# SOURCE: vendored from instadeepai/InstaNovo @ 55a9a3ff7c18e2748c0ef7fc6ba3213002ad6a0f
# https://raw.githubusercontent.com/instadeepai/InstaNovo/main/instanovo/transformer/model.py
# https://raw.githubusercontent.com/instadeepai/InstaNovo/main/instanovo/transformer/layers.py
# https://raw.githubusercontent.com/instadeepai/InstaNovo/main/instanovo/utils/residues.py
# https://raw.githubusercontent.com/instadeepai/InstaNovo/main/instanovo/inference/interfaces.py
#
# Eloff et al. 2025 (Nature Machine Intelligence) "InstaNovo: De novo peptide sequencing
# with a transformer model". This is the flagship InstaNovo transformer (the "transformer +
# multinomial diffusion (InstaNovo+)" family the queue notes describe -- this staging module
# covers the base transformer half; the diffusion refinement head lives in a separate
# `instanovo/diffusion/` subpackage). `InstaNovo` is a standard encoder-decoder transformer
# over mass-spectrometry spectra: `MultiScalePeakEmbedding` encodes each (m/z, intensity)
# peak with a multi-frequency sinusoidal m/z embedding (Voronov et al.) fed through two small
# MLPs, a learned `latent_spectrum` token is prepended, precursor mass/charge features are
# encoded the same way and prepended again, and the whole sequence runs through a standard
# `nn.TransformerEncoder`. The decoder embeds the (right-to-left) peptide token sequence with
# `nn.Embedding` + sinusoidal `PositionalEncoding`, applies a causal mask, and runs a standard
# `nn.TransformerDecoder` cross-attending to the encoder memory, ending in a linear
# vocab-projection head. `InstaNovo.forward`/`_encoder`/`_decoder`/`_get_causal_mask` are
# copied verbatim from the real `transformer/model.py`; the Flash-Attention variants
# (`_flash_encoder`/`_flash_decoder`, gated behind `use_flash_attention=False` by default and
# unused here), checkpoint I/O (`load`/`from_pretrained`/`get_pretrained`/`_whitelist_torch_
# omegaconf`), and non-architectural scoring/decoding utilities (`score_sequences`,
# `score_candidates`, `idx_to_aa`, `batch_idx_to_aa`) are omitted as they are not part of the
# forward-pass architecture. `PositionalEncoding`/`MultiScalePeakEmbedding`/`ConvPeakEmbedding`
# are copied verbatim from `transformer/layers.py` (`ConvPeakEmbedding` is included for
# fidelity even though the default `conv_peak_encoder=False` path used here does not call it).
# `ResidueSet` is copied verbatim from `utils/residues.py` (the real class InstaNovo takes as
# its `residue_set` constructor arg, needed for `vocab_size = len(residue_set)` and the
# `SOS_INDEX`/`EOS_INDEX`/`PAD_INDEX` token ids used by `_decoder`). `Decodable` (the ABC
# `InstaNovo` inherits from) is copied verbatim from `inference/interfaces.py`;
# `ScoredSequence` (a dataclass unused by any traced forward-pass code path) is omitted
# because `@dataclass` requires its defining module to be registered in `sys.modules`,
# which this file's staging-module `importlib.util.spec_from_file_location` loader does
# not do -- dropping the unused dataclass keeps the loader contract intact without
# touching any architecture.

from __future__ import annotations

import math
import re
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

MAX_SEQUENCE_LENGTH = 200


class SpecialTokens:
    """Special tokens used by the ResidueSet and model (mirrors constants.SpecialTokens)."""

    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "[EOS]"
    SOS_TOKEN = "[SOS]"


H2O_MASS = 18.0106
PROTON_MASS_AMU = 1.007276


# --- utils/residues.py (verbatim, ResidueSet class only) ---


class ResidueSet:
    """A class for managing sets of residues.

    Args:
        residue_masses (dict[str, float]):
            Dictionary of residues mapping to corresponding mass values.
        residue_remapping (dict[str, str] | None, optional):
            Dictionary of residues mapping to keys in `residue_masses`.
            This is used for dataset specific residue naming conventions.
            Residue remapping may be many-to-one.
    """

    def __init__(
        self,
        residue_masses: dict[str, float],
        residue_remapping: dict[str, str] | None = None,
    ) -> None:
        self.residue_masses = residue_masses
        self.residue_remapping = residue_remapping if residue_remapping else {}

        # Special tokens come first
        self.special_tokens = [
            SpecialTokens.PAD_TOKEN,
            SpecialTokens.SOS_TOKEN,
            SpecialTokens.EOS_TOKEN,
        ]

        self.vocab = self.special_tokens + list(self.residue_masses.keys())

        # Create mappings
        self.residue_to_index = {residue: index for index, residue in enumerate(self.vocab)}
        self.index_to_residue = dict(enumerate(self.vocab))
        self.tokenizer_regex = (
            r"(\[[^\]]+\]"
            r"|\([^)]+\)"
            r"|[+-]?\d+(?:\.\d+)?"
            r"|[+-]?\.\d+"
            r")|"
            r"([A-Z]"
            r"(?:\[[^\]]+\]"
            r"|\([^)]+\)"
            r"|[+-]?\d+(?:\.\d+)?"
            r"|[+-]?\.\d+)?"
            r")"
        )

        self.PAD_INDEX: int = self.residue_to_index[SpecialTokens.PAD_TOKEN]
        self.SOS_INDEX: int = self.residue_to_index[SpecialTokens.SOS_TOKEN]
        self.EOS_INDEX: int = self.residue_to_index[SpecialTokens.EOS_TOKEN]

    def update_remapping(self, mapping: dict[str, str]) -> None:
        """Update the residue remapping for specific datasets."""
        self.residue_remapping.update(mapping)

    def get_mass(self, residue: str) -> float:
        """Get the mass of a residue."""
        if self.residue_remapping and residue in self.residue_remapping:
            residue = self.residue_remapping[residue]
        return self.residue_masses[residue]

    def get_sequence_mass(self, sequence: str | list[str], charge: int | None) -> float:
        """Get the mass of a residue sequence."""
        mass = sum([self.get_mass(residue) for residue in sequence]) + H2O_MASS
        if charge:
            mass = (mass / charge) + PROTON_MASS_AMU
        return float(mass)

    def tokenize(self, sequence: str | list[str] | None) -> list[str]:
        """Split a peptide represented as a string into a list of residues."""
        if sequence is None:
            return []
        if isinstance(sequence, list):
            return sequence
        return [
            item
            for sublist in re.findall(self.tokenizer_regex, sequence)
            for item in sublist
            if item
        ]

    def detokenize(self, sequence: list[str]) -> str:
        """Joining a list of residues into a string representing the peptide."""
        return "".join(sequence)

    def encode(
        self,
        sequence: list[str],
        add_eos: bool = False,
        return_tensor: str | None = None,
        pad_length: int | None = None,
    ) -> torch.LongTensor | np.ndarray:
        """Map a sequence of residues to their indices and optionally pad them to a fixed length."""
        encoded_list = [
            self.residue_to_index[
                self.residue_remapping[residue] if residue in self.residue_remapping else residue
            ]
            for residue in sequence
        ]

        if add_eos:
            encoded_list.extend([self.EOS_INDEX])

        if pad_length:
            encoded_list.extend((pad_length - len(encoded_list)) * [self.PAD_INDEX])

        if return_tensor == "pt":
            return torch.tensor(encoded_list, dtype=torch.long)
        elif return_tensor == "np":
            return np.array(encoded_list, dtype=np.int32)
        else:
            return encoded_list

    def decode(self, sequence: torch.LongTensor | list[int], reverse: bool = False) -> list[str]:
        """Map a sequence of indices to the corresponding sequence of residues."""
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()

        residue_sequence = []
        for index in sequence:
            if index == self.EOS_INDEX:
                break
            if index == self.SOS_INDEX or index == self.PAD_INDEX:
                continue
            residue_sequence.append(index)

        if reverse:
            residue_sequence = residue_sequence[::-1]

        return [self.index_to_residue[index] for index in residue_sequence]

    def __len__(self) -> int:
        return len(self.index_to_residue)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResidueSet):
            return NotImplemented
        return self.vocab == other.vocab

    def __contains__(self, residue: str) -> bool:
        """Check if a residue is in the residue set."""
        return residue in self.residue_masses


# --- inference/interfaces.py (verbatim, Decodable only -- see header for why
# ScoredSequence is omitted) ---


class Decodable(metaclass=ABCMeta):
    """An interface for models that can be decoded.

    Algorithms should conform to the search interface.
    """

    @property
    @abstractmethod
    def residue_set(self) -> ResidueSet:
        """Every model must have a `residue_set` attribute."""
        pass

    @abstractmethod
    def init(self, spectra: Tensor, precursors: Tensor, *args, **kwargs):  # type:ignore
        """Initialize the search state."""
        pass


# --- transformer/layers.py (verbatim) ---


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Positional encoding forward pass.

        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiScalePeakEmbedding(nn.Module):
    """Multi-scale sinusoidal embedding based on Voronov et. al."""

    def __init__(
        self, h_size: int, dropout: float = 0, float_dtype: torch.dtype | str = torch.float64
    ) -> None:
        super().__init__()
        self.h_size = h_size
        self.float_dtype = (
            getattr(torch, float_dtype, None) if isinstance(float_dtype, str) else float_dtype
        )
        if self.float_dtype is None:
            raise ValueError(f"Unknown torch dtype string: {float_dtype}")

        self.mlp = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_size, h_size),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(h_size + 1, h_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_size, h_size),
            nn.Dropout(dropout),
        )

        freqs = 2 * np.pi / torch.logspace(-2, -3, int(h_size / 2), dtype=self.float_dtype)
        self.register_buffer("freqs", freqs)

    def forward(self, spectra: Tensor) -> Tensor:
        """Encode peaks."""
        mz_values, intensities = spectra[:, :, [0]], spectra[:, :, [1]]
        x = self.encode_mass(mz_values)
        x = self.mlp(x)
        x = torch.cat([x, intensities], axis=2)
        return self.head(x)

    def encode_mass(self, x: Tensor) -> Tensor:
        """Encode mz."""
        x = self.freqs[None, None, :] * x
        x = torch.cat([torch.sin(x), torch.cos(x)], axis=2)
        return x.float()


class ConvPeakEmbedding(nn.Module):
    """Convolutional peak embedding."""

    def __init__(self, h_size: int, dropout: float = 0) -> None:
        super().__init__()
        self.h_size = h_size

        self.conv = nn.Sequential(
            nn.Conv1d(1, h_size // 4, kernel_size=40_000, stride=100, padding=40_000 // 2 - 1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(h_size // 4, h_size, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Conv peak embedding."""
        x = x.unsqueeze(1)
        return self.conv(x).transpose(-1, -2)


# --- transformer/model.py (verbatim architecture; checkpoint I/O + flash-attention
# variants + non-forward-path scoring/decoding helpers omitted, see header) ---


class InstaNovo(nn.Module, Decodable):
    """The Instanovo model."""

    def __init__(
        self,
        residue_set: ResidueSet,
        dim_model: int = 768,
        n_head: int = 16,
        dim_feedforward: int = 2048,
        encoder_layers: int = 9,
        decoder_layers: int = 9,
        dropout: float = 0.1,
        max_charge: int = 5,
        use_flash_attention: bool = False,
        conv_peak_encoder: bool = False,
        peak_embedding_dtype: torch.dtype | str = torch.float64,
    ) -> None:
        super().__init__()
        self._residue_set = residue_set
        self.vocab_size = len(residue_set)
        self.use_flash_attention = use_flash_attention
        self.conv_peak_encoder = conv_peak_encoder

        self.latent_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        if self.use_flash_attention:
            self.pad_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        # Encoder
        self.peak_encoder = MultiScalePeakEmbedding(
            dim_model, dropout=dropout, float_dtype=peak_embedding_dtype
        )
        if self.conv_peak_encoder:
            self.conv_encoder = ConvPeakEmbedding(dim_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0 if self.use_flash_attention else dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
        )

        # Decoder
        self.aa_embed = nn.Embedding(self.vocab_size, dim_model, padding_idx=0)

        self.aa_pos_embed = PositionalEncoding(dim_model, dropout, max_len=MAX_SEQUENCE_LENGTH)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0 if self.use_flash_attention else dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
        )

        self.head = nn.Linear(dim_model, self.vocab_size)
        self.charge_encoder = nn.Embedding(max_charge, dim_model)

    @property
    def residue_set(self) -> ResidueSet:
        """Every model must have a `residue_set` attribute."""
        return self._residue_set

    @staticmethod
    def _get_causal_mask(seq_len: int, return_float: bool = False) -> Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        if return_float:
            return (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
        return ~mask.bool()

    def forward(
        self,
        x: Tensor,
        p: Tensor,
        y: Tensor,
        x_mask: Optional[Tensor] = None,
        y_mask: Optional[Tensor] = None,
        add_bos: bool = True,
        return_encoder_output: bool = False,
    ) -> Tensor:
        """Model forward pass.

        Args:
            x: Spectra, float Tensor (batch, n_peaks, 2)
            p: Precursors, float Tensor (batch, 3)
            y: Peptide, long Tensor (batch, seq_len, vocab)
            x_mask: Spectra padding mask, True for padded indices, bool Tensor (batch, n_peaks)
            y_mask: Peptide padding mask, bool Tensor (batch, seq_len)
            add_bos: Force add a <s> prefix to y, bool

        Returns:
            logits: float Tensor (batch, n, vocab_size),
            (batch, n+1, vocab_size) if add_bos==True.
        """
        x, x_mask = self._encoder(x, p, x_mask)
        y = self._decoder(x, y, x_mask, y_mask, add_bos)
        if return_encoder_output:
            return y, x
        return y

    def init(
        self,
        spectra: Tensor,
        precursors: Tensor,
        spectra_mask: Optional[Tensor] = None,
    ):
        """Initialise model encoder."""
        spectra, spectra_mask = self._encoder(spectra, precursors, spectra_mask)
        logits = self._decoder(spectra, None, spectra_mask, None, add_bos=False)
        return (spectra, spectra_mask), torch.log_softmax(logits[:, -1, :], -1)

    def get_residue_masses(self, mass_scale: int) -> Tensor:
        """Get the scaled masses of all residues."""
        residue_masses = torch.zeros(len(self.residue_set), dtype=torch.int64)
        for index, residue in self.residue_set.index_to_residue.items():
            if residue in self.residue_set.residue_masses:
                residue_masses[index] = round(mass_scale * self.residue_set.get_mass(residue))
        return residue_masses

    def get_eos_index(self) -> int:
        """Get the EOS token ID."""
        return int(self.residue_set.EOS_INDEX)

    def get_empty_index(self) -> int:
        """Get the PAD token ID."""
        return int(self.residue_set.PAD_INDEX)

    def decode(self, sequence: Tensor) -> list[str]:
        """Decode a single sequence of AA IDs."""
        return self.residue_set.decode(sequence, reverse=True)  # type: ignore

    def _encoder(
        self,
        x: Tensor,
        p: Tensor | None = None,
        x_mask: Optional[Tensor] = None,
    ):
        if self.conv_peak_encoder:
            x = self.conv_encoder(x)
            x_mask = torch.zeros((x.shape[0], x.shape[1]), device=x.device).bool()
        else:
            if x_mask is None:
                x_mask = ~x.sum(dim=2).bool()
            x = self.peak_encoder(x)

        # Self-attention on latent spectra AND peaks
        latent_spectra = self.latent_spectrum.expand(x.shape[0], -1, -1)
        x = torch.cat([latent_spectra, x], dim=1)
        latent_mask = torch.zeros((x_mask.shape[0], 1), dtype=bool, device=x_mask.device)
        x_mask = torch.cat([latent_mask, x_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=x_mask)

        # Prepare precursors
        if p is not None:
            masses = self.peak_encoder.encode_mass(p[:, None, [0]])
            charges = self.charge_encoder(p[:, 1].int() - 1)
            precursors = masses + charges[:, None, :]

            # Concatenate precursors
            x = torch.cat([precursors, x], dim=1)
            prec_mask = torch.zeros((x_mask.shape[0], 1), dtype=bool, device=x_mask.device)
            x_mask = torch.cat([prec_mask, x_mask], dim=1)

        return x, x_mask

    def _decoder(
        self,
        x: Tensor,
        y: Tensor,
        x_mask: Tensor,
        y_mask: Optional[Tensor] = None,
        add_bos: bool = True,
    ) -> Tensor:
        if y is None:
            y = torch.full((x.shape[0], 1), self.residue_set.SOS_INDEX, device=x.device)
        elif add_bos:
            bos = (
                torch.ones((y.shape[0], 1), dtype=y.dtype, device=y.device)
                * self.residue_set.SOS_INDEX
            )
            y = torch.cat([bos, y], dim=1)

            if y_mask is not None:
                bos_mask = torch.zeros((y_mask.shape[0], 1), dtype=bool, device=y_mask.device)
                y_mask = torch.cat([bos_mask, y_mask], dim=1)

        y = self.aa_embed(y)
        if y_mask is None:
            y_mask = ~y.sum(axis=2).bool()

        # concat bos
        y = self.aa_pos_embed(y)

        c_mask = self._get_causal_mask(y.shape[1]).to(y.device)

        y_hat = self.decoder(
            y,
            x,
            tgt_mask=c_mask,
            tgt_key_padding_mask=y_mask,
            memory_key_padding_mask=x_mask,
        )

        return self.head(y_hat)


# --- staging harness (tiny sizes; not part of the real repo) ---


def _example_residue_set() -> ResidueSet:
    # A tiny stand-in residue vocabulary (real InstaNovo ships a much larger
    # unimod-based residue_masses dict loaded from a config file); only the
    # vocab SIZE and special-token indices matter for the architecture.
    residue_masses = {
        "G": 57.02146,
        "A": 71.03711,
        "S": 87.03203,
        "P": 97.05276,
        "V": 99.06841,
        "T": 101.04768,
        "L": 113.08406,
        "N": 114.04293,
        "D": 115.02694,
        "K": 128.09496,
    }
    return ResidueSet(residue_masses)


def build_instanovo():
    # dim_model/n_head/dim_feedforward/encoder_layers/decoder_layers shrunk from the
    # real pretrained defaults (768/16/2048/9/9) to a tiny config for a fast trace;
    # max_charge and conv_peak_encoder=False (default path) preserved. peak_embedding_
    # dtype kept at the real default torch.float64 (MultiScalePeakEmbedding computes
    # its sinusoidal frequency table in this dtype before casting back to float32).
    residue_set = _example_residue_set()
    model = InstaNovo(
        residue_set=residue_set,
        dim_model=32,
        n_head=4,
        dim_feedforward=64,
        encoder_layers=2,
        decoder_layers=2,
        dropout=0.1,
        max_charge=5,
        use_flash_attention=False,
        conv_peak_encoder=False,
        peak_embedding_dtype=torch.float64,
    )
    model.eval()
    return model


def example_input_instanovo():
    # x: spectra (batch, n_peaks, 2) = [m/z, intensity] pairs, matching
    # `Float[Spectrum, " batch"]` used throughout transformer/model.py.
    # p: precursors (batch, 3) = [mass, charge, m/z]; charge_encoder does
    # `p[:, 1].int() - 1` so charge values must be >= 1 (charges 1..max_charge).
    # y: peptide token ids (batch, seq_len), long tensor of residue indices
    # (excluding SOS -- forward()'s default add_bos=True prepends it).
    batch = 1
    n_peaks = 6
    seq_len = 4
    x = torch.rand(batch, n_peaks, 2)
    p = torch.stack(
        [
            torch.rand(batch) * 1000 + 500,  # mass
            torch.randint(1, 5, (batch,)).float(),  # charge (>=1, see charge_encoder)
            torch.rand(batch) * 500 + 200,  # m/z
        ],
        dim=1,
    )
    residue_set = _example_residue_set()
    # Vocab has 3 special tokens ([PAD], [SOS], [EOS]) + 10 real residues = 13;
    # sample only from the real-residue index range to avoid emitting SOS/EOS mid-sequence.
    y = torch.randint(len(residue_set.special_tokens), len(residue_set), (batch, seq_len))
    return (x, p, y)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("InstaNovo", "build_instanovo", "example_input_instanovo", 2025, "vendored"),
]
