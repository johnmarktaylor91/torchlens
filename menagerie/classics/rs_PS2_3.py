# FAITHFUL REIMPLEMENTATION from published architecture descriptions (no public code)
#
# "21cmBERT" (candidate queue name) has no distinct named repo (confirmed by GitHub repo/code search
# and web search: no "21cmBERT"-named project exists). The candidate's own triage note identifies it
# as the same attention-on-21cm-sequences family as the "21cm Transformer" candidate, but using a
# BERT-style masked-pretraining objective, "well precedented (cf. Astromer)". ASTROMER (Donoso-Oliva
# et al., A&A 2023, "ASTROMER: A transformer-based embedding for the representation of light curves")
# is a public, well-documented architecture description (paper Table/Sec. 3): a bidirectional
# Transformer encoder pretrained with a BERT-style masked-reconstruction objective over astronomical
# time sequences -- randomly mask a subset of sequence positions, exclude the masked positions'
# original values from what the encoder attends to on input (replace with a learned mask token),
# run bidirectional self-attention, and reconstruct the masked values with a lightweight prediction
# head trained against the true (unmasked) values. This is detailed enough to faithfully port the
# MECHANISM (masked reconstruction over a Transformer encoder) to the 21-cm domain: instead of
# (magnitude, MJD) light-curve observations, the sequence elements are per-redshift 21-cm power
# spectra / brightness-temperature summary vectors. No code exists anywhere under the name
# "21cmBERT" to vendor or port (rungs 2/3 do not apply), so this is RUNG 4. As with the sibling
# "21cm Transformer" candidate, the actual self-attention mechanism is the real
# `torch.nn.TransformerEncoder` from the base library, not a hand-rolled approximation; only the
# masking scheme and reconstruction head are reimplemented per the Astromer-style description.
from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "reimpl-pytorch"


class MaskedReionizationBERT(nn.Module):
    """BERT/Astromer-style masked-reconstruction Transformer encoder over 21-cm sequences.

    A random subset of sequence positions is masked (replaced by a learned mask token) before the
    bidirectional self-attention encoder runs; a lightweight linear head reconstructs the masked
    feature vectors from the (bidirectionally) contextualized encoding, following the Astromer
    masked-light-curve-modeling recipe adapted to 21-cm power-spectrum/brightness-temperature
    sequences ("21cmBERT" per the candidate's triage note).
    """

    def __init__(
        self,
        feat_dim: int = 8,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        mask_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.mask_prob = mask_prob
        self.input_embed = nn.Linear(feat_dim, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstruction_head = nn.Linear(d_model, feat_dim)

    def forward(self, sequence: Tensor) -> Tensor:
        """sequence: (batch, seq_len, feat_dim) sequence of per-redshift 21-cm feature vectors.

        Returns the reconstructed feature vectors at every position (masked and unmasked); the
        masked-reconstruction loss is computed only over the masked positions during training.
        """
        batch, seq_len, _ = sequence.shape
        embedded = self.input_embed(sequence)
        mask = (torch.rand(batch, seq_len, 1, device=sequence.device) < self.mask_prob).float()
        masked_embedded = embedded * (1 - mask) + self.mask_token * mask
        masked_embedded = masked_embedded + self.pos_embed[:, :seq_len]
        encoded = self.encoder(masked_embedded)
        return self.reconstruction_head(encoded)


def build_21cmbert() -> nn.Module:
    model = MaskedReionizationBERT(
        feat_dim=8, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, mask_prob=0.5
    )
    model.eval()
    return model


def example_input_21cmbert() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(
        2, 20, 8
    )  # (batch, 20 redshift-bin sequence positions, 8-dim feature vector)


MENAGERIE_ENTRIES = [
    ("21cmBERT", "build_21cmbert", "example_input_21cmbert", 2023, MENAGERIE_ZOO),
]
