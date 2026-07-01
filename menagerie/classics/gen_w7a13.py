"""Protein/RNA structure- and molecule-modeling architecture family: gen_w7a13.

Sources checked (repo_url / desc_source from the build queue, web search for
architecture details where the repo itself is TF1-only, weight-only, or archived):
  - trRosetta: https://github.com/gjoni/trRosetta; Yang et al., PNAS 2020
    (arXiv:2004.02638 companion trRosetta2). A dilated-convolution residual network:
    an ``L x L x 526`` MSA co-evolution feature map (from direct-coupling-analysis /
    covariance features) is projected to 64 channels, then passed through a deep
    stack of residual blocks whose dilation cycles ``1, 2, 4, 8, 16`` (to grow the
    receptive field cheaply over the pairwise map), and finally branches into four
    independent 2D-conv + softmax heads predicting inter-residue distance and three
    orientation-angle (``omega``, ``theta``, ``phi``) distributions, with the
    symmetric d/omega maps explicitly symmetrized before their head.
  - trRosettaRNA: https://github.com/YangLab-SDU/trRosettaRNA2; Wang et al.,
    Nat. Commun. 2023 (Wang et al., Biology Methods and Protocols 2024 follow-up).
    Adapts the trRosetta idea to RNA with a transformer-based "RNAformer": an MSA
    representation and a pair representation (from sequence + secondary structure)
    are built, then alternately updated by row/column MSA self-attention and
    axial (row/column) pair self-attention -- an Evoformer-style two-track update --
    before 2D heads predict inter-nucleotide distance/orientation distributions
    used as folding restraints.
  - ZymCTRL: https://huggingface.co/AI4PD/ZymCTRL; Ferruz et al., MLSB 2022 /
    bioRxiv 2024 (arXiv:2410.03634 adapter follow-up). A CTRL-style (Keskar et al.
    2019) GPT decoder-only Transformer language model over enzyme sequences: an
    Enzyme-Commission (EC) control code is tokenized and prepended to the amino-acid
    sequence, so the causal self-attention stack conditions every generated residue
    on the EC-class control tokens through ordinary left-to-right attention (no
    separate conditioning pathway -- the control code IS part of the token stream).
  - ABT-MPNN: https://github.com/LCY02/ABT-MPNN; Yang et al., J. Cheminformatics
    2023. An atom-bond transformer message-passing neural network for molecular
    property prediction: standard D-MPNN directed-bond message passing is extended
    with a bond-level multi-head self-attention over the current bond-message set at
    each MPNN step (attention re-weights which bonds influence a given bond's
    update) and an atom-level self-attention re-weighting of the atom embeddings
    produced by the readout, before pooling to a molecule-level property.
  - Chemformer: https://github.com/MolecularAI/Chemformer (archived Feb 2026;
    functional); Irwin et al., Mach. Learn.: Sci. Technol. 2022. A standard BART
    (Lewis et al. 2019) encoder-decoder Transformer applied to tokenized SMILES:
    pretrained with a span-denoising objective (mask/shuffle SMILES tokens, decode
    the original string) and fine-tuned for reaction prediction, retrosynthesis,
    and property prediction. Reimplemented here as a compact bidirectional-encoder /
    causal-decoder Transformer over a SMILES-character vocabulary.
  - ANI (torchani, ANI-2x) is SKIPPED: already present as ``build_ani2x`` /
    ``example_input_ani2x`` in ``menagerie/classics/gen_w6a17.py`` (Behler-Parrinello
    atomic environment vector potential) -- a duplicate of this candidate.

All models below are compact, randomly initialized, faithful reimplementations of
each architecture's distinctive mechanism (not generic MLP/transformer stubs), sized
small so tracing and rendering stay fast.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# trRosetta -- dilated-residual 2D ConvNet over MSA co-evolution features,
# branching into distance + 3 orientation-angle softmax heads.
# ---------------------------------------------------------------------------


class TrRosettaResBlock(nn.Module):
    """One dilated 2D residual block: conv-BN-ELU-conv-BN(+dropout)-ELU with a skip."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Build a dilated residual block.

        Parameters
        ----------
        channels : int
            Number of feature-map channels (constant through the block).
        dilation : int
            Dilation rate for both convolutions.
        """

        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``x`` plus the dilated-conv residual branch, ELU-activated."""

        residual = x
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.elu(out + residual)


class TrRosetta(nn.Module):
    """Compact trRosetta: dilated-residual ConvNet -> distance/orientation heads."""

    def __init__(
        self,
        in_features: int = 42,
        channels: int = 16,
        n_blocks: int = 8,
        n_bins: int = 37,
        n_angle_bins: int = 25,
    ) -> None:
        """Build the input projection, dilated-residual stack, and 4 output heads.

        Parameters
        ----------
        in_features : int
            Number of input pairwise co-evolution feature channels.
        channels : int
            Hidden channel width of the residual stack.
        n_blocks : int
            Number of dilated residual blocks (dilation cycles 1, 2, 4, 8).
        n_bins : int
            Number of distance histogram bins.
        n_angle_bins : int
            Number of bins for each orientation-angle histogram.
        """

        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_features, channels, 1), nn.BatchNorm2d(channels), nn.ELU()
        )
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [TrRosettaResBlock(channels, dilations[i % len(dilations)]) for i in range(n_blocks)]
        )
        self.dist_head = nn.Conv2d(channels, n_bins, 1)
        self.omega_head = nn.Conv2d(channels, n_angle_bins, 1)
        self.theta_head = nn.Conv2d(channels, n_angle_bins, 1)
        self.phi_head = nn.Conv2d(channels, n_angle_bins, 1)

    def forward(self, pair_features: Tensor) -> dict[str, Tensor]:
        """Return distance/omega/theta/phi bin-probability maps for ``pair_features``.

        Parameters
        ----------
        pair_features : Tensor
            Co-evolution feature map of shape ``(1, in_features, L, L)``.

        Returns
        -------
        dict[str, Tensor]
            ``dist``, ``omega``, ``theta``, ``phi`` softmax probability maps.
        """

        x = self.in_proj(pair_features)
        for block in self.blocks:
            x = block(x)
        # d and omega are symmetric properties of the residue pair -> symmetrize
        # the feature map before those two heads (as in the reference network).
        x_sym = 0.5 * (x + x.transpose(-1, -2))
        dist = F.softmax(self.dist_head(x_sym), dim=1)
        omega = F.softmax(self.omega_head(x_sym), dim=1)
        theta = F.softmax(self.theta_head(x), dim=1)
        phi = F.softmax(self.phi_head(x), dim=1)
        return {"dist": dist, "omega": omega, "theta": theta, "phi": phi}


def build_trrosetta() -> nn.Module:
    """Build a compact random-init trRosetta dilated-residual network.

    Returns
    -------
    nn.Module
        ``TrRosetta`` in eval mode.
    """

    return TrRosetta(in_features=42, channels=16, n_blocks=8, n_bins=37, n_angle_bins=25).eval()


def example_input_trrosetta() -> Tensor:
    """Return a small ``L=20`` residue-pair co-evolution feature map.

    Returns
    -------
    Tensor
        Feature map of shape ``(1, 42, 20, 20)``.
    """

    torch.manual_seed(0)
    return torch.randn(1, 42, 20, 20)


# ---------------------------------------------------------------------------
# trRosettaRNA -- Evoformer-style "RNAformer": alternating MSA row/column
# self-attention and axial (row/column) pair self-attention over an RNA MSA
# representation and pair representation, feeding 2D geometry heads.
# ---------------------------------------------------------------------------


class AxialPairAttention(nn.Module):
    """Row-then-column self-attention over an ``(L, L, C)`` pair representation."""

    def __init__(self, dim: int, n_heads: int = 2) -> None:
        """Build the row-wise and column-wise multi-head self-attention modules."""

        super().__init__()
        self.row_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_row = nn.LayerNorm(dim)
        self.norm_col = nn.LayerNorm(dim)

    def forward(self, pair: Tensor) -> Tensor:
        """Apply row-wise then column-wise self-attention with residual updates.

        Parameters
        ----------
        pair : Tensor
            Pair representation ``(L, L, C)``.
        """

        length = pair.shape[0]
        # Row attention: each row i attends over columns j (batch over rows).
        row_in = self.norm_row(pair)
        row_out, _ = self.row_attn(row_in, row_in, row_in)
        pair = pair + row_out
        # Column attention: transpose so each column becomes a "row" to attend over.
        col_in = self.norm_col(pair).transpose(0, 1)
        col_out, _ = self.col_attn(col_in, col_in, col_in)
        pair = pair + col_out.transpose(0, 1)
        return pair.view(length, length, -1)


class RNAformerBlock(nn.Module):
    """One Evoformer-style block: MSA row-attention (biased by pair) + axial pair update."""

    def __init__(self, msa_dim: int, pair_dim: int, n_heads: int = 2) -> None:
        """Build the MSA row-attention, MSA->pair outer-product update, and pair axial attention."""

        super().__init__()
        self.msa_attn = nn.MultiheadAttention(msa_dim, n_heads, batch_first=True)
        self.msa_norm = nn.LayerNorm(msa_dim)
        self.outer_proj = nn.Linear(msa_dim, pair_dim)
        self.pair_attn = AxialPairAttention(pair_dim, n_heads)
        self.pair_norm = nn.LayerNorm(pair_dim)

    def forward(self, msa: Tensor, pair: Tensor) -> tuple[Tensor, Tensor]:
        """Update the MSA and pair representations for one RNAformer block.

        Parameters
        ----------
        msa : Tensor
            MSA representation ``(n_seq, L, msa_dim)``.
        pair : Tensor
            Pair representation ``(L, L, pair_dim)``.
        """

        n_seq, length, msa_dim = msa.shape
        # MSA row (per-sequence) self-attention across positions.
        flat = self.msa_norm(msa)
        attn_out, _ = self.msa_attn(flat, flat, flat)
        msa = msa + attn_out
        # Outer-product-mean style MSA -> pair communication: average the
        # per-position projected MSA features and broadcast into the pair map.
        summary = self.outer_proj(msa).mean(dim=0)  # (L, pair_dim)
        pair = pair + summary.unsqueeze(1) + summary.unsqueeze(0)
        pair = self.pair_norm(self.pair_attn(pair))
        return msa, pair


class TrRosettaRNA(nn.Module):
    """Compact trRosettaRNA: RNAformer (MSA + axial-pair attention) -> geometry heads."""

    def __init__(
        self,
        vocab_size: int = 5,
        msa_dim: int = 16,
        pair_dim: int = 16,
        n_blocks: int = 2,
        n_bins: int = 20,
    ) -> None:
        """Build the MSA/pair embeddings, RNAformer stack, and distance/angle heads.

        Parameters
        ----------
        vocab_size : int
            Nucleotide alphabet size (A, C, G, U, gap).
        msa_dim : int
            MSA representation channel width.
        pair_dim : int
            Pair representation channel width.
        n_blocks : int
            Number of RNAformer blocks.
        n_bins : int
            Number of geometry histogram bins.
        """

        super().__init__()
        self.msa_embed = nn.Embedding(vocab_size, msa_dim)
        self.pair_init = nn.Linear(2 * msa_dim, pair_dim)
        self.blocks = nn.ModuleList(
            [RNAformerBlock(msa_dim, pair_dim, n_heads=2) for _ in range(n_blocks)]
        )
        self.dist_head = nn.Linear(pair_dim, n_bins)
        self.orientation_head = nn.Linear(pair_dim, n_bins)

    def forward(self, msa_tokens: Tensor) -> dict[str, Tensor]:
        """Return distance/orientation bin logits for the RNA MSA ``msa_tokens``.

        Parameters
        ----------
        msa_tokens : Tensor
            Integer nucleotide tokens ``(n_seq, L)``.

        Returns
        -------
        dict[str, Tensor]
            ``dist`` and ``orientation`` logit maps of shape ``(L, L, n_bins)``.
        """

        msa = self.msa_embed(msa_tokens)  # (n_seq, L, msa_dim)
        length = msa.shape[1]
        query_seq = msa[0]  # (L, msa_dim) -- reference sequence row
        pair = self.pair_init(
            torch.cat(
                [
                    query_seq.unsqueeze(1).expand(-1, length, -1),
                    query_seq.unsqueeze(0).expand(length, -1, -1),
                ],
                dim=-1,
            )
        )
        for block in self.blocks:
            msa, pair = block(msa, pair)
        pair_sym = 0.5 * (pair + pair.transpose(0, 1))
        return {"dist": self.dist_head(pair_sym), "orientation": self.orientation_head(pair)}


def build_trrosettarna() -> nn.Module:
    """Build a compact random-init trRosettaRNA (RNAformer) model.

    Returns
    -------
    nn.Module
        ``TrRosettaRNA`` in eval mode.
    """

    return TrRosettaRNA(vocab_size=5, msa_dim=16, pair_dim=16, n_blocks=2, n_bins=20).eval()


def example_input_trrosettarna() -> Tensor:
    """Return a small RNA MSA of 6 sequences x 12 nucleotide positions.

    Returns
    -------
    Tensor
        Integer token tensor of shape ``(6, 12)``.
    """

    torch.manual_seed(0)
    return torch.randint(0, 5, (6, 12))


# ---------------------------------------------------------------------------
# ZymCTRL -- CTRL-style GPT decoder-only LM: EC-number control-code tokens are
# prepended to the enzyme-sequence token stream and conditioned on purely via
# ordinary causal self-attention (no separate conditioning pathway).
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Standard causal multi-head self-attention block."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Build the QKV and output projections for causal self-attention."""

        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Return the causally-masked self-attention output for ``x`` ``(B, T, C)``."""

        batch, seq_len, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t: Tensor) -> Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim) + mask
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out_proj(context)


class CtrlBlock(nn.Module):
    """One CTRL/GPT transformer block: pre-norm causal self-attention + MLP."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Build the pre-attention norm, causal self-attention, and feed-forward sublayers."""

        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the block output after residual self-attention and feed-forward."""

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ZymCtrl(nn.Module):
    """Compact ZymCTRL: CTRL-style GPT decoder-only LM over EC-code + enzyme tokens."""

    def __init__(
        self, vocab_size: int = 50, d_model: int = 32, n_heads: int = 4, n_layers: int = 3
    ) -> None:
        """Build the token/position embeddings, causal transformer stack, and LM head.

        Parameters
        ----------
        vocab_size : int
            Joint vocabulary size for EC-control-code tokens + amino-acid tokens.
        d_model : int
            Model (hidden) dimensionality.
        n_heads : int
            Number of self-attention heads.
        n_layers : int
            Number of stacked CTRL blocks.
        """

        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(256, d_model)
        self.blocks = nn.ModuleList([CtrlBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return next-token logits for ``token_ids`` (EC control code + sequence).

        Parameters
        ----------
        token_ids : Tensor
            Integer token ids ``(batch, seq_len)``; the leading positions are the
            EC-number control-code tokens, followed by amino-acid sequence tokens.

        Returns
        -------
        Tensor
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """

        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.tok_embed(token_ids) + self.pos_embed(positions).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


def build_zymctrl() -> nn.Module:
    """Build a compact random-init ZymCTRL CTRL-style enzyme language model.

    Returns
    -------
    nn.Module
        ``ZymCtrl`` in eval mode.
    """

    return ZymCtrl(vocab_size=50, d_model=32, n_heads=4, n_layers=3).eval()


def example_input_zymctrl() -> Tensor:
    """Return EC-control-code tokens followed by a short enzyme sequence, ``(1, 14)``.

    Returns
    -------
    Tensor
        Integer token ids of shape ``(1, 14)`` (4 EC-code tokens + 10 sequence tokens).
    """

    torch.manual_seed(0)
    ec_code = torch.tensor([[40, 41, 1, 1]])  # e.g. "1.1.1.2"-style control tokens
    sequence = torch.randint(2, 22, (1, 10))  # 20 amino-acid tokens
    return torch.cat([ec_code, sequence], dim=1)


# ---------------------------------------------------------------------------
# ABT-MPNN -- Atom-Bond Transformer MPNN: directed-bond D-MPNN message passing
# with a bond-level self-attention re-weighting at every message-passing step,
# plus an atom-level self-attention over the readout embeddings.
# ---------------------------------------------------------------------------


class BondAttention(nn.Module):
    """Bond-level multi-head self-attention over the current directed-bond messages."""

    def __init__(self, dim: int, n_heads: int = 2) -> None:
        """Build the bond-message self-attention module."""

        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, bond_messages: Tensor) -> Tensor:
        """Return attention-reweighted bond messages ``(1, n_bonds, dim)``."""

        normed = self.norm(bond_messages)
        out, _ = self.attn(normed, normed, normed)
        return bond_messages + out


class AbtMpnn(nn.Module):
    """Compact ABT-MPNN: D-MPNN directed message passing + bond/atom self-attention."""

    def __init__(
        self,
        atom_features: int = 12,
        bond_features: int = 6,
        hidden_dim: int = 24,
        n_steps: int = 3,
        n_out: int = 1,
    ) -> None:
        """Build the bond-message init, message-passing + bond-attention stack, and readout.

        Parameters
        ----------
        atom_features : int
            Raw per-atom feature dimensionality.
        bond_features : int
            Raw per-directed-bond feature dimensionality.
        hidden_dim : int
            Hidden message dimensionality.
        n_steps : int
            Number of D-MPNN message-passing / bond-attention steps.
        n_out : int
            Number of predicted molecular properties.
        """

        super().__init__()
        self.bond_init = nn.Linear(atom_features + bond_features, hidden_dim)
        self.bond_update = nn.Linear(hidden_dim, hidden_dim)
        self.bond_attns = nn.ModuleList([BondAttention(hidden_dim) for _ in range(n_steps)])
        self.n_steps = n_steps
        self.atom_readout = nn.Linear(atom_features + hidden_dim, hidden_dim)
        self.atom_attn = nn.MultiheadAttention(hidden_dim, 2, batch_first=True)
        self.atom_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_out)
        )

    def forward(
        self, atom_feats: Tensor, bond_feats: Tensor, bond_src: Tensor, incoming: Tensor
    ) -> Tensor:
        """Return the predicted molecular property vector.

        Parameters
        ----------
        atom_feats : Tensor
            Per-atom features ``(n_atoms, atom_features)``.
        bond_feats : Tensor
            Per-directed-bond features ``(n_bonds, bond_features)``.
        bond_src : Tensor
            Source-atom index of each directed bond ``(n_bonds,)``.
        incoming : Tensor
            Incoming-bond adjacency ``(n_bonds, n_bonds)`` (1 if bond j feeds into
            the source atom of bond i, excluding the reverse of bond i itself).

        Returns
        -------
        Tensor
            Molecule-level property prediction ``(n_out,)``.
        """

        src_feats = atom_feats[bond_src]
        bond_msg = F.relu(self.bond_init(torch.cat([src_feats, bond_feats], dim=-1)))
        for step in range(self.n_steps):
            incoming_sum = incoming @ bond_msg
            bond_msg = F.relu(bond_msg + self.bond_update(incoming_sum))
            bond_msg = self.bond_attns[step](bond_msg.unsqueeze(0)).squeeze(0)
        n_atoms = atom_feats.shape[0]
        atom_incoming = torch.zeros(n_atoms, bond_msg.shape[0], device=atom_feats.device)
        atom_incoming.scatter_(1, bond_src.unsqueeze(0).expand(n_atoms, -1), 1.0)
        atom_msg_sum = (
            atom_incoming @ bond_msg / atom_incoming.sum(dim=1, keepdim=True).clamp(min=1.0)
        )
        atom_repr = F.relu(self.atom_readout(torch.cat([atom_feats, atom_msg_sum], dim=-1)))
        normed = self.atom_norm(atom_repr).unsqueeze(0)
        attn_out, _ = self.atom_attn(normed, normed, normed)
        atom_repr = atom_repr + attn_out.squeeze(0)
        mol_repr = atom_repr.mean(dim=0)
        return self.ffn(mol_repr)


def build_abt_mpnn() -> nn.Module:
    """Build a compact random-init ABT-MPNN atom-bond transformer MPNN.

    Returns
    -------
    nn.Module
        ``AbtMpnn`` in eval mode.
    """

    return AbtMpnn(atom_features=12, bond_features=6, hidden_dim=24, n_steps=3, n_out=1).eval()


def example_input_abt_mpnn() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return a small toy-molecule directed-bond graph (e.g. an 8-atom ring-and-chain).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``atom_feats (8, 12)``, ``bond_feats (14, 6)``, ``bond_src (14,)`` source-atom
        indices, and ``incoming (14, 14)`` directed-bond incoming-message adjacency.
    """

    torch.manual_seed(0)
    n_atoms, n_bonds = 8, 14
    atom_feats = torch.randn(n_atoms, 12)
    bond_feats = torch.randn(n_bonds, 6)
    bond_src = torch.randint(0, n_atoms, (n_bonds,))
    incoming = (torch.rand(n_bonds, n_bonds) > 0.7).float()
    incoming.fill_diagonal_(0.0)
    return atom_feats, bond_feats, bond_src, incoming


# ---------------------------------------------------------------------------
# Chemformer -- BART-style bidirectional-encoder / causal-decoder Transformer
# pretrained via denoising over tokenized SMILES strings.
# ---------------------------------------------------------------------------


class Chemformer(nn.Module):
    """Compact Chemformer: BART encoder-decoder over a SMILES token vocabulary."""

    def __init__(
        self,
        vocab_size: int = 60,
        d_model: int = 32,
        n_heads: int = 4,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
    ) -> None:
        """Build the shared token embedding, BART encoder, and BART causal decoder.

        Parameters
        ----------
        vocab_size : int
            SMILES-character token vocabulary size.
        d_model : int
            Model (hidden) dimensionality.
        n_heads : int
            Number of attention heads.
        n_enc_layers : int
            Number of Transformer encoder layers.
        n_dec_layers : int
            Number of Transformer decoder layers.
        """

        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(128, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_enc_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_dec_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _embed(self, token_ids: Tensor) -> Tensor:
        """Return token + position embeddings for ``token_ids``."""

        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device)
        return self.tok_embed(token_ids) + self.pos_embed(positions).unsqueeze(0)

    def forward(self, noised_smiles_ids: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Return reconstruction logits for the denoising-decoded SMILES.

        Parameters
        ----------
        noised_smiles_ids : Tensor
            Corrupted (masked/shuffled) source SMILES tokens ``(batch, src_len)``,
            consumed bidirectionally by the encoder.
        decoder_input_ids : Tensor
            Right-shifted target SMILES tokens ``(batch, tgt_len)``, consumed
            causally by the decoder to reconstruct the original SMILES string.

        Returns
        -------
        Tensor
            Vocabulary logits ``(batch, tgt_len, vocab_size)``.
        """

        memory = self.encoder(self._embed(noised_smiles_ids))
        tgt_len = decoder_input_ids.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            decoder_input_ids.device
        )
        decoded = self.decoder(self._embed(decoder_input_ids), memory, tgt_mask=causal_mask)
        return self.lm_head(decoded)


def build_chemformer() -> nn.Module:
    """Build a compact random-init Chemformer BART encoder-decoder over SMILES.

    Returns
    -------
    nn.Module
        ``Chemformer`` in eval mode.
    """

    return Chemformer(vocab_size=60, d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2).eval()


def example_input_chemformer() -> tuple[Tensor, Tensor]:
    """Return a noised source SMILES token sequence and a shifted decoder target.

    Returns
    -------
    tuple[Tensor, Tensor]
        Encoder input ``(1, 18)`` and decoder input ``(1, 16)`` SMILES token ids.
    """

    torch.manual_seed(0)
    src = torch.randint(3, 60, (1, 18))
    tgt = torch.randint(3, 60, (1, 16))
    return src, tgt


MENAGERIE_ENTRIES = [
    ("trRosetta", "build_trrosetta", "example_input_trrosetta", "2020", "BIO"),
    ("trRosettaRNA", "build_trrosettarna", "example_input_trrosettarna", "2023", "BIO"),
    ("ZymCTRL", "build_zymctrl", "example_input_zymctrl", "2022", "BIO"),
    ("ABT-MPNN", "build_abt_mpnn", "example_input_abt_mpnn", "2023", "BIO"),
    ("Chemformer", "build_chemformer", "example_input_chemformer", "2022", "BIO"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")
