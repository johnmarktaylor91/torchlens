"""Wave 5 batch 10 menagerie classics: protein/RNA structure + biomedical NLP family.

Sources checked (repo_url / desc_source columns of the build queue, web research
2026-07-01; no cloning, no pip installs beyond the base env):
  - ProtenixFold: https://github.com/bytedance/Protenix ; arXiv:2503.01543
    "Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3
    Reproduction" (bioRxiv 2025.01.08.631967). AF3-style trunk (48 Pairformer
    blocks) + diffusion coordinate head; distinctive engineering choice vs. other
    AF3 reproductions is the *constrained/gated pair-update* Pairformer variant
    plus an explicit confidence (pLDDT-style) head trained jointly with structure.
  - PubMedBERT: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
    ; Gu et al. 2020, arXiv:2007.15779 "Domain-Specific Language Model
    Pretraining for Biomedical NLP". Standard BERT-base architecture trained
    from scratch (not fine-tuned from general-domain BERT) on PubMed abstracts
    with a domain-specific WordPiece vocabulary. Built here via
    ``transformers.BertConfig`` + ``BertForMaskedLM`` at tiny dims (the
    from-scratch-domain-vocab training choice, not architecture, is the
    distinctive contribution, so the standard BERT MLM head is faithful).
  - RF2NA (RoseTTAFold2NA): https://github.com/uw-ipd/RoseTTAFold2NA ;
    Baek et al. 2023, Nature Methods, "Accurate prediction of protein-nucleic
    acid complexes using RoseTTAFoldNA". Three-track network (1D sequence/MSA,
    2D residue-pair, 3D coordinate-frame) extended with nucleotide tokens in the
    1D track, protein-nucleotide pair interactions in the 2D track, and
    nucleotide backbone/torsion frames in the 3D track.
  - RfamGen: https://github.com/Shunsuke-1994/rfamgen ; Sumi, Hamada, Saito
    2024, Nature Methods, "Deep generative design of RNA family sequences"
    (10.1038/s41592-023-02148-8). Family-conditioned VAE: encodes an RNA
    sequence-plus-consensus-secondary-structure profile-HMM/covariance-model
    triplet (transcript, secondary structure, base-pairing) into a latent
    ``z``, decodes back to per-column emission (base) and pair-emission
    (base-pair) probabilities so sampling respects the family's consensus fold.
  - RFAA (RFdiffusion All-Atom): https://github.com/baker-laboratory/rf_diffusion_all_atom
    ; Krishna et al. 2024, Science, "Generalized biomolecular modeling and
    design with RoseTTAFold All-Atom" (10.1126/science.adl2528). Hybrid
    residue-frame (protein/DNA backbone) + atom-node (ligand/small-molecule)
    graph, three-track-style pair/coordinate updates, DDPM coordinate denoising
    conditioned on a fixed small-molecule motif.
  - RoseTTAFold2: https://github.com/uw-ipd/RoseTTAFold2 ; Baek et al. 2023,
    bioRxiv 10.1101/2023.05.24.542179, "Efficient and accurate prediction of
    protein structure using RoseTTAFold2". Three parallel tracks (1D MSA/seq,
    2D residue-pair distances, 3D structure) refined jointly across repeated
    blocks with frame-aligned coordinate updates (single-chain/multimer
    protein-only structure module, no nucleic-acid or ligand extensions --
    this is what distinguishes it from RF2NA and RFAA above).

All six are faithful compact reimplementations: random init, small dims, few
blocks, forward-only, kept just large enough to exercise each architecture's
distinctive mechanism so the traced/unrolled atlas graph renders quickly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM


def _heads(x: torch.Tensor, h: int) -> torch.Tensor:
    """Reshape the trailing channel dim of ``x`` into ``(h, c // h)`` heads.

    Parameters
    ----------
    x:
        Tensor whose last dimension will be split.
    h:
        Number of attention heads.

    Returns
    -------
    torch.Tensor
        Tensor with an extra trailing head dimension.
    """

    *lead, c = x.shape
    return x.view(*lead, h, c // h)


# ---------------------------------------------------------------------------
# 1. ProtenixFold: AF3-style Pairformer trunk (gated triangle updates) +
#    diffusion coordinate head + joint confidence head.
# ---------------------------------------------------------------------------


class GatedTriangleUpdate(nn.Module):
    """Gated triangle multiplicative pair update (Protenix's constrained variant)."""

    def __init__(self, c_z: int, c_hidden: int = 8, outgoing: bool = True) -> None:
        """Initialize the gated triangle-multiplication block.

        Parameters
        ----------
        c_z:
            Pair representation width.
        c_hidden:
            Hidden projection width for the triangle einsum.
        outgoing:
            Use outgoing (``ikc,jkc->ijc``) vs incoming edges.
        """

        super().__init__()
        self.outgoing = outgoing
        self.norm = nn.LayerNorm(c_z)
        self.a_proj = nn.Linear(c_z, c_hidden)
        self.a_gate = nn.Linear(c_z, c_hidden)
        self.b_proj = nn.Linear(c_z, c_hidden)
        self.b_gate = nn.Linear(c_z, c_hidden)
        self.norm_out = nn.LayerNorm(c_hidden)
        self.out_gate = nn.Linear(c_z, c_z)
        self.out = nn.Linear(c_hidden, c_z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply a gated triangle update to the pair representation.

        Parameters
        ----------
        z:
            Pair representation of shape ``(N, N, c_z)``.

        Returns
        -------
        torch.Tensor
            Updated pair representation, same shape as ``z``.
        """

        z = self.norm(z)
        a = torch.sigmoid(self.a_gate(z)) * self.a_proj(z)
        b = torch.sigmoid(self.b_gate(z)) * self.b_proj(z)
        x = (
            torch.einsum("ikc,jkc->ijc", a, b)
            if self.outgoing
            else torch.einsum("kic,kjc->ijc", a, b)
        )
        return torch.sigmoid(self.out_gate(z)) * self.out(self.norm_out(x))


class ProtenixPairformerBlock(nn.Module):
    """Compact Protenix Pairformer block: gated triangle updates + pair-biased attn."""

    def __init__(self, c_s: int, c_z: int, n_head: int = 4) -> None:
        """Build a single Pairformer block.

        Parameters
        ----------
        c_s:
            Single (per-token) representation width.
        c_z:
            Pair representation width.
        n_head:
            Attention head count for the single-rep update.
        """

        super().__init__()
        self.tri_out = GatedTriangleUpdate(c_z, outgoing=True)
        self.tri_in = GatedTriangleUpdate(c_z, outgoing=False)
        self.pair_transition = nn.Sequential(
            nn.LayerNorm(c_z), nn.Linear(c_z, c_z * 2), nn.ReLU(), nn.Linear(c_z * 2, c_z)
        )
        self.h = n_head
        self.c = c_s // n_head
        self.norm_s = nn.LayerNorm(c_s)
        self.q = nn.Linear(c_s, c_s, bias=False)
        self.k = nn.Linear(c_s, c_s, bias=False)
        self.v = nn.Linear(c_s, c_s, bias=False)
        self.pair_bias = nn.Linear(c_z, n_head, bias=False)
        self.gate = nn.Linear(c_s, c_s)
        self.attn_out = nn.Linear(c_s, c_s)
        self.single_transition = nn.Sequential(
            nn.LayerNorm(c_s), nn.Linear(c_s, c_s * 2), nn.ReLU(), nn.Linear(c_s * 2, c_s)
        )

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the single and pair representations.

        Parameters
        ----------
        s:
            Single representation of shape ``(N, c_s)``.
        z:
            Pair representation of shape ``(N, N, c_z)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(s, z)``.
        """

        z = z + self.tri_out(z)
        z = z + self.tri_in(z)
        z = z + self.pair_transition(z)
        h = self.norm_s(s)
        q = _heads(self.q(h), self.h).permute(1, 0, 2)
        k = _heads(self.k(h), self.h).permute(1, 0, 2)
        v = _heads(self.v(h), self.h).permute(1, 0, 2)
        bias = self.pair_bias(z).permute(2, 0, 1)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.c**0.5) + bias, dim=-1)
        out = torch.matmul(att, v).permute(1, 0, 2).reshape(s.shape[0], -1)
        s = s + self.attn_out(out * torch.sigmoid(self.gate(h)))
        s = s + self.single_transition(s)
        return s, z


class ProtenixDiffusionHead(nn.Module):
    """Denoising transformer coordinate head conditioned on trunk (s, z)."""

    def __init__(self, c_s: int, c_z: int, c_a: int = 16) -> None:
        """Build the diffusion coordinate head.

        Parameters
        ----------
        c_s:
            Single representation width.
        c_z:
            Pair representation width.
        c_a:
            Internal atom-token width.
        """

        super().__init__()
        self.in_proj = nn.Linear(3, c_a)
        self.time_embed = nn.Sequential(nn.Linear(1, c_s), nn.SiLU(), nn.Linear(c_s, c_s))
        self.cond = nn.Linear(c_s, c_a)
        self.pair_bias = nn.Linear(c_z, 1, bias=False)
        self.ff = nn.Sequential(nn.Linear(c_a, c_a * 2), nn.GELU(), nn.Linear(c_a * 2, c_a))
        self.out_proj = nn.Linear(c_a, 3)

    def forward(
        self, x_noised: torch.Tensor, s: torch.Tensor, z: torch.Tensor, t: float
    ) -> torch.Tensor:
        """Predict denoised atom coordinates.

        Parameters
        ----------
        x_noised:
            Noised coordinates of shape ``(N, 3)``.
        s:
            Trunk single representation of shape ``(N, c_s)``.
        z:
            Trunk pair representation of shape ``(N, N, c_z)``.
        t:
            Diffusion timestep scalar.

        Returns
        -------
        torch.Tensor
            Denoised coordinates of shape ``(N, 3)``.
        """

        a = self.in_proj(x_noised)
        tt = torch.full((s.shape[0], 1), float(t), device=s.device, dtype=s.dtype)
        cond = self.cond(s + self.time_embed(tt))
        pair_ctx = torch.matmul(torch.softmax(self.pair_bias(z).squeeze(-1), dim=-1), a)
        a = a + cond + pair_ctx
        a = a + self.ff(a)
        return x_noised + self.out_proj(a)


class ProtenixFold(nn.Module):
    """Compact ProtenixFold: gated-triangle Pairformer trunk + diffusion head + confidence head."""

    def __init__(self, c_s: int = 16, c_z: int = 16, n_block: int = 2, n_token: int = 24) -> None:
        """Build the ProtenixFold model.

        Parameters
        ----------
        c_s:
            Single representation width.
        c_z:
            Pair representation width.
        n_block:
            Number of Pairformer blocks in the trunk.
        n_token:
            Vocabulary size for the synthetic token-type input.
        """

        super().__init__()
        self.token_embed = nn.Embedding(n_token, c_s)
        self.left = nn.Embedding(n_token, c_z)
        self.right = nn.Embedding(n_token, c_z)
        self.blocks = nn.ModuleList([ProtenixPairformerBlock(c_s, c_z) for _ in range(n_block)])
        self.diffusion = ProtenixDiffusionHead(c_s, c_z)
        self.confidence_head = nn.Sequential(
            nn.Linear(c_s, c_s // 2), nn.ReLU(), nn.Linear(c_s // 2, 1)
        )

    def forward(self, token_types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict atom coordinates and a per-token confidence (pLDDT-style) score.

        Parameters
        ----------
        token_types:
            Integer token-type tensor of shape ``(N,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(coords, confidence)`` of shapes ``(N, 3)`` and ``(N, 1)``.
        """

        s = self.token_embed(token_types)
        z = self.left(token_types)[:, None, :] + self.right(token_types)[None, :, :]
        for block in self.blocks:
            s, z = block(s, z)
        x_noised = torch.randn(s.shape[0], 3, device=s.device, dtype=s.dtype)
        coords = self.diffusion(x_noised, s, z, t=0.5)
        confidence = torch.sigmoid(self.confidence_head(s))
        return coords, confidence


def build_protenixfold() -> nn.Module:
    """Build compact ProtenixFold.

    Returns
    -------
    nn.Module
        Random-initialized ProtenixFold in eval mode.
    """

    return ProtenixFold().eval()


def example_input_protenixfold() -> torch.Tensor:
    """Create a small token-type tensor for ProtenixFold.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape ``(10,)``.
    """

    return torch.randint(0, 24, (10,))


# ---------------------------------------------------------------------------
# 2. PubMedBERT (BiomedNLP-BiomedBERT): BERT-base pretrained from scratch on
#    PubMed abstracts with a domain vocabulary; standard architecture, built
#    via the transformers library at tiny dims.
# ---------------------------------------------------------------------------


def build_pubmedbert() -> nn.Module:
    """Build a tiny BERT-for-masked-LM standing in for PubMedBERT.

    PubMedBERT's distinctive contribution is training *from scratch* on
    PubMed text with a domain-specific vocabulary (Gu et al. 2020), not a
    novel architecture; the architecture itself is vanilla BERT-base. This
    builds that architecture at tiny dims via ``transformers``.

    Returns
    -------
    nn.Module
        Random-initialized ``BertForMaskedLM`` in eval mode.
    """

    cfg = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return BertForMaskedLM(cfg).eval()


def example_input_pubmedbert() -> torch.Tensor:
    """Create a small token-id tensor for PubMedBERT.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape ``(1, 16)``.
    """

    return torch.randint(0, 128, (1, 16))


# ---------------------------------------------------------------------------
# 3. RF2NA (RoseTTAFold2NA): three-track network with nucleotide tokens in
#    the 1D track, protein-nucleotide interactions in the 2D track, and
#    nucleotide backbone/torsion frames appended to the 3D track.
# ---------------------------------------------------------------------------


class ThreeTrackBlockNA(nn.Module):
    """One three-track update: 1D (seq) <-> 2D (pair) <-> 3D (coordinate frame)."""

    def __init__(self, c_1d: int, c_2d: int, n_head: int = 4) -> None:
        """Build a three-track update block extended for nucleic acids.

        Parameters
        ----------
        c_1d:
            1D sequence-track channel width.
        c_2d:
            2D pair-track channel width.
        n_head:
            Attention heads for the 1D self-attention.
        """

        super().__init__()
        self.h = n_head
        self.c = c_1d // n_head
        # 1D track: self-attention biased by the 2D pair track
        self.norm_1d = nn.LayerNorm(c_1d)
        self.q = nn.Linear(c_1d, c_1d, bias=False)
        self.k = nn.Linear(c_1d, c_1d, bias=False)
        self.v = nn.Linear(c_1d, c_1d, bias=False)
        self.pair_bias = nn.Linear(c_2d, n_head, bias=False)
        self.attn_out = nn.Linear(c_1d, c_1d)
        # 1D -> 2D: outer-product update (captures new protein-nucleotide pairs)
        self.outer_left = nn.Linear(c_1d, c_2d)
        self.outer_right = nn.Linear(c_1d, c_2d)
        self.pair_update = nn.Sequential(nn.LayerNorm(c_2d), nn.Linear(c_2d, c_2d), nn.ReLU())
        # 3D track: frame update from pooled pair context + torsion delta
        self.frame_from_pair = nn.Linear(c_2d, 3)
        self.torsion_head = nn.Linear(c_1d, 10)  # 10 torsions incl. NA backbone/side chain

    def forward(
        self, s: torch.Tensor, z: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Jointly refine sequence, pair, and coordinate-frame tracks.

        Parameters
        ----------
        s:
            1D sequence representation, shape ``(N, c_1d)``.
        z:
            2D pair representation, shape ``(N, N, c_2d)``.
        coords:
            3D coordinate-frame origins, shape ``(N, 3)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Updated ``(s, z, coords, torsions)``.
        """

        h = self.norm_1d(s)
        q = _heads(self.q(h), self.h).permute(1, 0, 2)
        k = _heads(self.k(h), self.h).permute(1, 0, 2)
        v = _heads(self.v(h), self.h).permute(1, 0, 2)
        bias = self.pair_bias(z).permute(2, 0, 1)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.c**0.5) + bias, dim=-1)
        out = torch.matmul(att, v).permute(1, 0, 2).reshape(s.shape[0], -1)
        s = s + self.attn_out(out)

        outer = self.outer_left(s)[:, None, :] + self.outer_right(s)[None, :, :]
        z = z + self.pair_update(outer)

        coords = coords + torch.einsum(
            "ij,ijc->ic", torch.softmax(z.mean(-1), dim=-1), self.frame_from_pair(z)
        )
        torsions = torch.tanh(self.torsion_head(s))
        return s, z, coords, torsions


class RoseTTAFold2NA(nn.Module):
    """Compact RF2NA: protein+nucleic-acid three-track network."""

    def __init__(self, c_1d: int = 16, c_2d: int = 16, n_block: int = 2, n_token: int = 25) -> None:
        """Build RF2NA.

        Parameters
        ----------
        c_1d:
            1D track channel width.
        c_2d:
            2D track channel width.
        n_block:
            Number of three-track blocks.
        n_token:
            Vocabulary size: 20 amino acids + 5 nucleotide tokens (A/C/G/U/T).
        """

        super().__init__()
        self.embed_1d = nn.Embedding(n_token, c_1d)
        self.left_2d = nn.Embedding(n_token, c_2d)
        self.right_2d = nn.Embedding(n_token, c_2d)
        self.blocks = nn.ModuleList([ThreeTrackBlockNA(c_1d, c_2d) for _ in range(n_block)])

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict per-residue/nucleotide coordinate frames and torsions.

        Parameters
        ----------
        tokens:
            Integer sequence tensor of shape ``(N,)`` mixing protein and
            nucleic-acid tokens.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(coords, torsions)`` of shapes ``(N, 3)`` and ``(N, 10)``.
        """

        s = self.embed_1d(tokens)
        z = self.left_2d(tokens)[:, None, :] + self.right_2d(tokens)[None, :, :]
        coords = torch.zeros(s.shape[0], 3, device=s.device, dtype=s.dtype)
        torsions = torch.zeros(s.shape[0], 10, device=s.device, dtype=s.dtype)
        for block in self.blocks:
            s, z, coords, torsions = block(s, z, coords)
        return coords, torsions


def build_rf2na() -> nn.Module:
    """Build compact RF2NA.

    Returns
    -------
    nn.Module
        Random-initialized RF2NA in eval mode.
    """

    return RoseTTAFold2NA().eval()


def example_input_rf2na() -> torch.Tensor:
    """Create a small mixed protein/nucleotide token sequence.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape ``(14,)`` with tokens in ``[0, 25)``.
    """

    return torch.randint(0, 25, (14,))


# ---------------------------------------------------------------------------
# 4. RfamGen: family-conditioned VAE over an RNA (transcript, secondary
#    structure, base-pair) triplet, encoding a covariance-model-style
#    consensus profile into a latent z that decodes to per-column emission
#    and per-pair base-pairing probabilities.
# ---------------------------------------------------------------------------


class RfamGen(nn.Module):
    """Compact RfamGen: (sequence, structure, base-pair) triplet VAE."""

    def __init__(
        self,
        n_col: int = 20,
        vocab: int = 5,
        c_hidden: int = 24,
        latent: int = 16,
    ) -> None:
        """Build the RfamGen family-conditioned VAE.

        Parameters
        ----------
        n_col:
            Number of consensus alignment columns (covariance-model match states).
        vocab:
            Nucleotide vocabulary size (A, C, G, U, gap).
        c_hidden:
            Encoder/decoder hidden width.
        latent:
            Latent dimensionality ``z``.
        """

        super().__init__()
        self.n_col = n_col
        self.vocab = vocab
        # encoder: consumes one-hot transcript + secondary-structure mask + base-pair mask
        in_dim = n_col * (vocab + 1)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, c_hidden), nn.ReLU(), nn.Linear(c_hidden, c_hidden), nn.ReLU()
        )
        self.mu = nn.Linear(c_hidden, latent)
        self.logvar = nn.Linear(c_hidden, latent)
        # decoder: latent -> per-column emission probabilities (covariance-model
        # match-state distribution) + per-pair base-pairing compatibility logits
        self.dec = nn.Sequential(
            nn.Linear(latent, c_hidden), nn.ReLU(), nn.Linear(c_hidden, c_hidden), nn.ReLU()
        )
        self.emission_head = nn.Linear(c_hidden, n_col * vocab)
        self.pair_head = nn.Linear(c_hidden, n_col * n_col)

    def forward(
        self, tr: torch.Tensor, ss: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a (transcript, secondary-structure) pair and decode a family sample.

        Parameters
        ----------
        tr:
            One-hot transcript tensor of shape ``(n_col, vocab)``.
        ss:
            Secondary-structure paired-column indicator of shape ``(n_col, 1)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``(emission_logits, pair_logits, mu, logvar)``: emission of shape
            ``(n_col, vocab)``, pair compatibility of shape ``(n_col, n_col)``.
        """

        x = torch.cat([tr, ss], dim=-1).reshape(-1)
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        d = self.dec(z)
        emission_logits = self.emission_head(d).view(self.n_col, self.vocab)
        pair_logits = self.pair_head(d).view(self.n_col, self.n_col)
        return emission_logits, pair_logits, mu, logvar


def build_rfamgen() -> nn.Module:
    """Build compact RfamGen.

    Returns
    -------
    nn.Module
        Random-initialized RfamGen in eval mode.
    """

    return RfamGen().eval()


def example_input_rfamgen() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small one-hot transcript + secondary-structure mask pair.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(tr, ss)`` of shapes ``(20, 5)`` and ``(20, 1)``.
    """

    tr = F.one_hot(torch.randint(0, 5, (20,)), num_classes=5).float()
    ss = torch.randint(0, 2, (20, 1)).float()
    return tr, ss


# ---------------------------------------------------------------------------
# 5. RFAA (RFdiffusion All-Atom): hybrid residue-frame + atom-node
#    representation with a DDPM coordinate denoiser conditioned on a fixed
#    small-molecule motif.
# ---------------------------------------------------------------------------


class HybridPairUpdateBlock(nn.Module):
    """Three-track-style pair/coordinate update over a mixed residue+atom node set."""

    def __init__(self, c_1d: int, c_2d: int, n_head: int = 4) -> None:
        """Build a hybrid residue/atom pair-update block.

        Parameters
        ----------
        c_1d:
            Per-node (residue-frame or atom) channel width.
        c_2d:
            Pair channel width.
        n_head:
            Attention head count.
        """

        super().__init__()
        self.h = n_head
        self.c = c_1d // n_head
        self.norm = nn.LayerNorm(c_1d)
        self.q = nn.Linear(c_1d, c_1d, bias=False)
        self.k = nn.Linear(c_1d, c_1d, bias=False)
        self.v = nn.Linear(c_1d, c_1d, bias=False)
        self.pair_bias = nn.Linear(c_2d, n_head, bias=False)
        self.attn_out = nn.Linear(c_1d, c_1d)
        self.outer_left = nn.Linear(c_1d, c_2d)
        self.outer_right = nn.Linear(c_1d, c_2d)
        self.pair_update = nn.Sequential(nn.LayerNorm(c_2d), nn.Linear(c_2d, c_2d), nn.ReLU())
        self.coord_from_pair = nn.Linear(c_2d, 3)

    def forward(
        self, s: torch.Tensor, z: torch.Tensor, coords: torch.Tensor, is_motif: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update node/pair features and denoise coordinates, holding the motif fixed.

        Parameters
        ----------
        s:
            Per-node representation of shape ``(N, c_1d)``.
        z:
            Pair representation of shape ``(N, N, c_2d)``.
        coords:
            Node coordinates of shape ``(N, 3)``.
        is_motif:
            Boolean mask of shape ``(N, 1)``; ``True`` marks the fixed
            small-molecule motif whose coordinates are never updated.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated ``(s, z, coords)``.
        """

        h = self.norm(s)
        q = _heads(self.q(h), self.h).permute(1, 0, 2)
        k = _heads(self.k(h), self.h).permute(1, 0, 2)
        v = _heads(self.v(h), self.h).permute(1, 0, 2)
        bias = self.pair_bias(z).permute(2, 0, 1)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.c**0.5) + bias, dim=-1)
        out = torch.matmul(att, v).permute(1, 0, 2).reshape(s.shape[0], -1)
        s = s + self.attn_out(out)

        outer = self.outer_left(s)[:, None, :] + self.outer_right(s)[None, :, :]
        z = z + self.pair_update(outer)

        delta = torch.einsum(
            "ij,ijc->ic", torch.softmax(z.mean(-1), dim=-1), self.coord_from_pair(z)
        )
        # motif atoms are held fixed (conditioning context, not denoised)
        coords = coords + delta * (~is_motif).float()
        return s, z, coords


class RFdiffusionAllAtom(nn.Module):
    """Compact RFAA: residue+atom hybrid nodes, DDPM denoising around a fixed motif."""

    def __init__(
        self, c_1d: int = 16, c_2d: int = 16, n_block: int = 2, n_node_type: int = 26
    ) -> None:
        """Build the RFdiffusion All-Atom model.

        Parameters
        ----------
        c_1d:
            Per-node channel width.
        c_2d:
            Pair channel width.
        n_block:
            Number of hybrid pair-update blocks.
        n_node_type:
            Vocabulary size: 20 amino acids + small-molecule atom element types.
        """

        super().__init__()
        self.embed = nn.Embedding(n_node_type, c_1d)
        self.left = nn.Embedding(n_node_type, c_2d)
        self.right = nn.Embedding(n_node_type, c_2d)
        self.time_embed = nn.Sequential(nn.Linear(1, c_1d), nn.SiLU(), nn.Linear(c_1d, c_1d))
        self.blocks = nn.ModuleList([HybridPairUpdateBlock(c_1d, c_2d) for _ in range(n_block)])

    def forward(
        self,
        node_types: torch.Tensor,
        coords: torch.Tensor,
        is_motif: torch.Tensor,
        t: float = 0.5,
    ) -> torch.Tensor:
        """Denoise protein coordinates around a fixed small-molecule motif.

        Parameters
        ----------
        node_types:
            Integer node-type tensor of shape ``(N,)`` (protein residues + motif atoms).
        coords:
            Noised coordinates of shape ``(N, 3)``.
        is_motif:
            Boolean mask of shape ``(N, 1)``; ``True`` marks fixed motif atoms.
        t:
            Diffusion timestep scalar.

        Returns
        -------
        torch.Tensor
            Denoised coordinates of shape ``(N, 3)``.
        """

        s = self.embed(node_types)
        tt = torch.full((s.shape[0], 1), float(t), device=s.device, dtype=s.dtype)
        s = s + self.time_embed(tt)
        z = self.left(node_types)[:, None, :] + self.right(node_types)[None, :, :]
        for block in self.blocks:
            s, z, coords = block(s, z, coords, is_motif)
        return coords


def build_rfaa() -> nn.Module:
    """Build compact RFdiffusion All-Atom (RFAA).

    Returns
    -------
    nn.Module
        Random-initialized RFAA in eval mode.
    """

    return RFdiffusionAllAtom().eval()


def example_input_rfaa() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a small mixed protein-residue + motif-atom system.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(node_types, coords, is_motif)`` for 10 protein residues + 4 motif atoms.
    """

    node_types = torch.randint(0, 26, (14,))
    coords = torch.randn(14, 3)
    is_motif = torch.zeros(14, 1, dtype=torch.bool)
    is_motif[10:] = True
    return node_types, coords, is_motif


# ---------------------------------------------------------------------------
# 6. RoseTTAFold2: protein-only three-track network (1D MSA/seq, 2D pair,
#    3D coordinate) with a frame-aligned-point-error-style structure module;
#    no nucleic-acid or ligand extension (contrast with RF2NA / RFAA above).
# ---------------------------------------------------------------------------


class ThreeTrackBlock(nn.Module):
    """One RoseTTAFold2 three-track block with a frame-update structure step."""

    def __init__(self, c_1d: int, c_2d: int, n_head: int = 4) -> None:
        """Build a protein-only three-track update block.

        Parameters
        ----------
        c_1d:
            1D sequence-track channel width.
        c_2d:
            2D pair-track channel width.
        n_head:
            Attention heads for the 1D self-attention.
        """

        super().__init__()
        self.h = n_head
        self.c = c_1d // n_head
        self.norm_1d = nn.LayerNorm(c_1d)
        self.q = nn.Linear(c_1d, c_1d, bias=False)
        self.k = nn.Linear(c_1d, c_1d, bias=False)
        self.v = nn.Linear(c_1d, c_1d, bias=False)
        self.pair_bias = nn.Linear(c_2d, n_head, bias=False)
        self.attn_out = nn.Linear(c_1d, c_1d)
        self.outer_left = nn.Linear(c_1d, c_2d)
        self.outer_right = nn.Linear(c_1d, c_2d)
        self.pair_update = nn.Sequential(nn.LayerNorm(c_2d), nn.Linear(c_2d, c_2d), nn.ReLU())
        # frame-aligned structure step: rotation (as a 3-vector) + translation update
        self.rotation_head = nn.Linear(c_1d, 3)
        self.translation_head = nn.Linear(c_2d, 3)

    def forward(
        self, s: torch.Tensor, z: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Jointly refine sequence, pair, and frame-aligned coordinate tracks.

        Parameters
        ----------
        s:
            1D sequence representation, shape ``(N, c_1d)``.
        z:
            2D pair representation, shape ``(N, N, c_2d)``.
        coords:
            3D residue-frame coordinates, shape ``(N, 3)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated ``(s, z, coords)``.
        """

        h = self.norm_1d(s)
        q = _heads(self.q(h), self.h).permute(1, 0, 2)
        k = _heads(self.k(h), self.h).permute(1, 0, 2)
        v = _heads(self.v(h), self.h).permute(1, 0, 2)
        bias = self.pair_bias(z).permute(2, 0, 1)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.c**0.5) + bias, dim=-1)
        out = torch.matmul(att, v).permute(1, 0, 2).reshape(s.shape[0], -1)
        s = s + self.attn_out(out)

        outer = self.outer_left(s)[:, None, :] + self.outer_right(s)[None, :, :]
        z = z + self.pair_update(outer)

        # frame-aligned point update: rotate by a small per-residue rotation
        # vector then translate by pair-context-weighted translation
        rot = self.rotation_head(s)
        translation = torch.einsum(
            "ij,ijc->ic", torch.softmax(z.mean(-1), dim=-1), self.translation_head(z)
        )
        coords = coords + rot + translation
        return s, z, coords


class RoseTTAFold2(nn.Module):
    """Compact RoseTTAFold2: protein-only three-track network, 36-block design (compacted)."""

    def __init__(self, c_1d: int = 16, c_2d: int = 16, n_block: int = 3, n_token: int = 21) -> None:
        """Build RoseTTAFold2.

        Parameters
        ----------
        c_1d:
            1D track channel width.
        c_2d:
            2D track channel width.
        n_block:
            Number of three-track blocks (compacted from the published 36).
        n_token:
            Vocabulary size: 20 amino acids + 1 unknown/gap token.
        """

        super().__init__()
        self.embed_1d = nn.Embedding(n_token, c_1d)
        self.left_2d = nn.Embedding(n_token, c_2d)
        self.right_2d = nn.Embedding(n_token, c_2d)
        self.blocks = nn.ModuleList([ThreeTrackBlock(c_1d, c_2d) for _ in range(n_block)])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict per-residue backbone-frame coordinates.

        Parameters
        ----------
        tokens:
            Integer amino-acid sequence tensor of shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Coordinates of shape ``(N, 3)``.
        """

        s = self.embed_1d(tokens)
        z = self.left_2d(tokens)[:, None, :] + self.right_2d(tokens)[None, :, :]
        coords = torch.zeros(s.shape[0], 3, device=s.device, dtype=s.dtype)
        for block in self.blocks:
            s, z, coords = block(s, z, coords)
        return coords


def build_rosettafold2() -> nn.Module:
    """Build compact RoseTTAFold2.

    Returns
    -------
    nn.Module
        Random-initialized RoseTTAFold2 in eval mode.
    """

    return RoseTTAFold2().eval()


def example_input_rosettafold2() -> torch.Tensor:
    """Create a small protein sequence tensor for RoseTTAFold2.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape ``(12,)`` with tokens in ``[0, 21)``.
    """

    return torch.randint(0, 21, (12,))


MENAGERIE_ENTRIES = [
    ("ProtenixFold", "build_protenixfold", "example_input_protenixfold", "2025", "BIO"),
    ("PubMedBERT", "build_pubmedbert", "example_input_pubmedbert", "2020", "NLP"),
    ("RF2NA", "build_rf2na", "example_input_rf2na", "2023", "BIO"),
    ("RfamGen", "build_rfamgen", "example_input_rfamgen", "2024", "BIO"),
    ("RFdiffusion-All-Atom (RFAA)", "build_rfaa", "example_input_rfaa", "2024", "BIO"),
    ("RoseTTAFold2", "build_rosettafold2", "example_input_rosettafold2", "2023", "BIO"),
]
