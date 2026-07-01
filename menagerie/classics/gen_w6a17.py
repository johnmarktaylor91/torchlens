"""Antibody/structural-biology and molecular-ML architecture family: gen_w6a17.

Sources checked (repo_url / desc_source from the build queue, web search for
architecture details where the repo itself ships only weights):
  - AbMPNN: https://github.com/Graylab/IgFold (weights at zenodo.org/records/8164693);
    AbMPNN is ProteinMPNN's residue-graph message-passing encoder/decoder fine-tuned
    on antibody structures -- same architecture as ProteinMPNN, antibody-specific data.
  - AbNatiV: https://github.com/oxpig/AbNatiV; Ramon et al., Nat. Mach. Intell. 2023.
    A VQ-VAE over aligned (IMGT-numbered) Fv sequence positions: positional one-hot
    encoder -> shared codebook vector quantization per position -> decoder back to
    per-position amino-acid distribution, trained with masked reconstruction.
  - ABodyBuilder2: https://github.com/brennanaba/ImmuneBuilder (part of ImmuneBuilder);
    Abanades et al., Bioinformatics 2024. A modified AlphaFold-Multimer trunk: paired
    heavy/light sequence embedding -> pairwise features -> a stack of shared-weight
    Invariant Point Attention structure-module blocks producing backbone frames
    (this reimplementation reuses the IPA/structure-module primitive established in
    menagerie/classics/openfold_af2.py, at antibody-appropriate small scale).
  - AlphaFlow: https://github.com/bjing2016/alphaflow; Jing, Berger, Jaakkola, ICML 2024
    (arXiv:2402.04845). Repurposes the AlphaFold trunk (Evoformer + structure module)
    as the vector field of a flow-matching / stochastic-interpolant generative model:
    the trunk takes a noised/interpolated structure at flow-time t (via a time
    embedding folded into the single representation) and predicts a denoised
    structure; sampling integrates the ODE from a harmonic-prior noise sample.
  - ANI-2x: https://github.com/aiqm/torchani; Behler-Parrinello atomic neural-network
    potential. Atomic Environment Vectors (AEV) built from radial (2-body) and
    angular (3-body) symmetry functions around each atom (rotation/translation
    invariant, smoothly cutoff), fed into a per-element feed-forward "atomic
    network" whose scalar outputs sum to the total molecular energy.
  - AntiFold: https://github.com/oxpig/AntiFold; Bioinformatics Advances 2024
    (arXiv:2405.03370). ESM-IF1 fine-tuned for antibodies: a Geometric Vector
    Perceptron (GVP) graph encoder over backbone-derived scalar+vector node/edge
    features (coordinates, orientations) followed by an autoregressive
    sequence-recovery decoder over the encoded structural context.

All models below are compact, randomly initialized, faithful reimplementations of
each architecture's distinctive mechanism (not generic MLP/transformer stubs), sized
small so tracing and rendering stay fast.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# AbMPNN -- ProteinMPNN's residue-graph message passing, antibody fine-tune
# ---------------------------------------------------------------------------


def _knn_edges(ca: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Build a residue k-nearest-neighbor graph from C-alpha coordinates.

    Parameters
    ----------
    ca:
        C-alpha coordinates ``(B, L, 3)``.
    k:
        Number of neighbors per residue.

    Returns
    -------
    tuple[Tensor, Tensor]
        Neighbor indices ``(B, L, K)`` and neighbor distances ``(B, L, K)``.
    """

    dist = torch.cdist(ca, ca)
    vals, idx = dist.topk(k + 1, largest=False)
    return idx[:, :, 1:], vals[:, :, 1:]


def _gather_nodes(nodes: Tensor, idx: Tensor) -> Tensor:
    """Gather node features at neighbor indices.

    Parameters
    ----------
    nodes:
        Node features ``(B, L, C)``.
    idx:
        Neighbor indices ``(B, L, K)``.

    Returns
    -------
    Tensor
        Neighbor features ``(B, L, K, C)``.
    """

    batch, length, channels = nodes.shape
    flat = nodes.reshape(batch * length, channels)
    offset = torch.arange(batch, device=nodes.device).view(batch, 1, 1) * length
    return flat[(idx + offset).reshape(-1)].view(batch, length, idx.shape[-1], channels)


class _MPNNLayer(nn.Module):
    """Edge-conditioned residue message-passing layer (ProteinMPNN-style)."""

    def __init__(self, dim: int) -> None:
        """Initialize message and edge-update projections.

        Parameters
        ----------
        dim:
            Hidden feature size.
        """

        super().__init__()
        self.msg = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.edge = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.node_norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(dim)

    def forward(self, node: Tensor, edge: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
        """Update node and edge features via one message-passing round.

        Parameters
        ----------
        node:
            Residue features ``(B, L, C)``.
        edge:
            Edge features ``(B, L, K, C)``.
        idx:
            Neighbor indices ``(B, L, K)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated node and edge features.
        """

        neigh = _gather_nodes(node, idx)
        src = node.unsqueeze(2).expand_as(neigh)
        messages = self.msg(torch.cat([src, neigh, edge], dim=-1))
        node = self.node_norm(node + messages.mean(dim=2))
        edge = self.edge_norm(edge + self.edge(torch.cat([edge, messages], dim=-1)))
        return node, edge


class AbMPNN(nn.Module):
    """Compact ProteinMPNN-architecture sequence designer, antibody fine-tune."""

    def __init__(self, dim: int = 32, k: int = 6, layers: int = 3, vocab: int = 21) -> None:
        """Initialize backbone featurizer, encoder stack, and decoder.

        Parameters
        ----------
        dim:
            Hidden width.
        k:
            Residue graph degree.
        layers:
            Number of encoder message-passing blocks.
        vocab:
            Amino-acid vocabulary size.
        """

        super().__init__()
        self.k = k
        self.node_in = nn.Linear(9, dim)
        self.edge_in = nn.Linear(8, dim)
        self.chain_embed = nn.Embedding(2, dim)  # heavy/light chain identity
        self.layers = nn.ModuleList([_MPNNLayer(dim) for _ in range(layers)])
        self.seq_embed = nn.Embedding(vocab, dim)
        self.decoder = _MPNNLayer(dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, backbone: Tensor, tokens: Tensor, chain_id: Tensor) -> Tensor:
        """Predict amino-acid logits from backbone coordinates and chain identity.

        Parameters
        ----------
        backbone:
            Backbone atom coordinates ``(B, L, 3, 3)`` for N, CA, C.
        tokens:
            Teacher-forced amino-acid ids ``(B, L)``.
        chain_id:
            Heavy(0)/light(1) chain id per residue ``(B, L)``.

        Returns
        -------
        Tensor
            Amino-acid logits ``(B, L, vocab)``.
        """

        n_coord, ca, c_coord = backbone.unbind(dim=2)
        idx, dist = _knn_edges(ca, self.k)
        forward_vec = F.normalize(c_coord - ca, dim=-1)
        backward_vec = F.normalize(n_coord - ca, dim=-1)
        node = self.node_in(torch.cat([ca, forward_vec, backward_vec], dim=-1))
        node = node + self.chain_embed(chain_id)
        neigh_ca = _gather_nodes(ca, idx)
        rel = neigh_ca - ca.unsqueeze(2)
        edge = self.edge_in(
            torch.cat([dist.unsqueeze(-1), rel, rel.abs(), rel.norm(dim=-1, keepdim=True)], dim=-1)
        )
        for layer in self.layers:
            node, edge = layer(node, edge, idx)
        causal = torch.tril(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device))
        neighbor_mask = torch.gather(causal.unsqueeze(0).expand(tokens.shape[0], -1, -1), 2, idx)
        seq_context = _gather_nodes(self.seq_embed(tokens), idx) * neighbor_mask.unsqueeze(-1)
        node = node + seq_context.mean(dim=2)
        node, _ = self.decoder(node, edge, idx)
        return self.out(node)


def build_abmpnn() -> nn.Module:
    """Build a compact random-init AbMPNN.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return AbMPNN().eval()


def example_input_abmpnn() -> tuple[Tensor, Tensor, Tensor]:
    """Return a small paired-chain Fv backbone, tokens, and chain ids.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Backbone coordinates, amino-acid tokens, and chain ids.
    """

    length = 14
    ca = torch.randn(1, length, 3).cumsum(dim=1)
    n_coord = ca + torch.tensor([0.2, 0.0, 0.0])
    c_coord = ca + torch.tensor([-0.2, 0.1, 0.0])
    backbone = torch.stack([n_coord, ca, c_coord], dim=2)
    tokens = torch.randint(0, 21, (1, length))
    chain_id = torch.cat(
        [
            torch.zeros(1, length // 2, dtype=torch.long),
            torch.ones(1, length - length // 2, dtype=torch.long),
        ],
        dim=1,
    )
    return backbone, tokens, chain_id


# ---------------------------------------------------------------------------
# AbNatiV -- VQ-VAE antibody/nanobody naturalness scorer
# ---------------------------------------------------------------------------


class _VectorQuantizer(nn.Module):
    """Straight-through vector quantizer over a shared codebook."""

    def __init__(self, n_codes: int, dim: int) -> None:
        """Initialize the codebook.

        Parameters
        ----------
        n_codes:
            Number of codebook entries.
        dim:
            Latent embedding dimension.
        """

        super().__init__()
        self.codebook = nn.Embedding(n_codes, dim)

    def forward(self, z: Tensor) -> Tensor:
        """Quantize latent vectors to their nearest codebook entry.

        Parameters
        ----------
        z:
            Continuous latents ``(..., dim)``.

        Returns
        -------
        Tensor
            Quantized latents (straight-through estimator applied), same shape as ``z``.
        """

        flat = z.reshape(-1, z.shape[-1])
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1)
        )
        idx = dist.argmin(dim=1)
        quantized = self.codebook(idx).view_as(z)
        return z + (quantized - z).detach()


class AbNatiV(nn.Module):
    """Positional VQ-VAE naturalness scorer for aligned antibody/nanobody Fv sequences."""

    def __init__(
        self,
        n_positions: int = 30,
        vocab: int = 21,
        dim: int = 32,
        n_codes: int = 64,
    ) -> None:
        """Initialize positional embedding, encoder, quantizer, and decoder.

        Parameters
        ----------
        n_positions:
            Number of IMGT-aligned Fv positions modeled.
        vocab:
            Amino-acid (+gap) vocabulary size.
        dim:
            Latent embedding width.
        n_codes:
            Codebook size.
        """

        super().__init__()
        self.n_positions = n_positions
        self.aa_embed = nn.Embedding(vocab, dim)
        self.pos_embed = nn.Parameter(torch.randn(n_positions, dim) * 0.02)
        self.encoder = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.quantizer = _VectorQuantizer(n_codes, dim)
        self.decoder = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, vocab))

    def forward(self, aligned_seq: Tensor) -> Tensor:
        """Reconstruct per-position amino-acid distributions (naturalness score input).

        Parameters
        ----------
        aligned_seq:
            IMGT-aligned amino-acid ids ``(B, n_positions)``.

        Returns
        -------
        Tensor
            Reconstructed per-position logits ``(B, n_positions, vocab)``.
        """

        z = self.aa_embed(aligned_seq) + self.pos_embed.unsqueeze(0)
        z = self.encoder(z)
        zq = self.quantizer(z)
        return self.decoder(zq)


def build_abnativ() -> nn.Module:
    """Build a compact random-init AbNatiV VQ-VAE.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return AbNatiV().eval()


def example_input_abnativ() -> Tensor:
    """Return a small IMGT-aligned Fv sequence batch.

    Returns
    -------
    Tensor
        Aligned amino-acid token ids ``(2, 30)``.
    """

    return torch.randint(0, 21, (2, 30))


# ---------------------------------------------------------------------------
# ABodyBuilder2 -- AlphaFold-Multimer-style structure module for paired Fv
# ---------------------------------------------------------------------------


class _InvariantPointAttention(nn.Module):
    """Invariant Point Attention: scalar + pair-bias + 3D-point attention terms."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int = 8,
        n_head: int = 4,
        n_qk_points: int = 4,
        n_v_points: int = 4,
    ) -> None:
        """Initialize scalar, point, and pair-bias attention projections.

        Parameters
        ----------
        c_s:
            Single (per-residue) representation width.
        c_z:
            Pair representation width.
        c_hidden:
            Per-head scalar attention width.
        n_head:
            Number of attention heads.
        n_qk_points:
            Query/key 3D point count per head.
        n_v_points:
            Value 3D point count per head.
        """

        super().__init__()
        self.n_head = n_head
        self.c_hidden = c_hidden
        self.n_qk_points = n_qk_points
        self.n_v_points = n_v_points
        self.linear_q = nn.Linear(c_s, c_hidden * n_head)
        self.linear_kv = nn.Linear(c_s, 2 * c_hidden * n_head)
        self.linear_q_points = nn.Linear(c_s, n_head * n_qk_points * 3)
        self.linear_kv_points = nn.Linear(c_s, n_head * (n_qk_points + n_v_points) * 3)
        self.linear_b = nn.Linear(c_z, n_head)
        self.head_weight = nn.Parameter(torch.zeros(n_head))
        concat = n_head * (c_hidden + c_z + n_v_points * 4)
        self.linear_o = nn.Linear(concat, c_s)

    def forward(self, s: Tensor, z: Tensor, rot: Tensor, trans: Tensor) -> Tensor:
        """Attend over residues using scalar, pair, and rigid-frame point terms.

        Parameters
        ----------
        s:
            Single representation ``(R, c_s)``.
        z:
            Pair representation ``(R, R, c_z)``.
        rot:
            Per-residue frame rotation matrices ``(R, 3, 3)``.
        trans:
            Per-residue frame translations ``(R, 3)``.

        Returns
        -------
        Tensor
            Updated single representation ``(R, c_s)``.
        """

        n_res = s.shape[0]
        h, c = self.n_head, self.c_hidden
        q = self.linear_q(s).view(n_res, h, c)
        kv = self.linear_kv(s).view(n_res, h, 2 * c)
        k, v = kv.split(c, dim=-1)

        def to_global(local: Tensor) -> Tensor:
            return torch.einsum("rij,rhpj->rhpi", rot, local) + trans[:, None, None, :]

        qp = self.linear_q_points(s).view(n_res, h, self.n_qk_points, 3)
        kvp = self.linear_kv_points(s).view(n_res, h, self.n_qk_points + self.n_v_points, 3)
        kp, vp = kvp.split([self.n_qk_points, self.n_v_points], dim=2)
        qp_g = to_global(qp)
        kp_g = to_global(kp)
        vp_g = to_global(vp)

        a_scalar = torch.einsum("ihc,jhc->hij", q, k) / (c**0.5)
        b = self.linear_b(z).permute(2, 0, 1)
        diff = qp_g[:, None] - kp_g[None]
        sq = diff.pow(2).sum(dim=(-1, -2)).permute(2, 0, 1)
        gamma = F.softplus(self.head_weight).view(h, 1, 1)
        a = a_scalar * (1.0 / 3.0) ** 0.5 + b * (1.0 / 3.0) ** 0.5 - 0.5 * gamma * sq
        a = torch.softmax(a, dim=-1)

        o_scalar = torch.einsum("hij,jhc->ihc", a, v).reshape(n_res, -1)
        o_pair = torch.einsum("hij,ijc->ihc", a, z).reshape(n_res, -1)
        o_pt_g = torch.einsum("hij,jhpk->ihpk", a, vp_g)
        rot_t = rot.transpose(-1, -2)
        o_pt_local = torch.einsum("rij,rhpj->rhpi", rot_t, o_pt_g - trans[:, None, None, :])
        o_pt_norm = torch.sqrt(o_pt_local.pow(2).sum(-1) + 1e-8)
        o_pt = torch.cat([o_pt_local.reshape(n_res, -1), o_pt_norm.reshape(n_res, -1)], dim=-1)

        out = torch.cat([o_scalar, o_pair, o_pt], dim=-1)
        return self.linear_o(out)


class _AntibodyStructureModule(nn.Module):
    """Shared-weight IPA iterations producing paired-chain Fv backbone frames."""

    def __init__(self, c_s: int, c_z: int, n_iter: int = 4) -> None:
        """Initialize normalization, IPA, transition, and backbone-update heads.

        Parameters
        ----------
        c_s:
            Single representation width.
        c_z:
            Pair representation width.
        n_iter:
            Number of shared-weight structure-module iterations.
        """

        super().__init__()
        self.n_iter = n_iter
        self.norm_s = nn.LayerNorm(c_s)
        self.norm_z = nn.LayerNorm(c_z)
        self.linear_in = nn.Linear(c_s, c_s)
        self.ipa = _InvariantPointAttention(c_s, c_z)
        self.norm_ipa = nn.LayerNorm(c_s)
        self.transition = nn.Sequential(
            nn.Linear(c_s, c_s), nn.ReLU(), nn.Linear(c_s, c_s), nn.ReLU(), nn.Linear(c_s, c_s)
        )
        self.norm_trans = nn.LayerNorm(c_s)
        self.bb_update = nn.Linear(c_s, 6)

    @staticmethod
    def _axis_angle_to_matrix(v: Tensor) -> Tensor:
        """Convert an axis-angle vector to a rotation matrix via Rodrigues' formula.

        Parameters
        ----------
        v:
            Axis-angle vectors ``(R, 3)``.

        Returns
        -------
        Tensor
            Rotation matrices ``(R, 3, 3)``.
        """

        theta = torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-8
        k = v / theta
        skew = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
        skew[:, 0, 1], skew[:, 0, 2] = -k[:, 2], k[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = k[:, 2], -k[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -k[:, 1], k[:, 0]
        eye = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0)
        th = theta.unsqueeze(-1)
        return eye + torch.sin(th) * skew + (1 - torch.cos(th)) * torch.matmul(skew, skew)

    def forward(self, s: Tensor, z: Tensor) -> Tensor:
        """Iteratively refine per-residue rigid frames from single/pair features.

        Parameters
        ----------
        s:
            Single representation ``(R, c_s)``.
        z:
            Pair representation ``(R, R, c_z)``.

        Returns
        -------
        Tensor
            Predicted CA coordinates ``(R, 3)``.
        """

        s = self.linear_in(self.norm_s(s))
        z = self.norm_z(z)
        n_res = s.shape[0]
        rot = torch.eye(3, device=s.device, dtype=s.dtype).unsqueeze(0).repeat(n_res, 1, 1)
        trans = torch.zeros(n_res, 3, device=s.device, dtype=s.dtype)
        for _ in range(self.n_iter):
            s = s + self.ipa(s, z, rot, trans)
            s = self.norm_ipa(s)
            s = s + self.transition(s)
            s = self.norm_trans(s)
            upd = self.bb_update(s)
            d_rot = self._axis_angle_to_matrix(upd[:, :3])
            rot = torch.matmul(rot, d_rot)
            trans = trans + torch.einsum("rij,rj->ri", rot, upd[:, 3:])
        return trans


class ABodyBuilder2(nn.Module):
    """Compact AlphaFold-Multimer-style structure predictor for paired VH/VL Fv."""

    def __init__(self, c_s: int = 24, c_z: int = 16, n_token: int = 21, n_iter: int = 4) -> None:
        """Initialize sequence/pair embeddings and the structure module.

        Parameters
        ----------
        c_s:
            Single representation width.
        c_z:
            Pair representation width.
        n_token:
            Amino-acid vocabulary size.
        n_iter:
            Structure module iterations.
        """

        super().__init__()
        self.seq_embed = nn.Embedding(n_token, c_s)
        self.chain_embed = nn.Embedding(2, c_s)
        self.left_embed = nn.Embedding(n_token, c_z)
        self.right_embed = nn.Embedding(n_token, c_z)
        self.structure = _AntibodyStructureModule(c_s, c_z, n_iter=n_iter)

    def forward(self, seq: Tensor, chain_id: Tensor) -> Tensor:
        """Predict CA backbone coordinates for a paired heavy+light Fv sequence.

        Parameters
        ----------
        seq:
            Concatenated heavy+light chain amino-acid ids ``(R,)``.
        chain_id:
            Heavy(0)/light(1) chain id per residue ``(R,)``.

        Returns
        -------
        Tensor
            Predicted CA coordinates ``(R, 3)``.
        """

        s = self.seq_embed(seq) + self.chain_embed(chain_id)
        z = self.left_embed(seq)[:, None, :] + self.right_embed(seq)[None, :, :]
        same_chain = (chain_id[:, None] == chain_id[None, :]).float().unsqueeze(-1)
        z = z * (0.5 + 0.5 * same_chain)
        return self.structure(s, z)


def build_abodybuilder2() -> nn.Module:
    """Build a compact random-init ABodyBuilder2.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return ABodyBuilder2().eval()


def example_input_abodybuilder2() -> tuple[Tensor, Tensor]:
    """Return a small paired heavy+light chain sequence and chain ids.

    Returns
    -------
    tuple[Tensor, Tensor]
        Amino-acid token ids and chain ids, each ``(R,)``.
    """

    heavy_len, light_len = 8, 7
    seq = torch.randint(0, 21, (heavy_len + light_len,))
    chain_id = torch.cat(
        [torch.zeros(heavy_len, dtype=torch.long), torch.ones(light_len, dtype=torch.long)]
    )
    return seq, chain_id


# ---------------------------------------------------------------------------
# AlphaFlow -- AlphaFold trunk repurposed as a flow-matching vector field
# ---------------------------------------------------------------------------


class _MSARowAttention(nn.Module):
    """MSA row-wise gated self-attention biased by the pair representation."""

    def __init__(self, c_m: int, c_z: int, n_head: int = 4) -> None:
        """Initialize per-head projections and pair bias.

        Parameters
        ----------
        c_m:
            MSA representation width.
        c_z:
            Pair representation width.
        n_head:
            Number of attention heads.
        """

        super().__init__()
        self.n_head = n_head
        self.head_dim = c_m // n_head
        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)
        self.qkv = nn.Linear(c_m, 3 * c_m)
        self.pair_bias = nn.Linear(c_z, n_head, bias=False)
        self.gate = nn.Linear(c_m, c_m)
        self.out = nn.Linear(c_m, c_m)

    def forward(self, m: Tensor, z: Tensor) -> Tensor:
        """Apply row-wise attention within each MSA sequence, biased by pair features.

        Parameters
        ----------
        m:
            MSA representation ``(S, R, c_m)``.
        z:
            Pair representation ``(R, R, c_z)``.

        Returns
        -------
        Tensor
            Updated MSA representation ``(S, R, c_m)``.
        """

        s_dim, r_dim, _ = m.shape
        mn = self.norm_m(m)
        qkv = self.qkv(mn).view(s_dim, r_dim, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        bias = self.pair_bias(self.norm_z(z)).permute(2, 0, 1)
        logits = torch.einsum("sihd,sjhd->hsij", q, k) / (self.head_dim**0.5) + bias.unsqueeze(1)
        attn = torch.softmax(logits, dim=-1)
        ctx = torch.einsum("hsij,sjhd->sihd", attn, v).reshape(s_dim, r_dim, -1)
        gate = torch.sigmoid(self.gate(mn))
        return self.out(ctx * gate)


class _Transition(nn.Module):
    """Two-layer feed-forward transition block."""

    def __init__(self, dim: int, expand: int = 2) -> None:
        """Initialize the expand/contract MLP.

        Parameters
        ----------
        dim:
            Feature width.
        expand:
            Hidden expansion factor.
        """

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expand), nn.ReLU(), nn.Linear(dim * expand, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the transition block.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        Tensor
            Transformed features, same shape as input.
        """

        return self.net(self.norm(x))


class _OuterProductMean(nn.Module):
    """MSA-to-pair communication via outer product mean over sequences."""

    def __init__(self, c_m: int, c_z: int, c_hidden: int = 8) -> None:
        """Initialize the outer-product projection.

        Parameters
        ----------
        c_m:
            MSA representation width.
        c_z:
            Pair representation width.
        c_hidden:
            Bottleneck width before the outer product.
        """

        super().__init__()
        self.norm = nn.LayerNorm(c_m)
        self.proj_a = nn.Linear(c_m, c_hidden)
        self.proj_b = nn.Linear(c_m, c_hidden)
        self.out = nn.Linear(c_hidden * c_hidden, c_z)

    def forward(self, m: Tensor) -> Tensor:
        """Compute a pair-representation update from the MSA representation.

        Parameters
        ----------
        m:
            MSA representation ``(S, R, c_m)``.

        Returns
        -------
        Tensor
            Pair representation update ``(R, R, c_z)``.
        """

        mn = self.norm(m)
        a = self.proj_a(mn)
        b = self.proj_b(mn)
        outer = torch.einsum("sic,sjd->ijcd", a, b) / m.shape[0]
        return self.out(outer.reshape(outer.shape[0], outer.shape[1], -1))


class _FlowTrunkBlock(nn.Module):
    """One Evoformer-style block co-evolving MSA and pair representations."""

    def __init__(self, c_m: int, c_z: int) -> None:
        """Initialize row attention, MSA transition, outer-product, and pair transition.

        Parameters
        ----------
        c_m:
            MSA representation width.
        c_z:
            Pair representation width.
        """

        super().__init__()
        self.row_attn = _MSARowAttention(c_m, c_z)
        self.msa_transition = _Transition(c_m)
        self.outer = _OuterProductMean(c_m, c_z)
        self.pair_transition = _Transition(c_z)

    def forward(self, m: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Update MSA and pair representations for one block.

        Parameters
        ----------
        m:
            MSA representation ``(S, R, c_m)``.
        z:
            Pair representation ``(R, R, c_z)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated MSA and pair representations.
        """

        m = m + self.row_attn(m, z)
        m = m + self.msa_transition(m)
        z = z + self.outer(m)
        z = z + self.pair_transition(z)
        return m, z


class AlphaFlow(nn.Module):
    """AlphaFold trunk repurposed as a flow-matching vector field over structures.

    A flow-time ``t`` embedding is folded into the single representation
    alongside the noised/interpolated CA coordinates at that flow-time, so the
    trunk regresses the flow-matching vector field (denoised structure) rather
    than a single static prediction -- the model input is the sequence, the
    current noisy coordinates, and the scalar flow time.
    """

    def __init__(
        self, c_m: int = 16, c_z: int = 16, c_s: int = 16, n_block: int = 2, n_token: int = 22
    ) -> None:
        """Initialize embeddings, Evoformer-style trunk, time conditioning, and IPA head.

        Parameters
        ----------
        c_m:
            MSA representation width.
        c_z:
            Pair representation width.
        c_s:
            Single representation width.
        n_block:
            Number of trunk blocks.
        n_token:
            Amino-acid vocabulary size.
        """

        super().__init__()
        self.msa_embed = nn.Embedding(n_token, c_m)
        self.left_embed = nn.Embedding(n_token, c_z)
        self.right_embed = nn.Embedding(n_token, c_z)
        self.blocks = nn.ModuleList([_FlowTrunkBlock(c_m, c_z) for _ in range(n_block)])
        self.s_proj = nn.Linear(c_m, c_s)
        self.coord_proj = nn.Linear(3, c_s)
        self.time_embed = nn.Sequential(nn.Linear(1, c_s), nn.SiLU(), nn.Linear(c_s, c_s))
        self.structure = _AntibodyStructureModule(c_s, c_z, n_iter=2)

    def forward(self, aatype_msa: Tensor, noisy_ca: Tensor, flow_time: Tensor) -> Tensor:
        """Predict the flow-matching vector field (denoised CA coordinates).

        Parameters
        ----------
        aatype_msa:
            Integer MSA tokens ``(S, R)``; row 0 is the target sequence.
        noisy_ca:
            Noised/interpolated CA coordinates at flow-time ``t``, ``(R, 3)``.
        flow_time:
            Scalar flow-matching time in ``[0, 1]``, shape ``(1,)``.

        Returns
        -------
        Tensor
            Predicted (denoised) CA coordinates ``(R, 3)``.
        """

        m = self.msa_embed(aatype_msa)
        seq = aatype_msa[0]
        z = self.left_embed(seq)[:, None, :] + self.right_embed(seq)[None, :, :]
        for blk in self.blocks:
            m, z = blk(m, z)
        s = self.s_proj(m[0]) + self.coord_proj(noisy_ca)
        t_embed = self.time_embed(flow_time.view(1, 1)).expand(s.shape[0], -1)
        s = s + t_embed
        return self.structure(s, z)


def build_alphaflow() -> nn.Module:
    """Build a compact random-init AlphaFlow vector-field model.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return AlphaFlow().eval()


def example_input_alphaflow() -> tuple[Tensor, Tensor, Tensor]:
    """Return a small MSA, noisy CA coordinates, and a flow-time scalar.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        MSA tokens ``(4, 10)``, noisy CA coordinates ``(10, 3)``, flow time ``(1,)``.
    """

    aatype_msa = torch.randint(0, 22, (4, 10))
    noisy_ca = torch.randn(10, 3)
    flow_time = torch.rand(1)
    return aatype_msa, noisy_ca, flow_time


# ---------------------------------------------------------------------------
# ANI-2x -- Behler-Parrinello atomic neural-network potential (AEV features)
# ---------------------------------------------------------------------------


def _radial_aev(dist: Tensor, centers: Tensor, eta: float, cutoff: float) -> Tensor:
    """Compute Behler-Parrinello radial (2-body) symmetry-function features.

    Parameters
    ----------
    dist:
        Pairwise distances ``(N, N)``.
    centers:
        Radial shift centers ``(n_centers,)``.
    eta:
        Radial Gaussian width parameter.
    cutoff:
        Smooth cutoff radius.

    Returns
    -------
    Tensor
        Radial features summed over neighbors, ``(N, n_centers)``.
    """

    fc = 0.5 * (torch.cos(math.pi * dist.clamp(max=cutoff) / cutoff) + 1.0)
    fc = fc * (dist < cutoff).float()
    diff = dist.unsqueeze(-1) - centers.view(1, 1, -1)
    gauss = torch.exp(-eta * diff.pow(2))
    terms = gauss * fc.unsqueeze(-1)
    eye_mask = 1.0 - torch.eye(dist.shape[0], device=dist.device).unsqueeze(-1)
    return (terms * eye_mask).sum(dim=1)


def _angular_aev(coords: Tensor, dist: Tensor, centers: Tensor, cutoff: float) -> Tensor:
    """Compute Behler-Parrinello angular (3-body) symmetry-function features.

    Parameters
    ----------
    coords:
        Atom coordinates ``(N, 3)``.
    dist:
        Pairwise distances ``(N, N)``.
    centers:
        Angular shift centers ``(n_centers,)`` in radians.
    cutoff:
        Smooth cutoff radius.

    Returns
    -------
    Tensor
        Angular features summed over neighbor pairs, ``(N, n_centers)``.
    """

    n_atom = coords.shape[0]
    vec = coords.unsqueeze(1) - coords.unsqueeze(0)
    unit = F.normalize(vec, dim=-1, eps=1e-8)
    cos_theta = torch.einsum("ijc,ikc->ijk", unit, unit).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    fc = (
        0.5 * (torch.cos(math.pi * dist.clamp(max=cutoff) / cutoff) + 1.0) * (dist < cutoff).float()
    )
    fc_pair = fc.unsqueeze(2) * fc.unsqueeze(1)
    not_self = 1.0 - torch.eye(n_atom, device=coords.device)
    triple_mask = (not_self.unsqueeze(2) * not_self.unsqueeze(1)).unsqueeze(-1)
    diff = theta.unsqueeze(-1) - centers.view(1, 1, 1, -1)
    ang = (1.0 + torch.cos(diff)) * fc_pair.unsqueeze(-1)
    ang = ang * triple_mask
    return ang.sum(dim=(1, 2))


class ANI2x(nn.Module):
    """Compact ANI-2x-style Behler-Parrinello atomic NN potential.

    Builds a rotation/translation-invariant Atomic Environment Vector (AEV)
    per atom from radial and angular symmetry functions, then routes each
    atom's AEV through a per-element feed-forward atomic network; per-atom
    scalar energies sum to the molecular energy.
    """

    def __init__(
        self, n_elements: int = 4, n_radial: int = 8, n_angular: int = 4, hidden: int = 32
    ) -> None:
        """Initialize symmetry-function centers and per-element atomic networks.

        Parameters
        ----------
        n_elements:
            Number of supported chemical elements (e.g. H, C, N, O).
        n_radial:
            Number of radial symmetry-function centers.
        n_angular:
            Number of angular symmetry-function centers.
        hidden:
            Atomic-network hidden width.
        """

        super().__init__()
        self.cutoff = 5.2
        self.radial_centers = nn.Parameter(
            torch.linspace(0.9, self.cutoff, n_radial), requires_grad=False
        )
        self.angular_centers = nn.Parameter(
            torch.linspace(0.0, math.pi, n_angular + 1)[:-1], requires_grad=False
        )
        self.eta = 16.0
        aev_dim = n_radial + n_angular
        self.atomic_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(aev_dim, hidden),
                    nn.CELU(alpha=0.1),
                    nn.Linear(hidden, hidden // 2),
                    nn.CELU(alpha=0.1),
                    nn.Linear(hidden // 2, 1),
                )
                for _ in range(n_elements)
            ]
        )

    def forward(self, coords: Tensor, species: Tensor) -> Tensor:
        """Predict total molecular energy from atom coordinates and element ids.

        Parameters
        ----------
        coords:
            Atom coordinates ``(N, 3)``.
        species:
            Element ids ``(N,)``, each in ``[0, n_elements)``.

        Returns
        -------
        Tensor
            Scalar predicted molecular energy.
        """

        dist = torch.cdist(coords, coords)
        radial = _radial_aev(dist, self.radial_centers, self.eta, self.cutoff)
        angular = _angular_aev(coords, dist, self.angular_centers, self.cutoff)
        aev = torch.cat([radial, angular], dim=-1)
        energies = torch.zeros(coords.shape[0], device=coords.device, dtype=coords.dtype)
        for elem_id, net in enumerate(self.atomic_nets):
            mask = species == elem_id
            if mask.any():
                energies = torch.where(mask, net(aev).squeeze(-1), energies)
        return energies.sum()


def build_ani2x() -> nn.Module:
    """Build a compact random-init ANI-2x atomic potential.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return ANI2x().eval()


def example_input_ani2x() -> tuple[Tensor, Tensor]:
    """Return a small organic-molecule-like coordinate/species pair.

    Returns
    -------
    tuple[Tensor, Tensor]
        Atom coordinates ``(9, 3)`` and element ids ``(9,)`` in ``{H, C, N, O}``.
    """

    coords = torch.randn(9, 3) * 1.5
    species = torch.tensor([1, 0, 0, 0, 1, 0, 0, 2, 3])
    return coords, species


# ---------------------------------------------------------------------------
# AntiFold -- GVP graph encoder (ESM-IF1 fine-tune) for antibody inverse folding
# ---------------------------------------------------------------------------


class _GVP(nn.Module):
    """Geometric Vector Perceptron: joint scalar+vector feature update.

    Vector features are transformed by norm-preserving linear maps (no bias,
    so rotation-equivariance holds) and gated by scalar-derived nonlinearities,
    following the GVP-GNN construction used in ESM-IF1 / AntiFold's encoder.
    """

    def __init__(
        self, s_dim: int, v_dim: int, s_out: int, v_out: int, h_dim: int | None = None
    ) -> None:
        """Initialize scalar/vector projection layers.

        Parameters
        ----------
        s_dim:
            Input scalar feature width.
        v_dim:
            Input vector feature count.
        s_out:
            Output scalar feature width.
        v_out:
            Output vector feature count.
        h_dim:
            Hidden vector-channel width; defaults to ``max(v_dim, v_out)``.
        """

        super().__init__()
        h_dim = h_dim or max(v_dim, v_out)
        self.wh = nn.Linear(v_dim, h_dim, bias=False)
        self.ws = nn.Linear(s_dim + h_dim, s_out)
        self.wv = nn.Linear(h_dim, v_out, bias=False)
        self.s_out = s_out
        self.v_out = v_out

    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Jointly update scalar and vector node features.

        Parameters
        ----------
        s:
            Scalar features ``(N, s_dim)``.
        v:
            Vector features ``(N, v_dim, 3)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated scalar ``(N, s_out)`` and vector ``(N, v_out, 3)`` features.
        """

        vh = self.wh(v.transpose(-1, -2)).transpose(-1, -2)
        vh_norm = torch.sqrt(vh.pow(2).sum(dim=-1) + 1e-8)
        s_cat = torch.cat([s, vh_norm], dim=-1)
        s_new = F.relu(self.ws(s_cat))
        v_new = self.wv(vh.transpose(-1, -2)).transpose(-1, -2)
        gate = torch.sigmoid(torch.sqrt(v_new.pow(2).sum(dim=-1, keepdim=True) + 1e-8))
        return s_new, v_new * gate


class AntiFold(nn.Module):
    """Compact GVP-graph inverse-folding model (ESM-IF1-style, antibody fine-tune)."""

    def __init__(
        self, s_dim: int = 32, v_dim: int = 8, k: int = 6, layers: int = 3, vocab: int = 21
    ) -> None:
        """Initialize backbone-geometry featurizer, GVP encoder stack, and decoder head.

        Parameters
        ----------
        s_dim:
            Scalar node-feature width.
        v_dim:
            Vector node-feature channel count.
        k:
            Residue graph degree.
        layers:
            Number of GVP encoder layers.
        vocab:
            Amino-acid vocabulary size.
        """

        super().__init__()
        self.k = k
        self.v_dim = v_dim
        self.s_in = nn.Linear(3, s_dim)
        self.v_in = nn.Linear(3, v_dim, bias=False)
        self.chain_embed = nn.Embedding(2, s_dim)
        self.gvps = nn.ModuleList([_GVP(s_dim, v_dim, s_dim, v_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(s_dim) for _ in range(layers)])
        self.decoder = nn.Sequential(
            nn.Linear(s_dim + v_dim, s_dim), nn.GELU(), nn.Linear(s_dim, vocab)
        )

    def forward(self, backbone: Tensor, chain_id: Tensor) -> Tensor:
        """Predict per-residue amino-acid recovery logits from backbone geometry.

        Parameters
        ----------
        backbone:
            Backbone atom coordinates ``(L, 3, 3)`` for N, CA, C.
        chain_id:
            Heavy(0)/light(1) chain id per residue ``(L,)``.

        Returns
        -------
        Tensor
            Amino-acid logits ``(L, vocab)``.
        """

        n_coord, ca, c_coord = backbone.unbind(dim=1)
        dist = torch.cdist(ca, ca)
        forward_vec = c_coord - ca
        backward_vec = n_coord - ca
        dihedral_proxy = torch.cross(forward_vec, backward_vec, dim=-1)

        s = self.s_in(
            torch.stack(
                [forward_vec.norm(dim=-1), backward_vec.norm(dim=-1), dihedral_proxy.norm(dim=-1)],
                dim=-1,
            )
        )
        s = s + self.chain_embed(chain_id)
        # raw geometric vectors (N->CA, CA->C, pseudo-dihedral normal) -> v_dim vector channels
        # via a bias-free linear map, preserving rotation-equivariance (GVP construction).
        raw_vectors = torch.stack([forward_vec, backward_vec, dihedral_proxy], dim=1)
        v = torch.einsum("lkc,vk->lvc", raw_vectors, self.v_in.weight)

        dist_no_diag = dist + torch.eye(ca.shape[0], device=ca.device) * 1e6
        _, idx = dist_no_diag.topk(min(self.k, ca.shape[0] - 1), largest=False)

        for gvp, norm in zip(self.gvps, self.norms):
            neigh_s = s[idx].mean(dim=1)
            neigh_v = v[idx].mean(dim=1)
            ds, dv = gvp(s + neigh_s, v + neigh_v)
            s = norm(s + ds)
            v = v + dv
        v_norm = torch.sqrt(v.pow(2).sum(dim=-1) + 1e-8)
        return self.decoder(torch.cat([s, v_norm], dim=-1))


def build_antifold() -> nn.Module:
    """Build a compact random-init AntiFold GVP inverse-folding model.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return AntiFold().eval()


def example_input_antifold() -> tuple[Tensor, Tensor]:
    """Return a small paired-chain antibody Fv backbone and chain ids.

    Returns
    -------
    tuple[Tensor, Tensor]
        Backbone coordinates ``(L, 3, 3)`` and chain ids ``(L,)``.
    """

    heavy_len, light_len = 8, 7
    length = heavy_len + light_len
    ca = torch.randn(length, 3).cumsum(dim=0)
    n_coord = ca + torch.tensor([0.2, 0.0, 0.0])
    c_coord = ca + torch.tensor([-0.2, 0.1, 0.0])
    backbone = torch.stack([n_coord, ca, c_coord], dim=1)
    chain_id = torch.cat(
        [torch.zeros(heavy_len, dtype=torch.long), torch.ones(light_len, dtype=torch.long)]
    )
    return backbone, chain_id


MENAGERIE_ENTRIES = [
    ("AbMPNN", "build_abmpnn", "example_input_abmpnn", "2023", "BIO"),
    ("AbNatiV", "build_abnativ", "example_input_abnativ", "2023", "BIO"),
    ("ABodyBuilder2", "build_abodybuilder2", "example_input_abodybuilder2", "2024", "BIO"),
    ("AlphaFlow", "build_alphaflow", "example_input_alphaflow", "2024", "BIO"),
    ("ANI-2x", "build_ani2x", "example_input_ani2x", "2020", "BIO"),
    ("AntiFold", "build_antifold", "example_input_antifold", "2024", "BIO"),
]
