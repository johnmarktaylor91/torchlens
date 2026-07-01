"""Five faithful, compact reimplementations of structural-biology / omics models.

Sources checked (repo code and/or paper, no clone/pip-install; base-env torch reimplementation
of the distinctive mechanism):
  - InstaNovo: https://github.com/instadeepai/InstaNovo (instanovo/transformer/model.py).
               Eloff et al., Nature Machine Intelligence 2025. https://www.nature.com/articles/s42256-025-01019-5
               De novo peptide sequencing: a spectrum encoder (multi-scale sinusoidal peak
               embeddings of m/z + intensity fed through transformer-encoder layers) and an
               autoregressive transformer decoder that cross-attends over the encoded spectrum
               peaks to predict amino-acid residues one at a time (causal self-attention +
               spectrum cross-attention), stopping at an end-of-sequence token.
  - Invariant Point Attention (IPA): https://github.com/lucidrains/invariant-point-attention
               (standalone library extracted from AlphaFold2's structure module).
               Jumper et al., Nature 2021. https://www.nature.com/articles/s41586-021-03819-2
               SE(3)-invariant attention over per-residue rigid frames (rotation + translation):
               attention logits combine (a) scalar q.k dot-product, (b) a pair-representation
               bias, and (c) a squared-distance term between "query points" and "key points"
               that are generated in each residue's local frame, mapped into the global frame,
               and compared there -- so the score is invariant to a global rotation/translation
               of all frames. This module reimplements that standalone primitive (not the full
               Evoformer trunk, which already exists faithfully in ``openfold_af2.py``).
  - DiffSBDD:  https://github.com/arneschneuing/DiffSBDD (equivariant_diffusion/dynamics.py,
               egnn/egnn_new.py). Schneuing et al., Nature Computational Science 2024
               (arXiv:2210.13695). SE(3)-equivariant DDPM over a joint ligand+pocket point
               cloud: each EGNN layer updates per-atom scalar features with permutation-
               invariant message passing, and updates 3D coordinates only via an equivariant
               weighted sum of *relative* coordinate vectors (never absolute coordinates enter
               a learned nonlinearity), which is what makes the whole coordinate-denoising
               stack rotation/translation-equivariant. Diffusion runs on ligand atoms only,
               conditioned on a fixed pocket context concatenated into the same graph each step.
  - Latent-space cryo-EM diffusion: base VAE at ``cryodrgn.py``; this entry is the diffusion
               extension. arXiv:2211.14169 (Kreis et al., "Latent Space Diffusion Models of
               Cryo-EM Structures"). A cryoDRGN-style encoder maps particle images to a latent
               conformational-heterogeneity code; instead of a plain VAE prior/decoder, a
               DDPM-style denoiser (a small time-conditioned MLP score network operating
               directly in that latent space, with sinusoidal timestep embedding FiLM-style
               modulation) is trained to reverse a noising process over the latent codes, then
               the cryoDRGN volume decoder (coordinate-MLP conditioned on the -- now diffusion-
               sampled -- latent code) renders a 3D density at query spatial coordinates.
  - MaxFuse:   https://github.com/shuxiaoc/maxfuse (maxfuse/model.py: Fusor / matching.py).
               Chen et al., Nature Biotechnology 2024. https://www.nature.com/articles/s41587-023-01935-0
               Cross-modal cell matching via iterative co-embedding + graph "fuzzy smoothing" +
               nearest-neighbor matching, for weakly-linked modalities (e.g. spatial proteomics
               vs. scRNA-seq) where few features are shared across modalities. The distinctive
               mechanism reimplemented here: two modality-specific linear projection heads learn
               a shared co-embedding space from the small set of *linked* (shared) features;
               each modality's *full* per-cell feature vector is then smoothed by propagating it
               over a k-nearest-neighbor graph built in that shared co-embedding space (a graph-
               diffusion / message-passing smoothing step, "fuzzy smoothing"), and the smoothed
               full-feature vectors of both modalities are re-projected into a second joint
               embedding used for cross-modal nearest-neighbor cell matching -- the co-embed ->
               smooth -> match cycle can be iterated.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# InstaNovo -- transformer de novo peptide sequencing (spectrum encoder + AR decoder)
# ---------------------------------------------------------------------------


class _PeakEmbedding(nn.Module):
    """Multi-scale sinusoidal embedding of (m/z, intensity) peak pairs."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1e4), half))
        self.register_buffer("freqs", freqs, persistent=False)
        self.intensity_proj = nn.Linear(1, d_model)

    def forward(self, mz: Tensor, intensity: Tensor) -> Tensor:
        # mz, intensity: (batch, n_peaks)
        angles = mz.unsqueeze(-1) * self.freqs  # (batch, n_peaks, d_model // 2)
        mz_embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return mz_embed + self.intensity_proj(intensity.unsqueeze(-1))


class InstaNovo(nn.Module):
    """Transformer encoder over spectrum peaks + autoregressive residue decoder."""

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        vocab_size: int = 22,
    ) -> None:
        super().__init__()
        self.peak_embed = _PeakEmbedding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers)
        self.residue_embed = nn.Embedding(vocab_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, mz: Tensor, intensity: Tensor, residue_prefix: Tensor) -> Tensor:
        """Predict next-residue logits given a spectrum and a partial peptide.

        Parameters
        ----------
        mz, intensity:
            Shape ``(batch, n_peaks)`` peak m/z values and intensities.
        residue_prefix:
            Shape ``(batch, seq_len)`` integer residue indices decoded so far.

        Returns
        -------
        Tensor
            Shape ``(batch, seq_len, vocab_size)`` next-residue logits.
        """
        memory = self.encoder(self.peak_embed(mz, intensity))
        tgt = self.residue_embed(residue_prefix)
        seq_len = tgt.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=tgt.device), diagonal=1
        )
        decoded = self.decoder(tgt, memory, tgt_mask=causal_mask)
        return self.out_proj(decoded)


def build_instanovo() -> nn.Module:
    """Build a small InstaNovo (spectrum transformer encoder + AR residue decoder)."""
    return InstaNovo(d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2, vocab_size=22).eval()


def example_input_instanovo() -> tuple[Tensor, Tensor, Tensor]:
    """Example input for InstaNovo: a batch of MS/MS spectra + partial peptide prefixes.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(mz, intensity, residue_prefix)`` with ``mz``/``intensity`` shape ``(2, 30)``
        and ``residue_prefix`` shape ``(2, 6)``.
    """
    mz = torch.rand(2, 30) * 2000.0
    intensity = torch.rand(2, 30)
    residue_prefix = torch.randint(0, 22, (2, 6))
    return mz, intensity, residue_prefix


# ---------------------------------------------------------------------------
# Invariant Point Attention (IPA) -- standalone SE(3)-invariant frame attention
# ---------------------------------------------------------------------------


class InvariantPointAttention(nn.Module):
    """Standalone IPA primitive: scalar + pair-bias + 3D-point attention terms.

    Reproduces the lucidrains ``invariant-point-attention`` standalone module (itself
    extracted from AlphaFold2's structure module): attention over per-token rigid
    frames that is invariant to a global rotation/translation of every frame.
    """

    def __init__(
        self,
        dim: int,
        pairwise_dim: int = 16,
        heads: int = 4,
        head_dim: int = 8,
        n_query_points: int = 4,
        n_value_points: int = 4,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.n_query_points = n_query_points
        self.n_value_points = n_value_points
        self.to_q = nn.Linear(dim, heads * head_dim)
        self.to_kv = nn.Linear(dim, 2 * heads * head_dim)
        self.to_q_points = nn.Linear(dim, heads * n_query_points * 3)
        self.to_kv_points = nn.Linear(dim, heads * (n_query_points + n_value_points) * 3)
        self.to_pairwise_bias = nn.Linear(pairwise_dim, heads)
        self.point_weight = nn.Parameter(torch.zeros(heads))
        concat_dim = heads * (head_dim + pairwise_dim + n_value_points * 4)
        self.to_out = nn.Linear(concat_dim, dim)

    def forward(
        self, single: Tensor, pairwise: Tensor, rotations: Tensor, translations: Tensor
    ) -> Tensor:
        # single: (n, dim), pairwise: (n, n, pairwise_dim)
        # rotations: (n, 3, 3), translations: (n, 3)  -- per-token rigid frame
        n = single.shape[0]
        h, d = self.heads, self.head_dim
        q = self.to_q(single).view(n, h, d)
        k, v = self.to_kv(single).view(n, h, 2 * d).split(d, dim=-1)

        def local_to_global(local_pts: Tensor) -> Tensor:
            return (
                torch.einsum("nij,nhpj->nhpi", rotations, local_pts)
                + translations[:, None, None, :]
            )

        qp = self.to_q_points(single).view(n, h, self.n_query_points, 3)
        kvp = self.to_kv_points(single).view(n, h, self.n_query_points + self.n_value_points, 3)
        kp, vp = kvp.split([self.n_query_points, self.n_value_points], dim=2)
        qp_g, kp_g, vp_g = local_to_global(qp), local_to_global(kp), local_to_global(vp)

        scalar_logits = torch.einsum("ihd,jhd->hij", q, k) / (d**0.5)
        pair_bias = self.to_pairwise_bias(pairwise).permute(2, 0, 1)
        point_dist2 = (qp_g[:, None] - kp_g[None]).pow(2).sum(dim=(-1, -2)).permute(2, 0, 1)
        gamma = F.softplus(self.point_weight).view(h, 1, 1)
        logits = scalar_logits / 3**0.5 + pair_bias / 3**0.5 - 0.5 * gamma * point_dist2
        attn = torch.softmax(logits, dim=-1)  # (h, n, n)

        out_scalar = torch.einsum("hij,jhd->ihd", attn, v).reshape(n, -1)
        out_pair = torch.einsum("hij,ijc->ihc", attn, pairwise).reshape(n, -1)
        out_pt_global = torch.einsum("hij,jhpk->ihpk", attn, vp_g)
        rot_t = rotations.transpose(-1, -2)
        out_pt_local = torch.einsum(
            "nij,nhpj->nhpi", rot_t, out_pt_global - translations[:, None, None, :]
        )
        out_pt_norm = torch.sqrt(out_pt_local.pow(2).sum(-1) + 1e-8)
        out_points = torch.cat([out_pt_local.reshape(n, -1), out_pt_norm.reshape(n, -1)], dim=-1)

        return self.to_out(torch.cat([out_scalar, out_pair, out_points], dim=-1))


def build_ipa() -> nn.Module:
    """Build a standalone Invariant Point Attention module."""
    return InvariantPointAttention(
        dim=32, pairwise_dim=16, heads=4, head_dim=8, n_query_points=4, n_value_points=4
    ).eval()


def example_input_ipa() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example input for IPA: single/pairwise reprs + a set of identity rigid frames.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(single, pairwise, rotations, translations)`` for ``n=10`` tokens.
    """
    n = 10
    single = torch.randn(n, 32)
    pairwise = torch.randn(n, n, 16)
    rotations = torch.eye(3).unsqueeze(0).repeat(n, 1, 1)
    translations = torch.randn(n, 3)
    return single, pairwise, rotations, translations


# ---------------------------------------------------------------------------
# DiffSBDD -- SE(3)-equivariant diffusion for structure-based drug design
# ---------------------------------------------------------------------------


class _EquivariantGraphLayer(nn.Module):
    """One EGNN layer: invariant feature message passing + equivariant coord update."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, h: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        # h: (n, feature_dim) scalar per-atom features, x: (n, 3) coordinates.
        rel = x.unsqueeze(1) - x.unsqueeze(0)  # (n, n, 3) -- relative, so rotation-equivariant
        dist2 = rel.pow(2).sum(-1, keepdim=True)  # (n, n, 1) -- invariant scalar
        h_i = h.unsqueeze(1).expand(-1, h.shape[0], -1)
        h_j = h.unsqueeze(0).expand(h.shape[0], -1, -1)
        msg = self.edge_mlp(torch.cat([h_i, h_j, dist2], dim=-1))  # (n, n, hidden)
        eye = torch.eye(h.shape[0], device=h.device, dtype=torch.bool)
        coord_weight = (
            self.coord_mlp(msg).squeeze(-1).masked_fill(eye, 0.0)
        )  # (n, n) invariant scalar
        x_update = (rel * coord_weight.unsqueeze(-1)).mean(
            dim=1
        )  # equivariant: scalar * relative vector
        x_out = x + x_update
        msg_agg = msg.masked_fill(eye.unsqueeze(-1), 0.0).sum(dim=1)
        h_out = h + self.node_mlp(torch.cat([h, msg_agg], dim=-1))
        return h_out, x_out


class DiffSBDD(nn.Module):
    """SE(3)-equivariant DDPM denoiser over a joint ligand+pocket point cloud.

    A stack of ``_EquivariantGraphLayer`` blocks jointly denoise ligand-atom
    features/coordinates conditioned on a fixed pocket context, at a given
    diffusion timestep (sinusoidal time embedding added to every atom's
    scalar feature). Only ligand atoms receive a coordinate/feature update;
    pocket atoms act as fixed geometric context, as in the real conditional
    DDPM variant of DiffSBDD.
    """

    def __init__(self, feature_dim: int = 16, hidden_dim: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim)
        )
        self.atom_type_embed = nn.Linear(10, feature_dim)
        self.layers = nn.ModuleList(
            [_EquivariantGraphLayer(feature_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.noise_head = nn.Linear(feature_dim, 10)

    def forward(
        self,
        ligand_x: Tensor,
        ligand_type: Tensor,
        pocket_x: Tensor,
        pocket_type: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict per-ligand-atom coordinate and type noise at diffusion step ``t``.

        Parameters
        ----------
        ligand_x:
            Shape ``(n_lig, 3)`` noised ligand atom coordinates.
        ligand_type:
            Shape ``(n_lig, 10)`` noised ligand atom-type one-hot/logits.
        pocket_x:
            Shape ``(n_pocket, 3)`` fixed protein-pocket atom coordinates.
        pocket_type:
            Shape ``(n_pocket, 10)`` fixed protein-pocket atom types.
        t:
            Shape ``(1,)`` diffusion timestep (scalar, broadcast to all atoms).

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(coord_noise_pred, type_noise_pred)`` for the ligand atoms only,
            shapes ``(n_lig, 3)`` and ``(n_lig, 10)``.
        """
        n_lig = ligand_x.shape[0]
        time_feat = self.time_embed(t.view(1, 1).expand(n_lig + pocket_x.shape[0], 1))
        h_lig = self.atom_type_embed(ligand_type) + time_feat[:n_lig]
        h_pocket = self.atom_type_embed(pocket_type) + time_feat[n_lig:]
        h = torch.cat([h_lig, h_pocket], dim=0)
        x = torch.cat([ligand_x, pocket_x], dim=0)
        for layer in self.layers:
            h, x = layer(h, x)
            # pocket atoms are fixed geometric context: restore their coordinates each step
            x = torch.cat([x[:n_lig], pocket_x], dim=0)
        pred = self.noise_head(h[:n_lig])
        coord_noise_pred = x[:n_lig] - ligand_x
        return coord_noise_pred, pred


def build_diffsbdd() -> nn.Module:
    """Build a small DiffSBDD SE(3)-equivariant ligand/pocket denoiser."""
    return DiffSBDD(feature_dim=16, hidden_dim=32, n_layers=3).eval()


def example_input_diffsbdd() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example input for DiffSBDD: noised ligand atoms + fixed pocket context + timestep.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        ``(ligand_x, ligand_type, pocket_x, pocket_type, t)``: 12 ligand atoms,
        20 pocket atoms, 10 atom types, scalar diffusion timestep.
    """
    ligand_x = torch.randn(12, 3)
    ligand_type = F.one_hot(torch.randint(0, 10, (12,)), 10).float()
    pocket_x = torch.randn(20, 3)
    pocket_type = F.one_hot(torch.randint(0, 10, (20,)), 10).float()
    t = torch.tensor([250.0])
    return ligand_x, ligand_type, pocket_x, pocket_type, t


# ---------------------------------------------------------------------------
# Latent-space cryo-EM diffusion (cryoDRGN latent + DDPM latent-diffusion prior)
# ---------------------------------------------------------------------------


class _SinusoidalTimeEmbedding(nn.Module):
    """Standard DDPM sinusoidal timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class CryoEMLatentDiffusion(nn.Module):
    """cryoDRGN-style image encoder -> latent DDPM score net -> coordinate-MLP volume decoder.

    Reimplements the Kreis et al. (arXiv:2211.14169) extension of cryoDRGN: instead of a
    plain VAE Gaussian prior over the conformational latent code, a time-conditioned score
    network is trained to denoise the latent code (standard latent-diffusion setup), and the
    cryoDRGN implicit coordinate-MLP volume decoder renders a density value at each queried
    3D spatial coordinate conditioned on that (denoised) latent code.
    """

    def __init__(self, latent_dim: int = 8, image_size: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.time_embed = _SinusoidalTimeEmbedding(latent_dim)
        self.score_net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.volume_decoder = nn.Sequential(
            nn.Linear(3 + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: Tensor, coords: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a particle image, denoise its latent code, and query a density volume.

        Parameters
        ----------
        image:
            Shape ``(1, image_size, image_size)`` cryo-EM particle image.
        coords:
            Shape ``(n_points, 3)`` query spatial coordinates for the implicit volume.
        t:
            Shape ``(1,)`` diffusion timestep.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(predicted_noise, density)``: predicted latent-space noise of shape
            ``(1, latent_dim)`` and per-query density of shape ``(n_points, 1)``.
        """
        z = self.encoder(image.unsqueeze(0))  # (1, latent_dim)
        time_feat = self.time_embed(t)  # (1, latent_dim)
        predicted_noise = self.score_net(torch.cat([z, time_feat], dim=-1))
        z_denoised = z - predicted_noise
        z_broadcast = z_denoised.expand(coords.shape[0], -1)
        density = self.volume_decoder(torch.cat([coords, z_broadcast], dim=-1))
        return predicted_noise, density


def build_cryoem_latent_diffusion() -> nn.Module:
    """Build a small cryoDRGN-style latent-diffusion cryo-EM model."""
    return CryoEMLatentDiffusion(latent_dim=8, image_size=16, hidden_dim=64).eval()


def example_input_cryoem_latent_diffusion() -> tuple[Tensor, Tensor, Tensor]:
    """Example input: one particle image + queried volume coordinates + a timestep.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(image, coords, t)`` with ``image`` shape ``(16, 16)``, ``coords`` shape
        ``(64, 3)``, and ``t`` shape ``(1,)``.
    """
    image = torch.randn(16, 16)
    coords = torch.rand(64, 3) - 0.5
    t = torch.tensor([100.0])
    return image, coords, t


# ---------------------------------------------------------------------------
# MaxFuse -- iterative co-embedding + fuzzy graph smoothing + cross-modal matching
# ---------------------------------------------------------------------------


class MaxFuse(nn.Module):
    """Cross-modal cell matching via co-embed -> kNN graph-smooth -> re-embed.

    Two modality-specific linear heads project each modality's *shared/linked*
    features into a common co-embedding space. A soft k-nearest-neighbor
    similarity graph is built per modality in that shared space, and each
    modality's *full* feature vector (linked + modality-specific) is smoothed
    by one step of graph diffusion over that graph ("fuzzy smoothing"). The
    smoothed full-feature vectors of both modalities are then re-projected by
    a second pair of linear heads into a joint matching space used for
    cross-modal nearest-neighbor cell matching.
    """

    def __init__(
        self, shared_dim: int = 20, full_dim_a: int = 40, full_dim_b: int = 30, embed_dim: int = 16
    ) -> None:
        super().__init__()
        self.co_embed_a = nn.Linear(shared_dim, embed_dim)
        self.co_embed_b = nn.Linear(shared_dim, embed_dim)
        self.match_embed_a = nn.Linear(full_dim_a, embed_dim)
        self.match_embed_b = nn.Linear(full_dim_b, embed_dim)

    @staticmethod
    def _fuzzy_smooth(features: Tensor, coembed: Tensor, temperature: float = 1.0) -> Tensor:
        # Soft kNN graph smoothing: propagate `features` over a similarity graph
        # built from `coembed` (cosine-similarity softmax adjacency).
        coembed_n = F.normalize(coembed, dim=-1)
        sim = coembed_n @ coembed_n.t()
        adjacency = torch.softmax(sim / temperature, dim=-1)
        return adjacency @ features

    def forward(
        self, shared_a: Tensor, shared_b: Tensor, full_a: Tensor, full_b: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Co-embed, fuzzy-smooth, and re-embed two weakly-linked modalities.

        Parameters
        ----------
        shared_a, shared_b:
            Shape ``(n_a, shared_dim)`` / ``(n_b, shared_dim)`` linked (shared)
            features for modality A and B.
        full_a, full_b:
            Shape ``(n_a, full_dim_a)`` / ``(n_b, full_dim_b)`` full per-cell
            feature vectors for modality A and B.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(match_embed_a, match_embed_b, match_scores)``: joint-space
            embeddings for both modalities and a cross-modal similarity
            matrix of shape ``(n_a, n_b)``.
        """
        coembed_a = self.co_embed_a(shared_a)
        coembed_b = self.co_embed_b(shared_b)
        smoothed_a = self._fuzzy_smooth(full_a, coembed_a)
        smoothed_b = self._fuzzy_smooth(full_b, coembed_b)
        match_a = self.match_embed_a(smoothed_a)
        match_b = self.match_embed_b(smoothed_b)
        match_scores = F.normalize(match_a, dim=-1) @ F.normalize(match_b, dim=-1).t()
        return match_a, match_b, match_scores


def build_maxfuse() -> nn.Module:
    """Build a small MaxFuse cross-modal co-embed/smooth/match model."""
    return MaxFuse(shared_dim=20, full_dim_a=40, full_dim_b=30, embed_dim=16).eval()


def example_input_maxfuse() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example input for MaxFuse: shared + full feature matrices for two modalities.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(shared_a, shared_b, full_a, full_b)`` for 15 cells in modality A
        (e.g. spatial proteomics) and 12 cells in modality B (e.g. scRNA-seq).
    """
    shared_a = torch.randn(15, 20)
    shared_b = torch.randn(12, 20)
    full_a = torch.randn(15, 40)
    full_b = torch.randn(12, 30)
    return shared_a, shared_b, full_a, full_b


MENAGERIE_ENTRIES = [
    ("InstaNovo", "build_instanovo", "example_input_instanovo", "2025", "BIO"),
    ("Invariant Point Attention (IPA)", "build_ipa", "example_input_ipa", "2021", "BIO"),
    (
        "Latent Diffusion for Drug Design (DiffSBDD)",
        "build_diffsbdd",
        "example_input_diffsbdd",
        "2024",
        "GEN",
    ),
    (
        "Latent-space cryo-EM diffusion",
        "build_cryoem_latent_diffusion",
        "example_input_cryoem_latent_diffusion",
        "2022",
        "BIO",
    ),
    ("MaxFuse", "build_maxfuse", "example_input_maxfuse", "2024", "BIO"),
]
