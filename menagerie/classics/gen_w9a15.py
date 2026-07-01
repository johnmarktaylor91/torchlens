"""Wave 9 batch 15 menagerie classics: vulnerability detection, and
climate/environment forecasting family (BLSTM vuln classifier, spherical
Fourier neural operator climate emulator, spatiotemporal patch transformer
for ENSO/ocean forecasting, dartboard-attention air-quality transformer,
physics-guided graph-ODE air-quality network, and a deep feedforward ANN
for glacier surface-mass-balance regression).

Sources checked (repo_url / desc_source columns of the build queue, web
research 2026-07-01; no cloning, no pip installs beyond the base env):
  - VulDeePecker: https://github.com/CGCL-codes/VulDeePecker; Li et al.,
    NDSS 2018, "VulDeePecker: A Deep Learning-Based System for Vulnerability
    Detection". The official repo ships only pre-extracted "code gadget"
    text corpora (``CWE-119/CGD/cwe119_cgd.txt``) and no PyTorch/Keras model
    file -- the architecture is fully specified in the paper instead: code
    gadgets are tokenized, mapped through a symbolic-token vocabulary,
    embedded via a token embedding table (paper uses pretrained word2vec;
    reimplemented here as a learned ``nn.Embedding``), fed through a
    **bidirectional LSTM** over the gadget's token sequence, and classified
    by dense softmax layers on the final concatenated forward/backward
    hidden state. Reimplemented here as ``VulDeePecker`` (token embedding ->
    stacked BLSTM -> dense classifier head), the paper's exact pipeline.
  - 3D-Geoformer: https://github.com/zhoulu327/Code_of_3D-Geoformer; Zhou &
    Zhang, 2023, "A self-attention-based neural network for
    three-dimensional multivariate modeling and its skillful ENSO
    prediction". Confirmed line-by-line from ``Code/Geoformer.py`` and
    ``Code/my_tools.py``: input space-time cubes are split into non-
    overlapping spatial patches (``unfold_func``), linearly embedded with
    added sinusoidal *temporal* position embeddings and learned *spatial*
    position embeddings (``make_embedding``), then processed by an
    encoder-decoder Transformer whose distinctive mechanism is **factorized
    space/time attention**: every attention block first runs standard
    multi-head attention along the *time* axis (``T_attention``, per grid
    patch) and then along the *space* axis (``S_attention``, per time step)
    rather than a single joint spatiotemporal attention -- the literal "3D"
    in the name. Reimplemented compactly as ``Geoformer3D`` (patch embed ->
    stacked factorized-attention encoder + causal factorized-attention
    decoder with cross-attention -> patch un-embed via ``fold``).
  - ACE (AI2 Climate Emulator): https://github.com/ai2cm/ace; Watt-Meyer et
    al., 2023-2025, "ACE: A fast, skillful learned global atmospheric
    model for climate prediction". Confirmed from
    ``fme/core/models/conditional_sfno/sfnonet.py``: the backbone is a
    **Spherical Fourier Neural Operator (SFNO)** -- an encoder projects
    gridded atmospheric channels to an embedding, then a stack of Fourier
    Neural Operator blocks each apply a *global spectral convolution*
    (forward spherical-harmonic transform -> per-mode complex/linear
    channel mixing -> inverse transform, implemented in
    ``s2convolutions.SpectralConvS2`` via the ``torch_harmonics`` library)
    followed by a pointwise MLP, with a "big skip" residual from the raw
    input to the decoder. ``torch_harmonics`` (SHT) is not in the base env,
    so the spectral-convolution mechanism is faithfully reproduced with the
    standard planar substitute used by the FNO/AFNO lineage that SFNO
    generalizes to the sphere: a 2D real FFT over the lat/lon grid, complex
    per-frequency-mode channel mixing restricted to the low modes (global
    receptive field in one layer, exactly the SFNO/FNO claim), and an
    inverse real FFT -- reimplemented as ``SpectralConvBlock`` inside
    ``ACEEmulator`` (encoder -> N spectral-FNO blocks with big-skip ->
    decoder), preserving the "global spectral mixing + local MLP" design.
  - AirFormer: https://github.com/yoshall/AirFormer; Liang et al., AAAI
    2023, "AirFormer: Predicting Nationwide Air Quality in China with
    Transformers". Confirmed line-by-line from ``src/models/airformer.py``:
    stacked AirFormer blocks alternate **DS-MSA** (Dartboard-partitioned
    Spatial Multi-head Self-Attention -- each station attends only to
    neighboring stations grouped into "dartboard" sectors via an
    ``einsum('bnc,mnr->bmrc', x, assignment)`` sector-pooling gather, with a
    learned per-sector relative attention bias) and **CT-MSA** (Causal
    windowed Temporal Multi-head Self-Attention over a local causal
    window), followed by a hierarchical latent-variable **stochastic
    generative/inference model** (a top-down ladder of Gaussian latents,
    one per block, reparameterized and concatenated) whose samples are
    fused with the deterministic states before the prediction head.
    Reimplemented compactly as ``AirFormer`` with a small synthetic
    dartboard sector-assignment tensor standing in for the real
    kilometre-scale station geometry (which requires an external, >500GB,
    pre-computed ``assignment.npy``/``mask.npy`` per the build-queue note),
    the DS-MSA sector-attention mechanism, CT-MSA windowed causal
    attention, and the top-down stochastic latent ladder.
  - AirPhyNet: https://github.com/kethmih/AirPhyNet; Hettige et al., ICLR
    2024, "AirPhyNet: Harnessing Physics-Guided Neural Networks for Air
    Quality Prediction". Confirmed from ``ode_func.py`` /
    ``diffeq_solver.py`` / ``airphynet_model.py``: a GRU encoder maps the
    observed sequence to a latent Gaussian initial state; a continuous-time
    **graph ODE** then integrates that latent forward, with the ODE's
    right-hand side computed as a **gated fusion of a diffusion term**
    (Chebyshev-polynomial graph convolution over the physical adjacency,
    modeling pollutant diffusion) **and an advection term** (a second graph
    convolution over a wind-direction-derived, batch-specific directed
    adjacency built from a small ``flow_net`` on wind covariates, modeling
    pollutant transport). The official repo integrates via
    ``torchdiffeq.odeint`` (not in the base env), so the ODE is integrated
    here with a compact, torch-only fixed-step RK4 loop over the same
    diffusion+advection+gated-fusion right-hand side -- reimplemented as
    ``AirPhyNet`` (GRU encoder -> reparameterized latent -> RK4-integrated
    diffusion/advection graph ODE -> linear decoder).
  - ALPGM (deep ANN SMB model): https://github.com/JordiBolibar/ALPGM;
    Bolibar et al., The Cryosphere 2020, "Deep learning applied to glacier
    evolution modelling". The shipped ANN weights are loaded from external
    ``.h5`` files (``smb_model_training.py`` calls ``load_model(...)``) and
    the Keras ``Sequential`` definition itself is not checked into the
    repo, but ``create_spatiotemporal_matrix`` in that same file pins down
    the *exact* input feature vector used to train it: 10 static/annual
    glacio-climatic covariates (cumulative positive degree days, winter
    and summer snowfall anomalies, mean/max glacier altitude, terrain
    slope, glacier area, longitude, latitude, aspect) concatenated with 12
    monthly temperature anomalies and 12 monthly snow anomalies (34
    features total), regressed onto a single scalar annual glacier
    surface-mass-balance (SMB) value; the paper describes the regressor as
    a small **deep feedforward ANN with dropout regularization**.
    Reimplemented compactly as ``ALPGMSmbNet`` (34-d input -> stacked
    ``Linear -> ReLU -> Dropout`` feedforward trunk -> scalar SMB head),
    matching the paper's described architecture and the repo's exact
    input-feature contract.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# VulDeePecker: BLSTM vulnerability detector over tokenized code gadgets
# ---------------------------------------------------------------------------


class VulDeePecker(nn.Module):
    """BLSTM-based binary vulnerability classifier over code-gadget tokens.

    Reimplements the VulDeePecker pipeline (Li et al., NDSS 2018): a code
    gadget (a slice of program statements related by data/control
    dependency) is tokenized into symbolic tokens, embedded, and classified
    by a bidirectional LSTM whose final forward/backward hidden states feed
    a dense softmax head.

    Parameters
    ----------
    vocab_size:
        Number of distinct symbolic code tokens.
    embed_dim:
        Token embedding dimensionality.
    hidden_dim:
        BLSTM hidden size (per direction).
    n_layers:
        Number of stacked BLSTM layers.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embed_dim: int = 32,
        hidden_dim: int = 48,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Classify a batch of tokenized code gadgets as vulnerable or not.

        Parameters
        ----------
        token_ids:
            Symbolic code-gadget token ids, shape ``(B, L)``.

        Returns
        -------
        torch.Tensor
            Binary class logits, shape ``(B, 2)``.
        """

        x = self.embed(token_ids)
        _, (h_n, _) = self.blstm(x)
        # h_n: (num_layers * 2, B, hidden_dim); take last layer's fwd+bwd.
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.classifier(h)


def build_vuldeepecker() -> nn.Module:
    """Build a compact VulDeePecker BLSTM classifier.

    Returns
    -------
    nn.Module
        Random-initialized VulDeePecker in eval mode.
    """

    return VulDeePecker().eval()


def example_input_vuldeepecker() -> torch.Tensor:
    """Create a small batch of tokenized code-gadget sequences.

    Returns
    -------
    torch.Tensor
        Long tensor of token ids, shape ``(4, 30)``.
    """

    torch.manual_seed(0)
    return torch.randint(0, 512, (4, 30))


# ---------------------------------------------------------------------------
# 3D-Geoformer: factorized space/time attention transformer for ENSO/ocean
# ---------------------------------------------------------------------------


def _unfold_patches(x: torch.Tensor, patch: tuple[int, int]) -> torch.Tensor:
    """Split a ``(B, T, C, H, W)`` field into non-overlapping spatial patches.

    Parameters
    ----------
    x:
        Input field, shape ``(B, T, C, H, W)``.
    patch:
        ``(patch_h, patch_w)`` patch size; ``H`` and ``W`` must be divisible.

    Returns
    -------
    torch.Tensor
        Patch cubes, shape ``(B, S, T, cube_dim)`` where ``S`` is the number
        of spatial patches and ``cube_dim = C * patch_h * patch_w``.
    """

    ph, pw = patch
    b, t, c, h, w = x.shape
    x = x.reshape(b, t, c, h // ph, ph, w // pw, pw)
    x = x.permute(0, 3, 5, 1, 2, 4, 6)  # (B, H//ph, W//pw, T, C, ph, pw)
    x = x.reshape(b, (h // ph) * (w // pw), t, c * ph * pw)
    return x


class _SpaceTimeAttention(nn.Module):
    """Factorized time-then-space multi-head self-attention.

    Applies standard scaled-dot-product multi-head attention first along
    the temporal axis (independently per spatial patch) and then along the
    spatial axis (independently per timestep), rather than a single joint
    spatiotemporal attention -- the "3D" mechanism of 3D-Geoformer.
    """

    def __init__(self, d_size: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_size // n_heads
        self.time_qkv = nn.Linear(d_size, d_size * 3)
        self.time_out = nn.Linear(d_size, d_size)
        self.space_qkv = nn.Linear(d_size, d_size * 3)
        self.space_out = nn.Linear(d_size, d_size)

    @staticmethod
    def _attend(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal_mask: torch.Tensor | None
    ) -> torch.Tensor:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def _mha_along(
        self,
        x: torch.Tensor,
        qkv_proj: nn.Linear,
        out_proj: nn.Linear,
        axis: int,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # x: (B, S, T, D); axis=2 attends over T (per S), axis=1 over S (per T).
        b, s, t, d = x.shape
        qkv = qkv_proj(x).reshape(b, s, t, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # (3, B, heads, S, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if axis == 2:
            # attend over T: fold (B, heads, S) into batch dim.
            q = q.reshape(b * self.n_heads * s, t, self.d_k)
            k = k.reshape(b * self.n_heads * s, t, self.d_k)
            v = v.reshape(b * self.n_heads * s, t, self.d_k)
            out = self._attend(q, k, v, causal_mask)
            out = out.reshape(b, self.n_heads, s, t, self.d_k)
        else:
            # attend over S: fold (B, heads, T) into batch dim.
            q = q.permute(0, 1, 3, 2, 4).reshape(b * self.n_heads * t, s, self.d_k)
            k = k.permute(0, 1, 3, 2, 4).reshape(b * self.n_heads * t, s, self.d_k)
            v = v.permute(0, 1, 3, 2, 4).reshape(b * self.n_heads * t, s, self.d_k)
            out = self._attend(q, k, v, None)
            out = out.reshape(b, self.n_heads, t, s, self.d_k).permute(0, 1, 3, 2, 4)
        out = out.permute(0, 2, 3, 1, 4).reshape(b, s, t, d)
        return out_proj(out)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply factorized time-then-space attention.

        Parameters
        ----------
        x:
            Patch embeddings, shape ``(B, S, T, D)``.
        causal_mask:
            Optional boolean mask of shape ``(T, T)`` applied to the
            temporal attention (``True`` positions are masked out).

        Returns
        -------
        torch.Tensor
            Attended embeddings, shape ``(B, S, T, D)``.
        """

        t_out = self._mha_along(x, self.time_qkv, self.time_out, axis=2, causal_mask=causal_mask)
        s_out = self._mha_along(t_out, self.space_qkv, self.space_out, axis=1, causal_mask=None)
        return s_out


class _GeoformerEncoderLayer(nn.Module):
    """One encoder block: factorized space/time self-attention + FFN."""

    def __init__(self, d_size: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = _SpaceTimeAttention(d_size, n_heads)
        self.norm1 = nn.LayerNorm(d_size)
        self.ffn = nn.Sequential(nn.Linear(d_size, d_ff), nn.ReLU(), nn.Linear(d_ff, d_size))
        self.norm2 = nn.LayerNorm(d_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class _GeoformerDecoderLayer(nn.Module):
    """One decoder block: causal factorized self-attn + cross-attn + FFN."""

    def __init__(self, d_size: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.self_attn = _SpaceTimeAttention(d_size, n_heads)
        self.norm1 = nn.LayerNorm(d_size)
        self.cross_time = nn.MultiheadAttention(d_size, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_size)
        self.ffn = nn.Sequential(nn.Linear(d_size, d_ff), nn.ReLU(), nn.Linear(d_ff, d_size))
        self.norm3 = nn.LayerNorm(d_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        b, s, t, d = x.shape
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        x = self.norm1(x + self.self_attn(x, causal_mask=causal))
        # cross-attend each (space, time) query against the encoder memory
        # flattened over its own time axis, per spatial patch.
        q = x.reshape(b * s, t, d)
        kv = memory.reshape(b * s, memory.size(2), d)
        cross_out, _ = self.cross_time(q, kv, kv)
        x = self.norm2(x + cross_out.reshape(b, s, t, d))
        x = self.norm3(x + self.ffn(x))
        return x


class Geoformer3D(nn.Module):
    """3D-Geoformer: patch-embedded factorized-attention encoder-decoder.

    Reimplements the 3D-Geoformer of Zhou & Zhang (2023) for skillful ENSO
    / upper-ocean prediction: input space-time cubes are unfolded into
    non-overlapping spatial patches, linearly embedded with sinusoidal
    temporal + learned spatial position embeddings, and processed by an
    encoder-decoder stack whose core mechanism is factorized (time-then-
    space) multi-head self-attention rather than joint spatiotemporal
    attention.

    Parameters
    ----------
    in_channels:
        Number of input physical channels (e.g. SST, wind stress u/v).
    patch:
        Spatial patch size ``(patch_h, patch_w)``.
    d_size:
        Embedding / model dimension.
    n_heads:
        Number of attention heads.
    d_ff:
        Feedforward hidden dimension.
    n_enc_layers:
        Number of encoder blocks.
    n_dec_layers:
        Number of decoder blocks.
    """

    def __init__(
        self,
        in_channels: int = 2,
        patch: tuple[int, int] = (4, 4),
        d_size: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch = patch
        self.cube_dim = in_channels * patch[0] * patch[1]
        self.d_size = d_size

        self.in_linear = nn.Linear(self.cube_dim, d_size)
        self.out_linear = nn.Linear(self.cube_dim, d_size)
        self.emb_space = nn.Embedding(64, d_size)  # supports up to 64 patches
        self.readout = nn.Linear(d_size, self.cube_dim)

        self.encoder = nn.ModuleList(
            [_GeoformerEncoderLayer(d_size, n_heads, d_ff) for _ in range(n_enc_layers)]
        )
        self.decoder = nn.ModuleList(
            [_GeoformerDecoderLayer(d_size, n_heads, d_ff) for _ in range(n_dec_layers)]
        )

    @staticmethod
    def _temporal_pe(t: int, d: int, device: torch.device) -> torch.Tensor:
        pe = torch.zeros(t, d, device=device)
        pos = torch.arange(0, t, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2, device=device).float() * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def _embed(self, cubes: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        b, s, t, _ = cubes.shape
        x = linear(cubes)
        x = x + self._temporal_pe(t, self.d_size, cubes.device)[None, None]
        space_idx = torch.arange(s, device=cubes.device)
        x = x + self.emb_space(space_idx)[None, :, None]
        return x

    def forward(self, predictor: torch.Tensor, predictand: torch.Tensor) -> torch.Tensor:
        """Predict a future space-time cube from a lookback window.

        Parameters
        ----------
        predictor:
            Lookback field, shape ``(B, T_in, C, H, W)``.
        predictand:
            Teacher-forcing target field (shifted by one step), shape
            ``(B, T_out, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Predicted field, shape ``(B, T_out, C, H, W)``.
        """

        h, w = predictor.shape[-2:]
        pred_cubes = _unfold_patches(predictor, self.patch)
        en = self._embed(pred_cubes, self.in_linear)
        for layer in self.encoder:
            en = layer(en)

        tgt_cubes = _unfold_patches(predictand, self.patch)
        de = self._embed(tgt_cubes, self.out_linear)
        for layer in self.decoder:
            de = layer(de, en)

        out = self.readout(de)  # (B, S, T, cube_dim)
        b, s, t, _ = out.shape
        ph, pw = self.patch
        out = out.reshape(b, h // ph, w // pw, t, self.in_channels, ph, pw)
        out = out.permute(0, 3, 4, 1, 5, 2, 6).reshape(b, t, self.in_channels, h, w)
        return out


def build_geoformer3d() -> nn.Module:
    """Build a compact 3D-Geoformer.

    Returns
    -------
    nn.Module
        Random-initialized Geoformer3D in eval mode.
    """

    return Geoformer3D().eval()


def example_input_geoformer3d() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small (predictor, predictand) space-time cube pair.

    Returns
    -------
    tuple of torch.Tensor
        ``predictor`` of shape ``(2, 3, 2, 8, 8)`` and ``predictand`` of
        shape ``(2, 2, 2, 8, 8)``.
    """

    torch.manual_seed(0)
    predictor = torch.randn(2, 3, 2, 8, 8)
    predictand = torch.randn(2, 2, 2, 8, 8)
    return predictor, predictand


# ---------------------------------------------------------------------------
# ACE: spherical Fourier neural operator climate emulator (planar FFT proxy)
# ---------------------------------------------------------------------------


class SpectralConvBlock(nn.Module):
    """Global spectral-convolution Fourier Neural Operator block.

    Reimplements the FNO/SFNO block used by ACE's ``sfnonet.py``: a global
    spectral convolution mixes channels per retained low-frequency Fourier
    mode (forward real FFT -> complex per-mode linear channel mixing on the
    low modes -> inverse real FFT, restoring the global receptive field in
    a single layer), followed by a pointwise MLP with a residual
    connection -- the planar-grid analogue of ACE's spherical-harmonic
    spectral convolution (which uses ``torch_harmonics``, not available in
    the base env).

    Parameters
    ----------
    channels:
        Number of channels.
    modes:
        Number of retained Fourier modes along each spatial axis.
    mlp_ratio:
        Hidden-dimension expansion ratio for the pointwise MLP.
    """

    def __init__(self, channels: int, modes: int = 4, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.channels = channels
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, modes, modes, dtype=torch.cfloat)
        )
        self.norm = nn.GroupNorm(1, channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one global-spectral-mixing + pointwise-MLP FNO block.

        Parameters
        ----------
        x:
            Gridded latent field, shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Updated latent field, shape ``(B, C, H, W)``.
        """

        residual = x
        x = self.norm(x)
        h, w = x.shape[-2:]
        x_ft = torch.fft.rfft2(x, norm="ortho")
        m1 = min(self.modes, x_ft.size(-2))
        m2 = min(self.modes, x_ft.size(-1))
        out_ft = torch.zeros_like(x_ft)
        mixed = torch.einsum("bcij,cdij->bdij", x_ft[:, :, :m1, :m2], self.weight[:, :, :m1, :m2])
        out_ft[:, :, :m1, :m2] = mixed
        x = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        x = residual + x
        x = x + self.mlp(x)
        return x


class ACEEmulator(nn.Module):
    """AI2 Climate Emulator (ACE): encoder + spectral-FNO stack + decoder.

    Reimplements the SFNO backbone of ACE (Watt-Meyer et al., 2023-2025): a
    pointwise-convolution encoder lifts gridded atmospheric channels to an
    embedding, a stack of global spectral-convolution FNO blocks
    (``SpectralConvBlock``) each mix all spatial locations in one layer via
    low-frequency-mode channel mixing, and a pointwise-convolution decoder
    projects back to physical output channels with a "big skip" residual
    from the raw input field.

    Parameters
    ----------
    in_channels:
        Number of input atmospheric state channels.
    out_channels:
        Number of predicted (next-step) output channels.
    embed_dim:
        Latent embedding dimension.
    n_layers:
        Number of stacked spectral FNO blocks.
    modes:
        Number of retained Fourier modes per spatial axis.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        embed_dim: int = 16,
        n_layers: int = 3,
        modes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, embed_dim, 1)
        self.blocks = nn.ModuleList(
            [SpectralConvBlock(embed_dim, modes=modes) for _ in range(n_layers)]
        )
        self.decoder = nn.Conv2d(embed_dim, out_channels, 1)
        self.big_skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next atmospheric state from the current gridded state.

        Parameters
        ----------
        x:
            Current atmospheric state, shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Predicted next state, shape ``(B, C_out, H, W)``.
        """

        skip = self.big_skip(x)
        h = self.encoder(x)
        for block in self.blocks:
            h = block(h)
        return self.decoder(h) + skip


def build_ace() -> nn.Module:
    """Build a compact ACE spherical-FNO climate emulator.

    Returns
    -------
    nn.Module
        Random-initialized ACEEmulator in eval mode.
    """

    return ACEEmulator().eval()


def example_input_ace() -> torch.Tensor:
    """Create a small gridded multi-channel atmospheric state.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(2, 4, 16, 32)`` (batch, channels, lat, lon).
    """

    torch.manual_seed(0)
    return torch.randn(2, 4, 16, 32)


# ---------------------------------------------------------------------------
# AirFormer: dartboard spatial attention + causal temporal attention + VAE
# ---------------------------------------------------------------------------


class _SectorSpatialAttention(nn.Module):
    """Dartboard-partitioned spatial self-attention (DS-MSA).

    Each station attends only to a pooled representative token per
    "dartboard" sector (a coarse angular/radial partition of nearby
    stations) via a learned assignment tensor, plus a learned per-sector
    relative attention bias -- reproducing AirFormer's local, geometry-
    aware spatial attention rather than dense all-pairs attention.

    Parameters
    ----------
    dim:
        Feature dimension.
    n_sectors:
        Number of dartboard sectors per station.
    n_heads:
        Number of attention heads.
    """

    def __init__(self, dim: int, n_sectors: int, n_heads: int = 2) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.n_sectors = n_sectors
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.rel_bias = nn.Parameter(torch.randn(n_heads, 1, n_sectors) * 0.02)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        """Apply dartboard sector-pooled spatial attention.

        Parameters
        ----------
        x:
            Station features, shape ``(B, N, C)``.
        assignment:
            Sector soft-assignment tensor, shape ``(N, N, n_sectors)``,
            row-normalized so ``assignment[n]`` pools neighbor features
            into ``n_sectors`` per-sector representatives for station
            ``n``.

        Returns
        -------
        torch.Tensor
            Updated station features, shape ``(B, N, C)``.
        """

        b, n, c = x.shape
        sector_kv = torch.einsum("bnc,mnr->bmrc", x, assignment)  # (B, N, R, C)
        r = sector_kv.size(2)
        q = self.q_proj(x).reshape(b * n, 1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(sector_kv.reshape(b * n, r, c))
        kv = kv.reshape(b * n, r, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (b*n, heads, 1, r)
        attn = attn.reshape(b, n, self.n_heads, 1, r) + self.rel_bias
        attn = attn.reshape(b * n, self.n_heads, 1, r).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.out_proj(out)


class _WindowedCausalTemporalAttention(nn.Module):
    """Causal temporal self-attention over local non-overlapping windows.

    Reproduces AirFormer's CT-MSA: full multi-head self-attention is
    applied within local causal windows of the time axis rather than the
    full sequence, letting window size shrink deeper into the stack.

    Parameters
    ----------
    dim:
        Feature dimension.
    n_heads:
        Number of attention heads.
    window_size:
        Local causal attention window length.
    """

    def __init__(self, dim: int, n_heads: int, window_size: int) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local causal windowed temporal attention.

        Parameters
        ----------
        x:
            Sequence features, shape ``(B, T, C)`` with ``T`` divisible by
            the window size.

        Returns
        -------
        torch.Tensor
            Updated sequence features, shape ``(B, T, C)``.
        """

        b, t, c = x.shape
        w = self.window_size
        x_win = x.reshape(b * (t // w), w, c)
        qkv = self.qkv(x_win).reshape(-1, w, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(w, w, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, w, c)
        out = out.reshape(b, t, c)
        return self.out_proj(out)


class _LatentLayer(nn.Module):
    """Gaussian latent layer producing per-node/time mean and std."""

    def __init__(self, det_dim: int, latent_in: int, latent_out: int, hidden: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(det_dim + latent_in, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 1),
            nn.ReLU(),
        )
        self.mu_head = nn.Conv2d(hidden, latent_out, 1)
        self.sigma_head = nn.Conv2d(hidden, latent_out, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu = torch.clamp(self.mu_head(h), max=10.0)
        log_sigma = torch.clamp(self.sigma_head(h), max=10.0)
        return mu, log_sigma


class AirFormer(nn.Module):
    """AirFormer: dartboard spatial attention + causal temporal attention.

    Reimplements the AirFormer architecture of Liang et al. (AAAI 2023):
    stacked blocks alternate dartboard-partitioned spatial self-attention
    (DS-MSA) with causal windowed temporal self-attention (CT-MSA), and a
    top-down hierarchical stochastic latent-variable ladder (one Gaussian
    latent per block, reparameterized) captures irreducible uncertainty
    before the deterministic and latent states are fused into the
    prediction head.

    Parameters
    ----------
    n_stations:
        Number of monitoring stations (spatial nodes).
    n_sectors:
        Number of dartboard sectors per station.
    seq_len:
        Input sequence length (must be divisible by ``2 ** (n_blocks - 1)``).
    in_dim:
        Number of input pollutant/weather channels per station per step.
    hidden:
        Hidden channel width.
    n_blocks:
        Number of stacked AirFormer blocks.
    n_heads:
        Number of attention heads.
    horizon:
        Number of future steps to predict.
    """

    def __init__(
        self,
        n_stations: int = 6,
        n_sectors: int = 4,
        seq_len: int = 8,
        in_dim: int = 3,
        hidden: int = 16,
        n_blocks: int = 2,
        n_heads: int = 2,
        horizon: int = 2,
    ) -> None:
        super().__init__()
        self.n_stations = n_stations
        self.n_sectors = n_sectors
        self.seq_len = seq_len
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.horizon = horizon

        self.start_proj = nn.Linear(in_dim, hidden)

        # Fixed random row-normalized dartboard sector assignment, standing
        # in for the real precomputed kilometre-scale geometry (which is an
        # external >500GB asset per the build-queue notes).
        assign = torch.rand(n_stations, n_stations, n_sectors)
        assign = assign / assign.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("assignment", assign)

        self.spatial_layers = nn.ModuleList(
            [_SectorSpatialAttention(hidden, n_sectors, n_heads) for _ in range(n_blocks)]
        )
        self.temporal_layers = nn.ModuleList(
            [
                _WindowedCausalTemporalAttention(
                    hidden, n_heads, window_size=max(1, seq_len // (2 ** (n_blocks - b - 1)))
                )
                for b in range(n_blocks)
            ]
        )
        self.block_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_blocks)])

        self.generative_layers = nn.ModuleList(
            [
                _LatentLayer(hidden, hidden if b < n_blocks - 1 else 0, hidden, hidden)
                for b in range(n_blocks)
            ]
        )
        self.inference_layers = nn.ModuleList(
            [
                _LatentLayer(hidden, hidden if b < n_blocks - 1 else 0, hidden, hidden)
                for b in range(n_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * n_blocks * 2, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, horizon * in_dim),
        )

    def _latent_ladder(self, layers: nn.ModuleList, det_states: list[torch.Tensor]) -> torch.Tensor:
        # top-down: last block first, no latent input; earlier blocks
        # condition on the previously sampled latent.
        b, n, t = det_states[0].shape[0], self.n_stations, det_states[0].shape[1]
        z_prev = None
        zs = []
        for i in reversed(range(self.n_blocks)):
            det_bcnt = det_states[i].permute(0, 3, 2, 1)  # (B, C, N, T)
            if z_prev is None:
                inp = det_bcnt
            else:
                inp = torch.cat([det_bcnt, z_prev], dim=1)
            mu, log_sigma = layers[i](inp)
            sigma = torch.exp(log_sigma) + 1e-3
            eps = torch.randn_like(sigma)
            z = mu + eps * sigma
            zs.append(z.permute(0, 3, 2, 1))  # back to (B, T, N, C)
            z_prev = z
        zs = list(reversed(zs))
        return torch.cat(zs, dim=-1)  # (B, T, N, hidden * n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict future pollutant concentrations from a station time series.

        Parameters
        ----------
        x:
            Station observations, shape ``(B, T, N, in_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted future values, shape ``(B, horizon, N, in_dim)``.
        """

        b, t, n, _ = x.shape
        h = self.start_proj(x)  # (B, T, N, hidden)

        det_states = []
        for i in range(self.n_blocks):
            h_flat = h.reshape(b * t, n, self.hidden)
            h_flat = self.spatial_layers[i](h_flat, self.assignment)
            h = h_flat.reshape(b, t, n, self.hidden)

            h_t = h.permute(0, 2, 1, 3).reshape(b * n, t, self.hidden)
            h_t = self.temporal_layers[i](h_t)
            h = h_t.reshape(b, n, t, self.hidden).permute(0, 2, 1, 3)

            h = self.block_norms[i](h)
            det_states.append(h)

        z_q = self._latent_ladder(self.inference_layers, det_states)  # (B, T, N, hidden*nb)
        det_cat = torch.cat(det_states, dim=-1)  # (B, T, N, hidden*nb)

        fused = torch.cat([det_cat[:, -1], z_q[:, -1]], dim=-1)  # (B, N, hidden*nb*2)
        out = self.head(fused)  # (B, N, horizon * in_dim)
        out = out.reshape(b, n, self.horizon, -1).permute(0, 2, 1, 3)
        return out


def build_airformer() -> nn.Module:
    """Build a compact AirFormer.

    Returns
    -------
    nn.Module
        Random-initialized AirFormer in eval mode.
    """

    return AirFormer().eval()


def example_input_airformer() -> torch.Tensor:
    """Create a small batch of multi-station pollutant time series.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(2, 8, 6, 3)`` (batch, time, stations, features).
    """

    torch.manual_seed(0)
    return torch.randn(2, 8, 6, 3)


# ---------------------------------------------------------------------------
# AirPhyNet: physics-guided diffusion+advection graph ODE for air quality
# ---------------------------------------------------------------------------


class _GraphConv(nn.Module):
    """Single-hop graph convolution: ``adj @ x`` followed by a linear map."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C); adj: (N, N) or (B, N, N).
        agg = torch.matmul(adj, x) if adj.dim() == 2 else torch.bmm(adj, x)
        return self.linear(agg)


class _DiffusionAdvectionODEFunc(nn.Module):
    """Physics-guided ODE right-hand side: gated diffusion + advection.

    Reimplements AirPhyNet's ``ODEFunc`` (``filter_type="diff_adv"``): the
    latent state's time-derivative is a learned gated fusion of (1) a
    diffusion term computed by graph convolution over the fixed physical
    adjacency and (2) an advection term computed by graph convolution over
    a wind-covariate-derived directed adjacency, modeling pollutant
    diffusion and wind-driven transport respectively.

    Parameters
    ----------
    n_nodes:
        Number of graph nodes (monitoring stations).
    latent_dim:
        Latent state dimension per node.
    hidden_dim:
        Hidden width for the diffusion/advection graph convolutions.
    """

    def __init__(self, n_nodes: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        self.diff_gc1 = _GraphConv(latent_dim, hidden_dim)
        self.diff_gc2 = _GraphConv(hidden_dim, latent_dim)
        self.adv_gc1 = _GraphConv(latent_dim, hidden_dim)
        self.adv_gc2 = _GraphConv(hidden_dim, latent_dim)
        self.gate = nn.Linear(latent_dim * 2, latent_dim)

    def forward(
        self, z: torch.Tensor, diff_adj: torch.Tensor, adv_adj: torch.Tensor
    ) -> torch.Tensor:
        """Compute ``dz/dt`` for the diffusion-advection graph ODE.

        Parameters
        ----------
        z:
            Latent state, shape ``(B, N, latent_dim)``.
        diff_adj:
            Fixed physical (diffusion) adjacency, shape ``(N, N)``.
        adv_adj:
            Wind-derived (advection) directed adjacency, shape
            ``(B, N, N)``.

        Returns
        -------
        torch.Tensor
            Time-derivative of the latent state, shape ``(B, N, latent_dim)``.
        """

        grad_diff = -0.1 * torch.tanh(
            self.diff_gc2(torch.tanh(self.diff_gc1(z, diff_adj)), diff_adj)
        )
        grad_adv = -torch.tanh(self.adv_gc2(torch.tanh(self.adv_gc1(z, adv_adj)), adv_adj))
        gate = torch.sigmoid(self.gate(torch.cat([grad_diff, grad_adv], dim=-1)))
        return gate * grad_diff + (1 - gate) * grad_adv


class AirPhyNet(nn.Module):
    """AirPhyNet: GRU encoder + physics-guided diffusion-advection graph ODE.

    Reimplements AirPhyNet (Hettige et al., ICLR 2024): a GRU encodes the
    observed multi-station pollutant sequence into a Gaussian latent
    initial condition, which is then integrated forward in continuous time
    through a graph ODE whose right-hand side is a gated fusion of a
    diffusion term (graph convolution over the fixed station adjacency) and
    an advection term (graph convolution over a wind-derived directed
    adjacency), integrated here with a compact fixed-step RK4 loop (the
    official repo uses ``torchdiffeq.odeint``, not available in the base
    env) before a linear decoder maps the integrated latent trajectory back
    to pollutant concentrations.

    Parameters
    ----------
    n_nodes:
        Number of monitoring stations (graph nodes).
    in_dim:
        Number of observed input features per node per step.
    latent_dim:
        Latent ODE state dimension per node.
    hidden_dim:
        GRU and graph-convolution hidden width.
    horizon:
        Number of future steps to predict.
    n_rk4_steps:
        Number of fixed RK4 integration steps per predicted horizon step.
    """

    def __init__(
        self,
        n_nodes: int = 6,
        in_dim: int = 2,
        latent_dim: int = 8,
        hidden_dim: int = 16,
        horizon: int = 3,
        n_rk4_steps: int = 2,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.n_rk4_steps = n_rk4_steps

        self.gru = nn.GRU(in_dim * n_nodes, hidden_dim, batch_first=True)
        self.mu_head = nn.Linear(hidden_dim, n_nodes * latent_dim)
        self.sigma_head = nn.Linear(hidden_dim, n_nodes * latent_dim)

        self.wind_flow_net = nn.Linear(in_dim, 1)
        self.ode_func = _DiffusionAdvectionODEFunc(n_nodes, latent_dim, hidden_dim)
        self.decoder = nn.Linear(latent_dim, in_dim)

        # Fixed physical adjacency (diffusion support); random symmetric
        # small-world stand-in for a real station distance graph.
        adj = torch.rand(n_nodes, n_nodes)
        adj = (adj + adj.t()) / 2
        adj.fill_diagonal_(0)
        adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("diff_adj", adj)

    def _advection_adjacency(self, last_wind: torch.Tensor) -> torch.Tensor:
        # last_wind: (B, N, in_dim) -> per-node scalar flow -> pairwise
        # directed edge weight from the flow difference, row-normalized.
        flow = self.wind_flow_net(last_wind)  # (B, N, 1)
        edge_weight = flow - flow.transpose(1, 2)  # (B, N, N)
        adv = torch.softmax(edge_weight, dim=-1)
        return adv

    def _rk4_integrate(
        self, z0: torch.Tensor, diff_adj: torch.Tensor, adv_adj: torch.Tensor
    ) -> torch.Tensor:
        dt = 1.0 / self.n_rk4_steps
        outs = []
        z = z0
        for _ in range(self.horizon):
            for _ in range(self.n_rk4_steps):
                k1 = self.ode_func(z, diff_adj, adv_adj)
                k2 = self.ode_func(z + 0.5 * dt * k1, diff_adj, adv_adj)
                k3 = self.ode_func(z + 0.5 * dt * k2, diff_adj, adv_adj)
                k4 = self.ode_func(z + dt * k3, diff_adj, adv_adj)
                z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            outs.append(z)
        return torch.stack(outs, dim=1)  # (B, horizon, N, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict future pollutant concentrations via graph-ODE integration.

        Parameters
        ----------
        x:
            Observed station sequence, shape ``(B, T, N, in_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted future values, shape ``(B, horizon, N, in_dim)``.
        """

        b, t, n, c = x.shape
        gru_in = x.reshape(b, t, n * c)
        _, h_n = self.gru(gru_in)
        h = h_n[-1]  # (B, hidden_dim)

        mu = self.mu_head(h).reshape(b, n, self.latent_dim)
        log_sigma = self.sigma_head(h).reshape(b, n, self.latent_dim)
        sigma = torch.exp(log_sigma) + 1e-3
        z0 = mu + torch.randn_like(sigma) * sigma

        adv_adj = self._advection_adjacency(x[:, -1])
        diff_adj = cast(torch.Tensor, self.diff_adj)
        z_traj = self._rk4_integrate(z0, diff_adj, adv_adj)
        return self.decoder(z_traj)


def build_airphynet() -> nn.Module:
    """Build a compact AirPhyNet physics-guided graph ODE model.

    Returns
    -------
    nn.Module
        Random-initialized AirPhyNet in eval mode.
    """

    return AirPhyNet().eval()


def example_input_airphynet() -> torch.Tensor:
    """Create a small batch of multi-station pollutant/wind time series.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(2, 6, 6, 2)`` (batch, time, stations, features).
    """

    torch.manual_seed(0)
    return torch.randn(2, 6, 6, 2)


# ---------------------------------------------------------------------------
# ALPGM: deep feedforward ANN for glacier surface-mass-balance regression
# ---------------------------------------------------------------------------


class ALPGMSmbNet(nn.Module):
    """Deep feedforward ANN for glacier surface-mass-balance (SMB) regression.

    Reimplements the ANN regressor from Bolibar et al. (The Cryosphere,
    2020), "Deep learning applied to glacier evolution modelling": a
    dropout-regularized feedforward network maps a 34-dimensional
    glacio-climatic feature vector -- 10 static/annual covariates
    (cumulative positive degree days, winter/summer snowfall anomalies,
    mean/max glacier altitude, terrain slope, glacier area, longitude,
    latitude, aspect) plus 12 monthly temperature anomalies and 12 monthly
    snow anomalies, exactly the feature layout built by
    ``create_spatiotemporal_matrix`` in the official repo's
    ``smb_model_training.py`` -- onto a single scalar annual glacier
    surface-mass-balance value.

    Parameters
    ----------
    in_features:
        Number of input glacio-climatic features (34 in the original
        paper's feature contract).
    hidden_dims:
        Sizes of the hidden feedforward layers.
    dropout:
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        in_features: int = 34,
        hidden_dims: tuple[int, ...] = (32, 16, 8),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Regress the annual glacier surface mass balance.

        Parameters
        ----------
        x:
            Glacio-climatic feature vectors, shape ``(B, in_features)``.

        Returns
        -------
        torch.Tensor
            Predicted scalar SMB values, shape ``(B, 1)``.
        """

        return self.net(x)


def build_alpgm() -> nn.Module:
    """Build a compact ALPGM glacier SMB feedforward ANN.

    Returns
    -------
    nn.Module
        Random-initialized ALPGMSmbNet in eval mode.
    """

    return ALPGMSmbNet().eval()


def example_input_alpgm() -> torch.Tensor:
    """Create a small batch of 34-dimensional glacio-climatic feature vectors.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(8, 34)``.
    """

    torch.manual_seed(0)
    return torch.randn(8, 34)


MENAGERIE_ENTRIES = [
    ("VulDeePecker", "build_vuldeepecker", "example_input_vuldeepecker", "2018", "SEC"),
    ("3D-Geoformer", "build_geoformer3d", "example_input_geoformer3d", "2023", "SCI"),
    ("ACE (AI2 Climate Emulator)", "build_ace", "example_input_ace", "2023", "SCI"),
    ("AirFormer", "build_airformer", "example_input_airformer", "2023", "SCI"),
    ("AirPhyNet", "build_airphynet", "example_input_airphynet", "2024", "SCI"),
    ("ALPGM (deep ANN SMB)", "build_alpgm", "example_input_alpgm", "2020", "SCI"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")
