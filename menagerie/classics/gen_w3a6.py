"""Menagerie batch w3a6: style-transfer / vector-graphics generative architectures.

Sources checked (reference only; no cloning, no pip installs):
  - StyleSwap ("Fast Patch-based Style Transfer of Arbitrary Style", Chen & Schmidt
    2016, arxiv:1612.04337): https://github.com/rtqichen/style-swap (styleswap.lua) --
    a fixed VGG encoder produces content/style feature maps at relu3_1; every 3x3
    content patch is matched via normalized cross-correlation (implemented as a
    convolution against the style patches) to its nearest-neighbour style patch, the
    matched style patches are pasted back with overlap-averaging ("style swap"), and a
    trained decoder maps the swapped feature map back to an RGB image.
  - StyTr2 ("Image Style Transfer with Transformers", Deng et al. 2021/CVPR 2022,
    arxiv:2105.14576): https://github.com/diyiiyiii/StyTR-2 (models/StyTR.py,
    models/transformer.py) -- two parallel transformer encoders turn a patchified
    content image (with a learned Content-Aware Positional Encoding, CAPE, added to
    the tokens) and a patchified style image into domain-specific token sequences; a
    multi-layer transformer decoder cross-attends the content queries against the
    style keys/values; a small CNN progressively upsamples the decoded tokens back to
    an RGB image.
  - SVG-VAE ("A Learned Representation for Scalable Vector Graphics", Lopes et al.,
    ICCV 2019, published under Google Magenta): https://github.com/magenta/magenta
    (magenta/models/svg_vae) -- a convolutional image encoder produces a VAE
    bottleneck (mu, logvar) over a rendered glyph image; the sampled latent is
    concatenated with a discrete class (font/glyph) embedding at every step and fed
    into a stack of LSTMs that autoregressively emit SVG drawing commands; the final
    LSTM layer is a Mixture-Density-Network head producing a Gaussian-mixture over
    continuous pen offsets plus categorical pen-state logits (draw/lift/end), matching
    the sketch-rnn-style decoder used for vector fonts.
  - SVGDreamer ("Text Guided SVG Generation with Diffusion Model", Xing et al.,
    CVPR 2024, arxiv:2312.16476): https://github.com/ximinng/SVGDreamer
    (svgdreamer/painter/) -- two distinctive stages captured here as one traceable
    module: (1) SIVE, semantic-driven image vectorization, which extracts a
    cross-attention map between text tokens and image feature-map locations so each
    of a set of learned Bezier-curve control-point primitives is assigned/pulled
    toward the region its attention mass concentrates on; (2) VPSD, vectorized
    particle-based score distillation, which keeps a *population* of vector particles
    (each an independent set of control points + colors) and scores every particle
    with a small conditional noise-prediction network (a compact proxy standing in for
    the frozen text-to-image diffusion UNet used for real score-distillation
    guidance), reweighting particles by their predicted noise residual.
  - UniColor ("A Unified Framework for Multi-Modal Colorization with Transformer",
    Huang et al., SIGGRAPH Asia 2022, arxiv:2209.11223):
    https://github.com/luckyhzt/unicolor (models/ColorTransformer.py,
    models/ChromaVQGAN.py) -- a Chroma-VQGAN encodes a downsampled chroma (ab) grid
    into a codebook of discrete chroma tokens; a Hybrid-Transformer then takes a mixed
    token sequence built from (a) continuous per-patch grayscale luminance embeddings,
    (b) sparse colour "hint points" (quantized chroma tokens placed at specific
    spatial locations, standing in for stroke/exemplar/text-derived conditions), and
    (c) mask tokens for the remaining chroma grid, and autoregressively/jointly
    predicts the full chroma-token grid, which the Chroma-VQGAN decoder turns back
    into a full-resolution colorized image.
  - Universal Style Transfer / WCT ("Universal Style Transfer via Feature
    Transforms", Li et al., NIPS 2017, arxiv:1705.08086):
    https://github.com/Yijunmaverick/UniversalStyleTransfer (Torch reference) /
    https://github.com/sunshineatnoon/PytorchWCT (PyTorch reimplementation) -- a
    single VGG encoder is truncated at five different relu levels (relu1_1 through
    relu5_1), each level paired with a trained symmetric decoder that inverts it back
    to pixel space; a genuine Whitening-and-Coloring Transform (WCT) -- centre the
    content feature map, whiten it by the inverse matrix square root of its own
    per-channel covariance (via eigendecomposition), then colour it with the matrix
    square root of the style feature map's covariance and re-add the style mean -- is
    applied at each level, and the five single-level stylizations are chained
    coarse-to-fine (level 5 down to level 1) for the full multi-level pipeline.

All models below are compact, faithfully-reimplemented-from-scratch nn.Modules with
random init and small dims for TorchLens architecture-catalog tracing (not a
trained-weights zoo).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# StyleSwap -- VGG-patch nearest-neighbour "style swap" + trained decoder
# ---------------------------------------------------------------------------


class StyleSwapEncoder(nn.Module):
    """Compact VGG-style encoder truncated at a relu3_1-like feature map."""

    def __init__(self, base: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``x`` into a relu3_1-analog feature map."""
        return self.net(x)


def _style_swap(content_feat: Tensor, style_feat: Tensor, patch_size: int = 3) -> Tensor:
    """Patch-wise nearest-neighbour style swap (Chen & Schmidt 2016).

    Extract overlapping ``patch_size``-sized patches from ``style_feat``, match each
    spatial location of ``content_feat`` to its highest-normalized-cross-correlation
    style patch via a convolution against the style patches, then reconstruct by
    pasting the matched style patches back with overlap-averaging.
    """
    b, c, h, w = style_feat.shape
    unfolded = F.unfold(style_feat, kernel_size=patch_size, padding=patch_size // 2)
    # unfolded: (b, c*k*k, n_patches) -- batch=1 catalog usage, so drop the batch dim.
    flat_patches = unfolded[0].transpose(0, 1)  # (n_patches, c*k*k)
    patches = flat_patches.view(-1, c, patch_size, patch_size)  # (n_patches, c, k, k)
    norm = flat_patches.norm(dim=1).clamp_min(1e-8).view(-1, 1, 1, 1)
    normalized_patches = patches / norm

    corr = F.conv2d(content_feat, normalized_patches, padding=patch_size // 2)
    best = corr.argmax(dim=1)  # (b, h, w) index of best-matching style patch

    matched = flat_patches[best[0].reshape(-1)]  # (h*w, c*k*k)
    matched = matched.transpose(0, 1).unsqueeze(0)  # (1, c*k*k, h*w)
    out = F.fold(
        matched,
        output_size=(h, w),
        kernel_size=patch_size,
        padding=patch_size // 2,
    )
    ones = F.fold(
        F.unfold(
            torch.ones_like(content_feat[:1]), kernel_size=patch_size, padding=patch_size // 2
        ),
        output_size=(h, w),
        kernel_size=patch_size,
        padding=patch_size // 2,
    )
    return out / ones.clamp_min(1e-8)


class StyleSwapDecoder(nn.Module):
    """Trained decoder mapping a swapped relu3_1-analog feature map back to RGB."""

    def __init__(self, base: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Decode the style-swapped feature map into an RGB image."""
        return self.net(x)


class StyleSwap(nn.Module):
    """StyleSwap: VGG encoder + patch nearest-neighbour style swap + trained decoder."""

    def __init__(self, base: int = 8, patch_size: int = 3) -> None:
        super().__init__()
        self.encoder = StyleSwapEncoder(base=base)
        self.decoder = StyleSwapDecoder(base=base)
        self.patch_size = patch_size

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Style-swap ``content`` toward ``style`` and decode back to RGB."""
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        swapped = _style_swap(content_feat, style_feat, patch_size=self.patch_size)
        return self.decoder(swapped)


def build_styleswap() -> nn.Module:
    """Build a compact StyleSwap model.

    Returns
    -------
    nn.Module
        StyleSwap in eval mode.
    """
    return StyleSwap(base=8, patch_size=3).eval()


def example_input_styleswap() -> tuple[Tensor, Tensor]:
    """Example (content, style) image pair for StyleSwap.

    Returns
    -------
    tuple[Tensor, Tensor]
        Two ``(1, 3, 32, 32)`` tensors.
    """
    return torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# StyTr2 -- dual-stream transformer style transfer with CAPE
# ---------------------------------------------------------------------------


class ContentAwarePositionalEncoding(nn.Module):
    """CAPE: a small conv net predicting a scale-invariant positional code per patch."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, feat: Tensor) -> Tensor:
        """Predict a content-conditioned positional code with the same shape as ``feat``."""
        return self.net(feat)


class StyTr2(nn.Module):
    """StyTr2: content transformer encoder (+CAPE) / style transformer encoder / decoder."""

    def __init__(self, dim: int = 16, n_heads: int = 2, n_layers: int = 1) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.cape = ContentAwarePositionalEncoding(dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.content_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        style_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.style_encoder = nn.TransformerEncoder(style_layer, num_layers=n_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 3, 3, padding=1),
        )

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Stylize ``content`` with ``style`` via cross-attending transformer streams."""
        c_feat = self.patch_embed(content)
        b, dim, h, w = c_feat.shape
        c_feat = c_feat + self.cape(c_feat)
        c_tokens = c_feat.flatten(2).transpose(1, 2)

        s_feat = self.patch_embed(style)
        s_tokens = s_feat.flatten(2).transpose(1, 2)

        content_seq = self.content_encoder(c_tokens)
        style_seq = self.style_encoder(s_tokens)
        stylized_seq = self.transformer_decoder(content_seq, style_seq)

        stylized_map = stylized_seq.transpose(1, 2).view(b, dim, h, w)
        return self.to_rgb(stylized_map)


def build_stytr2() -> nn.Module:
    """Build a compact StyTr2 model.

    Returns
    -------
    nn.Module
        StyTr2 in eval mode.
    """
    return StyTr2(dim=16, n_heads=2, n_layers=1).eval()


def example_input_stytr2() -> tuple[Tensor, Tensor]:
    """Example (content, style) image pair for StyTr2.

    Returns
    -------
    tuple[Tensor, Tensor]
        Two ``(1, 3, 32, 32)`` tensors.
    """
    return torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# SVG-VAE -- image VAE encoder + class/latent-conditioned LSTM + MDN decoder
# ---------------------------------------------------------------------------


class SvgVaeEncoder(nn.Module):
    """Convolutional image encoder producing a VAE bottleneck (mu, logvar)."""

    def __init__(self, latent_dim: int = 16, base: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_mu = nn.Linear(base * 2, latent_dim)
        self.fc_logvar = nn.Linear(base * 2, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a rendered glyph image into VAE (mu, logvar)."""
        feat = self.net(x).flatten(1)
        return self.fc_mu(feat), self.fc_logvar(feat)


class SvgVaeMdnDecoder(nn.Module):
    """Class+latent-conditioned LSTM stack with a Mixture-Density-Network head."""

    def __init__(
        self,
        latent_dim: int = 16,
        n_classes: int = 4,
        class_dim: int = 8,
        hidden: int = 24,
        n_layers: int = 4,
        n_mixtures: int = 3,
        cmd_dim: int = 2,
    ) -> None:
        super().__init__()
        self.class_embed = nn.Embedding(n_classes, class_dim)
        input_dim = cmd_dim + latent_dim + class_dim
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=n_layers, batch_first=True)
        self.n_mixtures = n_mixtures
        # MDN head: per-mixture (pi, mu_x, mu_y, sigma_x, sigma_y, rho) + 3 pen-state logits.
        self.mdn_head = nn.Linear(hidden, n_mixtures * 6 + 3)

    def forward(self, commands: Tensor, z: Tensor, class_id: Tensor) -> Tensor:
        """Autoregressively decode a step sequence into per-step MDN parameters."""
        b, t, _ = commands.shape
        class_vec = self.class_embed(class_id).unsqueeze(1).expand(-1, t, -1)
        z_vec = z.unsqueeze(1).expand(-1, t, -1)
        lstm_in = torch.cat([commands, z_vec, class_vec], dim=-1)
        hidden, _ = self.lstm(lstm_in)
        return self.mdn_head(hidden)


class SvgVae(nn.Module):
    """SVG-VAE: image VAE encoder + class/latent-conditioned LSTM+MDN SVG decoder."""

    def __init__(self, latent_dim: int = 16, n_classes: int = 4, hidden: int = 24) -> None:
        super().__init__()
        self.encoder = SvgVaeEncoder(latent_dim=latent_dim)
        self.decoder = SvgVaeMdnDecoder(latent_dim=latent_dim, n_classes=n_classes, hidden=hidden)

    def forward(self, image: Tensor, commands: Tensor, class_id: Tensor) -> Tensor:
        """Encode ``image`` and decode ``commands`` into per-step MDN parameters."""
        mu, logvar = self.encoder(image)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(commands, z, class_id)


def build_svg_vae() -> nn.Module:
    """Build a compact SVG-VAE model.

    Returns
    -------
    nn.Module
        SVG-VAE in eval mode.
    """
    return SvgVae(latent_dim=16, n_classes=4, hidden=24).eval()


def example_input_svg_vae() -> tuple[Tensor, Tensor, Tensor]:
    """Example (glyph image, command sequence, class id) inputs for SVG-VAE.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        A ``(1, 1, 32, 32)`` rendered glyph, a ``(1, 12, 2)`` pen-offset command
        sequence, and a ``(1,)`` integer class-id tensor.
    """
    return torch.randn(1, 1, 32, 32), torch.randn(1, 12, 2), torch.tensor([1])


# ---------------------------------------------------------------------------
# SVGDreamer -- SIVE attention-assigned primitives + VPSD particle population
# ---------------------------------------------------------------------------


class SiveAttentionAssigner(nn.Module):
    """SIVE: cross-attention between text tokens and image features assigns
    each learned Bezier-primitive control-point set to its attended region."""

    def __init__(self, feat_dim: int = 16, text_dim: int = 16, n_primitives: int = 4) -> None:
        super().__init__()
        self.image_proj = nn.Conv2d(feat_dim, text_dim, 1)
        self.text_proj = nn.Linear(text_dim, text_dim)
        self.primitive_query = nn.Parameter(torch.randn(n_primitives, text_dim))
        self.n_primitives = n_primitives

    def forward(self, image_feat: Tensor, text_tokens: Tensor) -> Tensor:
        """Return an attention-weighted spatial feature per primitive."""
        b, c, h, w = image_feat.shape
        img_kv = self.image_proj(image_feat).flatten(2).transpose(1, 2)  # (b, hw, d)
        text_kv = self.text_proj(text_tokens)  # (b, n_text, d)

        query = self.primitive_query.unsqueeze(0).expand(b, -1, -1)  # (b, n_prim, d)
        text_attn = torch.softmax(query @ text_kv.transpose(1, 2) / query.shape[-1] ** 0.5, dim=-1)
        text_context = text_attn @ text_kv  # (b, n_prim, d)

        spatial_attn = torch.softmax(
            text_context @ img_kv.transpose(1, 2) / query.shape[-1] ** 0.5, dim=-1
        )
        return spatial_attn @ img_kv  # (b, n_prim, d) -- primitive region assignment


class VpsdParticleScorer(nn.Module):
    """VPSD: score a population of vector particles with a compact noise-proxy net."""

    def __init__(self, ctrl_pts: int = 4, n_particles: int = 3) -> None:
        super().__init__()
        param_dim = ctrl_pts * 2 + 3  # (x, y) control points + RGB colour
        self.n_particles = n_particles
        self.param_dim = param_dim
        self.noise_proxy = nn.Sequential(
            nn.Linear(param_dim + 1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, param_dim),
        )

    def forward(self, particles: Tensor, timestep: Tensor) -> Tensor:
        """Predict the noise residual for each vector particle at ``timestep``."""
        b, n, d = particles.shape
        t = timestep.view(b, 1, 1).expand(-1, n, 1)
        return self.noise_proxy(torch.cat([particles, t], dim=-1))


class SvgDreamer(nn.Module):
    """SVGDreamer: SIVE attention-assigned primitives + VPSD particle-population scoring."""

    def __init__(
        self, feat_dim: int = 16, text_dim: int = 16, n_primitives: int = 4, n_particles: int = 3
    ) -> None:
        super().__init__()
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(3, feat_dim, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.sive = SiveAttentionAssigner(
            feat_dim=feat_dim, text_dim=text_dim, n_primitives=n_primitives
        )
        self.to_particle_params = nn.Linear(text_dim, 4 * 2 + 3)
        self.vpsd = VpsdParticleScorer(ctrl_pts=4, n_particles=n_particles)
        self.n_particles = n_particles

    def forward(
        self, image: Tensor, text_tokens: Tensor, timestep: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Assign primitives via SIVE, expand a particle population, score via VPSD."""
        image_feat = self.feat_encoder(image)
        primitive_ctx = self.sive(image_feat, text_tokens)  # (b, n_prim, text_dim)
        base_params = self.to_particle_params(primitive_ctx)  # (b, n_prim, param_dim)

        b, n_prim, param_dim = base_params.shape
        particles = base_params.unsqueeze(2).expand(-1, -1, self.n_particles, -1)
        noise = torch.randn_like(particles) * 0.1
        particles = (particles + noise).reshape(b, n_prim * self.n_particles, param_dim)

        noise_pred = self.vpsd(particles, timestep)
        return particles, noise_pred


def build_svgdreamer() -> nn.Module:
    """Build a compact SVGDreamer model.

    Returns
    -------
    nn.Module
        SVGDreamer in eval mode.
    """
    return SvgDreamer(feat_dim=16, text_dim=16, n_primitives=4, n_particles=3).eval()


def example_input_svgdreamer() -> tuple[Tensor, Tensor, Tensor]:
    """Example (image, text tokens, diffusion timestep) inputs for SVGDreamer.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        A ``(1, 3, 32, 32)`` raster proxy image, ``(1, 6, 16)`` text token
        embeddings, and a ``(1,)`` timestep tensor.
    """
    return torch.randn(1, 3, 32, 32), torch.randn(1, 6, 16), torch.tensor([500.0])


# ---------------------------------------------------------------------------
# UniColor -- Chroma-VQGAN codebook + Hybrid-Transformer over mixed tokens
# ---------------------------------------------------------------------------


class ChromaVectorQuantizer(nn.Module):
    """Chroma-VQGAN codebook: nearest-code lookup over a learned chroma embedding table."""

    def __init__(self, n_codes: int = 16, code_dim: int = 8) -> None:
        super().__init__()
        self.codebook = nn.Embedding(n_codes, code_dim)

    def forward(self, chroma_feat: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize ``chroma_feat`` (b, hw, d) against the codebook, straight-through."""
        flat = chroma_feat.reshape(-1, chroma_feat.shape[-1])
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        idx = dist.argmin(dim=1)
        quantized = self.codebook(idx).view_as(chroma_feat)
        quantized_st = chroma_feat + (quantized - chroma_feat).detach()
        return quantized_st, idx.view(chroma_feat.shape[0], chroma_feat.shape[1])


class UniColorHybridTransformer(nn.Module):
    """Hybrid-Transformer over mixed (grayscale, hint-point, mask) chroma tokens."""

    def __init__(
        self, dim: int = 16, n_heads: int = 2, n_layers: int = 2, n_codes: int = 16
    ) -> None:
        super().__init__()
        self.gray_proj = nn.Linear(1, dim)
        self.hint_embed = nn.Embedding(n_codes + 1, dim)  # +1 for the "no hint" mask token
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.chroma_head = nn.Linear(dim, n_codes)

    def forward(self, gray_patches: Tensor, hint_tokens: Tensor) -> Tensor:
        """Fuse grayscale luminance patches with quantized hint tokens, predict chroma logits."""
        gray_tok = self.gray_proj(gray_patches)
        hint_tok = self.hint_embed(hint_tokens)
        fused = gray_tok + hint_tok
        hidden = self.transformer(fused)
        return self.chroma_head(hidden)


class UniColor(nn.Module):
    """UniColor: Chroma-VQGAN chroma codebook + Hybrid-Transformer multi-modal colorizer."""

    def __init__(self, dim: int = 16, n_codes: int = 16) -> None:
        super().__init__()
        self.chroma_encoder = nn.Sequential(
            nn.Conv2d(2, dim, 3, padding=1, stride=2), nn.ReLU(inplace=True)
        )
        self.quantizer = ChromaVectorQuantizer(n_codes=n_codes, code_dim=dim)
        self.hybrid_transformer = UniColorHybridTransformer(dim=dim, n_codes=n_codes)
        self.chroma_decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 2, 3, padding=1),
        )

    def forward(
        self, gray: Tensor, exemplar_chroma: Tensor, hint_tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Quantize exemplar chroma hints and predict a full chroma grid from grayscale+hints."""
        chroma_feat = self.chroma_encoder(exemplar_chroma)
        b, d, h, w = chroma_feat.shape
        chroma_seq = chroma_feat.flatten(2).transpose(1, 2)
        _, hint_codes = self.quantizer(chroma_seq)

        gray_patches = F.avg_pool2d(gray, kernel_size=2).flatten(2).transpose(1, 2)
        chroma_logits = self.hybrid_transformer(gray_patches, hint_tokens)

        pred_idx = chroma_logits.argmax(dim=-1)
        pred_chroma = self.quantizer.codebook(pred_idx).transpose(1, 2).view(b, d, h, w)
        full_chroma = self.chroma_decoder(pred_chroma)
        return full_chroma, hint_codes


def build_unicolor() -> nn.Module:
    """Build a compact UniColor model.

    Returns
    -------
    nn.Module
        UniColor in eval mode.
    """
    return UniColor(dim=16, n_codes=16).eval()


def example_input_unicolor() -> tuple[Tensor, Tensor, Tensor]:
    """Example (grayscale, exemplar chroma, hint tokens) inputs for UniColor.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        A ``(1, 1, 32, 32)`` grayscale image, a ``(1, 2, 32, 32)`` exemplar
        chroma (ab) map, and a ``(1, 256)`` integer hint-token grid (16x16
        patches of a 16-code + 1 mask-token vocabulary).
    """
    return torch.randn(1, 1, 32, 32), torch.randn(1, 2, 32, 32), torch.randint(0, 17, (1, 256))


# ---------------------------------------------------------------------------
# Universal Style Transfer (WCT) -- multi-level whitening-coloring transform
# ---------------------------------------------------------------------------


def _whiten_color_transform(content_feat: Tensor, style_feat: Tensor, eps: float = 1e-5) -> Tensor:
    """Genuine Whitening-and-Coloring Transform (Li et al. 2017) for a single (b=1) map.

    Whiten ``content_feat`` by the inverse matrix square root of its own per-channel
    covariance (via eigendecomposition), then colour it with the matrix square root
    of ``style_feat``'s covariance and re-add the style mean.
    """
    b, c, h, w = content_feat.shape
    cf = content_feat.view(c, h * w)
    sf = style_feat.view(c, -1)

    c_mean = cf.mean(dim=1, keepdim=True)
    cf_centered = cf - c_mean
    c_cov = cf_centered @ cf_centered.t() / (cf_centered.shape[1] - 1) + eps * torch.eye(
        c, device=cf.device
    )
    c_eigval, c_eigvec = torch.linalg.eigh(c_cov)
    c_eigval = c_eigval.clamp_min(eps)
    whitening = c_eigvec @ torch.diag(c_eigval.pow(-0.5)) @ c_eigvec.t()
    whitened = whitening @ cf_centered

    s_mean = sf.mean(dim=1, keepdim=True)
    sf_centered = sf - s_mean
    s_cov = sf_centered @ sf_centered.t() / (sf_centered.shape[1] - 1) + eps * torch.eye(
        c, device=sf.device
    )
    s_eigval, s_eigvec = torch.linalg.eigh(s_cov)
    s_eigval = s_eigval.clamp_min(eps)
    coloring = s_eigvec @ torch.diag(s_eigval.pow(0.5)) @ s_eigvec.t()
    colored = coloring @ whitened + s_mean

    return colored.view(b, c, h, w)


class UstEncoderLevel(nn.Module):
    """One VGG-truncation encoder level (relu1_1 .. relu5_1 style, coarsened here)."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride), nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``x`` one relu-level deeper."""
        return self.net(x)


class UstDecoderLevel(nn.Module):
    """Symmetric decoder inverting one ``UstEncoderLevel`` back toward pixel space."""

    def __init__(self, in_ch: int, out_ch: int, upsample: bool) -> None:
        super().__init__()
        self.upsample = upsample
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        """Decode ``x`` one relu-level shallower."""
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.net(x)


class UniversalStyleTransfer(nn.Module):
    """Universal Style Transfer: multi-level VGG encoder/decoder pairs with a WCT
    (whitening-coloring transform) applied coarse-to-fine at every level."""

    def __init__(self, base: int = 4) -> None:
        super().__init__()
        # Five encoder levels (relu1_1 .. relu5_1 analogs). Only the first two levels
        # downsample so the deepest feature map still has far more spatial samples
        # than channels, keeping the per-channel covariance in the WCT well-posed.
        chans = [3, base, base, base, base, base]
        downsamples = [True, True, False, False, False]
        self.encoders = nn.ModuleList(
            [UstEncoderLevel(chans[i], chans[i + 1], downsample=downsamples[i]) for i in range(5)]
        )
        self.decoders = nn.ModuleList(
            [UstDecoderLevel(chans[i + 1], chans[i], upsample=downsamples[i]) for i in range(5)]
        )

    def _encode_to_level(self, x: Tensor, level: int) -> Tensor:
        """Run the encoder stack up through relu-level ``level`` (0-indexed)."""
        feat = x
        for enc in self.encoders[: level + 1]:
            feat = enc(feat)
        return feat

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Coarse-to-fine multi-level WCT stylization (level 5 down to level 1)."""
        stylized = content
        for level in reversed(range(5)):
            c_feat = self._encode_to_level(stylized, level)
            s_feat = self._encode_to_level(style, level)
            wct_feat = _whiten_color_transform(c_feat, s_feat)
            out = wct_feat
            for dec in reversed(self.decoders[: level + 1]):
                out = dec(out)
            stylized = out
        return stylized


def build_universal_style_transfer() -> nn.Module:
    """Build a compact Universal Style Transfer (WCT) model.

    Returns
    -------
    nn.Module
        UniversalStyleTransfer in eval mode.
    """
    return UniversalStyleTransfer(base=4).eval()


def example_input_universal_style_transfer() -> tuple[Tensor, Tensor]:
    """Example (content, style) image pair for Universal Style Transfer.

    Returns
    -------
    tuple[Tensor, Tensor]
        Two ``(1, 3, 32, 32)`` tensors.
    """
    return torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("StyleSwap", "build_styleswap", "example_input_styleswap", "2016", "VIS"),
    ("StyTr2", "build_stytr2", "example_input_stytr2", "2021", "VIS"),
    ("SVG-VAE", "build_svg_vae", "example_input_svg_vae", "2019", "GEN"),
    (
        "SVGDreamer (semantic-driven vector diffusion)",
        "build_svgdreamer",
        "example_input_svgdreamer",
        "2024",
        "GEN",
    ),
    (
        "UniColor (unified multi-modal colorization)",
        "build_unicolor",
        "example_input_unicolor",
        "2022",
        "VIS",
    ),
    (
        "Universal Style Transfer",
        "build_universal_style_transfer",
        "example_input_universal_style_transfer",
        "2017",
        "VIS",
    ),
]
