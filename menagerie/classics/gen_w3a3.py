"""Compact faithful reimplementations of six low-level vision (inpainting / denoising /
neural painting / unified diffusion restoration) architecture families.

Sources checked (paper + official/community source; reimplemented compactly from
scratch in base-env torch, no clone/pip-install):
  - Noise2Same: Xie, Wang & Ji, "Noise2Same: Optimizing a Self-Supervised Bound for
    Image Denoising" (NeurIPS 2020, arxiv:2010.11971); official (TensorFlow) repo
    divelab/Noise2Same (``models.py``, class ``UNetEstimator``, and ``network.py``,
    class ``UNet``); TF framework, reimplemented here in PyTorch. The distinctive
    mechanism is a SELF-SUPERVISED UPPER BOUND combining STRATIFIED-RANDOM MASKED-NOISE
    SUBSTITUTION with a blind-spot-consistency penalty: a small STRATIFIED random
    fraction of pixels (grid-jittered so masked pixels are spread evenly, not a fixed
    checkerboard as in Noise2Self) are REPLACED with Gaussian noise (the source's
    ``masking='gaussian'`` mode, ``masked = (1-mask)*features + mask*noise``) before the
    same plain res-block U-Net predicts the whole image; the training loss then combines
    a RECONSTRUCTION term (masked network output vs. noisy input, at masked locations)
    with an INVARIANCE term ``lmbd * sqrt(mse(pred(masked_input), pred(raw_input)))`` at
    those same masked locations, which upper-bounds the (unavailable) supervised
    denoising loss without Noise2Self's stricter J-invariance requirement. Reimplemented
    as a plain res-block U-Net wrapped by a stratified-random noise-substitution masker
    (build one random binary mask per call, substitute Gaussian noise at masked pixels)
    exposing both the masked-input prediction and a raw-input prediction so the
    invariance term is computable, matching the source's ``UNetEstimator.model_fn``
    two-pass structure.
  - MEDFE (Mutual Encoder-Decoder with Feature Equalizations): Liu, Jiang, Xie, Chen,
    Yang, Wang & Wang, "Rethinking Image Inpainting via a Mutual Encoder-Decoder with
    Feature Equalizations" (ECCV 2020 oral, arxiv:2007.06929); official repo
    KumapowerLIU/Rethinking-Inpainting-MEDFE (``models/PCconv.py``, class ``PCconv``).
    The distinctive mechanism is a MUTUAL two-branch encoder-decoder that keeps
    separate TEXTURE (fine, shallow-layer) and STRUCTURE (coarse, deep-layer) feature
    streams: three shallow feature maps are stacked as the texture branch, three deep
    feature maps as the structure branch, and EACH branch is filled independently by
    multi-scale masked convolutions at kernel sizes 3/5/7 (parallel receptive fields,
    concatenated and fused). The two filled branches are then concatenated and pushed
    through a FEATURE-EQUALIZATION block (channel SE-gate + a non-local self-similarity
    attention over the hole/valid split) before being added BACK onto BOTH original
    branches as a shared residual -- so structure and texture mutually regularize each
    other through one shared equalized code, then diverge again for their own decoders.
    Reimplemented as a compact texture/structure dual-branch inpainting network: three
    masked-conv scales (3/5/7) per branch, concatenation + fuse, a channel-SE
    ("feature equalization") bottleneck, and a residual add-back into two separate
    output heads.
  - Noise2Score: Kim & Ye, "Noise2Score: Tweedie's Approach to Self-Supervised Image
    Denoising without Clean Images" (NeurIPS 2021, arxiv:2106.07009); official repo
    cubeyoung/Noise2Score (``models/Gaussian_model.py``, class ``GaussianModel``, and
    ``models/networks.py``, function ``define_G`` building a U-Net). The distinctive
    mechanism is TWEEDIE'S FORMULA SCORE ESTIMATION: rather than predicting a clean
    image directly, the network ``f`` is trained (via an amortized residual /ARDAE-style
    denoising-score-matching objective on the noisy input alone) to approximate the
    SCORE FUNCTION of the noisy distribution, and Tweedie's formula reconstructs the
    denoised image in closed form as ``recon = noisy + sigma**2 * f(noisy)`` -- i.e. the
    network output is a per-pixel correction term scaled by the (known or estimated)
    noise variance and added back onto the noisy input, rather than an end-to-end
    clean-image regression. Reimplemented as a small U-Net score network whose output is
    scaled by a learProgrammable noise-variance parameter and added back onto the input
    exactly per Tweedie's formula.
  - Noise2Self: Batson & Royer, "Noise2Self: Blind Denoising by Self-Supervision"
    (ICML 2019, arxiv:1901.11365); official repo czbiohub-sf/noise2self (``mask.py``,
    class ``Masker``). The distinctive mechanism is J-INVARIANT MASKING: input pixels
    are partitioned into a checkerboard-style grid of ``J`` disjoint pixel subsets (a
    ``width x width`` phase grid), and for a chosen subset the network only ever SEES
    the OTHER pixels (the chosen subset is zeroed / interpolated out of the input before
    the forward pass) yet is asked to predict exactly those held-out pixels -- so no
    pixel's prediction depends on its own (noisy) value, giving a valid self-supervised
    denoising objective without any clean targets or explicit noise model. Reimplemented
    as a small U-Net-style denoiser wrapped by a grid-mask module that builds the phase
    mask, zeroes the selected pixel subset (optionally appending the mask as an extra
    input channel, as in the source's ``include_mask_as_input``), and runs the network on
    the masked input -- the exact same J-invariant construction as the source's
    ``Masker.mask``.
  - Paint Transformer: Liu, Rong, Li, Feng, Sheng, Ji & Zhang, "Paint Transformer: Feed
    Forward Neural Painting with Stroke Prediction" (ICCV 2021, arxiv:2108.03798);
    solid unofficial PyTorch reimplementation Huage001/PaintTransformer
    (``inference/network.py``, class ``Painter``). The distinctive mechanism is FEED-
    FORWARD PARALLEL STROKE-SET PREDICTION VIA A DETR-STYLE TRANSFORMER: a target image
    and the current canvas are each CNN-encoded, concatenated channel-wise and projected
    to a token grid with learned row/column positional embeddings, then a
    ``nn.Transformer`` encoder-decoder maps a FIXED bank of learned stroke QUERIES (like
    DETR object queries) onto that token grid to predict, IN ONE PASS, a parameter
    vector per stroke (shape/color/position of a differentiable brush stroke) PLUS a
    binary keep/discard decision logit per stroke -- painting the whole canvas with many
    strokes simultaneously rather than sequentially (as earlier RL-based neural painters
    did) and rather than optimizing per-image (as stroke-optimization methods did).
    Reimplemented as a compact CNN dual-encoder (image + canvas) feeding a small
    ``nn.Transformer`` with a learned stroke-query bank and row/column positional
    embeddings, producing per-stroke parameters and a keep/discard decision, matching
    the source's ``Painter.forward`` signature and structure.
  - Palette (unified image-to-image diffusion): Saharia, Chan, Chang, Lee, Ho,
    Salimans, Fleet & Norouzi, "Palette: Image-to-Image Diffusion Models" (SIGGRAPH
    2022, arxiv:2111.05826); no official Google code release, unofficial PyTorch
    reimplementation Janspiry/Palette-Image-to-Image-Diffusion-Models
    (``models/network.py``, class ``Network``, and ``models/sr3_modules/unet.py``). The
    distinctive mechanism is a SINGLE CONDITIONAL DDPM SHARED ACROSS TASKS: the
    conditioning image ``y_cond`` (masked-for-inpainting / grayscale-for-colorization /
    cropped-for-uncropping / degraded-for-restoration) is CHANNEL-CONCATENATED with the
    noisy target ``y_t`` and fed, together with a sinusoidal noise-level ("gamma")
    embedding, through ONE denoising U-Net -- so the same architecture and training
    objective (denoising-score-matching / eps-prediction) handles four different
    image-to-image restoration tasks purely by swapping what image is placed in the
    conditioning channels, with no task-specific heads or losses. Reimplemented as a
    compact conditional-DDPM U-Net: ``[y_cond ; y_t]`` channel-concatenation, a
    sinusoidal timestep/noise-level embedding injected into every residual block via
    FiLM-style scale-and-shift, and a standard down/bottleneck/up U-Net trunk with skip
    connections, matching the source's concat-conditioning + noise-level-embedding
    design.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Noise2Same: stratified random noise-substitution masking + invariance bound
# ---------------------------------------------------------------------------


class _ResBlock(nn.Module):
    """Plain conv-norm-ReLU residual block, standing in for the source's ``res_block``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block."""
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class _ResUNet(nn.Module):
    """Small res-block U-Net, standing in for the source's ``UNet`` (TF ``network.py``)."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.enc = _ResBlock(channels)
        self.down = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.bottom = _ResBlock(channels * 2)
        self.up = nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1)
        self.dec = _ResBlock(channels)
        self.head = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict a denoised image from ``x``."""
        s = self.enc(self.stem(x))
        d = self.bottom(self.down(s))
        u = self.dec(s + self.up(d))
        return self.head(u)


class Noise2SameDenoiser(nn.Module):
    """Self-supervised denoiser optimizing Noise2Same's reconstruction + invariance bound.

    A stratified-random fraction of pixels is replaced with Gaussian noise before the
    res-block U-Net predicts the image (the source's ``masking='gaussian'`` mode); the
    same U-Net is also run on the raw (unmasked) input so the training-time invariance
    term (difference between masked-input and raw-input predictions at the masked
    locations) is computable, matching ``UNetEstimator.model_fn``'s two-pass structure.
    """

    def __init__(self, channels: int = 16, mask_fraction: float = 0.02) -> None:
        super().__init__()
        self.net = _ResUNet(channels)
        self.mask_fraction = mask_fraction

    def forward(self, noisy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Denoise ``noisy`` via stratified random noise-substitution masking.

        Parameters
        ----------
        noisy : Tensor
            ``(B, 3, H, W)`` noisy input image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(masked_pred, raw_pred, mask)``: the network's prediction from the
            noise-substituted input, its prediction from the raw input (for the
            invariance term), and the ``(B, 1, H, W)`` stratified random mask used.
        """
        mask = (
            torch.rand(noisy.shape[0], 1, *noisy.shape[2:], device=noisy.device)
            < self.mask_fraction
        ).float()
        substitute_noise = torch.randn_like(noisy) * 0.2
        masked_input = (1 - mask) * noisy + mask * substitute_noise
        masked_pred = self.net(masked_input)
        raw_pred = self.net(noisy)
        return masked_pred, raw_pred, mask


def build_noise2same() -> nn.Module:
    """Build a small Noise2Same stratified-masking self-supervised denoiser.

    Returns
    -------
    nn.Module
        ``Noise2SameDenoiser`` in eval mode.
    """
    return Noise2SameDenoiser(channels=12, mask_fraction=0.05).eval()


def example_input_noise2same() -> Tensor:
    """Example noisy image for Noise2Same.

    Returns
    -------
    Tensor
        ``(1, 3, 32, 32)`` noisy image.
    """
    return torch.rand(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# MEDFE: mutual texture/structure encoder-decoder with feature equalization
# ---------------------------------------------------------------------------


class _MaskedConvBlock(nn.Module):
    """One masked-conv-norm-activation stage at a given kernel size.

    Approximates the source's ``PartialConv`` (renormalization by the local valid-pixel
    count) with a plain conv applied to the mask-gated input, keeping the reimplementation
    dependency-free while preserving the multi-scale masked-fill mechanism.
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, feat: Tensor, mask: Tensor) -> Tensor:
        """Apply one masked convolution stage.

        Parameters
        ----------
        feat : Tensor
            ``(B, C, H, W)`` feature map.
        mask : Tensor
            ``(B, 1, H, W)`` valid-region mask (1 = valid, 0 = hole).

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` updated feature map.
        """
        out = self.conv(feat * mask)
        return self.act(self.norm(out))


class _FeatureEqualization(nn.Module):
    """Channel squeeze-excite bottleneck standing in for the source's ``BASE`` block."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Gate ``x`` channel-wise and project, the mutual "equalization" code."""
        gate = self.gate(self.pool(x))
        return self.proj(x * gate)


class MEDFEInpainting(nn.Module):
    """Mutual texture/structure encoder-decoder with feature equalization for inpainting.

    Two independent feature streams (texture = shallow/fine, structure = deep/coarse)
    are each filled by parallel multi-scale (3/5/7) masked convolutions, fused into a
    shared "equalized" code via a channel-SE bottleneck, and added back as a residual
    onto BOTH branches before their separate 1x1 output heads.
    """

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.texture_in = nn.Conv2d(3, channels, 3, padding=1)
        self.structure_in = nn.Conv2d(3, channels, 3, padding=1)
        self.texture_scales = nn.ModuleList([_MaskedConvBlock(channels, k) for k in (3, 5, 7)])
        self.structure_scales = nn.ModuleList([_MaskedConvBlock(channels, k) for k in (3, 5, 7)])
        self.texture_fuse = nn.Conv2d(channels * 3, channels, 1)
        self.structure_fuse = nn.Conv2d(channels * 3, channels, 1)
        self.joint_fuse = nn.Conv2d(channels * 2, channels, 1)
        self.equalize = _FeatureEqualization(channels)
        self.texture_out = nn.Conv2d(channels, 3, 3, padding=1)
        self.structure_out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Inpaint an image via mutual texture/structure branches.

        Parameters
        ----------
        image : Tensor
            ``(B, 3, H, W)`` masked-out input image (holes zeroed).
        mask : Tensor
            ``(B, 1, H, W)`` binary valid-region mask (1 = valid, 0 = hole).

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(texture_out, structure_out)``, each ``(B, 3, H, W)``, the two branches'
            independently-decoded but mutually-equalized reconstructions.
        """
        tex = self.texture_in(image)
        struct = self.structure_in(image)

        tex_scales = [scale(tex, mask) for scale in self.texture_scales]
        struct_scales = [scale(struct, mask) for scale in self.structure_scales]

        tex_fused = self.texture_fuse(torch.cat(tex_scales, dim=1))
        struct_fused = self.structure_fuse(torch.cat(struct_scales, dim=1))

        joint = self.joint_fuse(torch.cat([tex_fused, struct_fused], dim=1))
        equalized = self.equalize(joint)

        tex_final = tex + equalized
        struct_final = struct + equalized
        return self.texture_out(tex_final), self.structure_out(struct_final)


def build_medfe() -> nn.Module:
    """Build a small MEDFE mutual encoder-decoder inpainting network.

    Returns
    -------
    nn.Module
        ``MEDFEInpainting`` in eval mode.
    """
    return MEDFEInpainting(channels=16).eval()


def example_input_medfe() -> tuple[Tensor, Tensor]:
    """Example masked image and valid-region mask for MEDFE.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(image, mask)``, both spatial size ``32x32``.
    """
    image = torch.rand(1, 3, 32, 32)
    mask = torch.ones(1, 1, 32, 32)
    mask[:, :, 10:20, 10:20] = 0.0
    image = image * mask
    return image, mask


# ---------------------------------------------------------------------------
# Noise2Score: Tweedie's-formula score estimation for self-supervised denoising
# ---------------------------------------------------------------------------


class _ScoreUNet(nn.Module):
    """Tiny U-Net score network (stands in for the source's ``define_G`` U-Net)."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(channels * 2, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict the (unscaled) score correction for input ``x``."""
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        up = self.dec1(f2)
        return self.out(torch.cat([f1, up], dim=1))


class Noise2ScoreDenoiser(nn.Module):
    """Self-supervised denoiser via Tweedie's formula score estimation.

    The score network ``f`` predicts a per-pixel correction term; Tweedie's formula
    reconstructs the denoised image as ``recon = noisy + sigma**2 * f(noisy)``, matching
    the source's ``GaussianModel.forward``
    (``self.recon = self.variance * self.netf(self.lr, 0)[0] + self.lr``).
    """

    def __init__(self, channels: int = 16, init_log_variance: float = -4.0) -> None:
        super().__init__()
        self.score_net = _ScoreUNet(channels)
        self.log_variance = nn.Parameter(torch.tensor(init_log_variance))

    def forward(self, noisy: Tensor) -> Tensor:
        """Denoise ``noisy`` via the learned score and Tweedie's formula.

        Parameters
        ----------
        noisy : Tensor
            ``(B, 3, H, W)`` noisy input image.

        Returns
        -------
        Tensor
            ``(B, 3, H, W)`` denoised reconstruction.
        """
        score = self.score_net(noisy)
        variance = torch.exp(self.log_variance)
        return noisy + variance * score


def build_noise2score() -> nn.Module:
    """Build a small Noise2Score Tweedie-formula denoiser.

    Returns
    -------
    nn.Module
        ``Noise2ScoreDenoiser`` in eval mode.
    """
    return Noise2ScoreDenoiser(channels=12).eval()


def example_input_noise2score() -> Tensor:
    """Example noisy image for Noise2Score.

    Returns
    -------
    Tensor
        ``(1, 3, 32, 32)`` noisy image.
    """
    return torch.rand(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# Noise2Self: J-invariant grid masking for blind self-supervised denoising
# ---------------------------------------------------------------------------


class _DenoiseUNet(nn.Module):
    """Small U-Net denoiser body shared by the Noise2Self wrapper."""

    def __init__(self, in_channels: int, channels: int = 16) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.mid = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict a denoised image from the (masked) input ``x``."""
        return self.out(self.mid(self.enc(x)))


class Noise2SelfDenoiser(nn.Module):
    """J-invariant blind denoiser: predicts held-out pixels the network never saw.

    A ``grid_size x grid_size`` checkerboard phase mask selects one disjoint pixel
    subset to zero out of the input (optionally appended as an extra mask channel);
    the network only ever sees the OTHER pixels when predicting the masked-out ones,
    matching the source's ``Masker.mask`` J-invariant construction.
    """

    def __init__(self, grid_size: int = 3, include_mask_as_input: bool = True) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.include_mask_as_input = include_mask_as_input
        in_channels = 4 if include_mask_as_input else 3
        self.net = _DenoiseUNet(in_channels)

    def _phase_mask(self, height: int, width: int, phase: int, device: torch.device) -> Tensor:
        """Build the ``(1, 1, H, W)`` checkerboard mask (1 = held out) for one phase."""
        phase_row = phase % self.grid_size
        phase_col = (phase // self.grid_size) % self.grid_size
        rows = torch.arange(height, device=device).view(-1, 1)
        cols = torch.arange(width, device=device).view(1, -1)
        held_out = (rows % self.grid_size == phase_row) & (cols % self.grid_size == phase_col)
        return held_out.float().view(1, 1, height, width)

    def forward(self, image: Tensor, phase: int = 0) -> tuple[Tensor, Tensor]:
        """Denoise ``image`` using J-invariant masking at grid ``phase``.

        Parameters
        ----------
        image : Tensor
            ``(B, 3, H, W)`` noisy input image.
        phase : int
            Which of the ``grid_size**2`` disjoint pixel subsets to hold out.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(prediction, held_out_mask)``: the full-image prediction (only the
            held-out pixels are J-invariant) and the ``(B, 1, H, W)`` mask used.
        """
        _, _, height, width = image.shape
        held_out = self._phase_mask(height, width, phase, image.device)
        keep = 1.0 - held_out
        masked = image * keep
        if self.include_mask_as_input:
            net_input = torch.cat(
                [masked, held_out.expand(image.shape[0], 1, height, width)], dim=1
            )
        else:
            net_input = masked
        prediction = self.net(net_input)
        return prediction, held_out.expand(image.shape[0], 1, height, width)


def build_noise2self() -> nn.Module:
    """Build a small Noise2Self J-invariant denoiser.

    Returns
    -------
    nn.Module
        ``Noise2SelfDenoiser`` in eval mode.
    """
    return Noise2SelfDenoiser(grid_size=3).eval()


def example_input_noise2self() -> tuple[Tensor, int]:
    """Example noisy image and grid phase for Noise2Self.

    Returns
    -------
    tuple[Tensor, int]
        ``(image, phase)``, image ``(1, 3, 24, 24)``, phase in ``[0, grid_size**2)``.
    """
    return torch.rand(1, 3, 24, 24), 0


# ---------------------------------------------------------------------------
# Paint Transformer: feed-forward parallel stroke-set prediction via DETR-style
# transformer
# ---------------------------------------------------------------------------


class PaintTransformerPainter(nn.Module):
    """Feed-forward stroke-set predictor: DETR-style transformer over stroke queries.

    Image and canvas are each CNN-encoded, concatenated and projected to a token grid
    with learned row/column positional embeddings; a ``nn.Transformer`` maps a fixed
    bank of learned stroke queries onto that grid to predict, in one pass, a parameter
    vector and a keep/discard decision logit per stroke -- matching the source's
    ``Painter`` class structure and forward signature.
    """

    def __init__(
        self,
        param_per_stroke: int = 8,
        total_strokes: int = 16,
        hidden_dim: int = 32,
        n_heads: int = 4,
        n_enc_layers: int = 1,
        n_dec_layers: int = 1,
    ) -> None:
        super().__init__()

        def _encoder() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc_img = _encoder()
        self.enc_canvas = _encoder()
        self.conv = nn.Conv2d(32 * 2, hidden_dim, 1)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=hidden_dim * 2,
        )
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, param_per_stroke),
        )
        self.linear_decision = nn.Linear(hidden_dim, 1)
        self.query_pos = nn.Parameter(torch.rand(total_strokes, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))

    def forward(self, image: Tensor, canvas: Tensor) -> tuple[Tensor, Tensor]:
        """Predict a set of strokes transforming ``canvas`` towards ``image``.

        Parameters
        ----------
        image : Tensor
            ``(B, 3, H, W)`` target image.
        canvas : Tensor
            ``(B, 3, H, W)`` current canvas state.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(stroke_params, decision)``: ``(total_strokes, B, param_per_stroke)``
            per-stroke parameters and ``(total_strokes, B, 1)`` keep/discard logits.
        """
        batch = image.shape[0]
        img_feat = self.enc_img(image)
        canvas_feat = self.enc_canvas(canvas)
        height, width = img_feat.shape[-2:]
        feat = torch.cat([img_feat, canvas_feat], dim=1)
        feat_conv = self.conv(feat)

        pos_embed = (
            torch.cat(
                [
                    self.col_embed[:width].unsqueeze(0).repeat(height, 1, 1),
                    self.row_embed[:height].unsqueeze(1).repeat(1, width, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )
        tokens = feat_conv.flatten(2).permute(2, 0, 1)
        hidden_state = self.transformer(
            pos_embed + tokens,
            self.query_pos.unsqueeze(1).repeat(1, batch, 1),
        )
        params = self.linear_param(hidden_state)
        decision = self.linear_decision(hidden_state)
        return params, decision


def build_paint_transformer() -> nn.Module:
    """Build a small Paint Transformer feed-forward stroke predictor.

    Returns
    -------
    nn.Module
        ``PaintTransformerPainter`` in eval mode.
    """
    return PaintTransformerPainter(
        param_per_stroke=8, total_strokes=12, hidden_dim=24, n_heads=4
    ).eval()


def example_input_paint_transformer() -> tuple[Tensor, Tensor]:
    """Example target image and current canvas for Paint Transformer.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(image, canvas)``, both ``(1, 3, 16, 16)``.
    """
    return torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)


# ---------------------------------------------------------------------------
# Palette: unified conditional-DDPM image-to-image diffusion
# ---------------------------------------------------------------------------


def _sinusoidal_embedding(noise_level: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding of a per-sample scalar noise level, as in the source's gamma
    embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=noise_level.device).float() / half
    )
    args = noise_level[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _FiLMResBlock(nn.Module):
    """Conv residual block with FiLM-style noise-level conditioning."""

    def __init__(self, channels: int, emb_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, channels)
        self.norm2 = nn.GroupNorm(4, channels)
        self.emb_proj = nn.Linear(emb_dim, channels * 2)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Apply the FiLM-conditioned residual block."""
        scale, shift = self.emb_proj(emb).chunk(2, dim=-1)
        h = self.act(self.norm1(self.conv1(x)))
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return x + h


class PaletteConditionalUNet(nn.Module):
    """Unified conditional-DDPM denoiser for image-to-image diffusion tasks.

    The conditioning image ``y_cond`` is channel-concatenated with the noisy target
    ``y_t`` and fed, together with a sinusoidal noise-level embedding injected via FiLM
    into every residual block, through one U-Net trunk -- the same network handles
    inpainting/colorization/uncropping/restoration by varying what is placed in
    ``y_cond``, matching the source's ``Network``/``sr3_modules.unet.UNet`` design.
    """

    def __init__(self, channels: int = 24, emb_dim: int = 32) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.stem = nn.Conv2d(6, channels, 3, padding=1)
        self.down = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.res_low = _FiLMResBlock(channels * 2, emb_dim)
        self.up = nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1)
        self.res_high = _FiLMResBlock(channels, emb_dim)
        self.head = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, y_cond: Tensor, y_t: Tensor, noise_level: Tensor) -> Tensor:
        """Predict the noise (or residual) added to the target given the condition.

        Parameters
        ----------
        y_cond : Tensor
            ``(B, 3, H, W)`` conditioning image (masked / grayscale / cropped / degraded
            input, depending on the task).
        y_t : Tensor
            ``(B, 3, H, W)`` noisy target at the current diffusion timestep.
        noise_level : Tensor
            ``(B,)`` per-sample scalar noise level (cumulative-alpha "gamma").

        Returns
        -------
        Tensor
            ``(B, 3, H, W)`` predicted noise / correction term.
        """
        emb = self.emb_mlp(_sinusoidal_embedding(noise_level, self.emb_dim))
        x = self.stem(torch.cat([y_cond, y_t], dim=1))
        low = self.down(x)
        low = self.res_low(low, emb)
        up = self.up(low)
        high = self.res_high(x + up, emb)
        return self.head(high)


def build_palette() -> nn.Module:
    """Build a small Palette unified conditional-DDPM restoration network.

    Returns
    -------
    nn.Module
        ``PaletteConditionalUNet`` in eval mode.
    """
    return PaletteConditionalUNet(channels=16, emb_dim=32).eval()


def example_input_palette() -> tuple[Tensor, Tensor, Tensor]:
    """Example condition image, noisy target, and noise level for Palette.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(y_cond, y_t, noise_level)``: two ``(1, 3, 32, 32)`` images and a ``(1,)``
        scalar noise level in ``[0, 1)``.
    """
    y_cond = torch.rand(1, 3, 32, 32)
    y_t = torch.rand(1, 3, 32, 32)
    noise_level = torch.tensor([0.5])
    return y_cond, y_t, noise_level


MENAGERIE_ENTRIES = [
    ("Noise2Same", "build_noise2same", "example_input_noise2same", "2020", "VIS"),
    ("MEDFE", "build_medfe", "example_input_medfe", "2020", "VIS"),
    ("Noise2Score", "build_noise2score", "example_input_noise2score", "2021", "VIS"),
    ("Noise2Self", "build_noise2self", "example_input_noise2self", "2019", "VIS"),
    (
        "Paint Transformer",
        "build_paint_transformer",
        "example_input_paint_transformer",
        "2021",
        "GEN",
    ),
    (
        "Palette (unified image-to-image diffusion)",
        "build_palette",
        "example_input_palette",
        "2022",
        "GEN",
    ),
]
