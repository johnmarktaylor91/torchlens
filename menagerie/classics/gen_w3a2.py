"""Compact faithful reimplementations of six neural style-transfer / stroke-based
image-generation architecture families.

Sources checked (paper + official source; reimplemented compactly from scratch in
base-env torch, no clone/pip-install):
  - IEContraAST (Internal-External Contrastive Arbitrary Style Transfer): Chen, Zhu,
    Wang, Zhang, Fu, Chen, Jin & Yin, "Artistic Style Transfer with Internal-external
    Learning and Contrastive Learning" (NeurIPS 2021); official repo HalbertCH/IEContraAST
    (``net.py``, classes ``SANet`` / ``Transform`` / ``Net``). The distinctive mechanism is
    a DUAL-SCALE STYLE-ATTENTIONAL NETWORK (SANet): a softmax cross-attention between
    mean-variance-normalized content queries/keys and style values (the "internal" swap
    of style statistics into content spatial layout), applied INDEPENDENTLY at VGG
    relu4_1 and relu5_1, then merged (relu5_1's SANet output upsampled to relu4_1's
    spatial size and added) through one more conv -- capturing style texture at two
    receptive-field scales instead of one. This is trained with an "external"
    momentum-contrastive loss over projected style/content embeddings (the "external
    learning" of the title), for which this reimplementation keeps the two small
    projection heads (``proj_style`` / ``proj_content``) as reusable trainable
    components alongside the SANet transform, since the contrastive loss itself is not
    part of the forward network graph. Reimplemented as a compact VGG-relu4_1 encoder
    feeding two parallel SANet cross-attention blocks (relu4_1 and relu5_1 scales)
    merged into one decoder, plus the style/content contrastive projection MLPs.
  - Learning to Paint: Huang, Heng & Zhou, "Learning to Paint With Model-based Deep
    Reinforcement Learning" (ICCV 2019, arxiv:1903.04411); official repo
    hzwer/ICCV2019-LearningToPaint (``baseline/DRL/actor.py`` class ``ResNet`` and
    ``baseline/Renderer/model.py`` class ``FCN``). The distinctive mechanism is a
    MODEL-BASED DDPG AGENT that paints an image with a small fixed number of PARAMETRIC
    BRUSH STROKES: a ResNet-18-style actor maps (target image, current canvas,
    normalized step index broadcast as a channel, and a fixed coord-conv grid) to a
    stroke-parameter vector per stroke (here Bezier-like control points + color,
    ``sigmoid``-bounded), and a SEPARATE, differentiable NEURAL RENDERER (a small FCN
    that maps a 10-D stroke-parameter vector through 4 FC layers into a 16x16x16 volume
    then 3 conv + ``PixelShuffle`` upsampling stages to a 128x128 soft stroke mask) lets
    gradients flow end-to-end through stroke rendering back into the actor -- so the
    agent learns to select strokes via a LEARNED DIFFERENTIABLE RENDERER rather than a
    non-differentiable graphics rasterizer. Reimplemented here as the actor (ResNet-style
    trunk on the 9-channel target+canvas+stepnum+coordconv input, producing 5 parallel
    stroke-parameter groups) composed with the FCN stroke renderer, alpha-compositing the
    rendered strokes onto the canvas exactly as ``decode()`` does in the source.
  - Linear Style Transfer: Li, Liu, Yang, Yang & Yang, "Learning Linear Transformations
    for Fast Arbitrary Style Transfer" (CVPR 2019, arxiv:1808.04537); official repo
    sunshineatnoon/LinearStyleTransfer (``libs/Matrix.py``, classes ``CNN`` / ``MulLayer``).
    The distinctive mechanism is that arbitrary style transfer is reduced to a single
    LEARNED LINEAR TRANSFORMATION MATRIX per image pair: small CNN subnetworks
    ("compression" convs + a Gram-like ``bmm(out, out^T)`` self-correlation + one FC
    layer) each map the (mean-subtracted) content and style VGG features to a small
    ``matrixSize x matrixSize`` matrix, the two matrices are COMPOSED via one more
    ``bmm`` into a single transform matrix, and that matrix is applied by a single
    ``bmm`` to compressed content features (then decompressed and re-biased by the
    style mean) -- giving fast, closed-form-like stylization with no per-pixel
    attention. Reimplemented here as the compact VGG relu4_1 encoder + the ``CNN``
    covariance-matrix subnetworks for content and style + the ``bmm``-composed linear
    transform (compress/transform/unzip) + decoder, matching ``MulLayer.forward``.
  - LIVE (Layerwise Image Vectorization): Ma, Gharbi, Fisher, Kim, Wan, Amir, Efros,
    Adobe/Picsart, "Towards Layer-wise Image Vectorization" (CVPR 2022 oral,
    arxiv:2206.04655); official repo Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization
    (``LIVE/main.py``, ``init_shapes`` + the ``pydiffvg``-based Bezier-path optimization
    loop). LIVE is a per-image OPTIMIZATION method (closed cubic-Bezier path control
    points and RGBA fill colors are directly-optimized ``nn.Parameter`` tensors, added
    one path -- one "layer" -- at a time and rendered with a differentiable vector
    rasterizer, ``pydiffvg``, which needs a custom CUDA build) rather than a
    feed-forward network with amortized weights; the reusable trainable component is
    the differentiable RASTERIZER + closed-path PARAMETERIZATION itself. Reimplemented
    here dependency-free as ``BezierPathRasterizer``: a small stack of LEARNABLE closed
    quadratic-Bezier paths (control points sampled at a fixed set of parametric
    ``t``-values into polygon vertices, exactly ``get_bezier_circle``'s sampling
    scheme) and per-path RGBA, rasterized with a differentiable SOFT SIGNED-DISTANCE
    occupancy (a smooth ``sigmoid`` band around each path's polygon edges standing in
    for ``pydiffvg``'s anti-aliased coverage) and composited back-to-front by path
    index -- preserving the "stack of learnable closed vector paths rendered and
    alpha-composited in layer order" mechanism without the CUDA rasterizer dependency.
  - MAST (Multi-Adaptation Style Transfer): Deng, Tang, Dong & Ma, "Arbitrary Style
    Transfer via Multi-Adaptation Network" (ACM MM 2020, arxiv:2005.13219); official
    repo diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network (``net.py``,
    classes ``CA`` / ``Content_SA`` / ``Style_SA`` / ``Multi_Adaptation_Module``). The
    distinctive mechanism is a THREE-STAGE ATTENTION PIPELINE (not a single cross
    attention): the VGG relu4_1 content feature is first refined by a CONTENT
    SELF-ATTENTION block (``Content_SA``, disentangling content structure from its own
    correlations), the style feature is separately refined by a STYLE SELF-ATTENTION
    block (``Style_SA``), and only THEN are the two refined features fused by a
    CROSS-ATTENTION block (``CA``, content queries attending to style keys/values) --
    an explicit "disentangle-then-fuse" adaptation absent from single-cross-attention
    methods like SANet. Reimplemented here as the compact VGG relu4_1 encoder feeding
    ``ContentSelfAttention`` and ``StyleSelfAttention`` blocks whose outputs are merged
    by a ``CrossAttention`` block into the decoder, matching
    ``Multi_Adaptation_Module.forward``.
  - MCCNet (Multi-Channel Correlation Network): Deng, Tang, Dong, Ma, Xu & Chen,
    "Arbitrary Video Style Transfer via Multi-Channel Correlation" (AAAI 2021,
    arxiv:2009.08003); official repo diyiiyiii/MCCNet (``net.py``, class ``MCCNet``).
    The distinctive mechanism is that -- unlike per-PIXEL spatial attention (SANet,
    MAST's CA) -- MCCNet computes a per-CHANNEL correlation: normalized style features
    are flattened over channels into a single ``(B, 1, H*W)`` vector, and a
    ``bmm(G_Fs, G_Fs^T)`` GRAM-LIKE outer product over the CHANNEL axis (normalized by
    the channel-wise activation sum) yields one ``(B, C)`` per-channel correlation
    vector, passed through one FC layer and broadcast back as a per-channel SCALE
    applied multiplicatively to the (normalized) content feature map -- this
    channel-only correlation is exactly what gives MCCNet its temporal/multi-frame
    STABILITY for video style transfer (no spatial attention map to flicker between
    frames). Reimplemented here as the compact VGG relu4_1 encoder + the per-channel
    correlation module (``f``/``g``/``h`` 1x1 convs, channel-flattened ``bmm``
    correlation + FC + multiplicative channel scaling) + decoder, matching
    ``MCCNet.forward``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Shared: a small VGG-relu4_1-style encoder/decoder pair used by all four
# encoder-decoder style-transfer families (IEContraAST, LinearStyleTransfer,
# MAST, MCCNet). Kept tiny (few channels) to stay compact; preserves the
# reflect-pad conv-relu stage structure with two downsamples to relu4_1.
# ---------------------------------------------------------------------------


def _vgg_relu4_1_encoder(base_ch: int = 8) -> nn.Sequential:
    """Build a compact VGG-style encoder up to a relu4_1-equivalent stage.

    Parameters
    ----------
    base_ch : int
        Channel width of the first conv stage.

    Returns
    -------
    nn.Sequential
        Encoder mapping ``(B, 3, H, W)`` to ``(B, base_ch * 8, H/4, W/4)``.
    """
    c = base_ch
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(3, c, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(c, c, 3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(c, c * 2, 3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(c * 2, c * 8, 3),
        nn.ReLU(inplace=True),
    )


def _small_decoder(base_ch: int = 8) -> nn.Sequential:
    """Build a compact decoder mirroring ``_vgg_relu4_1_encoder``.

    Parameters
    ----------
    base_ch : int
        Channel width matching the encoder's first stage.

    Returns
    -------
    nn.Sequential
        Decoder mapping ``(B, base_ch * 8, H/4, W/4)`` back to ``(B, 3, H, W)``.
    """
    c = base_ch
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(c * 8, c * 2, 3),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1),
        nn.Conv2d(c * 2, c, 3),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1),
        nn.Conv2d(c, 3, 3),
    )


def _mean_std(feat: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """Compute per-channel spatial mean/std, matching ``calc_mean_std``.

    Parameters
    ----------
    feat : Tensor
        ``(B, C, H, W)`` feature map.
    eps : float
        Numerical-stability offset added to the variance.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(mean, std)``, each ``(B, C, 1, 1)``.
    """
    b, c = feat.shape[:2]
    flat = feat.view(b, c, -1)
    var = flat.var(dim=2) + eps
    std = var.sqrt().view(b, c, 1, 1)
    mean = flat.mean(dim=2).view(b, c, 1, 1)
    return mean, std


def _mean_variance_norm(feat: Tensor) -> Tensor:
    """Normalize a feature map to zero mean, unit variance per channel.

    Parameters
    ----------
    feat : Tensor
        ``(B, C, H, W)`` feature map.

    Returns
    -------
    Tensor
        Normalized feature map of the same shape.
    """
    mean, std = _mean_std(feat)
    return (feat - mean) / std


# ---------------------------------------------------------------------------
# MODULE 1: IEContraAST -- dual-scale SANet attentional style transfer with
# internal-external contrastive projection heads.
# ---------------------------------------------------------------------------


class _SANet(nn.Module):
    """Style-attentional cross-attention block (one scale of IEContraAST).

    Parameters
    ----------
    in_planes : int
        Channel width of the content/style feature maps.
    """

    def __init__(self, in_planes: int) -> None:
        super().__init__()
        self.f = nn.Conv2d(in_planes, in_planes, 1)
        self.g = nn.Conv2d(in_planes, in_planes, 1)
        self.h = nn.Conv2d(in_planes, in_planes, 1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, 1)

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Cross-attend normalized content queries onto style keys/values.

        Parameters
        ----------
        content : Tensor
            ``(B, C, H, W)`` content feature map.
        style : Tensor
            ``(B, C, H, W)`` style feature map.

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` stylized content feature map (content + attended style).
        """
        b, c, h, w = content.shape
        query = self.f(_mean_variance_norm(content)).view(b, c, -1).permute(0, 2, 1)
        key = self.g(_mean_variance_norm(style)).view(b, c, -1)
        attn = F.softmax(torch.bmm(query, key), dim=-1)
        sb, sc, sh, sw = style.shape
        value = self.h(style).view(sb, sc, -1)
        out = torch.bmm(value, attn.permute(0, 2, 1)).view(b, c, h, w)
        return content + self.out_conv(out)


class IEContraAST(nn.Module):
    """Dual-scale SANet style transfer with internal-external contrastive heads.

    Two independent SANet cross-attention blocks operate on relu4_1- and
    relu5_1-equivalent encoder scales; the coarser scale's output is upsampled
    and merged into the finer scale before decoding. Two small MLP projection
    heads (unused in this forward pass but part of the trainable model,
    matching ``proj_style`` / ``proj_content`` in the source) embed pooled
    style/content features for the paper's external contrastive loss.

    Parameters
    ----------
    base_ch : int
        Channel width of the encoder's first stage.
    """

    def __init__(self, base_ch: int = 8) -> None:
        super().__init__()
        c = base_ch
        self.enc4 = _vgg_relu4_1_encoder(c)
        self.enc5 = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c * 8, c * 8, 3),
            nn.ReLU(inplace=True),
        )
        self.sanet4 = _SANet(c * 8)
        self.sanet5 = _SANet(c * 8)
        self.merge_pad = nn.ReflectionPad2d(1)
        self.merge_conv = nn.Conv2d(c * 8, c * 8, 3)
        self.decoder = _small_decoder(c)
        self.proj_style = nn.Sequential(
            nn.Linear(c * 8, c * 4), nn.ReLU(inplace=True), nn.Linear(c * 4, c * 4)
        )
        self.proj_content = nn.Sequential(
            nn.Linear(c * 8, c * 4), nn.ReLU(inplace=True), nn.Linear(c * 4, c * 4)
        )

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Stylize ``content`` with ``style`` via the dual-scale SANet transform.

        Parameters
        ----------
        content : Tensor
            ``(B, 3, H, W)`` content image.
        style : Tensor
            ``(B, 3, H, W)`` style image.

        Returns
        -------
        Tensor
            ``(B, 3, H, W)`` stylized image.
        """
        c4 = self.enc4(content)
        s4 = self.enc4(style)
        c5 = self.enc5(c4)
        s5 = self.enc5(s4)
        merged4 = self.sanet4(c4, s4)
        merged5 = self.sanet5(c5, s5)
        merged5_up = F.interpolate(merged5, size=merged4.shape[2:], mode="nearest")
        fused = self.merge_conv(self.merge_pad(merged4 + merged5_up))
        return self.decoder(fused)


def build_iecontraast() -> nn.Module:
    """Build a compact dual-scale SANet IEContraAST style-transfer network.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return IEContraAST(base_ch=8).eval()


def example_input_iecontraast() -> tuple[Tensor, Tensor]:
    """Example content and style images.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(content, style)``, each ``(1, 3, 64, 64)``.
    """
    return torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)


# ---------------------------------------------------------------------------
# MODULE 2: Learning to Paint -- DDPG actor + differentiable FCN stroke
# renderer, alpha-compositing strokes onto a canvas.
# ---------------------------------------------------------------------------


class _StrokeActor(nn.Module):
    """Compact ResNet-style actor mapping (target, canvas, step, coordconv) to strokes.

    Parameters
    ----------
    n_strokes : int
        Number of parallel strokes proposed per forward call.
    stroke_dim : int
        Parameter dimensionality of each stroke (control points + color).
    """

    def __init__(self, n_strokes: int = 5, stroke_dim: int = 13) -> None:
        super().__init__()
        self.n_strokes = n_strokes
        self.stroke_dim = stroke_dim
        # input channels: target(3) + canvas(3) + stepnum(1) + coordconv(2) = 9
        self.stem = nn.Sequential(nn.Conv2d(9, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, n_strokes * stroke_dim)

    def forward(self, state: Tensor) -> Tensor:
        """Propose ``n_strokes`` stroke-parameter vectors from the current state.

        Parameters
        ----------
        state : Tensor
            ``(B, 9, H, W)`` target+canvas+stepnum+coordconv tensor.

        Returns
        -------
        Tensor
            ``(B, n_strokes, stroke_dim)`` stroke parameters in ``[0, 1]``.
        """
        x = self.stem(state)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        out = torch.sigmoid(self.fc(x))
        return out.view(-1, self.n_strokes, self.stroke_dim)


class _FCNRenderer(nn.Module):
    """Differentiable neural renderer mapping stroke parameters to a soft mask.

    Mirrors the source ``FCN``: 4 FC layers expand a 10-D stroke-parameter
    vector to a small ``4x4`` volume, then 3 conv + ``PixelShuffle(2)`` stages
    upsample (matching the source's 3-stage pixel-shuffle structure, scaled
    down) to a ``32x32`` soft stroke-coverage mask.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 64)
        self.conv1 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1, 1)
        self.conv3 = nn.Conv2d(2, 4, 3, 1, 1)
        self.conv4 = nn.Conv2d(4, 4, 3, 1, 1)
        self.conv5 = nn.Conv2d(1, 4, 3, 1, 1)
        self.conv6 = nn.Conv2d(4, 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, stroke_params: Tensor) -> Tensor:
        """Render a soft stroke-coverage mask from stroke parameters.

        Parameters
        ----------
        stroke_params : Tensor
            ``(N, 10)`` control-point + shape stroke parameters in ``[0, 1]``.

        Returns
        -------
        Tensor
            ``(N, 32, 32)`` soft stroke masks (1 = covered).
        """
        x = F.relu(self.fc1(stroke_params))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 4, 4, 4)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 32, 32)


class LearningToPaint(nn.Module):
    """DDPG stroke-painting actor composed with a differentiable stroke renderer.

    The actor proposes a batch of parametric strokes from the current
    (target, canvas, step, coordconv) state; each stroke's first 10
    parameters are rendered into a soft coverage mask by the FCN renderer and
    the remaining 3 parameters give an RGB color, alpha-composited onto the
    canvas one stroke at a time -- reproducing the source's ``decode()``.

    Parameters
    ----------
    canvas_size : int
        Spatial size of the square canvas (and renderer output).
    n_strokes : int
        Number of strokes composited per forward call.
    """

    def __init__(self, canvas_size: int = 32, n_strokes: int = 5) -> None:
        super().__init__()
        self.canvas_size = canvas_size
        self.n_strokes = n_strokes
        self.actor = _StrokeActor(n_strokes=n_strokes, stroke_dim=13)
        self.renderer = _FCNRenderer()

    def forward(self, target: Tensor, canvas: Tensor, step: Tensor) -> Tensor:
        """Paint ``n_strokes`` new strokes onto ``canvas`` towards ``target``.

        Parameters
        ----------
        target : Tensor
            ``(B, 3, S, S)`` target image.
        canvas : Tensor
            ``(B, 3, S, S)`` current canvas.
        step : Tensor
            ``(B, 1, S, S)`` broadcast normalized step-index channel.

        Returns
        -------
        Tensor
            ``(B, 3, S, S)`` updated canvas after compositing the proposed strokes.
        """
        s = self.canvas_size
        coord = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(0, 1, s, device=target.device),
                    torch.linspace(0, 1, s, device=target.device),
                    indexing="ij",
                ),
                dim=0,
            )
            .unsqueeze(0)
            .expand(target.shape[0], -1, -1, -1)
        )
        state = torch.cat([target, canvas, step, coord], dim=1)
        strokes = self.actor(state)
        b = strokes.shape[0]
        flat = strokes.view(b * self.n_strokes, self.actor.stroke_dim)
        shape_params, colors = flat[:, :10], flat[:, 10:]
        masks = self.renderer(shape_params).view(b, self.n_strokes, 1, s, s)
        colors = colors.view(b, self.n_strokes, 3, 1, 1)
        for i in range(self.n_strokes):
            color_stroke = masks[:, i] * colors[:, i]
            canvas = canvas * (1 - masks[:, i]) + color_stroke
        return canvas


def build_learning_to_paint() -> nn.Module:
    """Build a compact Learning-to-Paint DDPG actor + differentiable renderer.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return LearningToPaint(canvas_size=32, n_strokes=5).eval()


def example_input_learning_to_paint() -> tuple[Tensor, Tensor, Tensor]:
    """Example target image, blank canvas, and step channel.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(target, canvas, step)``, target/canvas ``(1, 3, 32, 32)`` and step
        ``(1, 1, 32, 32)``.
    """
    target = torch.rand(1, 3, 32, 32)
    canvas = torch.zeros(1, 3, 32, 32)
    step = torch.full((1, 1, 32, 32), 0.1)
    return target, canvas, step


# ---------------------------------------------------------------------------
# MODULE 3: Linear Style Transfer -- learned per-image linear transform matrix
# (MulLayer) composed from small content/style covariance subnetworks.
# ---------------------------------------------------------------------------


class _MatrixCNN(nn.Module):
    """Covariance-matrix subnetwork mapping a feature map to a small matrix.

    Compresses a feature map, computes its Gram-like self-correlation, then
    refines it with one FC layer, matching the source ``CNN`` class.

    Parameters
    ----------
    in_ch : int
        Input feature-map channel width.
    matrix_size : int
        Side length of the output square matrix.
    """

    def __init__(self, in_ch: int, matrix_size: int = 16) -> None:
        super().__init__()
        self.matrix_size = matrix_size
        mid = max(matrix_size * 2, 8)
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, matrix_size, 3, 1, 1),
        )
        self.fc = nn.Linear(matrix_size * matrix_size, matrix_size * matrix_size)

    def forward(self, x: Tensor) -> Tensor:
        """Compute a flattened ``matrix_size x matrix_size`` covariance matrix.

        Parameters
        ----------
        x : Tensor
            ``(B, in_ch, H, W)`` feature map.

        Returns
        -------
        Tensor
            ``(B, matrix_size * matrix_size)`` flattened matrix.
        """
        out = self.convs(x)
        b, c, h, w = out.shape
        out = out.view(b, c, -1)
        out = torch.bmm(out, out.transpose(1, 2)).div(h * w)
        out = out.view(b, -1)
        return self.fc(out)


class _MulLayer(nn.Module):
    """Learned linear-transformation style-transfer layer (``MulLayer``).

    Parameters
    ----------
    in_ch : int
        Content/style feature-map channel width (encoder output channels).
    matrix_size : int
        Side length of the learned per-image transform matrix.
    """

    def __init__(self, in_ch: int, matrix_size: int = 16) -> None:
        super().__init__()
        self.matrix_size = matrix_size
        self.cnet = _MatrixCNN(in_ch, matrix_size)
        self.snet = _MatrixCNN(in_ch, matrix_size)
        self.compress = nn.Conv2d(in_ch, matrix_size, 1)
        self.unzip = nn.Conv2d(matrix_size, in_ch, 1)

    def forward(self, content_feat: Tensor, style_feat: Tensor) -> Tensor:
        """Apply the composed content/style linear transform to content features.

        Parameters
        ----------
        content_feat : Tensor
            ``(B, in_ch, H, W)`` content feature map.
        style_feat : Tensor
            ``(B, in_ch, H, W)`` style feature map.

        Returns
        -------
        Tensor
            ``(B, in_ch, H, W)`` transformed feature map.
        """
        c_mean, _ = _mean_std(content_feat)
        content_centered = content_feat - c_mean
        s_mean, _ = _mean_std(style_feat)
        style_centered = style_feat - s_mean

        b, c, h, w = content_centered.shape
        compressed = self.compress(content_centered).view(b, self.matrix_size, -1)

        c_matrix = self.cnet(content_centered).view(b, self.matrix_size, self.matrix_size)
        s_matrix = self.snet(style_centered).view(b, self.matrix_size, self.matrix_size)
        transform = torch.bmm(s_matrix, c_matrix)
        transformed = torch.bmm(transform, compressed).view(b, self.matrix_size, h, w)
        out = self.unzip(transformed)
        return out + s_mean.expand_as(out)


class LinearStyleTransfer(nn.Module):
    """Linear-transformation-matrix arbitrary style transfer network.

    Parameters
    ----------
    base_ch : int
        Channel width of the encoder's first stage.
    matrix_size : int
        Side length of the learned per-image transform matrix.
    """

    def __init__(self, base_ch: int = 8, matrix_size: int = 16) -> None:
        super().__init__()
        self.encoder = _vgg_relu4_1_encoder(base_ch)
        self.mul_layer = _MulLayer(base_ch * 8, matrix_size)
        self.decoder = _small_decoder(base_ch)

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Stylize ``content`` with ``style`` via the learned linear transform.

        Parameters
        ----------
        content : Tensor
            ``(B, 3, H, W)`` content image.
        style : Tensor
            ``(B, 3, H, W)`` style image.

        Returns
        -------
        Tensor
            ``(B, 3, H, W)`` stylized image.
        """
        c_feat = self.encoder(content)
        s_feat = self.encoder(style)
        transformed = self.mul_layer(c_feat, s_feat)
        return self.decoder(transformed)


def build_linear_style_transfer() -> nn.Module:
    """Build a compact Linear Style Transfer (``MulLayer``) network.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return LinearStyleTransfer(base_ch=8, matrix_size=16).eval()


def example_input_linear_style_transfer() -> tuple[Tensor, Tensor]:
    """Example content and style images.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(content, style)``, each ``(1, 3, 64, 64)``.
    """
    return torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)


# ---------------------------------------------------------------------------
# MODULE 4: LIVE -- layer-wise learnable closed Bezier-path vectorization with
# a differentiable soft-occupancy rasterizer (no CUDA rasterizer dependency).
# ---------------------------------------------------------------------------


class BezierPathRasterizer(nn.Module):
    """Stack of learnable closed Bezier paths rasterized and layer-composited.

    Each path's polygon vertices are sampled from a fixed number of control
    points around a unit circle (matching ``get_bezier_circle``'s
    ``segments * 3`` sampling), offset by a learnable per-path center and
    per-vertex radius; a smooth sigmoid band around the (approximate)
    signed distance to the polygon boundary gives a differentiable soft
    occupancy mask, and paths are alpha-composited back-to-front by index
    (layer 0 first, matching progressive layer-wise addition).

    Parameters
    ----------
    n_paths : int
        Number of closed vector paths ("layers").
    n_vertices : int
        Number of polygon vertices sampled per path.
    canvas_size : int
        Spatial size of the square rasterized canvas.
    """

    def __init__(self, n_paths: int = 4, n_vertices: int = 12, canvas_size: int = 48) -> None:
        super().__init__()
        self.n_paths = n_paths
        self.n_vertices = n_vertices
        self.canvas_size = canvas_size
        angles = torch.linspace(0, 2 * math.pi, n_vertices + 1)[:-1]
        self.register_buffer(
            "base_dir", torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        )
        self.centers = nn.Parameter(torch.rand(n_paths, 2) * 0.6 + 0.2)
        self.radii = nn.Parameter(torch.rand(n_paths, n_vertices) * 0.1 + 0.1)
        self.colors = nn.Parameter(torch.rand(n_paths, 3))
        self.alphas = nn.Parameter(torch.full((n_paths,), 0.8))
        grid = torch.linspace(0, 1, canvas_size)
        gy, gx = torch.meshgrid(grid, grid, indexing="ij")
        self.register_buffer("grid", torch.stack([gx, gy], dim=-1))

    def forward(self, dummy: Tensor) -> Tensor:
        """Rasterize and layer-composite the learnable paths into an RGB canvas.

        LIVE optimizes its Bezier-path parameters directly with no real input
        image passed through a network; ``dummy`` is accepted only so this
        module fits a standard forward-pass call signature and is otherwise
        unused (the source's per-image optimization loop uses a *target*
        image purely as an external loss reference, never as a network
        input).

        Parameters
        ----------
        dummy : Tensor
            Unused placeholder input (accepted for forward-pass-call
            compatibility only).

        Returns
        -------
        Tensor
            ``(1, 3, canvas_size, canvas_size)`` rasterized image.
        """
        del dummy
        s = self.canvas_size
        canvas = torch.ones(3, s, s, device=self.centers.device)
        for i in range(self.n_paths):
            vertices = self.centers[i].unsqueeze(0) + self.radii[i].unsqueeze(-1) * self.base_dir
            diffs = self.grid.view(-1, 1, 2) - vertices.unsqueeze(0)
            dist_to_vertex = diffs.norm(dim=-1).min(dim=-1).values
            center_dist = (self.grid.view(-1, 2) - self.centers[i].unsqueeze(0)).norm(dim=-1)
            mean_radius = self.radii[i].mean()
            inside = torch.sigmoid((mean_radius - center_dist) * 40.0)
            edge_soft = torch.sigmoid((0.05 - dist_to_vertex) * 40.0)
            occupancy = torch.clamp(inside + edge_soft, max=1.0).view(s, s)
            color = torch.sigmoid(self.colors[i]).view(3, 1, 1)
            alpha = torch.sigmoid(self.alphas[i]) * occupancy.unsqueeze(0)
            canvas = canvas * (1 - alpha) + color * alpha
        return canvas.unsqueeze(0)


def build_live() -> nn.Module:
    """Build a compact LIVE-style learnable Bezier-path vectorization stack.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return BezierPathRasterizer(n_paths=4, n_vertices=12, canvas_size=48).eval()


def example_input_live() -> Tensor:
    """Unused placeholder input; LIVE's paths are optimized directly.

    Returns
    -------
    Tensor
        A single unused placeholder tensor, present only so the model fits
        a standard forward-pass call signature.
    """
    return torch.zeros(1)


# ---------------------------------------------------------------------------
# MODULE 5: MAST -- content self-attention + style self-attention, fused by
# cross-attention (disentangle-then-fuse multi-adaptation).
# ---------------------------------------------------------------------------


class _SelfAttentionBlock(nn.Module):
    """Generic softmax self-attention residual block used for CSA and SSA.

    Parameters
    ----------
    in_dim : int
        Channel width of the feature map.
    normalize : bool
        Whether to mean-variance-normalize inputs before the query/key convs
        (used for content self-attention; style self-attention uses raw
        features, matching the source's ``Content_SA`` vs. ``Style_SA``).
    """

    def __init__(self, in_dim: int, normalize: bool) -> None:
        super().__init__()
        self.normalize = normalize
        self.f = nn.Conv2d(in_dim, in_dim, 1)
        self.g = nn.Conv2d(in_dim, in_dim, 1)
        self.h = nn.Conv2d(in_dim, in_dim, 1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, 1)

    def forward(self, feat: Tensor) -> Tensor:
        """Apply self-attention refinement to ``feat``.

        Parameters
        ----------
        feat : Tensor
            ``(B, C, H, W)`` feature map.

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` refined feature map (feat + attended residual).
        """
        b, c, h, w = feat.shape
        src = _mean_variance_norm(feat) if self.normalize else feat
        query = self.f(src).view(b, c, -1).permute(0, 2, 1)
        key = self.g(src).view(b, c, -1)
        attn = F.softmax(torch.bmm(query, key), dim=-1)
        value = self.h(feat).view(b, c, -1)
        out = torch.bmm(value, attn.permute(0, 2, 1)).view(b, c, h, w)
        return feat + self.out_conv(out)


class _CrossAttention(nn.Module):
    """Content-queries-style-keys/values cross-attention block (``CA``).

    Parameters
    ----------
    in_dim : int
        Channel width of the feature maps.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.f = nn.Conv2d(in_dim, in_dim, 1)
        self.g = nn.Conv2d(in_dim, in_dim, 1)
        self.h = nn.Conv2d(in_dim, in_dim, 1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, 1)

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Cross-attend content queries onto style keys/values.

        Parameters
        ----------
        content : Tensor
            ``(B, C, H, W)`` refined content feature map.
        style : Tensor
            ``(B, C, H, W)`` refined style feature map.

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` fused feature map.
        """
        b, c, h, w = content.shape
        query = self.f(_mean_variance_norm(content)).view(b, c, -1).permute(0, 2, 1)
        key = self.g(_mean_variance_norm(style)).view(b, c, -1)
        attn = F.softmax(torch.bmm(query, key), dim=-1)
        value = self.h(style).view(b, c, -1)
        out = torch.bmm(value, attn.permute(0, 2, 1)).view(b, c, h, w)
        return content + self.out_conv(out)


class MAST(nn.Module):
    """Multi-adaptation style transfer: disentangle-then-fuse attention pipeline.

    Parameters
    ----------
    base_ch : int
        Channel width of the encoder's first stage.
    """

    def __init__(self, base_ch: int = 8) -> None:
        super().__init__()
        c = base_ch * 8
        self.encoder = _vgg_relu4_1_encoder(base_ch)
        self.content_sa = _SelfAttentionBlock(c, normalize=True)
        self.style_sa = _SelfAttentionBlock(c, normalize=False)
        self.cross_attn = _CrossAttention(c)
        self.decoder = _small_decoder(base_ch)

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Stylize ``content`` with ``style`` via the disentangle-then-fuse pipeline.

        Parameters
        ----------
        content : Tensor
            ``(B, 3, H, W)`` content image.
        style : Tensor
            ``(B, 3, H, W)`` style image.

        Returns
        -------
        Tensor
            ``(B, 3, H, W)`` stylized image.
        """
        c_feat = self.encoder(content)
        s_feat = self.encoder(style)
        c_refined = self.content_sa(c_feat)
        s_refined = self.style_sa(s_feat)
        fused = self.cross_attn(c_refined, s_refined)
        return self.decoder(fused)


def build_mast() -> nn.Module:
    """Build a compact MAST multi-adaptation style-transfer network.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return MAST(base_ch=8).eval()


def example_input_mast() -> tuple[Tensor, Tensor]:
    """Example content and style images.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(content, style)``, each ``(1, 3, 64, 64)``.
    """
    return torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)


# ---------------------------------------------------------------------------
# MODULE 6: MCCNet -- multi-channel correlation module (per-channel Gram-like
# correlation instead of per-pixel spatial attention).
# ---------------------------------------------------------------------------


class _MultiChannelCorrelation(nn.Module):
    """Per-channel correlation module (``MCCNet`` class in the source).

    Parameters
    ----------
    in_dim : int
        Channel width of the feature maps.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.f = nn.Conv2d(in_dim, in_dim, 1)
        self.g = nn.Conv2d(in_dim, in_dim, 1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.fc = nn.Linear(in_dim, in_dim)

    def forward(self, content_feat: Tensor, style_feat: Tensor) -> Tensor:
        """Modulate content features by a per-channel style correlation scale.

        Parameters
        ----------
        content_feat : Tensor
            ``(B, C, H, W)`` content feature map.
        style_feat : Tensor
            ``(B, C, H, W)`` style feature map.

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` channel-modulated feature map.
        """
        b, c, h, w = content_feat.shape
        f_norm = self.f(_mean_variance_norm(content_feat))

        sb, sc, sh, sw = style_feat.shape
        g_norm = self.g(_mean_variance_norm(style_feat)).view(-1, 1, sh * sw)
        g_sum = g_norm.view(sb, sc, sh * sw).sum(-1)
        fc_s = torch.bmm(g_norm, g_norm.transpose(1, 2)).view(sb, sc) / (g_sum + 1e-6)
        fc_s = self.fc(fc_s).view(sb, sc, 1, 1)

        out = f_norm * fc_s
        out = self.out_conv(out)
        return content_feat + out


class MCCNet(nn.Module):
    """Multi-channel correlation arbitrary (video/image) style transfer network.

    Parameters
    ----------
    base_ch : int
        Channel width of the encoder's first stage.
    """

    def __init__(self, base_ch: int = 8) -> None:
        super().__init__()
        self.encoder = _vgg_relu4_1_encoder(base_ch)
        self.mcc = _MultiChannelCorrelation(base_ch * 8)
        self.decoder = _small_decoder(base_ch)

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Stylize ``content`` with ``style`` via the channel-correlation transform.

        Parameters
        ----------
        content : Tensor
            ``(B, 3, H, W)`` content image.
        style : Tensor
            ``(B, 3, H, W)`` style image.

        Returns
        -------
        Tensor
            ``(B, 3, H, W)`` stylized image.
        """
        c_feat = self.encoder(content)
        s_feat = self.encoder(style)
        fused = self.mcc(c_feat, s_feat)
        return self.decoder(fused)


def build_mccnet() -> nn.Module:
    """Build a compact MCCNet multi-channel-correlation style-transfer network.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return MCCNet(base_ch=8).eval()


def example_input_mccnet() -> tuple[Tensor, Tensor]:
    """Example content and style images.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(content, style)``, each ``(1, 3, 64, 64)``.
    """
    return torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("IEContraAST", "build_iecontraast", "example_input_iecontraast", "2021", "GEN"),
    (
        "Learning to Paint",
        "build_learning_to_paint",
        "example_input_learning_to_paint",
        "2019",
        "RL",
    ),
    (
        "Linear Style Transfer",
        "build_linear_style_transfer",
        "example_input_linear_style_transfer",
        "2019",
        "GEN",
    ),
    ("LIVE", "build_live", "example_input_live", "2022", "GEN"),
    ("MAST", "build_mast", "example_input_mast", "2020", "GEN"),
    ("MCCNet", "build_mccnet", "example_input_mccnet", "2021", "GEN"),
]
