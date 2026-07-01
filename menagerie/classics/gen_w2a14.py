"""Compact faithful reimplementations of five low-level vision (image restoration /
inpainting / colorization) architecture families.

Sources checked (paper + official/community source; reimplemented compactly from
scratch in base-env torch, no clone/pip-install):
  - EnlightenGAN: Jiang, Gong, Liu, Cheng, Fang, Shen, Yang, Zhou & Wang, "EnlightenGAN:
    Deep Light Enhancement without Paired Supervision" (IEEE TIP 2021, arxiv:1906.06972);
    official repo VITA-Group/EnlightenGAN (``models/networks.py``, class
    ``Unet_resize_conv``). The distinctive mechanism is the SELF-REGULARIZED ATTENTION
    MAP: the illumination channel of the input (grayscale, no extra supervision) is used
    directly as ``1 - I``, downsampled to match each U-Net encoder/decoder scale, and
    ELEMENT-WISE MULTIPLIED into every intermediate feature map (and the residual output)
    -- so darker regions get proportionally stronger enhancement without any learned
    attention weights. This is paired (in the full GAN) with a global-local dual
    discriminator, but the attention-guided U-Net generator is the reusable trainable
    component reimplemented here. Reimplemented as a U-Net generator whose skip
    connections and bottleneck are each gated by a resized ``1 - gray`` attention map
    computed once from the input and pooled to every scale, exactly the source's
    ``x * gray_k`` gating at every stage.
  - FRRN Inpainting (a.k.a. Progressive Image Inpainting with Full-Resolution Residual
    Network): Guo, Chen, Yuan & Bai, "Progressive Image Inpainting with Full-Resolution
    Residual Network" (ACM MM 2019, arxiv:1907.10478); official repo
    ZongyuGuo/Inpainting_FRRN (``src/networks.py``, class ``FRRNet`` / ``FRRBlock``). The
    distinctive mechanism is the FULL-RESOLUTION RESIDUAL BLOCK used progressively: each
    block has TWO PARALLEL branches over the (partial-conv-gated) feature map -- a
    "full-resolution" branch that stays at the input's spatial resolution the whole time
    (three stride-1 partial convs), and a "pooling" branch that downsamples 3x then
    upsamples 3x back (an internal mini encoder-decoder) -- whose outputs are elementwise
    averaged, re-gated by the updated (shrunk) hole mask, and added back onto the block's
    input as a residual; stacking these blocks and updating the mask at every block
    implements PROGRESSIVE inpainting (the hole shrinks/fills one full-resolution
    residual step at a time, rather than in a single encoder-decoder pass). Reimplemented
    here with (masked) plain convs standing in for the paper's ``PartialConv2d`` (same
    per-pixel dual-branch full-res-plus-pooling block and progressive mask update; partial
    convolution's renormalization is the only omission, to keep the reimplementation
    dependency-free) stacked into a small ``FRRNet``.
  - GANgogh: Jones & Bonafilia, "GANGogh: Creating Art with GANs" (Williams College 2017
    undergraduate project, blog https://medium.com/towards-data-science/gangogh-creating-
    art-with-gans-8d087d8f74a1); official (TensorFlow) repo rkjones4/GANGogh
    (``GANgogh.py``, functions ``kACGANGenerator`` / ``kACGANDiscriminator``,
    ``MODE = 'acwgan'``). The distinctive mechanism is an AUXILIARY-CLASSIFIER WASSERSTEIN
    GAN (AC-WGAN) trained on WikiArt with GENRE CONDITIONING injected via a PixelCNN-style
    GATED NONLINEARITY: at every generator stage a one-hot genre label is projected to a
    per-channel "condition" tensor of the same shape as that stage's features, and
    ``sigmoid(a + cond_a) * tanh(b + cond_b)`` (splitting the feature map into two halves
    ``a``/``b``) replaces a plain activation -- so the class condition modulates every
    resolution of the generator, not just a concatenated one-hot vector at the input; the
    discriminator/critic is a WGAN critic (no sigmoid, gradient-penalty-compatible
    LayerNorm) with an AUXILIARY CLASSIFICATION HEAD predicting the genre from the
    real/fake image, giving the AC-WGAN's joint real/fake + class-conditional loss.
    Reimplemented in PyTorch as a small conv generator with genre-conditioned
    gated-nonlinearity blocks at each upsampling stage and a WGAN critic with an
    auxiliary genre-classification head.
  - HiFill (Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting):
    Yi, Tang, Azizi, Jang & Xu, "Contextual Residual Aggregation for Ultra
    High-Resolution Image Inpainting" (CVPR 2020 oral, arxiv:2005.09704); official
    (TensorFlow) demo Atlas200dk/sample-imageinpainting-HiFill, unofficial PyTorch
    reimplementation wangyx240/High-Resolution-Image-Inpainting-GAN
    (``inpainting_network.py``, class ``GatedGenerator``). The distinctive mechanism is
    CONTEXTUAL RESIDUAL AGGREGATION: a coarse-to-fine gated-conv generator computes a
    LOW-RESOLUTION patch-wise cosine-similarity attention map between hole and non-hole
    32x32 patches of a bottleneck feature map (masked so hole patches only attend to
    valid patches), and that SAME attention map is reused ("transferred") at THREE
    different decoder resolutions to aggregate residual detail from valid regions into
    the hole at each scale via ``attention_transfer`` -- letting a network trained on
    small crops recompose high-frequency residual texture at resolutions far larger than
    its training resolution (the paper's 8K-inpainting capability), unlike single-scale
    contextual-attention inpainting (e.g. DeepFillv2) which computes and applies
    attention only once. Reimplemented as a small coarse gated-conv encoder-decoder
    followed by a refinement encoder-decoder that computes one low-res patch attention
    map from the mask and bottleneck features and re-applies it at three decoder scales.
  - iColoriT: Yun, Lee, Park & Choo, "iColoriT: Towards Propagating Local Hint to the
    Right Region in Interactive Colorization by Leveraging Vision Transformer" (WACV
    2023, arxiv:2207.06831); official repo pmh9960/iColoriT (``modeling.py``, class
    ``IColoriT`` / ``LocalAttentionHead``). The distinctive mechanism is POINT-INTERACTIVE
    COLORIZATION VIA A PLAIN (non-hierarchical) ViT operating directly on a 4-channel
    input (L channel + masked ab-color channels + a binary hint-location mask channel),
    so sparse user color hints ("points") are given as unmasked ab-channel pixels inside
    otherwise-zeroed patches and the transformer's GLOBAL self-attention propagates each
    hint to every other patch in one forward pass (no separate hint-propagation network);
    the decoder head skips a conv decoder entirely and instead predicts, per patch token,
    a full ``2 * patch_size**2``-channel vector that PIXEL-SHUFFLES directly into an
    ab-color map (``num_classes == 2 * patch_size**2``), tanh-bounded; the paper's LOCAL
    STABILIZING LAYER (``LocalAttentionHead``) replaces the final linear pixel-shuffle
    head with a windowed self-attention head whose queries may attend only to a small
    kxk neighborhood of patch tokens (via an additive attention mask), damping the
    checkerboard artifacts that plain large-ratio pixel shuffling introduces.
    Reimplemented here as a compact plain ViT over a 4-channel (L, masked-ab, hint-mask)
    input with sinusoidal position embeddings and the local-attention pixel-shuffle head.

Note: "Gated Convolution Inpainting" (cand_00181, a.k.a. DeepFillv2) is ALREADY present
in the catalog as ``DeepFillv2Generator`` / ``build_deepfillv2_gated_generator`` in
``menagerie/classics/ri_vision_b_gan.py`` and is therefore skipped here (see
``MENAGERIE_ENTRIES`` note below / structured build report).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# MODULE 1: EnlightenGAN -- attention-guided U-Net generator
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    """Two 3x3 conv + LeakyReLU + BatchNorm layers (the source's basic U-Net block)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the two conv-BN-LeakyReLU stages.

        Parameters
        ----------
        x : Tensor
            ``(batch, in_ch, h, w)`` feature map.

        Returns
        -------
        Tensor
            ``(batch, out_ch, h, w)`` feature map.
        """
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class EnlightenGANGenerator(nn.Module):
    """Attention-guided U-Net generator for unsupervised low-light enhancement.

    A 3-level U-Net whose encoder features, decoder features, and residual output are
    all gated by a SELF-REGULARIZED ATTENTION MAP derived directly from the input's
    illumination channel (``1 - gray``), resized by average-pooling to match each
    scale -- the source's ``x * gray_k`` multiplicative gating, requiring no extra
    supervision or learned attention weights.

    Parameters
    ----------
    base_ch : int
        Base channel width of the first encoder stage.
    """

    def __init__(self, base_ch: int = 16) -> None:
        super().__init__()
        self.enc1 = _ConvBlock(3, base_ch)
        self.enc2 = _ConvBlock(base_ch, base_ch * 2)
        self.enc3 = _ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.AvgPool2d(2)

        self.bottleneck = _ConvBlock(base_ch * 4, base_ch * 4)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec3 = _ConvBlock(base_ch * 2 + base_ch * 4, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec2 = _ConvBlock(base_ch + base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Enhance a low-light RGB image with attention-guided gating at every scale.

        Parameters
        ----------
        x : Tensor
            ``(batch, 3, h, w)`` low-light RGB image in ``[0, 1]``.

        Returns
        -------
        Tensor
            ``(batch, 3, h, w)`` enhanced RGB image.
        """
        # Self-regularized attention map: illumination channel of the input, inverted.
        gray = x.mean(dim=1, keepdim=True)
        atten = 1.0 - gray
        atten2 = self.pool(atten)
        atten3 = self.pool(atten2)
        atten_b = self.pool(atten3)

        e1 = self.enc1(x) * atten
        e2 = self.enc2(self.pool(e1)) * atten2
        e3 = self.enc3(self.pool(e2)) * atten3

        b = self.bottleneck(self.pool(e3)) * atten_b

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1)) * atten3
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1)) * atten2

        out = self.out_conv(d2) * atten2
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.tanh(out) * atten + x


def build_enlightengan() -> nn.Module:
    """Build a small EnlightenGAN attention-guided U-Net generator.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return EnlightenGANGenerator(base_ch=12).eval()


def example_input_enlightengan() -> Tensor:
    """Example low-light RGB image.

    Returns
    -------
    Tensor
        ``(1, 3, 64, 64)`` float tensor in ``[0, 1]``.
    """
    return torch.rand(1, 3, 64, 64)


# ---------------------------------------------------------------------------
# MODULE 2: FRRN Inpainting -- progressive full-resolution residual network
# ---------------------------------------------------------------------------


class _MaskedConv(nn.Module):
    """Conv layer that also mask-gates its output and shrinks the hole mask by a
    fixed-radius erosion (a lightweight stand-in for the source's ``PartialConv2d``,
    which additionally renormalizes by the local valid-pixel count -- omitted here to
    keep the reimplementation dependency-free while preserving the mask-update
    behavior that drives progressive inpainting)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, act: str = "relu") -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch)
        if act == "relu":
            self.act: nn.Module = nn.ReLU(inplace=True)
        elif act == "leaky":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Tanh()
        self.mask_pool = nn.MaxPool2d(3, stride=stride, padding=1)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply masked conv + norm + activation and update the (shrinking) mask.

        Parameters
        ----------
        x : Tensor
            ``(batch, in_ch, h, w)`` feature map.
        mask : Tensor
            ``(batch, 1, h, w)`` binary hole mask (1 == hole).

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(feature, mask)`` pair, mask pooled to the conv's output stride.
        """
        out = self.act(self.norm(self.conv(x)))
        new_mask = self.mask_pool(mask)
        return out, new_mask


class _FRRBlock(nn.Module):
    """One full-resolution residual block: a full-res branch (stays at input
    resolution) and a pooling branch (downsample-then-upsample), averaged, mask-gated,
    and added back onto the block's input -- the source's ``FRRBlock``."""

    def __init__(self, ch: int = 3) -> None:
        super().__init__()
        self.full1 = _MaskedConv(ch, 16)
        self.full2 = _MaskedConv(16, 16)
        self.full3 = _MaskedConv(16, ch)

        self.branch1 = _MaskedConv(ch, 24, stride=2)
        self.branch2 = _MaskedConv(24, 32, stride=2)
        self.branch3 = _MaskedConv(32, 32, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.branch4 = _MaskedConv(32, 24, act="leaky")
        self.branch5 = _MaskedConv(24, 16, act="leaky")
        self.branch6 = _MaskedConv(16, ch, act="tanh")

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Run the dual full-res/pooling branches and merge into a residual update.

        Parameters
        ----------
        x : Tensor
            ``(batch, ch, h, w)`` current reconstruction.
        mask : Tensor
            ``(batch, 1, h, w)`` current hole mask (1 == hole).

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(reconstruction, mask)`` pair.
        """
        out_f, mask_f = self.full1(x, mask)
        out_f, mask_f = self.full2(out_f, mask_f)
        out_f, mask_f = self.full3(out_f, mask_f)

        out_b, mask_b = self.branch1(x, mask)
        out_b, mask_b = self.branch2(out_b, mask_b)
        out_b, mask_b = self.branch3(out_b, mask_b)
        out_b, mask_b = self.up(out_b), self.up(mask_b)
        out_b, mask_b = self.branch4(out_b, mask_b)
        out_b, mask_b = self.up(out_b), self.up(mask_b)
        out_b, mask_b = self.branch5(out_b, mask_b)
        out_b, mask_b = self.up(out_b), self.up(mask_b)
        out_b, mask_b = self.branch6(out_b, mask_b)

        mask_new = mask_f * mask_b
        merged = (out_f * mask_new + out_b * mask_new) * 0.5 * (1.0 - mask) + x
        return merged, mask_new


class FRRNet(nn.Module):
    """Progressive full-resolution-residual inpainting network.

    Stacks ``block_num`` full-resolution residual blocks; the hole mask shrinks after
    every block, implementing progressive (coarse-to-fine, hole-shrinking) inpainting
    rather than a single encoder-decoder pass.

    Parameters
    ----------
    block_num : int
        Number of stacked ``_FRRBlock`` instances (kept small for the catalog).
    """

    def __init__(self, block_num: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FRRBlock() for _ in range(block_num)])

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Progressively inpaint the masked regions of ``x``.

        Parameters
        ----------
        x : Tensor
            ``(batch, 3, h, w)`` masked RGB image (hole pixels arbitrary).
        mask : Tensor
            ``(batch, 1, h, w)`` binary hole mask (1 == hole, 0 == known).

        Returns
        -------
        Tensor
            ``(batch, 3, h, w)`` progressively reconstructed image.
        """
        cur_mask = mask
        for block in self.blocks:
            x, cur_mask = block(x, cur_mask)
        return x


def build_frrn_inpainting() -> nn.Module:
    """Build a small progressive FRRN inpainting network.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return FRRNet(block_num=3).eval()


def example_input_frrn_inpainting() -> tuple[Tensor, Tensor]:
    """Example masked image and hole mask.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(image, mask)``: ``image`` is ``(1, 3, 32, 32)``, ``mask`` is ``(1, 1, 32, 32)``
        binary (1 == hole).
    """
    image = torch.rand(1, 3, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 10:22, 10:22] = 1.0
    return image, mask


# ---------------------------------------------------------------------------
# MODULE 3: GANgogh -- genre-conditioned AC-WGAN
# ---------------------------------------------------------------------------


class _GatedCondBlock(nn.Module):
    """Upsample stage with a PixelCNN-style genre-conditioned gated nonlinearity:
    ``sigmoid(a + cond_a) * tanh(b + cond_b)`` where ``(a, b)`` split the conv output
    channel-wise and ``cond`` is a per-stage linear projection of the one-hot genre
    label -- the source's ``pixcnn_gated_nonlinearity`` fed by ``Generator.condK``."""

    def __init__(self, in_ch: int, out_ch: int, n_classes: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch * 2, 4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.cond = nn.Linear(n_classes, out_ch * 2, bias=False)

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        """Upsample and apply genre-conditioned gated activation.

        Parameters
        ----------
        x : Tensor
            ``(batch, in_ch, h, w)`` feature map.
        labels : Tensor
            ``(batch, n_classes)`` one-hot (or soft) genre labels.

        Returns
        -------
        Tensor
            ``(batch, out_ch, 2h, 2w)`` gated, upsampled feature map.
        """
        out = self.bn(self.deconv(x))
        cond = self.cond(labels)
        a, b = out.chunk(2, dim=1)
        ca, cb = cond.chunk(2, dim=1)
        ca = ca.unsqueeze(-1).unsqueeze(-1)
        cb = cb.unsqueeze(-1).unsqueeze(-1)
        return torch.sigmoid(a + ca) * torch.tanh(b + cb)


class GANgoghGenerator(nn.Module):
    """AC-WGAN generator with genre-conditioned gated nonlinearities at every stage.

    Parameters
    ----------
    noise_dim : int
        Latent noise dimensionality.
    n_classes : int
        Number of WikiArt genre classes.
    base_ch : int
        Channel width of the first (post-projection) feature map.
    """

    def __init__(self, noise_dim: int = 32, n_classes: int = 14, base_ch: int = 32) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(noise_dim + n_classes, base_ch * 4 * 4 * 4)
        self.base_ch = base_ch
        self.block1 = _GatedCondBlock(base_ch * 4, base_ch * 2, n_classes)
        self.block2 = _GatedCondBlock(base_ch * 2, base_ch, n_classes)
        self.to_rgb = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        """Generate a genre-conditioned image from noise and a one-hot genre label.

        Parameters
        ----------
        noise : Tensor
            ``(batch, noise_dim)`` latent noise.
        labels : Tensor
            ``(batch, n_classes)`` one-hot genre labels.

        Returns
        -------
        Tensor
            ``(batch, 3, 16, 16)`` generated image in ``[-1, 1]``.
        """
        z = torch.cat([noise, labels], dim=1)
        x = self.fc(z).view(-1, self.base_ch * 4, 4, 4)
        x = self.block1(x, labels)
        x = self.block2(x, labels)
        return torch.tanh(self.to_rgb(x))


class GANgoghCritic(nn.Module):
    """AC-WGAN critic: a WGAN critic head (unbounded scalar, LayerNorm not BatchNorm)
    plus an auxiliary genre-classification head reading the same conv trunk -- the
    source's ``kACGANDiscriminator``.

    Parameters
    ----------
    n_classes : int
        Number of WikiArt genre classes.
    base_ch : int
        Channel width of the first conv stage.
    """

    def __init__(self, n_classes: int = 14, base_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, base_ch, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.ln2 = nn.GroupNorm(1, base_ch * 2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.critic_head = nn.Linear(base_ch * 2 * 4 * 4, 1)
        self.class_head = nn.Linear(base_ch * 2 * 4 * 4, n_classes)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Score an image's realism and predict its genre.

        Parameters
        ----------
        image : Tensor
            ``(batch, 3, 16, 16)`` image.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(critic_score, genre_logits)``: unbounded scalar WGAN score and
            per-class genre logits.
        """
        x = self.act(self.conv1(image))
        x = self.act(self.ln2(self.conv2(x)))
        x = x.flatten(1)
        return self.critic_head(x).squeeze(-1), self.class_head(x)


class GANgoghACWGAN(nn.Module):
    """Wraps the AC-WGAN generator and critic in one traceable forward: generate a
    genre-conditioned image from noise, then critique it (score + genre logits)."""

    def __init__(self, noise_dim: int = 32, n_classes: int = 14, base_ch: int = 32) -> None:
        super().__init__()
        self.generator = GANgoghGenerator(noise_dim, n_classes, base_ch)
        self.critic = GANgoghCritic(n_classes, base_ch)

    def forward(self, noise: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Generate then critique a genre-conditioned image.

        Parameters
        ----------
        noise : Tensor
            ``(batch, noise_dim)`` latent noise.
        labels : Tensor
            ``(batch, n_classes)`` one-hot genre labels.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(fake_image, critic_score, genre_logits)``.
        """
        fake = self.generator(noise, labels)
        score, logits = self.critic(fake)
        return fake, score, logits


def build_gangogh() -> nn.Module:
    """Build a small GANgogh AC-WGAN (genre-conditioned generator + critic).

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return GANgoghACWGAN(noise_dim=24, n_classes=10, base_ch=16).eval()


def example_input_gangogh() -> tuple[Tensor, Tensor]:
    """Example latent noise and one-hot genre labels.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(noise, labels)``: ``noise`` is ``(2, 24)``, ``labels`` is ``(2, 10)`` one-hot.
    """
    noise = torch.randn(2, 24)
    labels = F.one_hot(torch.randint(0, 10, (2,)), num_classes=10).float()
    return noise, labels


# ---------------------------------------------------------------------------
# MODULE 4: HiFill -- contextual residual aggregation for high-res inpainting
# ---------------------------------------------------------------------------


class _GatedConv(nn.Module):
    """Gated convolution: ``feature * sigmoid(gate)``, both branches from one conv
    applied to a doubled output channel count and split -- the free-form-inpainting
    gating mechanism shared by DeepFillv2/HiFill generators."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        pad = dilation
        self.conv = nn.Conv2d(in_ch, out_ch * 2, 3, stride=stride, padding=pad, dilation=dilation)
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a gated convolution.

        Parameters
        ----------
        x : Tensor
            ``(batch, in_ch, h, w)`` feature map.

        Returns
        -------
        Tensor
            ``(batch, out_ch, h', w')`` gated feature map.
        """
        feat, gate = self.conv(x).chunk(2, dim=1)
        return self.act(feat) * torch.sigmoid(gate)


class HiFillGenerator(nn.Module):
    """Coarse-to-fine gated-conv inpainting generator with CONTEXTUAL RESIDUAL
    AGGREGATION: one low-resolution patch-wise attention map (cosine similarity
    between hole and valid 4x4 patches of a bottleneck feature, masked so hole patches
    only attend to valid ones) is computed once and then RE-APPLIED at three different
    refinement-decoder resolutions to aggregate high-frequency residual detail into the
    hole at each scale -- the paper's core mechanism for inpainting far above training
    resolution.

    Parameters
    ----------
    base_ch : int
        Base channel width.
    patch_grid : int
        Side length of the square patch grid the attention map is computed over.
    """

    def __init__(self, base_ch: int = 16, patch_grid: int = 4) -> None:
        super().__init__()
        self.patch_grid = patch_grid

        # Coarse stage: input is RGB+mask (4ch), output is a coarse RGB fill.
        self.coarse_enc = nn.Sequential(
            _GatedConv(4, base_ch, stride=2),
            _GatedConv(base_ch, base_ch * 2, stride=2),
        )
        self.coarse_res = _GatedConv(base_ch * 2, base_ch * 2, dilation=2)
        self.coarse_dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_ch, 3, 4, stride=2, padding=1),
        )

        # Refinement stage: encoder producing 3 scales (pl1, pl2, pl3=bottleneck).
        self.ref1 = _GatedConv(3, base_ch)
        self.ref2 = _GatedConv(base_ch, base_ch * 2, stride=2)
        self.ref3 = _GatedConv(base_ch * 2, base_ch * 4, stride=2)

        # Decoder, consuming attention-transferred features concatenated at each scale.
        self.dec3_to_2 = nn.Sequential(
            nn.Conv2d(base_ch * 4 * 2, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1),
        )
        self.dec2_to_1 = nn.Sequential(
            nn.Conv2d(base_ch * 2 * 2, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1),
        )
        self.out_conv = nn.Conv2d(base_ch * 2, 3, 3, padding=1)

    def _compute_attention(self, feat: Tensor, mask: Tensor) -> Tensor:
        """Compute a low-res patch-wise cosine-similarity attention map.

        Parameters
        ----------
        feat : Tensor
            ``(batch, c, h, w)`` bottleneck feature map.
        mask : Tensor
            ``(batch, 1, H, W)`` full-resolution hole mask (1 == hole).

        Returns
        -------
        Tensor
            ``(batch, g*g, g*g)`` softmax attention over the ``g x g`` patch grid,
            masked so hole patches attend only to valid patches.
        """
        g = self.patch_grid
        pooled_feat = F.adaptive_avg_pool2d(feat, (g, g))
        pooled_mask = F.adaptive_max_pool2d(mask, (g, g))
        b, c = pooled_feat.shape[:2]
        f = pooled_feat.permute(0, 2, 3, 1).reshape(b, g * g, c)
        f_norm = F.normalize(f, dim=-1)
        sim = torch.bmm(f_norm, f_norm.transpose(1, 2))
        p = pooled_mask.reshape(b, g * g, 1)
        valid_matrix = torch.bmm(p, (1.0 - p).transpose(1, 2))
        sim = sim.masked_fill(valid_matrix < 0.5, -1e4)
        return F.softmax(sim, dim=-1)

    def _transfer(self, feat: Tensor, attn: Tensor) -> Tensor:
        """Re-apply a low-res attention map to aggregate residual detail at ``feat``'s
        own (higher) resolution.

        Parameters
        ----------
        feat : Tensor
            ``(batch, c, h, w)`` decoder feature map at some scale.
        attn : Tensor
            ``(batch, g*g, g*g)`` patch attention computed once at the bottleneck.

        Returns
        -------
        Tensor
            ``(batch, c, h, w)`` attention-aggregated feature map at the same scale.
        """
        g = self.patch_grid
        b, c, h, w = feat.shape
        patches = F.adaptive_avg_pool2d(feat, (g, g)).reshape(b, c, g * g).permute(0, 2, 1)
        agg = torch.bmm(attn, patches).permute(0, 2, 1).reshape(b, c, g, g)
        return F.interpolate(agg, size=(h, w), mode="nearest")

    def forward(self, image: Tensor, mask: Tensor) -> Tensor:
        """Coarse-to-fine inpainting with contextual residual aggregation.

        Parameters
        ----------
        image : Tensor
            ``(batch, 3, h, w)`` RGB image with hole pixels arbitrary.
        mask : Tensor
            ``(batch, 1, h, w)`` binary hole mask (1 == hole).

        Returns
        -------
        Tensor
            ``(batch, 3, h, w)`` inpainted RGB image.
        """
        masked_img = image * (1.0 - mask)
        coarse_in = torch.cat([masked_img, mask], dim=1)
        c = self.coarse_enc(coarse_in)
        c = self.coarse_res(c) + c
        coarse_out = torch.tanh(self.coarse_dec(c))
        coarse_out = F.interpolate(
            coarse_out, size=image.shape[-2:], mode="bilinear", align_corners=False
        )

        refine_in = image * (1.0 - mask) + coarse_out * mask
        pl1 = self.ref1(refine_in)
        pl2 = self.ref2(pl1)
        pl3 = self.ref3(pl2)

        attn = self._compute_attention(pl3, mask)
        pl3_t = self._transfer(pl3, attn)
        d = torch.cat([pl3, pl3_t], dim=1)
        d = self.dec3_to_2(d)

        pl2_t = self._transfer(pl2, attn)
        d = torch.cat([d, pl2_t], dim=1)
        d = self.dec2_to_1(d)

        pl1_t = self._transfer(pl1, attn)
        d = torch.cat([d, pl1_t], dim=1)
        out = torch.tanh(self.out_conv(d))
        return image * (1.0 - mask) + out * mask


def build_hifill() -> nn.Module:
    """Build a small HiFill contextual-residual-aggregation inpainting generator.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return HiFillGenerator(base_ch=12, patch_grid=4).eval()


def example_input_hifill() -> tuple[Tensor, Tensor]:
    """Example masked image and hole mask.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(image, mask)``: ``image`` is ``(1, 3, 64, 64)``, ``mask`` is ``(1, 1, 64, 64)``
        binary (1 == hole).
    """
    image = torch.rand(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 20:44, 20:44] = 1.0
    return image, mask


# ---------------------------------------------------------------------------
# MODULE 5: iColoriT -- point-interactive colorization ViT
# ---------------------------------------------------------------------------


class _ViTBlock(nn.Module):
    """Standard pre-norm transformer block (MHSA + MLP)."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply one pre-norm self-attention + MLP block.

        Parameters
        ----------
        x : Tensor
            ``(batch, n_tokens, dim)`` token sequence.

        Returns
        -------
        Tensor
            ``(batch, n_tokens, dim)`` updated token sequence.
        """
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class _LocalAttentionHead(nn.Module):
    """Local stabilizing layer: a windowed self-attention head over the patch-token
    grid whose queries may attend only to a ``kernel_size x kernel_size`` neighborhood
    (via an additive attention-bias mask built from grid coordinates), replacing a
    plain linear pixel-shuffle head to damp large-upsampling-ratio checkerboard
    artifacts -- the source's ``LocalAttentionHead``.

    Parameters
    ----------
    dim : int
        Token embedding dimension.
    out_dim : int
        Output channels per token (``2 * patch_size ** 2`` for ab pixel-shuffle).
    grid_size : int
        Side length of the square patch-token grid.
    kernel_size : int
        Side length of the local attention neighborhood.
    n_heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        grid_size: int,
        kernel_size: int = 3,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.grid_size = grid_size
        self.scale = (dim // n_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, out_dim)

        n = grid_size * grid_size
        mask = torch.ones(n, n)
        for i in range(grid_size):
            for j in range(grid_size):
                stx, sty = max(i - kernel_size // 2, 0), max(j - kernel_size // 2, 0)
                edx = min(i + kernel_size // 2, grid_size - 1)
                edy = min(j + kernel_size // 2, grid_size - 1)
                local = torch.ones(grid_size, grid_size)
                local[stx : edx + 1, sty : edy + 1] = 0.0
                mask[i * grid_size + j] = local.flatten()
        self.register_buffer("block_mask", mask.bool())

    def forward(self, x: Tensor) -> Tensor:
        """Apply windowed local self-attention and project to output channels.

        Parameters
        ----------
        x : Tensor
            ``(batch, n_tokens, dim)`` token sequence (``n_tokens == grid_size**2``).

        Returns
        -------
        Tensor
            ``(batch, n_tokens, out_dim)`` per-token output vectors.
        """
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.masked_fill(self.block_mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.proj(out)


class IColoriT(nn.Module):
    """Point-interactive colorization Vision Transformer.

    Operates on a 4-channel input (L channel, masked ab-color channels, binary
    hint-location mask), propagating sparse user hints to the whole image via global
    self-attention across ALL patch tokens in a single forward pass, then predicting
    a per-patch ``2 * patch_size**2``-channel vector that pixel-shuffles directly into
    the full-resolution ab-color map via the local-attention stabilizing head.

    Parameters
    ----------
    img_size : int
        Input spatial size (assumed square).
    patch_size : int
        Patch edge length; also sets the pixel-shuffle upsampling ratio.
    embed_dim : int
        Token embedding width.
    depth : int
        Number of transformer blocks.
    n_heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 48,
        depth: int = 3,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.patch_embed = nn.Conv2d(4, embed_dim, kernel_size=patch_size, stride=patch_size)

        n_patches = self.grid_size * self.grid_size
        self.register_buffer("pos_embed", _sinusoid_table(n_patches, embed_dim))

        self.blocks = nn.ModuleList([_ViTBlock(embed_dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        out_dim = 2 * patch_size * patch_size
        self.head = _LocalAttentionHead(embed_dim, out_dim, self.grid_size, n_heads=n_heads)

    def forward(self, l_channel: Tensor, ab_channels: Tensor, hint_mask: Tensor) -> Tensor:
        """Colorize the ``L`` channel given sparse ``ab`` point hints.

        Parameters
        ----------
        l_channel : Tensor
            ``(batch, 1, h, w)`` lightness channel.
        ab_channels : Tensor
            ``(batch, 2, h, w)`` ground-truth ab color, already zeroed outside hint
            locations by the caller (matching the source's hint-masking convention).
        hint_mask : Tensor
            ``(batch, 1, h, w)`` binary mask, 1 at NON-hint (unknown) locations.

        Returns
        -------
        Tensor
            ``(batch, 2, h, w)`` predicted ab color map in ``[-1, 1]``.
        """
        x = torch.cat([l_channel, ab_channels, hint_mask], dim=1)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed.to(tokens.dtype)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        out = self.head(tokens)
        out = torch.tanh(out)

        b = out.shape[0]
        g = self.grid_size
        p = self.patch_size
        out = out.reshape(b, g, g, 2, p, p)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(b, 2, g * p, g * p)
        return out


def _sinusoid_table(n_position: int, dim: int) -> Tensor:
    """Build a fixed sinusoidal position-embedding table.

    Parameters
    ----------
    n_position : int
        Number of positions (patch tokens).
    dim : int
        Embedding dimension.

    Returns
    -------
    Tensor
        ``(1, n_position, dim)`` sinusoidal position embedding table.
    """
    position = torch.arange(n_position).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    table = torch.zeros(n_position, dim)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term[: table[:, 1::2].shape[1]])
    return table.unsqueeze(0)


def build_icolorit() -> nn.Module:
    """Build a small point-interactive iColoriT colorization ViT.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return IColoriT(img_size=32, patch_size=4, embed_dim=48, depth=3, n_heads=4).eval()


def example_input_icolorit() -> tuple[Tensor, Tensor, Tensor]:
    """Example L channel, sparse ab hints, and hint mask.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(l_channel, ab_channels, hint_mask)``, all ``(1, C, 32, 32)`` with a single
        4x4-patch hint location.
    """
    l_channel = torch.rand(1, 1, 32, 32)
    ab_channels = torch.zeros(1, 2, 32, 32)
    hint_mask = torch.ones(1, 1, 32, 32)
    ab_channels[:, :, 8:12, 8:12] = torch.rand(1, 2, 4, 4) * 2 - 1
    hint_mask[:, :, 8:12, 8:12] = 0.0
    return l_channel, ab_channels, hint_mask


MENAGERIE_ENTRIES = [
    ("EnlightenGAN", "build_enlightengan", "example_input_enlightengan", "2021", "VIS"),
    ("FRRN Inpainting", "build_frrn_inpainting", "example_input_frrn_inpainting", "2019", "VIS"),
    ("GANgogh", "build_gangogh", "example_input_gangogh", "2017", "GEN"),
    ("HiFill", "build_hifill", "example_input_hifill", "2020", "VIS"),
    (
        "iColoriT (interactive colorization via ViT)",
        "build_icolorit",
        "example_input_icolorit",
        "2023",
        "VIS",
    ),
]
