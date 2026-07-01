"""Wave 9 batch 21 menagerie classics: remote-sensing change-detection,
wildfire/flood forecasting, and geography-aware self-supervised pretraining.

Sources checked (repo_url / desc_source columns of the build queue, web
research 2026-07-01; no cloning, no pip installs beyond the base env):

  - FC-EF: https://github.com/rcdaudt/fully_convolutional_change_detection
    (``unet.py``) ; Daudt, Le Saux & Boulch, ICIP 2018, "Fully Convolutional
    Siamese Networks for Change Detection". Confirmed from the official repo:
    FC-EF ("early fusion") is a plain U-Net whose *input* is the channel-wise
    concatenation of the bi-temporal image pair -- a single shared encoder
    sees both dates fused before any convolution, with skip connections from
    each of 4 encoder stages to the matching decoder stage (via
    ``ReplicationPad2d``-aligned concatenation) and a final log-softmax head.
    Reimplemented here compactly with the same fuse-then-U-Net topology (4
    down/up stages, skip concatenation, log-softmax output) at reduced width.

  - FC-Siam change nets: same repo, ``siamunet_diff.py`` / ``siamunet_conc.py``
    ; same paper. Confirmed from the official repo: unlike FC-EF, the two
    dates are each pushed independently through a *weight-shared* Siamese
    U-Net encoder (both branches literally call the same ``nn.Conv2d``/
    ``nn.BatchNorm2d`` modules), and the two encoder feature streams are
    fused into the decoder skip connections either by absolute difference
    (FC-Siam-diff: ``torch.abs(feat_1 - feat_2)``) or by channel
    concatenation (FC-Siam-conc: ``torch.cat([feat_1, feat_2], dim=1)``) at
    every one of the 4 stages, with a single shared decoder. Reimplemented
    here as one ``FCSiamUNet`` module parameterised by a ``fusion`` mode
    ("diff" / "conc") so both official variants are built from one class,
    matching the repo's own code-sharing between ``siamunet_diff.py`` and
    ``siamunet_conc.py`` (they differ only in the skip-fusion op and the
    resulting decoder input width).

  - FireCast: https://github.com/brian-xu/wildfire-prediction
    (``train/firecast.py``) ; Radke et al., IJWF 2019, "FireCast: An
    Intelligent Wildfire Forecasting System Applying Convolutional Neural
    Networks". Confirmed from the repo (community reimplementation of the
    official Radke et al. architecture, medium confidence per the build
    queue but architecturally explicit and unambiguous): a dual-branch model
    -- a small terrain CNN (2 conv+pool blocks with Sigmoid/ReLU activations
    over a ``terrain_features``-channel local grid, flattened to a dense
    layer) is concatenated with a raw weather-feature vector and passed
    through a final linear+sigmoid head that predicts fire-spread
    probability for the center cell. Reimplemented here with the same
    dual-branch-then-concat-then-sigmoid topology at reduced channel widths.

  - FloodCast: https://github.com/HydroPML/FloodCast ; Yan et al., Water
    Research 2024, doi:10.1016/j.watres.2024.122162. This candidate is a
    duplicate of ``cand_01364`` "DeepFlood", which maps to the same
    HydroPML/FloodCast repository and has already been built as
    ``build_geopins_floodcast`` / ``GeoPINSFloodModel`` in
    ``menagerie/classics/gen_w9a18.py`` (registered under the canonical name
    "DeepFlood", capturing the paper's actual novelty: a geometry-adaptive,
    physics-informed spectral flood solver). SKIPPED here as
    already_in_catalog -- see the build-queue's own POTENTIAL_DEDUP note.

  - GASSL (Geography-Aware Self-Supervised Learning):
    https://github.com/sustainlab-group/geography-aware-ssl
    (``moco_fmow/moco/builder_geo.py``, ``moco_fmow/main_moco_geo+tp.py``) ;
    Ayush et al., ICCV 2021, arXiv:2011.09980. Confirmed from the official
    repo: a standard MoCo-v2 query/key encoder pair (momentum-updated key
    encoder, normalized queue of negative keys, InfoNCE contrastive loss)
    extended with a ``classifier`` head (``nn.Linear(dim, num_geo_classes)``
    in ``main_moco_geo+tp.py``) applied to the *query* embedding and trained
    jointly, with cross-entropy against a geographic *cell* label (the
    "geo-aware" contribution: predicting which of N discretized geographic
    regions an image was captured in, using the same embedding the
    contrastive loss trains). Reimplemented here compactly with a small CNN
    backbone in place of ResNet-50, keeping the momentum key encoder + queue
    + geo-classification head as the distinctive combination.

  - GeoKR: repo URL in the build queue
    (flyakon/Geographical-Knowledge-driven-Representation-Learning) 404s;
    the actual official repo is
    https://github.com/flyakon/Geographical-Knowledge-driven-Representaion-Learning
    (note the repo owner's typo, "Representaion") ; Li et al., IEEE TGRS
    2021, arXiv:2107.05276. Confirmed from
    ``GeoKR/models/representation/{representation_net,mean_teacher_net}.py``
    and ``GeoKR/losses/mean_teacher_losses.py``: a *mean-teacher* pair of
    identical backbone+classification-head networks (student and teacher,
    same architecture, teacher weights are an EMA of the student's:
    ``w_t = alpha * w_t + (1 - alpha) * w_s``) where the classification head
    predicts *land-cover class* probabilities (the geographical-knowledge
    supervision signal -- pseudo-labels derived from a global land-cover
    product plus each image's lat/lon) from pooled backbone features, and
    training combines a soft-label classification loss on the student with a
    student/teacher consistency (KL-style) loss on the class distributions.
    Reimplemented here compactly with a small shared-architecture CNN
    backbone, keeping the mean-teacher EMA-consistency + land-cover
    classification-head mechanism that is GeoKR's actual novelty.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# 1. FC-EF: early-fusion fully convolutional Siamese U-Net (Daudt 2018).
# ---------------------------------------------------------------------------


class _ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> Dropout2d block used throughout."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(p=0.2)

    def forward(self, x: Tensor) -> Tensor:
        """Apply conv, batchnorm, ReLU, and dropout in sequence."""
        return self.drop(F.relu(self.bn(self.conv(x))))


class FCEFUNet(nn.Module):
    """FC-EF: early-fusion change-detection U-Net (Daudt et al., ICIP 2018).

    The bi-temporal image pair is concatenated channel-wise *before* the
    encoder, so a single ordinary U-Net (4 down-stages, 4 up-stages with
    skip connections) sees the fused pair and predicts a per-pixel
    change/no-change log-probability map.
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        widths = (16, 32, 64, 128)
        self.enc1 = nn.ModuleList(
            [_ConvBNReLU(2 * in_channels, widths[0]), _ConvBNReLU(widths[0], widths[0])]
        )
        self.enc2 = nn.ModuleList(
            [_ConvBNReLU(widths[0], widths[1]), _ConvBNReLU(widths[1], widths[1])]
        )
        self.enc3 = nn.ModuleList(
            [_ConvBNReLU(widths[1], widths[2]), _ConvBNReLU(widths[2], widths[2])]
        )
        self.enc4 = nn.ModuleList(
            [_ConvBNReLU(widths[2], widths[3]), _ConvBNReLU(widths[3], widths[3])]
        )

        self.up4 = nn.ConvTranspose2d(
            widths[3], widths[3], 3, padding=1, stride=2, output_padding=1
        )
        self.dec4 = _ConvBNReLU(2 * widths[3], widths[2])
        self.up3 = nn.ConvTranspose2d(
            widths[2], widths[2], 3, padding=1, stride=2, output_padding=1
        )
        self.dec3 = _ConvBNReLU(2 * widths[2], widths[1])
        self.up2 = nn.ConvTranspose2d(
            widths[1], widths[1], 3, padding=1, stride=2, output_padding=1
        )
        self.dec2 = _ConvBNReLU(2 * widths[1], widths[0])
        self.up1 = nn.ConvTranspose2d(
            widths[0], widths[0], 3, padding=1, stride=2, output_padding=1
        )
        self.dec1 = nn.Conv2d(2 * widths[0], num_classes, kernel_size=3, padding=1)

    @staticmethod
    def _run_stage(blocks: nn.ModuleList, x: Tensor) -> Tensor:
        for block in blocks:
            x = block(x)
        return x

    def forward(self, image_t1: Tensor, image_t2: Tensor) -> Tensor:
        """Predict a change-map log-probability from an image pair.

        Parameters
        ----------
        image_t1 : torch.Tensor
            Earlier-date image, shape ``(batch, in_channels, height, width)``.
        image_t2 : torch.Tensor
            Later-date image, same shape as ``image_t1``.

        Returns
        -------
        torch.Tensor
            Per-pixel log-softmax class scores, shape
            ``(batch, num_classes, height, width)``.
        """
        x = torch.cat([image_t1, image_t2], dim=1)

        f1 = self._run_stage(self.enc1, x)
        p1 = F.max_pool2d(f1, 2, 2)
        f2 = self._run_stage(self.enc2, p1)
        p2 = F.max_pool2d(f2, 2, 2)
        f3 = self._run_stage(self.enc3, p2)
        p3 = F.max_pool2d(f3, 2, 2)
        f4 = self._run_stage(self.enc4, p3)
        p4 = F.max_pool2d(f4, 2, 2)

        d4 = self.up4(p4)
        d4 = self.dec4(torch.cat([d4, f4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, f3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, f2], dim=1))
        d1 = self.up1(d2)
        out = self.dec1(torch.cat([d1, f1], dim=1))
        return F.log_softmax(out, dim=1)


def build_fc_ef() -> FCEFUNet:
    """Build a compact FC-EF early-fusion change-detection U-Net.

    Returns
    -------
    FCEFUNet
        Random-initialized FC-EF model in eval mode.
    """
    return FCEFUNet(in_channels=4, num_classes=2).eval()


def example_input_fc_ef() -> tuple[Tensor, Tensor]:
    """Create an example bi-temporal image pair for FC-EF.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two ``(2, 4, 64, 64)`` images (earlier and later acquisition dates).
    """
    torch.manual_seed(0)
    image_t1 = torch.randn(2, 4, 64, 64)
    image_t2 = torch.randn(2, 4, 64, 64)
    return image_t1, image_t2


# ---------------------------------------------------------------------------
# 2. FC-Siam-diff / FC-Siam-conc: weight-shared Siamese change-detection
#    U-Nets (Daudt 2018).
# ---------------------------------------------------------------------------


class FCSiamUNet(nn.Module):
    """FC-Siam-diff / FC-Siam-conc: Siamese change-detection U-Net.

    A single weight-shared encoder is applied independently to each of the
    two bi-temporal images. At every one of the 4 encoder stages the two
    resulting feature maps are fused into the corresponding decoder skip
    connection either by absolute difference (``fusion="diff"``,
    FC-Siam-diff) or by channel concatenation (``fusion="conc"``,
    FC-Siam-conc); a single shared decoder then produces the change map.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, fusion: str = "diff") -> None:
        super().__init__()
        if fusion not in ("diff", "conc"):
            raise ValueError(f"fusion must be 'diff' or 'conc', got {fusion!r}")
        self.fusion = fusion
        widths = (16, 32, 64, 128)

        # Weight-shared Siamese encoder (same modules used for both dates).
        self.enc1 = nn.ModuleList(
            [_ConvBNReLU(in_channels, widths[0]), _ConvBNReLU(widths[0], widths[0])]
        )
        self.enc2 = nn.ModuleList(
            [_ConvBNReLU(widths[0], widths[1]), _ConvBNReLU(widths[1], widths[1])]
        )
        self.enc3 = nn.ModuleList(
            [_ConvBNReLU(widths[1], widths[2]), _ConvBNReLU(widths[2], widths[2])]
        )
        self.enc4 = nn.ModuleList(
            [_ConvBNReLU(widths[2], widths[3]), _ConvBNReLU(widths[3], widths[3])]
        )

        # Decoder skip-input width depends on the fusion mode: "diff"
        # collapses the pair to one feature map (same width as the encoder
        # stage); "conc" keeps both (double width).
        mult = 1 if fusion == "diff" else 2

        self.up4 = nn.ConvTranspose2d(
            widths[3], widths[3], 3, padding=1, stride=2, output_padding=1
        )
        self.dec4 = _ConvBNReLU(widths[3] + mult * widths[3], widths[2])
        self.up3 = nn.ConvTranspose2d(
            widths[2], widths[2], 3, padding=1, stride=2, output_padding=1
        )
        self.dec3 = _ConvBNReLU(widths[2] + mult * widths[2], widths[1])
        self.up2 = nn.ConvTranspose2d(
            widths[1], widths[1], 3, padding=1, stride=2, output_padding=1
        )
        self.dec2 = _ConvBNReLU(widths[1] + mult * widths[1], widths[0])
        self.up1 = nn.ConvTranspose2d(
            widths[0], widths[0], 3, padding=1, stride=2, output_padding=1
        )
        self.dec1 = nn.Conv2d(widths[0] + mult * widths[0], num_classes, kernel_size=3, padding=1)

    @staticmethod
    def _run_stage(blocks: nn.ModuleList, x: Tensor) -> Tensor:
        for block in blocks:
            x = block(x)
        return x

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        f1 = self._run_stage(self.enc1, x)
        p1 = F.max_pool2d(f1, 2, 2)
        f2 = self._run_stage(self.enc2, p1)
        p2 = F.max_pool2d(f2, 2, 2)
        f3 = self._run_stage(self.enc3, p2)
        p3 = F.max_pool2d(f3, 2, 2)
        f4 = self._run_stage(self.enc4, p3)
        p4 = F.max_pool2d(f4, 2, 2)
        return f1, f2, f3, f4, p4

    def _fuse(self, feat_a: Tensor, feat_b: Tensor) -> Tensor:
        if self.fusion == "diff":
            return torch.abs(feat_a - feat_b)
        return torch.cat([feat_a, feat_b], dim=1)

    def forward(self, image_t1: Tensor, image_t2: Tensor) -> Tensor:
        """Predict a change-map log-probability from an image pair.

        Parameters
        ----------
        image_t1 : torch.Tensor
            Earlier-date image, shape ``(batch, in_channels, height, width)``.
        image_t2 : torch.Tensor
            Later-date image, same shape as ``image_t1``.

        Returns
        -------
        torch.Tensor
            Per-pixel log-softmax class scores, shape
            ``(batch, num_classes, height, width)``.
        """
        f1_a, f2_a, f3_a, f4_a, p4_a = self._encode(image_t1)
        f1_b, f2_b, f3_b, f4_b, p4_b = self._encode(image_t2)

        d4 = self.up4(p4_a)
        d4 = self.dec4(torch.cat([d4, self._fuse(f4_a, f4_b)], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self._fuse(f3_a, f3_b)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self._fuse(f2_a, f2_b)], dim=1))
        d1 = self.up1(d2)
        out = self.dec1(torch.cat([d1, self._fuse(f1_a, f1_b)], dim=1))
        return F.log_softmax(out, dim=1)


def build_fc_siam() -> FCSiamUNet:
    """Build a compact FC-Siam-diff Siamese change-detection U-Net.

    Returns
    -------
    FCSiamUNet
        Random-initialized FC-Siam-diff model in eval mode.
    """
    return FCSiamUNet(in_channels=3, num_classes=2, fusion="diff").eval()


def example_input_fc_siam() -> tuple[Tensor, Tensor]:
    """Create an example bi-temporal image pair for FC-Siam-diff.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two ``(2, 3, 64, 64)`` images (earlier and later acquisition dates).
    """
    torch.manual_seed(0)
    image_t1 = torch.randn(2, 3, 64, 64)
    image_t2 = torch.randn(2, 3, 64, 64)
    return image_t1, image_t2


# ---------------------------------------------------------------------------
# 3. FireCast: dual-branch terrain CNN + weather MLP wildfire spread model
#    (Radke et al., IJWF 2019).
# ---------------------------------------------------------------------------


class FireCast(nn.Module):
    """FireCast wildfire spread predictor (Radke et al., IJWF 2019).

    A small terrain CNN consumes a local grid of static/dynamic terrain
    layers, is flattened and projected to a dense embedding; that embedding
    is concatenated with a raw per-cell weather-feature vector and passed
    through a final linear+sigmoid head predicting the probability that the
    center cell will be on fire.
    """

    def __init__(
        self, area: int = 15, terrain_features: int = 6, weather_features: int = 5
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(terrain_features, 32, kernel_size=2, stride=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(),
        )

        def _conv2d_result(size: int, kernel_size: int, stride: int = 1) -> int:
            return (size - kernel_size) // stride + 1

        conv_size = _conv2d_result(area, 2)
        conv_size = _conv2d_result(conv_size, 2, 2)
        conv_size = _conv2d_result(conv_size, 2)
        conv_size = _conv2d_result(conv_size, 2, 2)
        self.linear_input_size = (conv_size**2) * 64

        self.dense1 = nn.Linear(self.linear_input_size, 128)
        self.dense2 = nn.Linear(128 + weather_features, 1)

    def forward(self, terrain: Tensor, weather: Tensor) -> Tensor:
        """Predict fire-spread probability from terrain and weather inputs.

        Parameters
        ----------
        terrain : torch.Tensor
            Local terrain feature grid, shape
            ``(batch, terrain_features, area, area)``.
        weather : torch.Tensor
            Per-cell weather feature vector, shape
            ``(batch, weather_features)``.

        Returns
        -------
        torch.Tensor
            Fire-spread probability, shape ``(batch, 1)``.
        """
        x1 = self.conv1(terrain)
        x1 = self.conv2(x1)
        x1 = x1.reshape(-1, self.linear_input_size)
        x1 = F.relu(self.dense1(x1))
        x = torch.cat([x1, weather], dim=1)
        return torch.sigmoid(self.dense2(x))


def build_firecast() -> FireCast:
    """Build a compact FireCast dual-branch wildfire spread model.

    Returns
    -------
    FireCast
        Random-initialized FireCast model in eval mode.
    """
    return FireCast(area=15, terrain_features=6, weather_features=5).eval()


def example_input_firecast() -> tuple[Tensor, Tensor]:
    """Create example terrain-grid and weather-vector inputs for FireCast.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A ``(2, 6, 15, 15)`` terrain grid and a ``(2, 5)`` weather vector.
    """
    torch.manual_seed(0)
    terrain = torch.randn(2, 6, 15, 15)
    weather = torch.randn(2, 5)
    return terrain, weather


# ---------------------------------------------------------------------------
# 4. GASSL: MoCo-v2 with a geography-aware classification head
#    (Ayush et al., ICCV 2021).
# ---------------------------------------------------------------------------


class _SmallEncoder(nn.Module):
    """Compact CNN encoder standing in for GASSL's ResNet-50 backbone."""

    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode an image batch into an embedding vector."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class GASSLMoCo(nn.Module):
    """Geography-Aware SSL: MoCo-v2 plus a geo-location classification head.

    A momentum-updated key encoder maintains a queue of negative embeddings
    for a standard InfoNCE contrastive loss between query and key crops of
    the same image (as in MoCo-v2). The geography-aware contribution is a
    linear classification head applied to the *query* embedding that
    predicts a discretized geographic cell label, trained jointly with the
    contrastive objective so the representation is pulled toward
    geographically discriminative structure.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        queue_size: int = 64,
        momentum: float = 0.999,
        temperature: float = 0.07,
        num_geo_classes: int = 10,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        self.encoder_q = _SmallEncoder(out_dim=embed_dim)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        self.geo_classifier = nn.Linear(embed_dim, num_geo_classes)

        queue = F.normalize(torch.randn(embed_dim, queue_size), dim=0)
        self.register_buffer("queue", queue)

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    def forward(self, image_q: Tensor, image_k: Tensor) -> tuple[Tensor, Tensor]:
        """Compute MoCo contrastive logits and geo-location class logits.

        Parameters
        ----------
        image_q : torch.Tensor
            Query-view image crop, shape ``(batch, 3, height, width)``.
        image_k : torch.Tensor
            Key-view image crop of the same underlying image, same shape as
            ``image_q``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(contrastive_logits, geo_logits)``: MoCo logits over
            ``1 + queue_size`` classes (positive key first, then the queue),
            and geo-location classification logits from the query embedding.
        """
        q = F.normalize(self.encoder_q(image_q), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(image_k), dim=1)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        contrastive_logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        geo_logits = self.geo_classifier(q)
        return contrastive_logits, geo_logits


def build_gassl() -> GASSLMoCo:
    """Build a compact GASSL geography-aware MoCo model.

    Returns
    -------
    GASSLMoCo
        Random-initialized GASSL model in eval mode.
    """
    return GASSLMoCo(embed_dim=32, queue_size=64, num_geo_classes=10).eval()


def example_input_gassl() -> tuple[Tensor, Tensor]:
    """Create an example query/key image-crop pair for GASSL.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two ``(2, 3, 64, 64)`` image crops (query view and key view).
    """
    torch.manual_seed(0)
    image_q = torch.randn(2, 3, 64, 64)
    image_k = torch.randn(2, 3, 64, 64)
    return image_q, image_k


# ---------------------------------------------------------------------------
# 5. GeoKR: mean-teacher land-cover-supervised representation learning
#    (Li et al., IEEE TGRS 2021).
# ---------------------------------------------------------------------------


class _GeoKRBackboneHead(nn.Module):
    """Small backbone + land-cover classification head (student or teacher)."""

    def __init__(
        self, in_channels: int = 3, hidden_channels: int = 32, num_land_cover: int = 8
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.class_fc = nn.Linear(hidden_channels, num_land_cover)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return land-cover class logits and softmax probabilities."""
        feat = self.backbone(x)
        feat = self.avg_pool(feat)
        feat = torch.flatten(feat, 1)
        logits = self.class_fc(feat)
        prob = F.softmax(logits, dim=1)
        return logits, prob


class GeoKRMeanTeacher(nn.Module):
    """GeoKR: mean-teacher pretraining with land-cover pseudo-labels.

    A student and a teacher network share the same backbone+classification-
    head architecture. The teacher's weights are an exponential moving
    average of the student's (``w_t = alpha * w_t + (1 - alpha) * w_s``,
    applied here once per forward call to keep the mechanism traceable).
    The classification head predicts land-cover class probabilities -- the
    geographical-knowledge supervision signal derived from a global
    land-cover product and each image's location -- and training combines a
    soft-label classification loss on the student with a student/teacher
    consistency loss on the class distributions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        num_land_cover: int = 8,
        teacher_momentum: float = 0.95,
    ) -> None:
        super().__init__()
        self.teacher_momentum = teacher_momentum
        self.student_model = _GeoKRBackboneHead(in_channels, hidden_channels, num_land_cover)
        self.teacher_model = _GeoKRBackboneHead(in_channels, hidden_channels, num_land_cover)
        for param_t, param_s in zip(
            self.teacher_model.parameters(), self.student_model.parameters()
        ):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def _update_teacher(self) -> None:
        alpha = self.teacher_momentum
        for param_t, param_s in zip(
            self.teacher_model.parameters(), self.student_model.parameters()
        ):
            param_t.data = alpha * param_t.data + (1.0 - alpha) * param_s.data

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the student and EMA-updated teacher on the same image.

        Parameters
        ----------
        image : torch.Tensor
            Remote-sensing image batch, shape
            ``(batch, in_channels, height, width)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``(logits_student, prob_student, logits_teacher, prob_teacher)``
            land-cover class logits/probabilities from each network.
        """
        logits_s, prob_s = self.student_model(image)
        self._update_teacher()
        with torch.no_grad():
            logits_t, prob_t = self.teacher_model(image)
        return logits_s, prob_s, logits_t, prob_t


def build_geokr() -> GeoKRMeanTeacher:
    """Build a compact GeoKR mean-teacher representation-learning model.

    Returns
    -------
    GeoKRMeanTeacher
        Random-initialized GeoKR model in eval mode.
    """
    return GeoKRMeanTeacher(in_channels=3, hidden_channels=32, num_land_cover=8).eval()


def example_input_geokr() -> Tensor:
    """Create an example remote-sensing image batch for GeoKR.

    Returns
    -------
    torch.Tensor
        A ``(2, 3, 64, 64)`` image batch.
    """
    torch.manual_seed(0)
    return torch.randn(2, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("FC-EF", "build_fc_ef", "example_input_fc_ef", "2018", "VIS"),
    ("FC-Siam change nets", "build_fc_siam", "example_input_fc_siam", "2018", "VIS"),
    ("FireCast", "build_firecast", "example_input_firecast", "2019", "GEO"),
    ("GASSL", "build_gassl", "example_input_gassl", "2021", "VIS"),
    ("GeoKR", "build_geokr", "example_input_geokr", "2021", "GEO"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")
