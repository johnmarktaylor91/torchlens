"""Dependency-free OpenMMLab/Paddle-style compact classics.

This module reconstructs install-hostile catalog entries as small random-init
``torch.nn.Module`` networks while keeping each family's distinctive primitive:
BLIP/BLIP-2 vision-language fusion, BMN/BSN/DRN temporal proposal heads,
HRNet-style bottom-up pose estimators, C2D/C3D/CSN video backbones, CAIN frame
interpolation, Cascade/Mask/Faster/Retina detector heads, CenterPoint-style BEV
dense heads, rotated detectors, DBNet/CRNN OCR, CycleGAN/DCGAN/DeepFill/Deblur
generators, DGCNN/Cylinder3D point-cloud operators, and diffusion/control U-Nets.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from menagerie.classics.dbnet_mobilenetv3 import (
    build_dbnet_mobilenetv3,
    build_dbnetpp_resnet50,
    example_det_input,
)
from menagerie.classics.dgcnn import build as build_dgcnn_base
from menagerie.classics.dgcnn import example_input as example_dgcnn_input
from menagerie.classics.paddledet_cascade_rcnn import PaddleDetCascadeRCNN
from menagerie.classics.paddledet_faster_rcnn import FasterRCNN
from menagerie.classics.paddledet_retinanet import RetinaNet
from menagerie.classics.ri_vision_b_gan import (
    build_cyclegan_resnet_generator,
    build_dcgan_generator,
    build_deepfillv2_gated_generator,
    example_input_image,
    example_input_inpaint,
    example_input_noise,
)
from menagerie.classics.ri_vision_b_video import build_c3d, example_input_clip
from menagerie.classics.reimpl5_2_compact import build_crnn, example_text_image
from menagerie.classics.stylegan2 import (
    build_stylegan2_generator,
    example_input_stylegan2_generator,
)


class ConvBNAct(nn.Module):
    """Convolution, batch-normalization, and ReLU block."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1
    ) -> None:
        """Initialize a compact convolution block.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        stride:
            Spatial or temporal stride.
        groups:
            Group count for grouped/depthwise convolution.
        """

        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Activated feature map.
        """

        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Two-layer residual block."""

    def __init__(self, channels: int) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.conv1 = ConvBNAct(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual update.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Residual output.
        """

        return F.relu(x + self.conv2(self.conv1(x)))


class TinyVisionTransformer(nn.Module):
    """Small ViT-style image encoder with patch embedding and class token."""

    def __init__(self, dim: int = 48, patches: int = 4) -> None:
        """Initialize the image transformer.

        Parameters
        ----------
        dim:
            Embedding width.
        patches:
            Patch size.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, dim, patches, stride=patches)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, 65, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode an image as patch tokens.

        Parameters
        ----------
        image:
            RGB image batch.

        Returns
        -------
        torch.Tensor
            Sequence including the class token.
        """

        patches = self.patch(image).flatten(2).transpose(1, 2)
        cls = self.cls.expand(image.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1)
        return self.encoder(tokens + self.pos[:, : tokens.shape[1]])


class TextTransformer(nn.Module):
    """Small text transformer encoder."""

    def __init__(self, vocab: int = 512, dim: int = 48) -> None:
        """Initialize token embeddings and transformer layers.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Hidden width.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 16, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Encode token ids.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Text sequence features.
        """

        return self.encoder(self.embed(ids) + self.pos[:, : ids.shape[1]])


class BLIPCompact(nn.Module):
    """BLIP image-text model with ViT encoder, text encoder, and caption decoder."""

    def __init__(self) -> None:
        """Initialize BLIP-style modules."""

        super().__init__()
        self.vision = TinyVisionTransformer()
        self.text = TextTransformer()
        decoder = nn.TransformerDecoderLayer(48, 4, 96, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder, 1)
        self.itm = nn.Linear(96, 2)
        self.lm = nn.Linear(48, 512)

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run image-text matching and caption-token decoding.

        Parameters
        ----------
        batch:
            Placeholder tensor controlling batch size.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Image-text logits and caption logits.
        """

        bsz = batch.shape[0]
        image = torch.randn(bsz, 3, 32, 32, device=batch.device, dtype=batch.dtype)
        ids = torch.arange(12, device=batch.device).unsqueeze(0).expand(bsz, -1)
        vision = self.vision(image)
        text = self.text(ids)
        match = self.itm(torch.cat((vision[:, 0], text[:, 0]), dim=-1))
        decoded = self.decoder(text, vision)
        return match, self.lm(decoded)


class BLIP2Compact(nn.Module):
    """BLIP-2 with frozen-style ViT tokens, Q-Former queries, and LM projection."""

    def __init__(self) -> None:
        """Initialize the BLIP-2 Q-Former stack."""

        super().__init__()
        self.vision = TinyVisionTransformer()
        self.query = nn.Parameter(torch.randn(1, 8, 48) * 0.02)
        layer = nn.TransformerDecoderLayer(48, 4, 96, batch_first=True)
        self.qformer = nn.TransformerDecoder(layer, 2)
        self.lm_proj = nn.Linear(48, 64)
        self.lm = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Linear(64, 512)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Generate compact language logits from visual query tokens.

        Parameters
        ----------
        batch:
            Placeholder tensor controlling batch size.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        image = torch.randn(batch.shape[0], 3, 32, 32, device=batch.device, dtype=batch.dtype)
        vision = self.vision(image)
        query = self.query.expand(batch.shape[0], -1, -1)
        qtokens = self.qformer(query, vision)
        out, _ = self.lm(self.lm_proj(qtokens))
        return self.head(out)


class ChineseCLIPCompact(nn.Module):
    """Chinese-CLIP dual encoder with contrastive image/text projections."""

    def __init__(self) -> None:
        """Initialize dual encoders and contrastive scale."""

        super().__init__()
        self.vision = TinyVisionTransformer()
        self.text = TextTransformer(vocab=1024)
        self.image_proj = nn.Linear(48, 32)
        self.text_proj = nn.Linear(48, 32)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute image-text similarity logits.

        Parameters
        ----------
        batch:
            Placeholder tensor controlling batch size.

        Returns
        -------
        torch.Tensor
            Contrastive similarity matrix.
        """

        image = torch.randn(batch.shape[0], 3, 32, 32, device=batch.device, dtype=batch.dtype)
        ids = torch.arange(12, device=batch.device).unsqueeze(0).expand(batch.shape[0], -1)
        image_feat = F.normalize(self.image_proj(self.vision(image)[:, 0]), dim=-1)
        text_feat = F.normalize(self.text_proj(self.text(ids)[:, 0]), dim=-1)
        return self.logit_scale.exp() * image_feat @ text_feat.T


class CLIP4ClipCompact(nn.Module):
    """CLIP4Clip video-text retrieval with framewise CLIP pooling."""

    def __init__(self) -> None:
        """Initialize frame encoder, text encoder, and temporal transformer."""

        super().__init__()
        self.frame = TinyVisionTransformer()
        self.text = TextTransformer()
        layer = nn.TransformerEncoderLayer(48, 4, 96, batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, 1)
        self.proj = nn.Linear(48, 32)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Compute video-text retrieval logits.

        Parameters
        ----------
        video:
            Video clip shaped ``(batch, channels, time, height, width)``.

        Returns
        -------
        torch.Tensor
            Similarity matrix.
        """

        bsz, channels, time, height, width = video.shape
        frames = video.transpose(1, 2).reshape(bsz * time, channels, height, width)
        frame_tokens = self.frame(frames)[:, 0].reshape(bsz, time, -1)
        video_feat = F.normalize(self.proj(self.temporal(frame_tokens).mean(dim=1)), dim=-1)
        ids = torch.arange(12, device=video.device).unsqueeze(0).expand(bsz, -1)
        text_feat = F.normalize(self.proj(self.text(ids)[:, 0]), dim=-1)
        return video_feat @ text_feat.T


class TemporalProposalNet(nn.Module):
    """BMN/BSN-style temporal proposal network."""

    def __init__(self, mode: str = "bmn", dim: int = 32) -> None:
        """Initialize temporal convolution and proposal heads.

        Parameters
        ----------
        mode:
            Proposal family: ``"bmn"``, ``"bsn"``, or ``"drn"``.
        dim:
            Hidden width.
        """

        super().__init__()
        self.mode = mode
        self.base = nn.Sequential(
            nn.Conv1d(16, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(),
        )
        self.start = nn.Conv1d(dim, 1, 1)
        self.end = nn.Conv1d(dim, 1, 1)
        self.conf = nn.Conv2d(dim, 2, 1)
        self.reg = nn.Conv1d(dim, 2, 1)
        self.refine = nn.GRU(dim, dim, batch_first=True)
        self.query_embed = nn.Parameter(torch.randn(1, 4, dim) * 0.02)
        self.query_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.distance = nn.Linear(dim, 2)
        self.iou = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict temporal boundaries, confidence maps, and refinements.

        Parameters
        ----------
        x:
            Temporal feature sequence ``(batch, channels, time)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Proposal tensors.
        """

        feat = self.base(x)
        start = torch.sigmoid(self.start(feat))
        end = torch.sigmoid(self.end(feat))
        pair = feat.unsqueeze(-1) * feat.unsqueeze(-2)
        out: dict[str, torch.Tensor] = {"start": start, "end": end, "confidence": self.conf(pair)}
        if self.mode in {"bsn", "drn"}:
            out["tem_reg"] = self.reg(feat)
        if self.mode == "drn":
            seq = feat.transpose(1, 2)
            query = self.query_embed.expand(x.shape[0], -1, -1)
            conditioned, _ = self.query_attn(seq, query, query)
            refined, _ = self.refine(seq + conditioned)
            out["query_conditioned_distance"] = self.distance(refined)
            out["iou"] = torch.sigmoid(self.iou(refined))
        return out


class HRBlock(nn.Module):
    """Two-resolution HRNet block with repeated cross-scale fusion."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize HRNet branches.

        Parameters
        ----------
        channels:
            High-resolution branch width.
        """

        super().__init__()
        self.high = ResBlock(channels)
        self.low = ResBlock(channels * 2)
        self.down = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.up = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse high- and low-resolution branches.

        Parameters
        ----------
        high:
            High-resolution branch.
        low:
            Low-resolution branch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated branches.
        """

        high_u = self.high(high) + F.interpolate(self.up(low), size=high.shape[-2:], mode="nearest")
        low_u = self.low(low) + self.down(high)
        return high_u, low_u


class BottomUpHRPose(nn.Module):
    """HRNet bottom-up pose estimator with associative/tag and offset variants."""

    def __init__(self, mode: str = "ae", joints: int = 17, channels: int = 24) -> None:
        """Initialize compact HRNet pose model.

        Parameters
        ----------
        mode:
            Head variant: ``ae``, ``cid``, ``dekr``, ``dark``, ``dwpose``, or ``cpm``.
        joints:
            Number of keypoints.
        channels:
            HRNet branch width.
        """

        super().__init__()
        self.mode = mode
        self.stem = nn.Sequential(ConvBNAct(3, channels), ConvBNAct(channels, channels))
        self.low_proj = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([HRBlock(channels) for _ in range(2)])
        self.heatmap = nn.Conv2d(channels, joints, 1)
        self.tag = nn.Conv2d(channels, joints, 1)
        self.offset = nn.Conv2d(channels, joints * 2, 1)
        self.instance = nn.Conv2d(channels, 1, 1)
        self.stage_refine = nn.Conv2d(joints, joints, 3, padding=1)
        self.context_proj = nn.Conv2d(channels, channels, 1)
        self.instance_proj = nn.Conv2d(channels, channels, 1)
        self.dekr_gate = nn.Conv2d(joints, joints * 2, 1)
        self.cpm_stage = nn.Sequential(
            ConvBNAct(channels + joints, channels), nn.Conv2d(channels, joints, 1)
        )
        self.dark_coord = nn.Linear(joints * 2, joints * 2)
        self.wholebody_simcc_x = nn.Linear(channels, joints * 64)
        self.wholebody_simcc_y = nn.Linear(channels, joints * 64)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict bottom-up pose maps.

        Parameters
        ----------
        x:
            RGB image.

        Returns
        -------
        dict[str, torch.Tensor]
            Heatmap plus variant-specific pose tensors.
        """

        high = self.stem(x)
        low = self.low_proj(high)
        for block in self.blocks:
            high, low = block(high, low)
        heat = self.heatmap(high)
        out: dict[str, torch.Tensor] = {"heatmap": heat}
        if self.mode == "ae":
            out["tag"] = self.tag(high)
        elif self.mode == "cid":
            instance = torch.sigmoid(self.instance(high))
            context = (self.context_proj(high) * instance).mean(dim=(2, 3), keepdim=True)
            decoupled = self.instance_proj(high) + context
            out["instance"] = instance
            out["context_decoupled_heatmap"] = self.heatmap(decoupled)
            out["tag"] = self.tag(decoupled)
        elif self.mode == "dekr":
            base_offset = self.offset(high)
            adaptive = torch.tanh(self.dekr_gate(heat))
            out["disentangled_keypoint_regression"] = base_offset + adaptive
        elif self.mode == "dark":
            out["udp_offset"] = self.offset(high)
            prob = torch.softmax(heat.flatten(2), dim=-1).reshape_as(heat)
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, heat.shape[-2], device=x.device),
                torch.linspace(0, 1, heat.shape[-1], device=x.device),
                indexing="ij",
            )
            coords = torch.stack([(prob * xx).sum((2, 3)), (prob * yy).sum((2, 3))], dim=-1)
            out["dark_unbiased_coordinate"] = self.dark_coord(coords.flatten(1)).view_as(coords)
            out["dark_refined"] = heat + 0.25 * self.stage_refine(heat)
        elif self.mode == "dwpose":
            pooled = high.mean(dim=(2, 3))
            out["wholebody_simcc_x"] = self.wholebody_simcc_x(pooled).view(x.shape[0], -1, 64)
            out["wholebody_simcc_y"] = self.wholebody_simcc_y(pooled).view(x.shape[0], -1, 64)
            out["distilled_body_hands_face"] = torch.cat((heat, self.offset(high)), dim=1)
        else:
            out["stage2"] = self.cpm_stage(torch.cat((high, heat), dim=1))
        return out


class YOLOXPoseCompact(nn.Module):
    """YOLOX/RTMO-style one-stage pose detector with decoupled keypoint head."""

    def __init__(self, channels: int = 24, joints: int = 17) -> None:
        """Initialize CSP-like backbone, PAN fusion, and pose head.

        Parameters
        ----------
        channels:
            Base feature width.
        joints:
            Number of keypoints.
        """

        super().__init__()
        self.c1 = ConvBNAct(3, channels, stride=2)
        self.c2 = ConvBNAct(channels, channels * 2, stride=2)
        self.c3 = ConvBNAct(channels * 2, channels * 4, stride=2)
        self.lat = nn.Conv2d(channels * 4, channels * 2, 1)
        self.fuse = ConvBNAct(channels * 4, channels * 2)
        self.cls = nn.Conv2d(channels * 2, 1, 1)
        self.box = nn.Conv2d(channels * 2, 4, 1)
        self.kpt_pool = nn.AdaptiveAvgPool2d(1)
        self.simcc_x = nn.Linear(channels * 2, joints * 64)
        self.simcc_y = nn.Linear(channels * 2, joints * 64)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict objectness, boxes, and keypoint triplets.

        Parameters
        ----------
        x:
            RGB image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Detection and keypoint maps.
        """

        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        up = F.interpolate(self.lat(c3), size=c2.shape[-2:], mode="nearest")
        feat = self.fuse(torch.cat((c2, up), dim=1))
        pooled = self.kpt_pool(feat).flatten(1)
        return (
            self.cls(feat),
            self.box(feat),
            self.simcc_x(pooled).view(x.shape[0], -1, 64),
            self.simcc_y(pooled).view(x.shape[0], -1, 64),
        )


class C2DRecognizer(nn.Module):
    """C2D action recognizer with 2D CNN frames and temporal consensus."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize frame CNN and temporal classifier.

        Parameters
        ----------
        classes:
            Number of action classes.
        """

        super().__init__()
        self.frame = nn.Sequential(
            ConvBNAct(3, 16, stride=2),
            ConvBNAct(16, 32, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Classify a clip by average frame consensus.

        Parameters
        ----------
        video:
            Video tensor ``(batch, channels, time, height, width)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        bsz, channels, time, height, width = video.shape
        frames = video.transpose(1, 2).reshape(bsz * time, channels, height, width)
        feat = self.frame(frames).flatten(1).reshape(bsz, time, -1).mean(dim=1)
        return self.fc(feat)


class CSNBlock(nn.Module):
    """Channel-separated 3D convolution block."""

    def __init__(self, channels: int) -> None:
        """Initialize depthwise temporal-spatial and pointwise convolutions.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.dw = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv3d(channels, channels, 1)
        self.bn = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-separated residual convolution.

        Parameters
        ----------
        x:
            Video feature tensor.

        Returns
        -------
        torch.Tensor
            Updated feature tensor.
        """

        return F.relu(x + self.bn(self.pw(self.dw(x))))


class CSNRecognizer(nn.Module):
    """CSN video recognizer with depthwise separable 3D residual blocks."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize compact CSN.

        Parameters
        ----------
        classes:
            Number of classes.
        """

        super().__init__()
        self.stem = nn.Conv3d(3, 24, 3, stride=(1, 2, 2), padding=1)
        self.blocks = nn.Sequential(CSNBlock(24), CSNBlock(24))
        self.fc = nn.Linear(24, classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Classify video with channel-separated 3D convolutions.

        Parameters
        ----------
        video:
            Video tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        feat = self.blocks(F.relu(self.stem(video))).mean(dim=(-1, -2, -3))
        return self.fc(feat)


class CAINCompact(nn.Module):
    """CAIN frame interpolation with channel attention and pixel shuffle."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize interpolation encoder and channel-attention mixer.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.enc = nn.Conv2d(6, channels, 3, padding=1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid()
        )
        self.res = nn.Sequential(ResBlock(channels), ResBlock(channels))
        self.up = nn.Sequential(nn.Conv2d(channels, 12, 3, padding=1), nn.PixelShuffle(2))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Interpolate the middle frame from two RGB frames.

        Parameters
        ----------
        frames:
            Concatenated two-frame tensor with six channels.

        Returns
        -------
        torch.Tensor
            Interpolated RGB frame.
        """

        feat = F.relu(self.enc(frames))
        feat = self.res(feat * self.attn(feat))
        return torch.tanh(self.up(feat))


class MaskRCNNCompact(nn.Module):
    """Mask R-CNN with backbone, FPN, RPN, ROI classifier, and mask head."""

    def __init__(self) -> None:
        """Initialize compact Mask R-CNN modules."""

        super().__init__()
        self.det = FasterRCNN(channels=24, proposals=6, classes=5)
        self.mask = nn.Sequential(
            nn.ConvTranspose2d(24, 24, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 5, 1),
        )

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run two-stage detection and a per-class mask branch.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            RPN boxes, class logits, box deltas, and masks.
        """

        feat = self.det.backbone(image)
        rpn_feat = F.relu(self.det.rpn(feat))
        obj = self.det.rpn_obj(rpn_feat)
        rpn_box = self.det.rpn_box(rpn_feat).flatten(2).transpose(1, 2)
        roi = self.det.roi_fc(self.det.roi_features(feat, obj))
        masks = self.mask(F.adaptive_avg_pool2d(feat, (7, 7)))
        return rpn_box, self.det.cls(roi), self.det.box(roi), masks


class CascadeMaskRCNNCompact(nn.Module):
    """Cascade Mask R-CNN with cascaded RoI box heads plus FCN mask branch."""

    def __init__(self, stages: int = 3) -> None:
        """Initialize the FPN/RPN/RoI cascade and mask head.

        Parameters
        ----------
        stages:
            Number of cascade box-refinement stages.
        """

        super().__init__()
        self.det = FasterRCNN(channels=24, proposals=6, classes=5)
        self.cascade_heads = nn.ModuleList([nn.Linear(24, 20) for _ in range(stages)])
        self.mask_info_flow = nn.Conv2d(5, 5, 3, padding=1)
        self.mask = nn.Sequential(
            nn.ConvTranspose2d(24, 24, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 5, 1),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run RPN proposals through cascade box stages and a mask branch.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[torch.Tensor, ...]
            RPN boxes, class logits, cascade boxes, and masks.
        """

        feat = self.det.backbone(image)
        rpn_feat = F.relu(self.det.rpn(feat))
        obj = self.det.rpn_obj(rpn_feat)
        rpn_box = self.det.rpn_box(rpn_feat).flatten(2).transpose(1, 2)
        roi = self.det.roi_fc(self.det.roi_features(feat, obj))
        cls = self.det.cls(roi)
        refined = self.det.box(roi)
        cascade_boxes = []
        for head in self.cascade_heads:
            refined = refined + 0.1 * head(roi)
            cascade_boxes.append(refined)
        mask = self.mask(F.adaptive_avg_pool2d(feat, (7, 7)))
        mask = mask + self.mask_info_flow(mask)
        return (rpn_box, cls, *cascade_boxes, mask)


class CenterPointCompact(nn.Module):
    """CenterPoint BEV detector with heatmap center and box-regression heads."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize pillar encoder and BEV heads.

        Parameters
        ----------
        channels:
            BEV feature width.
        """

        super().__init__()
        self.pillar = nn.Sequential(
            nn.Linear(4, channels), nn.ReLU(), nn.Linear(channels, channels)
        )
        self.bev = nn.Conv2d(channels, channels, 3, padding=1)
        self.heat = nn.Conv2d(channels, 3, 1)
        self.box = nn.Conv2d(channels, 8, 1)
        self.vel = nn.Conv2d(channels, 2, 1)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Voxelize points coarsely and predict BEV centers.

        Parameters
        ----------
        points:
            Point tensor ``(batch, points, xyzi)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Heatmap, box, and velocity maps.
        """

        feat = self.pillar(points).transpose(1, 2)
        side = int(math.sqrt(points.shape[1]))
        bev = feat[:, :, : side * side].reshape(points.shape[0], -1, side, side)
        bev = F.relu(self.bev(bev))
        return torch.sigmoid(self.heat(bev)), self.box(bev), self.vel(bev)

    def features(self, points: torch.Tensor) -> torch.Tensor:
        """Return the BEV feature map before CenterPoint prediction heads.

        Parameters
        ----------
        points:
            Point tensor ``(batch, points, xyzi)``.

        Returns
        -------
        torch.Tensor
            BEV feature map.
        """

        feat = self.pillar(points).transpose(1, 2)
        side = int(math.sqrt(points.shape[1]))
        bev = feat[:, :, : side * side].reshape(points.shape[0], -1, side, side)
        return F.relu(self.bev(bev))


class CenterFormerCompact(nn.Module):
    """CenterFormer-style CenterPoint head with BEV transformer refinement."""

    def __init__(self) -> None:
        """Initialize CenterPoint base and BEV transformer."""

        super().__init__()
        self.base = CenterPointCompact()
        layer = nn.TransformerDecoderLayer(24, 4, 48, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, 1)
        self.refine = nn.Linear(24, 10)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Predict transformer-refined center features.

        Parameters
        ----------
        points:
            Point tensor.

        Returns
        -------
        torch.Tensor
            Refined BEV center predictions.
        """

        bev = self.base.features(points)
        heat = torch.sigmoid(self.base.heat(bev))
        scores = heat.flatten(2).max(dim=1).values
        idx = torch.topk(scores, 8, dim=1).indices
        tokens = bev.flatten(2).transpose(1, 2)
        gather = idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        center_queries = torch.gather(tokens, 1, gather)
        decoded = self.transformer(center_queries, tokens)
        return self.refine(decoded)


class RotatedFCOSCompact(nn.Module):
    """Rotated FCOS/CSL/CFA detector with FPN and angle-aware dense head."""

    def __init__(self, mode: str = "csl") -> None:
        """Initialize rotated detector.

        Parameters
        ----------
        mode:
            Rotated-head variant: ``csl`` or ``cfa``.
        """

        super().__init__()
        self.mode = mode
        self.c3 = ConvBNAct(3, 24, stride=2)
        self.c4 = ConvBNAct(24, 24, stride=2)
        self.lat = nn.Conv2d(24, 24, 1)
        self.cls = nn.Conv2d(24, 5, 3, padding=1)
        self.box = nn.Conv2d(24, 5, 3, padding=1)
        self.angle = nn.Conv2d(24, 36 if mode == "csl" else 1, 3, padding=1)
        self.align = nn.Conv2d(24, 1, 3, padding=1)
        self.convex_points = nn.Conv2d(24, 18, 3, padding=1)
        self.feature_adapt = nn.Conv2d(24 + 18, 24, 3, padding=1)
        self.ciou_quality = nn.Conv2d(24, 1, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict rotated boxes and variant-specific angle outputs.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        dict[str, torch.Tensor]
            Dense rotated detection maps.
        """

        p = self.lat(self.c4(self.c3(image)))
        out = {"cls": self.cls(p), "rbox": self.box(p), "angle": self.angle(p)}
        if self.mode == "cfa":
            convex = self.convex_points(p)
            adapted = self.feature_adapt(torch.cat((p, torch.tanh(convex)), dim=1))
            out["convex_hull_points"] = convex
            out["convex_feature_align"] = torch.sigmoid(self.align(adapted))
            out["ciou_quality"] = torch.sigmoid(self.ciou_quality(adapted))
        return out


class Cylinder3DCompact(nn.Module):
    """Cylinder3D point-cloud segmentation with cylindrical bins and asymmetric 3D conv."""

    def __init__(self, channels: int = 16, classes: int = 8) -> None:
        """Initialize point embedding and sparse-like 3D convolution stack.

        Parameters
        ----------
        channels:
            Hidden channel count.
        classes:
            Number of semantic classes.
        """

        super().__init__()
        self.embed = nn.Linear(4, channels)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, (3, 1, 3), padding=(1, 0, 1)),
            nn.ReLU(),
            nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
        )
        self.head = nn.Conv3d(channels, classes, 1)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Segment a compact cylindrical point grid.

        Parameters
        ----------
        points:
            Point tensor ``(batch, 64, 4)``.

        Returns
        -------
        torch.Tensor
            Per-cylinder semantic logits.
        """

        feat = self.embed(points).transpose(1, 2).reshape(points.shape[0], -1, 4, 4, 4)
        return self.head(self.conv(feat))


class DeepFillV1Compact(nn.Module):
    """DeepFill v1 coarse-to-fine inpainting with dilated refinement."""

    def __init__(self) -> None:
        """Initialize coarse and refinement branches."""

        super().__init__()
        self.coarse = nn.Sequential(
            ConvBNAct(4, 24), ConvBNAct(24, 24), nn.Conv2d(24, 3, 3, padding=1)
        )
        self.refine = nn.Sequential(
            ConvBNAct(7, 24),
            nn.Conv2d(24, 24, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(24, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inpaint masked image with coarse-to-fine refinement.

        Parameters
        ----------
        x:
            RGB image plus mask.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Coarse and refined images.
        """

        coarse = torch.tanh(self.coarse(x))
        refined = torch.tanh(self.refine(torch.cat((x, coarse), dim=1)))
        return coarse, refined


class DeblurGANv2Compact(nn.Module):
    """DeblurGAN-v2 FPN generator with residual image prediction."""

    def __init__(self) -> None:
        """Initialize encoder, FPN lateral path, and restoration head."""

        super().__init__()
        self.e1 = ConvBNAct(3, 16)
        self.e2 = ConvBNAct(16, 32, stride=2)
        self.e3 = ConvBNAct(32, 64, stride=2)
        self.l2 = nn.Conv2d(32, 24, 1)
        self.l3 = nn.Conv2d(64, 24, 1)
        self.head = nn.Sequential(ResBlock(24), nn.Conv2d(24, 3, 3, padding=1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Restore a blurred image.

        Parameters
        ----------
        image:
            Blurred RGB image.

        Returns
        -------
        torch.Tensor
            Deblurred RGB image.
        """

        e1 = self.e1(image)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        f = self.l2(e2) + F.interpolate(self.l3(e3), size=e2.shape[-2:], mode="nearest")
        return torch.tanh(
            F.interpolate(self.head(f), size=image.shape[-2:], mode="nearest") + image
        )


class DICCompact(nn.Module):
    """DIC face super-resolution with iterative heatmap feedback."""

    def __init__(self, steps: int = 2) -> None:
        """Initialize SR trunk and landmark feedback loop.

        Parameters
        ----------
        steps:
            Number of iterative correction steps.
        """

        super().__init__()
        self.steps = steps
        self.feat = ConvBNAct(3, 24)
        self.landmark = nn.Conv2d(24, 5, 1)
        self.feedback = nn.Conv2d(29, 24, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1), nn.PixelShuffle(2), nn.Conv2d(12, 3, 3, padding=1)
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Super-resolve with landmark heatmap feedback.

        Parameters
        ----------
        image:
            Low-resolution face image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Super-resolved image and landmark heatmaps.
        """

        feat = self.feat(image)
        heat = self.landmark(feat)
        for _ in range(self.steps):
            feat = F.relu(self.feedback(torch.cat((feat, heat), dim=1)))
            heat = self.landmark(feat)
        return self.up(feat), heat


class DIMCompact(nn.Module):
    """Deep Image Matting encoder-decoder with alpha matte head."""

    def __init__(self) -> None:
        """Initialize trimap-conditioned matting network."""

        super().__init__()
        self.enc1 = ConvBNAct(4, 16)
        self.enc2 = ConvBNAct(16, 32, stride=2)
        self.dec = nn.Conv2d(48, 16, 3, padding=1)
        self.alpha = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict an alpha matte from RGB plus trimap.

        Parameters
        ----------
        x:
            Four-channel RGB+trimap tensor.

        Returns
        -------
        torch.Tensor
            Alpha matte.
        """

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        up = F.interpolate(e2, size=e1.shape[-2:], mode="nearest")
        return torch.sigmoid(self.alpha(F.relu(self.dec(torch.cat((e1, up), dim=1)))))


class ControlNetCompact(nn.Module):
    """ControlNet locked U-Net residual with zero-conv control branch."""

    def __init__(self) -> None:
        """Initialize base and conditioning branches."""

        super().__init__()
        self.locked = nn.Sequential(ConvBNAct(3, 24), ResBlock(24))
        self.cond = nn.Sequential(ConvBNAct(1, 24), nn.Conv2d(24, 24, 1))
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)
        self.out = nn.Conv2d(24, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply controlled residual denoising.

        Parameters
        ----------
        x:
            RGB image plus one-channel condition.

        Returns
        -------
        torch.Tensor
            Controlled output image.
        """

        return self.out(self.locked(x[:, :3]) + self.cond(x[:, 3:4]))


class TinyDiffusionUNet(nn.Module):
    """Diffusion U-Net with timestep and optional cross-conditioning."""

    def __init__(self, conditioned: bool = False) -> None:
        """Initialize compact diffusion U-Net.

        Parameters
        ----------
        conditioned:
            Whether to use text/control conditioning.
        """

        super().__init__()
        self.conditioned = conditioned
        self.time = nn.Linear(1, 24)
        self.down = ConvBNAct(4, 24, stride=2)
        self.mid = ResBlock(24)
        self.cond = nn.Linear(16, 24) if conditioned else nn.Identity()
        self.up = nn.ConvTranspose2d(24, 4, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise latent input.

        Parameters
        ----------
        x:
            Latent tensor.

        Returns
        -------
        torch.Tensor
            Predicted noise.
        """

        t = self.time(torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)).view(
            x.shape[0], -1, 1, 1
        )
        feat = self.mid(self.down(x) + t)
        if self.conditioned:
            cond = torch.zeros(x.shape[0], 16, device=x.device, dtype=x.dtype)
            feat = feat + self.cond(cond).view(x.shape[0], -1, 1, 1)
        return self.up(feat)


class DiffusersPipelineCompact(nn.Module):
    """Diffusers-style inference pipeline with text encoder, U-Net, VAE, and scheduler."""

    def __init__(self, dual_text: bool = False) -> None:
        """Initialize compact pipeline components.

        Parameters
        ----------
        dual_text:
            Whether to use SDXL-style dual text encoders.
        """

        super().__init__()
        self.dual_text = dual_text
        self.text = nn.Embedding(32, 16)
        self.text_2 = nn.Embedding(32, 16) if dual_text else None
        self.unet = TinyDiffusionUNet(conditioned=True)
        self.vae_decode = nn.ConvTranspose2d(4, 3, 4, stride=2, padding=1)
        self.scheduler = nn.Linear(1, 1)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one compact denoising and VAE decode pipeline step.

        Parameters
        ----------
        latent:
            Latent tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Decoded image and predicted noise.
        """

        ids = torch.arange(8, device=latent.device).unsqueeze(0).expand(latent.shape[0], -1)
        cond = self.text(ids).mean(dim=1)
        if self.text_2 is not None:
            cond = cond + self.text_2(ids).mean(dim=1)
        step_scale = self.scheduler(
            torch.ones(latent.shape[0], 1, device=latent.device, dtype=latent.dtype)
        )
        noise = self.unet(
            latent + cond[:, :4].view(latent.shape[0], 4, 1, 1) * step_scale.view(-1, 1, 1, 1)
        )
        denoised = latent - 0.1 * noise
        return torch.tanh(self.vae_decode(denoised)), noise


class DiscoDiffusionCompact(nn.Module):
    """CLIP-guided Disco Diffusion denoising loop."""

    def __init__(self) -> None:
        """Initialize denoiser and CLIP guidance projections."""

        super().__init__()
        self.unet = TinyDiffusionUNet(conditioned=True)
        self.image_clip = nn.Linear(4, 16)
        self.text_clip = nn.Embedding(32, 16)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one CLIP-guided diffusion update.

        Parameters
        ----------
        latent:
            Latent tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Guided latent and CLIP guidance score.
        """

        noise = self.unet(latent)
        ids = torch.arange(8, device=latent.device).unsqueeze(0).expand(latent.shape[0], -1)
        image_feat = F.normalize(self.image_clip(latent.mean(dim=(2, 3))), dim=-1)
        text_feat = F.normalize(self.text_clip(ids).mean(dim=1), dim=-1)
        guidance = (image_feat * text_feat).sum(dim=-1, keepdim=True)
        return latent - 0.1 * noise + guidance.view(-1, 1, 1, 1), guidance


class DragGANCompact(nn.Module):
    """DragGAN point-tracking and motion-supervised latent manipulation wrapper."""

    def __init__(self) -> None:
        """Initialize StyleGAN2 generator and motion supervision heads."""

        super().__init__()
        self.generator = build_stylegan2_generator()
        self.point_tracker = nn.Linear(64, 4)
        self.motion = nn.Linear(4, 64)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate an image after a compact DragGAN latent update.

        Parameters
        ----------
        latent:
            StyleGAN latent vector.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Manipulated image, tracked points, and latent update.
        """

        points = self.point_tracker(latent)
        target_motion = torch.tanh(points)
        delta = self.motion(target_motion)
        image = self.generator(latent + 0.05 * delta)
        return image, points, delta


class DreamBoothCompact(nn.Module):
    """DreamBooth personalization step with text conditioning and prior preservation."""

    def __init__(self) -> None:
        """Initialize personalized token encoder, U-Net, and prior branch."""

        super().__init__()
        self.text = nn.Embedding(64, 16)
        self.subject_token = nn.Parameter(torch.randn(16) * 0.02)
        self.unet = TinyDiffusionUNet(conditioned=True)
        self.prior = TinyDiffusionUNet(conditioned=True)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict subject-specific and prior-preservation noise.

        Parameters
        ----------
        latent:
            Latent tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Subject noise and class-prior noise.
        """

        ids = torch.arange(8, device=latent.device).unsqueeze(0).expand(latent.shape[0], -1)
        text = self.text(ids).mean(dim=1) + self.subject_token
        subject_latent = latent + text[:, :4].view(latent.shape[0], 4, 1, 1)
        return self.unet(subject_latent), self.prior(latent)


class DRRGCompact(nn.Module):
    """DRRG text detector with component graph reasoning."""

    def __init__(self) -> None:
        """Initialize text component CNN and graph message passing."""

        super().__init__()
        self.backbone = nn.Sequential(ConvBNAct(3, 24, stride=2), ConvBNAct(24, 24, stride=2))
        self.comp = nn.Conv2d(24, 8, 1)
        self.edge = nn.Linear(16, 8)
        self.node = nn.Linear(16, 8)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict text components and graph-linked relations.

        Parameters
        ----------
        image:
            RGB text image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Component maps and relation logits.
        """

        comp = self.comp(self.backbone(image))
        nodes = comp.flatten(2).transpose(1, 2)[:, :8]
        pair = torch.cat(
            (nodes.unsqueeze(2).expand(-1, -1, 8, -1), nodes.unsqueeze(1).expand(-1, 8, -1, -1)),
            dim=-1,
        )
        rel = self.edge(pair).mean(dim=-1) + self.node(pair.mean(dim=2)).mean(dim=-1).unsqueeze(-1)
        return comp, rel


def build_blip() -> nn.Module:
    """Build compact BLIP."""

    return BLIPCompact().eval()


def build_blip2() -> nn.Module:
    """Build compact BLIP-2."""

    return BLIP2Compact().eval()


def build_chinese_clip() -> nn.Module:
    """Build compact Chinese-CLIP."""

    return ChineseCLIPCompact().eval()


def build_clip4clip() -> nn.Module:
    """Build compact CLIP4Clip."""

    return CLIP4ClipCompact().eval()


def build_bmn() -> nn.Module:
    """Build compact BMN."""

    return TemporalProposalNet("bmn").eval()


def build_bsn() -> nn.Module:
    """Build compact BSN."""

    return TemporalProposalNet("bsn").eval()


def build_drn() -> nn.Module:
    """Build compact DRN temporal localization."""

    return TemporalProposalNet("drn").eval()


def build_pose_ae() -> nn.Module:
    """Build HRNet associative-embedding bottom-up pose estimator."""

    return BottomUpHRPose("ae").eval()


def build_pose_cid() -> nn.Module:
    """Build CID bottom-up pose estimator."""

    return BottomUpHRPose("cid").eval()


def build_pose_dekr() -> nn.Module:
    """Build DEKR displacement-regression pose estimator."""

    return BottomUpHRPose("dekr").eval()


def build_pose_dark() -> nn.Module:
    """Build DarkPose HRNet with UDP/DARK refinement."""

    return BottomUpHRPose("dark").eval()


def build_pose_dwpose() -> nn.Module:
    """Build DWPose-style whole-body HRNet estimator."""

    return BottomUpHRPose("dwpose").eval()


def build_pose_cpm() -> nn.Module:
    """Build convolutional pose machine with staged heatmap refinement."""

    return BottomUpHRPose("cpm").eval()


def build_yoloxpose() -> nn.Module:
    """Build YOLOX-Pose/RTMO compact one-stage pose detector."""

    return YOLOXPoseCompact().eval()


def build_c2d() -> nn.Module:
    """Build compact C2D video recognizer."""

    return C2DRecognizer().eval()


def build_csn() -> nn.Module:
    """Build compact CSN video recognizer."""

    return CSNRecognizer().eval()


def build_cain() -> nn.Module:
    """Build compact CAIN frame interpolation model."""

    return CAINCompact().eval()


def build_cascade_mask_rcnn() -> nn.Module:
    """Build Cascade Mask R-CNN with FPN/RPN/cascade/mask primitives."""

    return CascadeMaskRCNNCompact().eval()


def build_mask_rcnn() -> nn.Module:
    """Build compact Mask R-CNN."""

    return MaskRCNNCompact().eval()


def build_faster_rcnn() -> nn.Module:
    """Build compact Faster R-CNN."""

    return FasterRCNN().eval()


def build_retinanet() -> nn.Module:
    """Build compact RetinaNet."""

    return RetinaNet().eval()


def build_centerpoint() -> nn.Module:
    """Build compact CenterPoint."""

    return CenterPointCompact().eval()


def build_centerformer() -> nn.Module:
    """Build compact CenterFormer."""

    return CenterFormerCompact().eval()


def build_rotated_csl() -> nn.Module:
    """Build compact rotated FCOS with CSL angle-classification head."""

    return RotatedFCOSCompact("csl").eval()


def build_rotated_cfa() -> nn.Module:
    """Build compact CFA rotated detector."""

    return RotatedFCOSCompact("cfa").eval()


def build_cylinder3d() -> nn.Module:
    """Build compact Cylinder3D."""

    return Cylinder3DCompact().eval()


def build_deepfillv1() -> nn.Module:
    """Build compact DeepFill v1."""

    return DeepFillV1Compact().eval()


def build_deblurganv2() -> nn.Module:
    """Build compact DeblurGAN-v2."""

    return DeblurGANv2Compact().eval()


def build_dic() -> nn.Module:
    """Build compact DIC face super-resolution."""

    return DICCompact().eval()


def build_dim() -> nn.Module:
    """Build compact DIM matting model."""

    return DIMCompact().eval()


def build_controlnet() -> nn.Module:
    """Build compact ControlNet."""

    return ControlNetCompact().eval()


def build_diffusion_unet() -> nn.Module:
    """Build compact text-conditioned diffusion U-Net."""

    return TinyDiffusionUNet(conditioned=True).eval()


def build_diffusers_pipeline() -> nn.Module:
    """Build compact Diffusers pipeline with scheduler, text encoder, U-Net, and VAE."""

    return DiffusersPipelineCompact().eval()


def build_sdxl_pipeline() -> nn.Module:
    """Build compact SDXL-style Diffusers pipeline with dual text encoders."""

    return DiffusersPipelineCompact(dual_text=True).eval()


def build_disco_diffusion() -> nn.Module:
    """Build compact CLIP-guided Disco Diffusion sampler."""

    return DiscoDiffusionCompact().eval()


def build_draggan() -> nn.Module:
    """Build compact DragGAN manipulation wrapper around StyleGAN2."""

    return DragGANCompact().eval()


def build_dreambooth() -> nn.Module:
    """Build compact DreamBooth personalization/prior-preservation model."""

    return DreamBoothCompact().eval()


def build_drrg() -> nn.Module:
    """Build compact DRRG text detector."""

    return DRRGCompact().eval()


def example_placeholder() -> torch.Tensor:
    """Return placeholder batch tensor."""

    return torch.zeros(1, 1)


def example_temporal() -> torch.Tensor:
    """Return temporal features for localization."""

    return torch.randn(1, 16, 16)


def example_pose_image() -> torch.Tensor:
    """Return compact pose image."""

    return torch.randn(1, 3, 64, 64)


def example_video_small() -> torch.Tensor:
    """Return compact video clip."""

    return torch.randn(1, 3, 4, 32, 32)


def example_two_frames() -> torch.Tensor:
    """Return concatenated pair of RGB frames."""

    return torch.randn(1, 6, 32, 32)


def example_points() -> torch.Tensor:
    """Return compact point cloud."""

    return torch.randn(1, 64, 4)


def example_latent() -> torch.Tensor:
    """Return compact diffusion latent."""

    return torch.randn(1, 4, 32, 32)


MENAGERIE_ENTRIES = [
    ("mmpretrain:blip", "build_blip", "example_placeholder", "2022", "DC"),
    ("mmpretrain:blip2", "build_blip2", "example_placeholder", "2023", "DC"),
    ("mmaction2_localization_bmn", "build_bmn", "example_temporal", "2019", "DC"),
    ("mmaction:bmn", "build_bmn", "example_temporal", "2019", "DC"),
    (
        "mmpose_bottomup_pose_estimator_ae_hrnet_w32_8xb24_300e_coco_512x512",
        "build_pose_ae",
        "example_pose_image",
        "2019",
        "DC",
    ),
    (
        "mmpose_bottomup_pose_estimator_cid_hrnet_w32_8xb20_140e_coco_512x512",
        "build_pose_cid",
        "example_pose_image",
        "2022",
        "DC",
    ),
    (
        "mmpose_bottomup_pose_estimator_dekr_hrnet_w32_8xb10_140e_coco_512x512",
        "build_pose_dekr",
        "example_pose_image",
        "2021",
        "DC",
    ),
    (
        "mmpose_bottomup_pose_estimator_rtmo_l_16xb16_600e_body7_640x640",
        "build_yoloxpose",
        "example_pose_image",
        "2023",
        "DC",
    ),
    (
        "mmpose_bottomup_pose_estimator_yoloxpose_l_8xb32_300e_coco_640",
        "build_yoloxpose",
        "example_pose_image",
        "2022",
        "DC",
    ),
    ("mmaction2_localization_bsn", "build_bsn", "example_temporal", "2018", "DC"),
    ("mmaction:bsn", "build_bsn", "example_temporal", "2018", "DC"),
    ("mmaction2_c2d_r101_8x8_k400", "build_c2d", "example_video_small", "2018", "DC"),
    ("mmaction2_c2d_r50_8x8_k400", "build_c2d", "example_video_small", "2018", "DC"),
    ("mmaction2_recognition_c2d", "build_c2d", "example_video_small", "2018", "DC"),
    ("mmaction:c2d", "build_c2d", "example_video_small", "2018", "DC"),
    ("mmaction_c2d_r50", "build_c2d", "example_video_small", "2018", "DC"),
    ("mmaction2_c3d_sports1m_ucf101", "build_c3d", "example_input_clip", "2015", "DC"),
    ("mmaction2_recognition_c3d", "build_c3d", "example_input_clip", "2015", "DC"),
    ("mmaction:c3d", "build_c3d", "example_input_clip", "2015", "DC"),
    ("mmagic:cain", "build_cain", "example_two_frames", "2020", "DC"),
    ("mmagic_cain", "build_cain", "example_two_frames", "2020", "DC"),
    (
        "mmagic_cain_cain_g1b32_1xb5_vimeo90k_triplet",
        "build_cain",
        "example_two_frames",
        "2020",
        "DC",
    ),
    (
        "mmdet3d_cascade_rcnn_cascade_mask_rcnn_r50_fpn_coco_20e_nuim",
        "build_cascade_mask_rcnn",
        "example_pose_image",
        "2018",
        "DC",
    ),
    (
        "mmdet3d_cascade_rcnn_cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim",
        "build_cascade_mask_rcnn",
        "example_pose_image",
        "2018",
        "DC",
    ),
    ("mmdet3d_centerformer", "build_centerformer", "example_points", "2022", "DC"),
    ("mmdet3d:centerpoint", "build_centerpoint", "example_points", "2020", "DC"),
    (
        "mmdet3d_centerpoint_centerpoint_pillar02_second_secfpn_8xb4_cyclic_20e_nus_3d",
        "build_centerpoint",
        "example_points",
        "2020",
        "DC",
    ),
    ("openpcdet_centerpoint", "build_centerpoint", "example_points", "2020", "DC"),
    ("mmrotate:cfa", "build_rotated_cfa", "example_pose_image", "2021", "DC"),
    (
        "mmrotate_cfa_cfa_r50_fpn_1x_dota_le135",
        "build_rotated_cfa",
        "example_pose_image",
        "2021",
        "DC",
    ),
    ("mmrotate_cfa_r50_fpn", "build_rotated_cfa", "example_pose_image", "2021", "DC"),
    ("mmpretrain:chinese_clip", "build_chinese_clip", "example_placeholder", "2022", "DC"),
    ("CID-HRNet-W48", "build_pose_cid", "example_pose_image", "2022", "DC"),
    ("mmpose:cid", "build_pose_cid", "example_pose_image", "2022", "DC"),
    ("mmpose_cid", "build_pose_cid", "example_pose_image", "2022", "DC"),
    ("mmaction2_retrieval_clip4clip", "build_clip4clip", "example_video_small", "2021", "DC"),
    ("mmagic:controlnet", "build_controlnet", "example_input_inpaint", "2023", "DC"),
    ("mmagic:controlnet_animation", "build_controlnet", "example_input_inpaint", "2023", "DC"),
    ("mmpose_cpm", "build_pose_cpm", "example_pose_image", "2016", "DC"),
    ("mmocr:crnn", "build_crnn", "example_text_image", "2015", "DC"),
    ("mmocr_textrecog_crnn", "build_crnn", "example_text_image", "2015", "DC"),
    ("mmrotate:csl", "build_rotated_csl", "example_pose_image", "2020", "DC"),
    ("mmrotate_csl_rotated_fcos", "build_rotated_csl", "example_pose_image", "2020", "DC"),
    ("mmaction2_recognition_csn", "build_csn", "example_video_small", "2019", "DC"),
    ("mmaction:csn", "build_csn", "example_video_small", "2019", "DC"),
    ("mmaction_csn_ip152", "build_csn", "example_video_small", "2019", "DC"),
    ("mmaction_csn_ir152", "build_csn", "example_video_small", "2019", "DC"),
    ("mmaction:custom_backbones", "build_csn", "example_video_small", "2019", "DC"),
    ("mmagic:cyclegan", "build_cyclegan_resnet_generator", "example_input_image", "2017", "DC"),
    ("mmagic_cyclegan", "build_cyclegan_resnet_generator", "example_input_image", "2017", "DC"),
    (
        "mmagic_cyclegan_cyclegan_lsgan_id0_resnet_in_1xb1_250kiters_summer2winter",
        "build_cyclegan_resnet_generator",
        "example_input_image",
        "2017",
        "DC",
    ),
    ("mmdet3d:cylinder3d", "build_cylinder3d", "example_points", "2021", "DC"),
    (
        "mmdet3d_cylinder3d_cylinder3d_4xb4_3x_semantickitti",
        "build_cylinder3d",
        "example_points",
        "2021",
        "DC",
    ),
    ("DarkPose-HRNet-W48", "build_pose_dark", "example_pose_image", "2019", "DC"),
    ("mmpose_dark", "build_pose_dark", "example_pose_image", "2019", "DC"),
    ("mmocr:dbnet", "build_dbnet_mobilenetv3", "example_det_input", "2020", "DC"),
    ("mmocr_textdet_dbnet", "build_dbnet_mobilenetv3", "example_det_input", "2020", "DC"),
    ("mmocr:dbnetpp", "build_dbnetpp_resnet50", "example_det_input", "2022", "DC"),
    ("mmocr_textdet_dbnetpp", "build_dbnetpp_resnet50", "example_det_input", "2022", "DC"),
    ("mmagic:dcgan", "build_dcgan_generator", "example_input_noise", "2015", "DC"),
    ("mmagic_dcgan", "build_dcgan_generator", "example_input_noise", "2015", "DC"),
    (
        "mmagic_dcgan_dcgan_1xb128_300kiters_celeba_cropped_64",
        "build_dcgan_generator",
        "example_input_noise",
        "2015",
        "DC",
    ),
    ("mmagic:deblurganv2", "build_deblurganv2", "example_input_image", "2019", "DC"),
    (
        "mmagic_deblurganv2_deblurganv2_fpn_inception_1xb1_gopro",
        "build_deblurganv2",
        "example_input_image",
        "2019",
        "DC",
    ),
    ("mmagic:deepfillv1", "build_deepfillv1", "example_input_inpaint", "2018", "DC"),
    ("mmagic_deepfillv1", "build_deepfillv1", "example_input_inpaint", "2018", "DC"),
    (
        "mmagic_deepfillv1_deepfillv1_4xb4_celeba_256x256",
        "build_deepfillv1",
        "example_input_inpaint",
        "2018",
        "DC",
    ),
    (
        "mmagic:deepfillv2",
        "build_deepfillv2_gated_generator",
        "example_input_inpaint",
        "2019",
        "DC",
    ),
    (
        "mmagic_deepfillv2",
        "build_deepfillv2_gated_generator",
        "example_input_inpaint",
        "2019",
        "DC",
    ),
    (
        "mmagic_deepfillv2_deepfillv2_8xb2_celeba_256x256",
        "build_deepfillv2_gated_generator",
        "example_input_inpaint",
        "2019",
        "DC",
    ),
    ("DEKR-HRNet-W32", "build_pose_dekr", "example_pose_image", "2021", "DC"),
    ("mmpose:dekr", "build_pose_dekr", "example_pose_image", "2021", "DC"),
    ("mmpose_dekr", "build_pose_dekr", "example_pose_image", "2021", "DC"),
    ("mmdet3d:dgcnn", "build_dgcnn_base", "example_dgcnn_input", "2019", "DC"),
    (
        "mmdet3d_dgcnn_dgcnn_4xb32_cosine_100e_s3dis_seg_test_area1",
        "build_dgcnn_base",
        "example_dgcnn_input",
        "2019",
        "DC",
    ),
    ("mmagic:dic", "build_dic", "example_input_image", "2020", "DC"),
    (
        "mmagic_dic_dic_gan_x8c48b6_4xb2_500k_celeba_hq",
        "build_dic",
        "example_input_image",
        "2020",
        "DC",
    ),
    ("mmagic:diffusers_pipeline", "build_diffusers_pipeline", "example_latent", "2022", "DC"),
    (
        "mmagic_diffusers_pipeline_sd_xl_pipeline",
        "build_sdxl_pipeline",
        "example_latent",
        "2023",
        "DC",
    ),
    ("mmagic:dim", "build_dim", "example_input_inpaint", "2017", "DC"),
    (
        "mmagic_dim_dim_stage1_v16_1xb1_1000k_comp1k",
        "build_dim",
        "example_input_inpaint",
        "2017",
        "DC",
    ),
    ("mmagic:disco_diffusion", "build_disco_diffusion", "example_latent", "2021", "DC"),
    (
        "mmagic_disco_diffusion_disco_diffusion_adm_u_finetuned_imagenet_256x256",
        "build_disco_diffusion",
        "example_latent",
        "2021",
        "DC",
    ),
    ("mmagic:draggan", "build_draggan", "example_input_stylegan2_generator", "2023", "DC"),
    ("mmagic_draggan", "build_draggan", "example_input_stylegan2_generator", "2023", "DC"),
    (
        "mmagic_draggan_stylegan2_1024x1024",
        "build_draggan",
        "example_input_stylegan2_generator",
        "2023",
        "DC",
    ),
    ("mmagic:dreambooth", "build_dreambooth", "example_latent", "2022", "DC"),
    (
        "mmagic_dreambooth_dreambooth_finetune_text_encoder",
        "build_dreambooth",
        "example_latent",
        "2022",
        "DC",
    ),
    ("mmaction2_localization_drn", "build_drn", "example_temporal", "2020", "DC"),
    ("mmaction:drn", "build_drn", "example_temporal", "2020", "DC"),
    ("mmocr:drrg", "build_drrg", "example_pose_image", "2020", "DC"),
    ("mmocr_textdet_drrg", "build_drrg", "example_pose_image", "2020", "DC"),
    ("mmpose:dwpose", "build_pose_dwpose", "example_pose_image", "2023", "DC"),
    ("mmpose_dwpose", "build_pose_dwpose", "example_pose_image", "2023", "DC"),
]
