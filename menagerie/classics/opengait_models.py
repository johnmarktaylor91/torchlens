"""OpenGait silhouette and body-model gait-recognition architectures.

The OpenGait benchmark compares GaitSet, GaitPart, GaitGL, GaitBase, and
SMPLGait.  These compact reconstructions preserve the signatures that matter for
TorchLens rendering: unordered set pooling over silhouette frames, horizontal
part pooling, global-local temporal convolutions, residual GaitBase blocks, and
an SMPL-style joint/shape branch fused with silhouette features.

Sources: OpenGait CVPR 2023, GaitSet AAAI 2019, GaitPart CVPR 2020,
GaitGL ICCV 2021, and SMPLGait CVPR 2022.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

GaitName = Literal["base", "gl", "part", "set", "smpl"]


class HorizontalPyramidPooling(nn.Module):
    """Horizontal pyramid pooling used by silhouette gait models.

    Parameters
    ----------
    bins:
        Tuple of vertical split counts.
    """

    def __init__(self, bins: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.bins = bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool feature maps into horizontal body parts.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Part descriptor tensor of shape ``(batch, parts, channels)``.
        """

        parts = []
        for bins in self.bins:
            chunks = torch.chunk(x, bins, dim=2)
            for chunk in chunks:
                pooled = F.adaptive_avg_pool2d(chunk, 1).flatten(1)
                parts.append(pooled)
        return torch.stack(parts, dim=1)


class GaitBackbone(nn.Module):
    """Small shared silhouette backbone.

    Parameters
    ----------
    channels:
        Hidden channel count.
    """

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, channels, 3, padding=1)
        self.block1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.block2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a batch of silhouette frames.

        Parameters
        ----------
        frames:
            Tensor of shape ``(batch, time, height, width)``.

        Returns
        -------
        torch.Tensor
            Encoded frames of shape ``(batch, time, channels, height, width)``.
        """

        batch, time, height, width = frames.shape
        x = frames.reshape(batch * time, 1, height, width)
        x = F.relu(self.stem(x))
        residual = x
        x = F.relu(self.block1(x))
        x = self.block2(x)
        x = F.relu(self.proj(x + residual))
        return x.view(batch, time, -1, height, width)


class GaitModel(nn.Module):
    """Compact multi-architecture gait recognizer.

    Parameters
    ----------
    kind:
        Gait architecture variant.
    embedding_dim:
        Output descriptor dimension.
    """

    def __init__(self, kind: GaitName, embedding_dim: int = 32) -> None:
        super().__init__()
        self.kind = kind
        self.backbone = GaitBackbone()
        self.hpp = HorizontalPyramidPooling()
        self.focal_small = nn.Conv2d(16, 16, 3, padding=1, groups=4)
        self.focal_large = nn.Conv2d(16, 16, 5, padding=2, groups=4)
        self.micro_motion = nn.Conv1d(16, 16, 3, padding=1, groups=16)
        self.part_temporal = nn.Conv1d(7 * 16, 7 * 16, 3, padding=1, groups=7)
        self.local3d = nn.Conv3d(16, 16, (3, 3, 3), padding=1, groups=4)
        self.global3d = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.smpl_mlp = nn.Sequential(nn.Linear(24 * 3 + 10, 32), nn.ReLU(), nn.Linear(32, 16))
        self.head = nn.Linear(7 * 16 + 16, embedding_dim)

    def forward(self, inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Extract a gait descriptor.

        Parameters
        ----------
        inputs:
            Silhouette tensor ``(batch, time, height, width)`` or, for SMPLGait,
            ``(silhouettes, smpl_features)`` where ``smpl_features`` is
            ``(batch, 82)``.

        Returns
        -------
        torch.Tensor
            Descriptor tensor of shape ``(batch, embedding_dim)``.
        """

        smpl_feat: torch.Tensor | None = None
        if isinstance(inputs, tuple):
            silhouettes, smpl_feat = inputs
        else:
            silhouettes = inputs
        encoded = self.backbone(silhouettes)
        if self.kind == "gl":
            encoded = self._global_local(encoded)
        if self.kind == "part":
            parts = self._gaitpart_parts(encoded)
        else:
            frame_features = self._temporal_set(encoded)
            parts = self.hpp(frame_features)
        if self.kind == "base":
            parts = parts + torch.tanh(parts)
        flat = parts.flatten(1)
        if self.kind == "smpl":
            if smpl_feat is None:
                smpl_feat = torch.zeros(silhouettes.shape[0], 82, device=silhouettes.device)
            body = self.smpl_mlp(smpl_feat)
        else:
            body = torch.zeros(silhouettes.shape[0], 16, device=silhouettes.device)
        return self.head(torch.cat((flat, body), dim=-1))

    def _temporal_set(self, encoded: torch.Tensor) -> torch.Tensor:
        """Apply GaitSet-style frame-order-invariant pooling.

        Parameters
        ----------
        encoded:
            Encoded frame tensor.

        Returns
        -------
        torch.Tensor
            Pooled feature map.
        """

        mean_pool = encoded.mean(dim=1)
        max_pool = encoded.max(dim=1).values
        return 0.5 * (mean_pool + max_pool)

    def _part_temporal(self, parts: torch.Tensor) -> torch.Tensor:
        """Apply GaitPart's temporal micro-motion convolution over parts.

        Parameters
        ----------
        parts:
            Horizontal part descriptors.

        Returns
        -------
        torch.Tensor
            Refined part descriptors.
        """

        batch, part_count, channels = parts.shape
        x = parts.reshape(batch, part_count * channels, 1).repeat(1, 1, 4)
        x = self.part_temporal(x).mean(dim=-1)
        return x.view(batch, part_count, channels)

    def _gaitpart_parts(self, encoded: torch.Tensor) -> torch.Tensor:
        """Apply focal convolution and micro-motion templates per body part.

        Parameters
        ----------
        encoded:
            Encoded frame tensor ``(batch, time, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Per-part descriptors after short-range temporal modeling.
        """
        batch, time, channels, height, width = encoded.shape
        frames = encoded.reshape(batch * time, channels, height, width)
        focal = F.relu(self.focal_small(frames)) + F.relu(self.focal_large(frames))
        frame_parts = self.hpp(focal).view(batch, time, -1, channels)
        part_outputs: list[torch.Tensor] = []
        for part_idx in range(frame_parts.shape[2]):
            sequence = frame_parts[:, :, part_idx, :].transpose(1, 2)
            part_outputs.append(self.micro_motion(sequence).mean(dim=-1))
        return torch.stack(part_outputs, dim=1)

    def _global_local(self, encoded: torch.Tensor) -> torch.Tensor:
        """Fuse GaitGL global and local 3D temporal paths.

        Parameters
        ----------
        encoded:
            Encoded frame tensor.

        Returns
        -------
        torch.Tensor
            Global-local feature tensor.
        """

        x = encoded.transpose(1, 2)
        local = self.local3d(x)
        global_path = self.global3d(x)
        return F.relu(local + global_path).transpose(1, 2)


def _build(kind: GaitName) -> GaitModel:
    """Build a compact gait model.

    Parameters
    ----------
    kind:
        Architecture key.

    Returns
    -------
    GaitModel
        Random-initialized gait model.
    """

    return GaitModel(kind)


def example_silhouettes() -> torch.Tensor:
    """Create a compact silhouette sequence.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(2, 5, 32, 22)``.
    """

    return torch.rand(2, 5, 32, 22)


def example_smpl() -> tuple[torch.Tensor, torch.Tensor]:
    """Create silhouette and SMPL pose/shape inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Silhouette sequence and flattened joint/shape vector.
    """

    return example_silhouettes(), torch.rand(2, 82)


def build_gaitbase() -> GaitModel:
    """Build GaitBase.

    Returns
    -------
    GaitModel
        GaitBase reconstruction.
    """

    return _build("base")


def build_gaitgl() -> GaitModel:
    """Build GaitGL.

    Returns
    -------
    GaitModel
        GaitGL reconstruction.
    """

    return _build("gl")


def build_gaitpart() -> GaitModel:
    """Build GaitPart.

    Returns
    -------
    GaitModel
        GaitPart reconstruction.
    """

    return _build("part")


def build_gaitset() -> GaitModel:
    """Build GaitSet.

    Returns
    -------
    GaitModel
        GaitSet reconstruction.
    """

    return _build("set")


def build_smplgait() -> GaitModel:
    """Build SMPLGait.

    Returns
    -------
    GaitModel
        SMPLGait reconstruction.
    """

    return _build("smpl")


MENAGERIE_ENTRIES = [
    ("GaitBase", "build_gaitbase", "example_silhouettes", "2023", "vision/gait"),
    ("GaitGL", "build_gaitgl", "example_silhouettes", "2021", "vision/gait"),
    ("GaitPart", "build_gaitpart", "example_silhouettes", "2020", "vision/gait"),
    ("GaitSet", "build_gaitset", "example_silhouettes", "2019", "vision/gait"),
    ("SMPLGait", "build_smplgait", "example_smpl", "2022", "vision/gait"),
]
