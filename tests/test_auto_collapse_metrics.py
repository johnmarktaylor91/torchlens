"""Regression coverage for smart module auto-collapse."""

from __future__ import annotations

import os
import re
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torchvision.models as tvm
import torchvision.models.segmentation as tvs

import torchlens as tl
import torchlens.visualization.auto_collapse as auto_collapse
import torchlens.visualization.collapse_optimizer as collapse_optimizer
from torchlens.visualization.auto_collapse import (
    _assert_plan_count,
    _child_condensed_flow_graphs,
    _flow_graph_for_sibling_group,
    _iter_collapsible_runs,
    _run_fold_is_legal,
    _sibling_address_groups,
    analyze_collapse,
    resolve_collapse_fn,
    resolve_run_folds,
)
from torchlens.visualization.collapse_optimizer import select_collapse_plan
from torchlens.visualization.collapse_plan import (
    CollapsePlan,
    OpSegment,
    RawOp,
    RenderContext,
    count,
    plan_from_v1,
)
from torchlens.visualization._render_edges import _collapsed_module_should_show_remainder


SVG_NODE_RE = re.compile(r'class="node"')


class ResidualBlock(torch.nn.Module):
    """Small residual block with enough internal structure to collapse."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the block."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block."""

        return self.net(x) + x


class RepeatedResidual(torch.nn.Module):
    """Repeated residual blocks for peer and budget tests."""

    def __init__(self, depth: int = 8, width: int = 8) -> None:
        """Initialize repeated residual blocks."""

        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualBlock(width) for _ in range(depth)])
        self.out = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the repeated residual model."""

        for block in self.blocks:
            x = block(x)
        return self.out(x)


class VariableResidualBlock(torch.nn.Module):
    """Residual block whose internal depth is constructor-controlled."""

    def __init__(self, width: int = 8, extra_layers: int = 0) -> None:
        """Initialize a block with ``extra_layers`` additional Linear+ReLU pairs."""

        super().__init__()
        layers: list[torch.nn.Module] = [torch.nn.Linear(width, width), torch.nn.ReLU()]
        for _ in range(extra_layers):
            layers.extend([torch.nn.Linear(width, width), torch.nn.ReLU()])
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.net(x) + x


class OddHiddenMemberStack(torch.nn.Module):
    """Same-class residual run with one structurally-odd block at a hidden position.

    All blocks share ``class_name`` and output shape (so they are eligible to
    group into one run), but ``odd_index`` has extra internal layers/params
    -- a genuinely different module that must never be silently folded into
    a "+N more" box as a hidden member.
    """

    def __init__(self, total: int = 7, odd_index: int = 3, width: int = 8) -> None:
        """Initialize ``total`` blocks, all uniform except ``odd_index``."""

        super().__init__()
        self.blocks = torch.nn.ModuleList(
            VariableResidualBlock(width=width, extra_layers=(2 if i == odd_index else 0))
            for i in range(total)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all blocks in sequence."""

        for block in self.blocks:
            x = block(x)
        return x


class DimStepBlock(torch.nn.Module):
    """Convolutional block whose channel dimensions may change."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize a dim-stepping convolutional block."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.net(x)


class DimStepRun(torch.nn.Module):
    """Run of structurally identical blocks with varying channel widths."""

    def __init__(self, depth: int = 4, start_width: int = 4) -> None:
        """Initialize the dim-stepping run."""

        super().__init__()
        widths = list(range(start_width, start_width + depth + 1))
        self.blocks = torch.nn.ModuleList(
            DimStepBlock(in_channels, out_channels)
            for in_channels, out_channels in zip(widths[:-1], widths[1:], strict=True)
        )
        self.head = torch.nn.Conv2d(widths[-1], widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all dim-stepping blocks."""

        for block in self.blocks:
            x = block(x)
        return self.head(x)


class MobileNetPlateauBlock(torch.nn.Module):
    """MobileNetV2-style block with optional residual join."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize a pointwise-depthwise-pointwise block."""

        super().__init__()
        self.use_residual = in_channels == out_channels
        hidden_channels = max(in_channels, out_channels)
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_channels, 1),
            torch.nn.BatchNorm2d(hidden_channels),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
            torch.nn.BatchNorm2d(hidden_channels),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(hidden_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        y = self.net(x)
        if self.use_residual:
            return x + y
        return y


class MobileNetPlateauStack(torch.nn.Module):
    """Synthetic MobileNetV2 stage stack with channel plateaus."""

    def __init__(self) -> None:
        """Initialize two foldable channel plateaus."""

        super().__init__()
        self.stem = torch.nn.Conv2d(3, 32, 1)
        channels = [32, 32, 32, 64, 64, 64, 64]
        in_channels = [32, *channels[:-1]]
        self.features = torch.nn.ModuleList(
            MobileNetPlateauBlock(in_ch, out_ch)
            for in_ch, out_ch in zip(in_channels, channels, strict=True)
        )
        self.head = torch.nn.Conv2d(64, 64, 1)
        self.tail = torch.nn.ModuleList(torch.nn.ReLU() for _ in range(48))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all synthetic inverted residual blocks."""

        x = self.stem(x)
        for block in self.features:
            x = block(x)
        x = self.head(x)
        for relu in self.tail:
            x = relu(x)
        return x


class BatchNormFlowBlock(torch.nn.Module):
    """Convolutional child block with registered BatchNorm buffers."""

    def __init__(self) -> None:
        """Initialize the block."""

        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.relu(self.bn(self.conv(x)))


class BatchNormFlowStack(torch.nn.Module):
    """Repeated BatchNorm blocks for condensed-flow buffer-edge regression."""

    def __init__(self, depth: int = 4) -> None:
        """Initialize repeated BatchNorm blocks."""

        super().__init__()
        self.blocks = torch.nn.Sequential(*(BatchNormFlowBlock() for _ in range(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the sequential BatchNorm stack."""

        return self.blocks(x)


class StageUnit(torch.nn.Module):
    """Small convolutional unit used to build synthetic stages."""

    def __init__(self, width: int) -> None:
        """Initialize the unit."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.BatchNorm2d(width),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the unit."""

        return self.net(x)


class DepthStage(torch.nn.Module):
    """Same-class stage whose depth is constructor-controlled."""

    def __init__(self, width: int, depth: int) -> None:
        """Initialize a stage with ``depth`` units."""

        super().__init__()
        self.blocks = torch.nn.ModuleList([StageUnit(width) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all units in the stage."""

        for block in self.blocks:
            x = block(x)
        return x


class UnevenDepthStages(torch.nn.Module):
    """Same-class sibling stages with different internal depths."""

    def __init__(self, depths: tuple[int, ...] = (2, 3, 4)) -> None:
        """Initialize stages of depths two, three, and four."""

        super().__init__()
        self.stem = torch.nn.Conv2d(4, 4, 1)
        self.stages = torch.nn.ModuleList([DepthStage(4, depth) for depth in depths])
        self.head = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the uneven-depth stages."""

        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


class SpatialStepBlock(torch.nn.Module):
    """Structurally identical block that changes spatial resolution."""

    def __init__(self, width: int) -> None:
        """Initialize the spatial-step block."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.BatchNorm2d(width),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the spatial-step block."""

        return self.net(x)


class SpatialStepRun(torch.nn.Module):
    """Run that should not fold because it crosses spatial scales."""

    def __init__(self) -> None:
        """Initialize repeated spatial-step blocks."""

        super().__init__()
        self.blocks = torch.nn.ModuleList([SpatialStepBlock(4) for _ in range(3)])
        self.head = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all spatial-step blocks."""

        for block in self.blocks:
            x = block(x)
        return self.head(x)


class VggBnFeatures(torch.nn.Module):
    """Flat VGG-style conv-bn-relu-pool features container."""

    def __init__(self) -> None:
        """Initialize a small VGG-BN-style feature extractor."""

        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(4, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Linear(8 * 4 * 4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run features and classifier."""

        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


class ResidualBody(torch.nn.Module):
    """Residual branch body whose join lives in the parent module."""

    def __init__(self, width: int = 8) -> None:
        """Initialize residual body layers."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual branch body."""

        return self.net(x)


class ParentJoinResidual(torch.nn.Module):
    """Repeated residual stages with parent-level add junctions."""

    def __init__(self, depth: int = 5, width: int = 8) -> None:
        """Initialize repeated residual stages."""

        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualBody(width) for _ in range(depth)])
        self.out = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual stack."""

        for block in self.blocks:
            x = block(x) + x
        return self.out(x)


class ConvReluBlock(torch.nn.Module):
    """Small convolutional block without buffer-only bookkeeping."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize convolutional layers."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.net(x)


class SkipConcatUNet(torch.nn.Module):
    """Tiny U-Net-style model with a cross-module concat junction."""

    def __init__(self) -> None:
        """Initialize encoder and decoder blocks."""

        super().__init__()
        self.enc1 = ConvReluBlock(4, 4)
        self.enc2 = ConvReluBlock(4, 4)
        self.dec1 = ConvReluBlock(8, 4)
        self.out = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the skip-concat model."""

        skip = self.enc1(x)
        deep = torch.nn.functional.avg_pool2d(self.enc2(skip), 2)
        up = torch.nn.functional.interpolate(deep, scale_factor=2, mode="nearest")
        return self.out(self.dec1(torch.cat([skip, up], dim=1)))


class BranchConcat(torch.nn.Module):
    """Inception-like parallel branches with a parent-level concat."""

    def __init__(self) -> None:
        """Initialize branch modules."""

        super().__init__()
        self.a = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 1),
        )
        self.b = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 1),
        )
        self.c = torch.nn.Sequential(
            torch.nn.AvgPool2d(3, stride=1, padding=1),
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all branches and concatenate their outputs."""

        return torch.cat([self.a(x), self.b(x), self.c(x)], dim=1)


class OwnOutputResidualBlock(torch.nn.Module):
    """Residual block whose add is the module's own output op."""

    def __init__(self) -> None:
        """Initialize the residual block."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 1),
            torch.nn.BatchNorm2d(8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block."""

        return x + self.net(x)


class OwnOutputResidualModel(torch.nn.Module):
    """Wrapper model exposing one own-output residual block."""

    def __init__(self) -> None:
        """Initialize the residual wrapper."""

        super().__init__()
        self.pre = torch.nn.Conv2d(3, 8, 1)
        self.block = OwnOutputResidualBlock()
        self.out = torch.nn.Conv2d(8, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual wrapper."""

        return self.out(self.block(self.pre(x)))


class OwnOutputConcatBlock(torch.nn.Module):
    """Concat block whose cat is the module's own output op."""

    def __init__(self) -> None:
        """Initialize the concat block."""

        super().__init__()
        self.a = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 1),
        )
        self.b = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 1),
        )
        self.c = torch.nn.Sequential(
            torch.nn.AvgPool2d(3, stride=1, padding=1),
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the concat block."""

        return torch.cat([self.a(x), self.b(x), self.c(x)], dim=1)


class OwnOutputConcatModel(torch.nn.Module):
    """Wrapper model exposing one own-output concat block."""

    def __init__(self) -> None:
        """Initialize the concat wrapper."""

        super().__init__()
        self.pre = torch.nn.Conv2d(3, 4, 1)
        self.block = OwnOutputConcatBlock()
        self.out = torch.nn.Conv2d(12, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the concat wrapper."""

        return self.out(self.block(self.pre(x)))


class ParallelRepeatedBranches(torch.nn.Module):
    """Many identical parallel branches feeding one concat junction."""

    def __init__(self, depth: int = 24) -> None:
        """Initialize repeated branch modules."""

        super().__init__()
        self.branches = torch.nn.ModuleList([ConvReluBlock(4, 4) for _ in range(depth)])
        self.out = torch.nn.Conv2d(depth * 4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run branches in parallel and concatenate their outputs."""

        return self.out(torch.cat([branch(x) for branch in self.branches], dim=1))


class NestedStage(torch.nn.Module):
    """Stage block used inside a generic nested backbone container."""

    def __init__(self, width: int) -> None:
        """Initialize a two-convolution stage."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stage."""

        return self.net(x)


class NestedStageBackbone(torch.nn.Module):
    """Model whose repeated stages live under a generic container."""

    def __init__(self) -> None:
        """Initialize a stem, nested backbone, and head."""

        super().__init__()
        self.stem = torch.nn.Conv2d(4, 4, 1)
        self.backbone = torch.nn.Sequential(*(NestedStage(4) for _ in range(4)))
        self.head = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested-stage backbone."""

        return self.head(self.backbone(self.stem(x)))


class BackboneContainer(torch.nn.Module):
    """Backbone container with direct stem and repeated stage children."""

    def __init__(self) -> None:
        """Initialize the nested backbone container."""

        super().__init__()
        self.stem = torch.nn.Conv2d(4, 4, 1)
        self.stages = torch.nn.ModuleList([NestedStage(4) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stem and repeated stages."""

        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class BackboneContainerModel(torch.nn.Module):
    """Model whose collapsed children should remain inside ``backbone``."""

    def __init__(self) -> None:
        """Initialize the container model."""

        super().__init__()
        self.backbone = BackboneContainer()
        self.head = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the container model."""

        return self.head(self.backbone(x))


class StemStageBackbone(torch.nn.Module):
    """Backbone with a standalone leaf-block stem and repeated stage blocks."""

    def __init__(self) -> None:
        """Initialize a leaf-block stem, stages, and leaf-block head."""

        super().__init__()
        self.stem = ConvNormRelu(4, 4, 3)
        self.stages = torch.nn.ModuleList([ConvNormRelu(4, 4, 3) for _ in range(4)])
        self.head = ConvNormRelu(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stem, repeated stages, and head."""

        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


class ConvNormRelu(torch.nn.Module):
    """Fixed conv-batchnorm-relu leaf chain."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Initialize the chain."""

        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run convolution, normalization, and activation."""

        return torch.relu(self.bn(self.conv(x)))


class UniqueMixed(torch.nn.Module):
    """Inception-like module with unique branch internals."""

    def __init__(self, width: int, kernel_size: int) -> None:
        """Initialize parallel branches with varying structure."""

        super().__init__()
        self.a = ConvNormRelu(width, width, 1)
        self.b = torch.nn.Sequential(
            ConvNormRelu(width, width, 1),
            ConvNormRelu(width, width, kernel_size),
        )
        self.c = torch.nn.Sequential(
            torch.nn.AvgPool2d(3, stride=1, padding=1),
            ConvNormRelu(width, width, 1),
        )
        self.project = torch.nn.Conv2d(width * 3, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the parallel branches and project the concatenation."""

        return self.project(torch.cat([self.a(x), self.b(x), self.c(x)], dim=1))


class UniqueParallelStack(torch.nn.Module):
    """Stack of same-role but structurally non-identical mixed modules."""

    def __init__(self) -> None:
        """Initialize mixed modules with different kernels."""

        super().__init__()
        self.mixed_5b = UniqueMixed(4, 3)
        self.mixed_5c = UniqueMixed(4, 5)
        self.mixed_5d = UniqueMixed(4, 3)
        self.mixed_6a = UniqueMixed(4, 5)
        self.head = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the mixed-module stack."""

        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        x = self.mixed_5d(x)
        x = self.mixed_6a(x)
        return self.head(x)


class RecurrentWrapper(torch.nn.Module):
    """GRU wrapper for recurrent module collapse coverage."""

    def __init__(self) -> None:
        """Initialize recurrent layers."""

        super().__init__()
        self.rnn = torch.nn.GRU(8, 8, batch_first=True)
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run recurrent model."""

        y, _ = self.rnn(x)
        return self.head(y[:, -1])


class UnevenReusedBlock(torch.nn.Module):
    """Small reused block for rolled recurrence-label tests."""

    def __init__(self, width: int = 4) -> None:
        """Initialize the block."""

        super().__init__()
        self.fc = torch.nn.Linear(width, width)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.relu(self.fc(x))


class UnevenReusedSiblings(torch.nn.Module):
    """Digest-identical sibling modules called different numbers of times."""

    def __init__(self, width: int = 4) -> None:
        """Initialize the siblings."""

        super().__init__()
        self.short = UnevenReusedBlock(width)
        self.long = UnevenReusedBlock(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run sibling modules with different recurrence counts."""

        for _ in range(3):
            x = self.short(x)
        for _ in range(5):
            x = self.long(x)
        return x


class TrivialSingle(torch.nn.Module):
    """Single-op model whose only submodule should not be collapse-eligible."""

    def __init__(self) -> None:
        """Initialize the single-op module."""

        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the trivial model."""

        return self.relu(x)


class FlowUnit(torch.nn.Module):
    """Single child unit for flow-order graph tests."""

    def __init__(self) -> None:
        """Initialize a pointwise convolution."""

        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the unit."""

        return torch.relu(self.conv(x))


class RegistrationInterleavedAuxParent(torch.nn.Module):
    """GoogLeNet-style parent whose registration order differs from flow order."""

    def __init__(self) -> None:
        """Register an aux branch in the middle of trunk children."""

        super().__init__()
        self.trunk0 = FlowUnit()
        self.aux = FlowUnit()
        self.trunk1 = FlowUnit()
        self.trunk2 = FlowUnit()
        self.project = torch.nn.Conv2d(8, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run trunk first, then an aux branch from the early activation."""

        stem = self.trunk0(x)
        trunk = self.trunk2(self.trunk1(stem))
        aux = self.aux(stem)
        return self.project(torch.cat([trunk, aux], dim=1))


class LongFunctional(torch.nn.Module):
    """Large op-count model for signal-tally latency coverage."""

    def __init__(self, depth: int = 3000) -> None:
        """Initialize the operation depth."""

        super().__init__()
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run many functional operations."""

        for _ in range(self.depth):
            x = torch.relu(x + 1.0)
        return x


class DistinctFunctionalChain(torch.nn.Module):
    """Flat straight-line chain of distinct functional operations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a long non-recurrent functional chain."""

        x = x + 1.0
        x = torch.relu(x)
        x = x * 1.1
        x = torch.sigmoid(x)
        x = x - 0.2
        x = torch.tanh(x)
        x = x / 1.1
        x = torch.exp(x)
        x = torch.log(torch.abs(x) + 1.0)
        x = torch.sqrt(torch.abs(x) + 1.0)
        x = torch.square(x)
        x = -x
        x = torch.sin(x)
        x = torch.cos(x)
        x = x + 2.0
        x = torch.relu(x)
        x = x * 0.9
        x = torch.sigmoid(x)
        x = x - 0.1
        x = torch.tanh(x)
        x = x / 0.9
        x = torch.exp(x)
        x = torch.log(torch.abs(x) + 1.0)
        x = torch.sqrt(torch.abs(x) + 1.0)
        x = torch.square(x)
        x = -x
        x = torch.sin(x)
        x = torch.cos(x)
        x = x + 3.0
        x = torch.relu(x)
        x = x * 0.7
        x = torch.sigmoid(x)
        x = x - 0.3
        x = torch.tanh(x)
        x = x / 0.7
        x = torch.exp(x)
        x = torch.log(torch.abs(x) + 1.0)
        x = torch.sqrt(torch.abs(x) + 1.0)
        x = torch.square(x)
        x = -x
        x = torch.sin(x)
        return torch.cos(x)


class SegmentToyBlock(torch.nn.Module):
    """Small block used by max-mode segment rendering tests."""

    def __init__(self) -> None:
        """Initialize the block layers."""

        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block forward pass."""

        return self.relu(self.conv(x))


class SegmentToyNet(torch.nn.Module):
    """Sequential toy net with a long raw-op run for max condensation."""

    def __init__(self) -> None:
        """Initialize the segment toy network."""

        super().__init__()
        self.stem = torch.nn.Conv2d(3, 4, 1)
        self.blocks = torch.nn.Sequential(*(SegmentToyBlock() for _ in range(5)))
        self.head = torch.nn.Conv2d(4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the toy network forward pass."""

        return self.head(self.blocks(self.stem(x)))


def _trace(model: torch.nn.Module, x: torch.Tensor) -> tl.Trace:
    """Capture ``model`` under ``torch.no_grad``."""

    with torch.no_grad():
        return tl.trace(model.eval(), x)


def _draw_source(
    trace: tl.Trace,
    tmp_path: Path,
    name: str,
    collapse: str,
    fold_runs: bool | None = None,
) -> str:
    """Render a trace to SVG and return DOT source."""

    return str(
        trace.draw(
            vis_outpath=str(tmp_path / name),
            vis_save_only=True,
            vis_fileformat="svg",
            vis_node_placement="dot",
            collapse=collapse,
            fold_runs=fold_runs,
        )
    )


def _draw_source_mode(
    trace: tl.Trace,
    tmp_path: Path,
    name: str,
    collapse: str,
    vis_mode: str,
) -> str:
    """Render a trace in a specific visualization mode and return DOT source.

    Parameters
    ----------
    trace:
        Trace to render.
    tmp_path:
        Temporary output directory.
    name:
        Output file stem.
    collapse:
        Collapse mode.
    vis_mode:
        Visualization mode.

    Returns
    -------
    str
        DOT source returned by ``Trace.draw``.
    """

    return str(
        trace.draw(
            vis_mode=vis_mode,  # type: ignore[arg-type]
            vis_outpath=str(tmp_path / name),
            vis_save_only=True,
            vis_fileformat="svg",
            vis_node_placement="dot",
            collapse=collapse,  # type: ignore[arg-type]
        )
    )


def _landmark_swallow_count(trace: tl.Trace, result: Any) -> int:
    """Return the number of selected boxes or child segments hiding landmarks.

    Parameters
    ----------
    trace:
        Trace being inspected.
    result:
        Optimizer result to inspect.

    Returns
    -------
    int
        Number of selected landmark-hiding module boxes or child segments.
    """

    analysis = analyze_collapse(trace)
    total = sum(
        1
        for address in getattr(result, "selected", ())
        if analysis.signals.get(address) is not None
        and analysis.signals[address].landmark_edges >= 2
    )
    for segment in (getattr(result, "segments", {}) or {}).values():
        if getattr(segment, "kind", "") != "child":
            continue
        for address in getattr(segment, "members", ()):
            signal = analysis.signals.get(address)
            if signal is not None and signal.landmark_edges >= 2:
                total += 1
    return total


def _svg_node_count(path: Path) -> int:
    """Return the Graphviz SVG node count.

    Parameters
    ----------
    path:
        SVG file path.

    Returns
    -------
    int
        Number of rendered node groups.
    """

    return len(SVG_NODE_RE.findall(path.read_text(encoding="utf-8")))


def _svg_collapsed_box_layer_labels(path: Path) -> list[str]:
    """Return collapsed-box layer-count labels from a Graphviz SVG.

    Parameters
    ----------
    path:
        SVG file path.

    Returns
    -------
    list[str]
        Labels such as ``"20 layers total"`` in SVG order.
    """

    text = path.read_text(encoding="utf-8")
    return re.findall(r">([0-9]+ layers total)<", text)


def _draw_svg_layer_labels(
    trace: tl.Trace,
    tmp_path: Path,
    name: str,
    collapse: Any,
) -> list[str]:
    """Render ``trace`` and return collapsed-box layer labels from SVG output.

    Parameters
    ----------
    trace:
        Trace to render.
    tmp_path:
        Temporary output directory.
    name:
        Output filename stem.
    collapse:
        Public collapse mode or float level.

    Returns
    -------
    list[str]
        Collapsed-box layer-count labels in SVG order.
    """

    out = tmp_path / name
    trace.draw(
        vis_outpath=str(out),
        vis_save_only=True,
        vis_fileformat="svg",
        vis_node_placement="dot",
        collapse=collapse,
    )
    return _svg_collapsed_box_layer_labels(out.with_suffix(".svg"))


def _assert_plan_svg_parity(
    trace: tl.Trace,
    tmp_path: Path,
    name: str,
    mode: str,
    vis_mode: str = "unrolled",
) -> None:
    """Assert v1 plan count equals rendered SVG node count.

    Parameters
    ----------
    trace:
        Trace to render.
    tmp_path:
        Pytest temporary directory.
    name:
        Output filename stem.
    mode:
        Collapse mode.
    vis_mode:
        Visualization mode.
    """

    context = RenderContext(vis_mode=vis_mode)  # type: ignore[arg-type]
    collapse_fn = resolve_collapse_fn(trace, mode, vis_mode, context=context)  # type: ignore[arg-type]
    folds = resolve_run_folds(trace, collapse_fn, context=context)
    v2_plan = getattr(collapse_fn, "_torchlens_v2_plan", None)
    plan_count = (
        count(v2_plan)
        if v2_plan is not None
        else count(plan_from_v1(trace, collapse_fn, folds, context))
    )
    out = tmp_path / name
    trace.draw(
        vis_mode=vis_mode,  # type: ignore[arg-type]
        vis_outpath=str(out),
        vis_save_only=True,
        vis_fileformat="svg",
        vis_node_placement="dot",
        collapse=mode,  # type: ignore[arg-type]
        show_containers=False,
    )
    assert plan_count == _svg_node_count(out.with_suffix(".svg"))


def _box_count(source: str) -> int:
    """Return collapsed module node count from DOT source."""

    return source.count("shape=box3d")


def _dot_node_count(source: str) -> int:
    """Return an approximate rendered node count from DOT source."""

    names = re.findall(
        r'^\s*("[^"]+"|[A-Za-z0-9_.]+(?:pass\d+)?) \[',
        source,
        flags=re.MULTILINE,
    )
    return len([name for name in names if name not in {"graph", "node", "edge"}])


def _has_visible_node(source: str, prefix: str) -> bool:
    """Return whether DOT source contains a visible node with ``prefix``.

    Parameters
    ----------
    source:
        DOT source emitted by a render.
    prefix:
        Node identifier prefix to find.

    Returns
    -------
    bool
        True when an explicit node declaration starts with ``prefix``.
    """

    pattern = rf'^\s*"?{re.escape(prefix)}[A-Za-z0-9_.]*"? \['
    return re.search(pattern, source, flags=re.MULTILINE) is not None


def _collapsed_label_count(source: str, prefix: str) -> int:
    """Return collapsed module label count with ``prefix``."""

    return source.count(f"<B>@{prefix}")


def _collapsed_exact_label_count(source: str, address: str) -> int:
    """Return collapsed module label count for exactly ``address``."""

    return len(
        [
            line
            for line in source.splitlines()
            if "shape=box3d" in line and f"<B>@{address}</B>" in line
        ]
    )


def _collapsed_node_line(source: str, address: str) -> str:
    """Return the DOT node declaration for one collapsed module address."""

    for line in source.splitlines():
        if "shape=box3d" in line and f"<B>@{address}</B>" in line:
            return line
    raise AssertionError(f"missing collapsed module node for {address!r}")


def _atomic_own_output_ops(trace: tl.Trace, address: str) -> tuple[Any, ...]:
    """Return atomic own-output ops for ``address`` in trace order."""

    surfaced: list[Any] = []
    for label in trace.modules[address].layer_labels:
        op = trace.ops[label]
        modules = list(getattr(op, "modules", ()) or ())
        if (
            modules
            and modules[-1].rsplit(":", 1)[0] == address
            and getattr(op, "is_atomic_module", False)
            and not getattr(op, "is_buffer", False)
        ):
            surfaced.append(op)
    return tuple(surfaced)


def _run_fold_ellipsis_name(address: str) -> str:
    """Return the deterministic run-fold ellipsis node name for ``address``."""

    return f"{address}pass1___runfoldellipsis"


def _run_fold_ellipsis_count(source: str, multiplicity: int) -> int:
    """Return count of run-fold ellipsis labels for ``multiplicity`` folded modules."""

    return source.count(f"... +{multiplicity - 1} more ")


def _has_run_fold_multiplicity_label(source: str, multiplicity: int) -> bool:
    """Return whether DOT source contains the old run-fold ``xN`` label."""

    return bool(re.search(rf"\bx{multiplicity}\b", source))


def _select_first_stage_unit(module: object) -> bool:
    """Return whether ``module`` is the first unit inside a synthetic stage."""

    return re.match(r"^stages\.\d+\.blocks\.0$", str(getattr(module, "address", ""))) is not None


def _select_blocks_child(module: object) -> bool:
    """Return whether ``module`` is a direct child of a ``blocks`` container."""

    return re.match(r"^blocks\.\d+$", str(getattr(module, "address", ""))) is not None


def _select_features_child(module: object) -> bool:
    """Return whether ``module`` is a direct child of a ``features`` container."""

    return re.match(r"^features\.\d+$", str(getattr(module, "address", ""))) is not None


def _select_branches_child(module: object) -> bool:
    """Return whether ``module`` is a direct child of a ``branches`` container."""

    return re.match(r"^branches\.\d+$", str(getattr(module, "address", ""))) is not None


def _select_backbone_stage(module: object) -> bool:
    """Return whether ``module`` is a direct stage child of ``backbone``."""

    return re.match(r"^backbone\.stages\.\d+$", str(getattr(module, "address", ""))) is not None


def _edge_count(source: str, tail: str, head: str) -> int:
    """Return Graphviz edge count between two rendered node names."""

    return len(
        re.findall(
            rf'^\s*"?{re.escape(tail)}"? -> "?{re.escape(head)}"?\s+\[',
            source,
            flags=re.MULTILINE,
        )
    )


def _incoming_edge_count(source: str, head: str) -> int:
    """Return Graphviz edge count into one rendered node."""

    return len(
        re.findall(
            rf'^\s*"?[^"]+"? -> "?{re.escape(head)}"?\s+\[',
            source,
            flags=re.MULTILINE,
        )
    )


def _outgoing_edge_count(source: str, tail: str) -> int:
    """Return Graphviz edge count out of one rendered node."""

    return len(
        re.findall(
            rf'^\s*"?{re.escape(tail)}"? -> "?[^"]+"?\s+\[',
            source,
            flags=re.MULTILINE,
        )
    )


def _cluster_body(source: str, cluster_name: str) -> str:
    """Return the DOT body for one named subgraph cluster."""

    markers = (f'subgraph "{cluster_name}" {{', f"subgraph {cluster_name} {{")
    marker = next((candidate for candidate in markers if candidate in source), None)
    if marker is None:
        raise ValueError(f"Cluster {cluster_name!r} is not present")
    start = source.index(marker)
    body_start = start + len(marker)
    depth = 1
    index = body_start
    while index < len(source):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[body_start:index]
        index += 1
    raise ValueError(f"Cluster {cluster_name!r} is not closed")


def test_collapse_plan_parity_fast_synthetic_models(tmp_path: Path) -> None:
    """CollapsePlan count matches SVG nodes on fast synthetic collapse fixtures."""

    cases = [
        ("residual", RepeatedResidual(depth=8), torch.randn(2, 8)),
        ("branches", ParallelRepeatedBranches(depth=8), torch.randn(1, 4, 8, 8)),
    ]
    for case_name, model, x in cases:
        trace = _trace(model, x)
        try:
            for mode in ("auto", "max"):
                _assert_plan_svg_parity(trace, tmp_path, f"{case_name}_{mode}", mode)
        finally:
            trace.cleanup()


def test_child_condensed_flow_graph_uses_execution_order_for_aux_branch() -> None:
    """Flow graph orders children by execution and keeps aux outside trunk chain."""

    model = RegistrationInterleavedAuxParent()
    assert tuple(model._modules) == ("trunk0", "aux", "trunk1", "trunk2", "project")
    trace = _trace(model, torch.randn(1, 4, 8, 8))
    try:
        graph = _child_condensed_flow_graphs(trace)["self"]
        assert graph.flow_children == ("trunk0", "trunk1", "trunk2", "aux", "project")
        assert ("trunk0", "trunk1") in graph.edges
        assert ("trunk1", "trunk2") in graph.edges
        assert ("trunk0", "aux") in graph.edges
        assert ("trunk2", "aux") not in graph.edges
        assert ("aux", "trunk1") not in graph.edges
        assert graph.flow_children.index("aux") != graph.flow_children.index("trunk1") - 1
    finally:
        trace.cleanup()


def test_child_condensed_flow_graph_ignores_batchnorm_buffer_edges() -> None:
    """BatchNorm buffer provenance does not create inter-child flow edges."""

    trace = _trace(BatchNormFlowStack(depth=4), torch.randn(1, 4, 8, 8))
    try:
        graph = _child_condensed_flow_graphs(trace)["blocks"]
        child_set = set(graph.flow_children)
        inter_child_edges = tuple(
            edge for edge in graph.edges if edge[0] in child_set and edge[1] in child_set
        )

        assert graph.flow_children == tuple(f"blocks.{index}" for index in range(4))
        assert inter_child_edges == tuple(
            (f"blocks.{index}", f"blocks.{index + 1}") for index in range(3)
        )
        assert graph.child_external_endpoint_counts["blocks.0"][0] == 1
        assert graph.child_external_endpoint_counts["blocks.3"][1] == 1
    finally:
        trace.cleanup()


@pytest.mark.heavy
def test_child_condensed_flow_graph_mobilenet_v2_features_exact_chain() -> None:
    """MobileNetV2 features children form one exact forward dataflow chain."""

    trace = _trace(tvm.mobilenet_v2(weights=None), torch.randn(1, 3, 224, 224))
    try:
        graph = _child_condensed_flow_graphs(trace)["features"]
        child_set = set(graph.flow_children)
        inter_child_edges = tuple(
            edge for edge in graph.edges if edge[0] in child_set and edge[1] in child_set
        )

        assert graph.flow_children == tuple(f"features.{index}" for index in range(19))
        assert inter_child_edges == tuple(
            (f"features.{index}", f"features.{index + 1}") for index in range(18)
        )
        assert graph.child_external_endpoint_counts["features.0"][0] == 1
        assert graph.child_external_endpoint_counts["features.18"][1] == 1
    finally:
        trace.cleanup()


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("name", "builder", "x"),
    [
        ("resnet50", lambda: tvm.resnet50(weights=None), torch.randn(1, 3, 224, 224)),
        ("vit_b_16", lambda: tvm.vit_b_16(weights=None), torch.randn(1, 3, 224, 224)),
        ("swin_s", lambda: tvm.swin_s(weights=None), torch.randn(1, 3, 224, 224)),
        ("mobilenet_v2", lambda: tvm.mobilenet_v2(weights=None), torch.randn(1, 3, 224, 224)),
        (
            "deeplabv3_resnet50",
            lambda: tvs.deeplabv3_resnet50(weights=None, weights_backbone=None),
            torch.randn(1, 3, 224, 224),
        ),
        (
            "googlenet",
            lambda: tvm.googlenet(weights=None, aux_logits=False, init_weights=False),
            torch.randn(1, 3, 224, 224),
        ),
    ],
)
def test_collapse_plan_parity_heavy_torchvision(
    tmp_path: Path,
    name: str,
    builder: object,
    x: torch.Tensor,
) -> None:
    """CollapsePlan count matches SVG nodes on representative torchvision models."""

    model = builder()
    trace = _trace(model, x)  # type: ignore[arg-type]
    try:
        for mode in ("auto", "max"):
            _assert_plan_svg_parity(trace, tmp_path, f"{name}_{mode}", mode)
    finally:
        trace.cleanup()


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("name", "builder", "x"),
    [
        ("resnet50", lambda: tvm.resnet50(weights=None), torch.randn(1, 3, 224, 224)),
        ("vit_b_16", lambda: tvm.vit_b_16(weights=None), torch.randn(1, 3, 224, 224)),
        ("gru_toy", RecurrentWrapper, torch.randn(1, 6, 8)),
    ],
)
def test_rolled_collapse_plan_parity_requested_models(
    tmp_path: Path,
    name: str,
    builder: object,
    x: torch.Tensor,
) -> None:
    """Rolled auto plan count matches SVG nodes on requested R3c models."""

    model = builder()
    trace = _trace(model, x)  # type: ignore[arg-type]
    try:
        _assert_plan_svg_parity(trace, tmp_path, f"{name}_rolled_auto", "auto", "rolled")
    finally:
        trace.cleanup()


def test_rolled_auto_distinct_recurrence_labels_for_uneven_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rolled labels keep ``(xN)`` strictly tied to recurrence counts."""

    trace = _trace(UnevenReusedSiblings(), torch.randn(2, 4))
    try:
        source = _draw_source_mode(trace, tmp_path, "uneven_rolled_auto", "auto", "rolled")

        assert "@short (x3)" in source
        assert "@long (x5)" in source
        assert "... +3 more" not in source
        assert "... +5 more" not in source
    finally:
        trace.cleanup()


def test_auto_collapse_budget_boxes_grain_and_determinism(tmp_path: Path) -> None:
    """Auto collapse hits the overview budget and renders deterministically."""

    trace = _trace(RepeatedResidual(depth=8), torch.randn(2, 8))
    try:
        none_source = _draw_source(trace, tmp_path, "none", "none")
        auto_source = _draw_source(trace, tmp_path, "auto1", "auto")
        auto_source_again = _draw_source(trace, tmp_path, "auto2", "auto")
        max_source = _draw_source(trace, tmp_path, "max", "max")

        assert auto_source == auto_source_again
        assert _box_count(none_source) == 0
        assert _box_count(auto_source) >= 1
        assert _box_count(max_source) >= _box_count(auto_source)
        assert _dot_node_count(auto_source) <= _dot_node_count(none_source)
        assert 4 <= _dot_node_count(auto_source) <= 40

        collapsed_scores = [score for _, score in trace.module_collapse_order if score > 0]
        assert collapsed_scores
        assert trace.module_collapse_order == sorted(
            trace.module_collapse_order,
            key=lambda item: (-item[1], item[0]),
        )

        selected_sizes = [
            analyze_collapse(trace).signals[address].hidden_ops
            for address, score in trace.module_collapse_order
            if score > 0
        ]
        assert max(selected_sizes) - min(selected_sizes) <= max(selected_sizes)
        assert "input_" in auto_source
        assert "output_" in auto_source
    finally:
        trace.cleanup()


def test_auto_collapse_enables_global_ranking_for_collapsed_layout(tmp_path: Path) -> None:
    """Collapsed DOT renders enable global ranking to keep sequential boxes ordered."""

    trace = _trace(RepeatedResidual(depth=8), torch.randn(2, 8))
    try:
        none_source = _draw_source(trace, tmp_path, "rank_none", "none")
        auto_source = _draw_source(trace, tmp_path, "rank_auto", "auto")

        assert "newrank=true" not in none_source
        assert "newrank=true" in auto_source
    finally:
        trace.cleanup()


def test_auto_collapse_places_collapsed_children_inside_parent_cluster(tmp_path: Path) -> None:
    """Collapsed child module nodes are emitted inside their parent cluster."""

    trace = _trace(BackboneContainerModel(), torch.randn(1, 4, 16, 16))
    try:
        auto_source = str(
            trace.draw(
                vis_outpath=str(tmp_path / "nested_stage_cluster_auto"),
                vis_save_only=True,
                vis_fileformat="svg",
                vis_node_placement="dot",
                collapse="auto",
                collapse_fn=_select_backbone_stage,
            )
        )
        backbone_body = _cluster_body(auto_source, "cluster_backbone_pass1")

        assert '"backbone.stages.0pass1" [' in backbone_body
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_representative_uses_single_instance_stats(
    tmp_path: Path,
) -> None:
    """Auto-mode chain fold representatives display single-instance stats."""

    trace = _trace(RepeatedResidual(depth=24), torch.randn(2, 8))
    try:
        collapse_fn = resolve_collapse_fn(trace, "auto", "unrolled")
        folds = resolve_run_folds(trace, collapse_fn, fold_runs=True)
        fold = folds["blocks.0"]
        representative = trace.modules[fold.representative]
        source = _draw_source(
            trace,
            tmp_path,
            "run_fold_auto_rep_stats",
            "auto",
            fold_runs=True,
        )
        rep_line = _collapsed_node_line(source, "blocks.0")

        assert fold.num_layers != representative.num_layers
        assert fold.num_params != representative.num_params
        assert f"... +{fold.multiplicity - 1} more {fold.class_name}" in source
        assert f"{representative.num_layers} layers total" in rep_line
        assert f"{representative.num_params} params (all trainable)" in rep_line
        assert f"{fold.num_layers} layers total" not in rep_line
        assert f"{fold.num_params} params (all trainable)" not in rep_line
        assert f"{trace.num_tensors} tensors total" in source
        assert f"{trace.num_params} params" in source
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_parallel_ellipsis_stays_in_flow(tmp_path: Path) -> None:
    """Parallel run folds keep ellipsis edges from source to sink."""

    trace = _trace(ParallelRepeatedBranches(depth=40), torch.randn(1, 4, 16, 16))
    try:
        auto_source = str(
            trace.draw(
                vis_outpath=str(tmp_path / "parallel_run_fold_auto"),
                vis_save_only=True,
                vis_fileformat="svg",
                vis_node_placement="dot",
                collapse="auto",
                collapse_fn=_select_branches_child,
            )
        )
        ellipsis_name = _run_fold_ellipsis_name("branches.0")

        assert _run_fold_ellipsis_count(auto_source, 40) == 1
        assert _incoming_edge_count(auto_source, ellipsis_name) >= 1
        assert _outgoing_edge_count(auto_source, ellipsis_name) >= 1
        assert _edge_count(auto_source, "input_1pass1", ellipsis_name) == 1
        assert _edge_count(auto_source, ellipsis_name, "cat_1_161pass1") == 1
    finally:
        trace.cleanup()


def test_auto_collapse_parallel_fold_representative_uses_single_instance_stats(
    tmp_path: Path,
) -> None:
    """Auto-mode parallel fold representatives display single-instance stats."""

    trace = _trace(ParallelRepeatedBranches(depth=40), torch.randn(1, 4, 16, 16))
    try:
        folds = resolve_run_folds(trace, _select_branches_child)
        fold = folds["branches.0"]
        representative = trace.modules[fold.representative]
        source = str(
            trace.draw(
                vis_outpath=str(tmp_path / "parallel_run_fold_auto_rep_stats"),
                vis_save_only=True,
                vis_fileformat="svg",
                vis_node_placement="dot",
                collapse="auto",
                collapse_fn=_select_branches_child,
            )
        )
        rep_line = _collapsed_node_line(source, "branches.0")

        assert fold.num_layers != representative.num_layers
        assert fold.num_params != representative.num_params
        assert f"... +{fold.multiplicity - 1} more {fold.class_name}" in source
        assert f"{representative.num_layers} layers total" in rep_line
        assert f"{representative.num_params} params (all trainable)" in rep_line
        assert f"{fold.num_layers} layers total" not in rep_line
        assert f"{fold.num_params} params (all trainable)" not in rep_line
        assert f"{trace.num_tensors} tensors total" in source
        assert f"{trace.num_params} params" in source
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_splits_same_spatial_channel_steps(tmp_path: Path) -> None:
    """Run-fold splits same-spatial channel changes into stage boundaries."""

    trace = _trace(DimStepRun(depth=24, start_width=32), torch.randn(1, 32, 8, 8))
    try:
        auto_source = _draw_source(trace, tmp_path, "dim_step_stage_boundary_auto", "auto")
        folds = resolve_run_folds(trace, _select_blocks_child)

        assert _collapsed_exact_label_count(auto_source, "blocks.1") == 1
        assert "runfoldellipsis" not in auto_source
        assert "... +" not in auto_source
        assert "shapes " not in auto_source
        assert folds == {}
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_folds_mobilenet_channel_plateaus() -> None:
    """MobileNetV2-style same-class plateaus fold and channel transitions split."""

    trace = _trace(MobileNetPlateauStack(), torch.randn(1, 3, 16, 16))
    try:
        folds = resolve_run_folds(trace, _select_features_child)

        assert folds["features.0"].addresses == ("features.0", "features.1", "features.2")
        assert folds["features.3"].addresses == (
            "features.3",
            "features.4",
            "features.5",
            "features.6",
        )
        assert folds["features.0"].addresses[-1] != folds["features.3"].addresses[0]
        assert folds["features.3"].hidden_member_composition == {
            "hidden_with_residual_join": 3,
            "hidden_without_residual_join": 0,
        }
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_folds_residual_mix_without_digest_key() -> None:
    """Same-class equal-shape blocks fold even when residual topology differs."""

    trace = _trace(MobileNetPlateauStack(), torch.randn(1, 3, 16, 16))
    try:
        folds = resolve_run_folds(trace, _select_features_child)

        assert folds["features.3"].addresses == (
            "features.3",
            "features.4",
            "features.5",
            "features.6",
        )
        assert folds["features.3"].hidden_member_composition["hidden_with_residual_join"] == 3
    finally:
        trace.cleanup()


def test_auto_collapse_fold_runs_true_splits_run_around_odd_hidden_member() -> None:
    """``fold_runs=True`` folds the maximal legal sub-runs around an odd hidden member.

    Regression for the round-3 honesty gate's own adjacent gap: pre-fix,
    ``_iter_collapsible_runs`` (the "shared substrate" v1 grouper reachable
    via ``fold_runs=True`` or a custom ``collapse_fn``, as opposed to the
    default v2 optimizer's ``_maximal_legal_runs``) yielded exactly one
    whole-run candidate per class/stem group with no backtracking, so
    ``_run_fold_hidden_members_uniform`` rejecting that single candidate
    (because ``blocks.3`` is structurally odd) meant *zero* folds for the
    entire 7-block run -- even though ``(blocks.0, blocks.1, blocks.2)`` and
    ``(blocks.3, blocks.4, blocks.5, blocks.6)`` are each independently
    legal, hidden-uniform runs, exactly as the default v2 engine already
    handles for the identical input.
    """

    trace = _trace(OddHiddenMemberStack(total=7, odd_index=3), torch.randn(2, 8))
    try:
        folds = resolve_run_folds(trace, _select_blocks_child, fold_runs=True)

        assert folds["blocks.0"].addresses == ("blocks.0", "blocks.1", "blocks.2")
        assert folds["blocks.3"].addresses == (
            "blocks.3",
            "blocks.4",
            "blocks.5",
            "blocks.6",
        )
        assert folds["blocks.3"].representative == "blocks.3"
        # The odd block is only ever the visible representative of its own
        # fold -- it must never appear as a *hidden* member of any fold.
        assert all("blocks.3" not in fold.addresses[1:] for fold in folds.values())
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_keeps_different_depth_stages_separate(tmp_path: Path) -> None:
    """Run-fold does not merge same-class sibling stages with different depths."""

    trace = _trace(UnevenDepthStages(depths=(2, 3, 4) * 8), torch.randn(1, 4, 16, 16))
    try:
        auto_source = _draw_source(trace, tmp_path, "uneven_depth_stages_auto", "auto")
        folds = resolve_run_folds(trace, _select_first_stage_unit)

        assert _run_fold_ellipsis_count(auto_source, 3) == 0
        assert "stages.0" not in folds
        assert "stages.1" not in folds
        assert "stages.2" not in folds
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_rejects_spatial_span() -> None:
    """Run-fold does not create a box across spatial-resolution changes."""

    trace = _trace(SpatialStepRun(), torch.randn(1, 4, 32, 32))
    try:
        folds = resolve_run_folds(trace, _select_blocks_child)

        assert "blocks.0" not in folds
        assert "blocks.1" not in folds
        assert "blocks.2" not in folds
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_parallel_fan_keeps_junction_visible(tmp_path: Path) -> None:
    """Parallel fan folds branches while keeping the shared concat outside."""

    trace = _trace(ParallelRepeatedBranches(depth=40), torch.randn(1, 4, 16, 16))
    try:
        auto_source = str(
            trace.draw(
                vis_outpath=str(tmp_path / "parallel_fan_contract_auto"),
                vis_save_only=True,
                vis_fileformat="svg",
                vis_node_placement="dot",
                collapse="auto",
                collapse_fn=_select_branches_child,
            )
        )
        ellipsis_name = _run_fold_ellipsis_name("branches.0")

        assert _run_fold_ellipsis_count(auto_source, 40) == 1
        assert _has_visible_node(auto_source, "cat_")
        assert _edge_count(auto_source, ellipsis_name, "cat_1_161pass1") == 1
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_rejects_flow_parallel_aux_chain() -> None:
    """Flow-parallel aux children do not join a trunk chain run."""

    trace = _trace(RegistrationInterleavedAuxParent(), torch.randn(1, 4, 8, 8))
    try:
        analysis = analyze_collapse(trace)
        graph = _flow_graph_for_sibling_group(
            trace,
            "self",
            _sibling_address_groups(trace)["self"],
            analysis,
        )
        assert graph is not None
        runs = list(_iter_collapsible_runs(trace, list(graph.flow_children), lambda module: True))

        assert runs == [("trunk0", "trunk1", "trunk2")]
        assert not _run_fold_is_legal(("trunk0", "trunk1", "trunk2"), graph)
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_skips_readable_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run-fold does not elide a stack whose collapsed render is readable."""

    trace = _trace(RepeatedResidual(depth=12), torch.randn(2, 8))
    try:
        collapse_fn = resolve_collapse_fn(trace, "auto", "unrolled")
        folds = resolve_run_folds(trace, collapse_fn)
        auto_source = _draw_source(trace, tmp_path, "run_fold_readable_auto", "auto")

        assert folds == {}
        assert "runfoldellipsis" not in auto_source
        assert "... +" not in auto_source
    finally:
        trace.cleanup()


def test_auto_collapse_fold_runs_false_disables_default_run_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``fold_runs=False`` disables otherwise-default auto run folding."""

    trace = _trace(RepeatedResidual(depth=24), torch.randn(2, 8))
    try:
        auto_source = _draw_source(
            trace,
            tmp_path,
            "run_fold_disabled_auto",
            "auto",
            fold_runs=False,
        )

        assert "runfoldellipsis" not in auto_source
        assert "... +" not in auto_source
    finally:
        trace.cleanup()


def test_auto_collapse_fold_runs_true_folds_readable_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``fold_runs=True`` folds eligible runs even inside the readable band."""

    trace = _trace(RepeatedResidual(depth=12), torch.randn(2, 8))
    try:
        auto_source = _draw_source(
            trace,
            tmp_path,
            "run_fold_forced_auto",
            "auto",
            fold_runs=True,
        )

        assert _run_fold_ellipsis_count(auto_source, 12) == 1
        assert "... +11 more ResidualBlock" in auto_source
        assert _collapsed_exact_label_count(auto_source, "blocks.11") == 0
    finally:
        trace.cleanup()


def test_auto_collapse_fold_runs_true_standalone_keeps_parallel_junction(
    tmp_path: Path,
) -> None:
    """Standalone run folding works with ``collapse='none'`` and preserves junctions."""

    trace = _trace(ParallelRepeatedBranches(depth=40), torch.randn(1, 4, 16, 16))
    try:
        source = str(
            trace.draw(
                vis_outpath=str(tmp_path / "standalone_run_fold_parallel"),
                vis_save_only=True,
                vis_fileformat="svg",
                vis_node_placement="dot",
                collapse="none",
                fold_runs=True,
            )
        )
        ellipsis_name = _run_fold_ellipsis_name("branches.0")

        assert _run_fold_ellipsis_count(source, 40) == 1
        assert _has_visible_node(source, "cat_")
        assert _edge_count(source, "input_1pass1", ellipsis_name) == 1
        assert _edge_count(source, ellipsis_name, "cat_1_161pass1") == 1
    finally:
        trace.cleanup()


def test_auto_collapse_fold_runs_rejects_invalid_value(tmp_path: Path) -> None:
    """Run-fold policy validation rejects non-tristate values."""

    trace = _trace(RepeatedResidual(depth=4), torch.randn(2, 8))
    try:
        with pytest.raises(ValueError, match="fold_runs must be None, True, or False"):
            trace.draw(
                vis_outpath=str(tmp_path / "invalid_fold_runs"),
                vis_save_only=True,
                vis_fileformat="svg",
                vis_node_placement="dot",
                collapse="auto",
                fold_runs="yes",  # type: ignore[arg-type]
            )
    finally:
        trace.cleanup()


def test_repeat_capture_and_trivial_collapse_score() -> None:
    """Scores are deterministic across captures, and trivial modules score zero."""

    first = _trace(RepeatedResidual(depth=4), torch.randn(2, 8))
    second = _trace(RepeatedResidual(depth=4), torch.randn(2, 8))
    trivial = _trace(TrivialSingle(), torch.randn(2, 8))
    try:
        assert first.module_collapse_order == second.module_collapse_order
        assert trivial.modules["relu"].collapse_score == 0.0
    finally:
        first.cleanup()
        second.cleanup()
        trivial.cleanup()


def test_max_collapse_is_never_less_collapsed_than_auto(tmp_path: Path) -> None:
    """Max collapse renders no more nodes than auto on representative toy models."""

    cases: list[tuple[str, torch.nn.Module, torch.Tensor]] = [
        ("repeated", RepeatedResidual(depth=5), torch.randn(2, 8)),
        ("parent_residual", ParentJoinResidual(depth=4), torch.randn(2, 8)),
        ("skip_concat", SkipConcatUNet(), torch.randn(1, 4, 16, 16)),
        ("branch_concat", BranchConcat(), torch.randn(1, 4, 16, 16)),
        ("stem_stage", StemStageBackbone(), torch.randn(1, 4, 16, 16)),
        ("recurrent", RecurrentWrapper(), torch.randn(1, 6, 8)),
    ]
    for name, model, x in cases:
        trace = _trace(model, x)
        try:
            auto_source = _draw_source(trace, tmp_path, f"{name}_auto", "auto")
            max_source = _draw_source(trace, tmp_path, f"{name}_max", "max")

            assert _dot_node_count(max_source) <= _dot_node_count(auto_source)
        finally:
            trace.cleanup()


@pytest.mark.parametrize(
    ("model_factory", "input_shape", "op_prefix"),
    (
        (OwnOutputResidualModel, (1, 3, 8, 8), "add_"),
        (OwnOutputConcatModel, (1, 3, 8, 8), "cat_"),
    ),
)
def test_max_collapsed_box_labels_exclude_surfaced_own_output_ops(
    tmp_path: Path,
    model_factory: Callable[[], torch.nn.Module],
    input_shape: tuple[int, ...],
    op_prefix: str,
) -> None:
    """Max-mode collapsed boxes summarize the remainder beside surfaced output ops."""

    trace = _trace(model_factory(), torch.randn(*input_shape))
    try:

        def collapse_block(module: Any) -> bool:
            """Select the fixture block as a v2 max collapsed module."""

            return getattr(module, "address", None) == "block"

        setattr(collapse_block, "_torchlens_v2_mode", "max")
        source = str(
            trace.draw(
                vis_outpath=str(tmp_path / f"{op_prefix}own_output_remainder"),
                vis_save_only=True,
                vis_fileformat="svg",
                vis_node_placement="dot",
                collapse_fn=collapse_block,
            )
        )
        module = trace.modules["block"]
        surfaced_ops = _atomic_own_output_ops(trace, "block")
        surfaced_params = sum(int(getattr(op, "num_params", 0) or 0) for op in surfaced_ops)
        remainder_layers = module.num_layers - len(surfaced_ops)
        remainder_params = module.num_params - surfaced_params
        block_line = _collapsed_node_line(source, "block")

        assert len(surfaced_ops) == 1
        assert _has_visible_node(source, op_prefix)
        assert f"{remainder_layers} layers total" in block_line
        assert f"{module.num_layers} layers total" not in block_line
        assert f"{remainder_params} params" in block_line
        assert remainder_layers + len(surfaced_ops) == module.num_layers
        assert remainder_params + surfaced_params == module.num_params
    finally:
        trace.cleanup()


def test_auto_plan_remainder_honesty_does_not_require_max_level() -> None:
    """Auto-mode v2 plans use plan structure even when OptimizerResult.level is unset."""

    trace = _trace(OwnOutputResidualModel(), torch.randn(1, 3, 8, 8))
    try:
        surfaced_ops = _atomic_own_output_ops(trace, "block")
        module_call = trace.module_calls["block:1"]

        def collapse_block(module: Any) -> bool:
            """Select the fixture block as a collapsed module."""

            return getattr(module, "address", None) == "block"

        setattr(
            collapse_block,
            "_torchlens_v2_plan",
            CollapsePlan((RawOp(surfaced_ops[0]),), RenderContext()),
        )
        setattr(collapse_block, "_torchlens_v2_result", type("Result", (), {"level": None})())

        assert _collapsed_module_should_show_remainder(
            trace,
            "block",
            module_call.ops,
            collapse_block,
        )
    finally:
        trace.cleanup()


@pytest.mark.heavy
def test_equivalent_max_plans_render_same_collapsed_box_layer_labels(tmp_path: Path) -> None:
    """Equivalent max endpoint plans render identical collapsed-box layer labels."""

    torch.set_num_threads(4)
    trace = _trace(tvm.mobilenet_v2(weights=None), torch.randn(1, 3, 224, 224))
    try:
        max_plan = trace.collapse_plan(mode="max")
        level_plan = trace.collapse_plan(mode=1.0)
        max_labels = _draw_svg_layer_labels(trace, tmp_path, "mobilenet_v2_max", "max")
        level_labels = _draw_svg_layer_labels(trace, tmp_path, "mobilenet_v2_level_1", 1.0)

        assert max_plan == level_plan
        assert max_labels == level_labels
        assert "20 layers total" in max_labels
        assert "21 layers total" not in level_labels
    finally:
        trace.cleanup()


def test_auto_collapse_descends_into_nested_stage_container(tmp_path: Path) -> None:
    """Auto keeps nested stage boxes instead of swallowing the whole container."""

    trace = _trace(NestedStageBackbone(), torch.randn(1, 4, 16, 16))
    try:
        auto_source = _draw_source(trace, tmp_path, "nested_stage_auto", "auto")
        max_source = _draw_source(trace, tmp_path, "nested_stage_max", "max")

        assert _collapsed_label_count(auto_source, "backbone.") >= 4
        assert "<B>@backbone</B></TD>" not in auto_source
        assert _dot_node_count(max_source) <= _dot_node_count(auto_source)
        assert auto_source != max_source
    finally:
        trace.cleanup()


def test_auto_collapse_groups_structurally_similar_unique_parallel_modules(
    tmp_path: Path,
) -> None:
    """Auto groups same-role unique mixed modules and avoids node-wall renders."""

    trace = _trace(UniqueParallelStack(), torch.randn(1, 4, 16, 16))
    try:
        none_source = _draw_source(trace, tmp_path, "unique_parallel_none", "none")
        auto_source = _draw_source(trace, tmp_path, "unique_parallel_auto", "auto")

        assert _dot_node_count(none_source) > 40
        assert _collapsed_label_count(auto_source, "mixed_") >= 4
        assert _dot_node_count(auto_source) < 30
        assert auto_source != none_source
    finally:
        trace.cleanup()


def test_v2_max_mode_uses_op_segments_without_changing_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Max mode condenses raw op runs while auto remains segment-free."""

    trace = _trace(SegmentToyNet(), torch.randn(1, 3, 8, 8))
    try:
        context = RenderContext()
        auto_fn = resolve_collapse_fn(trace, "auto", "unrolled", context=context)
        max_fn = resolve_collapse_fn(trace, "max", "unrolled", context=context)
        auto_plan = getattr(auto_fn, "_torchlens_v2_plan")
        max_plan = getattr(max_fn, "_torchlens_v2_plan")

        assert not any(isinstance(node, OpSegment) for node in auto_plan.nodes)
        assert any(isinstance(node, OpSegment) for node in max_plan.nodes)
        assert count(max_plan) < count(auto_plan)
    finally:
        trace.cleanup()


def test_v2_auto_uses_op_segments_as_over_band_last_resort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto condenses over-band flat op chains when module cuts cannot help."""

    trace = _trace(DistinctFunctionalChain(), torch.randn(1, 8))
    try:
        context = RenderContext()
        full_count = count(plan_from_v1(trace, None, None, context))
        result = select_collapse_plan(trace, context, mode="auto")

        assert full_count > 40
        assert any(isinstance(node, OpSegment) for node in result.plan.nodes)
        assert 1 <= count(result.plan) <= 40
        assert count(result.plan) < full_count
    finally:
        trace.cleanup()


def test_v2_zero_frontier_falls_back_to_visible_full_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V2 never returns an empty plan when every DP point is over the cap."""

    monkeypatch.setattr(
        collapse_optimizer,
        "_RESULT_CACHE",
        collapse_optimizer.weakref.WeakKeyDictionary(),
    )
    monkeypatch.setattr(collapse_optimizer, "_select_best_decision", lambda **_: None)
    trace = _trace(LongFunctional(depth=80), torch.randn(1, 8))
    try:
        result = select_collapse_plan(trace, RenderContext(), mode="auto")

        assert not result.declined
        assert count(result.plan) >= 1
        assert result.visible_count == count(result.plan)
        assert result.reason == "floor_fallback: no optimizer frontier was produced"
    finally:
        trace.cleanup()


def test_incremental_count_mismatch_warns_once_outside_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production count-model gaps warn and continue instead of raising."""

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("TORCHLENS_COLLAPSE_STRICT", raising=False)
    monkeypatch.setattr(auto_collapse, "_COUNT_MISMATCH_WARNING_EMITTED", False)
    trace = _trace(SegmentToyNet(), torch.randn(1, 3, 8, 8))
    try:
        with pytest.warns(RuntimeWarning, match="incremental collapse count mismatch"):
            _assert_plan_count(trace, None, None, RenderContext(), -1)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            _assert_plan_count(trace, None, None, RenderContext(), -1)
        assert not records
    finally:
        trace.cleanup()


def test_incremental_count_mismatch_stays_hard_in_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict collapse count-model gaps remain hard tripwires."""

    monkeypatch.setenv("TORCHLENS_COLLAPSE_STRICT", "1")
    trace = _trace(SegmentToyNet(), torch.randn(1, 3, 8, 8))
    try:
        with pytest.raises(AssertionError, match="incremental collapse count mismatch"):
            _assert_plan_count(trace, None, None, RenderContext(), -1)
    finally:
        trace.cleanup()


def test_transformer_attention_landmarks_remain_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPT-2 attention residual landmarks are not hidden by auto or max."""

    transformers = pytest.importorskip("transformers")

    class DictInputWrapper(torch.nn.Module):
        """Wrap keyword-input transformer modules for TorchLens tracing."""

        def __init__(self, model: torch.nn.Module) -> None:
            """Initialize the wrapped module."""

            super().__init__()
            self.model = model

        def forward(self, batch: dict[str, torch.Tensor]) -> Any:
            """Forward a tensor dictionary as keyword arguments."""

            return self.model(**batch)

    cases = (
        (
            "gpt2",
            transformers.GPT2Model(
                transformers.GPT2Config(
                    vocab_size=50257,
                    n_embd=128,
                    n_layer=2,
                    n_head=4,
                    n_positions=32,
                    n_ctx=32,
                )
            ),
            {"input_ids": torch.randint(0, 50257, (1, 16))},
        ),
        (
            "bert",
            transformers.BertModel(
                transformers.BertConfig(
                    vocab_size=30522,
                    hidden_size=128,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    intermediate_size=256,
                )
            ),
            {
                "input_ids": torch.randint(0, 30522, (1, 16)),
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            },
        ),
        (
            "distilbert",
            transformers.DistilBertModel(
                transformers.DistilBertConfig(
                    vocab_size=30522,
                    dim=128,
                    n_layers=2,
                    n_heads=4,
                    hidden_dim=256,
                )
            ),
            {
                "input_ids": torch.randint(0, 30522, (1, 16)),
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            },
        ),
    )
    for _name, model, batch in cases:
        trace = _trace(DictInputWrapper(model), batch)  # type: ignore[arg-type]
        try:
            for mode in ("auto", "max"):
                result = select_collapse_plan(trace, RenderContext(), mode=mode)
                assert _landmark_swallow_count(trace, result) == 0
        finally:
            trace.cleanup()


def test_v2_max_op_segment_renders_dashed_box_and_contracts_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendered max op segments suppress members and keep contracted flow edges."""

    trace = _trace(SegmentToyNet(), torch.randn(1, 3, 8, 8))
    try:
        source = _draw_source(trace, tmp_path, "max_op_segment", "max")

        assert "__segment__" in source
        assert 'style="rounded,dashed,filled"' in source
        assert "conv2d_1_1 ... conv2d_5_8 -- 8 ops" in source
        assert "conv2d_2_2pass1 [" not in source
        assert "input_1pass1 -> conv2d_1_1__segment__conv2d_5_8pass1" in source
        assert "relu_4_9__segment__conv2d_7_12pass1 -> output_1pass1" in source
    finally:
        trace.cleanup()


@pytest.mark.serial
def test_signal_tally_latency_under_budget() -> None:
    """Signal tally stays under a load-scaled per-3k-node budget."""

    trace = _trace(LongFunctional(depth=1500), torch.randn(1, 8))
    try:
        start = time.perf_counter()
        analyze_collapse(trace)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        scaled_ms = elapsed_ms * (3000.0 / max(1, len(trace.ops)))
        budget_factor = float(os.environ.get("TORCHLENS_TIMING_BUDGET_FACTOR", "2.0"))
        assert scaled_ms < 50.0 * budget_factor
    finally:
        trace.cleanup()
