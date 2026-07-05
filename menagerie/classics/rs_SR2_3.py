# FAITHFUL REIMPLEMENTATION from Afshar, Oikonomou, Naderkhani, Tyrrell,
# Plataniotis, Farahani & Mohammadi, "3D-MCN: A 3D Multi-scale Capsule Network
# for Lung Nodule Malignancy Prediction" (Scientific Reports 10, 7948, 2020,
# https://doi.org/10.1038/s41598-020-64824-5, open access).
#
# `gh search repos "3D-MCN"` and web search found no public repository for this
# exact model. Sibling capsule-routing repos exist (VinAIResearch/3D-UCaps,
# UARK-AICV/3DConvCaps) but they implement DIFFERENT architectures (UNet-style
# 3D segmentation capsule networks) -- not 3D-MCN's own network -- so they are
# not vendorable/portable source for this candidate under rung 2/3; they only
# confirm that 3D capsule-routing layers are a real, established mechanism.
#
# The paper's Results section ("Proposed 3D-MCN Model") gives an exact
# architecture description, transcribed below:
#   - 3D-MCN = three INDEPENDENT CapsNets, each taking a 3D nodule crop at a
#     different spatial scale (visible surrounding-tissue extent) as input.
#   - Each single-scale CapsNet = "a convolutional layer, a primary capsule
#     layer that predicts the output of the next layer, and a classification
#     capsule layer that outputs the probability of each class along with the
#     instantiation parameters" (routing-by-agreement per Sabour, Frosst &
#     Hinton, "Dynamic Routing Between Capsules", NeurIPS 2017 -- the paper
#     reproduces that routing algorithm's exact update equations:
#     a_ij = s_j . u_hat_{j|i}; b_ij += a_ij; c_ij = softmax_j(b_ij);
#     s_j = sum_i c_ij * u_hat_{j|i}).
#   - "Each output class is of dimension 16. Having two output classes (benign
#     and malignant) results in a vector of dimension 32 for each CapsNet, and
#     having three CapsNets results in an input of size 96 to the multi-scale
#     network." -- i.e. class_dim=16, num_classes=2, 3 scales -> fusion_in=96.
#   - "For each CapsNet, the output vector of the lower probability class was
#     masked (set to zero)" before concatenation across scales.
#   - "Our multi-scale model was a fully connected neural network with three
#     hidden layers of sizes 1028, 512, and 256" feeding the final
#     benign/malignant classification.
#
# The paper's text does not state the exact routing-iteration count or
# primary-capsule conv hyperparameters for its own CapsNet (only for a baseline
# comparison 3D-CNN); this module uses the canonical default of 3 routing
# iterations (Sabour et al. 2017's own default) and a small conv/primary-capsule
# stem sized for fast random-init tracing, faithfully reproducing every
# mechanism the paper specifies exactly (routing algorithm, capsule/class
# dimensions, masking, three-scale fusion MLP widths).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "reimpl-pytorch"


def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Capsule squashing nonlinearity (Sabour, Frosst & Hinton 2017, Eq. 1)."""
    squared_norm = (s**2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * s / torch.sqrt(squared_norm + eps)


class PrimaryCapsule3D(nn.Module):
    """3D primary capsule layer: a Conv3d producing `out_capsule_types *
    capsule_dim` channels, reshaped into capsule vectors (one per spatial
    position per capsule type) and squashed -- the 3D analogue of Sabour et
    al.'s PrimaryCaps layer, as used by 3D-MCN's "primary capsule layer that
    predicts the output of the next layer"."""

    def __init__(self, in_channels, out_capsule_types, capsule_dim, kernel_size=5, stride=2):
        super().__init__()
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv3d(
            in_channels,
            out_capsule_types * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)  # (N, types*dim, D, H, W)
        n = out.shape[0]
        spatial = out.shape[2:]
        out = out.view(n, -1, self.capsule_dim, *spatial)
        out = out.permute(0, 1, 3, 4, 5, 2).reshape(n, -1, self.capsule_dim)
        return squash(out, dim=-1)


class ClassCapsule3D(nn.Module):
    """Classification capsule layer with dynamic routing-by-agreement (Sabour,
    Frosst & Hinton 2017, Algorithm 1), reproducing the paper's own transcribed
    update equations for a_ij, b_ij, c_ij, s_j. 3 routing iterations (canonical
    default; exact count unspecified for 3D-MCN's own CapsNet)."""

    def __init__(self, in_capsules, in_dim, num_classes, out_dim, routing_iters=3):
        super().__init__()
        self.routing_iters = routing_iters
        self.num_classes = num_classes
        self.out_dim = out_dim
        # Per-pair transformation matrices W_ij predicting u_hat_{j|i} = W_ij @ u_i.
        self.W = nn.Parameter(0.01 * torch.randn(1, in_capsules, num_classes, out_dim, in_dim))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (N, in_capsules, in_dim)
        n = u.shape[0]
        u_exp = u.unsqueeze(2).unsqueeze(-1)  # (N, in_caps, 1, in_dim, 1)
        w = self.W.expand(n, -1, -1, -1, -1)  # (N, in_caps, classes, out_dim, in_dim)
        u_hat = torch.matmul(w, u_exp).squeeze(-1)  # (N, in_caps, classes, out_dim)

        b = torch.zeros(n, u.shape[1], self.num_classes, device=u.device, dtype=u.dtype)
        v = None
        for _ in range(self.routing_iters):
            c = F.softmax(b, dim=2)  # c_ij
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)  # (N, classes, out_dim)
            v = squash(s, dim=-1)
            agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)  # a_ij: (N, in_caps, classes)
            b = b + agreement
        return v  # (N, classes, out_dim)


def _primary_capsule_count(input_size: int, primary_capsule_types: int) -> int:
    """Deterministic capsule count for a fixed cubic crop, given the conv/
    primary-capsule stem's fixed kernel/stride/padding below (kernel=5,
    stride=1, pad=2 for the conv layer -- size-preserving; kernel=5, stride=2,
    pad=2 for the primary-capsule conv -- halves each spatial dim, rounding per
    the standard conv output-size formula), avoiding any lazy/dynamic module
    construction inside forward()."""
    conv_out = input_size  # kernel=5, stride=1, pad=2 preserves spatial size
    primary_out = (conv_out + 2 * 2 - 5) // 2 + 1
    return primary_capsule_types * (primary_out**3)


class SingleScaleCapsNet3D(nn.Module):
    """Single-scale 3D CapsNet: conv layer -> primary capsule layer ->
    classification capsule layer, per 3D-MCN Fig. 7's asterisked
    sub-architecture (the template shared by all three scale-branches)."""

    def __init__(
        self,
        input_size=8,
        in_channels=1,
        conv_channels=8,
        primary_capsule_types=4,
        primary_dim=4,
        num_classes=2,
        class_dim=16,
        routing_iters=3,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, conv_channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.primary = PrimaryCapsule3D(
            conv_channels, primary_capsule_types, primary_dim, kernel_size=5, stride=2
        )
        in_capsules = _primary_capsule_count(input_size, primary_capsule_types)
        self.classify = ClassCapsule3D(
            in_capsules, primary_dim, num_classes, class_dim, routing_iters
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv(x))
        u = self.primary(h)  # (N, in_capsules, primary_dim)
        return self.classify(u)  # (N, num_classes, class_dim)


class MultiScaleCapsNet3D(nn.Module):
    """3D-MCN: three independent single-scale 3D CapsNets, one per input
    spatial scale. Each branch's classification-capsule output is masked (the
    lower-probability class vector zeroed, per the paper's masking scheme),
    the three masked 32-dim (2 classes x 16-dim) vectors are concatenated into
    a 96-dim vector, and a 3-hidden-layer fusion MLP (sizes 1028, 512, 256, per
    the paper) produces the final benign/malignant classification."""

    def __init__(self, input_size=8, num_classes=2, class_dim=16, fusion_hidden=(1028, 512, 256)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                SingleScaleCapsNet3D(
                    input_size=input_size, num_classes=num_classes, class_dim=class_dim
                )
                for _ in range(3)
            ]
        )
        self.num_classes = num_classes
        self.class_dim = class_dim
        fusion_in = 3 * num_classes * class_dim  # = 96 for the paper's config
        layers = []
        width = fusion_in
        for h in fusion_hidden:
            layers += [nn.Linear(width, h), nn.ReLU(inplace=True)]
            width = h
        layers.append(nn.Linear(width, num_classes))
        self.fusion = nn.Sequential(*layers)

    @staticmethod
    def _mask_lower_probability_class(v: torch.Tensor) -> torch.Tensor:
        # v: (N, num_classes, class_dim). Zero out every class capsule except
        # the highest-norm (predicted) one, per "the output vector of the
        # lower probability class was masked (set to zero)".
        norms = v.norm(dim=-1)  # (N, num_classes)
        idx = norms.argmax(dim=-1)  # (N,)
        mask = F.one_hot(idx, num_classes=v.shape[1]).unsqueeze(-1).to(v.dtype)
        return v * mask

    def forward(
        self, scale1: torch.Tensor, scale2: torch.Tensor, scale3: torch.Tensor
    ) -> torch.Tensor:
        outs = []
        for branch, x in zip(self.branches, (scale1, scale2, scale3)):
            v = branch(x)  # (N, num_classes, class_dim)
            v = self._mask_lower_probability_class(v)
            outs.append(v.flatten(1))
        fused = torch.cat(outs, dim=1)  # (N, 3*num_classes*class_dim)
        return self.fusion(fused)


def build_3dmcn():
    return MultiScaleCapsNet3D(
        input_size=8, num_classes=2, class_dim=16, fusion_hidden=(1028, 512, 256)
    )


def example_input_3dmcn():
    # Three independent 3D nodule crops (one per multi-scale branch); the real
    # pipeline differs only in physical voxel extent per scale, not tensor
    # shape, so a tiny fixed-size crop tensor per branch is a faithful stand-in
    # for tracing.
    scale1 = torch.rand(1, 1, 8, 8, 8)
    scale2 = torch.rand(1, 1, 8, 8, 8)
    scale3 = torch.rand(1, 1, 8, 8, 8)
    return (scale1, scale2, scale3)


MENAGERIE_ENTRIES = [
    (
        "3D Multi-Scale Capsule Network (3D-MCN)",
        "build_3dmcn",
        "example_input_3dmcn",
        2020,
        "reimpl-pytorch",
    ),
]
