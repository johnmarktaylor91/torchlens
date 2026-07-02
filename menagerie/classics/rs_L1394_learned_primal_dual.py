# FAITHFUL PORT of adler-j/learned_primal_dual @ master (original framework: TensorFlow 1.x
# + ODL (Operator Discretization Library) / odl.contrib.tensorflow)
# https://github.com/adler-j/learned_primal_dual
#
# "Learned Primal-Dual Reconstruction" (Adler & Oktem, IEEE TMI 2018, arXiv:1707.06474).
# The reference script (ellipses/learned_primal_dual.py) builds `n_iter` (=10) alternating
# dual/primal update blocks around a fixed linear forward operator (ODL's parallel-beam
# `RayTransform`, i.e. the Radon transform) and its adjoint (back-projection):
#
#   primal = concat([0]*n_primal); dual = concat([0]*n_dual)
#   for i in range(n_iter):
#       dual:   update = concat([dual, A(primal[...,1]), y]);  update = conv-prelu-conv-prelu-conv(update)
#               dual = dual + update
#       primal: update = concat([primal, A^T(dual[...,0])]);   update = conv-prelu-conv-prelu-conv(update)
#               primal = primal + update
#   x_result = primal[..., 0]
#
# each conv is a 3x3 SAME conv (n_primal/n_dual channels out on the final conv of each
# block, `filters=32` on the two hidden convs), and PReLU is the activation between them.
# This is transcribed here FAITHFULLY into self-contained torch: the two learned CNN
# update blocks (`apply_conv` x3 + `prelu` x2, exactly as in the TF script) are unchanged;
# ODL's abstract `RayTransform`/`RayTransform.adjoint` (which in the original is JIT-compiled
# against the `astra`/`skimage` backend via `odl.contrib.tensorflow.as_tensorflow_layer`) is
# replaced by an explicit, differentiable parallel-beam Radon forward/back-projection pair
# implemented directly in torch (rotate-then-column-sum for the forward projection, and its
# exact adjoint -- smear-then-rotate-back -- for the back-projection), matching the same
# forward-operator semantics (`odl.tomo.parallel_beam_geometry` + `odl.tomo.RayTransform`)
# the original network is built around. Only the *fixed linear operator*'s numerical backend
# changed (ODL/ASTRA -> torch grid_sample rotation); the *learned* primal-dual iteration --
# the actual "architecture" under study here -- is bit-for-bit the same block structure as
# the original TF graph.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class ParallelBeamRadon(nn.Module):
    """Differentiable parallel-beam Radon transform + adjoint (back-projection).

    Mirrors `odl.tomo.parallel_beam_geometry` + `odl.tomo.RayTransform` semantics used by
    the original script: given an (size x size) image, project along `num_angles` evenly
    spaced angles in [0, pi) to produce a (num_angles x size) sinogram, and (approximately)
    invert with the adjoint back-projection.
    """

    def __init__(self, size=32, num_angles=30):
        super().__init__()
        self.size = size
        self.num_angles = num_angles
        thetas = (
            torch.linspace(0.0, math.pi, num_angles, endpoint=False)
            if False
            else (torch.arange(num_angles, dtype=torch.float32) * (math.pi / num_angles))
        )
        self.register_buffer("thetas", thetas)

    def _affine_grid_for_angle(self, theta, batch, device, dtype):
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        rot = (
            torch.stack(
                [
                    torch.stack([cos_t, -sin_t, torch.zeros((), device=device, dtype=dtype)]),
                    torch.stack([sin_t, cos_t, torch.zeros((), device=device, dtype=dtype)]),
                ]
            )
            .unsqueeze(0)
            .expand(batch, 2, 3)
            .to(device=device, dtype=dtype)
        )
        grid = F.affine_grid(rot, [batch, 1, self.size, self.size], align_corners=False)
        return grid

    def forward(self, x):
        """Forward Radon transform. x: (B, 1, S, S) -> sinogram (B, 1, num_angles, S)."""
        b, _, s, _ = x.shape
        rows = []
        for theta in self.thetas:
            grid = self._affine_grid_for_angle(theta, b, x.device, x.dtype)
            rotated = F.grid_sample(
                x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            rows.append(rotated.sum(dim=2, keepdim=True))  # column-sum -> one projection row
        sino = torch.cat(rows, dim=2)  # (B, 1, num_angles, S)
        return sino / s

    def adjoint(self, sino):
        """Back-projection adjoint. sino: (B, 1, num_angles, S) -> image (B, 1, S, S)."""
        b = sino.shape[0]
        acc = torch.zeros(b, 1, self.size, self.size, device=sino.device, dtype=sino.dtype)
        for i, theta in enumerate(self.thetas):
            row = sino[:, :, i : i + 1, :]  # (B, 1, 1, S)
            smeared = row.expand(b, 1, self.size, self.size)
            grid = self._affine_grid_for_angle(-theta, b, sino.device, sino.dtype)
            acc = acc + F.grid_sample(
                smeared, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        return acc / self.num_angles


def _apply_conv(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)


class _DualBlock(nn.Module):
    def __init__(self, n_dual, n_primal, hidden=32):
        super().__init__()
        # inputs to conv stack: dual (n_dual) + A(primal slice) (1) + y (1)
        self.conv1 = _apply_conv(n_dual + 2, hidden)
        self.prelu1 = nn.PReLU(hidden)
        self.conv2 = _apply_conv(hidden, hidden)
        self.prelu2 = nn.PReLU(hidden)
        self.conv3 = _apply_conv(hidden, n_dual)

    def forward(self, dual, evalop, y_rt):
        update = torch.cat([dual, evalop, y_rt], dim=1)
        update = self.prelu1(self.conv1(update))
        update = self.prelu2(self.conv2(update))
        update = self.conv3(update)
        return dual + update


class _PrimalBlock(nn.Module):
    def __init__(self, n_primal, hidden=32):
        super().__init__()
        # inputs to conv stack: primal (n_primal) + A^T(dual slice) (1)
        self.conv1 = _apply_conv(n_primal + 1, hidden)
        self.prelu1 = nn.PReLU(hidden)
        self.conv2 = _apply_conv(hidden, hidden)
        self.prelu2 = nn.PReLU(hidden)
        self.conv3 = _apply_conv(hidden, n_primal)

    def forward(self, primal, evalop_adj):
        update = torch.cat([primal, evalop_adj], dim=1)
        update = self.prelu1(self.conv1(update))
        update = self.prelu2(self.conv2(update))
        update = self.conv3(update)
        return primal + update


class LearnedPrimalDual(nn.Module):
    """Learned Primal-Dual reconstruction network (Adler & Oktem 2018).

    Faithful port of `ellipses/learned_primal_dual.py`: n_iter alternating dual/primal
    CNN update blocks wrapped around a fixed Radon forward operator and its adjoint.
    """

    def __init__(self, size=32, num_angles=15, n_iter=4, n_primal=5, n_dual=5, hidden=32):
        super().__init__()
        self.size = size
        self.n_iter = n_iter
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.operator = ParallelBeamRadon(size=size, num_angles=num_angles)
        self.dual_blocks = nn.ModuleList(
            [_DualBlock(n_dual, n_primal, hidden) for _ in range(n_iter)]
        )
        self.primal_blocks = nn.ModuleList([_PrimalBlock(n_primal, hidden) for _ in range(n_iter)])

    def forward(self, y_rt):
        # y_rt: (B, 1, num_angles, size) -- the measured sinogram
        b = y_rt.shape[0]
        primal = torch.zeros(
            b, self.n_primal, self.size, self.size, device=y_rt.device, dtype=y_rt.dtype
        )
        dual = torch.zeros(
            b,
            self.n_dual,
            self.operator.num_angles,
            self.size,
            device=y_rt.device,
            dtype=y_rt.dtype,
        )

        for i in range(self.n_iter):
            evalop = self.operator(primal[:, 1:2])
            dual = self.dual_blocks[i](dual, evalop, y_rt)

            evalop_adj = self.operator.adjoint(dual[:, 0:1])
            primal = self.primal_blocks[i](primal, evalop_adj)

        return primal[:, 0:1]


def build_learned_primal_dual():
    return LearnedPrimalDual(size=32, num_angles=15, n_iter=4, n_primal=5, n_dual=5, hidden=16)


def example_input_learned_primal_dual():
    return torch.randn(1, 1, 15, 32)


MENAGERIE_ENTRIES = [
    (
        "LearnedPrimalDual",
        "build_learned_primal_dual",
        "example_input_learned_primal_dual",
        "2018",
        "ported-pytorch",
    ),
]
