# FAITHFUL PORT of https://github.com/dgilton/neumann_networks_code @ master
# (src/neumann_network.py::NeumannNet + src/learned_component_resnet_nblock.py
#  ::nblock_resnet + src/operators_blur_cifar.py's blur_model/blur_gramian,
#  original framework: TensorFlow 1.x / tf.contrib + Keras BatchNormalization)
#
# Neumann Networks for linear inverse problems in imaging (Gilton, Ongie,
# Willett, "Neumann Networks for Linear Inverse Problems in Imaging",
# IEEE Trans. Computational Imaging 2019 / arXiv:1901.03707): a fixed-depth
# truncation of the Neumann series for (A^T A + lambda*R)^-1 A^T y, where the
# per-term linear residual `runner - eta * A^T A(runner)` is corrected each
# iteration by a *learned* nonlinear regularizer network (here, a leaky-relu
# residual CNN, `nblock_resnet`) instead of the analytic proximal/gradient
# term, and the partial Neumann sums are accumulated across iterations.
#
# The real repo (`dgilton/neumann_networks_code`, note: distinct from
# `dgilton/neumann_networks`, which is only a gh-pages docs mirror with no
# code) ships this as TensorFlow 1.x graph-mode code: `tf.placeholder`,
# `tf.get_variable`, `tf.contrib.layers.xavier_initializer_conv2d`,
# `tf.keras.layers.BatchNormalization().apply(...)`, and `.ckpt` v1
# checkpoint files -- none of which is runnable in this torch-only base env
# (TF1.x graph mode + tf.contrib is long removed from modern TensorFlow).
# This is therefore a faithful PORT (rung 3): every op in
# `NeumannNet.__init__`'s "Build the network" loop and every conv/batchnorm/
# activation in `nblock_resnet.residual_block` / `nblock_resnet.network` is
# transcribed 1:1 below (kernel sizes, channel counts, patch-mean
# subtract/restore, leaky-relu slope 0.1, xavier_conv2d init) using the real
# CIFAR-deblurring instantiation from `operators_blur_cifar.py` /
# `driver_neumann_cifar_deblur.py` (Gaussian blur forward operator,
# gramian = blur(blur(x)) since the symmetric Gaussian blur is self-adjoint,
# n_blocks=6 matching the driver's own `n_blocks = 6 # B in the Neumann
# networks paper`) as the concrete forward/gramian operator pair.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# src/operators_blur_cifar.py (verbatim numerics; TF ops -> torch equivalents)
# ---------------------------------------------------------------------------


def _fspecial_gauss(size: int, sigma: float) -> torch.Tensor:
    """Transcribed from `fspecial_gauss` (numpy meshgrid Gaussian kernel).
    Uses floor-division semantics matching `-size//2 + 1 : size//2 + 1`
    (Python `//` floors toward -inf, e.g. for size=5: range(-2, 3))."""
    lo = (-size) // 2 + 1
    hi = size // 2 + 1
    ax = torch.arange(lo, hi, dtype=torch.float64)
    y, x = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    g = g / g.sum()
    return g.float()


class DepthwiseGaussianBlur(nn.Module):
    """Real forward operator from `operators_blur_cifar.py`: a fixed (not
    learned) 5x5 sigma=2.0 Gaussian depthwise blur, `SAME` padding, applied
    per color channel (`tf.nn.depthwise_conv2d`). `blur_gramian(x) =
    blur_model(blur_model(x))`, matching the source's `A^T A` since a
    symmetric real blur kernel is self-adjoint under convolution."""

    def __init__(self, color_channels: int = 3, kernel_size: int = 5, sigma: float = 2.0):
        super().__init__()
        kernel = _fspecial_gauss(kernel_size, sigma)
        weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(color_channels, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.color_channels = color_channels
        self.padding = kernel_size // 2

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.color_channels)

    def gramian(self, x):
        return self.forward(self.forward(x))


# ---------------------------------------------------------------------------
# src/learned_component_resnet_nblock.py::nblock_resnet (verbatim topology;
# TF `tf.get_variable`/`tf.nn.conv2d`/Keras BatchNormalization -> torch
# nn.Conv2d/nn.BatchNorm2d; `leaky_relu(x) = max(0.1x, x)` transcribed as-is)
# ---------------------------------------------------------------------------


def _leaky_relu(x):
    return torch.maximum(0.1 * x, x)


class _ResidualBlock(nn.Module):
    """`nblock_resnet.residual_block`: two 3x3 conv+BN+leaky_relu stages with
    an outer residual (identity) add -- `input + conv_layer_2`."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels, eps=1e-5, momentum=0.01)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels, eps=1e-5, momentum=0.01)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = _leaky_relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = _leaky_relu(h)
        return x + h


class NBlockResNet(nn.Module):
    """`nblock_resnet.network`: patch-mean subtraction -> 1x1 dimension-fit
    conv (3 -> n_intermediate_channels) -> `n_residual_blocks` residual
    blocks -> three 1x1 convs (last one back down to 3 channels, no
    activation on the final one) -> add patch means back."""

    def __init__(
        self,
        color_channels: int = 3,
        n_intermediate_channels: int = 128,
        n_residual_blocks: int = 2,
    ):
        super().__init__()
        self.dimension_fit = nn.Conv2d(color_channels, n_intermediate_channels, kernel_size=1)
        self.residual_blocks = nn.ModuleList(
            [_ResidualBlock(n_intermediate_channels) for _ in range(n_residual_blocks)]
        )
        self.head_conv0 = nn.Conv2d(n_intermediate_channels, n_intermediate_channels, kernel_size=1)
        self.head_conv1 = nn.Conv2d(n_intermediate_channels, n_intermediate_channels, kernel_size=1)
        self.head_conv2 = nn.Conv2d(n_intermediate_channels, color_channels, kernel_size=1)

    def forward(self, x):
        patch_means = x.mean(dim=(2, 3), keepdim=True)
        h = x - patch_means

        h = self.dimension_fit(h)
        for block in self.residual_blocks:
            h = block(h)

        h = self.head_conv0(h)
        h = _leaky_relu(h)
        h = self.head_conv1(h)
        h = _leaky_relu(h)
        h = self.head_conv2(h)

        return h + patch_means


# ---------------------------------------------------------------------------
# src/neumann_network.py::NeumannNet (verbatim unrolled-loop topology; the
# TF constructor also *builds the training graph* including a placeholder
# `true_beta` and `self.output` -- that is training-loop wiring, not
# architecture, so the port exposes the same "Build the network" loop as a
# proper `forward(y)` taking the already-corrupted/measured image `y`
# directly, matching `network_input = forward_adjoint(corruption_model(
# self.true_beta))` with `corruption_model(self.true_beta)` supplied as the
# module input.)
# ---------------------------------------------------------------------------


class NeumannNet(nn.Module):
    def __init__(
        self,
        iterations: int,
        color_channels: int = 3,
        n_intermediate_channels: int = 128,
        n_residual_blocks: int = 2,
    ):
        super().__init__()
        self.num_iters = iterations
        self.forward_operator = DepthwiseGaussianBlur(color_channels=color_channels)
        self.resnet = NBlockResNet(
            color_channels=color_channels,
            n_intermediate_channels=n_intermediate_channels,
            n_residual_blocks=n_residual_blocks,
        )
        # `self.eta = tf.get_variable(name='eta', initializer=0.1, ...,
        # trainable=True)` -- a single learned scalar step size.
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, y):
        """`y` is the corrupted measurement (`corruption_model(true_beta)`
        in the source); `network_input = forward_adjoint(y)`."""
        network_input = self.forward_operator(y)
        network_input = self.eta * network_input
        runner = network_input
        neumann_sum = runner

        for _ii in range(self.num_iters):
            linear_component = runner - self.eta * self.forward_operator.gramian(runner)
            regularizer_output = self.resnet(runner)
            learned_component = -regularizer_output
            runner = linear_component + learned_component
            neumann_sum = neumann_sum + runner

        return neumann_sum


# ---------------------------------------------------------------------------
# Tiny build/example for TorchLens tracing. `n_blocks=6` (real value from
# `driver_neumann_cifar_deblur.py`'s "B in the Neumann networks paper"),
# CIFAR-sized 3-channel image input; `n_intermediate_channels` shrunk from
# the real 128 for a fast trace (channel count is a width knob, not part of
# the architectural topology being ported).
# ---------------------------------------------------------------------------
def build_neumann_network():
    torch.manual_seed(0)
    model = NeumannNet(
        iterations=6, color_channels=3, n_intermediate_channels=16, n_residual_blocks=2
    )
    model.eval()
    return model


def example_input_neumann_network():
    torch.manual_seed(0)
    return torch.randn(2, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "NeumannNetwork",
        "build_neumann_network",
        "example_input_neumann_network",
        2019,
        MENAGERIE_ZOO,
    ),
]
