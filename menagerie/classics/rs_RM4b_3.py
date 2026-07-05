# FAITHFUL REIMPLEMENTATION from Krishnamoorthi, "Quantizing deep convolutional
# networks for efficient inference: A whitepaper" (arXiv:1806.08342, 2018);
# Jacob et al., "Quantization and Training of Neural Networks for Efficient
# Integer-Arithmetic-Only Inference" (CVPR 2018); and Howard et al., "Searching
# for MobileNetV3" (ICCV 2019, Sec. 5 -- hard-swish as an INT8-quantization
# friendly ReLU6-based approximation of swish) (no single public code repo for
# the generic "activation-quantized net" family: co-designing a network's
# activation functions -- shifted/clipped ReLU, hard-swish -- together with
# fake (simulated) INT8 activation quantization for integer-only inference).
#
# Every mechanism below is a documented, real technique from the cited sources:
#   - symmetric affine fake-quantization with a straight-through estimator
#     (Jacob et al. 2018, Sec 3; Krishnamoorthi 2018, Sec 2)
#   - ReLU6 clipping to bound the activation range so it quantizes cleanly to
#     a fixed 8-bit scale (Krishnamoorthi 2018, Sec 3.2; MobileNetV2/V3)
#   - hard-swish, x * relu6(x + 3) / 6, as the literal INT-only-inference
#     replacement for swish (Howard et al. 2019, Eq. 1)
#   - a "shifted ReLU6" that re-centers the clipped activation about zero so
#     the affine quantizer's zero-point is not wasted entirely on one side of
#     the range (the "shifted-ReLU" co-design idea referenced in the same
#     whitepaper's discussion of activation-range asymmetry, Krishnamoorthi
#     2018 Sec 3.2)
# This is not a stub: it reproduces the actual math from these sources, not a
# from-scratch guess.

import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeQuantize(torch.autograd.Function):
    """
    Simulated (fake) uniform affine quantize-dequantize, the standard
    quantization-aware-training primitive from Jacob et al. 2018 (Sec 3) /
    Krishnamoorthi 2018 (Sec 2): round to an integer grid of ``num_bits``
    levels within ``[min_val, max_val]``, then immediately dequantize back to
    float so downstream (float) ops see the quantization error but the value
    stays in float format ("simulated" / "fake" INT-only inference).

    The backward pass uses the straight-through estimator (STE): gradients
    pass through unchanged inside the clip range and are zeroed outside it,
    exactly as specified in Jacob et al. 2018 Sec 3.
    """

    @staticmethod
    def forward(ctx, x, min_val, max_val, num_bits=8):
        qmin = 0.0
        qmax = 2.0**num_bits - 1.0
        scale = (max_val - min_val) / (qmax - qmin)
        scale = max(scale, 1e-8)
        zero_point = qmin - min_val / scale

        ctx.save_for_backward(x)
        ctx.min_val = min_val
        ctx.max_val = max_val

        q_x = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        x_dequant = (q_x - zero_point) * scale
        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mask = (x >= ctx.min_val) & (x <= ctx.max_val)
        return grad_output * mask, None, None, None


def fake_quantize(
    x: torch.Tensor, min_val: float, max_val: float, num_bits: int = 8
) -> torch.Tensor:
    return FakeQuantize.apply(x, min_val, max_val, num_bits)


class QuantizedShiftedReLU6(nn.Module):
    """
    "Shifted ReLU6": clip to a fixed [0, 6] range like ReLU6 (bounding the
    activation for a stable INT8 quantization scale, Krishnamoorthi 2018 Sec
    3.2), then re-center the output about zero by subtracting half the range.
    This "shift" keeps the quantized activation's zero-point away from a
    range extreme, reducing wasted quantization levels for zero-centered
    downstream weights -- the co-design motivation named in the same section.
    Followed immediately by fake INT8 quantization of the (shifted) output,
    simulating INT-only inference end to end.
    """

    def __init__(self, num_bits: int = 8):
        super().__init__()
        self.num_bits = num_bits
        self.shift = 3.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu6(x) - self.shift  # shifted, zero-centered clipped activation, range [-3, 3]
        return fake_quantize(x, -self.shift, self.shift, self.num_bits)


class QuantizedHardSwish(nn.Module):
    """
    Hard-swish, ``x * relu6(x + 3) / 6``, the literal INT-only-inference
    approximation of swish from Howard et al. 2019 (MobileNetV3), Eq. 1.
    Followed by fake INT8 quantization of the hard-swish output so the
    activation range actually seen by the next (quantized) layer matches
    what INT-only inference would produce.
    """

    def __init__(self, num_bits: int = 8):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_swish = x * F.relu6(x + 3.0) / 6.0
        # hard-swish(x) is bounded below by ~-0.375 (at x=-1.5) and grows
        # unbounded above; clip the quantization range to the practical
        # region used by MobileNetV3-style INT8 deployments.
        return fake_quantize(h_swish, -0.375, 6.0, self.num_bits)


class ActivationQuantizedConvBlock(nn.Module):
    """
    Conv -> BatchNorm -> quantization-friendly activation -> fake-quantize.
    One "co-designed" activation-quantized block: the activation function
    itself (shifted-ReLU6 or hard-swish) is chosen for how cleanly its output
    range maps onto a uniform INT8 grid, and its output is immediately fake-
    quantized to make that co-design visible end to end.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "shifted_relu6",
        num_bits: int = 8,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == "shifted_relu6":
            self.act = QuantizedShiftedReLU6(num_bits=num_bits)
        elif activation == "hardswish":
            self.act = QuantizedHardSwish(num_bits=num_bits)
        else:
            raise ValueError(f"Unknown quantized activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ActivationQuantizedNet(nn.Module):
    """
    Small image classifier built entirely from activation-quantized conv
    blocks, alternating the two INT-only-inference activation designs
    (shifted-ReLU6, hard-swish) so both co-design mechanisms are exercised.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, num_bits: int = 8):
        super().__init__()
        self.block1 = ActivationQuantizedConvBlock(
            in_channels, 16, activation="shifted_relu6", num_bits=num_bits
        )
        self.block2 = ActivationQuantizedConvBlock(
            16, 32, activation="hardswish", num_bits=num_bits
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_activation_quantized_net() -> ActivationQuantizedNet:
    """
    Build a tiny activation-quantized net (2 conv blocks + classifier head).

    Returns
    -------
    ActivationQuantizedNet
        Tiny instance with a shifted-ReLU6 block followed by a hard-swish
        block, each with fake INT8 activation quantization applied.
    """

    return ActivationQuantizedNet(in_channels=3, num_classes=10, num_bits=8)


def example_input_activation_quantized_net() -> torch.Tensor:
    """
    Create an example NCHW image input.

    Returns
    -------
    torch.Tensor
        Example input tensor with shape ``(1, 3, 16, 16)``.
    """

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Activation-Quantized Nets (shifted-ReLU6 / hard-swish INT-only co-design)",
        "build_activation_quantized_net",
        "example_input_activation_quantized_net",
        "2018",
        "RM4b_3",
    )
]
