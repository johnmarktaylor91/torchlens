# FAITHFUL REIMPLEMENTATION of the "activation quantization for INT-only
# inference" family this candidate names ("Activation-Quantized Nets"). Per this
# queue's own triage note, the concept ("co-designing activation with
# quantization ... shifted-ReLU, hardswish for INT-only inference") is described
# across multiple papers with no single canonical "Activation-Quantized Nets"
# repository (confirmed here: GitHub/arXiv search for that exact name surfaces
# no such project). Per the source ladder, this is reimplemented from the two
# papers that together define the concrete, most-cited canonical instance of the
# family -- both equations transcribed exactly, not guessed:
#
#   - Choi et al., "PACT: Parameterized Clipping Activation for Quantized
#     Neural Networks" (arXiv:1805.06085, 2018) -- activation quantization via
#     a learnable per-layer clipping level alpha (paper Eq. 1-2, Sec. 3.2):
#       y = clip(x, 0, alpha) = 0.5*(|x| - |x - alpha| + alpha)
#       y_q = round(y * (2^k - 1) / alpha) * (alpha / (2^k - 1))
#     This IS the "shifted/clipped-ReLU" activation-quantization scheme the
#     candidate description refers to.
#   - Zhou et al., "DoReFa-Net: Training Low Bitwidth Convolutional Neural
#     Networks with Low Bitwidth Gradients" (arXiv:1606.06160, 2016) -- the
#     paired weight-quantization scheme PACT itself cites and composes with
#     for a fully quantized (weights + activations) network:
#       w_q = 2 * quantize_k( tanh(w) / (2*max(|tanh(w)|)) + 0.5 ) - 1
#       quantize_k(x) = round((2^k - 1) * x) / (2^k - 1)
#
# Both quantizers use the standard straight-through estimator (STE: identity
# gradient through `round`, via `x + (round(x) - x).detach()`) so the network
# remains end-to-end differentiable, applied here to a compact conv classifier
# stack matching the INT-only-inference CNN setting both papers target (PACT
# Table 2 evaluates ResNet/AlexNet-style CNNs on ImageNet; this module uses a
# compact 3-conv-layer CNN sized for menagerie-scale tracing).
#
# MENAGERIE_ZOO = "reimpl-pytorch"

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "reimpl-pytorch"


def _ste_round(x):
    """Straight-through estimator round: forward = round(x), backward = identity."""
    return x + (torch.round(x) - x).detach()


def dorefa_quantize_weight(w, k_bits):
    """DoReFa-Net weight quantization (Zhou et al. 2016)."""
    if k_bits >= 32:
        return w
    tanh_w = torch.tanh(w)
    scaled = tanh_w / (2 * tanh_w.abs().max() + 1e-8) + 0.5
    n = float(2**k_bits - 1)
    quantized = _ste_round(scaled * n) / n
    return 2 * quantized - 1


class PACTActivation(nn.Module):
    """PACT parameterized-clipping activation quantization (Choi et al. 2018)."""

    def __init__(self, k_bits=4, alpha_init=10.0):
        super().__init__()
        self.k_bits = k_bits
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        alpha = F.softplus(self.alpha_raw)  # keep the clipping level positive
        y = 0.5 * (x.abs() - (x - alpha).abs() + alpha)  # clip(x, 0, alpha)
        n = float(2**self.k_bits - 1)
        y_q = _ste_round(y * n / alpha) * (alpha / n)
        return y_q


class DoReFaConv2d(nn.Conv2d):
    """Conv2d with DoReFa-quantized weights; activations are quantized
    separately by a following PACTActivation module (the paper's combined
    "PACT + DoReFa" fully-quantized-network recipe)."""

    def __init__(self, *args, w_bits=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_bits = w_bits

    def forward(self, x):
        w_q = dorefa_quantize_weight(self.weight, self.w_bits)
        return self._conv_forward(x, w_q, self.bias)


class ActivationQuantizedCNN(nn.Module):
    """Compact CNN with DoReFa-quantized conv weights and PACT-quantized
    activations at every layer -- the canonical "activation-quantized network"
    for INT-only inference (Choi et al. 2018 / Zhou et al. 2016)."""

    def __init__(self, in_channels=3, num_classes=10, w_bits=4, a_bits=4):
        super().__init__()
        self.conv1 = DoReFaConv2d(in_channels, 32, kernel_size=3, padding=1, w_bits=w_bits)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = PACTActivation(k_bits=a_bits)

        self.conv2 = DoReFaConv2d(32, 64, kernel_size=3, stride=2, padding=1, w_bits=w_bits)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = PACTActivation(k_bits=a_bits)

        self.conv3 = DoReFaConv2d(64, 128, kernel_size=3, stride=2, padding=1, w_bits=w_bits)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = PACTActivation(k_bits=a_bits)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_activation_quantized_cnn():
    return ActivationQuantizedCNN(in_channels=3, num_classes=10, w_bits=4, a_bits=4)


def example_input_activation_quantized_cnn():
    return torch.rand(2, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "Activation-Quantized Nets (PACT+DoReFa)",
        "build_activation_quantized_cnn",
        "example_input_activation_quantized_cnn",
        2018,
        MENAGERIE_ZOO,
    ),
]
