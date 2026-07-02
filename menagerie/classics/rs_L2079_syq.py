# FAITHFUL PORT of julianfaraone/SYQ @ master (original framework: TensorFlow 1.x /
# tensorpack, Python 2)
# https://raw.githubusercontent.com/julianfaraone/SYQ/master/tensorpack/examples/SYQ-AlexNet/syq-alexnet.py
# https://raw.githubusercontent.com/julianfaraone/SYQ/master/tensorpack/examples/SYQ-AlexNet/quantize.py
#
# Faraone, Fraser, Blott, Leong, 2018 (CVPR 2018) "SYQ: Learning Symmetric
# Quantization for Efficient Deep Neural Networks". The real repo is TF1.x
# tensorpack (ancient `tf.get_variable`/`tf.select`/`tf.scalar_summary` APIs,
# `.pyc`-only modules, Python-2-only `map()`-as-list usage) and cannot run in
# a modern base torch/TF env, so the architecture is TRANSCRIBED faithfully
# from the real `syq-alexnet.py` + `quantize.py` source (not guessed from the
# paper): an AlexNet backbone (conv0 unquantized -> conv1..conv4 grouped
# convs with `split=2` matching the original 2-GPU AlexNet split -> fc0/fc1
# fine-grained-quantized -> fct unquantized final classifier), with SYQ's two
# real mechanisms ported 1:1:
#   1. `fine_grained_quant`: per-output-channel-group WEIGHT quantization --
#      `w = sign(x) * mask_z * masker`, where `masker` applies a learned
#      per-(kernel_h, kernel_w) location scale `Ws` to elements whose
#      magnitude exceeds a data-dependent threshold `eta * max(|x|)` (ternary
#      {-Ws, 0, +Ws} weights for conv layers; `binary=True` -> `mask_z` is
#      identity so weights are the true fine-grained SYQ representation).
#      conv0 and the final fc ("fct") are left UNQUANTIZED, exactly matching
#      the real `new_get_variable` monkey-patch's skip condition
#      (`name != 'W' or 'conv0' in v.op.name or 'fct' in v.op.name`).
#   2. `quantize` (activations): `floor(x * (2**BITA - 1) + 0.5) / (2**BITA - 1)`
#      fixed-point rounding after `relu -> clip[0, 1]`, with `BITA=8` matching
#      the repo's default.
# The straight-through-estimator `gradient_override_map` calls in the real
# code only affect the backward pass (irrelevant for a random-init forward
# trace) and are omitted; `tf.stop_gradient` on `eta_x`/`mask_z` is likewise
# a training-only no-op for a pure forward pass. BatchNorm/MaxPool/ReLU
# structure, layer order, channel counts, kernel sizes, strides, and the
# grouped-conv `split=2` pattern for conv1/conv3/conv4 are preserved exactly
# from `syq-alexnet.py`.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _syq_quantize_activation(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Port of quantize.py::quantize(x, k=BITA, fraclength=None) applied after the
    relu+clip[0,1] the real activate() performs (forward-only; STE omitted)."""
    x = F.relu(x)
    x = torch.clamp(x, 0.0, 1.0)
    n = float(2**bits - 1)
    return torch.floor(x * n + 0.5) / n


class FineGrainedQuantWeight(nn.Module):
    """Port of quantize.py::fine_grained_quant for conv weights (binary=True path,
    i.e. mask_z == 1 -- matches the real repo's default call site). Learns one
    scale `Ws` per (kernel_h, kernel_w) spatial location, applied to any weight
    element whose magnitude exceeds a data-dependent threshold `eta * max(|W|)`.
    Ternary-like effective weight: {-Ws, 0(masked by sign only when below
    threshold -> falls back to +-1 * 1.0), +Ws} per spatial location."""

    def __init__(self, kh: int, kw: int, eta: float = 0.05):
        super().__init__()
        self.eta = eta
        # real repo: w_s shape [(kh*kw), 1], initialized to 1.0 (non-INITIAL branch)
        self.w_s = nn.Parameter(torch.ones(kh * kw, 1))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        # weight layout here is torch conv layout: (out_ch, in_ch/groups, kh, kw)
        out_ch, in_ch_g, kh, kw = weight.shape
        eta_x = (weight.abs().max() * self.eta).detach()

        ws_grid = self.w_s.view(kh, kw)  # per-(kh,kw) scale, broadcast over out/in ch
        pos_mask = weight > eta_x
        neg_mask = weight < -eta_x
        scale = ws_grid.view(1, 1, kh, kw).expand_as(weight)
        masker = torch.where(pos_mask | neg_mask, scale, torch.ones_like(weight))

        w = torch.sign(weight) * masker
        return w


class FineGrainedQuantConv2d(nn.Module):
    """Conv2d whose weight is fine-grained-quantized at forward time via
    FineGrainedQuantWeight, matching SYQ's `new_get_variable` monkeypatch applied
    to every conv layer's 'W' except conv0 (see module header)."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, quantize_w=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.quantize_w = quantize_w
        if quantize_w:
            kh, kw = self.conv.kernel_size
            self.quantizer = FineGrainedQuantWeight(kh, kw)
        else:
            self.quantizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv.weight
        if self.quantize_w:
            weight = self.quantizer(weight)
        return F.conv2d(
            x,
            weight,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
        )


class FineGrainedQuantLinear(nn.Module):
    """FC counterpart of fine_grained_quant's else-branch (single scalar `Wn` scale
    instead of a per-pixel grid, matching the real repo's non-conv path)."""

    def __init__(self, in_features, out_features, quantize_w=True, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.quantize_w = quantize_w
        if quantize_w:
            self.wn = nn.Parameter(torch.tensor(1.0))
        self.eta = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fc.weight
        if self.quantize_w:
            eta_x = (weight.abs().max() * self.eta).detach()
            pos_mask = weight > eta_x
            neg_mask = weight < -eta_x
            masker = torch.where(
                pos_mask | neg_mask, self.wn.expand_as(weight), torch.ones_like(weight)
            )
            weight = torch.sign(weight) * masker
        return F.linear(x, weight, self.fc.bias)


class SYQAlexNet(nn.Module):
    """Port of syq-alexnet.py::Model._build_graph -- SYQ-quantized AlexNet.
    conv0 and the final classifier ('fct') keep full-precision weights (matches
    the real repo's `new_get_variable` skip condition); all other conv/fc weights
    are fine-grained SYQ-quantized, and every intermediate activation is
    relu -> clip[0,1] -> 8-bit fixed-point quantized (BITA=8)."""

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.conv0 = FineGrainedQuantConv2d(3, 96, 12, stride=4, padding=0, quantize_w=False)

        self.conv1 = FineGrainedQuantConv2d(96, 256, 5, stride=1, padding=2, groups=2)
        self.bn1 = nn.BatchNorm2d(256, momentum=0.1, eps=1e-4)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = FineGrainedQuantConv2d(256, 384, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(384, momentum=0.1, eps=1e-4)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv3 = FineGrainedQuantConv2d(384, 384, 3, stride=1, padding=1, groups=2)
        self.bn3 = nn.BatchNorm2d(384, momentum=0.1, eps=1e-4)

        self.conv4 = FineGrainedQuantConv2d(384, 256, 3, stride=1, padding=1, groups=2)
        self.bn4 = nn.BatchNorm2d(256, momentum=0.1, eps=1e-4)
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=0)

        # spatial size after conv0(k12,s4)->pool1->pool2->pool4 on a 224x224 input:
        # 224->54(conv0)->54->27(pool1)->27->14(pool2)->14->14->6(pool4) => 256*6*6
        self._flatten_dim = 256 * 6 * 6

        self.fc0 = FineGrainedQuantLinear(self._flatten_dim, 4096, bias=False)
        self.bnfc0 = nn.BatchNorm1d(4096, momentum=0.1, eps=1e-4)

        self.fc1 = FineGrainedQuantLinear(4096, 4096, bias=False)
        self.bnfc1 = nn.BatchNorm1d(4096, momentum=0.1, eps=1e-4)

        self.fct = FineGrainedQuantLinear(4096, num_classes, quantize_w=False, bias=True)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        return _syq_quantize_activation(x, bits=8)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image / 255.0

        x = self.conv0(x)
        x = self._activate(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self._activate(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self._activate(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self._activate(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool4(x)
        x = self._activate(x)

        x = x.flatten(1)

        x = self.fc0(x)
        x = self.bnfc0(x)
        x = self._activate(x)

        x = self.fc1(x)
        x = self.bnfc1(x)
        x = self._activate(x)

        logits = self.fct(x)
        return logits


# ============================================================================
# menagerie staging glue
# ============================================================================


def build_syq_alexnet():
    torch.manual_seed(0)
    model = SYQAlexNet(num_classes=10)
    model.eval()
    return model


def example_input_syq_alexnet():
    torch.manual_seed(0)
    return torch.randn(2, 3, 224, 224) * 64 + 128


MENAGERIE_ENTRIES = [
    ("SYQ-AlexNet", "build_syq_alexnet", "example_input_syq_alexnet", 2018, "ported-pytorch"),
]
