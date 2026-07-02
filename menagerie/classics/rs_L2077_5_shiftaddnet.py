# SOURCE: vendored from GATECH-EIC/ShiftAddNet @ main
# https://raw.githubusercontent.com/GATECH-EIC/ShiftAddNet/main/models/resnet20_shiftadd_se.py
# https://raw.githubusercontent.com/GATECH-EIC/ShiftAddNet/main/se_shift/conv_mask_shift.py
# https://raw.githubusercontent.com/GATECH-EIC/ShiftAddNet/main/se_shift/utils_quantize.py
# https://raw.githubusercontent.com/GATECH-EIC/ShiftAddNet/main/deepshift/ste.py
# https://raw.githubusercontent.com/GATECH-EIC/ShiftAddNet/main/deepshift/utils.py
# https://raw.githubusercontent.com/GATECH-EIC/ShiftAddNet/main/adder/adder_slow.py
#
# You, Chen, Zhang, Li, Wang, Lin, 2020 (NeurIPS 2020) "ShiftAddNet: A Hardware-
# Inspired Deep Network". ShiftAddNet replaces standard convolutions with a
# cascade of a bit-shift convolution (`SEConv2d`, from `se_shift/`, powers-of-2
# weights selected via a learnable sign/shift-magnitude reparameterization) and
# an L1-distance-based "adder" convolution (`Adder2D` in `adder/`, computing
# `-|W - X|` instead of `W @ X`) -- the multiplication-free shift+add cascade
# IS the paper's whole architectural contribution, so this is vendored (real
# code), not built from a stock library conv/resnet class.
#
# The real `adder/adder.py::Adder2D` compiles a custom CUDA extension at
# import time (`torch.utils.cpp_extension.load(...)`), which this task's base
# env cannot build. The SAME official repo ships a pure-torch reference
# implementation of the identical adder-convolution math (`adder/adder_slow.py
# ::adder2d`, using `torch.cdist`/`unfold` instead of the custom CUDA kernel)
# -- it is vendored here in place of the CUDA `Adder2D`, wrapped in a thin
# `Adder2D` adapter that accepts (and no-ops, matching the CUDA class's
# `quantize=False` default no-op branch) the `quantize`/`weight_bits`/
# `sparsity` kwargs the real `resnet20_shiftadd_se.py` passes at every call
# site with defaults `quantize=False` -- i.e. the code path actually exercised
# below is architecturally identical to the CUDA version's default behavior.
#
# `se_shift/conv_mask_shift.py::SEConv2d` is reproduced verbatim (only the
# unused `VEC_2_SHAPE` import -- a plain lookup-table dict from `se_shift/
# alg.py`, never referenced anywhere in `SEConv2d`'s actual forward path --
# is dropped rather than dragging in `alg.py`'s `joblib`/`scipy.io` imports).
# `deepshift/ste.py` and `deepshift/utils.py`'s `round`/`round_power_of_2`/
# `sign`/`clamp` straight-through-estimator ops and `se_shift/utils_quantize.py
# ::sparsify_and_nearestpow2` are reproduced verbatim (only the functions
# `SEConv2d` actually calls).
#
# `models/resnet20_shiftadd_se.py::ResNet`/`BasicBlock`/`conv3x3`/
# `resnet20_shiftadd_se` are reproduced verbatim (the full CIFAR-style
# ResNet-20 shift+add backbone; the paper's headline architecture).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torch.autograd.function import Function as _AutogradFunction  # noqa: F401 (parity import)
from torch.nn.modules.utils import _pair

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# deepshift/utils.py :: round_to_fixed / round_power_of_2 / round (verbatim,
# functions actually used by deepshift/ste.py's straight-through wrappers)
# ============================================================================


def _ds_round_power_of_2(x, rounding="deterministic"):
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    shift = _ds_round(torch.log(x_abs) / math.log(2), rounding)
    x_rounded = (2.0**shift) * sign
    return x_rounded


def _ds_round(x, rounding="deterministic"):
    assert rounding in ["deterministic", "stochastic"]
    if rounding == "stochastic":
        x_floor = x.floor()
        return x_floor + torch.bernoulli(x - x_floor)
    else:
        return x.round()


# ============================================================================
# deepshift/ste.py (verbatim, straight-through-estimator Functions used by
# SEConv2d.reset_parameters)
# ============================================================================


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, input, rounding="deterministic"):
        return _ds_round(input, rounding)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def ste_round(input, rounding="deterministic"):
    return RoundFunction.apply(input, rounding)


class SignFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_sign(input):
    return SignFunction.apply(input)


class ClampFunction(Function):
    @staticmethod
    def forward(ctx, input, min, max):
        return torch.clamp(input, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def ste_clamp(input, min, max):
    return ClampFunction.apply(input, min, max)


# ============================================================================
# se_shift/utils_quantize.py :: sparsify_and_nearestpow2 (verbatim, the one
# function SEConv2d.sparsify_and_quantize_weight actually calls)
# ============================================================================


class SparsifyAndNearestPow2(Function):
    @staticmethod
    def forward(ctx, input, threshold, eps=1e-5):
        with torch.no_grad():
            output = input.new_zeros(input.size())
            input_sign = input.sign()
            input_abs = input.abs()

            nnz_idx = input_abs >= (threshold - eps)
            input_abs_nnz = input_abs[nnz_idx]

            nextpow2 = 2 ** input_abs_nnz.log2().ceil()
            prevpow2 = nextpow2 / 2.0
            lerr = input_abs_nnz - prevpow2
            rerr = nextpow2 - input_abs_nnz
            lbetter = (lerr < rerr).float()
            output_abs_nnz = prevpow2 * lbetter + nextpow2 * (1 - lbetter)

            output[nnz_idx] = output_abs_nnz * input_sign[nnz_idx]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None


def sparsify_and_nearestpow2(input, threshold):
    return SparsifyAndNearestPow2().apply(input, threshold)


# ============================================================================
# se_shift/conv_mask_shift.py :: SEConv2d (verbatim, minus the unused
# VEC_2_SHAPE import -- see header note)
# ============================================================================


def round_act_to_fixed(input, bits=16):
    if bits == 1:
        return torch.sign(input)
    S = 2.0 ** (bits - 1)
    input_round = torch.round(input * S) / S
    return input_round


class RoundActFixedPoint(Function):
    @staticmethod
    def forward(ctx, input, bits):
        return round_act_to_fixed(input, bits)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def round_act_fixed_point(input, bits):
    return RoundActFixedPoint.apply(input, bits)


def dynamic_range_for_sign(sign, threshold):
    sign[sign < -threshold] = -1
    sign[sign > threshold] = 1
    sign[(-threshold <= sign) & (sign <= threshold)] = 0
    return sign


class RoundFunctionSign(Function):
    @staticmethod
    def forward(ctx, input, threshold):
        return dynamic_range_for_sign(input, threshold)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def round_sign(input, threshold):
    return RoundFunctionSign.apply(input, threshold)


class SEConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        size_splits=64,
        threshold=5e-3,
        sign_threshold=0.5,
        distribution="uniform",
    ):
        super(SEConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.size_splits = size_splits
        self.sign_threshold = sign_threshold
        self.distribution = distribution
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.weight = torch.nn.Parameter(
            nn.init.normal_(
                torch.randn(self.out_channels, self.in_channels, kernel_size, kernel_size)
            )
        )
        self.p = torch.nn.Parameter(
            nn.init.uniform_(
                torch.randn(self.out_channels, self.in_channels, kernel_size, kernel_size)
            )
        )
        self.s = torch.nn.Parameter(
            nn.init.uniform_(
                torch.randn(self.out_channels, self.in_channels, kernel_size, kernel_size)
            )
        )
        self.register_buffer("mask", torch.Tensor(*self.weight.size()).float())
        self.threshold = threshold
        for i in range(-10, 1):
            if 2**i >= threshold:
                self.min_p = -i
                break
        self.shift_range = (-1 * self.min_p, 0)

        self.reset_parameters()

    def reset_dweight_counter(self):
        self.dweight_counter = self.weight.new_zeros(self.weight.size()).float()

    def reset_parameters(self):
        if self.distribution == "kaiming_normal":
            init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
            self.set_mask()  # quantize
            self.s.data.uniform_(-1, 1)
            sign = ste_sign(round_sign(self.s, self.sign_threshold))
            self.weight.data *= abs(sign)
        else:
            if self.distribution == "uniform":
                self.p.data.uniform_(-self.min_p - 0.5, -1 + 0.5)
            elif self.distribution == "normal":
                self.p.data.normal_(-self.min_p / 2, 1)
            self.p.data = ste_clamp(self.p.data, *self.shift_range)
            self.p.data = ste_round(self.p.data, "deterministic")
            self.s.data.uniform_(-1, 1)
            sign = ste_sign(round_sign(self.s, self.sign_threshold))
            self.weight.data = sign * (2**self.p.data)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def set_mask(self):
        self.weight.data = self.sparsify_and_quantize_weight(mask=False)
        self.mask.data = (self.weight != 0.0).float()
        assert self.mask.requires_grad is False

    def sparsify_and_quantize_weight(self, mask=True):
        qweight = sparsify_and_nearestpow2(self.weight, self.threshold)
        if mask:
            qweight = qweight * self.mask
        return qweight

    def get_weight(self, mask=True):
        qweight = self.sparsify_and_quantize_weight(mask=mask)
        return qweight

    def forward(self, input):
        weight = self.weight
        input = round_act_fixed_point(input, bits=16)

        output = F.conv2d(
            input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return output


# ============================================================================
# adder/adder_slow.py (verbatim) -- pure-torch reference implementation of
# the same L1-distance "adder" convolution the real (CUDA-extension) Adder2D
# computes; see header note.
# ============================================================================


def _new_cdist(p, eta):
    class cdist(torch.autograd.Function):
        @staticmethod
        def forward(ctx, W, X):
            ctx.save_for_backward(W, X)
            out = -torch.cdist(W, X, p)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            W, X = ctx.saved_tensors
            grad_W = grad_X = None
            if ctx.needs_input_grad[0]:
                X_unsqueeze = torch.unsqueeze(X, 0).expand(W.shape[0], X.shape[0], X.shape[1])
                W_unsqueeze = torch.unsqueeze(W, 1).expand(W.shape[0], X.shape[0], W.shape[1])
                grad_unsqueeze = torch.unsqueeze(grad_output, 2).expand(
                    grad_output.shape[0], grad_output.shape[1], W.shape[1]
                )
                grad_W = ((X_unsqueeze - W_unsqueeze) * grad_unsqueeze).sum(1)
                grad_W = eta * (grad_W.numel() ** 0.5) / torch.norm(grad_W) * grad_W
            if ctx.needs_input_grad[1]:
                grad_X = (
                    torch.nn.functional.hardtanh(
                        (W_unsqueeze - X_unsqueeze), min_val=-1.0, max_val=1.0
                    )
                    * grad_unsqueeze
                ).sum(0)
            return grad_W, grad_X

    return cdist().apply


_ADDER_ETA = 0.2
_adder_cdist = _new_cdist(1, _ADDER_ETA)


def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)

    X_col = torch.nn.functional.unfold(
        X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride
    ).view(n_x, -1, h_out * w_out)
    X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
    W_col = W.view(n_filters, -1)

    out = _adder_cdist(W_col, X_col.transpose(0, 1))

    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()

    return out


class _AdderSlow2D(nn.Module):
    """Verbatim `adder/adder_slow.py::adder2d` (pure-torch reference adder
    convolution, no custom CUDA extension)."""

    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False):
        super(_AdderSlow2D, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(
            nn.init.normal_(torch.randn(output_channel, input_channel, kernel_size, kernel_size))
        )
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x, self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return output


class Adder2D(nn.Module):
    """Adapter over the real, pure-torch `_AdderSlow2D` matching the call
    signature `models/resnet20_shiftadd_se.py` uses for the CUDA
    `adder/adder.py::Adder2D` (`quantize=`/`weight_bits=`/`sparsity=` kwargs).
    Every call site in the vendored ResNet passes `quantize=False` (the
    default), which is the CUDA class's own no-op branch for those kwargs --
    so this adapter's forward path is architecturally identical to the real
    Adder2D's default behavior; only the quantization side-branches (never
    exercised by the default construction below) are omitted."""

    def __init__(
        self,
        input_channel,
        output_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        eta=0.2,
        quantize=False,
        weight_bits=8,
        sparsity=0,
        momentum=0.9,
        quantize_v="sbm",
    ):
        super().__init__()
        if quantize:
            raise NotImplementedError(
                "quantize=True path of the CUDA-backed Adder2D is not "
                "reproduced by this pure-torch adapter (see module header)."
            )
        self._impl = _AdderSlow2D(
            input_channel, output_channel, kernel_size, stride=stride, padding=padding, bias=bias
        )

    def forward(self, input):
        return self._impl(input)


# ============================================================================
# models/resnet20_shiftadd_se.py (verbatim)
# ============================================================================


def conv3x3(
    in_planes,
    out_planes,
    threshold,
    sign_threshold,
    distribution,
    stride=1,
    quantize=False,
    weight_bits=8,
    sparsity=0,
):
    "3x3 convolution with padding"
    shift = SEConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        threshold=threshold,
        sign_threshold=sign_threshold,
        distribution=distribution,
    )
    add = Adder2D(
        out_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        quantize=quantize,
        weight_bits=weight_bits,
        sparsity=sparsity,
    )
    return nn.Sequential(shift, add)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        threshold,
        sign_threshold,
        distribution,
        stride=1,
        downsample=None,
        quantize=False,
        weight_bits=8,
        sparsity=0,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(
            inplanes,
            planes,
            threshold=threshold,
            sign_threshold=sign_threshold,
            distribution=distribution,
            stride=stride,
            quantize=quantize,
            weight_bits=weight_bits,
            sparsity=sparsity,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(
            planes,
            planes,
            threshold=threshold,
            sign_threshold=sign_threshold,
            distribution=distribution,
            quantize=quantize,
            weight_bits=weight_bits,
            sparsity=sparsity,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes,
        threshold,
        sign_threshold,
        distribution,
        quantize=False,
        weight_bits=8,
        sparsity=0,
    ):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.quantize = quantize
        self.threshold = threshold
        self.sign_threshold = sign_threshold
        self.distribution = distribution
        self.weight_bits = weight_bits
        self.sparsity = sparsity
        self.conv1 = SEConv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            threshold=threshold,
            sign_threshold=sign_threshold,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # use conv as fc layer (addernet)
        self.fc = SEConv2d(
            64 * block.expansion,
            num_classes,
            1,
            bias=False,
            threshold=threshold,
            sign_threshold=sign_threshold,
        )
        self.bn2 = nn.BatchNorm2d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SEConv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    threshold=self.threshold,
                    sign_threshold=self.sign_threshold,
                    distribution=self.distribution,
                ),  # shift
                Adder2D(
                    planes * block.expansion,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    quantize=self.quantize,
                    weight_bits=self.weight_bits,
                    sparsity=self.sparsity,
                ),  # add
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                threshold=self.threshold,
                sign_threshold=self.sign_threshold,
                distribution=self.distribution,
                stride=stride,
                downsample=downsample,
                quantize=self.quantize,
                weight_bits=self.weight_bits,
                sparsity=self.sparsity,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    threshold=self.threshold,
                    sign_threshold=self.sign_threshold,
                    distribution=self.distribution,
                    quantize=self.quantize,
                    weight_bits=self.weight_bits,
                    sparsity=self.sparsity,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)
        return x.view(x.size(0), -1)


def resnet20_shiftadd_se(threshold, sign_threshold, distribution, num_classes=10, **kwargs):
    return ResNet(
        BasicBlock,
        [3, 3, 3],
        num_classes=num_classes,
        threshold=threshold,
        sign_threshold=sign_threshold,
        distribution=distribution,
        **kwargs,
    )


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_shiftaddnet():
    torch.manual_seed(0)
    model = resnet20_shiftadd_se(
        threshold=5e-3,
        sign_threshold=0.5,
        distribution="uniform",
        num_classes=10,
    )
    model.eval()
    return model


def example_input_shiftaddnet():
    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("ShiftAddNet", "build_shiftaddnet", "example_input_shiftaddnet", 2020, "vendored-pytorch"),
]
