# SOURCE: vendored from mostafaelhoushi/DeepShift @ master (pytorch/deepshift/{modules.py,
# ste.py, utils.py, convert.py} + pytorch/cifar10_models/resnet.py). DeepShift = "DeepShift:
# Towards Multiplication-Less Neural Networks" (Elhoushi et al.). LinearShift/Conv2dShift (the
# DeepShift-PS variant: parameterized ternary-sign + power-of-2-shift weights, trained with a
# straight-through estimator) are the repo's actual multiplication-free layer replacements for
# nn.Linear/nn.Conv2d; convert_to_shift() is the repo's own mechanism for converting an arbitrary
# base network into its DeepShift counterpart (used throughout the repo's CIFAR10/ImageNet
# scripts). Architecture is unchanged from the repo. The only adaptation: the repo's `use_kernel`
# path calls into a compiled `deepshift.kernels` CUDA/CPU extension (build-from-source, not
# installed); we never set `use_kernel=True`, so that dependency is dropped along with the one
# `utils.compress_bits` helper that is exclusively used by it (not on this model's forward path).
# The base network converted here is the repo's own CIFAR10 ResNet (pytorch/cifar10_models/
# resnet.py, resnet20(), option='A' -- the exact "CIFAR10 ResNet paper" architecture the repo's
# own scripts train/convert), scaled down only via num_classes/input resolution for a fast trace.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import init
from torch.nn.modules.utils import _pair

MENAGERIE_ZOO = "vendored-pytorch"

log2 = math.log(2)

# --- vendored from DeepShift pytorch/deepshift/utils.py (kernel-only compress_bits dropped) ---


def round_to_fixed(input, integer_bits=16, fraction_bits=16):
    assert integer_bits >= 1, integer_bits
    if integer_bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -(fraction_bits))
    bound = math.pow(2.0, integer_bits - 1)
    min_val = -bound
    max_val = bound - 1
    rounded = torch.floor(input / delta) * delta

    clipped_value = torch.clamp(rounded, min_val, max_val)
    return clipped_value


def ds_round(x, rounding="deterministic"):
    assert rounding in ["deterministic", "stochastic"]
    if rounding == "stochastic":
        x_floor = x.floor()
        return x_floor + torch.bernoulli(x - x_floor)
    else:
        return x.round()


def get_shift_and_sign(x, rounding="deterministic"):
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    shift = ds_round(torch.log(x_abs) / math.log(2), rounding)
    return shift, sign


# --- vendored from DeepShift pytorch/deepshift/ste.py (straight-through estimator ops) ---


class RoundFixedPoint(Function):
    @staticmethod
    def forward(ctx, input, act_integer_bits=16, act_fraction_bits=16):
        return round_to_fixed(input, act_integer_bits, act_fraction_bits)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def round_fixed_point(input, act_integer_bits=16, act_fraction_bits=16):
    return RoundFixedPoint.apply(input, act_integer_bits, act_fraction_bits)


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, input, rounding="deterministic"):
        return ds_round(input, rounding)

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


class UnsymmetricGradMulFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        return torch.mul(input1, input2)

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        return grad_output * input2, grad_output


def unsym_grad_mul(input1, input2):
    return UnsymmetricGradMulFunction.apply(input1, input2)


# --- vendored from DeepShift pytorch/deepshift/modules.py (DeepShift-PS layers; use_kernel path
# and its Function-based autograd.Function forward/backward dropped -- forward-only trace never
# takes that branch since use_kernel=False is the constructor default we always use) ---


class LinearShift(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        check_grad=False,
        freeze_sign=False,
        rounding="deterministic",
        weight_bits=5,
        act_integer_bits=16,
        act_fraction_bits=16,
    ):
        super(LinearShift, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.check_grad = check_grad
        self.rounding = rounding
        self.shift_range = (-1 * (2 ** (weight_bits - 1) - 2), 0)
        self.act_integer_bits, self.act_fraction_bits = act_integer_bits, act_fraction_bits

        tensor_constructor = torch.DoubleTensor if check_grad else torch.Tensor

        self.shift = nn.Parameter(tensor_constructor(out_features, in_features))
        self.sign = nn.Parameter(
            tensor_constructor(out_features, in_features), requires_grad=(freeze_sign is False)
        )

        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.shift.data.uniform_(*self.shift_range)
        self.sign.data.uniform_(-1, 1)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.shift)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        self.shift.data = ste_clamp(self.shift.data, *self.shift_range)
        shift_rounded = ste_round(self.shift, rounding=self.rounding)
        sign_rounded_signed = ste_sign(ste_round(self.sign, rounding=self.rounding))
        weight_ps = unsym_grad_mul(2**shift_rounded, sign_rounded_signed)

        return torch.nn.functional.linear(input, weight_ps, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class _ConvNdShift(nn.Module):
    __constants__ = ["stride", "padding", "dilation", "groups", "bias", "padding_mode"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        check_grad=False,
        freeze_sign=False,
        rounding="deterministic",
        weight_bits=5,
        act_integer_bits=16,
        act_fraction_bits=16,
    ):
        super(_ConvNdShift, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.rounding = rounding
        self.shift_range = (-1 * (2 ** (weight_bits - 1) - 2), 0)
        self.act_integer_bits, self.act_fraction_bits = act_integer_bits, act_fraction_bits

        tensor_constructor = torch.DoubleTensor if check_grad else torch.Tensor

        if transposed:
            self.shift = nn.Parameter(
                tensor_constructor(in_channels, out_channels // groups, *kernel_size)
            )
            self.sign = nn.Parameter(
                tensor_constructor(in_channels, out_channels // groups, *kernel_size),
                requires_grad=(freeze_sign is False),
            )
        else:
            self.shift = nn.Parameter(
                tensor_constructor(out_channels, in_channels // groups, *kernel_size)
            )
            self.sign = nn.Parameter(
                tensor_constructor(out_channels, in_channels // groups, *kernel_size),
                requires_grad=(freeze_sign is False),
            )
        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        self.shift.data.uniform_(-10, -1)
        self.sign.data.uniform_(-1, 1)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.shift)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


class Conv2dShift(_ConvNdShift):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        check_grad=False,
        freeze_sign=False,
        rounding="deterministic",
        weight_bits=5,
        act_integer_bits=16,
        act_fraction_bits=16,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dShift, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            check_grad,
            freeze_sign,
            rounding,
            weight_bits,
            act_integer_bits,
            act_fraction_bits,
        )

    def forward(self, input):
        self.shift.data = ste_clamp(self.shift.data, *self.shift_range)
        shift_rounded = ste_round(self.shift, self.rounding)
        sign_rounded_signed = ste_sign(ste_round(self.sign, self.rounding))
        weight_ps = unsym_grad_mul(2**shift_rounded, sign_rounded_signed)
        input_fixed_point = round_fixed_point(input, self.act_integer_bits, self.act_fraction_bits)
        if self.bias is not None:
            bias_fixed_point = round_fixed_point(
                self.bias, self.act_integer_bits, self.act_fraction_bits
            )
        else:
            bias_fixed_point = None

        if self.padding_mode == "circular":
            expanded_padding = (
                (self.padding[1] + 1) // 2,
                self.padding[1] // 2,
                (self.padding[0] + 1) // 2,
                self.padding[0] // 2,
            )
            input_padded = F.pad(input_fixed_point, expanded_padding, mode="circular")
            padding = _pair(0)
        else:
            input_padded = input_fixed_point
            padding = self.padding

        return torch.nn.functional.conv2d(
            input_padded,
            weight_ps,
            bias_fixed_point,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )


# --- vendored from DeepShift pytorch/deepshift/convert.py (drops the shift_type=='Q' branch and
# use_kernel/use_cuda plumbing -- this staging module only exercises the PS variant) ---


def convert_to_shift(
    model,
    shift_depth,
    convert_all_linear=True,
    convert_weights=False,
    freeze_sign=False,
    weight_bits=5,
):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name], num_converted = convert_to_shift(
                model=module,
                shift_depth=shift_depth - conversion_count,
                convert_all_linear=convert_all_linear,
                convert_weights=convert_weights,
                freeze_sign=freeze_sign,
                weight_bits=weight_bits,
            )
            conversion_count += num_converted
        if type(module) == nn.Linear and (
            convert_all_linear is True or conversion_count < shift_depth
        ):
            linear = module
            shift_linear = LinearShift(
                linear.in_features,
                linear.out_features,
                linear.bias is not None,
                freeze_sign=freeze_sign,
                weight_bits=weight_bits,
            )
            if convert_weights:
                shift_linear.shift.data, shift_linear.sign.data = get_shift_and_sign(linear.weight)
                shift_linear.bias = linear.bias
            model._modules[name] = shift_linear
            if convert_all_linear is False:
                conversion_count += 1

        if type(module) == nn.Conv2d and conversion_count < shift_depth:
            conv2d = module
            shift_conv2d = Conv2dShift(
                conv2d.in_channels,
                conv2d.out_channels,
                conv2d.kernel_size,
                conv2d.stride,
                conv2d.padding,
                conv2d.dilation,
                conv2d.groups,
                conv2d.bias is not None,
                conv2d.padding_mode,
                freeze_sign=freeze_sign,
                weight_bits=weight_bits,
            )
            if convert_weights:
                shift_conv2d.shift.data, shift_conv2d.sign.data = get_shift_and_sign(conv2d.weight)
                shift_conv2d.bias = conv2d.bias
            model._modules[name] = shift_conv2d
            conversion_count += 1

    return model, conversion_count


# --- vendored from DeepShift pytorch/cifar10_models/resnet.py (the repo's own CIFAR10 ResNet,
# the base network its scripts convert into DeepShift form) ---


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if option == "A":
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option="A"):
        super(ResNet, self).__init__()
        self.option = option
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, option="A")


def build_deepshift_ps_resnet20():
    """DeepShift-PS ResNet20: the repo's CIFAR10 resnet20() with every Conv2d/Linear converted to
    Conv2dShift/LinearShift via the repo's own convert_to_shift(shift_type='PS')."""
    model = resnet20(num_classes=10)
    model, _ = convert_to_shift(
        model, shift_depth=1000, convert_all_linear=True, convert_weights=True
    )
    return model


def example_input_deepshift_ps_resnet20():
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "DeepShift-PS (ResNet20, multiplication-free shift-and-sign conv/linear)",
        build_deepshift_ps_resnet20,
        example_input_deepshift_ps_resnet20,
        2021,
        "CODE",
    ),
]
