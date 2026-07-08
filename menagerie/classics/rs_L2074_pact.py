# SOURCE: vendored from ZouJiu1/Dorefa_Pact @ fbbab6f7c834fbdad6c55194c0d24bf2a8fcce7e
# Files: quantization/dorefa.py (DorefaWeightQuantizer, needed as a pact.py import
# dependency) + quantization/pact.py (PactActivationQuantizer, QuantConv2d, QuantLinear,
# add_quant_op, prepare) -- the actual PACT (Parameterized Clipping Activation) mechanism
# from Choi et al. 2018 (arxiv 1805.06085): a learnable per-layer clipping threshold
# `alpha` on activations, trained via a custom straight-through-estimator autograd.Function
# (`quantize_pact`), applied by walking a base CNN's modules and swapping ReLU layers for
# `PactActivationQuantizer` (clip+round to `a_bits`) and Conv2d/Linear layers for quantized
# variants whose weights are quantized via DoReFa (`DorefaWeightQuantizer`, Zhou et al. 2016).
# Code is copied verbatim from the two source files; only import paths were flattened
# (single-file staging module, no `quantization.` package) and the `PactActivationQuantizer`
# weight_quantizer wiring in QuantConv2d/QuantLinear is the one actually used for a_bits/w_bits
# activation clipping (matches upstream: pact.py's QuantConv2d only imports
# DorefaWeightQuantizer for weights and swaps ReLU->PactActivationQuantizer for activations).
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

MENAGERIE_ZOO = "vendored-pytorch"


# ********************* quantizers (from quantization/dorefa.py) *********************
class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class quantizek(Function):
    @staticmethod
    def forward(ctx, values, q_range):
        ctx.save_for_backward(values)
        ctx.other = q_range
        values = Round.apply(values * q_range) / q_range
        return values

    @staticmethod
    def backward(ctx, grad_weight):
        return grad_weight, None


class DorefaWeightQuantizer(nn.Module):
    def __init__(self, w_bits):
        super(DorefaWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.q_range = 2**w_bits - 1

    def forward(self, weight):
        maxvalue = torch.tanh(weight).abs().max()
        tmp = torch.tanh(weight) / maxvalue * 0.5 + 0.5
        tmp = 2 * quantizek.apply(tmp, self.q_range) - 1
        # for my opinion, need to restore the original weight range
        tmp = maxvalue * tmp
        q_w = torch.arctanh(tmp)
        return q_w


# ********************* quantizers (from quantization/pact.py) *********************
class quantize_pact(Function):
    @staticmethod
    def forward(ctx, values, alpha, q_range):
        ctx.save_for_backward(values, alpha)
        ctx.other = q_range
        tmp = values.clone()
        tmp[tmp > alpha] = alpha
        tmp[tmp < 0] = 0
        tmp = Round.apply(tmp * q_range / alpha) / q_range * alpha
        return tmp

    @staticmethod
    def backward(ctx, grad_weight):
        values, alpha = ctx.saved_tensors
        copyvalue = values.clone()
        double = values.clone()
        double[double < alpha] = 0
        double[double >= alpha] = 1
        grad_alpha = torch.sum(grad_weight * double).unsqueeze(dim=0)

        copyvalue[copyvalue < 0] = -1
        copyvalue[copyvalue >= alpha] = -1
        copyvalue[copyvalue >= 0] = 1  # copyvalue >= 0 & copyvalue < alpha  == copyvalue >= 0
        copyvalue[copyvalue < 0] = 0
        grad_weight = grad_weight * copyvalue
        return grad_weight, grad_alpha, None


class PactActivationQuantizer(nn.Module):
    def __init__(self, a_bits):
        super(PactActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.q_range = 2**self.a_bits - 1
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.init = 0

    def forward(self, activation):
        if self.init == 0:  # initization
            self.alpha.data = activation.abs().max()
            self.init = 1
        q_a = quantize_pact.apply(activation, self.alpha, self.q_range)
        return q_a


class QuantConv2d(nn.Conv2d):
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
        a_bits=8,
        w_bits=8,
        quant_inference=False,
    ):
        super(QuantConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_inference = quant_inference
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits)

    def forward(self, inputs):
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        output = F.conv2d(
            inputs, quant_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return output


class QuantLinear(nn.Linear):
    def __init__(
        self, in_features, out_features, bias=True, a_bits=8, w_bits=8, quant_inference=False
    ):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits)

    def forward(self, inputs):
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(inputs, quant_weight, self.bias)
        return output


def add_quant_op(module, layer_counter, a_bits=8, w_bits=8, quant_inference=False):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1:  # first layer is quantized too
                if child.bias is not None:
                    quant_conv = QuantConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=True,
                        padding_mode=child.padding_mode,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                    )
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=False,
                        padding_mode=child.padding_mode,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                    )
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.ReLU):
            relu = PactActivationQuantizer(a_bits=a_bits)
            module._modules[name] = relu
        elif isinstance(child, nn.Linear):
            layer_counter[0] += 1
            if layer_counter[0] >= 1:  # first layer is quantized too
                if child.bias is not None:
                    quant_linear = QuantLinear(
                        child.in_features,
                        child.out_features,
                        bias=True,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                    )
                    quant_linear.bias.data = child.bias
                else:
                    quant_linear = QuantLinear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                    )
                quant_linear.weight.data = child.weight
                module._modules[name] = quant_linear
        else:
            add_quant_op(
                child, layer_counter, a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference
            )


def prepare(model, inplace=False, a_bits=8, w_bits=8, quant_inference=False):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(
        model, layer_counter, a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference
    )
    return model


# ---- staging harness: tiny base CNN (mirrors the repo's own models.py ResNet pattern of
# conv/bn/relu/linear) run through the real PACT `prepare()` quantization-swap pass ----
class _TinyBaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_pact():
    base = _TinyBaseCNN()
    base.eval()
    model = prepare(base, inplace=False, a_bits=4, w_bits=4, quant_inference=False)
    model.eval()
    return model


def example_input_pact():
    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "PACT (Parameterized Clipping Activation)",
        "build_pact",
        "example_input_pact",
        2018,
        MENAGERIE_ZOO,
    ),
]
