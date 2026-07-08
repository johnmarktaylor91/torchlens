# SOURCE: vendored from cornell-zhang/FracBNN @ main
# https://raw.githubusercontent.com/cornell-zhang/FracBNN/main/model/fracbnn_cifar10.py
# https://raw.githubusercontent.com/cornell-zhang/FracBNN/main/utils/quantization.py
#
# Zhang, Yang, Wei, Chen, Zhang, Li, Zhang 2021 (FPGA'21) "FracBNN: Accurate and
# FPGA-Efficient Binary Neural Networks with Fractional Activations". Official Cornell
# repo. This vendors the software (PyTorch-only) portion of FracBNN: the CIFAR-10
# ResNet-20 binary-neural-network variant built from `InputEncoder` (multi-bit
# thermometer/"fractional" input encoding that turns each pixel into `b` bipolar
# {-1,+1} planes instead of a single binarized plane), `BinaryConv2d`/`PGBinaryConv2d`
# (sign-binarized-weight convolutions, the latter a "precision-gated" conv that mixes a
# cheap binary-MSB conv with a full-precision conv per spatial location via a learned
# threshold + differentiable mask), `RPReLU`/`RSign`/`FastSign`/`QuantSign` (ReActNet-
# style learnable-bias-sandwiched activations and straight-through quantizers), and the
# ResNet-20 BasicBlock/ResNet wiring that composes them. The HLS/FPGA accelerator
# directories (`xcel-cifar10/`, `xcel-imagenet/`) are C++/HLS, not portable, and are
# correctly excluded per the queue notes ("FPGA accelerator code included but SW
# PyTorch portion is portable").
#
# No architectural changes were made; only mechanical fixes for import isolation and to
# run without a GPU:
#   - `import utils.quantization as q` (relative sibling-package import assuming the
#     original repo layout) is replaced by defining the `quantization.py` classes
#     directly in this module (still verbatim) since the `utils` package is not
#     installed standalone.
#   - `InputEncoder.__init__` unconditionally called `.cuda()` on its registered
#     buffer when `torch.cuda.is_available()` (a training-convenience shortcut in the
#     original script, not part of the architecture); that conditional is dropped so
#     the buffer is always built on the model's default (CPU) device, matching how
#     every other layer in the network is constructed. The buffer's values and shape
#     are identical either way.
#   - The `__main__` block (`thop.profile` benchmarking script) is dropped; irrelevant
#     to tracing a single instance.

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# utils/quantization.py (verbatim, ReAct-style quantized layers)
# ---------------------------------------------------------------------------


class LearnableBias(nn.Module):
    def __init__(self, in_channels):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1), requires_grad=True)

    def forward(self, input):
        return input + self.bias.expand_as(input)


class RPReLU(nn.Module):
    """RPReLU is a PReLU sandwitched by learnable biases"""

    def __init__(self, in_channels):
        super(RPReLU, self).__init__()
        self.shift_x = LearnableBias(in_channels)
        self.shift_y = LearnableBias(in_channels)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, input):
        input = self.shift_x(input)
        input = self.prelu(input)
        input = self.shift_y(input)
        return input


class FastSign(nn.Module):
    def __init__(self):
        super(FastSign, self).__init__()

    def forward(self, input):
        out_forward = torch.sign(input)
        """
        Only inputs in the range [-t_clip,t_clip]
        have gradient 1.
        """
        t_clip = 1.3
        out_backward = torch.clamp(input, -t_clip, t_clip)
        return out_forward.detach() - out_backward.detach() + out_backward


class QuantSign(torch.autograd.Function):
    """
    Quantize Sign activation to arbitrary bitwidth.
    Usage:
        output = QuantSign.apply(input, bits)
    """

    @staticmethod
    def forward(ctx, input, bits=2):
        ctx.save_for_backward(input)
        input = torch.clamp(input, -1.0, 1.0)
        delta = 2.0 / (2.0**bits - 1.0)
        input = torch.round((input + 1.0) / delta) * delta - 1.0
        return input

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        """
        Only inputs in the range [-t_clip,t_clip]
        have gradient 1.
        """
        t_clip = 1.0
        grad_input = grad_output.clone()
        grad_input *= (input > -t_clip).float()
        grad_input *= (input < t_clip).float()
        return grad_input, None


class SparseGreaterThan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, torch.tensor(threshold))
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        (
            input,
            threshold,
        ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        """ Identity gradients only when input >= threshold """
        grad_input *= (input >= threshold).float()
        return grad_input, None


class BinaryConv2d(nn.Conv2d):
    """
    A convolutional layer with its weight tensor binarized to {-1, +1}.
    """

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
    ):
        super(BinaryConv2d, self).__init__(
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
        self.binarize = FastSign()

    def forward(self, input):
        return F.conv2d(
            input,
            self.binarize(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PGBinaryConv2d(nn.Conv2d):
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
        padding_mode="zeros",
        sparse_bp=True,
        init=-1.0,
    ):
        super(PGBinaryConv2d, self).__init__(
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
        self.binarize = FastSign()
        self.gt = (
            SparseGreaterThan.apply
            if sparse_bp
            else lambda x, t: torch.Tensor.float(torch.gt(x, t))
        )

        """
        zero initialization
        nan loss while using torch.Tensor to initialize the thresholds
        """
        self.threshold = nn.Parameter(torch.ones(1, out_channels, 1, 1) * init)

        """ number of output features """
        self.num_out = torch.zeros(1)
        """ number of output features computed at high precision """
        self.num_high = torch.zeros(1)

    def forward(self, input):
        """MSB convolution"""
        out_msb = (
            F.conv2d(
                self.binarize(input),
                self.binarize(self.weight),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            * 2.0
            / 3.0
        )
        """ Calculate the mask """
        mask = self.gt(torch.sigmoid(5.0 * (out_msb - self.threshold)), 0.5)
        """ update report """
        self.num_out.fill_(mask.numel())
        self.num_high.fill_((mask > 0).sum().item())
        """ full convolution """
        out_full = F.conv2d(
            input,
            self.binarize(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        """ combine outputs """
        return (1 - mask) * out_msb + mask * out_full


class InputEncoder(nn.Module):
    """
    Encode the input images to bipolar strings using thermometer encoding.
    Request:
        Know the input size beforehand.
    """

    def __init__(self, input_size, resolution):
        super(InputEncoder, self).__init__()
        self.n, self.c, self.h, self.w = input_size
        self.resolution = int(resolution)
        self.b = int(round(255.0 / self.resolution))
        # Real repo: conditionally .cuda() the buffer when torch.cuda.is_available().
        # Dropped here so the buffer always matches the model's (CPU) device -- see
        # header note.
        placeholder = torch.ones(self.n, self.c, self.b, self.h, self.w, dtype=torch.float32)
        placeholder *= torch.arange(self.b).view(1, 1, -1, 1, 1)
        self.register_buffer("placeholder", placeholder)

    def forward(self, x):
        x = (x * 255.0).view(-1, self.c, 1, self.h, self.w)
        output = (self.placeholder < torch.round(x / self.resolution)).float()
        output *= 2.0
        output -= 1.0
        return output.view(-1, self.b * self.c, self.h, self.w).detach()


# ---------------------------------------------------------------------------
# model/fracbnn_cifar10.py (verbatim ResNet-20 BNN variant)
# ---------------------------------------------------------------------------


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    """
    Proposed ReActNet model variant.
    For details, please refer to our paper:
        https://arxiv.org/abs/2012.12206
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.rprelu1 = RPReLU(in_channels=planes)
        self.rprelu2 = RPReLU(in_channels=planes)

        self.conv1 = PGBinaryConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PGBinaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                LambdaLayer(lambda x: torch.cat((x, x), dim=1)),
            )

        self.binarize = QuantSign.apply
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn4 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.rprelu1(self.bn1(self.conv1(self.binarize(x)))) + self.shortcut(x)
        x = self.bn3(x)
        x = self.rprelu2(self.bn2(self.conv2(self.binarize(x)))) + x
        x = self.bn4(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, batch_size=128, num_gpus=1):
        super(ResNet, self).__init__()
        self.in_planes = 16

        """ The input layer is binarized! """
        self.conv1 = BinaryConv2d(96, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        """ N = batch_size / num_gpus """
        assert batch_size % num_gpus == 0, (
            "Given batch size cannot evenly distributed to available gpus."
        )
        N = batch_size // num_gpus
        self.encoder = InputEncoder(input_size=(N, 3, 32, 32), resolution=8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10, batch_size=128, num_gpus=1):
    return ResNet(
        BasicBlock, [3, 3, 3], num_classes=num_classes, batch_size=batch_size, num_gpus=num_gpus
    )


def build_fracbnn():
    model = resnet20(num_classes=10, batch_size=1, num_gpus=1)
    model.eval()
    return model


def example_input_fracbnn():
    torch.manual_seed(0)
    return (torch.rand(1, 3, 32, 32),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("FracBNN", "build_fracbnn", "example_input_fracbnn", 2021, "vendored"),
]
