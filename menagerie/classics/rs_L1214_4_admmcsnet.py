# SOURCE: vendored from yangyan92/Pytorch_ADMM-CSNet @ main (network/CSNet_Layers.py,
# torchpwl/pwl.py)
#
# ADMM-CSNet ("ADMM-CSNet: A Deep Learning Approach for Image Compressive Sensing", Yang,
# Sun, Li, Xie, TPAMI 2019). `network/CSNet_Layers.py::ADMMCSNetLayer` unrolls 10 ADMM
# iterations of the compressive-sensing reconstruction update, each iteration composed of a
# closed-form reconstruction step in the Fourier domain (`ReconstructionUpdateLayer`, using
# a learned scalar `rho`), a learned nonlinear transform sandwiched between two learned
# convolutions (`ConvolutionLayer1` -> `torchpwl.PWL` piecewise-linear nonlinearity ->
# `ConvolutionLayer2`, operating on real/imag parts separately then recombined as a complex
# tensor), a subtraction/multiplier update (`MinusLayer`/`Multiple*Layer`, using a learned
# scalar `gamma`), all vendored verbatim below. The `torchpwl.PWL` module used as the
# nonlinearity is itself vendored from the same repo's `torchpwl/pwl.py` (a copy of the
# published `torchpwl` package, pure-torch, no extra deps).
#
# Two minimal device-portability fixes are applied to the vendored code (NOT architectural
# changes): the original hardcodes `.cuda()` in `ReconstructionOriginalLayer`,
# `ReconstructionUpdateLayer`, `ReconstructionFinalLayer` (network/CSNet_Layers.py) and in
# `BaseSlopedPWL.forward` (torchpwl/pwl.py line 236, `sorted_x_positions.cuda()`) -- these
# four call sites are rewritten to follow the input tensor's own device instead of assuming
# CUDA is present, so the real math runs unmodified on CPU too.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Vendored from torchpwl/pwl.py (yangyan92/Pytorch_ADMM-CSNet @ main)
# ---------------------------------------------------------------------------
class BasePWL(torch.nn.Module):
    def __init__(self, num_breakpoints):
        super(BasePWL, self).__init__()
        if not num_breakpoints >= 1:
            raise ValueError(
                "Piecewise linear function only makes sense when you have 1 or more breakpoints."
            )
        self.num_breakpoints = num_breakpoints

    def slope_at(self, x):
        dx = 1e-3
        return -(self.forward(x) - self.forward(x + dx)) / dx


class BasePWLX(BasePWL):
    def __init__(self, num_channels, num_breakpoints, num_x_points):
        super(BasePWLX, self).__init__(num_breakpoints)
        self.num_channels = num_channels
        self.num_x_points = num_x_points
        self.x_positions = torch.Tensor(self.num_channels, self.num_x_points)
        self._reset_x_points()

    def _reset_x_points(self):
        self.x_positions = (
            torch.linspace(-1, 1, self.num_x_points)
            .unsqueeze(0)
            .expand(self.num_channels, self.num_x_points)
        )

    def get_x_positions(self):
        return self.x_positions

    def get_sorted_x_positions(self):
        return torch.sort(self.get_x_positions(), dim=1)[0]

    def get_spreads(self):
        sorted_x_positions = self.get_sorted_x_positions()
        return (torch.roll(sorted_x_positions, shifts=-1, dims=1) - sorted_x_positions)[:, :-1]

    def unpack_input(self, x):
        shape = list(x.shape)
        if len(shape) == 2:
            return x
        elif len(shape) < 2:
            raise ValueError(
                "Invalid input, the input to the PWL module must have at least 2 dimensions with channels at dimension dim(1)."
            )
        assert shape[1] == self.num_channels, (
            "Invalid input, the size of dim(1) must be equal to num_channels (%d)"
            % self.num_channels
        )
        x = torch.transpose(x, 1, len(shape) - 1)
        assert x.shape[-1] == self.num_channels
        return x.reshape(-1, self.num_channels)

    def repack_input(self, unpacked, old_shape):
        old_shape = list(old_shape)
        if len(old_shape) == 2:
            return unpacked
        transposed_shape = old_shape[:]
        transposed_shape[1] = old_shape[-1]
        transposed_shape[-1] = old_shape[1]
        unpacked = unpacked.view(*transposed_shape)
        return torch.transpose(unpacked, 1, len(old_shape) - 1)


class BaseSlopedPWL(BasePWLX):
    def get_biases(self):
        raise NotImplementedError()

    def get_slopes(self):
        raise NotImplementedError()

    def forward(self, x):
        old_shape = x.shape
        x = self.unpack_input(x)
        bs = x.shape[0]
        # DEVICE FIX (was `.cuda()` in the original repo): follow the input tensor's device
        # instead of hardcoding CUDA, so this runs on CPU too. No math change.
        sorted_x_positions = self.get_sorted_x_positions().to(x.device)
        skips = torch.roll(sorted_x_positions, shifts=-1, dims=1) - sorted_x_positions
        slopes = self.get_slopes()
        skip_deltas = skips * slopes[:, 1:]
        biases = self.get_biases().unsqueeze(1)
        cumsums = torch.cumsum(skip_deltas, dim=1)[:, :-1]

        betas = torch.cat([biases, biases, cumsums + biases], dim=1)
        breakpoints = torch.cat([sorted_x_positions[:, 0].unsqueeze(1), sorted_x_positions], dim=1)

        s = x.unsqueeze(2) - sorted_x_positions.unsqueeze(0)
        s = torch.where(s < 0, torch.tensor(float("inf"), device=x.device), s)
        b_ids = torch.where(
            sorted_x_positions[:, 0].unsqueeze(0) <= x,
            torch.argmin(s, dim=2) + 1,
            torch.tensor(0, device=x.device),
        ).unsqueeze(2)

        selected_betas = torch.gather(
            betas.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids
        ).squeeze(2)
        selected_breakpoints = torch.gather(
            breakpoints.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids
        ).squeeze(2)
        selected_slopes = torch.gather(
            slopes.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids
        ).squeeze(2)
        cand = selected_betas + (x - selected_breakpoints) * selected_slopes
        return self.repack_input(cand, old_shape)


class PWL(BaseSlopedPWL):
    r"""Piecewise Linear Function (PWL) module (vendored from torchpwl)."""

    def __init__(self, num_channels, num_breakpoints):
        super(PWL, self).__init__(num_channels, num_breakpoints, num_x_points=num_breakpoints)
        self.slopes = torch.nn.Parameter(torch.Tensor(self.num_channels, self.num_breakpoints + 1))
        self.biases = torch.nn.Parameter(torch.Tensor(self.num_channels))
        self._reset_params()

    def _reset_params(self):
        BasePWLX._reset_x_points(self)
        torch.nn.init.ones_(self.slopes)
        self.slopes.data[:, : (self.num_breakpoints + 1) // 2] = 0.0
        with torch.no_grad():
            self.biases.copy_(torch.zeros_like(self.biases))

    def get_biases(self):
        return self.biases

    def get_x_positions(self):
        return self.x_positions

    def get_slopes(self):
        return self.slopes


# ---------------------------------------------------------------------------
# Vendored from network/CSNet_Layers.py (yangyan92/Pytorch_ADMM-CSNet @ main)
# ---------------------------------------------------------------------------
class ADMMCSNetLayer(nn.Module):
    def __init__(self, mask, in_channels: int = 1, out_channels: int = 128, kernel_size: int = 5):
        super(ADMMCSNetLayer, self).__init__()

        self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.mask = mask
        self.re_org_layer = ReconstructionOriginalLayer(self.rho, self.mask)
        self.conv1_layer = ConvolutionLayer1(in_channels, out_channels, kernel_size)
        self.nonlinear_layer = NonlinearLayer()
        self.conv2_layer = ConvolutionLayer2(out_channels, in_channels, kernel_size)
        self.min_layer = MinusLayer()
        self.multiple_org_layer = MultipleOriginalLayer(self.gamma)
        self.re_update_layer = ReconstructionUpdateLayer(self.rho, self.mask)
        self.add_layer = AdditionalLayer()
        self.multiple_update_layer = MultipleUpdateLayer(self.gamma)
        self.re_final_layer = ReconstructionFinalLayer(self.rho, self.mask)
        layers = []

        layers.append(self.re_org_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_org_layer)

        for i in range(8):
            layers.append(self.re_update_layer)
            layers.append(self.add_layer)
            layers.append(self.conv1_layer)
            layers.append(self.nonlinear_layer)
            layers.append(self.conv2_layer)
            layers.append(self.min_layer)
            layers.append(self.multiple_update_layer)

        layers.append(self.re_update_layer)
        layers.append(self.add_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_update_layer)

        layers.append(self.re_final_layer)

        self.cs_net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1_layer.conv.weight = torch.nn.init.normal_(
            self.conv1_layer.conv.weight, mean=0, std=1
        )
        self.conv2_layer.conv.weight = torch.nn.init.normal_(
            self.conv2_layer.conv.weight, mean=0, std=1
        )
        self.conv1_layer.conv.weight.data = self.conv1_layer.conv.weight.data * 0.025
        self.conv2_layer.conv.weight.data = self.conv2_layer.conv.weight.data * 0.025

    def forward(self, x):
        y = torch.mul(x, self.mask)
        x = self.cs_net(y)
        x = torch.fft.ifft2(y + (1 - self.mask) * torch.fft.fft2(x))
        return x


class ReconstructionOriginalLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionOriginalLayer, self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        mask = self.mask
        # DEVICE FIX (was `mask.cuda()` / `value = ....cuda()` in the original repo): follow
        # the input tensor's device. No math change.
        denom = torch.add(mask.to(x.device), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a, device=x.device)
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)

        orig_output2 = torch.mul(x, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        cs_data = dict()
        cs_data["input"] = x
        cs_data["conv1_input"] = orig_output3
        return cs_data


class ReconstructionUpdateLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionUpdateLayer, self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        minus_output = x["minus_output"]
        multiple_output = x["multi_output"]
        input = x["input"]
        mask = self.mask
        number = torch.add(
            input, self.rho * torch.fft.fft2(torch.sub(minus_output, multiple_output))
        )
        # DEVICE FIX (was hardcoded `.cuda()`): follow the input tensor's device.
        denom = torch.add(mask.to(input.device), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a, device=input.device)
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        x["re_mid_output"] = orig_output3
        return x


class ReconstructionFinalLayer(nn.Module):
    def __init__(self, rho, mask):
        super(ReconstructionFinalLayer, self).__init__()
        self.rho = rho
        self.mask = mask

    def forward(self, x):
        minus_output = x["minus_output"]
        multiple_output = x["multi_output"]
        input = x["input"]
        mask = self.mask
        number = torch.add(
            input, self.rho * torch.fft.fft2(torch.sub(minus_output, multiple_output))
        )
        # DEVICE FIX (was hardcoded `.cuda()`): follow the input tensor's device.
        denom = torch.add(mask.to(input.device), self.rho)
        a = 1e-6
        value = torch.full(denom.size(), a, device=input.device)
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        x["re_final_output"] = orig_output3
        return x["re_final_output"]


class MultipleOriginalLayer(nn.Module):
    def __init__(self, gamma):
        super(MultipleOriginalLayer, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        org_output = x["conv1_input"]
        minus_output = x["minus_output"]
        output = torch.mul(self.gamma, torch.sub(org_output, minus_output))
        x["multi_output"] = output
        return x


class MultipleUpdateLayer(nn.Module):
    def __init__(self, gamma):
        super(MultipleUpdateLayer, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        multiple_output = x["multi_output"]
        re_mid_output = x["re_mid_output"]
        minus_output = x["minus_output"]
        output = torch.add(
            multiple_output, torch.mul(self.gamma, torch.sub(re_mid_output, minus_output))
        )
        x["multi_output"] = output
        return x


class ConvolutionLayer1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            stride=1,
            dilation=1,
            bias=True,
        )

    def forward(self, x):
        conv1_input = x["conv1_input"]
        real = self.conv(conv1_input.real)
        imag = self.conv(conv1_input.imag)
        output = torch.complex(real, imag)
        x["conv1_output"] = output
        return x


class ConvolutionLayer2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            stride=1,
            dilation=1,
            bias=True,
        )

    def forward(self, x):
        nonlinear_output = x["nonlinear_output"]
        real = self.conv(nonlinear_output.real)
        imag = self.conv(nonlinear_output.imag)
        output = torch.complex(real, imag)
        x["conv2_output"] = output
        return x


class NonlinearLayer(nn.Module):
    def __init__(self):
        super(NonlinearLayer, self).__init__()
        self.pwl = PWL(num_channels=128, num_breakpoints=101)

    def forward(self, x):
        conv1_output = x["conv1_output"]
        y_real = self.pwl(conv1_output.real)
        y_imag = self.pwl(conv1_output.imag)
        output = torch.complex(y_real, y_imag)
        x["nonlinear_output"] = output
        return x


class MinusLayer(nn.Module):
    def __init__(self):
        super(MinusLayer, self).__init__()

    def forward(self, x):
        minus_input = x["conv1_input"]
        conv2_output = x["conv2_output"]
        output = torch.sub(minus_input, conv2_output)
        x["minus_output"] = output
        return x


class AdditionalLayer(nn.Module):
    def __init__(self):
        super(AdditionalLayer, self).__init__()

    def forward(self, x):
        mid_output = x["re_mid_output"]
        multi_output = x["multi_output"]
        output = torch.add(mid_output, multi_output)
        x["conv1_input"] = output
        return x


def build_admmcsnet():
    # Real repo default: out_channels=128 (see ConvolutionLayer1/2 default and NonlinearLayer's
    # PWL(num_channels=128, ...)). mask is a fixed (non-trainable) undersampling mask buffer
    # the real train.py/test.py load from a .mat file; a small deterministic checkerboard mask
    # here plays the same structural role (a {0,1}-valued k-space sampling mask), same shape
    # contract as `torch.mul(x, self.mask)` in ADMMCSNetLayer.forward.
    h, w = 32, 32
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, ::2, ::2] = 1.0
    model = ADMMCSNetLayer(mask=mask, in_channels=1, out_channels=128, kernel_size=5)
    model.eval()
    return model


def example_input_admmcsnet():
    # Complex-valued k-space input, matching `torch.mul(x, self.mask)` / `torch.fft.ifft2`
    # usage throughout ADMMCSNetLayer.forward and the Reconstruction*Layer classes.
    h, w = 32, 32
    real = torch.randn(2, 1, h, w)
    imag = torch.randn(2, 1, h, w)
    return torch.complex(real, imag)


MENAGERIE_ENTRIES = [
    ("ADMM-CSNet", "build_admmcsnet", "example_input_admmcsnet", 2019, "CODE"),
]
