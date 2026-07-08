# SOURCE: vendored from https://github.com/sweetcocoa/DeepComplexUNetPyTorch @ master
# (DCUNet/unet.py, DCUNet/complex_nn.py)
#
# Deep Complex U-Net (Choi et al., ICLR 2019) speech-enhancement network: a
# complex-valued encoder-decoder U-Net operating directly on the complex STFT
# spectrogram, using complex-valued convolution (`ComplexConv2d`/
# `ComplexConvTranspose2d`, each holding two real Conv2d/ConvTranspose2d
# sub-layers combined via the complex-multiplication identity) and complex
# batch norm (`ComplexBatchNorm2d`, two real BatchNorm2d sub-layers applied
# to the real/imaginary channels separately). The classes below --
# `ComplexConv2d`, `ComplexConvTranspose2d`, `ComplexBatchNorm2d`, `Encoder`,
# `Decoder`, and `UNet` -- are copied verbatim from `complex_nn.py` and
# `unet.py` of the real PyTorch port. No architecture was altered; only the
# unused `import DCUNet.complex_nn as complex_nn` package-relative import is
# inlined (this file merges both source files) and `set_size`'s `model_depth
# == 20` branch (an alternate, larger channel schedule the real code also
# supports) is omitted for brevity since the staged entry point below uses
# the real `model_depth=10` schedule verbatim.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class ComplexConv2d(nn.Module):
    """Real DCUNet/complex_nn.py:ComplexConv2d, verbatim."""

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.conv_re = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        self.conv_im = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )

    def forward(self, x):  # shape of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    """Real DCUNet/complex_nn.py:ComplexConvTranspose2d, verbatim."""

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.tconv_re = nn.ConvTranspose2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            **kwargs,
        )
        self.tconv_im = nn.ConvTranspose2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x):  # shape of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    """Real DCUNet/complex_nn.py:ComplexBatchNorm2d, verbatim."""

    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, **kwargs
    ):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(
            num_features=num_features,
            momentum=momentum,
            affine=affine,
            eps=eps,
            track_running_stats=track_running_stats,
            **kwargs,
        )
        self.bn_im = nn.BatchNorm2d(
            num_features=num_features,
            momentum=momentum,
            affine=affine,
            eps=eps,
            track_running_stats=track_running_stats,
            **kwargs,
        )

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


class Encoder(nn.Module):
    """Real DCUNet/unet.py:Encoder, verbatim."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=None,
        complex=False,
        padding_mode="zeros",
    ):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding

        if complex:
            conv = ComplexConv2d
            bn = ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.conv = conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    """Real DCUNet/unet.py:Decoder, verbatim."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), complex=False
    ):
        super().__init__()
        if complex:
            tconv = ComplexConvTranspose2d
            bn = ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d

        self.transconv = tconv(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    """Real DCUNet/unet.py:UNet (Deep Complex U-Net), verbatim, restricted to
    the real `model_depth == 10` channel schedule (the other real schedule,
    `model_depth == 20`, is a larger alternate config the source also
    supports; omitted here to keep the trace fast, per module docstring)."""

    def __init__(
        self,
        input_channels=1,
        complex=False,
        model_complexity=45,
        model_depth=10,
        padding_mode="zeros",
    ):
        super().__init__()

        if complex:
            model_complexity = int(model_complexity // 1.414)

        self.set_size(
            model_complexity=model_complexity,
            input_channels=input_channels,
            model_depth=model_depth,
        )
        self.encoders = []
        self.model_length = model_depth // 2

        for i in range(self.model_length):
            module = Encoder(
                self.enc_channels[i],
                self.enc_channels[i + 1],
                kernel_size=self.enc_kernel_sizes[i],
                stride=self.enc_strides[i],
                padding=self.enc_paddings[i],
                complex=complex,
                padding_mode=padding_mode,
            )
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            module = Decoder(
                self.dec_channels[i] + self.enc_channels[self.model_length - i],
                self.dec_channels[i + 1],
                kernel_size=self.dec_kernel_sizes[i],
                stride=self.dec_strides[i],
                padding=self.dec_paddings[i],
                complex=complex,
            )
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        if complex:
            conv = ComplexConv2d
        else:
            conv = nn.Conv2d

        linear = conv(self.dec_channels[-1], 1, 1)

        self.add_module("linear", linear)
        self.complex = complex
        self.padding_mode = padding_mode

        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, bd):
        if self.complex:
            x = bd["X"]
        else:
            x = bd["mag_X"]
        # go down
        xs = []
        for _i, encoder in enumerate(self.encoders):
            xs.append(x)
            x = encoder(x)

        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            p = torch.cat([p, xs[self.model_length - 1 - i]], dim=1)

        mask = self.linear(p)
        mask = torch.tanh(mask)
        bd["M_hat"] = mask
        return bd

    def set_size(self, model_complexity, model_depth=10, input_channels=1):
        if model_depth == 10:
            self.enc_channels = [
                input_channels,
                model_complexity,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
            ]
            self.enc_kernel_sizes = [(7, 5), (7, 5), (5, 3), (5, 3), (5, 3)]
            self.enc_strides = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 1)]
            self.enc_paddings = [(2, 1), None, None, None, None]

            self.dec_channels = [
                0,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
            ]

            self.dec_kernel_sizes = [(4, 3), (4, 4), (6, 4), (6, 4), (7, 5)]
            self.dec_strides = [(2, 1), (2, 2), (2, 2), (2, 2), (2, 2)]
            self.dec_paddings = [(1, 1), (1, 1), (2, 1), (2, 1), (2, 1)]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------
_MODEL_COMPLEXITY = 8  # real default is 45; shrunk for a tiny CPU trace
_BATCH = 1
_FREQ = 65  # matches the real model_depth=10 conv/deconv stride-2 schedule's
_TIME = 33  # size-divisibility requirements (verified against real forward)


def build_deep_complex_unet():
    torch.manual_seed(0)
    model = UNet(input_channels=1, complex=True, model_complexity=_MODEL_COMPLEXITY, model_depth=10)
    model.eval()
    return model


def example_input_deep_complex_unet():
    torch.manual_seed(0)
    # Complex STFT spectrogram packed as a trailing real/imaginary axis of
    # size 2, matching the real UNet.forward's `bd['X']` input convention.
    x = torch.randn(_BATCH, 1, _FREQ, _TIME, 2)
    return {"X": x}


MENAGERIE_ENTRIES = [
    (
        "DeepComplexUNet",
        "build_deep_complex_unet",
        "example_input_deep_complex_unet",
        2019,
        MENAGERIE_ZOO,
    ),
]
