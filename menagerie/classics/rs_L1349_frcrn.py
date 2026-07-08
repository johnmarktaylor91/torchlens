# SOURCE: vendored from modelscope/ClearerVoice-Studio @ main
#   Vendored files: clearvoice/clearvoice/models/frcrn_se/frcrn.py (DCCRN, FRCRN_SE_16K),
#   clearvoice/clearvoice/models/frcrn_se/conv_stft.py (ConvSTFT, ConviSTFT),
#   clearvoice/clearvoice/models/frcrn_se/unet.py (UNet, Encoder, Decoder),
#   clearvoice/clearvoice/models/frcrn_se/se_layer.py (SELayer),
#   clearvoice/clearvoice/models/frcrn_se/complex_nn.py (ComplexConv2d, ComplexConvTranspose2d,
#   ComplexBatchNorm2d, ComplexUniDeepFsmn, ComplexUniDeepFsmn_L1, UniDeepFsmn).
# https://github.com/modelscope/ClearerVoice-Studio
#
# FRCRN (Frequency Recurrent Convolutional Recurrent Network for speech enhancement,
# Zhao/Alibaba Speech Lab, ICASSP 2022 -- 2nd place in the DNS Challenge 2022). Built
# on top of the DCCRN architecture (complex convolutional STFT encoder/decoder), FRCRN
# adds a Feedback Complex Recurrent module using two chained complex-valued UNets with
# complex FSMN (Feedforward Sequential Memory Network) bottlenecks and complex
# Squeeze-and-Excitation attention on each UNet stage, estimating a complex ratio mask
# applied to the (fixed, non-trainable) ConvSTFT complex spectrogram; the mask is
# inverted back to a waveform via ConviSTFT, so the whole pipeline is waveform-in,
# waveform-out and traceable end to end.
#
# Minimal API-compat fixes (NOT architecture changes):
#   - `np.int` (removed from modern numpy) -> Python builtin `int`.
# Everything else (UNet topology, complex FSMN/SE/conv arithmetic, channel counts,
# STFT kernel construction) is untouched from the original source.

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window

# ---------------------------------------------------------------------------
# frcrn_se/conv_stft.py
# ---------------------------------------------------------------------------


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == "None" or win_type is None:
        window = np.ones(win_len)
    else:
        if scipy.__version__ >= "1.10.1" and win_type == "hanning":
            win_type = "hann"
        window = get_window(win_type, win_len, fftbins=True) ** 0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(
        window[None, :, None].astype(np.float32)
    )


class ConvSTFT(nn.Module):
    def __init__(
        self, win_len, win_inc, fft_len=None, win_type="hamming", feature_type="real", fix=True
    ):
        super(ConvSTFT, self).__init__()

        if fft_len is None:
            self.fft_len = int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)

        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == "complex":
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase


class ConviSTFT(nn.Module):
    def __init__(
        self, win_len, win_inc, fft_len=None, win_type="hamming", feature_type="real", fix=True
    ):
        super(ConviSTFT, self).__init__()

        if fft_len is None:
            self.fft_len = int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.win_inc = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer("window", window)
        self.register_buffer("enframe", torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)

        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        return outputs


# ---------------------------------------------------------------------------
# frcrn_se/complex_nn.py
# ---------------------------------------------------------------------------


class UniDeepFsmn(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(
            output_dim, output_dim, [lorder, 1], [1, 1], groups=output_dim, bias=False
        )

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)

        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, 0])

        out = x_per + self.conv1(y)

        out1 = out.permute(0, 3, 2, 1)
        return input + out1.squeeze()


class ComplexUniDeepFsmn(nn.Module):
    def __init__(self, nIn, nHidden=128, nOut=128):
        super(ComplexUniDeepFsmn, self).__init__()

        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_re_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)
        self.fsmn_im_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)

    def forward(self, x):
        b, c, h, T, d = x.size()
        x = torch.reshape(x, (b, c * h, T, d))
        x = torch.transpose(x, 1, 2)

        real_L1 = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary_L1 = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])

        real = self.fsmn_re_L2(real_L1) - self.fsmn_im_L2(imaginary_L1)
        imaginary = self.fsmn_re_L2(imaginary_L1) + self.fsmn_im_L2(real_L1)

        output = torch.stack((real, imaginary), dim=-1)
        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (b, c, h, T, d))

        return output


class ComplexUniDeepFsmn_L1(nn.Module):
    def __init__(self, nIn, nHidden=128, nOut=128):
        super(ComplexUniDeepFsmn_L1, self).__init__()

        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)

    def forward(self, x):
        b, c, h, T, d = x.size()
        x = torch.transpose(x, 1, 3)
        x = torch.reshape(x, (b * T, h, c, d))

        real = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])

        output = torch.stack((real, imaginary), dim=-1)
        output = torch.reshape(output, (b, T, h, c, d))
        output = torch.transpose(output, 1, 3)

        return output


class ComplexConv2d(nn.Module):
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

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)

        return output


class ComplexConvTranspose2d(nn.Module):
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

    def forward(self, x):
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)

        return output


class ComplexBatchNorm2d(nn.Module):
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


# ---------------------------------------------------------------------------
# frcrn_se/se_layer.py
# ---------------------------------------------------------------------------


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_r = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

        self.fc_i = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()

        x_r = self.avg_pool(x[:, :, :, :, 0]).view(b, c)
        x_i = self.avg_pool(x[:, :, :, :, 1]).view(b, c)

        y_r = self.fc_r(x_r).view(b, c, 1, 1, 1) - self.fc_i(x_i).view(b, c, 1, 1, 1)
        y_i = self.fc_r(x_i).view(b, c, 1, 1, 1) + self.fc_i(x_r).view(b, c, 1, 1, 1)

        y = torch.cat([y_r, y_i], 4)

        return x * y


# ---------------------------------------------------------------------------
# frcrn_se/unet.py
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
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
            padding = [(i - 1) // 2 for i in kernel_size]

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
    def __init__(
        self,
        input_channels=1,
        complex=False,
        model_complexity=45,
        model_depth=20,
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
        self.fsmn = ComplexUniDeepFsmn(128, 128, 128)
        self.se_layers_enc = []
        self.fsmn_enc = []

        for i in range(self.model_length):
            fsmn_enc = ComplexUniDeepFsmn_L1(128, 128, 128)
            self.add_module("fsmn_enc{}".format(i), fsmn_enc)
            self.fsmn_enc.append(fsmn_enc)
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
            se_layer_enc = SELayer(self.enc_channels[i + 1], 8)
            self.add_module("se_layer_enc{}".format(i), se_layer_enc)
            self.se_layers_enc.append(se_layer_enc)

        self.decoders = []
        self.fsmn_dec = []
        self.se_layers_dec = []

        for i in range(self.model_length):
            fsmn_dec = ComplexUniDeepFsmn_L1(128, 128, 128)
            self.add_module("fsmn_dec{}".format(i), fsmn_dec)
            self.fsmn_dec.append(fsmn_dec)
            module = Decoder(
                self.dec_channels[i] * 2,
                self.dec_channels[i + 1],
                kernel_size=self.dec_kernel_sizes[i],
                stride=self.dec_strides[i],
                padding=self.dec_paddings[i],
                complex=complex,
            )
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)
            if i < self.model_length - 1:
                se_layer_dec = SELayer(self.dec_channels[i + 1], 8)
                self.add_module("se_layer_dec{}".format(i), se_layer_dec)
                self.se_layers_dec.append(se_layer_dec)

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
        self.se_layers_enc = nn.ModuleList(self.se_layers_enc)
        self.se_layers_dec = nn.ModuleList(self.se_layers_dec)
        self.fsmn_enc = nn.ModuleList(self.fsmn_enc)
        self.fsmn_dec = nn.ModuleList(self.fsmn_dec)

    def forward(self, inputs):
        x = inputs
        xs = []
        xs_se = []
        xs_se.append(x)

        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            if i > 0:
                x = self.fsmn_enc[i](x)
            x = encoder(x)
            xs_se.append(self.se_layers_enc[i](x))

        x = self.fsmn(x)
        p = x

        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i < self.model_length - 1:
                p = self.fsmn_dec[i](p)
            if i == self.model_length - 1:
                break
            if i < self.model_length - 2:
                p = self.se_layers_dec[i](p)
            p = torch.cat([p, xs_se[self.model_length - 1 - i]], dim=1)

        cmp_spec = self.linear(p)
        return cmp_spec

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        if model_depth == 14:
            self.enc_channels = [input_channels, 128, 128, 128, 128, 128, 128, 128]

            self.enc_kernel_sizes = [(5, 2), (5, 2), (5, 2), (5, 2), (5, 2), (5, 2), (2, 2)]

            self.enc_strides = [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)]

            self.enc_paddings = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]

            self.dec_channels = [64, 128, 128, 128, 128, 128, 128, 1]

            self.dec_kernel_sizes = [(2, 2), (5, 2), (5, 2), (5, 2), (6, 2), (5, 2), (5, 2)]

            self.dec_strides = [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)]

            self.dec_paddings = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]

        elif model_depth == 20:
            self.enc_channels = [
                input_channels,
                model_complexity,
                model_complexity,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                128,
            ]

            self.enc_kernel_sizes = [
                (7, 1),
                (1, 7),
                (6, 4),
                (7, 5),
                (5, 3),
                (5, 3),
                (5, 3),
                (5, 3),
                (5, 3),
                (5, 3),
            ]

            self.enc_strides = [
                (1, 1),
                (1, 1),
                (2, 2),
                (2, 1),
                (2, 2),
                (2, 1),
                (2, 2),
                (2, 1),
                (2, 2),
                (2, 1),
            ]

            self.enc_paddings = [(3, 0), (0, 3), None, None, None, None, None, None, None, None]

            self.dec_channels = [
                0,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
                model_complexity * 2,
            ]

            self.dec_kernel_sizes = [
                (4, 3),
                (4, 2),
                (4, 3),
                (4, 2),
                (4, 3),
                (4, 2),
                (6, 3),
                (7, 4),
                (1, 7),
                (7, 1),
            ]

            self.dec_strides = [
                (2, 1),
                (2, 2),
                (2, 1),
                (2, 2),
                (2, 1),
                (2, 2),
                (2, 1),
                (2, 2),
                (1, 1),
                (1, 1),
            ]

            self.dec_paddings = [
                (1, 1),
                (1, 0),
                (1, 1),
                (1, 0),
                (1, 1),
                (1, 0),
                (2, 1),
                (2, 1),
                (0, 3),
                (3, 0),
            ]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))


# ---------------------------------------------------------------------------
# frcrn_se/frcrn.py
# ---------------------------------------------------------------------------


class DCCRN(nn.Module):
    """
    FRCRN is built on top of the DCCRN rep (https://github.com/huyanxin/DeepComplexCRN)
    for complex speech enhancement: a convolutional STFT front end feeding two chained
    complex UNets (FSMN bottleneck + SE attention) that jointly estimate a complex mask.
    """

    def __init__(
        self,
        complex,
        model_complexity,
        model_depth,
        log_amp,
        padding_mode,
        win_len=400,
        win_inc=100,
        fft_len=512,
        win_type="hanning",
    ):
        super().__init__()
        self.feat_dim = fft_len // 2 + 1

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        fix = True
        self.stft = ConvSTFT(
            self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type="complex", fix=fix
        )
        self.istft = ConviSTFT(
            self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type="complex", fix=fix
        )

        self.unet = UNet(
            1,
            complex=complex,
            model_complexity=model_complexity,
            model_depth=model_depth,
            padding_mode=padding_mode,
        )
        self.unet2 = UNet(
            1,
            complex=complex,
            model_complexity=model_complexity,
            model_depth=model_depth,
            padding_mode=padding_mode,
        )

    def forward(self, inputs):
        out_list = []
        cmp_spec = self.stft(inputs)
        cmp_spec = torch.unsqueeze(cmp_spec, 1)

        cmp_spec = torch.cat(
            [
                cmp_spec[:, :, : self.feat_dim, :],
                cmp_spec[:, :, self.feat_dim :, :],
            ],
            1,
        )

        cmp_spec = torch.unsqueeze(cmp_spec, 4)
        cmp_spec = torch.transpose(cmp_spec, 1, 4)

        unet1_out = self.unet(cmp_spec)
        cmp_mask1 = torch.tanh(unet1_out)

        unet2_out = self.unet2(unet1_out)
        cmp_mask2 = torch.tanh(unet2_out)
        cmp_mask2 = cmp_mask2 + cmp_mask1

        est_spec, est_wav, est_mask = self.apply_mask(cmp_spec, cmp_mask2)
        out_list.append(est_spec)
        out_list.append(est_wav)
        out_list.append(est_mask)
        return out_list

    def apply_mask(self, cmp_spec, cmp_mask):
        est_spec = torch.cat(
            [
                cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 0]
                - cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 1],
                cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 1]
                + cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 0],
            ],
            1,
        )

        est_spec = torch.cat([est_spec[:, 0, :, :], est_spec[:, 1, :, :]], 1)
        cmp_mask = torch.squeeze(cmp_mask, 1)
        cmp_mask = torch.cat([cmp_mask[:, :, :, 0], cmp_mask[:, :, :, 1]], 1)

        est_wav = self.istft(est_spec)
        est_wav = torch.squeeze(est_wav, 1)
        return est_spec, est_wav, cmp_mask


class FRCRN_SE_16K(nn.Module):
    """FRCRN configured for 16 kHz speech-enhancement input."""

    def __init__(self, win_len=640, win_inc=320, fft_len=640, win_type="hanning"):
        super(FRCRN_SE_16K, self).__init__()
        self.model = DCCRN(
            complex=True,
            model_complexity=45,
            model_depth=14,
            log_amp=False,
            padding_mode="zeros",
            win_len=win_len,
            win_inc=win_inc,
            fft_len=fft_len,
            win_type=win_type,
        )

    def forward(self, x):
        output = self.model(x)
        return output[1]  # Return estimated waveform


MENAGERIE_ZOO = "vendored-pytorch"


def build_frcrn():
    # The depth=14 UNet.set_size branch (the one FRCRN_SE_16K uses) hardcodes its
    # encoder/decoder kernel/stride/padding schedule assuming the repo-default
    # feat_dim = fft_len // 2 + 1 = 321 (win_len=win_inc=fft_len=640 divided/padded
    # down to exactly reach the fsmn bottleneck's fixed h=128); shrinking win_len/
    # fft_len breaks the encoder's implicit shape assumptions (verified: every
    # feat_dim != 321 raises a shape/padding error inside the real UNet code), so
    # win_len/win_inc/fft_len are kept at the real repo defaults. The only knob left
    # to keep the trace tractable is the input waveform length (kept to ~1 STFT
    # frame beyond the window).
    model = FRCRN_SE_16K(win_len=640, win_inc=320, fft_len=640, win_type="hann")
    model.eval()
    return model


def example_input_frcrn():
    return torch.randn(1, 2000)


MENAGERIE_ENTRIES = [
    ("FRCRN", "build_frcrn", "example_input_frcrn", 2022, MENAGERIE_ZOO),
]
