# FAITHFUL PORT of alexxu1988/ADJSCC @ main (original framework: TensorFlow/Keras +
# tensorflow_compression)
#
# ADJSCC ("Wireless Image Transmission Using Deep Source Channel Coding With Attention
# Modules", Xu, Bao, Ma, Ni, Chen, Zhang, Zhang, IEEE TCSVT 2022). The official repo is
# TensorFlow/Keras (`util_module.py::Attention_Encoder/Attention_Decoder`,
# `util_channel.py::Channel`), not PyTorch, and depends on Google's
# `tensorflow_compression` package (GDN activation + SignalConv2D) which is TF-only and not
# installable alongside our base torch stack. This module transcribes the real architecture
# faithfully into torch:
#   - `GDN`/`IGDN`: ports `tensorflow_compression.python.layers.gdn.GDN` exactly --
#     norm_pool[i] = beta[i] + sum_j gamma[j,i] * x[j]^2 (a 1x1-conv channel mixing of the
#     squared input using a non-negative-parameterized gamma matrix), then y = x / sqrt(norm_pool)
#     (GDN, inverse=False) or y = x * sqrt(norm_pool) (IGDN, inverse=True). This is the exact
#     math in gdn.py::GDN.call, just expressed as an `nn.Conv2d` 1x1 kernel instead of a
#     tf.nn.convolution "VALID" matmul.
#   - `SignalConv2D` with corr=True (encoder, "same_zeros" padding, arbitrary stride-down) or
#     corr=False (decoder, transposed conv, stride-up) maps directly onto torch's
#     `nn.Conv2d`/`nn.ConvTranspose2d` with `padding="same"` semantics; TF's default conv
#     initializer differs from torch's default, but the math is the same.
#   - `AF_Module` (channel/spatial-agnostic attention-feature module): global-average-pool
#     the feature map, concat the scalar SNR (in dB) as an extra channel, two dense layers
#     (relu -> sigmoid) producing a per-channel gate, multiply into the feature map. Ported
#     from `util_module.py::AF_Module` verbatim (Dense -> nn.Linear, Multiply -> elementwise
#     mul).
#   - `Attention_Encoder`/`Attention_Decoder`: the exact 5-stage GFR_Encoder_Module /
#     GFR_Decoder_Module stack from `util_module.py`, with an `AF_Module` inserted after
#     every encoder/decoder stage (matching the repo's SNR-adaptive "AD" variant used in
#     `adjscc_cifar10.py::main`).
#   - `Channel` (awgn): ports `util_channel.py::Channel.call`/`awgn` -- flatten features,
#     split into real/imag halves to form a complex baseband symbol stream, normalize to
#     unit average symbol power, add complex AWGN at the given SNR (dB), unflatten. Uses the
#     'awgn' branch only (the branch actually exercised by `main()` in adjscc_cifar10.py with
#     default `--channel_type awgn`); `slow_fading`/`slow_fading_eq`/`burst` branches are
#     dropped as this module targets the shipped default configuration.
#
# MENAGERIE_ZOO = "ported-pytorch"

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# GDN / IGDN, ported from tensorflow_compression.python.layers.gdn.GDN
# ---------------------------------------------------------------------------
class GDN(nn.Module):
    def __init__(self, num_channels, inverse=False, gamma_init=0.1, beta_min=1e-6):
        super().__init__()
        self.inverse = inverse
        self.num_channels = num_channels
        # NonnegativeParameterizer: store raw params, reparameterize to be non-negative via
        # a squared reparameterization at call time, matching tensorflow_compression's
        # default lower-bounded-at-`beta_min` reparameterization in spirit (the exact
        # smoothed reparameterization function is an implementation detail of TF's Parameterizer;
        # a square + offset reproduces the same "always non-negative" invariant the real
        # GDN math relies on).
        self.beta_raw = nn.Parameter(torch.sqrt(torch.ones(num_channels) - beta_min))
        self.gamma_raw = nn.Parameter(torch.sqrt(gamma_init * torch.eye(num_channels)))
        self.beta_min = beta_min

    def forward(self, x):
        beta = self.beta_raw**2 + self.beta_min
        gamma = self.gamma_raw**2
        # norm_pool[i] = beta[i] + sum_j gamma[j, i] * x[j]^2, applied as a 1x1 conv over
        # channels (matches gdn.py's `tf.nn.convolution(x**2, gamma_reshaped, "VALID")`).
        weight = gamma.t().reshape(self.num_channels, self.num_channels, 1, 1)
        norm_pool = F.conv2d(x**2, weight) + beta.view(1, -1, 1, 1)
        if self.inverse:
            norm_pool = torch.sqrt(norm_pool)
        else:
            norm_pool = torch.rsqrt(norm_pool)
        return x * norm_pool


# ---------------------------------------------------------------------------
# SignalConv2D "same_zeros" stand-ins, ported as plain torch conv/deconv (see module header).
# ---------------------------------------------------------------------------
def _same_padding(kernel_size):
    return kernel_size // 2


class GFREncoderModule(nn.Module):
    """Ported from util_module.py::GFR_Encoder_Module (corr=True SignalConv2D + GDN + PReLU)."""

    def __init__(self, in_channels, num_filter, kernel_size, stride, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, num_filter, kernel_size, stride=stride, padding=_same_padding(kernel_size)
        )
        self.gdn = GDN(num_filter, inverse=False)
        self.activation = activation
        if activation == "prelu":
            self.prelu = nn.PReLU(num_parameters=num_filter)

    def forward(self, x):
        x = self.gdn(self.conv(x))
        if self.activation == "prelu":
            x = self.prelu(x)
        return x


class GFRDecoderModule(nn.Module):
    """Ported from util_module.py::GFR_Decoder_Module (corr=False SignalConv2D + IGDN + PReLU/sigmoid)."""

    def __init__(self, in_channels, num_filter, kernel_size, stride, activation=None):
        super().__init__()
        pad = _same_padding(kernel_size)
        # output_padding=stride-1 reproduces TF's "same" upsampling output size for stride>1.
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            num_filter,
            kernel_size,
            stride=stride,
            padding=pad,
            output_padding=max(stride - 1, 0),
        )
        self.igdn = GDN(num_filter, inverse=True)
        self.activation = activation
        if activation == "prelu":
            self.prelu = nn.PReLU(num_parameters=num_filter)

    def forward(self, x):
        x = self.igdn(self.deconv(x))
        if self.activation == "prelu":
            x = self.prelu(x)
        elif self.activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


class AFModule(nn.Module):
    """Ported from util_module.py::AF_Module (SNR-conditioned channel attention gate)."""

    def __init__(self, ch_num):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(ch_num + 1, max(ch_num // 16, 1))
        self.dense2 = nn.Linear(max(ch_num // 16, 1), ch_num)

    def forward(self, x, snr):
        m = self.pool(x).flatten(1)  # (batch, ch_num)
        m = torch.cat([m, snr], dim=1)  # concat scalar SNR(dB) channel
        m = F.relu(self.dense1(m))
        m = torch.sigmoid(self.dense2(m))
        return x * m.unsqueeze(-1).unsqueeze(-1)


class AttentionEncoder(nn.Module):
    """Ported from util_module.py::Attention_Encoder."""

    def __init__(self, in_channels, tcn):
        super().__init__()
        self.en1 = GFREncoderModule(in_channels, 256, 9, 2, "prelu")
        self.af1 = AFModule(256)
        self.en2 = GFREncoderModule(256, 256, 5, 2, "prelu")
        self.af2 = AFModule(256)
        self.en3 = GFREncoderModule(256, 256, 5, 1, "prelu")
        self.af3 = AFModule(256)
        self.en4 = GFREncoderModule(256, 256, 5, 1, "prelu")
        self.af4 = AFModule(256)
        self.en5 = GFREncoderModule(256, tcn, 5, 1, None)

    def forward(self, x, snr):
        x = self.af1(self.en1(x), snr)
        x = self.af2(self.en2(x), snr)
        x = self.af3(self.en3(x), snr)
        x = self.af4(self.en4(x), snr)
        x = self.en5(x)
        return x


class AttentionDecoder(nn.Module):
    """Ported from util_module.py::Attention_Decoder."""

    def __init__(self, tcn, out_channels=3):
        super().__init__()
        self.de1 = GFRDecoderModule(tcn, 256, 5, 1, "prelu")
        self.af1 = AFModule(256)
        self.de2 = GFRDecoderModule(256, 256, 5, 1, "prelu")
        self.af2 = AFModule(256)
        self.de3 = GFRDecoderModule(256, 256, 5, 1, "prelu")
        self.af3 = AFModule(256)
        self.de4 = GFRDecoderModule(256, 256, 5, 2, "prelu")
        self.af4 = AFModule(256)
        self.de5 = GFRDecoderModule(256, out_channels, 9, 2, "sigmoid")

    def forward(self, x, snr):
        x = self.af1(self.de1(x), snr)
        x = self.af2(self.de2(x), snr)
        x = self.af3(self.de3(x), snr)
        x = self.af4(self.de4(x), snr)
        x = self.de5(x)
        return x


class AWGNChannel(nn.Module):
    """Ported from util_channel.py::Channel (channel_type='awgn' branch only)."""

    def forward(self, features, snr_db):
        b, c, h, w = features.shape
        f = features.reshape(b, -1)
        dim_z = f.shape[1] // 2
        z_real, z_imag = f[:, :dim_z], f[:, dim_z:]
        # power constraint: average complex symbol power is 1
        norm_factor = (z_real**2 + z_imag**2).sum(dim=1, keepdim=True)
        scale = torch.sqrt(dim_z / norm_factor)
        z_real, z_imag = z_real * scale, z_imag * scale
        noise_stddev = torch.sqrt(10 ** (-snr_db / 10)) / math.sqrt(2)
        noise_real = torch.randn_like(z_real) * noise_stddev
        noise_imag = torch.randn_like(z_imag) * noise_stddev
        z_real, z_imag = z_real + noise_real, z_imag + noise_imag
        out = torch.cat([z_real, z_imag], dim=1)
        return out.reshape(b, c, h, w)


class ADJSCC(nn.Module):
    """Ported from adjscc_cifar10.py::main (channel_type='awgn' branch: the shipped default)."""

    def __init__(self, in_channels=3, tcn=16):
        super().__init__()
        self.encoder = AttentionEncoder(in_channels, tcn)
        self.channel = AWGNChannel()
        self.decoder = AttentionDecoder(tcn, in_channels)

    def forward(self, imgs, snr_db):
        x = imgs / 255.0
        x = self.encoder(x, snr_db)
        x = self.channel(x, snr_db)
        x = self.decoder(x, snr_db)
        return x * 255.0


def build_adjscc():
    model = ADJSCC(in_channels=3, tcn=8)
    model.eval()
    return model


def example_input_adjscc():
    # (imgs, snr_db) matching the real repo's Model(inputs=[input_imgs, input_snrdb], ...)
    # call convention in adjscc_cifar10.py::main (channel_type='awgn').
    imgs = torch.rand(2, 3, 32, 32) * 255.0
    snr_db = torch.full((2, 1), 10.0)
    return (imgs, snr_db)


MENAGERIE_ENTRIES = [
    (
        "ADJSCC (Attention Deep Joint Source-Channel Coding)",
        "build_adjscc",
        "example_input_adjscc",
        2022,
        "PORT",
    ),
]
