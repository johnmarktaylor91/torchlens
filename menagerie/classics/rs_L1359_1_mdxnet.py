# SOURCE: vendored from kuielab/mdx-net @ main
# https://github.com/kuielab/mdx-net/blob/main/src/models/mdxnet.py
# https://github.com/kuielab/mdx-net/blob/main/src/models/modules.py
#
# The original AbstractMDXNet/ConvTDFNet classes subclass pytorch_lightning.LightningModule
# purely for training-loop plumbing (training_step/validation_step/configure_optimizers/
# save_hyperparameters/load_from_checkpoint). None of that plumbing is architecture, so the
# base class here is swapped for plain torch.nn.Module (mechanical import fix per the vendoring
# rung -- "fix only imports/relative-paths minimally"). The STFT/ISTFT framing and the
# TFC / DenseTFC / TFC_TDF / ConvTDFNet forward-pass code is transcribed verbatim from the
# real repo. torch.stft is called with return_complex=False (explicit) to reproduce the old
# torch default the original code relied on -- current torch requires the arg to be given.

import torch
import torch.nn as nn


class TFC(nn.Module):
    def __init__(self, c, l, k):  # noqa: E741 (matches upstream param name)
        super(TFC, self).__init__()

        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k // 2
                    ),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x


class DenseTFC(nn.Module):
    def __init__(self, c, l, k):  # noqa: E741 (matches upstream param name)
        super(DenseTFC, self).__init__()

        self.conv = nn.ModuleList()
        for i in range(l):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k // 2
                    ),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.conv[:-1]:
            x = torch.cat([layer(x), x], 1)
        return self.conv[-1](x)


class TFC_TDF(nn.Module):
    def __init__(self, c, l, f, k, bn, dense=False, bias=True):  # noqa: E741 (matches upstream param name)
        super(TFC_TDF, self).__init__()

        self.use_tdf = bn is not None

        self.tfc = DenseTFC(c, l, k) if dense else TFC(c, l, k)

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(nn.Linear(f, f, bias=bias), nn.BatchNorm2d(c), nn.ReLU())
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                )

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tdf(x) if self.use_tdf else x


class AbstractMDXNet(nn.Module):
    def __init__(self, target_name, dim_c, dim_f, dim_t, n_fft, hop_length, overlap):
        super().__init__()
        self.target_name = target_name
        self.dim_c = dim_c
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length

        self.chunk_size = hop_length * (self.dim_t - 1)
        self.overlap = overlap
        self.window = nn.Parameter(
            torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False
        )
        self.freq_pad = nn.Parameter(
            torch.zeros([1, dim_c, self.n_bins - self.dim_f, self.dim_t]), requires_grad=False
        )
        self.input_sample_shape = (self.stft(torch.zeros([1, 2, self.chunk_size]))).shape

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=False,
        )
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, spec):
        spec = torch.cat([spec, self.freq_pad.repeat([spec.shape[0], 1, 1, 1])], -2)
        spec = spec.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        spec = spec.permute([0, 2, 3, 1])
        # torch.istft now requires a complex tensor (old default accepted the real [...,2] layout
        # this repo builds); view_as_complex reproduces the old behavior identically.
        spec = torch.view_as_complex(spec.contiguous())
        spec = torch.istft(
            spec, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True
        )
        return spec.reshape([-1, 2, self.chunk_size])


class ConvTDFNet(AbstractMDXNet):
    def __init__(
        self,
        target_name,
        dim_c,
        dim_f,
        dim_t,
        n_fft,
        hop_length,
        num_blocks,
        l,
        g,
        k,
        bn,
        bias,
        overlap,
    ):  # noqa: E741 (matches upstream param name)
        super(ConvTDFNet, self).__init__(
            target_name, dim_c, dim_f, dim_t, n_fft, hop_length, overlap
        )

        self.num_blocks = num_blocks
        self.l = l
        self.g = g
        self.k = k
        self.bn = bn
        self.bias = bias

        self.n = num_blocks // 2
        scale = (2, 2)

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_c, out_channels=g, kernel_size=(1, 1)),
            nn.BatchNorm2d(g),
            nn.ReLU(),
        )

        f = self.dim_f
        c = g
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()
        for i in range(self.n):
            self.encoding_blocks.append(TFC_TDF(c, l, f, k, bn, bias=bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    nn.BatchNorm2d(c + g),
                    nn.ReLU(),
                )
            )
            f = f // 2
            c += g

        self.bottleneck_block = TFC_TDF(c, l, f, k, bn, bias=bias)

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale
                    ),
                    nn.BatchNorm2d(c - g),
                    nn.ReLU(),
                )
            )
            f = f * 2
            c -= g

            self.decoding_blocks.append(TFC_TDF(c, l, f, k, bn, bias=bias))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c, kernel_size=(1, 1)),
        )

    def forward(self, x):
        x = self.first_conv(x)

        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        x = self.bottleneck_block(x)

        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x


class ConvTDFNetEndToEnd(nn.Module):
    """Waveform-in/waveform-out wrapper: stft -> ConvTDFNet -> istft, matching how
    kuielab/mdx-net actually runs the separator on raw audio (see AbstractMDXNet.stft/
    istft and Mixer.training_step in src/models/mdxnet.py)."""

    def __init__(self, net: ConvTDFNet):
        super().__init__()
        self.net = net

    def forward(self, mix_wave):
        mix_spec = self.net.stft(mix_wave)
        out_spec = self.net(mix_spec)
        return self.net.istft(out_spec)


# Tiny random-init preset (architecture-preserving shrink of the real
# configs/model/ConvTDFNet_vocals.yaml preset: num_blocks/l/g/k/bn/bias unchanged;
# n_fft/dim_f/dim_t/hop_length/overlap are input-size params only, shrunk so a
# forward pass is cheap. dim_f=64 is divisible by 2**(num_blocks//2)=4 as required
# by the encoder/decoder downsample/upsample stack.)
def build_mdxnet_convtdf():
    net = ConvTDFNet(
        target_name="vocals",
        dim_c=4,
        dim_f=64,
        dim_t=8,
        n_fft=126,
        hop_length=16,
        num_blocks=4,
        l=2,
        g=4,
        k=3,
        bn=2,
        bias=True,
        overlap=4,
    )
    return ConvTDFNetEndToEnd(net)


def example_input_mdxnet_convtdf():
    net = build_mdxnet_convtdf().net
    return torch.randn(1, 2, net.chunk_size)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "KUIELAB-MDX-Net (ConvTDFNet)",
        "build_mdxnet_convtdf",
        "example_input_mdxnet_convtdf",
        2021,
        "vendored-pytorch",
    ),
]
