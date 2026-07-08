# SOURCE: vendored from etzinis/unsup_speech_enh_adaptation @ main
# https://github.com/etzinis/unsup_speech_enh_adaptation/blob/main/baseline/models/improved_sudormrf.py
# The improved SuDO-RM-RF universal sound-source-separation network (Tzinis
# et al., JSPS 2022 / MLSP 2020) as vendored verbatim by the RemixIT repo
# (etzinis/unsup_speech_enh_adaptation), which trains this exact separator
# with the RemixIT self-supervised remixing procedure (teacher-student
# adaptation on unlabeled mixtures). RemixIT's architectural contribution
# IS this SuDO-RM-RF separator; the self-supervised objective/training loop
# is orthogonal to the traced forward architecture. Transcribed verbatim;
# only change is dropping the `if __name__ == "__main__"` smoke-test block
# (kept as build_/example_input_ helpers below) and using a small config for
# fast tracing.
import math

import torch
import torch.nn as nn


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, chanel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    """
    This class defines a normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: number of output channels
        """
        super().__init__()
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class UConvBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1)
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=2 * stride + 1,
                    stride=stride,
                    groups=in_channels,
                    d=1,
                )
            )
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2)
        self.final_norm = NormAct(in_channels)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # Gather them now in reverse order
        for _ in range(self.depth - 1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        expanded = self.final_norm(output[-1])

        return self.res_conv(expanded) + residual


class SuDORMRF(nn.Module):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=2,
    ):
        super(SuDORMRF, self).__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.n_least_samples_req = self.enc_kernel_size // 2 * 2**self.upsampling_depth

        # Front end
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=enc_num_basis,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=enc_num_basis, out_channels=out_channels, kernel_size=1
        )

        # Separation module
        self.sm = nn.Sequential(
            *[
                UConvBlock(
                    out_channels=out_channels,
                    in_channels=in_channels,
                    upsampling_depth=upsampling_depth,
                )
                for _ in range(num_blocks)
            ]
        )

        mask_conv = nn.Conv1d(out_channels, num_sources * enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=1,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()

    # Forward pass
    def forward(self, input_wav):
        # Front end
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        # Split paths
        s = x.clone()
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        return self.remove_trailing_zeros(estimated_waveforms, input_wav)

    def pad_to_appropriate_length(self, x):
        input_length = x.shape[-1]
        if input_length < self.n_least_samples_req:
            values_to_pad = self.n_least_samples_req
        else:
            res = 1 if input_length % self.n_least_samples_req else 0
            least_number_of_pads = input_length // self.n_least_samples_req
            values_to_pad = (least_number_of_pads + res) * self.n_least_samples_req

        padded_x = torch.zeros(list(x.shape[:-1]) + [values_to_pad], dtype=torch.float32)
        padded_x[..., : x.shape[-1]] = x
        return padded_x.to(x.device)

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., : initial_x.shape[-1]]


MENAGERIE_ZOO = "vendored-pytorch"


def build_remixit_sudormrf():
    torch.manual_seed(0)
    model = SuDORMRF(
        out_channels=32,
        in_channels=64,
        num_blocks=2,
        upsampling_depth=3,
        enc_kernel_size=21,
        enc_num_basis=64,
        num_sources=2,
    )
    model.eval()
    return model


def example_input_remixit_sudormrf():
    torch.manual_seed(0)
    return (torch.rand(1, 1, 4000),)


MENAGERIE_ENTRIES = [
    (
        "RemixIT (self-supervised sep)",
        "build_remixit_sudormrf",
        "example_input_remixit_sudormrf",
        2022,
        MENAGERIE_ZOO,
    ),
]
