# SOURCE: vendored from pheepa/DCUnet @ master (dcunet.ipynb, code cells 23-29)
# https://raw.githubusercontent.com/pheepa/DCUnet/master/dcunet.ipynb
#
# Deep Complex U-Net (Choi et al., ICLR 2019, "Phase-Aware Speech Enhancement with Deep
# Complex U-Net", https://openreview.net/pdf?id=SkeRTsAcYm) for complex ratio masking:
# a U-Net over the complex STFT of a noisy waveform, built entirely from complex-valued
# convolution/batchnorm layers (real/imaginary channels processed by two real-valued
# convs each, combined via complex multiplication rules), producing a complex ratio
# mask applied to the noisy spectrogram. Classes `CConv2d`, `CConvTranspose2d`,
# `CBatchNorm2d`, `Encoder`, `Decoder`, `DCUnet10` are transcribed verbatim from the
# notebook's code cells; only change is dropping the `is_istft` STFT/ISTFT wrapping
# at the tail of `forward` (kept as an opt-out flag exactly as in the source) and the
# training/data-loading cells, which are not part of the model definition.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.real_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        self.im_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)

        output = torch.stack([c_real, c_im], dim=-1)
        return output


class CConvTranspose2d(nn.Module):
    """
    Class of complex valued dilation convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride

        self.real_convt = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            output_padding=self.output_padding,
            padding=self.padding,
            stride=self.stride,
        )

        self.im_convt = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            output_padding=self.output_padding,
            padding=self.padding,
            stride=self.stride,
        )

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)

        output = torch.stack([ct_real, ct_im], dim=-1)
        return output


class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """

    def __init__(
        self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.real_b = nn.BatchNorm2d(
            num_features=self.num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )
        self.im_b = nn.BatchNorm2d(
            num_features=self.num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)

        output = torch.stack([n_real, n_im], dim=-1)
        return output


class Encoder(nn.Module):
    """
    Class of upsample block
    """

    def __init__(
        self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45, padding=(0, 0)
    ):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.filter_size,
            stride=self.stride_size,
            padding=self.padding,
        )

        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)

        return acted


class Decoder(nn.Module):
    """
    Class of downsample block
    """

    def __init__(
        self,
        filter_size=(7, 5),
        stride_size=(2, 2),
        in_channels=1,
        out_channels=45,
        output_padding=(0, 0),
        padding=(0, 0),
        last_layer=False,
    ):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding

        self.last_layer = last_layer

        self.cconvt = CConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.filter_size,
            stride=self.stride_size,
            output_padding=self.output_padding,
            padding=self.padding,
        )

        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        conved = self.cconvt(x)

        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag

        return output


class DCUnet10(nn.Module):
    """
    Deep Complex U-Net class of the model.
    """

    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()

        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length

        # downsampling/encoding
        self.downsample0 = Encoder(
            filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45
        )
        self.downsample1 = Encoder(
            filter_size=(7, 5), stride_size=(2, 2), in_channels=45, out_channels=90
        )
        self.downsample2 = Encoder(
            filter_size=(5, 3), stride_size=(2, 2), in_channels=90, out_channels=90
        )
        self.downsample3 = Encoder(
            filter_size=(5, 3), stride_size=(2, 2), in_channels=90, out_channels=90
        )
        self.downsample4 = Encoder(
            filter_size=(5, 3), stride_size=(2, 1), in_channels=90, out_channels=90
        )

        # upsampling/decoding
        self.upsample0 = Decoder(
            filter_size=(5, 3), stride_size=(2, 1), in_channels=90, out_channels=90
        )
        self.upsample1 = Decoder(
            filter_size=(5, 3), stride_size=(2, 2), in_channels=180, out_channels=90
        )
        self.upsample2 = Decoder(
            filter_size=(5, 3), stride_size=(2, 2), in_channels=180, out_channels=90
        )
        self.upsample3 = Decoder(
            filter_size=(7, 5), stride_size=(2, 2), in_channels=180, out_channels=45
        )
        self.upsample4 = Decoder(
            filter_size=(7, 5),
            stride_size=(2, 2),
            in_channels=90,
            output_padding=(0, 1),
            out_channels=1,
            last_layer=True,
        )

    def forward(self, x, is_istft=False):
        # downsampling/encoding
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)

        # upsampling/decoding
        u0 = self.upsample0(d4)
        # skip-connection
        c0 = torch.cat((u0, d3), dim=1)

        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)

        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)

        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)

        u4 = self.upsample4(c3)

        # u4 - the complex ratio mask
        output = u4 * x
        if is_istft:
            output = torch.squeeze(output, 1)
            output = torch.istft(
                output, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True
            )

        return output


def build_dcunet10():
    # n_fft/hop_length only matter for the optional istft path (unused here, is_istft
    # defaults False); the architecture-relevant knob is the input spectrogram shape
    # below. Kept at a nominal size; source used a 48kHz-derived n_fft=1024/hop=784.
    model = DCUnet10(n_fft=64, hop_length=16)
    # Deepest encoder stage collapses freq/time down to 1x1 (see example_input_dcunet10
    # comment below), which BatchNorm2d in training mode rejects (needs >1 value per
    # channel per batch element); eval mode uses running stats instead and traces fine.
    model.eval()
    return model


def example_input_dcunet10():
    # Complex STFT tensor in the source's "last-dim-is-real/imag" convention:
    # [batch, channel=1, freq_bins, time_frames, 2]. The encoder is a "valid" (no
    # padding) conv stack; the decoder's transpose convs must land on the exact
    # same spatial size as each encoder skip-connection level (torch.cat requires
    # matching shapes). freq_bins=131, time_frames=70 is the smallest size found by
    # solving the encoder (5,3)/(7,5)-kernel, stride-(2,2)/(2,1) downsample chain
    # against the decoder's mirrored ConvTranspose2d chain (131->63->29->13->5->1 in
    # freq; 70->33->15->7->3->1 in time) so every skip-connection concat lines up.
    torch.manual_seed(0)
    return (torch.randn(1, 1, 131, 70, 2),)


MENAGERIE_ENTRIES = [
    (
        "DCUnet10 (Deep Complex U-Net, complex ratio masking)",
        "build_dcunet10",
        "example_input_dcunet10",
        2019,
        "vendored-pytorch",
    ),
]
