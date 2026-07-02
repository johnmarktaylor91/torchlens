# FAITHFUL PORT of AI4EPS/DeepDenoiser @ master (original framework: TensorFlow 1.x,
# deepdenoiser/model.py `UNet.add_prediction_op`)
# DeepDenoiser (Zhu & Beroza, 2019) denoises seismic waveforms via a depth-6 2D U-Net
# operating on a spectrogram-like [time, freq, channel] representation, trained with a
# soft-mask cross-entropy/softmax objective. The official repo is TF1
# (`tf.compat.v1.disable_eager_execution()`, `tf.compat.v1.layers.*`) with no PyTorch
# release, so this ports the architecture layer-for-layer into torch: same depth-6
# down/up conv stack, same stride-2 strided-conv downsampling + transpose-conv
# upsampling (no pooling layers -- the original uses strided conv2d for downsampling),
# same batchnorm+relu+dropout order per block, same crop-and-concat skip connections,
# same 1x1 output conv + softmax head. Only the input/weight-decay/training
# scaffolding (placeholders, TF summaries, optimizer) is dropped since we only need
# the forward (`add_prediction_op`) path.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class ConvBNReLU(nn.Module):
    """conv2d(bias=False) -> batchnorm -> relu, mirroring the repeated
    tf.compat.v1.layers.conv2d/batch_normalization/relu blocks in UNet.add_prediction_op."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=1, dilation=1):
        super().__init__()
        pad = tuple(((k - 1) * d) // 2 for k, d in zip(kernel_size, (dilation, dilation)))
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class UpConvBNReLU(nn.Module):
    """conv2d_transpose -> batchnorm -> relu, mirroring the up_conv0 block."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(2, 2)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=1, output_padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


def crop_and_concat(net1, net2):
    """Center-crop net2 (the larger, upsampled feature map) to net1's spatial size and
    concat on the channel dim -- port of the TF `crop_and_concat` helper."""
    h1, w1 = net1.shape[-2], net1.shape[-1]
    h2, w2 = net2.shape[-2], net2.shape[-1]
    top = (h2 - h1) // 2
    left = (w2 - w1) // 2
    net2_resized = net2[..., top : top + h1, left : left + w1]
    return torch.cat([net1, net2_resized], dim=1)


class UNet(nn.Module):
    """DeepDenoiser U-Net (Zhu & Beroza 2019): depth-6 2D U-Net over a
    [channel, time, freq]-style spectrogram input, strided-conv downsampling,
    transpose-conv upsampling with skip concatenation, 1x1 conv + softmax mask head."""

    def __init__(
        self, depths=6, filters_root=8, n_channel=2, n_class=2, kernel_size=(3, 3), pool_size=(2, 2)
    ):
        super().__init__()
        self.depths = depths
        self.filters_root = filters_root

        self.input_conv = ConvBNReLU(n_channel, filters_root, kernel_size)

        self.down_conv1 = nn.ModuleList()
        self.down_conv3 = nn.ModuleList()
        in_ch = filters_root
        for depth in range(depths):
            filters = int(2**depth * filters_root)
            self.down_conv1.append(ConvBNReLU(in_ch, filters, kernel_size))
            if depth < depths - 1:
                self.down_conv3.append(ConvBNReLU(filters, filters, kernel_size, stride=pool_size))
            in_ch = filters

        self.up_conv0 = nn.ModuleList()
        self.up_conv1 = nn.ModuleList()
        for depth in range(depths - 2, -1, -1):
            filters = int(2**depth * filters_root)
            self.up_conv0.append(UpConvBNReLU(in_ch, filters, kernel_size, stride=pool_size))
            self.up_conv1.append(ConvBNReLU(2 * filters, filters, kernel_size))
            in_ch = filters

        self.output_conv = nn.Conv2d(in_ch, n_class, kernel_size=(1, 1))

    def forward(self, x):
        net = self.input_conv(x)

        convs = [None] * self.depths
        down_idx = 0
        for depth in range(self.depths):
            net = self.down_conv1[depth](net)
            convs[depth] = net
            if depth < self.depths - 1:
                net = self.down_conv3[down_idx](net)
                down_idx += 1

        up_idx = 0
        for depth in range(self.depths - 2, -1, -1):
            net = self.up_conv0[up_idx](net)
            net = crop_and_concat(convs[depth], net)
            net = self.up_conv1[up_idx](net)
            up_idx += 1

        logits = self.output_conv(net)
        return F.softmax(logits, dim=1)


def build_deepdenoiser():
    # Menagerie-scale config: depths=3 (down from released 6), filters_root=4 (down
    # from released 8), n_channel=2 (real+imag STFT channels), n_class=2 (signal/noise
    # soft mask), matching ModelConfig defaults in deepdenoiser/model.py.
    return UNet(depths=3, filters_root=4, n_channel=2, n_class=2)


def example_input_deepdenoiser():
    torch.manual_seed(0)
    # Spatial dims chosen divisible by 2**(depths-1) so strided-conv/transpose-conv
    # round-trip to matching shapes (real usage: X_shape=[31, 201, 2] spectrogram).
    return (torch.randn(1, 2, 16, 16),)


MENAGERIE_ENTRIES = [
    (
        "DeepDenoiser",
        "build_deepdenoiser",
        "example_input_deepdenoiser",
        2019,
        "ported-pytorch",
    ),
]
