# FAITHFUL PORT of lzx551402/geodesc @ master (original framework: TensorFlow 1.x)
#
# GeoDesc: a local image-patch descriptor CNN, "GeoDesc: Learning Local
# Descriptors by Integrating Geometry Constraints" (ECCV'18). The official
# inference repo (lzx551402/geodesc) ships only a frozen TF1 GraphDef
# (model/geodesc.pb) with no source model-definition file. The training
# code -- including the actual network architecture class -- lives in the
# companion repo lzx551402/tfmatch (cnn_wrapper/descnet.py's GeoDesc class,
# built on cnn_wrapper/network.py's TF1.x layer-wrapper DSL), so the
# architecture is transcribed faithfully here rather than vendored (no
# PyTorch/TF2 code path exists for either repo).
#
# Ported architecture (from tfmatch/cnn_wrapper/descnet.py, GeoDesc.setup()):
#   .conv_bn(3, 32, 1, name='conv0')
#   .conv_bn(3, 32, 1, name='conv1')
#   .conv_bn(3, 64, 2, name='conv2')
#   .conv_bn(3, 64, 1, name='conv3')
#   .conv_bn(3, 128, 2, name='conv4')
#   .conv_bn(3, 128, 1, name='conv5')
#   .conv(8, 128, 1, biased=False, relu=False, padding='VALID', name='conv6')
#   .l2norm(name='l2norm').squeeze(axis=[1, 2])
#
# Ported layer semantics (from tfmatch/cnn_wrapper/network.py):
#   conv_bn(kernel, filters, stride, ...) = Conv2d(kernel, filters, stride,
#       bias=False, padding='SAME' by default) -> BatchNorm2d(affine=False,
#       i.e. center=False/scale=False in the original -- no learnable
#       gamma/beta, only running mean/var normalization) -> ReLU.
#   conv6 uses explicit padding='VALID' (no padding) with an 8x8 kernel,
#       biased=True by default per the `conv` signature is overridden False
#       here (biased=False as called) and relu=False, producing a single
#       128-d spatial-collapsed descriptor from a 32x32 input patch.
#   l2norm -> L2-normalize along the channel axis.
#   squeeze(axis=[1, 2]) -> drop the now-1x1 spatial dims, leaving a
#       [batch, 128] descriptor vector (this port keeps NCHW layout, so the
#       squeeze is over the H,W dims which are dims 2,3 in NCHW).
#
# Caffe-like 'SAME' padding (caffe_like_padding) reduces to standard
# TensorFlow/PyTorch 'same' zero-padding for the odd, stride-{1,2} 3x3
# kernels used in conv0..conv5, so this port uses PyTorch's
# padding=kernel//2 convention, which is numerically equivalent for these
# kernel/stride combinations.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class ConvBN(nn.Module):
    """Port of Network.conv_bn: unbiased conv -> BN (no affine, i.e.
    center=False/scale=False in the original TF wrapper) -> ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, affine=False, eps=1e-5)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class GeoDesc(nn.Module):
    """Port of tfmatch.cnn_wrapper.descnet.GeoDesc.setup(): a 7-layer
    fully-convolutional local descriptor network producing an L2-normalized
    128-d embedding per input patch."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.conv0 = ConvBN(in_channels, 32, kernel_size=3, stride=1)
        self.conv1 = ConvBN(32, 32, kernel_size=3, stride=1)
        self.conv2 = ConvBN(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvBN(64, 64, kernel_size=3, stride=1)
        self.conv4 = ConvBN(64, 128, kernel_size=3, stride=2)
        self.conv5 = ConvBN(128, 128, kernel_size=3, stride=1)
        # conv6: 8x8 'VALID' (no padding), unbiased, no ReLU -- collapses
        # the remaining spatial extent to 1x1 for a 32x32 input patch.
        self.conv6 = nn.Conv2d(128, 128, kernel_size=8, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = x.squeeze(dim=(2, 3))
        return x


# ---------------------------------------------------------------------------
# Staging build/example helpers. Original repo runs on 32x32 grayscale
# patches (standard local-descriptor patch size, matching HPatches/SIFT
# patch extraction); kept at native 32x32 here since the architecture's
# spatial reductions (2 stride-2 convs + final 8x8 VALID conv) are exactly
# sized for that input, same architecture shape.
# ---------------------------------------------------------------------------
def build_geodesc():
    torch.manual_seed(0)
    model = GeoDesc(in_channels=1)
    model.eval()
    return model


def example_input_geodesc():
    torch.manual_seed(0)
    return torch.randn(2, 1, 32, 32)


MENAGERIE_ENTRIES = [
    ("GeoDesc", "build_geodesc", "example_input_geodesc", 2018, MENAGERIE_ZOO),
]
