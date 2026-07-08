# SOURCE: vendored from wangjuan001/hicplus @ master (hicplus/model.py, class Net)
#
# HiCPlus resolution-enhancement CNN for Hi-C contact-map super-resolution (Zhang et al.,
# bioRxiv 2017 / Bioinformatics 2018). The original zhangyan32/HiCPlus repo (the queue.tsv
# target) implements this in Theano/Lasagne/nolearn (Python 2, unmaintained); its own README
# explicitly deprecates that implementation in favor of the same lab's official PyTorch port,
# wangjuan001/hicplus, whose `hicplus/model.py::Net` is the real, currently-usable
# architecture: a 3-layer 2D CNN (9x9 conv -> ReLU -> 1x1 conv -> ReLU -> 5x5 conv -> ReLU)
# operating on (N, 1, n, n) Hi-C sub-matrix patches. Copied verbatim aside from dropping the
# unused training-script imports (`torch.autograd.Variable`, `torch.utils.data`, `gzip`,
# `torch.optim`) that model.py carries at module scope for the (separate, non-architectural)
# training loop commented out at the bottom of that file.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5


class Net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, conv2d1_filters_numbers, conv2d1_filters_size)
        self.conv2 = nn.Conv2d(
            conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size
        )
        self.conv3 = nn.Conv2d(conv2d2_filters_numbers, 1, conv2d3_filters_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x


def build_hicplus_net():
    return Net(40, 28)


def example_input_hicplus_net():
    # (N, 1, n, n) Hi-C sub-matrix patch; n=40 matches the repo's default training sample size.
    return torch.rand(2, 1, 40, 40)


MENAGERIE_ENTRIES = [
    ("HiCPlus-Net", build_hicplus_net, example_input_hicplus_net, 2017, "SOURCE_AVAILABLE"),
]
