# SOURCE: vendored from tiangexiang/BiO-Net @ 75fce3c0a38c8df3acbefcc63d1ef38920f5918f
# https://raw.githubusercontent.com/tiangexiang/BiO-Net/master/pytorch_version/model.py
#
# Xiang, Wang, Cai, Wang, Liu, Zheng, Zhang 2020 (MICCAI) "BiO-Net: Learning Recurrent
# Bi-directional Networks with Fewer Parameters for Medical Image Segmentation". A U-Net
# style encoder-decoder where the encoder/decoder conv blocks at each depth are
# WEIGHT-SHARED and re-used across `iterations` recurrent passes: on iteration 0 the
# decoder features feed back as extra input channels to the encoder on the next
# iteration (concatenated with the raw encoder features), and the encoder's skip features
# similarly feed the decoder -- the "bidirectional" recurrence lets encoder and decoder
# refine each other's representations with no extra parameters (only extra compute).
# `BiONet` is copied verbatim from the real `pytorch_version/model.py` (only the import
# line was split across two lines for readability; no architectural changes).

from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    ConvTranspose2d,
    ReLU,
    MaxPool2d,
    Sigmoid,
)
from torch import cat
import torch


class BiONet(Module):
    def __init__(
        self,
        num_classes: int = 1,
        iterations: int = 2,
        multiplier: float = 1.0,
        num_layers: int = 4,
        integrate: bool = False,
    ):
        super(BiONet, self).__init__()
        self.iterations = iterations
        self.multiplier = multiplier
        self.num_layers = num_layers
        self.integrate = integrate
        self.batch_norm_momentum = 0.01
        self.filters_list = [int(32 * (2**i) * self.multiplier) for i in range(self.num_layers + 1)]
        self.pre_transform_conv_block = Sequential(
            Conv2d(3, self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(
                self.filters_list[0],
                self.filters_list[0],
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(
                self.filters_list[0],
                self.filters_list[0],
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
        )
        self.reuse_convs = []
        self.encoders = []
        self.reuse_deconvs = []
        self.decoders = []
        for iteration in range(self.iterations):
            for layer in range(self.num_layers):
                in_channel = self.filters_list[layer] * 2
                mid_channel = self.filters_list[layer]
                out_channel = self.filters_list[layer + 1]
                if iteration == 0:
                    conv1 = Conv2d(
                        in_channel, mid_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
                    )
                    conv2 = Conv2d(
                        mid_channel, mid_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
                    )
                    conv3 = Conv2d(
                        mid_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
                    )
                    self.reuse_convs.append((conv1, conv2, conv3))
                convs = Sequential(
                    self.reuse_convs[layer][0],
                    ReLU(),
                    BatchNorm2d(mid_channel, momentum=self.batch_norm_momentum),
                    self.reuse_convs[layer][1],
                    ReLU(),
                    BatchNorm2d(mid_channel, momentum=self.batch_norm_momentum),
                )
                down = Sequential(
                    self.reuse_convs[layer][2],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum),
                    MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
                )
                self.add_module(
                    "iteration{0}_layer{1}_encoder_convs".format(iteration, layer), convs
                )
                self.add_module("iteration{0}_layer{1}_encoder_down".format(iteration, layer), down)
                self.encoders.append((convs, down))

                in_channel = (
                    self.filters_list[self.num_layers - layer]
                    + self.filters_list[self.num_layers - 1 - layer]
                )
                out_channel = self.filters_list[self.num_layers - 1 - layer]
                if iteration == 0:
                    conv1 = Conv2d(
                        in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
                    )
                    conv2 = Conv2d(
                        out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
                    )
                    conv3 = ConvTranspose2d(
                        out_channel,
                        out_channel,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        stride=(2, 2),
                        output_padding=(1, 1),
                    )
                    self.reuse_deconvs.append((conv1, conv2, conv3))
                convs = Sequential(
                    self.reuse_deconvs[layer][0],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum),
                    self.reuse_deconvs[layer][1],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum),
                )
                up = Sequential(
                    self.reuse_deconvs[layer][2],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum),
                )
                self.add_module(
                    "iteration{0}_layer{1}_decoder_convs".format(iteration, layer), convs
                )
                self.add_module("iteration{0}_layer{1}_decoder_up".format(iteration, layer), up)
                self.decoders.append((convs, up))
        self.middles = Sequential(
            Conv2d(
                self.filters_list[-1],
                self.filters_list[-1],
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            Conv2d(
                self.filters_list[-1],
                self.filters_list[-1],
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            ConvTranspose2d(
                self.filters_list[-1],
                self.filters_list[-1],
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(2, 2),
                output_padding=(1, 1),
            ),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
        )
        self.post_transform_conv_block = Sequential(
            Conv2d(
                self.filters_list[0] * self.iterations,
                self.filters_list[0],
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            )
            if self.integrate
            else Conv2d(
                self.filters_list[0],
                self.filters_list[0],
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(
                self.filters_list[0],
                self.filters_list[0],
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], 1, kernel_size=(1, 1), stride=(1, 1)),
            Sigmoid(),
        )

    def forward(self, x):
        enc = [None for i in range(self.num_layers)]
        dec = [None for i in range(self.num_layers)]
        all_output = [None for i in range(self.iterations)]
        x = self.pre_transform_conv_block(x)
        e_i = 0
        d_i = 0
        for iteration in range(self.iterations):
            for layer in range(self.num_layers):
                if layer == 0:
                    x_in = x
                x_in = self.encoders[e_i][0](
                    cat([x_in, x_in if dec[-1 - layer] is None else dec[-1 - layer]], dim=1)
                )
                enc[layer] = x_in
                x_in = self.encoders[e_i][1](x_in)
                e_i = e_i + 1
            x_in = self.middles(x_in)
            for layer in range(self.num_layers):
                x_in = self.decoders[d_i][0](cat([x_in, enc[-1 - layer]], dim=1))
                dec[layer] = x_in
                x_in = self.decoders[d_i][1](x_in)
                d_i = d_i + 1
            all_output[iteration] = x_in
        if self.integrate:
            x_in = cat(all_output, dim=1)
        x_in = self.post_transform_conv_block(x_in)
        return x_in


def build_bionet():
    return BiONet(num_classes=1, iterations=2, multiplier=0.25, num_layers=2, integrate=False)


def example_input_bionet():
    # num_layers=2 -> 2 downsamples (factor 4); use a spatial size divisible by 4.
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("BiO-Net", "build_bionet", "example_input_bionet", 2020, "vendored"),
]
