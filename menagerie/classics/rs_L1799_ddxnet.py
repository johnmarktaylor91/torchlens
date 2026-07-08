# SOURCE: vendored from Drajan/DDxNet @ master
# https://raw.githubusercontent.com/Drajan/DDxNet/master/model/ddxnet_model.py
#
# Rajan & Thiagarajan 2019 (IBM) "A Deep Neural Framework for Continuous Sign Language
# Recognition by Iterative Training" -- DDxNet: a DenseNet-style 1-D dilated-causal-CNN
# stack for ECG/EEG/EHR time-series classification. DDx_block/DDxNet copied verbatim from
# the real model/ddxnet_model.py (only the unused `torchvision`/`torchvision.transforms`
# imports and the module-level CUDA `device` selection are dropped; no architectural change).
"""DDxNet: densely-connected dilated-causal 1-D CNN for physiological time series."""

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


def relu_conv(in_channels, n_filters=32, k=1, dilation_factor=1, s=1, p=1):
    return nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Conv1d(
            in_channels, n_filters, kernel_size=k, dilation=dilation_factor, stride=s, padding=p
        ),
    )


class DDx_block(nn.Module):
    def __init__(self, in_channels, dilation_factor=1, causal=False):
        super(DDx_block, self).__init__()
        self.causal = causal
        ker_size = 3
        d = dilation_factor

        if self.causal:
            self.pad = (ker_size - 1) * d
        else:
            self.pad = (ker_size - 1) * (d) // 2

        self.cnn1 = relu_conv(in_channels, 128, k=ker_size, dilation_factor=d, s=1, p=self.pad)
        self.cnn2 = relu_conv(128, 32, k=ker_size, dilation_factor=d, s=1, p=self.pad)

    def forward(self, x):
        h = self.cnn1(x)
        if self.causal:
            h = h[:, :, : -self.pad]
        h = self.cnn2(h)
        if self.causal:
            h = h[:, :, : -self.pad]
        out = torch.cat((h, x), 1)
        return out


class DDxNet(nn.Module):
    def __init__(self, in_channels, seqlen, block, stacks, output_dim, causal, use_dilation):
        super(DDxNet, self).__init__()
        self.n_filters_entry = 64
        self.seqlen = seqlen
        self.samp_factor = 2 ** (len(stacks))
        self.causal = causal
        self.use_dilation = use_dilation

        self.cb_entry = nn.Sequential(
            nn.Conv1d(in_channels, self.n_filters_entry, kernel_size=7, stride=1, padding=3),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )

        self.stack1, self.n_filters1 = self.make_stack(block, self.n_filters_entry, stacks[0])
        self.trans1 = self.transition_block(self.n_filters1)

        self.stack2, self.n_filters2 = self.make_stack(block, self.n_filters1 // 2, stacks[1])
        self.trans2 = self.transition_block(self.n_filters2)

        self.stack3, self.n_filters3 = self.make_stack(block, self.n_filters2 // 2, stacks[2])
        self.trans3 = self.transition_block(self.n_filters3)

        self.stack4, self.n_filters4 = self.make_stack(block, self.n_filters3 // 2, stacks[3])
        self.trans4 = self.transition_block(self.n_filters4)

        self.avg_pool = nn.AvgPool1d(self.seqlen // self.samp_factor)
        self.fc = nn.Linear(1 * self.n_filters4 // 2, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_stack(self, block, in_channels, num_ddx_blocks):
        stack = []
        stack.append(block(in_channels))
        in_channels += 32

        for i in range(1, num_ddx_blocks):
            if self.use_dilation:
                dilation_factor = np.minimum(128, 2 ** (i + 2))
            else:
                dilation_factor = 1
            stack.append(block(in_channels, dilation_factor, self.causal))
            in_channels += 32

        return nn.Sequential(*stack), in_channels

    def transition_block(self, in_channels, k=1, s=1, p=0):
        return nn.Sequential(
            relu_conv(in_channels, in_channels // 2, k=1, s=1, p=0),
            nn.AvgPool1d(kernel_size=2, stride=2, padding=0),
        )

    def forward(self, x):
        h = self.cb_entry(x)
        h = self.stack1(h)
        h = self.trans1(h)
        h = self.stack2(h)
        h = self.trans2(h)
        h = self.stack3(h)
        h = self.trans3(h)
        h = self.stack4(h)
        h = self.trans4(h)
        h = self.avg_pool(h)
        out = h.view(h.size(0), -1)
        out = self.fc(out)
        return out


def build_ddxnet():
    # Reference DDxNet configs use 4 dense stacks of ~4-6 DDx_blocks each over long ECG/EEG
    # windows; scaled down here (2 blocks/stack, seqlen=64) for a fast trace while keeping
    # the real 4-stack + transition + causal-dilated-conv architecture intact.
    return DDxNet(
        in_channels=4,
        seqlen=64,
        block=DDx_block,
        stacks=[2, 2, 2, 2],
        output_dim=5,
        causal=True,
        use_dilation=True,
    )


def example_input_ddxnet():
    torch.manual_seed(0)
    return (torch.randn(2, 4, 64),)


MENAGERIE_ENTRIES = [
    ("DDxNet", "build_ddxnet", "example_input_ddxnet", 2019, "vendored"),
]
