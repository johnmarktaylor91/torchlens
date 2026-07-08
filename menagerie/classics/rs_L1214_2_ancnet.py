# SOURCE: vendored from Luo-Zhengding/SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning @ main
# (Network.py)
#
# SFANC control-filter-selection CNN from "Deep learning based selective active noise
# control system" / "A Hybrid SFANC-FxNLMS Algorithm for Active Noise Control based on Deep
# Learning" (Luo, Shi, Ni, Chen, Xiao 2022/2023). The repo trains a 1-D CNN classifier over
# short noise segments to select among a bank of pre-trained control filters (a "Selective
# Fixed-filter Active Noise Control", SFANC, front end that hands off to FxNLMS). Network.py
# defines a plain CNN and a residual CNN (`CNNRes`, built from `ResBlock`s); the module-level
# `m6_res = CNNRes(...)` instantiation is the actual network shipped/trained in the repo
# (`SFANC-FxNLMS for ANC.ipynb` loads its weights from `Trained models/model.pth`). Classes
# are copied verbatim; only the trailing repo-hardcoded `m6_res = CNNRes(...)` instantiation
# is reproduced inside `build_ancnet()` below (same call, same architecture).
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch

MENAGERIE_ZOO = "vendored-pytorch"


class CNN(torch.nn.Module):
    def __init__(
        self, channels, conv_kernels, conv_strides, conv_padding, pool_padding, num_classes=15
    ):
        assert len(conv_kernels) == len(channels) == len(conv_strides) == len(conv_padding)
        super(CNN, self).__init__()

        # create conv blocks
        self.conv_blocks = torch.nn.ModuleList()
        prev_channel = 1

        for i in range(len(channels)):
            # add stacked conv layer
            block = []
            for j, conv_channel in enumerate(channels[i]):
                block.append(
                    torch.nn.Conv1d(
                        in_channels=prev_channel,
                        out_channels=conv_channel,
                        kernel_size=conv_kernels[i],
                        stride=conv_strides[i],
                        padding=conv_padding[i],
                    )
                )
                prev_channel = conv_channel
                # add batch norm layer
                block.append(torch.nn.BatchNorm1d(prev_channel))
                # adding ReLU
                block.append(torch.nn.ReLU())
            self.conv_blocks.append(torch.nn.Sequential(*block))

        # create pooling blocks
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(len(pool_padding)):
            # adding Max Pool (drops dims by a factor of 4)
            self.pool_blocks.append(
                torch.nn.MaxPool1d(kernel_size=4, stride=4, padding=pool_padding[i])
            )

        # global pooling
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(prev_channel, num_classes)

    def forward(self, inwav):
        for i in range(len(self.conv_blocks)):
            # apply conv layer
            inwav = self.conv_blocks[i](inwav)
            # apply max_pool
            if i < len(self.pool_blocks):
                inwav = self.pool_blocks[i](inwav)
        # apply global pooling
        out = self.global_pool(inwav).squeeze()  # [batch_size, 256, 1] [batch_size, 256]
        out = self.linear(out)  # [batch_size, 15]
        return out


class ResBlock(torch.nn.Module):
    def __init__(self, prev_channel, channel, conv_kernel, conv_stride, conv_pad):
        super(ResBlock, self).__init__()
        self.res = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=prev_channel,
                out_channels=channel,
                kernel_size=conv_kernel,
                stride=conv_stride,
                padding=conv_pad,
            ),
            torch.nn.BatchNorm1d(channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=conv_kernel,
                stride=conv_stride,
                padding=conv_pad,
            ),
            torch.nn.BatchNorm1d(channel),
        )
        self.bn = torch.nn.BatchNorm1d(channel)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.res(x)
        if x.shape[1] == identity.shape[1]:
            x += identity
        # repeat the smaller block till it reaches the size of the bigger block
        elif x.shape[1] > identity.shape[1]:
            if x.shape[1] % identity.shape[1] == 0:
                x += identity.repeat(1, x.shape[1] // identity.shape[1], 1)
            else:
                raise RuntimeError("Dims in ResBlock needs to be divisible on the previous dims!!")
        else:
            if identity.shape[1] % x.shape[1] == 0:
                identity += x.repeat(1, identity.shape[1] // x.shape[1], 1)
            else:
                raise RuntimeError("Dims in ResBlock needs to be divisible on the previous dims!!")
            x = identity
        x = self.bn(x)
        x = self.relu(x)
        return x


class CNNRes(torch.nn.Module):
    def __init__(
        self, channels, conv_kernels, conv_strides, conv_padding, pool_padding, num_classes=15
    ):
        assert len(conv_kernels) == len(channels) == len(conv_strides) == len(conv_padding)
        super(CNNRes, self).__init__()

        # create conv block
        prev_channel = 1
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=prev_channel,
                out_channels=channels[0][0],
                kernel_size=conv_kernels[0],
                stride=conv_strides[0],
                padding=conv_padding[0],
            ),
            # add batch norm layer
            torch.nn.BatchNorm1d(channels[0][0]),
            # adding ReLU
            torch.nn.ReLU(),
            # adding max pool
            torch.nn.MaxPool1d(kernel_size=4, stride=4, padding=pool_padding[0]),
        )

        # create res
        prev_channel = channels[0][0]
        self.res_blocks = torch.nn.ModuleList()
        for i in range(1, len(channels)):
            # add stacked res layer
            block = []
            for j, conv_channel in enumerate(channels[i]):
                block.append(
                    ResBlock(
                        prev_channel,
                        conv_channel,
                        conv_kernels[i],
                        conv_strides[i],
                        conv_padding[i],
                    )
                )
                prev_channel = conv_channel
            self.res_blocks.append(torch.nn.Sequential(*block))

        # create pool blocks
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(1, len(pool_padding)):
            # adding Max Pool (drops dims by a factor of 4)
            self.pool_blocks.append(
                torch.nn.MaxPool1d(kernel_size=4, stride=4, padding=pool_padding[i])
            )

        # global pooling
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(prev_channel, num_classes)

    def forward(self, inwav):
        inwav = self.conv_block(inwav)
        for i in range(len(self.res_blocks)):
            # apply conv layer
            inwav = self.res_blocks[i](inwav)
            # apply max_pool
            if i < len(self.pool_blocks):
                inwav = self.pool_blocks[i](inwav)
        # apply global pooling
        out = self.global_pool(inwav).squeeze()
        out = self.linear(out)
        return out


def build_ancnet():
    # Same constructor call the repo makes for its shipped `m6_res` model (Network.py, final
    # module-level statement), just with a smaller final linear layer implied by num_classes
    # left at the real default (15 control filters) -- no shape hyperparameters were shrunk,
    # this is the exact real-repo call.
    model = CNNRes(
        channels=[[128], [128] * 2],
        conv_kernels=[80, 3],
        conv_strides=[4, 1],
        conv_padding=[38, 1],
        pool_padding=[0, 0],
    )
    model.eval()
    return model


def example_input_ancnet():
    # (batch, 1, samples) raw noise waveform segment, matching CNNRes.forward's Conv1d(in=1,...)
    # front end. 1024 samples is enough to survive the repo's stride-4 conv + two stride-4
    # maxpools without collapsing to zero length.
    return torch.randn(2, 1, 1024)


MENAGERIE_ENTRIES = [
    (
        "SFANC Control-Filter Selector CNN (ANC-Net)",
        "build_ancnet",
        "example_input_ancnet",
        2022,
        "CODE",
    ),
]
