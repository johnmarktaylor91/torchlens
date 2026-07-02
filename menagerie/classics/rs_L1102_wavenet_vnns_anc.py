# SOURCE: vendored from https://github.com/Lu-Baihh/WaveNet-VNNs-for-ANC @ main
# (WaveNet_VNNs/networks.py, WaveNet_VNNs/utils.py:slicing)
#
# "Deep ANC" -- the queue's Deep Active Noise Control candidate points to this
# repo (Lu-Baihh/WaveNet-VNNs-for-ANC), the official implementation
# accompanying the WaveNet-VNNs active-noise-control paper (secondary-path
# nonlinearity compensation via a Volterra-style quadratic neural network
# fused with a WaveNet-style dilated causal-convolution stack). The classes
# below -- `Causal_Conv1d`, `VNN2` (2nd-order truncated Volterra neural
# network: parallel linear + Hadamard-product quadratic causal-conv branches,
# summed), `dilated_residual_block` (real WaveNet gated residual block:
# tanh/sigmoid gating + residual/skip split), and `WaveNet_VNNs` (the full
# stack: input causal convs -> dilated residual stack -> summed skip
# connections -> output causal convs -> VNN2 head) -- are copied verbatim
# from `networks.py`, together with the real `slicing` helper from
# `utils.py` that `dilated_residual_block.forward` calls. No architecture
# was altered. `config.json`'s real channel/kernel/stack values are reused
# unmodified from the repo; only `num_stacks`/`dilations` are shrunk (3->1
# stack, 9->3 dilation levels) purely so a CPU trace on a short input runs
# quickly -- the module topology (WaveNet resblock + VNN2 head) is unchanged.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


def slicing(x, slice_idx, axes):
    """Real WaveNet_VNNs/utils.py:slicing, verbatim."""
    dimensionality = len(x.shape)
    if dimensionality == 3:
        if axes == 1:
            return x[:, slice_idx, :]
        if axes == 2:
            return x[:, :, slice_idx]
    if dimensionality == 2:
        if axes == 0:
            return x[slice_idx, :]
        if axes == 1:
            return x[:, slice_idx]
    return None


class Causal_Conv1d(nn.Module):
    """Real WaveNet_VNNs/networks.py:Causal_Conv1d, verbatim."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
            dilation=dilation,
        )

    def forward(self, x):
        x = F.pad(x, (self.dilation * (self.kernel_size - 1), 0), mode="constant", value=0)
        x = self.conv(x)
        return x


class VNN2(nn.Module):
    """Real WaveNet_VNNs/networks.py:VNN2 (2nd-order Volterra neural
    network head), verbatim."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.Q2 = config["VNN2"]["Q2"]
        self.out_channel = config["VNN2"]["conv1d"]["out"][0]
        self.conv1 = Causal_Conv1d(
            self.config["VNN2"]["conv1d"]["input"][0],
            self.config["VNN2"]["conv1d"]["out"][0],
            self.config["VNN2"]["conv1d"]["kernel"][0],
            stride=1,
            bias=False,
        )
        self.conv2 = Causal_Conv1d(
            self.config["VNN2"]["conv1d"]["input"][1],
            2 * self.Q2 * self.config["VNN2"]["conv1d"]["out"][1],
            self.config["VNN2"]["conv1d"]["kernel"][1],
            stride=1,
            bias=False,
        )

    def forward(self, x):
        linear_term = self.conv1(x)  # first-order

        x2 = self.conv2(x)
        x2_mul = torch.mul(
            x2[:, 0 : self.Q2 * self.out_channel, :],
            x2[:, self.Q2 * self.out_channel : 2 * self.Q2 * self.out_channel, :],
        )
        x2_add = torch.zeros_like(linear_term)
        for q in range(self.Q2):
            x2_add = torch.add(
                x2_add,
                x2_mul[:, (q * self.out_channel) : ((q * self.out_channel) + self.out_channel), :],
            )
        quad_term = x2_add  # second-order
        x = torch.add(linear_term, quad_term)
        return x.squeeze()


class dilated_residual_block(nn.Module):
    """Real WaveNet_VNNs/networks.py:dilated_residual_block (WaveNet gated
    residual block), verbatim."""

    def __init__(self, dilation, config):
        super().__init__()
        self.dilation = dilation
        self.config = config
        self.conv1 = Causal_Conv1d(
            self.config["WaveNet"]["Resblock"]["conv1d"]["res"],
            2 * self.config["WaveNet"]["Resblock"]["conv1d"]["res"],
            kernel_size=self.config["WaveNet"]["Resblock"]["conv1d"]["kernel"][0],
            stride=1,
            bias=False,
            dilation=self.dilation,
        )
        self.conv2 = Causal_Conv1d(
            self.config["WaveNet"]["Resblock"]["conv1d"]["res"],
            self.config["WaveNet"]["Resblock"]["conv1d"]["res"]
            + self.config["WaveNet"]["Resblock"]["conv1d"]["skip"],
            self.config["WaveNet"]["Resblock"]["conv1d"]["kernel"][1],
            stride=1,
            bias=False,
        )

    def forward(self, data_x):
        original_x = data_x
        data_out = self.conv1(data_x)
        data_out_1 = slicing(
            data_out, slice(0, self.config["WaveNet"]["Resblock"]["conv1d"]["res"], 1), 1
        )
        data_out_2 = slicing(
            data_out,
            slice(
                self.config["WaveNet"]["Resblock"]["conv1d"]["res"],
                2 * self.config["WaveNet"]["Resblock"]["conv1d"]["res"],
                1,
            ),
            1,
        )
        tanh_out = torch.tanh(data_out_1)
        sigm_out = torch.sigmoid(data_out_2)
        data_x = tanh_out * sigm_out
        data_x = self.conv2(data_x)
        res_x = slicing(data_x, slice(0, self.config["WaveNet"]["Resblock"]["conv1d"]["res"], 1), 1)
        skip_x = slicing(
            data_x,
            slice(
                self.config["WaveNet"]["Resblock"]["conv1d"]["res"],
                self.config["WaveNet"]["Resblock"]["conv1d"]["res"]
                + self.config["WaveNet"]["Resblock"]["conv1d"]["skip"],
                1,
            ),
            1,
        )
        res_x = res_x + original_x
        return res_x, skip_x


class WaveNet_VNNs(nn.Module):
    """Real WaveNet_VNNs/networks.py:WaveNet_VNNs, verbatim (Deep-ANC
    secondary-path nonlinearity compensator)."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_stacks = self.config["WaveNet"]["num_stacks"]
        if isinstance(self.config["WaveNet"]["dilations"], int):
            self.dilations = [2**i for i in range(0, self.config["WaveNet"]["dilations"] + 1)]
        elif isinstance(self.config["WaveNet"]["dilations"], list):
            self.dilations = self.config["WaveNet"]["dilations"]

        self.num_residual_blocks = len(self.dilations) * self.num_stacks

        self.conv1 = Causal_Conv1d(
            self.config["WaveNet"]["conv"]["input"][0],
            self.config["WaveNet"]["conv"]["out"][0],
            self.config["WaveNet"]["conv"]["kernel"][0],
            stride=1,
            bias=False,
        )

        self.conv2 = Causal_Conv1d(
            self.config["WaveNet"]["conv"]["input"][1],
            self.config["WaveNet"]["conv"]["out"][1],
            self.config["WaveNet"]["conv"]["kernel"][1],
            stride=1,
            bias=False,
        )

        self.conv3 = Causal_Conv1d(
            self.config["WaveNet"]["conv"]["input"][2],
            self.config["WaveNet"]["conv"]["out"][2],
            self.config["WaveNet"]["conv"]["kernel"][2],
            stride=1,
            bias=False,
        )

        self.conv4 = Causal_Conv1d(
            self.config["WaveNet"]["conv"]["input"][3],
            self.config["WaveNet"]["conv"]["out"][3],
            self.config["WaveNet"]["conv"]["kernel"][3],
            stride=1,
            bias=False,
        )
        self.dilated_layers = nn.ModuleList(
            [dilated_residual_block(dilation, self.config) for dilation in self.dilations]
        )
        self.VNN = VNN2(self.config)

    def forward(self, x):
        data_input = x
        data_expanded = data_input
        data_out = self.conv1(data_expanded)
        skip_connections = []
        for _ in range(self.num_stacks):
            for layer in self.dilated_layers:
                data_out, skip_out = layer(data_out)
                if skip_out is not None:
                    skip_connections.append(skip_out)

        data_out = torch.stack(skip_connections, dim=0).sum(dim=0)
        data_out = F.tanh(data_out)
        data_out = self.conv2(data_out)
        data_out = F.tanh(data_out)
        data_out = self.conv3(data_out)
        data_out = F.tanh(data_out)
        data_out = self.conv4(data_out)
        data_out = F.tanh(data_out)
        data_out = self.VNN(data_out).squeeze()
        return data_out


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------
# Real config.json channel/kernel values reused unmodified; only num_stacks
# (3->1) and dilations (9->3, i.e. dilations [1,2,4,8]) shrunk for a quick
# CPU trace on a short 1-D waveform input.
_CONFIG = {
    "WaveNet": {
        "num_stacks": 1,
        "dilations": 3,
        "Resblock": {
            "conv1d": {
                "res": 32,
                "skip": 32,
                "kernel": [3, 1],
            }
        },
        "conv": {
            "input": [1, 32, 128, 32],
            "out": [32, 128, 32, 16],
            "kernel": [3, 3, 3, 1],
        },
    },
    "VNN2": {
        "Q2": 4,
        "conv1d": {
            "input": [16, 16],
            "out": [1, 1],
            "kernel": [16, 16],
        },
    },
}

_BATCH = 2
_SEQ_LEN = 256


def build_wavenet_vnns_anc():
    torch.manual_seed(0)
    model = WaveNet_VNNs(_CONFIG)
    model.eval()
    return model


def example_input_wavenet_vnns_anc():
    torch.manual_seed(0)
    return torch.randn(_BATCH, 1, _SEQ_LEN)


MENAGERIE_ENTRIES = [
    (
        "WaveNet-VNNs-DeepANC",
        "build_wavenet_vnns_anc",
        "example_input_wavenet_vnns_anc",
        2024,
        MENAGERIE_ZOO,
    ),
]
