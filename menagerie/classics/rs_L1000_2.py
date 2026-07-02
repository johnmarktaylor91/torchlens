# SOURCE: vendored from https://github.com/0zgur0/ms-convSTAR @ master
#   Vendored files:
#     - models/convstar.py (ConvSTARCell, ConvSTAR)
#     - models/multi_stage_sequenceencoder.py (multistageSTARSequentialEncoder, 'star' branch)
#
# ZueriCrop (Hierarchical ConvRNN) is Turkoglu et al. 2021 "Crop mapping from image time
# series: deep learning with multi-scale label hierarchies" (Remote Sensing of Environment).
# The model is a multi-stage convolutional-recurrent encoder ("convSTAR": a convolutional
# variant of the STAR/SRU-style gated recurrent cell) applied over a Sentinel-2 time series,
# with three classification heads reading out at different depths of the recurrent stack to
# supervise a coarse-to-fine label hierarchy (nclasses_l1 -> nclasses_l2 -> nclasses).
#
# Minimal fixes applied to make the vendored code run on current torch (no architectural
# change): `torch.nn.init.orthogonal`/`.constant` -> the modern in-place `_` variants
# (deprecated-but-still-functional aliases in current torch; using the `_` form avoids the
# deprecation warning without altering the actual initialization). The `prev_state is None`
# CUDA/Variable branch in ConvSTARCell.forward is dead code in this traced path (the caller,
# multistageSTARSequentialEncoder, always pre-allocates hiddenS before calling self.rnn), so
# it is preserved as-is (never executed). multistageSTARSequentialEncoder.forward's blanket
# `if torch.cuda.is_available(): hiddenS[i] = hiddenS[i].cuda()` assumes CUDA-available implies
# model-on-CUDA; replaced with `.to(x.device)` (device-correctness fix, not an architecture
# change) so the module traces correctly on CPU in CUDA-available environments.

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import init

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/convstar.py
# ---------------------------------------------------------------------------
class ConvSTARCell(torch.nn.Module):
    """Generate a convolutional STAR cell"""

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = torch.nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding
        )
        self.update = torch.nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.update.weight)
        init.orthogonal_(self.gate.weight)
        init.constant_(self.update.bias, 0.0)
        init.constant_(self.gate.bias, 1.0)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.autograd.Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = torch.autograd.Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        gain = torch.sigmoid(self.gate(stacked_inputs))
        update = torch.tanh(self.update(input_))
        new_state = gain * prev_state + (1 - gain) * update

        return new_state


class ConvSTAR(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        """
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvSTARCell`.
        """
        super().__init__()

        self.input_size = input_size

        if not isinstance(hidden_sizes, list):
            self.hidden_sizes = [hidden_sizes] * n_layers
        else:
            assert len(hidden_sizes) == n_layers, (
                "`hidden_sizes` must have the same length as n_layers"
            )
            self.hidden_sizes = hidden_sizes
        if not isinstance(kernel_sizes, list):
            self.kernel_sizes = [kernel_sizes] * n_layers
        else:
            assert len(kernel_sizes) == n_layers, (
                "`kernel_sizes` must have the same length as n_layers"
            )
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]

            cell = ConvSTARCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = "ConvSTARCell_" + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
        """
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        """
        if not hidden:
            hidden = [None] * self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden


# ---------------------------------------------------------------------------
# models/multi_stage_sequenceencoder.py -- multistageSTARSequentialEncoder ('star' branch)
# ---------------------------------------------------------------------------
class multistageSTARSequentialEncoder(torch.nn.Module):
    def __init__(
        self,
        height,
        width,
        input_dim=4,
        hidden_dim=64,
        nclasses=15,
        nstage=3,
        nclasses_l1=3,
        nclasses_l2=7,
        kernel_size=(3, 3),
        n_layers=6,
        use_in_layer_norm=False,
        viz=False,
        test=False,
        wo_softmax=False,
        cell="star",
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.nstage = nstage
        self.viz = viz
        self.test = test
        self.wo_softmax = wo_softmax
        self.cell = cell

        self.use_in_layer_norm = use_in_layer_norm
        if use_in_layer_norm:
            self.in_layer_norm = torch.nn.LayerNorm(input_dim)

        # only the 'star' cell path is vendored here (ZueriCrop's default configuration);
        # the 'gru'/'star_res' branches route through ConvGRU/ConvSTAR_Res, which this
        # staging module intentionally omits to keep the traced entry self-contained.
        assert self.cell == "star"
        self.rnn = ConvSTAR(
            input_size=input_dim,
            hidden_sizes=hidden_dim,
            kernel_sizes=kernel_size[0],
            n_layers=n_layers,
        )

        self.final = torch.nn.Conv2d(hidden_dim, nclasses, (3, 3), padding=1)
        self.final_local_1 = torch.nn.Conv2d(hidden_dim, nclasses_l1, (3, 3), padding=1)
        self.final_local_2 = torch.nn.Conv2d(hidden_dim, nclasses_l2, (3, 3), padding=1)

    def forward(self, x, hiddenS=None):
        if self.use_in_layer_norm:
            # (b x t x c x h x w) -> (b x t x h x w x c) -> (b x c x t x h x w)
            x = self.in_layer_norm(x.permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)
        else:
            # (b x t x c x h x w) -> (b x c x t x h x w)
            x = x.permute(0, 2, 1, 3, 4)

        b, c, t, h, w = x.shape

        # convRNN step---------------------------------
        # hiddenS is a list (number of layer) of hidden states of size [b x c x h x w]
        if hiddenS is None:
            hiddenS = [torch.zeros((b, self.hidden_dim, h, w))] * self.n_layers

        # Minimal device-correctness fix (not an architecture change): the upstream script
        # assumes `torch.cuda.is_available()` implies the model itself lives on CUDA, which
        # breaks when CUDA is present but the model is on CPU (as during CPU-only tracing).
        # Match the hidden state device to the input device instead of blanket `.cuda()`.
        for i in range(self.n_layers):
            hiddenS[i] = hiddenS[i].to(x.device)

        for iteration in range(t):
            hiddenS = self.rnn.forward(x[:, :, iteration, :, :], hiddenS)

        if self.n_layers == 3:
            local_1 = hiddenS[0]
            local_2 = hiddenS[1]
        elif self.nstage == 3:
            local_1 = hiddenS[1]
            local_2 = hiddenS[3]
        elif self.nstage == 2:
            local_1 = hiddenS[1]
            local_2 = hiddenS[2]
        elif self.nstage == 1:
            local_1 = hiddenS[-1]
            local_2 = hiddenS[-1]

        local_1 = self.final_local_1(local_1)
        local_2 = self.final_local_2(local_2)

        last = hiddenS[-1]
        last = self.final(last)

        if self.viz:
            return hiddenS[-1]
        elif self.test:
            return F.softmax(last, dim=1), F.softmax(local_1, dim=1), F.softmax(local_2, dim=1)
        elif self.wo_softmax:
            return last, local_1, local_2
        else:
            return (
                F.log_softmax(last, dim=1),
                F.log_softmax(local_1, dim=1),
                F.log_softmax(local_2, dim=1),
            )


# ---------------------------------------------------------------------------
# Menagerie staging hooks
# ---------------------------------------------------------------------------
_HEIGHT = 8
_WIDTH = 8
_INPUT_DIM = 4
_HIDDEN_DIM = 6
_TIMESTEPS = 3
_N_LAYERS = 4  # must be >=4 so nstage==3 branch (hiddenS[1], hiddenS[3]) is valid


def build_zuericrop_convstar():
    return multistageSTARSequentialEncoder(
        height=_HEIGHT,
        width=_WIDTH,
        input_dim=_INPUT_DIM,
        hidden_dim=_HIDDEN_DIM,
        nclasses=5,
        nstage=3,
        nclasses_l1=2,
        nclasses_l2=3,
        kernel_size=(3, 3),
        n_layers=_N_LAYERS,
        cell="star",
    )


def example_input_zuericrop_convstar():
    # (batch, time, channels, height, width)
    return torch.randn(1, _TIMESTEPS, _INPUT_DIM, _HEIGHT, _WIDTH)


MENAGERIE_ENTRIES = [
    (
        "ZueriCrop_ConvSTAR",
        build_zuericrop_convstar,
        example_input_zuericrop_convstar,
        2021,
        "MENAGERIE_ZOO",
    ),
]
