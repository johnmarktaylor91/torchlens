# SOURCE: vendored from georgeyiasemis/RIM-PyTorch @ d8917c7aea926db516ba659a62c7ceea98c5f877
# https://raw.githubusercontent.com/georgeyiasemis/RIM-PyTorch/main/rim.ipynb
#
# Recurrent Inference Machine (RIM), "Recurrent Inference Machines for Solving
# Inverse Problems" (Putzky & Welling, 2017) -- a learned iterative solver that
# unrolls a convolutional-GRU-style recurrent network over the gradient of a
# data-consistency (measurement) term. Every class (`Conv2dRNNCell`, `RNN`, `RIM`)
# is transcribed verbatim from the `rim.ipynb` code cells. The only change is
# replacing the notebook's free (module-global) `device` variable -- set once at
# notebook top-level from `torch.cuda.is_available()` and referenced inside
# `RIM.forward` -- with `x.device`, since a standalone vendored module has no such
# global; this is a portability fix (device placement), not an architectural
# change (identical tensor ops on identical devices as the input).

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class Conv2dRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True, nonlinearity="tanh"):
        super(Conv2dRNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if type(kernel_size) == tuple and len(kernel_size) == 2:
            self.kernel_size = kernel_size
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
            self.padding = (kernel_size // 2, kernel_size // 2)
        else:
            raise ValueError("Invalid kernel size.")

        self.bias = bias
        self.nonlinearity = nonlinearity

        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        self.x2h = nn.Conv2d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias,
        )

        self.h2h = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        # Inputs:
        #       input: of shape (batch_size, input_size, height_size, width_size)
        #       hx: of shape (batch_size, hidden_size, height_size, width_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size, height_size, width_size)

        if hx is None:
            hx = Variable(
                input.new_zeros(input.size(0), self.hidden_size, input.size(2), input.size(3))
            )
        # print(input.shape)
        hy = self.x2h(input) + self.h2h(hx)

        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)

        return hy


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_sizes, output_size, activation="relu"):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.rnncell1 = Conv2dRNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            kernel_size=kernel_sizes[0],
            nonlinearity=activation,
        )
        self.rnncell2 = Conv2dRNNCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
            kernel_size=kernel_sizes[1],
            nonlinearity=activation,
        )
        self.conv = nn.Conv2d(
            in_channels=hidden_size * 2,
            out_channels=output_size,
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2] // 2,
        )

        # nn.init.xavier_normal_(self.fc.weight, 0.1)

    def forward(self, xt, st=None):
        if st is None:
            st = [
                Variable(xt.new_zeros(xt.size(0), self.hidden_size, xt.size(2), xt.size(3))),
                Variable(xt.new_zeros(xt.size(0), self.hidden_size, xt.size(2), xt.size(3))),
            ]

        st_1 = self.rnncell1(xt, st[0])
        st_2 = self.rnncell2(st_1, st[1])
        dxt = self.conv(torch.cat((st_1, st[1]), 1))

        st = [st_1, st_2]

        return dxt, st


class RIM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_sizes,
        output_size,
        sequence_size,
        gradient_fun,
        activation="relu",
    ):
        super(RIM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        self.output_size = output_size
        self.sequence_size = sequence_size
        self.gradient_fun = gradient_fun

        # input is ([x, grad_{y|x}])
        self.rnn = RNN(
            input_size=input_size * 2,
            hidden_size=hidden_size,
            kernel_sizes=kernel_sizes,
            output_size=output_size,
            activation=activation,
        )

    def forward(self, x, y):
        device = x.device

        # x is only used for shape matching
        x0 = torch.zeros(x.shape)
        xt = x0.clone().to(device)

        grad_xt = self.gradient_fun(xt, y).clone().to(device)

        input_t = Variable(torch.cat((xt, grad_xt), 1)).to(device)
        st = None
        X = torch.zeros(
            (self.sequence_size + 1, x0.size(0), self.input_size, x0.size(2), x0.size(3))
        )
        X[0] = y.clone()
        for t in range(self.sequence_size):
            dxt, st = self.rnn(input_t, st)

            xt = xt + dxt

            grad_xt = self.gradient_fun(xt, y).clone()
            input_t = Variable(torch.cat((xt, grad_xt), 1))

            X[t + 1] = xt.clone()

        return X.to(device)


def _quadratic_data_gradient(x, y):
    """Gradient of the squared-error data-consistency term 0.5*||x - y||^2, i.e. x - y.

    Standard RIM usage (see `rim.ipynb` cell 11, `def gradient(X, Y): return X - Y`):
    the network is unrolled over the gradient of a per-problem measurement/data
    term; this quadratic denoising term is the notebook's own worked example.
    """
    return x - y


def build_rim():
    torch.manual_seed(0)
    model = RIM(
        input_size=1,
        hidden_size=8,
        kernel_sizes=(5, 3, 3),
        output_size=1,
        sequence_size=3,
        gradient_fun=_quadratic_data_gradient,
        activation="relu",
    )
    model.eval()
    return model


def example_input_rim():
    torch.manual_seed(0)
    # (x, y): x is only used for shape matching (zeroed internally), y is the
    # noisy/degraded measurement being iteratively denoised. Both (N, 1, H, W).
    x = torch.randn(1, 1, 16, 16)
    y = torch.randn(1, 1, 16, 16)
    return (x, y)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RIM", "build_rim", "example_input_rim", 2017, MENAGERIE_ZOO),
]
