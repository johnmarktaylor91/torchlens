# SOURCE: vendored from fteufel/PyTorch-GRU-D @ master
#   src/GRUD_layer.py :: GRUD_cell
#   src/GRUD_model.py :: grud_model
#
# GRU-D (Che et al., "Recurrent Neural Networks for Multivariate Time Series
# with Missing Values", Scientific Reports 2018) -- a GRU variant with
# trainable exponential decay of both the input toward its empirical mean and
# the hidden state, driven by an explicit missingness mask and elapsed-time
# ("delta") channel per timestep. This is a PyTorch reimplementation of the
# original paper's Keras/TF1 code (github.com/PeterChe1990/GRU-D, listed in
# the queue notes), vendored verbatim (imports/formatting only adjusted; no
# architectural changes) since it is a pure-torch, base-lib-only port of the
# real architecture and directly importable in this environment.

import math
import numbers
import warnings

import torch

MENAGERIE_ZOO = "vendored-pytorch"


# ---- src/GRUD_layer.py (verbatim, GRUD_cell) ----


class GRUD_cell(torch.nn.Module):
    """
    Implementation of GRUD.
    Inputs: x_mean
            n_smp x 3 x n_channels x len_seq tensor (0: data, 1: mask, 2: deltat)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        x_mean=0,
        bias=True,
        batch_first=False,
        bidirectional=False,
        dropout_type="mloss",
        dropout=0,
        return_hidden=False,
    ):
        super(GRUD_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_hidden = (
            return_hidden  # controls the output, True if another GRU-D layer follows
        )

        x_mean = torch.tensor(x_mean, requires_grad=True)
        self.register_buffer("x_mean", x_mean)
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional

        if (
            not isinstance(dropout, numbers.Number)
            or not 0 <= dropout <= 1
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        if dropout > 0 and num_layers == 1:
            warnings.warn(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                "num_layers greater than 1, but got dropout={} and "
                "num_layers={}".format(dropout, num_layers)
            )

        # set up all the operations that are needed in the forward pass
        self.w_dg_x = torch.nn.Linear(input_size, input_size, bias=True)
        self.w_dg_h = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.w_xz = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mz = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.w_xr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_xh = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mh = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.w_hy = torch.nn.Linear(hidden_size, output_size, bias=True)

        Hidden_State = torch.zeros(self.hidden_size, requires_grad=True)
        # we use buffers because pytorch will take care of pushing them to GPU for us
        self.register_buffer("Hidden_State", Hidden_State)
        self.register_buffer("X_last_obs", torch.zeros(input_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input):
        # input.size = (3, 33,49) : num_input or num_hidden, num_layer or step
        X = input[:, 0, :, :]
        Mask = input[:, 1, :, :]
        Delta = input[:, 2, :, :]

        output = None
        h = getattr(self, "Hidden_State")
        x_mean = getattr(self, "x_mean")
        x_last_obsv = getattr(self, "X_last_obs")

        device = next(self.parameters()).device
        output_tensor = torch.empty(
            [X.size()[0], X.size()[2], self.output_size], dtype=X.dtype, device=device
        )
        hidden_tensor = torch.empty(
            X.size()[0], X.size()[2], self.hidden_size, dtype=X.dtype, device=device
        )

        # iterate over seq
        for timestep in range(X.size()[2]):
            x = torch.squeeze(X[:, :, timestep])
            m = torch.squeeze(Mask[:, :, timestep])
            d = torch.squeeze(Delta[:, :, timestep])

            # (4)
            gamma_x = torch.exp(-1 * torch.nn.functional.relu(self.w_dg_x(d)))
            gamma_h = torch.exp(-1 * torch.nn.functional.relu(self.w_dg_h(d)))

            # (5)
            x_last_obsv = torch.where(m > 0, x, x_last_obsv)
            x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * x_mean)
            x = m * x + (1 - m) * (gamma_x * x_last_obsv + (1 - gamma_x) * x_mean)

            # (6) -- dropout_type == 'mloss' path (recurrent dropout without memory loss, arXiv 1603.05118)
            h = gamma_h * h
            z = torch.sigmoid(self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
            r = torch.sigmoid(self.w_xr(x) + self.w_hr(h) + self.w_mr(m))

            dropout = torch.nn.Dropout(p=self.dropout)  # noqa: F841 -- built but unused in real code (mloss branch never applies it)
            h_tilde = torch.tanh(self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m))

            h = (1 - z) * h + z * h_tilde

            step_output = self.w_hy(h)
            step_output = torch.sigmoid(step_output)
            output_tensor[:, timestep, :] = step_output
            hidden_tensor[:, timestep, :] = h

        output = output_tensor, hidden_tensor
        return output


# ---- src/GRUD_model.py (verbatim, grud_model) ----


class grud_model(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        x_mean=0,
        bias=True,
        batch_first=False,
        bidirectional=False,
        dropout_type="mloss",
        dropout=0,
    ):
        super(grud_model, self).__init__()

        self.gru_d = GRUD_cell(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
            dropout_type=dropout_type,
            x_mean=x_mean,
        )
        self.hidden_to_output = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if self.num_layers > 1:
            # (batch, seq, feature)
            self.gru_layers = torch.nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=self.num_layers - 1,
                dropout=dropout,
            )

    def initialize_hidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(self.num_layers - 1, batch_size, self.hidden_size, device=device)

    def forward(self, input):
        # pass through GRU-D
        output, hidden = self.gru_d(input)

        if self.num_layers > 1:
            init_hidden = self.initialize_hidden(hidden.size()[0])  # noqa: F841 -- real code marks this dead ("#TODO remove init hidden, not necessary")
            output, hidden = self.gru_layers(hidden)
            output = self.hidden_to_output(output)
            output = torch.sigmoid(output)

        return output


def build_grud():
    torch.manual_seed(0)
    input_size, hidden_size, output_size = 12, 16, 1
    return grud_model(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=1,
        x_mean=0.0,
        dropout=0.0,
    ).eval()


def example_input_grud():
    torch.manual_seed(0)
    batch, n_channels, seq_len = 2, 12, 8
    data = torch.randn(batch, n_channels, seq_len)
    mask = torch.randint(0, 2, (batch, n_channels, seq_len)).float()
    delta = torch.rand(batch, n_channels, seq_len)
    x = torch.stack([data, mask, delta], dim=1)  # (batch, 3, n_channels, seq_len)
    return (x,)


MENAGERIE_ENTRIES = [
    ("GRU-D", build_grud, example_input_grud, 2018, "vendored-pytorch"),
]
