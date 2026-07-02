# SOURCE: vendored from ml-jku/mc-lstm @ 8bbaece3ecb4187a76c6318d4c6e40c1dcc71303
# (mclstm.py)
#
# MC-LSTM (Mass-Conserving LSTM): an LSTM variant that enforces exact conservation of a
# subset of ("mass") inputs across the cell state -- the input gate and redistribution
# matrix are both column-normalized (softmax) so mass entering a cell is exactly
# accounted for, and the output gate is the only place mass can leave. Used for physically
# constrained hydrology / rainfall-runoff modeling (ICML 2021). MassConservingLSTM/MCGate
# below are the REAL model code from mclstm.py, copied verbatim (only base-lib deps:
# torch, numpy).

import numpy as np
import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class MassConservingLSTM(nn.Module):
    """Pytorch implementation of Mass-Conserving LSTMs."""

    def __init__(
        self,
        in_dim: int,
        aux_dim: int,
        out_dim: int,
        in_gate: nn.Module = None,
        out_gate: nn.Module = None,
        redistribution: nn.Module = None,
        time_dependent: bool = True,
        batch_first: bool = False,
    ):
        """
        Parameters
        ----------
        in_dim : int
            The number of mass inputs.
        aux_dim : int
            The number of auxiliary inputs.
        out_dim : int
            The number of cells or, equivalently, outputs.
        in_gate : nn.Module, optional
            A module computing the (normalised!) input gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `in_dim` x `out_dim` matrix for every sample.
            Defaults to a time-dependent softmax input gate.
        out_gate : nn.Module, optional
            A module computing the output gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` vector for every sample.
        redistribution : nn.Module, optional
            A module computing the redistribution matrix.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` x `out_dim` matrix for every sample.
        time_dependent : bool, optional
            Use time-dependent gates if `True` (default).
            Otherwise, use only auxiliary inputs for gates.
        batch_first : bool, optional
            Expects first dimension to represent samples if `True`,
            Otherwise, first dimension is expected to represent timesteps (default).
        """
        super().__init__()
        self.in_dim = in_dim
        self.aux_dim = aux_dim
        self.out_dim = out_dim
        self._seq_dim = 1 if batch_first else 0

        gate_kwargs = {
            "aux_dim": aux_dim,
            "out_dim": out_dim if time_dependent else None,
            "normaliser": nn.Softmax(dim=-1),
        }
        if redistribution is None:
            redistribution = MCGate((out_dim, out_dim), **gate_kwargs)
        if in_gate is None:
            in_gate = MCGate((in_dim, out_dim), **gate_kwargs)
        if out_gate is None:
            gate_kwargs["normaliser"] = nn.Sigmoid()
            out_gate = MCGate((out_dim,), **gate_kwargs)

        self.redistribution = redistribution
        self.in_gate = in_gate
        self.out_gate = out_gate

    @property
    def batch_first(self) -> bool:
        return self._seq_dim != 0

    def reset_parameters(self, out_bias: float = -3.0):
        """
        Parameters
        ----------
        out_bias : float, optional
            The initial bias value for the output gate (default to -3).
        """
        self.redistribution.reset_parameters(bias_init=nn.init.eye_)
        self.in_gate.reset_parameters(bias_init=nn.init.zeros_)
        self.out_gate.reset_parameters(bias_init=lambda b: nn.init.constant_(b, val=out_bias))

    def forward(self, xm, xa, state=None):
        xm = xm.unbind(dim=self._seq_dim)
        xa = xa.unbind(dim=self._seq_dim)

        if state is None:
            state = self.init_state(len(xa[0]))

        hs, cs = [], []
        for xm_t, xa_t in zip(xm, xa):
            h, state = self._step(xm_t, xa_t, state)
            hs.append(h)
            cs.append(state)

        hs = torch.stack(hs, dim=self._seq_dim)
        cs = torch.stack(cs, dim=self._seq_dim)
        return hs, cs

    @torch.no_grad()
    def init_state(self, batch_size: int):
        """Create the default initial state."""
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.out_dim, device=device)

    def _step(self, xm_t, xa_t, c_t):
        """Implementation of MC-LSTM recurrence."""
        r = self.redistribution(xm_t, xa_t, c_t)
        i = self.in_gate(xm_t, xa_t, c_t)
        o = self.out_gate(xm_t, xa_t, c_t)

        c = torch.matmul(c_t.unsqueeze(-2), r).squeeze(-2)
        c = c + torch.matmul(xm_t.unsqueeze(-2), i).squeeze(-2)
        h = o * c
        c = c - h
        return h, c

    def autoregress(self, c0: torch.Tensor, xa: torch.Tensor, xm: torch.Tensor = None):
        """
        Use MC-LSTM in an autoregressive fashion.

        By operating on the cell states of MC-LSTM directly,
        the MC-LSTM can be used as an auto-regressive model.

        Parameters
        ----------
        c0 : (B, out_dim) torch.Tensor
            The initial cell state for the MC-LSTM or, equivalently,
            the starting point for the auto-regression.
        xa : (L, B, aux_dim) torch.Tensor
            A sequence of auxiliary inputs for the MC-LSTM.
            The output sequence will have the same length `L` as the given sequence.
            If not specified, the sequence consists of
            `length` equally spaced points between 0 and 1.
        xm : (L, B, in_dim) torch.Tensor, optional
            A sequence of mass inputs for the MC-LSTM.
            This sequence must have the same length as `xa`.
            If not specified, a sequence of zeros is used.

        Returns
        -------
        y : (L, B, out_dim) torch.Tensor
            The sequence of cell states from the MC-LSTM or, equivalently,
            the outputs of the autoregression.
            The length of the sequence is specified is the length of `xa`.
        h : (L, B, out_dim) torch.Tensor
            The sequence of outputs produced by the MC-LSTM.
            This sequence should contain all mass that disappeared over time,
            and has the same length as `y`.

        Notes
        -----
        If `self.batch_first` is `True`, the documented shapes of
        input and output sequences should be `(B, L, ...)` instead of `(L, B, ...)`.

        """
        if len(c0.shape) != 2 or c0.size(1) != self.out_dim:
            raise ValueError(f"cell state must have shape (?, {self.out_dim})")
        if xa.size(-1) != self.aux_dim:
            raise ValueError(f"auxiliary input must have shape (..., {self.aux_dim})")
        if xm is None:
            xm = torch.zeros(*xa.shape[:-1], self.in_dim)
        elif xm.size(-1) != self.in_dim:
            raise ValueError(f"mass input must have shape (..., {self.in_dim})")

        h, y = self.forward(xm, xa, state=c0)
        return y, h


class MCGate(nn.Module):
    """Default gating logic for MC-LSTM."""

    def __init__(
        self,
        shape: tuple,
        aux_dim: int,
        out_dim: int = None,
        in_dim: int = None,
        normaliser: nn.Module = nn.Softmax(dim=-1),
    ):
        """
        Parameters
        ----------
        shape : tuple of ints
            The output shape for this gate.
        aux_dim : int
            The number of auxiliary inputs per timestep.
        out_dim : int, optional
            The number of accumulation cells.
            If `None`, the cell states are not used in the gating (default).
        in_dim : int, optional
            The number of mass inputs per timestep.
            If `None`, the mass inputs are not used in the gating (default).
        normaliser : nn.Module, optional
            The activation function to use for computing the gates.
            This function is responsible for any normalisation of the gate.
        """
        super().__init__()
        batch_dim = 1 if any(n == 0 for n in shape) else -1
        self.out_shape = (batch_dim, *shape)
        self.use_mass = in_dim is not None
        self.use_state = out_dim is not None

        gate_dim = aux_dim
        if self.use_mass:
            gate_dim += in_dim
        if self.use_state:
            gate_dim += out_dim

        self.connections = nn.Linear(gate_dim, np.prod(shape).item())
        self.normaliser = normaliser

    def reset_parameters(self, bias_init=nn.init.zeros_):
        """
        Parameters
        ----------
        bias_init : callable
            Initialisation function for the bias parameter (in-place).
        """
        bias_init(self.connections.bias.view(self.out_shape[1:]))
        nn.init.orthogonal_(self.connections.weight)

    def forward(self, xm, xa, c):
        inputs = [xa]
        if self.use_mass:
            xm_sum = torch.sum(xm, dim=-1, keepdims=True)
            scale = torch.where(xm_sum == 0, xm_sum.new_ones(1), xm_sum)
            inputs.append(xm / scale)
        if self.use_state:
            c_sum = torch.sum(c, dim=-1, keepdims=True)
            scale = torch.where(c_sum == 0, c_sum.new_ones(1), c_sum)
            inputs.append(c / scale)

        x_ = torch.cat(inputs, dim=-1)
        s = self.connections(x_)
        s = s.view(self.out_shape)
        return self.normaliser(s)


def build_mclstm():
    """Tiny MassConservingLSTM for tracing: small mass/aux/out dims."""
    model = MassConservingLSTM(in_dim=2, aux_dim=3, out_dim=4, batch_first=False)
    model.eval()
    return model


def example_input_mclstm():
    seq_len, batch = 5, 2
    xm = torch.rand(seq_len, batch, 2)
    xa = torch.rand(seq_len, batch, 3)
    return (xm, xa)


MENAGERIE_ENTRIES = [
    ("MC-LSTM", build_mclstm, example_input_mclstm, 2021, "vendored-pytorch"),
]
