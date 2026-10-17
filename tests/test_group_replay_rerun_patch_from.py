"""``tl.patch_from`` under the rerun engine with a live selector.

A hook fired during a live forward sees the provisional capture label
(``<type>_<n>_<raw index>_raw``); the donor lookup must map it to the op the
source trace captured at the same position, pass by pass, and refuse when the
source holds no op of that type and shape there. The rerun path historically
looked the provisional label up among the source's final labels and refused
every live firing (``patch_from source trace has no site 'rsqrt_..._raw:1'``).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import HookValueError, MultiMatchWarning

RERUN = tl.options.InterventionOptions(engine="rerun")
READY = tl.options.CaptureOptions(intervention_ready=True)


class _RmsHead(nn.Module):
    """A linear map, an RMS normalisation through ``torch.rsqrt``, and a readout."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.out = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return self.out(h)


class _ShiftedRmsHead(_RmsHead):
    """The same head with one extra op before the normalisation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.lin(x))
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return self.out(h)


class _TwoPass(nn.Module):
    """One Linear called twice, so the relu after it is a two-pass layer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.lin(x))
        x = torch.relu(self.lin(x))
        return x


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(3)
    return torch.randn(2, 4, generator=gen), torch.randn(2, 4, generator=gen)


def _rerun_patch(model: nn.Module, donor: tl.Trace, x: torch.Tensor, func: str) -> tl.Trace:
    base = tl.trace(model, x, capture=READY)
    fork = base.fork()
    fork.do({tl.func(func): tl.patch_from(donor)}, model=model, x=x, intervention=RERUN)
    return fork


def test_live_rsqrt_takes_the_donor_value() -> None:
    """The live rsqrt receives the donor's rsqrt; the output is the closed form."""

    torch.manual_seed(0)
    model = _RmsHead().eval()
    xa, xb = _inputs()
    donor = tl.trace(model, xb, capture=READY)
    fork = _rerun_patch(model, donor, xa, "rsqrt")

    with torch.no_grad():
        hb = model.lin(xb)
        donor_scale = torch.rsqrt(hb.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = model.out(model.lin(xa) * donor_scale)
    (rsqrt_op,) = [op for op in fork if op.func_name == "rsqrt"]
    (donor_rsqrt,) = [op for op in donor if op.func_name == "rsqrt"]
    assert torch.equal(rsqrt_op.out, donor_rsqrt.out)
    out_label = str(fork.modules["out"].output_ops[0])
    torch.testing.assert_close(fork[out_label].out, expected, rtol=0, atol=1e-6)


def test_live_multipass_firing_takes_its_own_donor_pass() -> None:
    """Each pass of a two-pass layer receives the donor value of the same pass."""

    torch.manual_seed(0)
    model = _TwoPass().eval()
    xa, xb = _inputs()
    donor = tl.trace(model, xb, capture=READY)
    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        fork = _rerun_patch(model, donor, xa, "relu")

    first, second = donor["relu_1_2:1"].out, donor["relu_1_2:2"].out
    assert not torch.equal(first, second)
    assert torch.equal(fork["relu_1_2:1"].out, first)
    assert torch.equal(fork["relu_1_2:2"].out, second)
    with torch.no_grad():
        torch.testing.assert_close(fork["linear_1_1:2"].out, model.lin(first), rtol=0, atol=1e-6)


def test_live_firing_refuses_a_donor_without_that_position() -> None:
    """A donor whose capture order differs refuses instead of patching another op."""

    torch.manual_seed(0)
    model = _RmsHead().eval()
    shifted = _ShiftedRmsHead().eval()
    shifted.load_state_dict(model.state_dict())
    xa, xb = _inputs()
    donor = tl.trace(shifted, xb, capture=READY)
    with pytest.raises(HookValueError, match="same position"):
        _rerun_patch(model, donor, xa, "rsqrt")
