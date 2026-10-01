"""Intervention honesty: Selection-shaped ``do(sel, value)`` validates like
the string-label path (WT A-II item 9 -- the validation bypass).

A raw value on the Selection lane was lifted through the tensor-only
``replace_with`` helper unchecked, so ``do(sel, 0.0)`` crashed bare
(``AttributeError: 'float' object has no attribute 'to'``) at push time on
all three kinds (ACT/EDGE/PARAM). Non-tensor replacement values now refuse
typed at the lift, teaching the tensor and helper spellings; tensor values
keep working under the scatter contract.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.selection import SelectionError

pytestmark = pytest.mark.smoke


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))) + 1.0)


@pytest.fixture(scope="module")
def capture():
    torch.manual_seed(0)
    model = _Net().eval()
    x = torch.randn(1, 1, 12, 12)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.mark.parametrize("bad_value", [0.0, 1, [1.0, 2.0], "zero"])
def test_act_selection_scalar_refuses_typed(capture, bad_value) -> None:
    """ACT lane: non-tensor values refuse with teaching, never crash bare."""

    fork = capture.fork()
    sel = tl.units("relu_1_2", [(0, 0, 1, 1)])
    with pytest.raises(ValueError) as excinfo:
        fork.do(sel, bad_value)
    message = str(excinfo.value)
    assert "replacement value" in message
    assert "full_like" in message or "zero_ablate" in message
    # No partial mutation happened: the fork still matches capture truth.
    assert torch.equal(fork["relu_1_2"].out, capture["relu_1_2"].out)


def test_edge_selection_scalar_refuses_typed(capture) -> None:
    """EDGE lane: same refusal, same teaching."""

    edge = next(e for e in capture.edges if e.parent_label == "relu_1_2")
    fork = capture.fork()
    with pytest.raises(SelectionError) as excinfo:
        fork.do(edge.__selection__(), 0.0)
    assert excinfo.value.fields["code"] == "selection_apply_invalid"
    assert "replacement value" in str(excinfo.value)


def test_param_selection_scalar_refuses_typed(capture) -> None:
    """PARAM lane: same refusal, same teaching."""

    fork = capture.fork()
    with pytest.raises(SelectionError) as excinfo:
        fork.do(tl.params("c1.weight"), 0.0)
    assert excinfo.value.fields["code"] == "selection_apply_invalid"
    assert "replacement value" in str(excinfo.value)


def test_full_shape_tensor_value_still_works_masked() -> None:
    """A full-shape tensor value keeps working under the scatter contract."""

    torch.manual_seed(0)
    model = _Net().eval()
    x = torch.randn(1, 1, 12, 12)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    fork = log.fork()
    replacement = torch.full_like(log["relu_1_2"].out, 7.0)
    fork.do(tl.units("relu_1_2", [(0, 0, 1, 1)]), replacement)
    patched = fork["relu_1_2"].out
    assert patched[0, 0, 1, 1].item() == 7.0
    mask = torch.zeros_like(patched, dtype=torch.bool)
    mask[0, 0, 1, 1] = True
    assert torch.equal(patched[~mask], log["relu_1_2"].out[~mask])


def test_wrong_shape_tensor_value_refuses_typed_not_bare(capture) -> None:
    """Shape mismatches stay typed refusals on the Selection lane."""

    fork = capture.fork()
    sel = capture["relu_1_2"].__selection__()
    with pytest.raises(Exception) as excinfo:
        fork.do(sel, torch.ones(2, 2))
    assert not isinstance(excinfo.value, AttributeError)
    assert "shape" in str(excinfo.value)
