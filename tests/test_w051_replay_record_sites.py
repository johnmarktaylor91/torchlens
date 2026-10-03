"""``do(<Op record>, edit)`` / ``do(<Layer record>, edit)`` (AUD-CODE 4.10).

A record is the most precise site spelling a user can hold, yet the hook
door refused it ("Unsupported site query") and the spec lowering would have
dropped an ``Op`` to its BARE ``layer_label`` (the last pass of a multi-pass
layer) had it been accepted. Records now lower at hook-plan normalization
to the PASS-QUALIFIED ``Op.label`` of each addressed pass.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.errors import MultiMatchWarning
from torchlens.intervention.hooks import lower_record_site_target
from torchlens.intervention.selectors import CompositeSelector, LabelSelector

_N_PASSES = 3


class _Loop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for step in range(_N_PASSES):
            hidden = torch.relu(self.cell(hidden)) + float(step)
        return hidden


@pytest.fixture(scope="module")
def loop():
    torch.manual_seed(0)
    model = _Loop().eval()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _truth(model, x, zeroed: set[int]) -> dict[str, torch.Tensor]:
    out = {}
    with torch.no_grad():
        hidden = x
        for k in range(1, _N_PASSES + 1):
            relu = torch.relu(model.cell(hidden))
            if k in zeroed:
                relu = torch.zeros_like(relu)
            out[f"relu_1_2:{k}"] = relu
            hidden = relu + float(k - 1)
        out["output_1"] = hidden
    return out


def _assert_matches(fork, truth):
    for k in range(1, _N_PASSES + 1):
        assert torch.allclose(fork[f"relu_1_2:{k}"].out, truth[f"relu_1_2:{k}"], atol=1e-6), k
    assert torch.allclose(fork["output_1"].out, truth["output_1"], atol=1e-6)


def test_do_op_record_edits_exactly_its_pass(loop) -> None:
    model, x, trace = loop
    fork = trace.fork()
    fork.do(trace["relu_1_2:2"].ops[0], tl.zero_ablate())
    _assert_matches(fork, _truth(model, x, {2}))
    assert torch.equal(fork["relu_1_2:1"].out, trace["relu_1_2:1"].out)
    fires = fork["relu_1_2:2"].ops[0].interventions
    assert fires and all(r.call_label == "relu_1_2:2" for r in fires)
    assert not fork["relu_1_2:3"].ops[0].interventions


def test_do_single_pass_getitem_record(loop) -> None:
    model, x, trace = loop
    fork = trace.fork()
    fork.do(fork["relu_1_2:3"], tl.zero_ablate())  # __getitem__ returns the Op for a pass
    _assert_matches(fork, _truth(model, x, {3}))


def test_do_layer_record_is_the_all_passes_spelling(loop) -> None:
    model, x, trace = loop
    fork = trace.fork()
    # The all-passes spelling fans out to every pass; the fan-out is
    # DISCLOSED (MultiMatchWarning), exactly like ``tl.contains("relu_1")``.
    with pytest.warns(MultiMatchWarning):
        fork.do(fork["relu_1_2"], tl.zero_ablate())
    _assert_matches(fork, _truth(model, x, {1, 2, 3}))


def test_do_single_pass_layer_record(loop) -> None:
    _model, _x, trace = loop
    fork = trace.fork()
    fork.do(fork["add_1_3"], tl.zero_ablate())
    assert bool((fork["add_1_3"].out == 0).all())


def test_attach_hooks_op_record_then_push(loop) -> None:
    model, x, trace = loop
    fork = trace.fork()
    fork.attach_hooks(trace["relu_1_2:2"].ops[0], tl.zero_ablate())
    fork.push()
    _assert_matches(fork, _truth(model, x, {2}))


def test_spec_target_is_pass_qualified(loop) -> None:
    _model, _x, trace = loop
    fork = trace.fork()
    fork.do(trace["relu_1_2:2"].ops[0], tl.zero_ablate())
    spec = fork._ensure_intervention_spec()
    values = [
        (spec_hook.site_target.selector_kind, spec_hook.site_target.selector_value)
        for spec_hook in spec.hook_specs
    ]
    assert ("label", "relu_1_2:2") in values
    assert all(value != "relu_1_2" for _kind, value in values)


def test_lower_record_site_target_shapes(loop) -> None:
    _model, _x, trace = loop
    op_lowered = lower_record_site_target(trace["relu_1_2:2"].ops[0])
    assert isinstance(op_lowered, LabelSelector) and op_lowered.selector_value == "relu_1_2:2"
    layer_lowered = lower_record_site_target(trace["relu_1_2"])
    assert isinstance(layer_lowered, CompositeSelector) and layer_lowered.operator == "or"
    assert [s.selector_value for s in layer_lowered.selectors] == [
        "relu_1_2:1",
        "relu_1_2:2",
        "relu_1_2:3",
    ]
    single = lower_record_site_target(trace["add_1_3"])
    assert isinstance(single, LabelSelector) and single.selector_value == "add_1_3:1"
    selector = tl.label("relu_1_2:1")
    assert lower_record_site_target(selector) is selector
    assert lower_record_site_target("relu_1_2:1") == "relu_1_2:1"
