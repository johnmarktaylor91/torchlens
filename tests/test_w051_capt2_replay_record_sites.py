"""``set()``-door and target-spec record lowering (W051-REPLAY out-of-fence 1-2).

Lane W051-REPLAY lowered ``Op``/``Layer`` records at HOOK-PLAN normalization,
so ``do(record, helper)`` worked, but the ``set()`` door (``do(record,
tensor)``), ``resolve_sites(record)`` and the mutator-site validation still
refused a record ("Unsupported site query"), and ``_target_spec_from_site``
would have lowered an ``Op`` to its BARE ``layer_label`` -- the LAST pass of
a multi-pass layer. Records now lower to their PASS-QUALIFIED ``Op.label``
at the post-hoc resolver and at target-spec conversion.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.errors import MultiMatchWarning
from torchlens.intervention.selectors import CompositeSelector, LabelSelector
from torchlens.ir.selector_eval import normalize_selector_like

pytestmark = pytest.mark.smoke

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


def test_set_door_op_record_edits_exactly_its_pass(loop) -> None:
    """``do(op_record, tensor)`` (the ``set()`` door) addresses ONE pass."""

    model, x, trace = loop
    fork = trace.fork()
    fork.do(trace["relu_1_2:2"].ops[0], torch.zeros(2, 4))
    _assert_matches(fork, _truth(model, x, {2}))
    assert torch.equal(fork["relu_1_2:1"].out, trace["relu_1_2:1"].out)
    assert torch.allclose(fork["relu_1_2:3"].out, _truth(model, x, {2})["relu_1_2:3"], atol=1e-6)


def test_set_door_layer_record_is_the_all_passes_spelling(loop) -> None:
    model, x, trace = loop
    fork = trace.fork()
    with pytest.warns(MultiMatchWarning):
        fork.do(fork["relu_1_2"], torch.zeros(2, 4))
    _assert_matches(fork, _truth(model, x, {1, 2, 3}))


def test_set_door_stored_target_is_pass_qualified(loop) -> None:
    """The recipe stores ``relu_1_2:2``, never the bare (last-pass) layer label."""

    _model, _x, trace = loop
    fork = trace.fork()
    fork.do(trace["relu_1_2:2"].ops[0], torch.zeros(2, 4))
    spec = fork._ensure_intervention_spec()
    targets = [
        (s.site_target.selector_kind, s.site_target.selector_value) for s in spec.target_value_specs
    ]
    assert ("label", "relu_1_2:2") in targets
    assert all(value != "relu_1_2" for _kind, value in targets)


def test_target_spec_from_op_record_is_its_own_pass(loop) -> None:
    _model, _x, trace = loop
    op_spec = trace._target_spec_from_site(trace["relu_1_2:2"].ops[0], strict=False)
    assert (op_spec.selector_kind, op_spec.selector_value) == ("label", "relu_1_2:2")
    layer_spec = trace._target_spec_from_site(trace["relu_1_2"], strict=False)
    assert layer_spec.selector_kind == "or"
    single = trace._target_spec_from_site(trace["add_1_3"], strict=False)
    assert (single.selector_kind, single.selector_value) == ("label", "add_1_3:1")
    passthrough = trace._target_spec_from_site("relu_1_2:1", strict=True)
    assert (passthrough.selector_kind, passthrough.selector_value) == ("label", "relu_1_2:1")
    assert passthrough.strict is True


def test_resolve_sites_accepts_records_pass_qualified(loop) -> None:
    _model, _x, trace = loop
    resolved = trace.resolve_sites(trace["relu_1_2:2"].ops[0])
    labels = {getattr(site, "label", None) or str(site) for site in resolved}
    assert any("relu_1_2:2" in label for label in labels), labels
    assert not any("relu_1_2:1" in label or "relu_1_2:3" in label for label in labels), labels


def test_normalize_selector_like_lowers_records_in_both_lifecycles(loop) -> None:
    _model, _x, trace = loop
    op = trace["relu_1_2:2"].ops[0]
    for lifecycle in ("live", "site"):
        lowered = normalize_selector_like(op, lifecycle=lifecycle)
        assert isinstance(lowered, LabelSelector) and lowered.selector_value == "relu_1_2:2"
    layer = normalize_selector_like(trace["relu_1_2"], lifecycle="site")
    assert isinstance(layer, CompositeSelector) and layer.operator == "or"
    assert [s.selector_value for s in layer.selectors] == ["relu_1_2:1", "relu_1_2:2", "relu_1_2:3"]
