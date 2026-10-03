"""Edge substitution into a NON-LAST pass of a multi-pass child (AUD-CODE 1.2).

Before the fix the edge door committed ``{child.layer_label: new_out}`` and
``{child.layer_label: [fire_record]}`` -- the BARE label, which
``layer_dict_all_keys`` resolves to the LAST pass -- then ``push_from``
pushed the addressed pass's UNCHANGED captured out downstream: a silent
no-op whose FireRecord landed on the wrong pass while the audit row claimed
success (and the fork was un-validatable, ``edge_substitution_uncorroborated``).

Test-design rules (do not weaken): ground truth is computed BY HAND from the
model's own weights, fidelity assertions are PER PASS, and the loop cell uses
``bias=True`` plus a per-pass ``+ i`` offset so zero is not a fixed point.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.edge_substitution import _consumed_value, tensor_content_digest

_N_PASSES = 3


class _Loop(nn.Module):
    """One reused Linear(bias=True) applied 3 times with a per-pass offset."""

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
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _hand_truth(model: _Loop, x: torch.Tensor, edits: dict[int, str]) -> dict[str, torch.Tensor]:
    """Per-pass relu outputs and the final output with ``edits`` = {pass: 'zero'}."""

    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        hidden = x
        for pass_index in range(1, _N_PASSES + 1):
            pre = model.cell(hidden)
            if edits.get(pass_index) == "zero":
                pre = torch.zeros_like(pre)
            relu = torch.relu(pre)
            out[f"relu_1_2:{pass_index}"] = relu
            hidden = relu + float(pass_index - 1)
        out["output_1"] = hidden
    return out


def _edge_into(trace: tl.Trace, pass_index: int):
    return next(e for e in trace.edges if e.child_label == f"relu_1_2:{pass_index}")


@pytest.mark.parametrize("pass_index", [1, 2])
def test_edge_into_non_last_pass_edits_exactly_that_pass(loop, pass_index: int) -> None:
    model, x, trace = loop
    fork = trace.fork()
    fork.do(_edge_into(trace, pass_index).__selection__(), tl.zero_ablate())
    truth = _hand_truth(model, x, {pass_index: "zero"})

    for k in range(1, _N_PASSES + 1):
        label = f"relu_1_2:{k}"
        assert torch.allclose(fork[label].out, truth[label], atol=1e-6), label
        if k < pass_index:
            assert torch.equal(fork[label].out, trace[label].out), f"{label} must be untouched"
    assert torch.allclose(fork["output_1"].out, truth["output_1"], atol=1e-6)
    assert not torch.equal(fork["output_1"].out, trace["output_1"].out)


def test_fire_record_and_store_land_on_the_addressed_pass(loop) -> None:
    _model, _x, trace = loop
    fork = trace.fork()
    edge = _edge_into(trace, 1)
    fork.do(edge.__selection__(), tl.zero_ablate())
    address = (edge.child_func_call_id, edge.arg_kind, edge.arg_path)

    hit = fork["relu_1_2:1"].ops[0]
    assert list(hit.edge_substitutions) == [(edge.arg_kind, edge.arg_path)]
    assert hit.edge_replacement_stamps[(edge.arg_kind, edge.arg_path)]["verdict"] is True
    fires = [r for r in hit.interventions if r.edge_address is not None]
    assert [r.edge_address for r in fires] == [address]
    assert fires[0].call_label == "relu_1_2:1"
    for other in (2, 3):
        op = fork[f"relu_1_2:{other}"].ops[0]
        assert not (op.edge_substitutions or {})
        assert not [r for r in op.interventions if r.edge_address is not None]


def test_audit_row_describes_the_real_effect(loop) -> None:
    _model, _x, trace = loop
    fork = trace.fork()
    edge = _edge_into(trace, 1)
    fork.do(edge.__selection__(), tl.zero_ablate())
    audit = fork.intervention_audit[-1]
    assert audit["kind"] == "EDGE"
    row = audit["edges"][0]
    assert row["child"] == "relu_1_2:1" and row["parent"] == "linear_1_1:1"
    store_value = fork["relu_1_2:1"].ops[0].edge_substitutions[(edge.arg_kind, edge.arg_path)]
    assert row["value_digest"] == tensor_content_digest(store_value["value"])
    assert bool((store_value["value"] == 0).all())
    assert fork.last_run["origins"] == ("relu_1_2:1",)


def test_multipass_edge_fork_validates_and_reaches_the_boundary_check(loop) -> None:
    model, x, trace = loop
    fork = trace.fork()
    fork.do(_edge_into(trace, 1).__selection__(), tl.zero_ablate())
    truth = _hand_truth(model, x, {1: "zero"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fork.validate_forward_pass(truth["output_1"]) is True
    boundary = [
        d
        for d in fork.validation_replay_status.decisions
        if d.get("decision") == "edge_intervention_boundary" and d.get("op_label") == "relu_1_2:1"
    ]
    assert boundary, "the tier-(ii) boundary check must be REACHED in a passing validation"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Wrong claim (the ORIGINAL output) must not validate.
        assert fork.validate_forward_pass(model(x)) is False


def test_consumed_value_prefers_the_pass_qualified_version() -> None:
    pass_value = torch.ones(2)
    last_pass_value = torch.full((2,), 7.0)
    parent = SimpleNamespace(
        out=torch.zeros(2),
        label="linear:1",
        out_versions_by_child={"relu:1": pass_value, "relu": last_pass_value},
    )
    child = SimpleNamespace(label="relu:1", layer_label="relu")
    assert _consumed_value(None, parent, child) is pass_value
    single = SimpleNamespace(label="relu:1", layer_label="relu")
    parent_single = SimpleNamespace(
        out=torch.zeros(2), label="linear:1", out_versions_by_child={"relu": last_pass_value}
    )
    assert _consumed_value(None, parent_single, single) is last_pass_value


def test_tensor_content_digest_is_dtype_agnostic_and_byte_exact() -> None:
    value = torch.randn(3, 2)
    assert tensor_content_digest(value) == tensor_content_digest(value.clone())
    assert tensor_content_digest(value) != tensor_content_digest(value + 1e-7)
    half = value.to(torch.bfloat16)
    assert tensor_content_digest(half) == tensor_content_digest(half.clone())
    assert tensor_content_digest(torch.tensor(3.0)) == tensor_content_digest(torch.tensor(3.0))
    assert tensor_content_digest(torch.empty(0)) == tensor_content_digest(torch.empty(0))
