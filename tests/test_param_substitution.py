"""Parameter substitution ("as if" the parameter were changed) — replay-only.

Pins the decided param-operand contract (2026-08-17): ``fork.do(
tl.params(...), edit)`` substitutes the VALUE each consuming op sees at its
derived occurrence address on the replay engine, through the shipped
tier-(ii) edge-substitution store, and the live parameter object is never
written (bit-identical proof below). Engine scope (rerun/set_only refuse
``param_substitution_engine_unsupported``), fail-closed occurrence
derivation (``param_substitution_occurrence_underivable``), edit-then-
scatter masking, validation boundary corroboration (DIFFERENT check, never
NO check), and re-splice persistence under later pushes are all pinned.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.selection import SelectionError
from torchlens.validation.core import _check_edge_intervention_boundary


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _capture():
    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(2, 4)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    return model, x, trace


@pytest.fixture(scope="module")
def capture():
    model, x, trace = _capture()
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _as_if_forward(model: _Net, x: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.fc2(torch.relu(x @ w1.T + model.fc1.bias))


def test_basic_as_if_substitution(capture):
    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.0))
    expected = _as_if_forward(model, x, torch.zeros_like(model.fc1.weight))
    assert torch.allclose(fork.output_ops[0].out, expected, atol=1e-6)
    # the source trace is untouched (fork is the sanctioned mutation surface)
    with torch.no_grad():
        assert torch.allclose(trace.output_ops[0].out, model(x))


def test_live_parameter_bit_identical(capture):
    """THE ruling's core guarantee: the parameter object is never written."""

    model, x, trace = capture
    weight = model.fc1.weight
    before_bytes = weight.detach().clone()
    before_ptr = weight.data_ptr()
    before_version = weight._version
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(3.0))
    assert weight.data_ptr() == before_ptr
    assert weight._version == before_version
    assert torch.equal(weight.detach(), before_bytes)
    assert weight.detach().view(torch.uint8).equal(before_bytes.view(torch.uint8)), (
        "parameter storage must be BIT-identical after an intervened replay"
    )


@pytest.mark.smoke
def test_masked_param_edit_scatters_only_selected_rows(capture):
    model, x, trace = capture
    mask = torch.zeros_like(model.fc1.weight, dtype=torch.bool)
    mask[0] = True
    fork = trace.fork()
    fork.do(tl.params("fc1.weight", mask=mask), tl.scale(0.0))
    as_if = model.fc1.weight.detach().clone()
    as_if[0] = 0.0
    expected = _as_if_forward(model, x, as_if)
    assert torch.allclose(fork.output_ops[0].out, expected, atol=1e-6)


def test_helpers_and_raw_tensor_compose(capture):
    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.add(0.25))
    expected = _as_if_forward(model, x, model.fc1.weight.detach() + 0.25)
    assert torch.allclose(fork.output_ops[0].out, expected, atol=1e-6)
    replacement = torch.randn_like(model.fc1.weight)
    fork2 = trace.fork()
    fork2.do(tl.params("fc1.weight"), replacement)
    expected2 = _as_if_forward(model, x, replacement)
    assert torch.allclose(fork2.output_ops[0].out, expected2, atol=1e-6)


def test_engine_scoping_refusals(capture):
    model, x, trace = capture
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(
            tl.params("fc1.weight"),
            tl.scale(0.0),
            intervention=tl.options.InterventionOptions(engine="set_only"),
        )
    assert excinfo.value.fields["code"] == "param_substitution_engine_unsupported"
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(
            tl.params("fc1.weight"),
            tl.scale(0.0),
            model=model,
            x=x,
            intervention=tl.options.InterventionOptions(engine="rerun"),
        )
    assert excinfo.value.fields["code"] == "param_substitution_engine_unsupported"
    # auto with model+x resolves to rerun -> same refusal
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(tl.params("fc1.weight"), tl.scale(0.0), model=model, x=x)
    assert excinfo.value.fields["code"] == "param_substitution_engine_unsupported"


class _RecurrentNet(nn.Module):
    """Hand-rolled recurrence reusing ONE nn.Linear (bias=True is deliberate:
    with bias=False, relu(cell(0)) == 0 is a fixed point and a wrong replay
    coincidentally matches ground truth — the confound that masked the
    original pass-blind bug)."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(3):
            h = torch.relu(self.cell(h))
        return h


def _recurrent_capture():
    torch.manual_seed(0)
    model = _RecurrentNet()
    x = torch.randn(2, 4)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    return model, x, trace


@pytest.fixture(scope="module")
def recurrent_capture():
    model, x, trace = _recurrent_capture()
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _recurrent_ground_truth(x, weight, bias):
    """Hand-compute every pass from the model's own weights (never the engine)."""

    per_pass = []
    with torch.no_grad():
        h = x
        for _ in range(3):
            lin = h @ weight.T + bias
            h = torch.relu(lin)
            per_pass.append((lin, h))
    return per_pass


def test_recurrent_reused_param_substitutes_every_pass(recurrent_capture):
    """Tied/recurrent params substitute at EVERY consumption, per-pass exact."""

    model, x, trace = recurrent_capture
    fork = trace.fork()
    fork.do(tl.params("cell.weight"), tl.add(0.25))
    expected = _recurrent_ground_truth(
        x, model.cell.weight.detach() + 0.25, model.cell.bias.detach()
    )
    for k, (lin, relu) in enumerate(expected, start=1):
        lin_op = fork.layer_dict_all_keys[f"linear_1_1:{k}"]
        relu_op = fork.layer_dict_all_keys[f"relu_1_2:{k}"]
        assert torch.allclose(lin_op.out, lin, atol=1e-6), f"pass {k} linear out is stale/wrong"
        assert torch.allclose(relu_op.out, relu, atol=1e-6), f"pass {k} relu out is stale/wrong"
        fires = [r for r in lin_op.interventions if r.edge_address is not None]
        assert len(fires) == 1, f"pass {k} must carry its OWN occurrence FireRecord"
        payload = lin_op.edge_substitutions[next(iter(lin_op.edge_substitutions))]
        assert payload["substitution_kind"] == "param"
    assert torch.allclose(fork.output_ops[0].out, expected[-1][1], atol=1e-6)
    audit = fork.intervention_audit[-1]["params"][0]
    assert audit["consumers"] == ["linear_1_1:1", "linear_1_1:2", "linear_1_1:3"]
    assert len(audit["occurrences"]) == 3


@pytest.mark.smoke
def test_recurrent_bias_param_substitutes_every_pass(recurrent_capture):
    model, x, trace = recurrent_capture
    replacement = torch.randn(4)
    fork = trace.fork()
    fork.do(tl.params("cell.bias"), replacement)
    expected = _recurrent_ground_truth(x, model.cell.weight.detach(), replacement)
    for k, (lin, relu) in enumerate(expected, start=1):
        assert torch.allclose(fork.layer_dict_all_keys[f"linear_1_1:{k}"].out, lin, atol=1e-6)
        assert torch.allclose(fork.layer_dict_all_keys[f"relu_1_2:{k}"].out, relu, atol=1e-6)


def test_tied_multipass_matmul_substitutes_every_pass():
    """The historically refused case: one param at structurally corresponding
    sites (recurrence-grouped matmuls) now substitutes at both passes."""

    torch.manual_seed(0)

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 4) * 0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x @ self.w) @ self.w

    model = Tied()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    try:
        param = next(p for p in trace.params if p.address == "w")
        assert list(param.used_by_ops) == ["matmul_1_1:1", "matmul_1_1:2"]
        fork = trace.fork()
        fork.do(tl.params("w"), tl.scale(2.0))
        w2 = model.w.detach() * 2.0
        with torch.no_grad():
            pass1 = x @ w2
            pass2 = torch.relu(pass1) @ w2
        assert torch.allclose(fork.layer_dict_all_keys["matmul_1_1:1"].out, pass1, atol=1e-6)
        assert torch.allclose(fork.layer_dict_all_keys["matmul_1_1:2"].out, pass2, atol=1e-6)
        assert torch.allclose(fork.output_ops[0].out, pass2, atol=1e-6)
    finally:
        trace.cleanup()


def test_tied_embedding_projection_substitutes_both_sites():
    """Classic weight tying: one param consumed by embedding AND projection."""

    import torch.nn.functional as F

    torch.manual_seed(0)

    class TiedLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(10, 4)
            self.fc = nn.Linear(4, 4, bias=True)

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.fc(self.emb(idx)))
            return F.linear(h, self.emb.weight)

    model = TiedLM()
    idx = torch.tensor([[1, 3, 7]])
    trace = tl.trace(model, idx, capture=tl.options.CaptureOptions(intervention_ready=True))
    try:
        replacement = torch.randn(10, 4)
        fork = trace.fork()
        fork.do(tl.params("emb.weight"), replacement)
        with torch.no_grad():
            h = torch.relu(model.fc(F.embedding(idx, replacement)))
            expected = F.linear(h, replacement)
        assert torch.allclose(fork.output_ops[0].out, expected, atol=1e-6)
        consumers = fork.intervention_audit[-1]["params"][0]["consumers"]
        assert len(consumers) == 2, "both tied consumption sites must be addressed"
    finally:
        trace.cleanup()


def test_multipass_live_parameter_bit_identical(recurrent_capture):
    """The ruling's core guarantee holds on the newly admitted multi-pass path."""

    model, x, trace = recurrent_capture
    weight = model.cell.weight
    before_bytes = weight.detach().clone()
    before_ptr = weight.data_ptr()
    before_version = weight._version
    fork = trace.fork()
    fork.do(tl.params("cell.weight"), tl.scale(3.0))
    assert weight.data_ptr() == before_ptr
    assert weight._version == before_version
    assert torch.equal(weight.detach(), before_bytes)
    assert weight.detach().view(torch.uint8).equal(before_bytes.view(torch.uint8)), (
        "parameter storage must be BIT-identical after a multi-pass intervened replay"
    )


def test_multipass_validation_corroborates_every_pass(recurrent_capture):
    model, x, trace = recurrent_capture
    fork = trace.fork()
    fork.do(tl.params("cell.weight"), tl.add(0.25))
    for k in (1, 2, 3):
        verdict = _check_edge_intervention_boundary(
            fork, fork.layer_dict_all_keys[f"linear_1_1:{k}"]
        )
        assert verdict is not None
        assert verdict.decision == "edge_intervention_boundary", f"pass {k} not corroborated"
        assert not verdict.failed


def test_multipass_validation_tamper_strip_fire_record_fails(recurrent_capture):
    """Tripwire intact on the newly admitted path: an uncorroborated entry on a
    NON-LAST pass (the pass the historical bug would have missed) FAILS."""

    model, x, trace = recurrent_capture
    fork = trace.fork()
    fork.do(tl.params("cell.weight"), tl.scale(0.5))
    tampered = fork.layer_dict_all_keys["linear_1_1:1"]
    tampered._internal_set(
        "interventions", [r for r in tampered.interventions if not r.edge_address]
    )
    verdict = _check_edge_intervention_boundary(fork, tampered)
    assert verdict is not None and verdict.failed
    assert verdict.reason == "edge_substitution_uncorroborated"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fork.validate_forward_pass(model(x)) is False


def test_multipass_bare_consumer_spelling_refuses():
    """A bare multi-pass consumer spelling addresses only the LAST pass —
    staging from it would silently substitute a subset, so it refuses."""

    model, x, trace = _recurrent_capture()
    try:
        param = next(p for p in trace.params if p.address == "cell.weight")
        param.used_by_ops = ["linear_1_1"]
        with pytest.raises(SelectionError) as excinfo:
            trace.fork().do(tl.params("cell.weight"), tl.scale(2.0))
        assert excinfo.value.fields["code"] == "param_substitution_occurrence_underivable"
    finally:
        trace.cleanup()


def test_multipass_incomplete_pass_coverage_refuses():
    """A consumer inventory omitting a pass refuses (fail-closed: a partial
    parameter substitution is a wrong replay that looks healthy)."""

    model, x, trace = _recurrent_capture()
    try:
        param = next(p for p in trace.params if p.address == "cell.weight")
        param.used_by_ops = ["linear_1_1:1", "linear_1_1:3"]
        with pytest.raises(SelectionError) as excinfo:
            trace.fork().do(tl.params("cell.weight"), tl.scale(2.0))
        assert excinfo.value.fields["code"] == "param_substitution_occurrence_underivable"
        assert "pass(es) [2]" in str(excinfo.value)
    finally:
        trace.cleanup()


@pytest.mark.smoke
def test_zero_mask_is_disclosure_not_error(capture):
    model, x, trace = capture
    fork = trace.fork()
    before = fork.output_ops[0].out.clone()
    fork.do(
        tl.params("fc1.weight", mask=torch.zeros_like(model.fc1.weight, dtype=torch.bool)),
        tl.scale(0.0),
    )
    assert torch.equal(fork.output_ops[0].out, before)
    assert fork.intervention_audit[-1]["kind"] == "PARAM"
    assert fork.intervention_audit[-1]["params"] == []


def test_audit_record_discloses_substitution_not_change(capture):
    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.5))
    audit = fork.intervention_audit[-1]
    assert audit["kind"] == "PARAM"
    assert "substituted at consumption" in audit["disclosure"]
    assert "live parameters unchanged" in audit["disclosure"]
    (param_row,) = audit["params"]
    assert param_row["param_address"] == "fc1.weight"
    (occurrence,) = param_row["occurrences"]
    assert occurrence["consumer"] == param_row["consumers"][0]
    assert occurrence["value_digest"]


def test_fire_record_and_store_mark_param_kind(capture):
    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.5))
    child = fork["fc1"].ops[0]
    (store_key,) = child.edge_substitutions.keys()
    payload = child.edge_substitutions[store_key]
    assert payload["substitution_kind"] == "param"
    assert payload["param_address"] == "fc1.weight"
    stamp = child.edge_replacement_stamps[store_key]
    assert stamp["substitution_kind"] == "param"
    fire = next(r for r in child.interventions if r.edge_address is not None)
    assert fire.replaced is False  # substitution replaces no node's OUTPUT
    assert fire.engine == "replay"


def test_validation_boundary_corroborates(capture):
    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.5))
    child = fork["fc1"].ops[0]
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None
    assert verdict.decision == "edge_intervention_boundary"  # DISTINCT term, never "exempted"
    assert not verdict.failed


@pytest.mark.smoke
def test_validation_tamper_strip_fire_record_fails(capture):
    """Tripwire intact: an uncorroborated param entry FAILS validation."""

    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.5))
    child = fork["fc1"].ops[0]
    child._internal_set("interventions", [r for r in child.interventions if not r.edge_address])
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None and verdict.failed
    assert verdict.reason == "edge_substitution_uncorroborated"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fork.validate_forward_pass(model(x)) is False


def test_substitution_survives_later_push(capture):
    """Param-kind entries re-splice during cone recomputation (never revert)."""

    model, x, trace = capture
    fork = trace.fork()
    fork.do(tl.params("fc1.weight"), tl.scale(0.0))
    after_do = fork.output_ops[0].out.clone()

    def _noop(out: torch.Tensor, *, hook) -> torch.Tensor:
        return out

    fork.do(tl.func("relu"), _noop)
    assert torch.allclose(fork.output_ops[0].out, after_do)


def test_foreign_resolved_selection_refuses(capture):
    model, x, trace = capture
    other_model, other_x, other = _capture()
    try:
        resolved = tl.params("fc1.weight").resolve(other)
        with pytest.raises(SelectionError) as excinfo:
            trace.fork().do(resolved, tl.scale(0.0))
        assert excinfo.value.fields["code"] == "selection_trace_mismatch"
    finally:
        other.cleanup()
