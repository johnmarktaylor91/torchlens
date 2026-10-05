"""R08 discrete-bool exemption hardening tests (validation tripwire).

Moved verbatim from ``tests/test_validation_hardening.py`` (at its size cap).
Pre-fix, a bool-dtype output was a blanket posthoc ``discrete_bool_output``
pass, so a dead or invented parent edge on a comparison or bool-predicate op
was unfalsifiable by perturbation. Each armed-proof here freezes a real
captured op and asserts the tripwire now fires; each honest control asserts a
real edge validates or exempts WITH probe evidence. R08-3 pins that a bare
ground-truth tensor is not iterated as rows.
"""

import pytest
import torch
import torch.nn as nn
from _validation_capture import _capture, _quiet_validate

# ---------------------------------------------------------------------------
# R08 (b1-opus round-5): the discrete-bool blanket posthoc exemption
# ---------------------------------------------------------------------------


class _FarThresholdGate(nn.Module):
    """``(x > 1e30).float()`` -- no feasible perturbation crosses the threshold."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 1.0e30).float()


def test_bool_comparison_spurious_edge_now_fails() -> None:
    """Armed-proof (R08): a dead edge on a COMPARISON op is no longer blessed.

    Freeze the ``gt`` op's replay callable to return its saved bool output
    regardless of inputs -- the recorded parent provably does not influence
    the output. Pre-fix, ``dtype == torch.bool`` was a blanket posthoc pass,
    so this exact spurious-edge class was unfalsifiable by perturbation
    (red-capable: pre-fix this returns ``exempted``/``discrete_bool_output``).
    The threshold-straddle probe now proves no value influence and the
    tripwire fires.
    """

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    trace, _ground_truth = _capture(_FarThresholdGate(), torch.randn(3, 4))
    gt_op = [op for op in trace.layer_list if op.func_name == "__gt__"][0]
    saved = gt_op.out.detach().clone()
    object.__setattr__(gt_op, "func", lambda *args, **kwargs: saved.clone())
    parent_label = gt_op.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, gt_op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "failed"
    assert result.reason == "perturbation_insensitive"


def test_far_threshold_comparison_keeps_evidence_backed_exemption() -> None:
    """A REAL comparison edge no feasible perturbation can flip stays exempt,
    and the exemption now carries straddle-probe evidence instead of being a
    blanket dtype pass (red-capable: pre-fix the justification was None)."""

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    trace, _ground_truth = _capture(_FarThresholdGate(), torch.randn(3, 4))
    gt_op = [op for op in trace.layer_list if op.func_name == "__gt__"][0]
    parent_label = gt_op.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, gt_op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "exempted"
    assert result.reason == "discrete_bool_output"
    assert result.justification and "straddle" in result.justification


def test_isnan_edge_validates_via_nan_probe() -> None:
    """The NaN retry rung upgrades a real ``isnan`` edge from exempt to
    VALIDATED (red-capable: pre-fix the finite draws never flipped the output
    and the blanket bool exemption absorbed it)."""

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    class _NanGate(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.isnan(x).float()

    trace, _ground_truth = _capture(_NanGate(), torch.randn(3, 4))
    isnan_op = [op for op in trace.layer_list if op.func_name == "isnan"][0]
    parent_label = isnan_op.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, isnan_op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "validated"
    assert result.reason == "perturbation_changed"


def test_bool_output_models_still_validate_true_end_to_end() -> None:
    """No false-fail regression: correct captures with bool ops validate True."""

    class _BoolMix(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate = (x > 0.5).float()
            nan_gate = torch.isnan(x).float()
            eq_gate = torch.eq(x, x.detach().clone() + 3.0).float()
            return gate + nan_gate + eq_gate

    torch.manual_seed(0)
    assert _quiet_validate(_BoolMix(), torch.randn(3, 4)) is True
    torch.manual_seed(0)
    assert _quiet_validate(_FarThresholdGate(), torch.randn(3, 4)) is True


class _PredicateZoo(nn.Module):
    """Bool-output NON-comparison ops: the R08-2 residual family.

    ``isnan``/``isfinite``/``logical_and``/``any``/``bitwise_not`` each get a
    perturbable float (or derived-bool) parent so the spurious-edge freeze
    harness and the honest controls run on real captured edges.
    """

    def __init__(self, func_name: str) -> None:
        super().__init__()
        self.func_name = func_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        if self.func_name == "isnan":
            return torch.isnan(h).float()
        if self.func_name == "isfinite":
            return torch.isfinite(h).float()
        if self.func_name == "logical_and":
            return torch.logical_and(h > 0, h < 1.0e30).float()
        if self.func_name == "any":
            return torch.any(h > 0).float()
        if self.func_name == "bitwise_not":
            return torch.bitwise_not(h > 0).float()
        raise AssertionError(self.func_name)


@pytest.mark.parametrize("func_name", ["isnan", "isfinite", "logical_and", "any", "bitwise_not"])
def test_bool_predicate_spurious_edge_now_fails(func_name: str) -> None:
    """Armed-proof (r7 R08-2): a dead edge on a NON-comparison bool op fails.

    Round-6 residual of the R08 comparison fix: freezing the replay callable
    of ``logical_and``/``isnan``/``isfinite``/``any``/``bitwise_not`` settled
    ``exempted``/``discrete_bool_output`` with an empty justification -- the
    ``probe is None`` fallback was still a blanket pass, so a capture bug
    that drops or invents a parent edge on the predicate family was
    unfalsifiable by perturbation. The substitution-battery probe now proves
    no value influence and the tripwire fires (red-capable: pre-fix every
    parametrization settles ``exempted``).
    """

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    trace, _ground_truth = _capture(_PredicateZoo(func_name), torch.randn(3, 4))
    target = [op for op in trace.layer_list if op.func_name == func_name][0]
    saved = target.out.detach().clone()
    object.__setattr__(target, "func", lambda *args, **kwargs: saved.clone())
    parent_label = target.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, target.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "failed", (func_name, result.decision, result.reason)
    assert result.reason == "perturbation_insensitive"


@pytest.mark.parametrize("func_name", ["isnan", "isfinite", "logical_and", "any", "bitwise_not"])
def test_bool_predicate_honest_edge_stays_green(func_name: str) -> None:
    """Honest control: the real edge validates (rungs) or exempts WITH
    battery evidence -- never the evidence-free blanket, never a failure."""

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    trace, _ground_truth = _capture(_PredicateZoo(func_name), torch.randn(3, 4))
    target = [op for op in trace.layer_list if op.func_name == func_name][0]
    parent_label = target.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, target.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision in ("validated", "exempted"), (
        func_name,
        result.decision,
        result.reason,
    )
    if result.decision == "exempted" and result.reason == "discrete_bool_output":
        assert result.justification, (
            f"{func_name}: honest edge exempted through the evidence-free blanket"
        )


def test_validate_forward_pass_accepts_a_bare_ground_truth_tensor() -> None:
    """r7 b1-opus R08-3: ``validate_forward_pass(model(x))`` must not FAIL.

    A bare tensor was iterated as rows, so the arity check counted the BATCH
    as expected outputs and a byte-correct capture reported
    ``ground_truth_missing_out`` ("1 logged vs N expected") -- a validation
    FAILURE for a caller-arity slip. The bare tensor now normalizes to
    ``[tensor]`` (red-capable: pre-fix this asserts False).
    """

    model = _FarThresholdGate()
    x = torch.randn(4, 3)
    trace, ground_truth = _capture(model, x)
    assert trace.validate_forward_pass(ground_truth) is True
    assert trace.validate_forward_pass([ground_truth]) is True


class _InplaceMaskGate(nn.Module):
    """In-place bool op whose probe used to corrupt the retained operand."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x > 0
        gate = x < 1.0e30
        mask.logical_and_(gate)
        return mask.float()


def test_bool_predicate_probe_never_mutates_retained_saved_args() -> None:
    """r8 R08 (fable MH): the battery must execute against CLONES.

    The bool universe includes in-place ops (``logical_and_``); the probe
    executed the captured func against the record's retained ``saved_args``
    with only the probed slot substituted, so one probe run MUTATED the
    capture evidence in place and returned a verdict computed on the
    corrupted operand (red-capable: pre-fix the byte comparison fails).
    """

    from torchlens.validation.exemptions import _bool_predicate_influence_probe

    trace, _ground_truth = _capture(_InplaceMaskGate(), torch.randn(3, 4))
    target = [op for op in trace.layer_list if op.func_name == "logical_and_"][0]
    saved_args = list(target.saved_args or ())
    assert saved_args and isinstance(saved_args[0], torch.Tensor)
    snapshots = [
        arg.detach().clone() if isinstance(arg, torch.Tensor) else arg for arg in saved_args
    ]
    arg_positions = (getattr(target, "parent_arg_positions", None) or {}).get("args", {})
    assert arg_positions, "expected positional parent metadata on the in-place op"
    # Probe through each single-parent slot the metadata knows about.
    for position, parent_label in arg_positions.items():
        _bool_predicate_influence_probe(target, [parent_label])
        _ = position
    for index, snapshot in enumerate(snapshots):
        if isinstance(snapshot, torch.Tensor):
            assert torch.equal(saved_args[index], snapshot), (
                f"probe mutated retained saved_args[{index}] in place"
            )


def test_bool_predicate_probe_catch_is_typed() -> None:
    """r8 R08 (sol fault-injection): unexpected probe crashes PROPAGATE.

    The old blanket ``except Exception`` swallowed genuine capture-bug
    crashes into the evidence-free heuristic pass; benign substitute
    refusals (TypeError/ValueError/RuntimeError) still settle "cannot run".
    """

    from torchlens.validation.exemptions import _bool_predicate_influence_probe

    trace, _ground_truth = _capture(_PredicateZoo("isnan"), torch.randn(3, 4))
    target = [op for op in trace.layer_list if op.func_name == "isnan"][0]
    parent_label = target.parents[0]

    object.__setattr__(
        target, "func", lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("capture bug"))
    )
    with pytest.raises(KeyError, match="capture bug"):
        _bool_predicate_influence_probe(target, [parent_label])

    def _benign_refusal(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - test shim
        raise RuntimeError("substitute rejected")

    object.__setattr__(target, "func", _benign_refusal)
    assert _bool_predicate_influence_probe(target, [parent_label]) is None
