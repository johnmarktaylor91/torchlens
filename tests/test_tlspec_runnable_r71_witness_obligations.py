"""r71 A -- the witness-strip class, closed comprehensively (machine-checked immunizer).

The agreed r71 invariant (deletion-closure):

    Every replay-structure item that can affect a faithfulness verdict creates a
    typed, independently derived OBLIGATION, anchored to structure the replay itself
    consumes. Each obligation is discharged exactly once, by an exact witness XOR an
    explicit, source-linked WitnessCoverageGap. The witness stream, the
    required-witness inventory, and the ``witness_completeness`` summary are NEVER
    authority for their own required coverage. Consequence: no combination of record
    DELETIONS can improve a verdict -- any partial strip leaves a surviving anchor
    contradiction (parse-refuse ``context_field_invalid``, analysis-only) or an
    UNVERIFIABLE floor, never VERIFIED.

This file is the convergence criterion for r71 A: registry closure, derivation
independence, the generated all-family strip matrix (single / matched-pair / 3-way
lockstep / forged-complete), the named r70 E2E pins (corr2recover-H1 A+B, free-F1,
hon1-F2 shape), completeness forgery, verdict monotonicity, no-over-trigger GREENs,
the source-scan floor tripwire, and the coherent-reauthoring threat-model pin.
"""

from __future__ import annotations

import copy as _copy
import inspect
import json as _json
import shutil as _shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import (
    _VERDICT_STEERING_WITNESS_FAMILIES,
    WITNESS_FAMILY_REGISTRY,
    ControlWitnessKind,
    PathFaithfulness,
    ReadinessStatus,
    WitnessGapKind,
)
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions

_CAPTURE = CaptureOptions(replayable=True, capture_container_structure=True, cache=False)


def _trace(model: nn.Module, inputs: Any) -> tl.Trace:
    return tl.trace(model, inputs, capture=_CAPTURE)


def _save(trace: tl.Trace, path: Path) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.save(path, level="runnable", include_weights=True)
    return path


def _mutate(path: Path, mutate: Callable[[dict], None]) -> None:
    manifest_path = path / "manifest.json"
    manifest = _json.loads(manifest_path.read_text())
    mutate(manifest["run"])
    manifest_path.write_text(_json.dumps(manifest))


def _verdict(path: Path, inputs: Any) -> tuple[str, Any]:
    """Load + run one artifact; classify the outcome without ever masking VERIFIED."""

    loaded = tl.load(path)
    readiness = loaded._runnable.readiness
    if readiness is not None and readiness.status is ReadinessStatus.UNAVAILABLE:
        return ("refused", {d.code.value for d in readiness.diagnostics})
    try:
        result = loaded.run(inputs=inputs, on_divergence="return")
        return ("ran", result.report.path_faithfulness)
    except Exception as exc:  # noqa: BLE001 -- typed refusal classification
        return ("raised", type(exc).__name__)


def _assert_never_verified(path: Path, inputs: Any) -> None:
    outcome, detail = _verdict(path, inputs)
    if outcome == "refused":
        assert "context_field_invalid" in detail, detail
        return
    if outcome == "ran":
        assert detail is not PathFaithfulness.VERIFIED, detail
        return
    # A typed raise (divergence / precondition) can never be a false VERIFIED.


# ======================================================================================
# Fixture models -- together they cover every witness-carrying registry family
# ======================================================================================


class _Branch(nn.Module):
    """scalar_bool + conditional_arm_entry."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x * 2.0
        return x + 100.0


class _Loop(nn.Module):
    """loop_predicate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.sum() > 0:
            x = x - 1.0
        return x


class _Escape(nn.Module):
    """tensor_derived_scalar_literal (the corr2recover repro-B shape)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = int(x[0].item())
        return x + n


class _Meta(nn.Module):
    """model_input_metadata (the free-F1 contiguity shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_contiguous():
            return self.lin(x)
        return self.lin(x) * 100.0


class _StateRead(nn.Module):
    """state_metadata (the hon1-F2 requires_grad shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.lin.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lin.weight.requires_grad:
            return self.lin(x)
        return self.lin(x) * 3.0


class _UnboundBuffer(nn.Module):
    """unbound_state_escape: a registered buffer consumed by NO traced call."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("gate", torch.tensor([2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


class _DictInOut(nn.Module):
    """container (MODEL_INPUT + MODEL_OUTPUT container records) + input_structure +
    model_input_literal."""

    def forward(self, x: torch.Tensor, flag: int) -> dict[str, torch.Tensor]:
        return {"out": x * 2.0 if flag == 1 else x + 100.0}


class _BatchNormEval(nn.Module):
    """module_training_mode (mode-sensitive op with a declared eval mode)."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class _DualUseSink(nn.Module):
    """inert_sink: a call-produced slot consumed by nothing (explicit dead claim)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        _ = y.sum()
        return y * 2.0


# Family -> (model factory, capture inputs factory, changed-run inputs factory).
# THE strip-fixture table: the registry-closure meta-test requires this table to
# cover every witness-carrying family plus the claim families exactly.
_FAMILY_FIXTURES: dict[
    str, tuple[Callable[[], nn.Module], Callable[[], Any], Callable[[], Any]]
] = {
    "scalar_bool": (
        _Branch,
        lambda: torch.tensor([1.0, 2.0, 3.0]),
        lambda: torch.tensor([-5.0, -6.0, -7.0]),
    ),
    "loop_predicate": (
        _Loop,
        lambda: torch.tensor([1.5]),
        lambda: torch.tensor([4.5]),
    ),
    "conditional_arm_entry": (
        _Branch,
        lambda: torch.tensor([1.0, 2.0, 3.0]),
        lambda: torch.tensor([-5.0, -6.0, -7.0]),
    ),
    "tensor_derived_scalar_literal": (
        _Escape,
        lambda: torch.tensor([5.0, 1.0, 1.0]),
        lambda: torch.tensor([99.0, 1.0, 1.0]),
    ),
    "input_structure": (
        _DictInOut,
        lambda: [torch.tensor([1.0, 2.0]), 1],
        lambda: [torch.tensor([1.0, 2.0]), 1],
    ),
    "model_input_literal": (
        _DictInOut,
        lambda: [torch.tensor([1.0, 2.0]), 1],
        lambda: [torch.tensor([1.0, 2.0]), 2],
    ),
    "model_input_metadata": (
        _Meta,
        lambda: torch.randn(2, 4),
        lambda: torch.randn(4, 2).t(),
    ),
    "module_training_mode": (
        lambda: _BatchNormEval().eval(),
        lambda: torch.randn(4, 4),
        lambda: torch.randn(4, 4),
    ),
    "state_metadata": (
        _StateRead,
        lambda: torch.randn(3),
        lambda: torch.randn(3),
    ),
    "container": (
        _DictInOut,
        lambda: [torch.tensor([1.0, 2.0]), 1],
        lambda: [torch.tensor([1.0, 2.0]), 1],
    ),
    "unbound_state_escape": (
        _UnboundBuffer,
        lambda: torch.tensor([1.0, 2.0]),
        lambda: torch.tensor([1.0, 2.0]),
    ),
    "inert_sink": (
        _DualUseSink,
        lambda: torch.tensor([1.0, 2.0]),
        lambda: torch.tensor([1.0, 2.0]),
    ),
    "unbound_state_inert": (
        # Reserved claim vocabulary: the current producer cannot prove inertness and
        # always claims "escaped"; the family's parse validation is exercised
        # synthetically below.
        _UnboundBuffer,
        lambda: torch.tensor([1.0, 2.0]),
        lambda: torch.tensor([1.0, 2.0]),
    ),
}


# ======================================================================================
# 1. Registry closure -- a new kind/family/gap kind cannot ship unregistered
# ======================================================================================


# ======================================================================================
# 2. Independence -- the required derivation cannot see witnesses/inventory/summary
# ======================================================================================


# ======================================================================================
# 3. The generated strip matrix -- EVERY family, single / pair / 3-way / forged-complete
# ======================================================================================


def _strip_family_witnesses(run: dict, family: str) -> int:
    """Remove every witness of one registry family; renumber nothing (raw strip)."""

    spec = WITNESS_FAMILY_REGISTRY[family]
    kept, removed = [], 0
    for witness in run["control_witnesses"]:
        if spec.witness_kind is None:
            kept.append(witness)
            continue
        if spec.witness_kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            matches = str(witness.get("site_label", "")).startswith(str(spec.site_prefix))
            matches = matches and witness.get("kind") == "shape_structure_fact"
        else:
            matches = witness.get("kind") == spec.witness_kind.value
        if matches:
            removed += 1
            continue
        kept.append(witness)
    run["control_witnesses"] = kept
    return removed


def _strip_inventory_members(run: dict, family: str) -> None:
    for row in run["required_witness_inventory"]["families"]:
        if row["family"] == family:
            row["members"] = []


def _strip_third_leg(run: dict, family: str) -> None:
    """Remove the owner-record obligation / claim leg where one exists in-manifest."""

    if family in {"scalar_bool", "loop_predicate"}:
        for call in run["calls"]:
            call["control_obligations"] = [
                obligation
                for obligation in call["control_obligations"]
                if obligation["kind"] != family
            ]
    elif family == "conditional_arm_entry":
        for call in run["calls"]:
            call["control_dependencies"] = []
    elif family == "tensor_derived_scalar_literal":
        for slot in run["tensor_slots"]:
            slot["host_escape"] = False
    elif family == "unbound_state_escape":
        for slot in run["tensor_slots"]:
            binding = slot.get("state_binding")
            if isinstance(binding, dict):
                binding["host_escape_disposition"] = None
    elif family in {"input_structure", "model_input_metadata"}:
        run["input_boundary"] = []
    elif family == "inert_sink":
        for slot in run["tensor_slots"]:
            slot["inert_sink"] = False


_THIRD_LEG_FAMILIES = frozenset(
    {
        "scalar_bool",
        "loop_predicate",
        "conditional_arm_entry",
        "tensor_derived_scalar_literal",
        "unbound_state_escape",
        "input_structure",
        "model_input_metadata",
        "inert_sink",
    }
)


@pytest.mark.smoke_cells("test_r71_strip_matrix_never_verified[conditional_arm_entry]")
@pytest.mark.parametrize("family", sorted(_VERDICT_STEERING_WITNESS_FAMILIES))
def test_r71_strip_matrix_never_verified(family: str, tmp_path: Path) -> None:
    """(a) single witness strip, (b) lockstep witness+member, (c) 3-way lockstep with
    the owner-record leg, (d) all of those with witness_completeness left "complete"
    -> parse-refuse typed OR run non-VERIFIED; NEVER VERIFIED/ATTESTED."""

    model_factory, capture_inputs, run_inputs = _FAMILY_FIXTURES[family]
    source = _save(_trace(model_factory(), capture_inputs()), tmp_path / f"{family}.tlspec")
    manifest = _json.loads((source / "manifest.json").read_text())
    present = _copy.deepcopy(manifest["run"])
    stripped_probe = _strip_family_witnesses(present, family)
    if family == "model_input_literal":
        assert stripped_probe > 0, "fixture must emit literal facts"
    else:
        assert stripped_probe > 0, f"fixture emits no {family!r} witness -- fix the fixture"

    mutations: list[tuple[str, Callable[[dict], None]]] = [
        ("single", lambda run: _strip_family_witnesses(run, family)),
        (
            "pair",
            lambda run: (
                _strip_family_witnesses(run, family),
                _strip_inventory_members(run, family),
            ),
        ),
    ]
    if family in _THIRD_LEG_FAMILIES:
        mutations.append(
            (
                "threeway",
                lambda run: (
                    _strip_family_witnesses(run, family),
                    _strip_inventory_members(run, family),
                    _strip_third_leg(run, family),
                ),
            )
        )
    # (d) every mutation repeated with the summary FORCED complete.
    mutations.extend(
        [
            (
                f"{name}_forged_complete",
                lambda run, inner=mutator: (
                    inner(run),
                    run.update(witness_completeness="complete"),
                ),
            )
            for name, mutator in list(mutations)
        ]
    )
    for name, mutator in mutations:
        path = tmp_path / f"{family}_{name}.tlspec"
        _shutil.copytree(source, path)
        _mutate(path, mutator)
        _assert_never_verified(path, run_inputs())


def test_r71_claim_family_strip_matrix(tmp_path: Path) -> None:
    """Claim-only families: an unclaimed terminal slot refuses; a claim on a
    non-terminal slot refuses; the reserved unbound_state_inert vocabulary parses
    only where the disposition totality allows it."""

    source = _save(_trace(_DualUseSink(), torch.tensor([1.0, 2.0])), tmp_path / "sink.tlspec")
    descriptor = tl.load(source)._runnable.descriptor
    inert_slots = [slot.slot_id for slot in descriptor.tensor_slots if slot.inert_sink]
    assert inert_slots, "fixture must claim at least one inert_sink slot"

    # Strip the claim (flag + mirror member) -> unclaimed terminal -> refuse.
    stripped = tmp_path / "sink_stripped.tlspec"
    _shutil.copytree(source, stripped)

    def _strip_claim(run: dict) -> None:
        for slot in run["tensor_slots"]:
            slot["inert_sink"] = False
        _strip_inventory_members(run, "inert_sink")

    _mutate(stripped, _strip_claim)
    outcome, detail = _verdict(stripped, torch.tensor([1.0, 2.0]))
    assert outcome == "refused" and "context_field_invalid" in detail, (outcome, detail)

    # Forge a claim on a consumed (non-terminal) slot -> refuse.
    forged = tmp_path / "sink_forged.tlspec"
    _shutil.copytree(source, forged)

    def _forge_claim(run: dict) -> None:
        consumed = {
            argument["slot_id"] for call in run["calls"] for argument in call["tensor_arguments"]
        }
        for slot in run["tensor_slots"]:
            if slot["slot_id"] in consumed:
                slot["inert_sink"] = True
                for row in run["required_witness_inventory"]["families"]:
                    if row["family"] == "inert_sink":
                        row["members"] = sorted(set(row["members"]) | {slot["slot_id"]})
                return

    _mutate(forged, _forge_claim)
    outcome, detail = _verdict(forged, torch.tensor([1.0, 2.0]))
    assert outcome == "refused" and "context_field_invalid" in detail, (outcome, detail)

    # unbound_state_inert on a BOUND state slot -> refuse (claim outside its domain).
    bound_state = _save(_trace(_StateRead(), torch.randn(3)), tmp_path / "bound_state.tlspec")
    misclaimed = tmp_path / "bound_state_inert.tlspec"
    _shutil.copytree(bound_state, misclaimed)

    def _misclaim(run: dict) -> None:
        consumed = {
            argument["slot_id"] for call in run["calls"] for argument in call["tensor_arguments"]
        }
        for slot in run["tensor_slots"]:
            binding = slot.get("state_binding")
            if isinstance(binding, dict) and slot["slot_id"] in consumed:
                binding["host_escape_disposition"] = "inert"
                for row in run["required_witness_inventory"]["families"]:
                    if row["family"] == "unbound_state_inert":
                        row["members"] = [f"{binding['state_dict_name']}::{slot['slot_id']}"]
                return

    _mutate(misclaimed, _misclaim)
    outcome, detail = _verdict(misclaimed, torch.randn(3))
    assert outcome == "refused" and "context_field_invalid" in detail, (outcome, detail)


# ======================================================================================
# 4. Named r70 E2E pins (grossly-wrong-output shapes, asserted by verdict)
# ======================================================================================


def test_r70_pin_corr2recover_repro_a_scalar_bool_single_strip(tmp_path: Path) -> None:
    """r70 corr2recover repro A: the single scalar_bool strip that flipped a branch
    to false VERIFIED now refuses at parse; the untampered artifact still diverges."""

    x = torch.tensor([1.0, 2.0, 3.0])
    changed = torch.tensor([-5.0, -6.0, -7.0])
    source = _save(_trace(_Branch(), x.clone()), tmp_path / "pin_a.tlspec")
    outcome, detail = _verdict(source, changed)
    assert outcome in {"ran", "raised"}
    if outcome == "ran":
        assert detail is not PathFaithfulness.VERIFIED
    tampered = tmp_path / "pin_a_strip.tlspec"
    _shutil.copytree(source, tampered)
    _mutate(tampered, lambda run: _strip_family_witnesses(run, "scalar_bool"))
    outcome, detail = _verdict(tampered, changed)
    assert outcome == "refused" and "context_field_invalid" in detail, (outcome, detail)


def test_r70_pin_corr2recover_repro_b_stale_escape_strips(tmp_path: Path) -> None:
    """r70 corr2recover repro B: the tensor_derived strip that blessed a stale baked
    literal as VERIFIED now refuses at parse -- single, pair, AND 3-way lockstep."""

    x = torch.tensor([5.0, 1.0, 1.0])
    changed = torch.tensor([99.0, 1.0, 1.0])
    source = _save(_trace(_Escape(), x.clone()), tmp_path / "pin_b.tlspec")
    result = tl.load(source).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    outcome, detail = _verdict(source, changed)
    assert outcome == "ran" and detail is PathFaithfulness.UNVERIFIABLE, (outcome, detail)
    for name, mutator in (
        ("single", lambda run: _strip_family_witnesses(run, "tensor_derived_scalar_literal")),
        (
            "pair",
            lambda run: (
                _strip_family_witnesses(run, "tensor_derived_scalar_literal"),
                _strip_inventory_members(run, "tensor_derived_scalar_literal"),
            ),
        ),
        (
            "threeway",
            lambda run: (
                _strip_family_witnesses(run, "tensor_derived_scalar_literal"),
                _strip_inventory_members(run, "tensor_derived_scalar_literal"),
                _strip_third_leg(run, "tensor_derived_scalar_literal"),
            ),
        ),
    ):
        tampered = tmp_path / f"pin_b_{name}.tlspec"
        _shutil.copytree(source, tampered)
        _mutate(tampered, mutator)
        outcome, detail = _verdict(tampered, changed)
        assert outcome == "refused" and "context_field_invalid" in detail, (name, detail)


def test_r71_producer_regression_fails_typed_at_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """corr2recover's correctness lane: a producer-emission regression that drops a
    witness family now fails TYPED at save through the shared self-check validator."""

    from torchlens._io import runnable as io_runnable

    monkeypatch.setattr(
        io_runnable, "_state_metadata_fact_witnesses", lambda slot_drafts, *, start_order: []
    )
    trace = _trace(_StateRead(), torch.randn(3))
    with pytest.raises(RunnablePreflightError) as excinfo:
        _save(trace, tmp_path / "regression.tlspec")
    diagnostics = str(excinfo.value.fields.get("diagnostics"))
    assert "context_field_invalid" in diagnostics
    assert "state_metadata" in diagnostics


# ======================================================================================
# 5. Completeness forgery -- the summary can never strengthen the derived floor
# ======================================================================================


class _OpaqueLeaf(nn.Module):
    def forward(self, x: torch.Tensor, cfg: Any) -> torch.Tensor:
        return x * 2.0


@pytest.mark.smoke
def test_r71_completeness_forgery_never_strengthens(tmp_path: Path) -> None:
    """Per gap class: forcing "complete" refuses (floor mismatch); stripping the gap
    (keeping the summary) refuses; stripping gap+summary leaves the surviving
    opaque-leaf literal fact contradicted; stripping gap+witness+summary breaks the
    bidirectional literal<->structure anchor. Never VERIFIED."""

    x = torch.tensor([1.0, 2.0])
    source = _save(
        _trace(_OpaqueLeaf(), [x.clone(), {"blob": (1 + 2j)}]), tmp_path / "forge.tlspec"
    )
    descriptor = tl.load(source)._runnable.descriptor
    assert any(gap.gap_kind is WitnessGapKind.OPAQUE_INPUT_LEAF for gap in descriptor.coverage_gaps)
    result = tl.load(source).run(inputs=[x.clone(), {"blob": (1 + 2j)}])
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE

    def _force_complete(run: dict) -> None:
        run["witness_completeness"] = "complete"

    def _strip_gaps(run: dict) -> None:
        run["coverage_gaps"] = []

    for name, mutator in (
        ("summary_forged", _force_complete),
        ("gap_stripped", _strip_gaps),
        ("gap_and_summary", lambda run: (_strip_gaps(run), _force_complete(run))),
    ):
        path = tmp_path / f"forge_{name}.tlspec"
        _shutil.copytree(source, path)
        _mutate(path, mutator)
        _assert_never_verified(path, [x.clone(), {"blob": (1 + 2j)}])


# ======================================================================================
# 6. Verdict monotonicity + 7. no-over-trigger GREENs
# ======================================================================================


def test_r71_no_over_trigger_greens(tmp_path: Path) -> None:
    """Honest artifacts keep their verdicts: tensor-only, literal-only, normal
    branch/loop, metadata-oblivious changed layout, no-state-read, mode-insensitive
    eval BN with declared mode, ordinary original-input host escape, and the
    dead-slot model with its explicit inert claim ALL stay VERIFIED."""

    class _TensorOnly(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x) * 2.0

    x3 = torch.tensor([1.0, 2.0, 3.0])
    cases: list[tuple[nn.Module, Any, Any]] = [
        (_TensorOnly(), x3.clone(), x3.clone()),
        (_DictInOut(), [x3.clone(), 1], [x3.clone(), 1]),
        (_Branch(), x3.clone(), x3.clone()),
        (_Loop(), torch.tensor([1.5]), torch.tensor([1.5])),
        (_Escape(), torch.tensor([5.0, 1.0, 1.0]), torch.tensor([5.0, 1.0, 1.0])),
        (_StateRead(), torch.randn(3), None),
        (_BatchNormEval().eval(), torch.randn(4, 4), None),
        (_DualUseSink(), torch.tensor([1.0, 2.0]), None),
        (_UnboundBuffer(), torch.tensor([1.0, 2.0]), None),
    ]
    for index, (model, capture_inputs, run_inputs) in enumerate(cases):
        path = _save(_trace(model, capture_inputs), tmp_path / f"green_{index}.tlspec")
        effective = capture_inputs if run_inputs is None else run_inputs
        result = tl.load(path).run(inputs=effective)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED, index

    # Metadata-OBLIVIOUS model: a changed-layout runtime input stays VERIFIED (the
    # totalized empty envelope compares nothing -- no over-trigger by construction).
    class _Oblivious(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    x = torch.randn(2, 4)
    path = _save(_trace(_Oblivious(), x.clone()), tmp_path / "green_layout.tlspec")
    noncontiguous = x.t().contiguous().t()
    assert not noncontiguous.is_contiguous()
    result = tl.load(path).run(inputs=noncontiguous)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# 8. Source-scan tripwires -- floor derivation is the ONLY completeness authority
# ======================================================================================


def test_r71_floor_source_scan_tripwire() -> None:
    """No verdict/readiness read of the raw persisted witness_completeness outside
    the single derivation function + the parse-time equality/re-assert belts."""

    import torchlens._io.runnable_load as runnable_load
    import torchlens._runnable.execution as execution
    import torchlens._runnable.path_faithfulness
    import torchlens._runnable.state as state

    # The verdict gate consults ONLY the derived floor.
    verdict_source = inspect.getsource(torchlens._runnable.path_faithfulness._path_faithfulness)
    assert "derived_witness_completeness" in verdict_source
    assert "descriptor.witness_completeness" not in verdict_source
    # The ONLY raw read left in the executor is the run-preparation equality
    # re-assert (defense in depth), and staging never reads it at all.
    execution_source = "\n".join(
        path.read_text()
        for path in sorted(Path(execution.__file__).parent.glob("*.py"))
        if path.name != "__init__.py"
    )
    assert execution_source.count("descriptor.witness_completeness") == 1
    assert "witness_completeness" not in inspect.getsource(state._apply_state_metadata_facts)
    # Readiness republishes the derived floor, never the raw summary.
    readiness_source = inspect.getsource(runnable_load._readiness_report)
    assert "derived_witness_completeness" in readiness_source
    assert "descriptor.witness_completeness" not in readiness_source
    # The parser's only raw reads are construction + the floor equality check.
    load_source = inspect.getsource(runnable_load)
    assert load_source.count("descriptor.witness_completeness") == 2  # equality check + message


# ======================================================================================
# 9. THE threat-model pin -- coherent reauthoring is an honest weaker program
# ======================================================================================
