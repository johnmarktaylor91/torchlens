"""Protocol-era conformance cells (SPEC 4.2): torch passes, every plant fails its cell.

The torch reference and the unplanted fake must pass every new cell before any
plant is trusted; each new fake plant must then fail exactly the cell it
targets (validation cells may also go red for the two payload plants, because
validation is the tripwire for exactly those defects, and that collateral is
pinned here rather than hidden). Framework-free tests cover the pure oracles
with stub traces and statuses.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from torchlens.backends._backend_protocol import REFUSABLE_CAPABILITY_FLAGS
from torchlens.conformance import ConformanceReport, run_conformance
from torchlens.conformance._adapters import (
    FAKE_PLANTS,
    FakeProviderAdapter,
    RosterModel,
    TorchReferenceAdapter,
)
from torchlens.conformance._protocol_cases import (
    C1_CONSTANTS_CASE,
    C1_INTERMEDIATES_CASE,
    C1_PROTOCOL_ROW_CASES,
    C2_PROTOCOL_ROW_CASES,
    PLANT_INTENDED_CELL,
    c1_row_case_ids,
    c1_tail_case_ids,
    c2_row_case_ids,
    constant_parents_problems,
    intermediates_problems,
    payload_equality_problems,
)
from torchlens.conformance._refusal_cases import CAPABILITY_REQUESTS, refusal_code_of
from torchlens.conformance._scorecard import (
    SCORECARD_VERSION,
    capability_summary,
    scorecard_from_json,
    scorecard_from_report,
    scorecards_differ,
)
from torchlens.conformance._validation_cells import (
    names_reason_problems,
    validation_status_problems,
)

#: The six v1 plants, whose intended cells ``tests/test_ecosystem_rt_conformance.py`` pins.
_LEGACY_PLANTS = frozenset(
    {
        "structural_failure",
        "capability_undispatched",
        "output_corruption",
        "roundtrip_field_loss",
        "tamper_tripwire",
        "shared_adapter_assumption",
    }
)

#: A plant whose intended cell is also a pinned gap must show its own failure there,
#: or the pin would mask it: the detail the plant's row carries.
_PLANT_SIGNATURE: dict[str, str] = {"bare_bool_validation": "validate returned bool"}

#: Cells a plant may also redden because validation catches the same defect.
_PLANT_ALLOWED_COLLATERAL: dict[str, frozenset[str]] = {
    "missing_constant_payload": frozenset({"c1_validation_passes", "c1_validation_names_reason"}),
    "shape_drift_after_load": frozenset({"c2_validation_after_load"}),
    "shape_metadata_drift": frozenset({"c1_validation_passes"}),
}


def _build_tiny() -> tuple[Any, Any]:
    """Cheap hermetic MLP."""

    import torch

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).eval()
    return model, torch.randn(2, 8)


def _build_with_constants() -> tuple[Any, Any]:
    """A model whose forward uses a closure constant and a literal producer."""

    import torch

    class WithConstants(torch.nn.Module):
        """Linear over a plain tensor attribute, plus a doubled ``torch.ones``.

        The literal producer ``ones`` is the row's only constant; ``mul``
        (its doubling) is the input-free intermediate the NumPy oracle pins,
        so a dropped constant payload trips the constants cell alone.
        """

        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.lin = torch.nn.Linear(4, 3)
            self.scale = torch.tensor([2.0, 0.5, -1.0])

        def forward(self, x: Any) -> Any:
            return self.lin(x) / self.scale + torch.ones(3) * 2

    torch.manual_seed(1)
    return WithConstants().eval(), torch.randn(2, 4)


def _roster() -> tuple[RosterModel, ...]:
    """Two rows: the tiny MLP and the constants model with its NumPy oracle."""

    return (
        RosterModel(family="tiny", realism="config_built", builder=_build_tiny),
        RosterModel(
            family="constants",
            realism="config_built",
            builder=_build_with_constants,
            has_constants=True,
            expected_intermediates={"__mul__": np.full(3, 2.0, dtype=np.float32)},
        ),
    )


def _rows(report: ConformanceReport, case_id: str) -> list[Any]:
    """The result rows for one case id."""

    return [row for row in report.results if row.case_id == case_id]


# ---------------------------------------------------------------------------
# Slot enumeration.


def test_case_ids_follow_the_row_flags_and_the_protocol_view() -> None:
    """Protocol cells exist only for protocol-bearing adapters and follow the row flags."""

    torch_adapter = TorchReferenceAdapter()
    tiny, constants = _roster()
    ids = c1_row_case_ids(torch_adapter, tiny)
    assert ids[:3] == ("c1_capture_completes", "c1_reference_parity", "c1_deterministic_repeat")
    assert all(case_id in ids for case_id in C1_PROTOCOL_ROW_CASES)
    assert C1_CONSTANTS_CASE not in ids and C1_INTERMEDIATES_CASE not in ids
    flagged = c1_row_case_ids(torch_adapter, constants)
    assert C1_CONSTANTS_CASE in flagged and C1_INTERMEDIATES_CASE in flagged
    assert flagged.index(C1_INTERMEDIATES_CASE) < flagged.index("c1_validation_passes")
    tail = c1_tail_case_ids(torch_adapter)
    assert tail[0] == "c1_capabilities_consumed"
    assert "c1_refusal_intermediate_derived_grads" in tail
    assert not any(case_id.startswith("c1_refusal_trace_option_") for case_id in tail)
    assert c2_row_case_ids(torch_adapter, tiny)[-2:] == C2_PROTOCOL_ROW_CASES
    bare = SimpleNamespace(name="bare")
    assert c1_row_case_ids(bare, constants) == (
        "c1_capture_completes",
        "c1_reference_parity",
        "c1_deterministic_repeat",
    )
    assert c1_tail_case_ids(bare) == ("c1_capabilities_consumed",)
    assert c2_row_case_ids(bare, tiny) == (
        "c2_roundtrip_completes",
        "c2_reload_is_distinct",
        "c2_structure_equal",
    )


def test_every_refusable_flag_has_a_request_recipe() -> None:
    """The one request table covers every refusable capability flag."""

    assert set(CAPABILITY_REQUESTS) == set(REFUSABLE_CAPABILITY_FLAGS)


def test_plant_tables_partition_the_closed_plant_set() -> None:
    """Legacy plants plus the protocol plants are exactly ``FAKE_PLANTS``."""

    assert _LEGACY_PLANTS.isdisjoint(PLANT_INTENDED_CELL)
    assert _LEGACY_PLANTS | set(PLANT_INTENDED_CELL) == set(FAKE_PLANTS)


# ---------------------------------------------------------------------------
# The reference and the unplanted fake pass every new cell (harness sanity).

#: Torch-side gaps the suite reports today, pinned by exact cell id with the detail each
#: carries (cross-packet requests in the lane's RESULT.md): the tamper verdict names the
#: consumer with torch's ``arg_logging_mismatch``, a reason the closed list does not spell
#: yet (a spelling for JMT, never added here). A fix flips the pin, and the entry must then
#: be removed; the cells stay strict.
TORCH_KNOWN_GAPS: dict[str, str] = {
    "c1_validation_names_reason": "not in VALIDATION_REASONS",
}
#: The fake refuses its one False flag typed before capture; it validates through torch, so
#: it inherits the reason-spelling gap.
FAKE_KNOWN_GAPS: dict[str, str] = {
    "c1_validation_names_reason": "not in VALIDATION_REASONS",
}


def _assert_only_known_gaps(report: ConformanceReport, gaps: dict[str, str]) -> None:
    """Every non-passed cell is a pinned gap, and every pinned gap is still red."""

    red = [
        (row.case_id, row.family, row.detail) for row in report.results if row.outcome != "passed"
    ]
    assert {case_id for case_id, _, _ in red} == set(gaps), red
    for case_id, _family, detail in red:
        assert gaps[case_id] in (detail or ""), (case_id, detail)


@pytest.mark.heavy
def test_torch_reference_passes_every_protocol_cell(tmp_path: Path) -> None:
    """Torch passes graph, shape, gap, intermediates, validation, constants, refusal, C2 cells."""

    report = run_conformance("torch", tiers=("C0", "C1", "C2"), roster=_roster(), workdir=tmp_path)
    _assert_only_known_gaps(report, TORCH_KNOWN_GAPS)
    ids = {row.case_id for row in report.results}
    expected = set(C1_PROTOCOL_ROW_CASES) | set(C2_PROTOCOL_ROW_CASES)
    expected |= {C1_CONSTANTS_CASE, C1_INTERMEDIATES_CASE, "c1_refusal_intermediate_derived_grads"}
    assert expected <= ids, sorted(expected - ids)
    card = scorecard_from_report(
        report, _roster(), framework_version="x", adapter=TorchReferenceAdapter()
    )
    assert card.scorecard_version == SCORECARD_VERSION
    assert card.maturity == "stable"
    assert card.framework_requirement["pinned"]
    # The naming gap sits in C1, so the ladder stops at C0 until the status names the op.
    assert card.highest_tier == "C0"
    assert card.roster_groups == {"config_built": 2}
    # SPEC 4.5: a False flag reads "unsupported" once its refusal cell passes.
    assert card.capabilities["intermediate_derived_grads"] == "unsupported"
    assert all(
        state == "supported"
        for flag, state in card.capabilities.items()
        if flag != "intermediate_derived_grads"
    ), card.capabilities
    assert [item.id for item in card.graduation] == ["G1", "G2", "G3", "G4", "G5", "G6"]
    # G1 and G2 read the pinned gaps; they flip with the gaps. G1 names the first red cell
    # in suite order (a pinned one); the coded grad_options refusal leaves no flag untested,
    # and the executable-level reload (P1b) takes c2_validation_after_load out of G2's detail.
    g1 = card.graduation[0]
    assert not g1.met and any(gap in g1.detail for gap in TORCH_KNOWN_GAPS), g1
    assert "c1_refusal_intermediate_derived_grads" not in g1.detail, g1
    assert "capabilities untested" not in g1.detail, g1
    g2 = card.graduation[1]
    assert not g2.met and "c1_validation_names_reason" in g2.detail, g2
    assert "c2_validation_after_load" not in g2.detail, g2
    assert card.counts["denominator"] == len(card.cells)
    assert scorecard_from_json(card.to_json()) == card


@pytest.mark.heavy
def test_unplanted_fake_passes_every_protocol_cell(tmp_path: Path) -> None:
    """The well-behaved fake passes before any plant is trusted to fail."""

    report = run_conformance("fake", tiers=("C0", "C1", "C2"), roster=_roster(), workdir=tmp_path)
    _assert_only_known_gaps(report, FAKE_KNOWN_GAPS)


@pytest.mark.heavy
@pytest.mark.parametrize("plant", sorted(PLANT_INTENDED_CELL))
def test_each_protocol_plant_fails_exactly_its_cell(plant: str, tmp_path: Path) -> None:
    """Each plant reddens its intended cell, and nothing else but its pinned collateral."""

    intended = PLANT_INTENDED_CELL[plant]
    tiers = ("C0", "C1", "C2") if intended.startswith("c2_") else ("C0", "C1")
    report = run_conformance("fake", tiers=tiers, roster=_roster(), workdir=tmp_path, _plant=plant)
    rows = _rows(report, intended)
    assert rows, f"plant {plant} produced no {intended} rows"
    assert any(row.outcome == "failed" for row in rows), [(r.case_id, r.detail) for r in rows]
    if intended in FAKE_KNOWN_GAPS:
        signature = _PLANT_SIGNATURE[plant]
        assert any(signature in (row.detail or "") for row in rows), [r.detail for r in rows]
    allowed = {intended} | _PLANT_ALLOWED_COLLATERAL.get(plant, frozenset()) | set(FAKE_KNOWN_GAPS)
    stray = [
        (row.case_id, row.family, row.detail)
        for row in report.results
        if row.outcome != "passed" and row.case_id not in allowed
    ]
    assert not stray, stray


# ---------------------------------------------------------------------------
# Pure oracles on stub inputs (framework-free).


class _Status:
    """Stub replay status."""

    def __init__(self, state: str, message: str = "", decisions: Any = (), unverified: Any = None):
        self.state = state
        self.message = message
        self.decisions = tuple(decisions)
        self.unverified_reason_counts = dict(unverified or {})


def _op(
    label: str,
    out: Any,
    parents: tuple[str, ...] = (),
    children: tuple[str, ...] = (),
    **extra: Any,
) -> Any:
    """Stub op record."""

    return SimpleNamespace(
        label=label,
        layer_label=extra.pop("layer_label", label),
        out=out,
        parents=parents,
        children=children,
        func_name=extra.pop("func_name", "op"),
        is_input=extra.pop("is_input", False),
        is_buffer=extra.pop("is_buffer", False),
        **extra,
    )


def test_validation_status_oracle_rejects_bare_values_and_foreign_reasons() -> None:
    """A bare bool, a non-passed state and an unregistered reason each fail."""

    assert validation_status_problems(True)[0].startswith("validate returned bool")
    assert validation_status_problems(_Status("passed")) == ()
    assert "state='failed'" in validation_status_problems(_Status("failed", "boom"))[0]
    bad = validation_status_problems(_Status("passed", unverified={"made_up_reason": 1}))
    assert "made_up_reason" in bad[0]
    assert validation_status_problems(_Status("passed", unverified={"replay_mismatch": 2})) == ()


def test_failed_replay_status_names_the_first_failed_decision() -> None:
    """The torch status message names the failed op and reason (the naming cell's source)."""

    from torchlens.validation.status import ValidationStatus

    rows = [
        {"op_label": "relu_1_2:1", "decision": "validated", "reason": "replay_matched"},
        {"op_label": "linear_1_1:1", "decision": "failed", "reason": "replay_mismatch"},
        {"op_label": None, "decision": "failed", "reason": "bfs_incomplete"},
    ]
    status = ValidationStatus.from_replay_counts(
        backend="torch",
        source="live",
        num_replayed_ops=1,
        num_unverified_ops=0,
        num_failed_ops=2,
        decisions=rows,
    )
    assert status.state == "failed"
    assert status.message == "Replay validation failed at linear_1_1:1 (replay_mismatch)."
    anonymous = ValidationStatus.from_replay_counts(
        backend="torch",
        source="live",
        num_replayed_ops=0,
        num_unverified_ops=0,
        num_failed_ops=1,
        decisions=rows[2:],
    )
    assert anonymous.message == "Replay validation failed (bfs_incomplete)."
    passed = ValidationStatus.from_replay_counts(
        backend="torch", source="live", num_replayed_ops=1, num_unverified_ops=0
    )
    assert passed.message == "Replay validation passed."


def test_names_reason_oracle_requires_the_op_and_a_registered_reason() -> None:
    """The failed verdict must name the tampered op (or its consumer) with a reason."""

    target = _op("mul_1", np.ones(2), children=("add_1",))
    assert names_reason_problems(False, target)[0].startswith("validate returned bool")
    assert "validated as 'passed'" in names_reason_problems(_Status("passed"), target)[0]
    anonymous = _Status("failed", "something broke")
    problems = names_reason_problems(anonymous, target)
    assert any("no decision names mul_1" in line for line in problems)
    assert any("message does not name mul_1" in line for line in problems)
    named = _Status(
        "failed",
        "replay of add_1 mismatched",
        decisions=[{"op_label": "add_1", "reason": "replay_mismatch"}],
    )
    assert names_reason_problems(named, target) == ()
    # A child list names the consumer by one key; the decision may use another
    # key of the same op, resolved through the trace.
    consumer = _op("add_1:1", np.ones(2), parents=("mul_1",), layer_label="add_1")
    trace = SimpleNamespace(_layer_list=[target, consumer])
    by_pass_label = _Status(
        "failed",
        "replay of add_1:1 mismatched",
        decisions=[{"op_label": "add_1:1", "reason": "replay_mismatch"}],
    )
    assert any("no decision names" in line for line in names_reason_problems(by_pass_label, target))
    assert names_reason_problems(by_pass_label, target, trace) == ()
    foreign = _Status(
        "failed", "mul_1 failed", decisions=[{"op_label": "mul_1", "reason": "vibes"}]
    )
    assert "not in VALIDATION_REASONS" in names_reason_problems(foreign, target)[0]


def test_payload_and_intermediate_oracles_compare_arrays_bit_for_bit_or_within_band() -> None:
    """Reload equality is exact including shape; intermediates use the dtype band."""

    live = SimpleNamespace(_layer_list=[_op("a", np.arange(4.0)), _op("b", np.ones((2, 2)))])
    same = SimpleNamespace(_layer_list=[_op("a", np.arange(4.0)), _op("b", np.ones((2, 2)))])
    assert payload_equality_problems(live, same, np.asarray) == ()
    drifted = SimpleNamespace(_layer_list=[_op("a", np.arange(4.0)), _op("b", np.ones((1, 2, 2)))])
    assert (
        "shape (2, 2) != reloaded (1, 2, 2)"
        in payload_equality_problems(live, drifted, np.asarray)[0]
    )
    missing = SimpleNamespace(_layer_list=[_op("a", np.arange(4.0))])
    assert "b: missing after reload" in payload_equality_problems(live, missing, np.asarray)
    trace = SimpleNamespace(
        _layer_list=[_op("ones_1", np.ones(3, dtype=np.float32), func_name="ones")]
    )
    assert intermediates_problems(trace, {"ones": np.ones(3)}, np.asarray) == ()
    assert intermediates_problems(trace, {"ones": [np.ones(3), np.ones(3)]}, np.asarray)[
        0
    ].startswith("ones: 1 captured ops, 2 expected")
    off = intermediates_problems(trace, {"ones": np.ones(3) * 1.5}, np.asarray)
    assert "max abs difference" in off[0]


def test_constant_parents_oracle_needs_payload_and_consumer() -> None:
    """A flagged row needs a constant op with a payload and a consuming op."""

    empty = SimpleNamespace(_layer_list=[_op("input_1", np.ones(1), is_input=True)])
    assert "no constant op recorded" in constant_parents_problems(empty, np.asarray)[0]
    good = SimpleNamespace(
        _layer_list=[
            _op("input_1", np.ones(1), is_input=True),
            _op("constant_1", np.ones(1), children=("mul_1",)),
            _op("mul_1", np.ones(1), parents=("input_1", "constant_1")),
        ]
    )
    assert constant_parents_problems(good, np.asarray) == ()
    orphan = SimpleNamespace(_layer_list=[_op("constant_1", None)])
    problems = constant_parents_problems(orphan, np.asarray)
    assert "without a payload" in problems[0]


def test_capability_summary_and_shipped_record_comparison() -> None:
    """Summary vocabulary follows the cells; the comparison ignores the TorchLens version."""

    from torchlens.conformance._scorecard import PreviewScorecard, ScorecardCell

    protocol = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(
            **{flag: flag != "fastlog" for flag in REFUSABLE_CAPABILITY_FLAGS}
        )
    )
    consumed = ScorecardCell("C1", "c1_capabilities_consumed", "", "passed")
    refused = ScorecardCell("C1", "c1_refusal_fastlog", "", "passed")
    summary = capability_summary(protocol, (consumed, refused))
    assert summary["fastlog"] == "unsupported" and summary["backward_capture"] == "supported"
    summary = capability_summary(protocol, (consumed,))
    assert summary["fastlog"] == "untested"
    summary = capability_summary(protocol, ())
    assert set(summary.values()) == {"untested"}
    assert capability_summary(None, ()) == {}
    card = PreviewScorecard(
        SCORECARD_VERSION, "x", "1.0", "s", "2.0.0", (consumed,), {"passed": 1}, "C0"
    )
    newer = PreviewScorecard(
        SCORECARD_VERSION, "x", "1.0", "s", "2.1.0", (consumed,), {"passed": 1}, "C0"
    )
    assert scorecards_differ(card, newer) == []
    changed = PreviewScorecard(
        SCORECARD_VERSION, "x", "1.0", "s", "2.0.0", (refused,), {"passed": 1}, "C0"
    )
    lines = scorecards_differ(card, changed)
    assert any("c1_capabilities_consumed" in line for line in lines)
    assert any("c1_refusal_fastlog" in line for line in lines)


def test_refusal_code_reads_the_instance_field_before_the_class_code() -> None:
    """``BackendUnsupportedError(code=...)`` carries the code on ``fields``; the class code is the fallback."""

    from torchlens.backends.registry import BackendUnsupportedError

    assert refusal_code_of(BackendUnsupportedError("x")) == "backend_unsupported"
    assert refusal_code_of(BackendUnsupportedError("x", code="fastlog_unsupported")) == (
        "fastlog_unsupported"
    )
    assert refusal_code_of(RuntimeError("x")) == "RuntimeError"


def test_fake_adapter_refuses_its_false_flag_typed_and_untyped_when_planted() -> None:
    """The fake's own refusal path: typed by default, a plain raise under the plant."""

    from torchlens.backends.registry import BackendUnsupportedError

    model, example = _build_tiny()
    with pytest.raises(BackendUnsupportedError) as excinfo:
        FakeProviderAdapter().trace_with(model, example, grad_options={"derived": True})
    assert refusal_code_of(excinfo.value) == "intermediate_derived_grads_unsupported"
    with pytest.raises(RuntimeError):
        FakeProviderAdapter("untyped_refusal").trace_with(
            model, example, grad_options={"derived": True}
        )


# ---------------------------------------------------------------------------
# The command.


def test_scorecard_command_writes_checks_and_grades(tmp_path: Path, monkeypatch: Any) -> None:
    """``--write`` then ``--check`` agree; a changed cell fails ``--check``; ``--graduation`` reads the file."""

    from torchlens.conformance import _scorecard, scorecard
    from torchlens.conformance._graduation import GraduationItem
    from torchlens.conformance._scorecard import PreviewScorecard, ScorecardCell

    cell = ScorecardCell("C0", "c0_provider_identity", "", "passed")
    card = _scorecard._with_digest(
        PreviewScorecard(
            SCORECARD_VERSION,
            "stub",
            "1.0",
            "suite",
            "2.0.0",
            (cell,),
            {"passed": 1, "denominator": 1},
            "C0",
            maturity="preview",
            graduation=(GraduationItem("G1", False, "stub"),),
        )
    )
    target = tmp_path / "stub" / "scorecard.json"
    monkeypatch.setattr(scorecard, "_preview_scorecard", lambda backend, **kw: card)
    monkeypatch.setattr(scorecard, "shipped_scorecard_path", lambda backend: target)
    assert scorecard.main(["--backend", "stub", "--graduation"]) == 2
    assert scorecard.main(["--backend", "stub", "--check"]) == 2
    assert scorecard.main(["--backend", "stub", "--write"]) == 0
    assert scorecard_from_json(target.read_text(encoding="utf-8")) == card
    assert scorecard.main(["--backend", "stub", "--check"]) == 0
    assert scorecard.main(["--backend", "stub", "--graduation"]) == 0
    changed = _scorecard._with_digest(
        PreviewScorecard(
            SCORECARD_VERSION,
            "stub",
            "1.0",
            "suite",
            "2.0.0",
            (ScorecardCell("C0", "c0_provider_identity", "", "failed", "x"),),
            {"failed": 1, "denominator": 1},
            "",
        )
    )
    monkeypatch.setattr(scorecard, "_preview_scorecard", lambda backend, **kw: changed)
    assert scorecard.main(["--backend", "stub", "--check"]) == 1
