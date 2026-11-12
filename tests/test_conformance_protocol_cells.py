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

from tests.conformance import ConformanceReport, run_conformance
from tests.conformance._adapters import (
    FAKE_PLANTS,
    RosterModel,
)
from tests.conformance._protocol_cases import (
    PLANT_INTENDED_CELL,
    constant_parents_problems,
)
from tests.conformance._refusal_cases import CAPABILITY_REQUESTS
from tests.conformance._scorecard import (
    SCORECARD_VERSION,
    capability_summary,
    scorecard_from_json,
    scorecards_differ,
)
from tests.conformance._validation_cells import (
    names_reason_problems,
    validation_status_problems,
)
from torchlens.backends._backend_protocol import REFUSABLE_CAPABILITY_FLAGS

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


def test_every_refusable_flag_has_a_request_recipe() -> None:
    """The one request table covers every refusable capability flag."""

    assert set(CAPABILITY_REQUESTS) == set(REFUSABLE_CAPABILITY_FLAGS)


def test_plant_tables_partition_the_closed_plant_set() -> None:
    """Legacy plants plus the protocol plants are exactly ``FAKE_PLANTS``."""

    assert _LEGACY_PLANTS.isdisjoint(PLANT_INTENDED_CELL)
    assert _LEGACY_PLANTS | set(PLANT_INTENDED_CELL) == set(FAKE_PLANTS)


# ---------------------------------------------------------------------------
# The reference and the unplanted fake pass every new cell (harness sanity).

#: Torch-side gaps the suite reports, pinned by exact cell id with the detail each carries.
#: Empty: torch's tamper verdict reason (``arg_logging_mismatch``) is in the closed list. A
#: new gap is pinned here with its detail and removed when it closes; the cells stay strict.
TORCH_KNOWN_GAPS: dict[str, str] = {}
#: The fake refuses its one False flag typed before capture; it validates through torch, so
#: it inherits torch's gaps.
FAKE_KNOWN_GAPS: dict[str, str] = {}


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_protocols/torch.py.
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

    from tests.conformance._scorecard import PreviewScorecard, ScorecardCell

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


# ---------------------------------------------------------------------------
# The command.


def test_scorecard_command_writes_checks_and_grades(tmp_path: Path, monkeypatch: Any) -> None:
    """``--write`` then ``--check`` agree; a changed cell fails ``--check``; ``--graduation`` reads the file."""

    from tests.conformance import _scorecard, scorecard
    from tests.conformance._graduation import GraduationItem
    from tests.conformance._scorecard import PreviewScorecard, ScorecardCell

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
