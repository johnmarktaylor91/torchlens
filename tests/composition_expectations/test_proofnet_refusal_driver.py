"""The refusal-contract driver (compo waves A-D; the three test laws, foldB D18).

Law 1: every protective claim gets a test that TRIES the forbidden thing and
asserts the refusal fired. Law 3: the refusal fired for the RIGHT REASON --
stable ``fields['code']`` plus the failed predicate named in the message.
Each row below is one forbidden thing, driven for real against the merged
tree; "it raised something" is never accepted.

Rows whose refusal is currently CODELESS or SILENT are ledgered inline
(state EXPECTED_GAP) so the fix must flip the row deliberately.
Ground truth probed live, 2026-08-30.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.heavy, pytest.mark.compo]


#: Memo D12 fixture-economics license (test_galleries.py lint):
COMPO_CONSTRUCTION_LICENSE = "refusal rows TRY forbidden constructions at entry doors (trace/save/load are the doors under test)"


@dataclass(frozen=True)
class RefusalRow:
    """One forbidden thing + its contracted refusal.

    Parameters
    ----------
    row_id:
        Stable row id.
    code:
        The contracted stable refusal code (empty only for EXPECTED_GAP rows).
    reason_needle:
        A substring of the teaching message naming the FAILED PREDICATE --
        law 3's "right reason" witness, deliberately short so wording can
        breathe while the predicate stays named.
    provoke:
        Zero-arg callable performing the forbidden thing.
    expected_gap:
        Non-empty = the refusal is currently absent/codeless; the row pins
        the CURRENT behavior and names the ledger entry a fix must burn down.
    """

    row_id: str
    code: str
    reason_needle: str
    provoke: Callable[[], Any]
    expected_gap: str = ""


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()


def _x() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 4)


def _tl() -> Any:
    import torchlens

    return torchlens


def _structure_only_trace() -> Any:
    tl = _tl()
    return tl.trace(_model(), _x(), capture=tl.options.CaptureOptions(structure_only=True))


REFUSAL_ROWS: tuple[RefusalRow, ...] = (
    RefusalRow(
        "RF-structopt-conflict",
        "structure_only_option_conflict",
        "structure_only",
        lambda: _tl().trace(
            _model(),
            _x(),
            capture=_tl().options.CaptureOptions(structure_only=True, raise_on_nan=True),
        ),
    ),
    RefusalRow(
        "RF-input-rung-conflict",
        "input_rung_conflict",
        "input",
        lambda: _tl().trace(_model(), _x(), input_size=(2, 4)),
    ),
    RefusalRow(
        "RF-unknown-backend",
        "unknown_backend",
        "Registered backends",
        lambda: _tl().trace(_model(), _x(), backend="theano"),
    ),
    RefusalRow(
        "RF-selection-kind-mix",
        "selection_kind_incompatible",
        "kind",
        lambda: _tl().units("relu_1_2", [(0, 0)]) & _tl().params("0.weight"),
    ),
    RefusalRow(
        "RF-halted-runnable-save",
        "halted_capture_not_runnable",
        "HALTED",
        lambda: _tl().save(
            _tl().trace(_model(), _x(), halt=_tl().func("relu")),
            "/tmp/proofnet-halted-runnable-refused.tlspec",
            level="runnable",
        ),
    ),
    RefusalRow(
        "RF-edges-without-arming",
        "edge_provenance_unavailable",
        "intervention_ready",
        lambda: _tl().trace(_model(), _x()).edges,
    ),
    RefusalRow(
        "RF-summary-axis-conflict",
        "summary_option_conflict",
        "depth=",
        lambda: _tl().trace(_model(), _x()).summary(level="op", depth=2),
    ),
    RefusalRow(
        "RF-summary-removed-spelling",
        "summary_option_invalid",
        "flop_convention=",
        lambda: _tl().trace(_model(), _x()).summary(count_fma_as_two=False),
    ),
    RefusalRow(
        "RF-topk-param-population",
        "selection_kind_incompatible",
        "ACT",
        lambda: _tl().top_k(_tl().params("0.weight"), 3),
    ),
    RefusalRow(
        "RF-changed-self-comparison",
        "selection_unresolvable",
        "own reference",
        lambda: (lambda tl, tr: tl.changed(tr).resolve(tr))(_tl(), _tl().trace(_model(), _x())),
    ),
    RefusalRow(
        "RF-load-missing-artifact",
        "manifest_missing",
        "Manifest not found",
        lambda: _tl().load("/tmp/proofnet-definitely-missing.tlspec"),
    ),
    RefusalRow(
        "RF-closure-root-unsupported",
        "model_type_unsupported",
        "function",
        lambda: (lambda tl, m: tl.trace(lambda z: m(z), _x()))(_tl(), _model()),
    ),
    RefusalRow(
        "RF-agent-json-bad-max-ops",
        "",
        "positive integer",
        lambda: _tl().trace(_model(), _x()).to_agent_json(max_ops=-3),
        expected_gap="bare ValueError (DIGEST-AUDIT agent-path holes: typed"
        " degradation class); typing it must flip this row",
    ),
)


@pytest.mark.parametrize("row", REFUSAL_ROWS, ids=lambda row: row.row_id)
def test_forbidden_thing_refuses_with_code_and_reason(row: RefusalRow) -> None:
    """TRY the forbidden thing; assert code + named predicate (laws 1+3)."""

    with pytest.raises(Exception) as excinfo:
        row.provoke()
    exc = excinfo.value
    code = getattr(exc, "fields", {}).get("code") if hasattr(exc, "fields") else None
    if row.expected_gap:
        assert not code, (
            f"{row.row_id}: the ledgered codeless refusal now carries"
            f" code={code!r} -- promote the row (gap: {row.expected_gap})"
        )
    else:
        assert code == row.code, (
            f"{row.row_id}: refusal fired with code={code!r}, contract says"
            f" {row.code!r} (law 3: stable code)"
        )
    assert row.reason_needle.lower() in str(exc).lower(), (
        f"{row.row_id}: message does not name the failed predicate"
        f" ({row.reason_needle!r}): {str(exc)[:160]!r}"
    )


def test_refusal_rows_are_unique_and_typed_rows_carry_codes() -> None:
    """Row-schema teeth for the driver table itself."""

    ids = [row.row_id for row in REFUSAL_ROWS]
    assert len(ids) == len(set(ids))
    for row in REFUSAL_ROWS:
        assert row.expected_gap or row.code, f"{row.row_id}: no code and no ledgered gap"


def test_structure_only_payload_read_gap_is_pinned() -> None:
    """LEDGERED FINDING (F36, 2026-08-30): reading ``.out`` on a
    structure-only capture returns silent ``None`` instead of the typed
    payload refusal the L7a chokepoint owes (SG#4's silent-None class on a
    NEW door). Pinned so the fix flips this test, never silently."""

    trace = _structure_only_trace()
    payload = trace["relu_1_2"].out
    assert payload is None, (
        "structure-only .out no longer returns silent None -- if it now"
        " refuses typed, move this row into REFUSAL_ROWS with its code;"
        " if it returns a VALUE, structure-only is leaking payloads (halt)"
    )
