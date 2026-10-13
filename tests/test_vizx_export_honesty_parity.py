"""Export honesty parity lockstep (R2-11, N-VZ-4).

Every capture-honesty fact ``torchlens._capture_honesty.capture_honesty_facts``
emits must reach every graph and agent surface -- draw, svg, html, Model
Explorer, Netron, the TensorBoard exporter and agent JSON -- or be refused by
that surface explicitly. The fact list is NOT hand-copied: it is the key set
the shared fact source returns for planted captures that drive every fact
into its disclosing state. A new fact therefore fails this test until it is
classified below (a text marker, plus a carry-or-refuse decision per
surface), and a refusal that a surface later starts carrying fails too, so
the table cannot go stale in either direction.

Two carriage kinds exist. Block surfaces (Netron, Model Explorer, agent
JSON) carry the fact block itself and are checked value by value. Text
surfaces (draw caption, svg and html comments, the TensorBoard text
summary) carry the shared banner or preamble lines and are checked for each
fact's disclosure marker. TorchLens has no TensorBoard graph-plugin export;
``tl.export.tensorboard`` (scalars plus the ``capture_honesty`` text) is the
TensorBoard surface.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_honesty import capture_honesty_facts
from torchlens.capture.outcome import CaptureStatus
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

#: The block's own schema stamp is not a fact about the capture.
_NOT_A_FACT = frozenset({"schema"})

#: Disclosure marker per fact for text surfaces: a fixed phrase of the shared
#: banner/preamble wording, or ``None`` to match the fact's own value (matched
#: case-insensitively).
TEXT_MARKERS: dict[str, str | None] = {
    "capture_status": None,
    "capture_verified": "unverified",
    "capture_verification_reason": None,
    "rescue_rerun": "rescue re-run",
    "structure_only": "structure-only",
    "claim_status": None,
    "values_available": "no tensor payloads",
    "substrate": None,
    "structure_evidence": None,
    "discharge": "real-run discharge",
    "poisoned": "poisoned",
    "path_faithfulness": None,
    "first_mismatch": None,
    "episode": "episode capture",
    "interventions": "intervened",
    "advisories": "probe_advisory",
}

SURFACES = (
    "draw",
    "svg",
    "html",
    "model_explorer",
    "netron",
    "tensorboard",
    "agent_json",
)
TEXT_SURFACES = frozenset({"draw", "svg", "html", "tensorboard"})

#: Explicit refusals: (surface, fact) -> why the surface does not carry it.
#: Each is re-checked to be really absent, so a surface that starts carrying
#: the fact forces the row's removal.
REFUSED: dict[tuple[str, str], str] = {
    **{
        (surface, "first_mismatch"): (
            "the caption, svg comment and TensorBoard text carry the one-line "
            "POISONED disclosure with path_faithfulness; the bounded "
            "first-mismatch text rides html and the block surfaces"
        )
        for surface in ("draw", "svg", "tensorboard")
    },
    **{
        (surface, "structure_evidence"): (
            "text surfaces name the substrate from the evidence envelope; the "
            "envelope dict itself rides the block surfaces only"
        )
        for surface in TEXT_SURFACES
    },
}


class _Tiny(nn.Module):
    """Two-op toy every surface renders quickly."""

    def __init__(self) -> None:
        """Build one linear layer."""

        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.fc(x))


def _fresh_trace() -> tl.Trace:
    """Capture the toy once."""

    torch.manual_seed(0)
    return tl.trace(_Tiny(), torch.randn(2, 3))


def _planted_disclosing_trace() -> tl.Trace:
    """A trace whose every non-structure-only fact is in its disclosing state."""

    trace = _fresh_trace()
    # ``Trace.outcome`` is a read-only property over the settled sidecar.
    trace.__dict__["_capture_outcome"] = dataclasses.replace(
        trace.outcome, status=CaptureStatus.UNATTESTED
    )
    trace.capture_verified = False
    trace.capture_verification_reason = "PROBE_REASON"
    trace.rescue_rerun = True
    trace._runnable.poisoned = True
    trace._runnable.path_faithfulness = PathFaithfulness.DIVERGED
    trace._runnable.first_mismatch = "PROBE_MISMATCH"
    trace.annotations["episode"] = {
        "header": {
            "capture_kind": "episode",
            "n_steps_declared": 3,
            "token_feed": "forced",
            "fidelity_basis": "forced",
        },
        "rows": [],
    }
    trace.annotations["capture_advisories"] = [
        {"kind": "PROBE_ADVISORY", "count": 2, "first_location": None, "message": "probe"}
    ]
    trace.intervention_audit.append({"kind": "PROBE_AUDIT"})
    return trace


def _discharged_structure_only_trace() -> tl.Trace:
    """A real structure-only capture, discharged against a real run (CORROBORATED)."""

    torch.manual_seed(0)
    model = _Tiny()
    structure = tl.trace(model, torch.randn(2, 3), capture=CaptureOptions(structure_only=True))
    structure.discharge_against(tl.trace(model, torch.randn(2, 3)))
    return structure


@pytest.fixture(scope="module")
def planted() -> Iterator[dict[str, tl.Trace]]:
    """Both planted captures, keyed by family, released at teardown."""

    traces = {
        "disclosing": _planted_disclosing_trace(),
        "structure_only": _discharged_structure_only_trace(),
    }
    try:
        yield traces
    finally:
        for trace in traces.values():
            trace.cleanup()


def _enumerated_facts(planted: dict[str, tl.Trace]) -> dict[str, tuple[str, Any]]:
    """Fact key -> (family, value) in its DISCLOSING state, from the shared source.

    A fact is attributed to the first family where it differs from a clean
    capture (or is absent from one), so a fact every capture carries is
    checked where it says something.
    """

    clean = capture_honesty_facts(_fresh_trace())
    facts: dict[str, tuple[str, Any]] = {}
    for family, trace in planted.items():
        for key, value in capture_honesty_facts(trace).items():
            if key in _NOT_A_FACT or (key in clean and clean[key] == value):
                continue
            facts.setdefault(key, (family, value))
    return facts


def _text_marker(key: str, value: Any) -> str:
    """The text a banner/preamble surface must contain for one fact."""

    marker = TEXT_MARKERS[key]
    return str(value).lower() if marker is None else marker


class _Writer:
    """Duck-typed TensorBoard writer that records every emitted row."""

    def __init__(self) -> None:
        """Start with no rows."""

        self.rows: list[tuple[Any, ...]] = []

    def add_scalar(self, *args: Any) -> None:
        """Record one scalar row."""

        self.rows.append(("scalar", *args))

    def add_text(self, *args: Any) -> None:
        """Record one text row."""

        self.rows.append(("text", *args))


def _draw(trace: tl.Trace, tmp: Path) -> str:
    """The draw caption source (DOT)."""

    return str(trace.draw(vis_save_only=True))


def _svg(trace: tl.Trace, tmp: Path) -> str:
    """The svg export text."""

    return tl.export.svg(trace, tmp / "graph.svg").read_text(encoding="utf-8")


def _html(trace: tl.Trace, tmp: Path) -> str:
    """The html export text."""

    return tl.export.html(trace, tmp / "page.html").read_text(encoding="utf-8")


def _tensorboard(trace: tl.Trace, tmp: Path) -> str:
    """The TensorBoard exporter's capture_honesty text summary."""

    writer = _Writer()
    tl.export.tensorboard(trace, writer, step=0)
    texts = [row[2] for row in writer.rows if row[0] == "text" and "capture_honesty" in row[1]]
    assert len(texts) == 1, "the TensorBoard exporter must write exactly one honesty text"
    return str(texts[0])


def _netron(trace: tl.Trace, tmp: Path) -> dict[str, Any]:
    """The Netron artifact's capture_honesty metadata block."""

    path = tl.export.netron(trace, tmp / "graph.json", attachment=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    props = {row["key"]: row["value"] for row in payload["metadataProps"]}
    return dict(json.loads(props["torchlens.capture_honesty"]))


def _model_explorer(trace: tl.Trace, tmp: Path) -> dict[str, Any]:
    """The Model Explorer collection's root disclosure row (first graph)."""

    path = tl.export.model_explorer(trace, tmp / "me.json", per_step=False)
    graphs = json.loads(path.read_text(encoding="utf-8"))["graphs"]
    return dict(graphs[0]["groupNodeAttributes"][""])


def _agent_json(trace: tl.Trace, tmp: Path) -> dict[str, Any]:
    """The agent JSON dump's capture block."""

    return dict(json.loads(json.dumps(trace.to_agent_json(max_ops=1), default=str))["capture"])


EXTRACTORS: dict[str, Callable[[tl.Trace, Path], Any]] = {
    "draw": _draw,
    "svg": _svg,
    "html": _html,
    "tensorboard": _tensorboard,
    "netron": _netron,
    "model_explorer": _model_explorer,
    "agent_json": _agent_json,
}


def _model_explorer_text(value: Any) -> str:
    """Model Explorer's string encoding of one fact (``_base_root_facts``)."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)[:500]


def _carries(surface: str, carried: Any, key: str, value: Any) -> bool:
    """Whether one extracted surface payload carries one fact."""

    if surface in TEXT_SURFACES:
        return _text_marker(key, value) in str(carried).lower()
    if key not in carried:
        return False
    if surface == "model_explorer":
        return bool(carried[key] == _model_explorer_text(value))
    expected = json.loads(json.dumps(value, default=str))
    return bool(carried[key] == expected)


def test_every_enumerated_fact_is_classified(planted: dict[str, tl.Trace]) -> None:
    """A new honesty fact fails here until it gets a marker and a decision."""

    facts = _enumerated_facts(planted)
    unclassified = sorted(set(facts) - set(TEXT_MARKERS))
    assert not unclassified, (
        f"new capture-honesty fact(s) {unclassified}: add a text marker to "
        "TEXT_MARKERS and carry or explicitly refuse each on every surface"
    )
    stale_markers = sorted(set(TEXT_MARKERS) - set(facts))
    assert not stale_markers, (
        f"TEXT_MARKERS names fact(s) the shared source no longer emits: {stale_markers}"
    )
    stale_refusals = sorted(f for _s, f in REFUSED if f not in facts)
    assert not stale_refusals, f"REFUSED names unknown fact(s): {stale_refusals}"


@pytest.mark.parametrize("surface", SURFACES)
def test_surface_carries_or_refuses_every_fact(
    surface: str, planted: dict[str, tl.Trace], tmp_path: Path
) -> None:
    """Each surface carries every fact, or refuses it explicitly (and truly lacks it)."""

    facts = _enumerated_facts(planted)
    carried_by_family = {
        family: EXTRACTORS[surface](trace, tmp_path / family) for family, trace in planted.items()
    }
    missing: list[str] = []
    stale: list[str] = []
    for key, (family, value) in sorted(facts.items()):
        carried = _carries(surface, carried_by_family[family], key, value)
        refused = (surface, key) in REFUSED
        if refused and carried:
            stale.append(key)
        elif not refused and not carried:
            missing.append(key)
    assert not missing, (
        f"{surface} neither carries nor explicitly refuses honesty fact(s) {missing}"
    )
    assert not stale, (
        f"{surface} now carries fact(s) {stale} listed in REFUSED: drop those refusal rows"
    )


def test_clean_capture_adds_no_banner_to_draw() -> None:
    """The draw banner stays empty on a clean plain capture (plain captions unchanged)."""

    from torchlens._capture_honesty import honesty_banner_lines

    assert honesty_banner_lines(_fresh_trace()) == []
