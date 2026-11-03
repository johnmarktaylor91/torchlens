"""F09 CP3: sumfam items 11, 13-16, 19-20 -- the family re-base, the
SurfaceRegistry, the capability card, PartialTrace typed refusals, and
the family-gate composition rows.

Named rows exercised: D22 (executable-spelling CI: every registry
spelling runs on a real fixture in its declared state), D24 (the
capability card is metadata-only), D25 (typed refusal, never
AttributeError, on the five partial members), D17/item 14 (explain-json
x agent_json overlapping-field parity), D4 (work-metric: no
metadata-only member triggers a payload scan), D3 (count vocabulary:
one value per grain across surfaces), and composition row 10
(non-mutation).
"""

from __future__ import annotations

import pathlib

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.errors import TraceCaptureError
from torchlens.report._factcore import factcore
from torchlens.report._registry import (
    SURFACE_REGISTRY,
    capability_card,
    registry_entry,
    which_do_i_use,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class FamilyModel(nn.Module):
    """Small nested model shared by the family gate."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 12), nn.ReLU())
        self.head = nn.Linear(12, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain nested forward."""

        return self.head(self.encoder(x))


@pytest.fixture(scope="module")
def family_trace():
    """One shared finished capture for the registry CI."""

    trace = tl.trace(FamilyModel().eval(), torch.randn(2, 6))
    yield trace
    trace.cleanup()


@pytest.fixture()
def partial_trace():
    """A real failed capture recovered as a PartialTrace."""

    class Boom(nn.Module):
        """Fails after one real op."""

        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one op then fail."""

            _ = self.a(x)
            raise RuntimeError("planted forward failure")

    with pytest.warns(Warning):
        try:
            tl.trace(Boom(), torch.randn(2, 4))
        except TraceCaptureError as exc:
            assert isinstance(exc.__cause__, RuntimeError)
            return exc.partial_trace
    raise AssertionError("the planted failure did not raise")


# ---------------------------------------------------------------------------
# SurfaceRegistry (D22) + executable-spelling CI


def test_registry_unknown_key_refuses_typed() -> None:
    """Lookups teach the roster."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        registry_entry("nonexistent_surface")
    assert excinfo.value.fields["code"] == "surface_registry_unknown"


@pytest.mark.heavy  # executes every registry spelling; crossed the 7s smoke budget at T58
def test_every_registry_spelling_executes(family_trace) -> None:
    """D22 executable-spelling CI: the hand-written map rotted; this one
    cannot -- every example runs on a real fixture in its declared state."""

    model = FamilyModel().eval()
    x = torch.randn(2, 6)
    namespace = {"tl": tl, "trace": family_trace, "model": model, "x": x}
    for entry in SURFACE_REGISTRY:
        result = eval(entry.example, namespace)  # noqa: S307 -- registry-owned strings
        assert result is not None, entry.key


def test_which_do_i_use_docs_page_is_lockstepped() -> None:
    """Item 20: the docs page IS the generator output (no rot possible)."""

    page = REPO_ROOT / "docs" / "reference" / "report_family.md"
    assert page.read_text(encoding="utf-8") == which_do_i_use()


# ---------------------------------------------------------------------------
# Capability card (D24)


def test_capability_card_is_metadata_only(family_trace, monkeypatch) -> None:
    """D24: building the card never scans, captures, or runs a forward."""

    import torchlens._data_classes._nonfinite as nonfinite_module

    def _no_scan(*_args, **_kwargs):
        raise AssertionError("capability_card triggered a payload scan")

    monkeypatch.setattr(nonfinite_module, "_scan", _no_scan)
    card = family_trace.capabilities()
    text = str(card)
    assert "capability card" in text
    assert any(row.is_available for row in card.rows)


def test_capability_card_on_partial_names_remedies(partial_trace) -> None:
    """D24 x D25: the card tells a partial-capture user where to go."""

    card = capability_card(partial_trace)
    assert card.object_kind == "partial_trace"
    refused = {row.key: row for row in card.rows if not row.is_available}
    assert "summary" in refused
    assert "explain" not in refused  # explain degrades, never refuses
    assert "audit" in str(refused["summary"].reason) or "explain" in str(refused["summary"].reason)


# ---------------------------------------------------------------------------
# PartialTrace typed refusals (D25 / item 15)


def test_partial_members_refuse_typed_never_attributeerror(partial_trace) -> None:
    """D25: the five bare AttributeErrors are gone; refusals teach."""

    for member in ("summary", "profile", "to_pandas", "to_agent_json", "output_table"):
        with pytest.raises(InvalidArgumentError) as excinfo:
            getattr(partial_trace, member)()
        assert excinfo.value.fields["code"] == "partial_trace_member_unavailable"
        assert "explain" in str(excinfo.value)
        assert "audit" in str(excinfo.value)


def test_partial_explain_and_audit_still_answer(partial_trace) -> None:
    """The two remedies the refusal names actually work."""

    text = partial_trace.explain()
    assert "partial capture" in text
    audit = partial_trace.audit()
    assert audit is not None


# ---------------------------------------------------------------------------
# Machine-core parity (item 14 / D17)


def test_agent_json_is_the_one_machine_projection(family_trace) -> None:
    """One machine core (D17; defaults.7): explain is prose only, and the
    agent JSON counts read the same FactCore the summary family reads (the
    69/151 bug class regression gate)."""

    import inspect

    assert "format" not in inspect.signature(tl.Trace.explain).parameters
    core = factcore(family_trace)
    agent_json = family_trace.to_agent_json()
    assert agent_json["counts"]["layers"] == core.counts.layers
    assert agent_json["counts"]["operations"] == core.counts.tracked_tensor_rows
    assert agent_json["counts"]["tensors_saved"] == core.memory.captured_saved_ops
    assert agent_json["capture"]["model_class"] == family_trace.source.class_name


# ---------------------------------------------------------------------------
# Family gate rows (item 19)
