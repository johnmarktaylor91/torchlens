"""F29: provoking tests for the agent refusal codes the other suites skip.

Every shipped code must be provoked by a test (the error-code coverage
gate): ``agent_artifact_kind_unsupported``, ``agent_name_unknown``, and
``agent_view_invalid`` land here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import torchlens as tl
from tests.test_agent_surface_helpers import save_clean_artifact
from torchlens.agent import _artifacts, call_tool


@pytest.fixture()
def clean(tmp_path: Path) -> Path:
    """The deterministic clean fixture artifact."""

    return save_clean_artifact(tmp_path)


def test_non_trace_artifact_refuses_kind_typed(
    clean: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-Trace load refuses agent_artifact_kind_unsupported, teaching tl.load.

    Real Bundle/InterventionSpec artifacts come from their own save doors; the
    agent seam's contract is the isinstance gate itself, provoked here by
    substituting the loader's return value.
    """

    _artifacts._TRACE_CACHE.clear()

    class NotATrace:
        """Stand-in for a Bundle/InterventionSpec load result."""

    monkeypatch.setattr(tl, "load", lambda *args, **kwargs: NotATrace())
    with pytest.raises(ValueError, match="not a Trace") as exc:
        call_tool("torchlens_overview", {"path": str(clean)})
    assert exc.value.fields["code"] == "agent_artifact_kind_unsupported"
    assert "tl.load" in exc.value.fields["remedy"]


def test_api_map_detail_unknown_name_refuses_typed() -> None:
    """Detail mode on a spelling off the surface refuses agent_name_unknown."""

    with pytest.raises(ValueError, match="not a torchlens root surface name") as exc:
        call_tool("torchlens_api_map", {"name": "definitely_not_a_surface_name"})
    assert exc.value.fields["code"] == "agent_name_unknown"


def test_dump_unknown_class_id_refuses_typed(clean: Path) -> None:
    """Fold drill-down on a bogus class id refuses agent_view_invalid."""

    with pytest.raises(ValueError, match="names no fold class") as exc:
        call_tool("torchlens_dump", {"path": str(clean), "view": "graph", "class_id": "c9999"})
    assert exc.value.fields["code"] == "agent_view_invalid"


def test_dump_view_gate_refuses_outside_the_vocabulary(clean: Path) -> None:
    """The dump_view seam's own vocabulary check refuses agent_view_invalid.

    The registry enum catches bad views on the tool path; the seam keeps its
    own fail-closed check for direct in-process callers.
    """

    from torchlens.agent._overview import dump_view

    log, _, _ = _artifacts.load_trace(str(clean))
    with pytest.raises(ValueError, match="not a dump view") as exc:
        dump_view(log, view="prose", max_rows=10)
    assert exc.value.fields["code"] == "agent_view_invalid"
