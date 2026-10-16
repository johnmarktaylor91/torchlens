"""``torchlens_explain`` takes a ``max_tokens`` above its 4,000-token default.

Every other agent tool resolves ``max_tokens`` through the served ceiling, where
a request may lower the limit and a request above it is refused
(``agent_limit_invalid``). ``torchlens_explain`` is the documented exception: its
handler budgets the report itself, so the default is 4,000 tokens and a caller
may ask for more. These tests pin that behaviour so a change to it is deliberate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import torchlens as tl
from tests.test_w051_agent_helpers import save_realistic_artifact
from torchlens.agent import _budgets, call_tool, spec_for
from torchlens.agent._artifacts import load_trace


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_realistic_artifact(tmp_path_factory.mktemp("findfix_explain"))


def test_explain_default_is_the_declared_budget(artifact: Path) -> None:
    declared = spec_for("torchlens_explain").default_limits["max_tokens"]
    assert declared == _budgets.ORIENTATION_MAX_TOKENS == 4_000

    envelope = call_tool("torchlens_explain", {"path": str(artifact)})

    assert envelope["limits"]["max_tokens"] == declared


@pytest.mark.parametrize("requested", [1_000, 10_000])
def test_explain_honours_a_request_below_or_above_the_default(
    artifact: Path, requested: int
) -> None:
    envelope = call_tool("torchlens_explain", {"path": str(artifact), "max_tokens": requested})

    assert envelope["status"] == "ok"
    assert envelope["limits"]["max_tokens"] == requested
    # The handler passes the request through unchanged: the report equals a direct
    # explain over the same loaded trace at the same budget.
    loaded, _plan, _block = load_trace(str(artifact))
    direct = tl.report.explain(loaded, max_tokens=requested)
    assert envelope["data"]["report"] == str(direct)


def test_ceiling_tools_still_refuse_a_request_above_their_default(artifact: Path) -> None:
    """The exception is explain's alone: the overview refuses past its ceiling."""

    ceiling = spec_for("torchlens_overview").default_limits["max_tokens"]
    with pytest.raises(ValueError) as exc:
        call_tool("torchlens_overview", {"path": str(artifact), "max_tokens": ceiling + 1})

    assert exc.value.fields["code"] == "agent_limit_invalid"
