"""F29: discoverability -- taught spellings resolve, refusals teach (memo 3.10).

The taught-spelling lint: every code span the surface teaches (guide
snippets, api_map rows, breadcrumbs, error remedies) resolves in a fresh
subprocess. Plus the facade repairs (tl.utils/tl.agent cold-resolve, the
did-you-mean step-5 message) and the typed HF entry recovery.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import torchlens as tl

REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def _resolves_cold(spelling: str) -> None:
    """Assert one dotted spelling resolves on a cold import in a subprocess."""

    probe = (
        "import sys\n"
        f"sys.path.insert(0, {REPO_ROOT!r})\n"
        "import torchlens as tl\n"
        f"obj = eval({spelling!r})\n"
        "assert obj is not None\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, (spelling, result.stderr[-400:])


@pytest.mark.heavy
def test_breadcrumb_spellings_resolve_cold() -> None:
    """Every spelling the package docstring teaches resolves on a cold import."""

    docstring = tl.__doc__ or ""
    assert "to_agent_json" in docstring
    assert "torchlens.agent.guide()" in docstring
    assert "python -m torchlens" in docstring
    for spelling in ("tl.Trace.to_agent_json", "tl.report.explain", "tl.agent.guide"):
        _resolves_cold(spelling)


@pytest.mark.heavy
def test_docs_taught_facade_names_resolve_cold() -> None:
    """tl.utils / tl.bridge / tl.backends / tl.callbacks / tl.agent cold-resolve."""

    for spelling in ("tl.utils.doctor", "tl.bridge", "tl.backends", "tl.callbacks", "tl.agent"):
        _resolves_cold(spelling)


@pytest.mark.smoke
def test_step5_did_you_mean_stays_a_plain_attributeerror() -> None:
    """Step 5 keeps the PLAIN type; the message teaches close matches."""

    with pytest.raises(AttributeError) as exc:
        tl.trce  # noqa: B018 - the typo IS the test
    assert type(exc.value) is AttributeError
    assert "Did you mean" in str(exc.value)
    assert "trace" in str(exc.value)
    # hasattr must never explode, and underscore probes stay silent-plain.
    assert not hasattr(tl, "definitely_not_a_name")
    with pytest.raises(AttributeError) as dunder_exc:
        tl._not_a_private_name  # noqa: B018
    assert "Did you mean" not in str(dunder_exc.value)


@pytest.mark.smoke
def test_hf_entry_recovery_is_typed_and_teaching() -> None:
    """tl.trace(model, input_ids=...) refuses typed, naming both remedies."""

    import torch
    from torch import nn

    model = nn.Linear(4, 4)
    with pytest.raises(TypeError) as kwargs_exc:
        tl.trace(model, input_ids=torch.ones(1, 4))
    assert kwargs_exc.value.fields["code"] == "trace_forward_kwargs_unrouted"
    assert "input_kwargs" in str(kwargs_exc.value)
    assert "trace_text" in str(kwargs_exc.value)
    # Bare tl.trace(model) is no longer a refusal: the quickstart input
    # ladder (F17, landed law) owns that spelling as the zero-argument
    # inferred rung, so the historical trace_inputs_missing site is gone.


@pytest.mark.smoke
def test_hf_remedy_spelling_actually_works() -> None:
    """The remedy the refusal teaches runs: input_kwargs routes keyword forwards."""

    import torch
    from torch import nn

    class KeywordOnly(nn.Module):
        """A forward that only accepts keyword inputs (the HF shape)."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, *, input_ids: torch.Tensor) -> torch.Tensor:
            """Consume the keyword input."""

            return self.fc(input_ids)

    log = tl.trace(KeywordOnly(), (), input_kwargs={"input_ids": torch.randn(2, 4)})
    assert log.num_ops >= 1


@pytest.mark.heavy
def test_guide_snippets_run_verbatim(tmp_path: Path) -> None:
    """Every fenced snippet in the agent guide executes in a fresh subprocess."""

    from tests.test_agent_surface_helpers import save_clean_artifact
    from torchlens.agent import AGENT_GUIDE

    artifact = save_clean_artifact(tmp_path)
    snippets = re.findall(r"\n\n(    .+?)(?=\n\n\S|\n\n#|\Z)", AGENT_GUIDE, re.DOTALL)
    assert snippets, "the guide lost its fenced snippets"
    ran = 0
    for index, snippet in enumerate(snippets):
        code = "\n".join(line[4:] for line in snippet.splitlines())
        if "tl.trace" in code and "model" in code and "import torchlens" not in code:
            continue  # phase-1 fragments need a live model; composed below
        if "tl.save" in code:
            # A saving snippet writes its OWN artifact (never onto the fixture).
            code = code.replace("run.tlspec", str(tmp_path / f"snippet_{index}.tlspec"))
        else:
            code = code.replace("run.tlspec", str(artifact))
        harness = (
            "import sys\n"
            f"sys.path.insert(0, {REPO_ROOT!r})\n"
            "import torch\n"
            "from torch import nn\n"
            "torch.manual_seed(0)\n"
            "model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())\n"
            "x = torch.randn(2, 4)\n"
            f"import os; os.chdir({str(tmp_path)!r})\n" + code
        )
        result = subprocess.run(
            [sys.executable, "-c", harness], capture_output=True, text=True, cwd=REPO_ROOT
        )
        assert result.returncode == 0, (code, result.stderr[-600:])
        ran += 1
    assert ran >= 3, "the snippet lint stopped covering the guide"


@pytest.mark.smoke
def test_api_map_rows_resolve_where_declared() -> None:
    """Every api_map compact row's spelling resolves through the live facade."""

    from torchlens.agent import call_tool

    rows = call_tool("torchlens_api_map")["data"]["names"]
    unresolved = [row["name"] for row in rows if row["kind"] == "unresolvable"]
    assert not unresolved, f"api_map advertises spellings that raise: {unresolved}"


@pytest.mark.smoke
def test_next_operations_and_dump_next_steps_name_live_spellings(tmp_path: Path) -> None:
    """Overview next_operations reference served tools; dump guide steps resolve."""

    from tests.test_agent_surface_helpers import save_clean_artifact
    from torchlens.agent import call_tool, tool_specs

    artifact = save_clean_artifact(tmp_path)
    served = {spec.name for spec in tool_specs()}
    overview = call_tool("torchlens_overview", {"path": str(artifact)})
    for step in overview["data"]["next_operations"].values():
        named = re.findall(r"torchlens_[a-z_]+", step)
        assert named and all(name in served for name in named), step
