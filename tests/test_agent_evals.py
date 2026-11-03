"""Offline mode of the agent-from-docs harness (``tools/agent_evals/``).

Every task's reference solution was written from the shipped docs alone. Executing it against
the current code and grading it with the task's checks catches docs and API drift with no model
call. Negative controls prove each grader fails a script that does nothing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from agent_evals import harness  # noqa: E402
from agent_evals.tasks import TASKS, TASKS_BY_ID  # noqa: E402

TASK_IDS = [task.task_id for task in TASKS]


def _failures(result: dict) -> list[str]:
    return [f"{c['name']}: {c['detail']}" for c in result["checks"] if not c["passed"]]


@pytest.mark.heavy  # one subprocess per task, 3-6 s each, about 25 s per family (measured)
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reference_solution_passes_its_checks(task_id: str, tmp_path: Path) -> None:
    """The docs-only reference solution runs on the current code and passes every check."""

    (result,) = harness.run_offline([TASKS_BY_ID[task_id]], tmp_path)
    assert result["passed"], _failures(result)
    assert len(result["checks"]) >= 3


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_grader_fails_a_script_that_does_nothing(task_id: str, tmp_path: Path) -> None:
    """Negative control: an exit-0 script with no outputs fails the task's checks."""

    task = TASKS_BY_ID[task_id]
    task_dir = harness.prepare_task(task, tmp_path / task_id, tmp_path / "docs")
    (task_dir / "solution.py").write_text("print('nothing')\n", encoding="utf-8")
    result = harness.grade_task(task, task_dir)
    assert result["checks"][0] == {"name": "script exits 0", "passed": True, "detail": ""}
    assert not result["passed"]


def test_reference_solutions_cite_docs_that_exist() -> None:
    """Each reference names the doc pages it was written from, and those pages exist."""

    for task_id in TASK_IDS:
        source = (harness.REFERENCE_DIR / f"{task_id}.py").read_text(encoding="utf-8")
        cited = re.findall(r"(docs/[\w/.-]+\.md)", source)
        assert cited, f"{task_id} cites no docs page"
        missing = [page for page in cited if not (REPO_ROOT / page).is_file()]
        assert not missing, f"{task_id} cites missing pages {missing}"


def test_docs_bundle_is_markdown_only(tmp_path: Path) -> None:
    """The live-mode bundle carries the docs and the agent guide, never source files."""

    bundle = harness.build_docs_bundle(tmp_path / "docs")
    files = [path for path in bundle.rglob("*") if path.is_file()]
    assert {path.suffix for path in files} == {".md"}
    assert (bundle / "agent_guide.md").read_text(encoding="utf-8").strip()
    assert (bundle / "docs" / "for-ai-agents.md").is_file()


def test_prompt_names_docs_dir_and_solution_file() -> None:
    """Every prompt states the docs directory, the fixture import and the output file."""

    for task in TASKS:
        prompt = task.full_prompt("/bundle/docs")
        assert "/bundle/docs" in prompt
        assert "from model import build_model, make_input" in prompt
        assert "solution.py" in prompt


def test_unknown_task_id_is_refused() -> None:
    with pytest.raises(SystemExit, match="unknown task ids"):
        harness.select_tasks("capture_inspect,not_a_task")
