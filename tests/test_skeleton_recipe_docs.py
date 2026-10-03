"""Realism gate for the skeleton-change recipe page (workstream SKELETON-RECIPE).

``docs/skeleton_change_recipe.md`` is the documented, tested capability that
survived the dropped ``tl.edited`` verb (foldA D13/item 13): the user edits a
model's skeleton in their OWN code, traces the edited model, and re-aligns
saved analyses through C03's site-key-first compatibility checker. This suite
EXECUTES the page's python fences top to bottom, in one shared namespace, on
the real ``transformers`` GPT-2 classes (config-built, zero network), then
asserts the verdicts and diffs the page prints -- so the page cannot rot
silently. The page is DOC_FENCE_EXEMPT in ``test_docs_snippets.py`` because
its fences are one sequential program, not independently runnable blocks;
this suite is the named executor.
"""

from __future__ import annotations

import linecache
import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("transformers")

import torchlens as tl
from torchlens.intervention.errors import GraphShapeMismatchError

RECIPE_PAGE = Path(__file__).resolve().parents[1] / "docs" / "skeleton_change_recipe.md"
PYTHON_FENCE_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)

#: The page's fence roles, in order. A structural rewrite of the page that
#: adds or removes a program step must update this roster consciously.
EXPECTED_FENCE_COUNT = 7

LM_HEAD_SITE_KEY = "s1|lm_head|linear||1"
ADAPTER_LINEAR_SITE_KEY = "s1|transformer/transformer.h.1/transformer.h.1.adapter|linear||1"


def _page_fences() -> list[str]:
    """Extract the recipe page's python fences in document order.

    Returns
    -------
    list[str]
        Code bodies of every ```python fence on the page.
    """

    text = RECIPE_PAGE.read_text(encoding="utf-8")
    return [match.group("code") for match in PYTHON_FENCE_RE.finditer(text)]


def _run_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Execute every page fence sequentially in one namespace.

    Parameters
    ----------
    tmp_path:
        Scratch directory the page's relative ``.tlspec`` saves land in.
    monkeypatch:
        Fixture used to chdir into ``tmp_path`` for the run.

    Returns
    -------
    dict[str, Any]
        The shared namespace after the final fence.
    """

    fences = _page_fences()
    assert len(fences) == EXPECTED_FENCE_COUNT, (
        f"docs/skeleton_change_recipe.md carries {len(fences)} python fences, "
        f"expected {EXPECTED_FENCE_COUNT}; the page's program changed shape -- "
        "update this suite's roster and assertions together"
    )
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {"__name__": "docs_skeleton_change_recipe"}
    for index, code in enumerate(fences, start=1):
        synthetic = f"docs/skeleton_change_recipe.md:python-block-{index}"
        linecache.cache[synthetic] = (
            len(code),
            None,
            [f"{line}\n" for line in code.splitlines()],
            synthetic,
        )
        exec(compile(code, synthetic, "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture(scope="module")
def recipe_namespace(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Run the whole page once and share the final namespace across asserts.

    Returns
    -------
    dict[str, Any]
        Namespace produced by :func:`_run_page`.
    """

    monkeypatch = pytest.MonkeyPatch()
    try:
        return _run_page(tmp_path_factory.mktemp("skeleton_recipe"), monkeypatch)
    finally:
        monkeypatch.undo()


@pytest.mark.heavy
@pytest.mark.real_model
def test_surviving_target_aligns_with_label_drift_disclosure(
    recipe_namespace: dict[str, Any],
) -> None:
    """The lm_head recipe re-aligns: confirmation verdict, key matched, labels drifted."""

    compat = recipe_namespace["compat"]
    assert compat.outcome == "COMPATIBLE_WITH_CONFIRMATION"
    assert compat.targets_resolve_identically is True
    assert f"matched   {LM_HEAD_SITE_KEY}" in compat.site_diff
    assert compat.diff.missing_site_keys == []
    assert compat.diff.new_site_keys == []

    drift_rows = recipe_namespace["drift_rows"]
    assert drift_rows, "the skeleton edit must renumber the lm_head label"
    for row in drift_rows:
        assert row["saved_labels"] != row["resolved_labels"]
        assert sorted(row["saved_site_keys"]) == sorted(row["resolved_site_keys"])
        assert row["resolved_site_keys"] == [LM_HEAD_SITE_KEY]


@pytest.mark.heavy
@pytest.mark.real_model
def test_moved_target_refuses_typed(recipe_namespace: dict[str, Any]) -> None:
    """A recipe whose target vanished from the edited graph refuses typed."""

    refusal = recipe_namespace["refusal"]
    assert isinstance(refusal, GraphShapeMismatchError)
    assert "graph_shape_hash" in str(refusal)


@pytest.mark.heavy
@pytest.mark.real_model
def test_site_key_diff_localizes_the_edit(recipe_namespace: dict[str, Any]) -> None:
    """Every moved address sits inside the wrapped block; outside sites held."""

    moved_out = recipe_namespace["moved_out"]
    moved_in = recipe_namespace["moved_in"]
    assert moved_out and moved_in
    for key in moved_out:
        assert "/transformer.h.1" in key, f"address outside the edited block moved: {key}"
    for key in moved_in:
        assert "/transformer.h.1" in key, f"new address outside the edited block: {key}"
    assert ADAPTER_LINEAR_SITE_KEY in moved_in
    assert LM_HEAD_SITE_KEY not in moved_out


@pytest.mark.heavy
@pytest.mark.real_model
def test_reauthored_recipe_checks_exact(recipe_namespace: dict[str, Any]) -> None:
    """Re-authored against the edited capture, the recipe verdict is EXACT."""

    compat_v2 = recipe_namespace["compat_v2"]
    assert compat_v2.outcome == "EXACT"
    assert compat_v2.targets_resolve_identically is True


@pytest.mark.smoke
def test_page_states_no_torchlens_verb_edits_the_model() -> None:
    """The page keeps its core honesty sentence and its executable-page pledge."""

    text = RECIPE_PAGE.read_text(encoding="utf-8")
    assert "no TorchLens verb edits your model" in text
    assert "tests/test_skeleton_recipe_docs.py" in text
    # The authored spellings the fences depend on; a rename must retouch the page.
    for spelling in (
        "tl.intervention.site(",
        "attach_hooks(",
        "tl.io.save_intervention(",
        "tl.io.load_intervention_spec(",
        "tl.validation.check_spec_compat(",
        "site_key",
    ):
        assert spelling in text, f"recipe page lost its {spelling!r} step"
    assert isinstance(tl.__all__, list)
