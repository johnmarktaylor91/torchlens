"""The agent kit generated from the task cards stays fresh, small and true to the code.

``tools/docs/build_agent_index.py`` renders the in-wheel agent index, the skill router, the
capabilities list and ``llms.txt`` from ``docs/field-guide/task_cards.yaml``. These tests fail
when an output is stale, the index outgrows its budget, a card links a missing page, or a card's
canonical call names a TorchLens function or keyword that does not exist.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import torchlens as tl

REPO_ROOT = Path(__file__).resolve().parents[1]


def _generator() -> ModuleType:
    """Import the generator by path (``tools/`` is not a package)."""

    name = "_tl_build_agent_index_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "tools" / "docs" / "build_agent_index.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GEN = _generator()
CARD_SET = GEN.load_card_set()
CARDS = {card.slug: card for card in CARD_SET.cards}


def test_generated_outputs_are_fresh() -> None:
    """Every tracked output matches a fresh render (run the generator after editing cards)."""

    outputs = GEN.render_all(CARD_SET, GEN.DEFAULT_LINK_BASE)
    assert GEN.stale_outputs(REPO_ROOT, outputs) == []


def test_index_fits_its_budget_and_router_matches() -> None:
    """The shipped index stays within 8 KB; the skill router carries the same text."""

    index = (REPO_ROOT / GEN.INDEX_PATH).read_bytes()
    assert len(index) <= GEN.INDEX_LIMIT_BYTES
    assert (REPO_ROOT / GEN.ROUTER_PATH).read_bytes() == index


def test_index_size_limit_is_enforced(tmp_path: Path) -> None:
    """An index over budget is a card error, not a silently large file."""

    long_rules = GEN.CardSet(("x" * GEN.INDEX_LIMIT_BYTES,), CARD_SET.cards, CARD_SET.capabilities)
    with pytest.raises(GEN.CardError, match="limit"):
        GEN.render_all(long_rules, GEN.DEFAULT_LINK_BASE)


@pytest.mark.parametrize("slug", sorted(CARDS))
def test_card_page_exists(slug: str) -> None:
    """Each card links a page that exists in the repository."""

    assert (REPO_ROOT / CARDS[slug].page).is_file(), CARDS[slug].page


def _resolve(dotted: str) -> Any:
    """Resolve ``tl.a.b`` on the installed package."""

    target: Any = tl
    for part in dotted.split(".")[1:]:
        target = getattr(target, part)
    return target


def _tl_calls(source: str) -> list[ast.Call]:
    """Every call in ``source`` whose function is a ``tl.`` attribute chain."""

    calls = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            parts, func = [], node.func
            while isinstance(func, ast.Attribute):
                parts.append(func.attr)
                func = func.value
            if isinstance(func, ast.Name) and func.id == "tl" and parts:
                calls.append(node)
    return calls


def _dotted(call: ast.Call) -> str:
    return ast.unparse(call.func)


@pytest.mark.parametrize("slug", sorted(CARDS))
def test_canonical_names_and_keywords_exist(slug: str) -> None:
    """Each canonical call parses, and every ``tl.`` name and keyword it uses is real."""

    canonical = CARDS[slug].canonical
    if canonical == GEN.NO_CALL:
        pytest.skip("card says TorchLens is the wrong tool")
    for call in _tl_calls(canonical):
        target = _resolve(_dotted(call))
        assert callable(target), _dotted(call)
        try:
            params = inspect.signature(target).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue
        for keyword in call.keywords:
            assert keyword.arg in params, f"{_dotted(call)} has no keyword {keyword.arg!r}"


def test_every_accepted_and_avoided_call_resolves() -> None:
    """``accept`` and ``avoid_calls`` name real TorchLens calls."""

    for card in CARD_SET.cards:
        for dotted in (*card.accept, *card.avoid_calls):
            assert callable(_resolve(dotted)), f"{card.slug}: {dotted}"


def test_capabilities_cover_cards_or_say_why_not() -> None:
    """Each capability name maps to cards or carries a note; the rendered list names them all."""

    rendered = (REPO_ROOT / GEN.CAPABILITIES_PATH).read_text(encoding="utf-8")
    for capability in CARD_SET.capabilities:
        assert capability.cards or capability.note, capability.name
        assert capability.name in rendered


def test_skill_and_plugin_manifests_name_the_skill() -> None:
    """The installable skill has Agent Skills front matter; the plugin manifests parse."""

    skill = (REPO_ROOT / "skills" / "torchlens" / "SKILL.md").read_text(encoding="utf-8")
    meta, _ = GEN.split_front_matter(skill)
    assert meta is not None and meta["name"] == "torchlens" and meta["description"]
    assert "ROUTER.md" in skill and "torchlens agent init" in skill
    plugin = json.loads((REPO_ROOT / ".claude-plugin" / "plugin.json").read_text("utf-8"))
    market = json.loads((REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text("utf-8"))
    assert plugin["name"] == "torchlens"
    assert [entry["name"] for entry in market["plugins"]] == ["torchlens"]
