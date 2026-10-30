"""Task-card schema (``tools/docs/task_card.py``) on fixture cards.

Real cards arrive with the Field guide pages. Until then these fixtures pin the
contract the docs generator and the agent index rely on: the required fields, the
closed ``tier`` and ``doors`` vocabularies, one-line ``canonical`` calls, unique
tasks, and loading from page front matter, YAML files and folders.

The check that each card's ``canonical`` executes and resolves to a decided public
name is not here: it needs the decided names and plugs into ``load_cards`` through
its ``resolve_canonical`` hook (exercised below with a stub).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "docs" / "task_card.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_task_card_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


task_card = _load_module()

VALID_CARD = """\
task: Extract features from a dataset
one_liner: Run a model over a dataset and keep chosen activations, streamed to disk.
canonical: tl.extract_dataset(model, loader, save=tl.module("layer4"))
avoid: "Calling tl.trace once per batch in a loop: slower and keeps every operation."
tier: core
doors: [neuroai, vision]
related: [select-sites, save-share]
"""


def _page(card: str, body: str = "# Extract features\n\nText.\n") -> str:
    return f"---\n{card}---\n\n{body}"


def test_valid_page_front_matter_loads(tmp_path: Path) -> None:
    page = tmp_path / "extract.md"
    page.write_text(_page(VALID_CARD), encoding="utf-8")
    (card,) = task_card.load_cards(page)
    assert card.task == "Extract features from a dataset"
    assert card.tier == "core"
    assert card.doors == ("neuroai", "vision")
    assert card.related == ("select-sites", "save-share")
    assert card.tags == ()


def test_yaml_list_and_folder_load(tmp_path: Path) -> None:
    second = VALID_CARD.replace("Extract features from a dataset", "Intervene").replace(
        "tier: core", "tier: kit"
    )
    indented = [
        "- " + "\n  ".join(card.rstrip("\n").splitlines()) + "\n" for card in (VALID_CARD, second)
    ]
    (tmp_path / "cards.yaml").write_text("".join(indented), encoding="utf-8")
    sub = tmp_path / "guide"
    sub.mkdir()
    third = (
        VALID_CARD.replace("Extract features from a dataset", "Draw a model") + "tags: [graph]\n"
    )
    (sub / "draw.md").write_text(_page(third), encoding="utf-8")
    (sub / "notes.txt").write_text("not a card", encoding="utf-8")
    cards = task_card.load_cards(tmp_path)
    assert [c.task for c in cards] == [
        "Extract features from a dataset",
        "Intervene",
        "Draw a model",
    ]
    assert cards[2].tags == ("graph",)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.replace("tier: core", "tier: basic"), "tier 'basic' is not one of"),
        (lambda c: c.replace("[neuroai, vision]", "[neuro]"), "doors has unknown values"),
        (lambda c: c.replace("doors: [neuroai, vision]\n", ""), "doors is required"),
        (lambda c: c.replace("related: [select-sites, save-share]\n", ""), "related is required"),
        (
            lambda c: c.replace("related: [select-sites, save-share]", "related: x"),
            "must be a list",
        ),
        (lambda c: c.replace("[neuroai, vision]", "[vision, vision]"), "duplicate entries"),
        (lambda c: c.replace('avoid: "Calling', 'avoid: ""\nold: "Calling'), "unknown field 'old'"),
        (lambda c: c.replace("one_liner: Run", "one_liner: ''\nx_: Run"), "one_liner must be"),
        (lambda c: c.replace("canonical: tl", "canonical: |\n  a()\n  tl"), "one call on one line"),
        (lambda c: c + "tags: [ok, '']\n", "tags[1] must be a non-empty string"),
    ],
)
def test_schema_violations_are_named(tmp_path: Path, change, message: str) -> None:
    page = tmp_path / "bad.md"
    page.write_text(_page(change(VALID_CARD)), encoding="utf-8")
    with pytest.raises(task_card.TaskCardError, match=message):
        task_card.load_cards(page)


def test_every_problem_is_reported_at_once(tmp_path: Path) -> None:
    page = tmp_path / "bad.md"
    page.write_text(_page("task: X\ntier: core\n"), encoding="utf-8")
    with pytest.raises(task_card.TaskCardError) as info:
        task_card.load_cards(page)
    text = str(info.value)
    for name in ("one_liner", "canonical", "avoid", "doors", "related"):
        assert name in text


def test_page_without_front_matter_is_refused(tmp_path: Path) -> None:
    page = tmp_path / "plain.md"
    page.write_text("# No card\n", encoding="utf-8")
    with pytest.raises(task_card.TaskCardError, match="no front matter"):
        task_card.load_cards(page)
    page.write_text("---\ntask: X\n", encoding="utf-8")
    with pytest.raises(task_card.TaskCardError, match="not closed"):
        task_card.load_cards(page)


def test_duplicate_tasks_and_empty_folders_are_refused(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(_page(VALID_CARD), encoding="utf-8")
    (tmp_path / "b.md").write_text(_page(VALID_CARD), encoding="utf-8")
    with pytest.raises(task_card.TaskCardError, match="also in"):
        task_card.load_cards(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(task_card.TaskCardError, match="no task cards found"):
        task_card.load_cards(empty)


def test_resolve_canonical_hook_is_applied(tmp_path: Path) -> None:
    page = tmp_path / "extract.md"
    page.write_text(_page(VALID_CARD), encoding="utf-8")
    seen: list[str] = []

    def refuse(call: str) -> str | None:
        seen.append(call)
        return "tl.extract_dataset is not a decided name"

    with pytest.raises(task_card.TaskCardError, match="canonical does not resolve"):
        task_card.load_cards(page, resolve_canonical=refuse)
    assert seen == ['tl.extract_dataset(model, loader, save=tl.module("layer4"))']
    assert task_card.load_cards(page, resolve_canonical=lambda call: None)


def test_vocabularies_match_the_contract() -> None:
    assert task_card.TIERS == ("core", "kit", "advanced")
    assert task_card.DOORS == ("mechinterp", "neuroai", "vision", "agents")
    assert task_card.SECTION_LABEL == "Field guide"
