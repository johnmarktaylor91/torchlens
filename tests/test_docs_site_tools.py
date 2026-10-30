"""The docs site tools: task-card generator, site map, page mover and link checker.

The generator's committed outputs (``tools/docs/generated``) must match a fresh
run over the Field guide cards, and ``mkdocs.yml`` must match the site map; both
are lockstep checks in the spirit of the other generated-docs gates.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]


def _load(name: str) -> ModuleType:
    path = REPO / "tools" / "docs" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_docs_tool_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


index_tool = _load("build_agent_index")
site_nav = _load("site_nav")
move_pages = _load("move_pages")
links = _load("check_site_links")

CARD = """---
task: {task}
one_liner: {task} in one line.
canonical: {call}
avoid: "Calling {call} in a loop."
tier: {tier}
doors: [mechinterp]
related: {related}
---

# {task}

Body of {task}.
"""


def _write_cards(folder: Path) -> Path:
    folder.mkdir()
    (folder / "see.md").write_text(
        CARD.format(task="See a model", call="tl.trace(model, x)", tier="core", related="[edit]")
    )
    (folder / "edit.md").write_text(
        CARD.format(task="Edit a run", call="trace.do(site, edit)", tier="advanced", related="[]")
    )
    (folder / "pick.md").write_text(
        CARD.format(task="Pick | sites", call="tl.select(x)", tier="kit", related="[see]")
    )
    (folder / "plain.md").write_text("# A page with no card\n")
    return folder


def test_generator_renders_cards_in_tier_order(tmp_path: Path) -> None:
    cards = index_tool.load_cards(_write_cards(tmp_path / "cards"))
    assert [card.slug for card in cards] == ["see", "pick", "edit"]
    outputs = index_tool.render_all(cards, "https://example.org/docs/", "9.x")
    index = outputs["AGENT_INDEX.md"]
    core_block = index.split("```python\n", 1)[1].split("```", 1)[0]
    assert core_block == "tl.trace(model, x)\n"
    assert "| Pick \\| sites | `tl.select(x)` | Calling tl.select(x) in a loop. | kit |" in index
    assert "https://example.org/docs/9.x/field-guide/edit/" in index
    assert "- [See a model](https://example.org/docs/9.x/field-guide/see/): " in outputs["llms.txt"]
    full = outputs["llms-full.txt"]
    assert "Body of Edit a run." in full and "tier:" not in full
    assert outputs == index_tool.render_all(cards, "https://example.org/docs/", "9.x")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tier", "core\nextra: 1", "unknown"),
        ("tier", "basic", "tier must be one of"),
        ("doors", "[mechinterp, astronomy]", "unknown doors"),
        ("related", "[nowhere]", "related names no carded page"),
        ("canonical", '""', "non-empty string"),
    ],
)
def test_generator_refuses_malformed_cards(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    folder = _write_cards(tmp_path / "cards")
    page = folder / "see.md"
    lines = [
        f"{field}: {value}" if line.startswith(f"{field}:") else line
        for line in page.read_text().splitlines()
    ]
    page.write_text("\n".join(lines) + "\n")
    with pytest.raises(index_tool.CardError, match=message):
        index_tool.load_cards(folder)


def test_generator_enforces_the_index_size_limit(tmp_path: Path) -> None:
    folder = _write_cards(tmp_path / "cards")
    for number in range(60):
        (folder / f"bulk{number:02d}.md").write_text(
            CARD.format(
                task=f"Task number {number} " + "x" * 80, call="tl.x()", tier="kit", related="[]"
            )
        )
    with pytest.raises(index_tool.CardError, match="limit is 8192"):
        index_tool.render_all(index_tool.load_cards(folder), "https://e.org/", "2.x")


def test_generator_check_mode_detects_drift(tmp_path: Path) -> None:
    folder = _write_cards(tmp_path / "cards")
    out = tmp_path / "out"
    argv = ["--cards", str(folder), "--out", str(out)]
    assert index_tool.main([*argv, "--check"]) == 1
    assert index_tool.main(argv) == 0
    assert index_tool.main([*argv, "--check"]) == 0
    (out / "llms.txt").write_text("edited by hand\n")
    assert index_tool.main([*argv, "--check"]) == 1


def test_committed_agent_index_matches_a_fresh_run() -> None:
    assert index_tool.main(["--check"]) == 0, (
        "tools/docs/generated is stale: run python tools/docs/build_agent_index.py"
    )


def test_site_map_covers_every_page_and_mkdocs_matches_it() -> None:
    site_map = site_nav.load_site_map()
    assert site_nav.coverage_problems(site_map) == []
    current = site_nav.MKDOCS.read_text("utf-8")
    assert site_nav.render_mkdocs(current, site_map) == current, (
        "mkdocs.yml is stale: run python tools/docs/site_nav.py"
    )


def test_site_map_never_publishes_internal_pages(tmp_path: Path) -> None:
    bad = tmp_path / "map.yaml"
    bad.write_text(
        "sections:\n  - {id: internal, title: Internal, internal: true}\n"
        "pages:\n  - {path: a.md, section: internal, publish: true}\n"
    )
    with pytest.raises(ValueError, match="never published"):
        site_nav.load_site_map(bad)


def test_page_mover_retargets_links_when_depth_changes() -> None:
    moves = {"docs/reference/a.md": "docs/internal/reference/a.md"}
    linking = "See [a](reference/a.md#part) and `docs/reference/a.md`."
    assert move_pages.rewrite_text(linking, "docs/index.md", moves) == (
        "See [a](internal/reference/a.md#part) and `docs/internal/reference/a.md`."
    )
    inside = "Back to [the guide](../quickstart.md)."
    assert move_pages.rewrite_text(inside, "docs/internal/reference/a.md", moves) == (
        "Back to [the guide](../../quickstart.md)."
    )
    assert move_pages.rewrite_text('"reference/a.md"', "tests/x.py", moves) == (
        '"internal/reference/a.md"'
    )


def test_link_checker_finds_missing_pages_and_anchors(tmp_path: Path) -> None:
    site = tmp_path / "site"
    (site / "page").mkdir(parents=True)
    (site / "page" / "index.html").write_text('<h2 id="here">x</h2><a href="../#top">up</a>')
    (site / "index.html").write_text(
        '<p id="top"></p><a href="page/#here">ok</a><a href="page/#gone">bad anchor</a>'
        '<a href="missing/">bad page</a><a href="https://example.org/x">external</a>'
        '<img src="/docs/page/index.html">'
    )
    problems = links.check_site(site, base_path="/docs/")
    assert problems == [
        "index.html: page/#gone (no anchor #gone)",
        "index.html: missing/ (no such file)",
    ]
