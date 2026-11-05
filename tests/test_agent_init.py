"""``torchlens agent init``: the marked agent-index block in a project's instruction files.

The command must never touch text outside its own markers, must refuse a file whose markers
are damaged instead of guessing, and must stay torch-free (it is a tier-0 CLI verb; the purity
probe in ``tests/test_agent_surface_cli.py`` runs it in a fresh interpreter).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torchlens.agent import _cli as cli, _init_block as init_block

BLOCK = init_block.render_block(Path("/site/torchlens/_docs/AGENT_INDEX.md"), "9.9.9")


def test_index_path_names_the_shipped_index() -> None:
    """The block points at the generated index inside the package itself."""

    path = init_block.index_path()
    assert path.parts[-3:] == ("torchlens", "_docs", "AGENT_INDEX.md")
    assert path.is_file()


def test_block_names_index_and_version_between_markers() -> None:
    """The block opens and closes with the markers and states the installed version."""

    lines = BLOCK.splitlines()
    assert lines[0] == init_block.BLOCK_START and lines[-1] == init_block.BLOCK_END
    assert "`/site/torchlens/_docs/AGENT_INDEX.md`" in BLOCK
    assert "torchlens 9.9.9" in BLOCK
    unknown = init_block.render_block(Path("/i.md"), "unknown")
    assert "unknown" not in unknown


def test_create_then_rerun_is_unchanged(tmp_path: Path) -> None:
    """A missing file is created; running again changes nothing."""

    target = tmp_path / "AGENTS.md"
    assert init_block.install_block(target, BLOCK).action == "created"
    assert target.read_text(encoding="utf-8") == BLOCK
    assert init_block.install_block(target, BLOCK).action == "unchanged"


def test_append_keeps_existing_text(tmp_path: Path) -> None:
    """An existing instruction file keeps its content; the block goes after a blank line."""

    target = tmp_path / "AGENTS.md"
    target.write_text("# My project\n\nUse uv.\n", encoding="utf-8")
    assert init_block.install_block(target, BLOCK).action == "added"
    assert target.read_text(encoding="utf-8") == "# My project\n\nUse uv.\n\n" + BLOCK
    no_newline = tmp_path / "CLAUDE.md"
    no_newline.write_text("@AGENTS.md", encoding="utf-8")
    init_block.install_block(no_newline, BLOCK)
    assert no_newline.read_text(encoding="utf-8") == "@AGENTS.md\n\n" + BLOCK


def test_update_replaces_only_the_block(tmp_path: Path) -> None:
    """An older block (another version or path) is replaced; text around it is kept."""

    old = init_block.render_block(Path("/old/AGENT_INDEX.md"), "1.0.0")
    target = tmp_path / "AGENTS.md"
    target.write_text("before\n\n" + old + "\nafter\n", encoding="utf-8")
    assert init_block.install_block(target, BLOCK).action == "updated"
    assert target.read_text(encoding="utf-8") == "before\n\n" + BLOCK + "\nafter\n"


def test_remove_keeps_other_text_and_deletes_a_block_only_file(tmp_path: Path) -> None:
    """``--remove`` takes out exactly the block; a file left empty is deleted."""

    shared = tmp_path / "AGENTS.md"
    shared.write_text("# Notes\n\n" + BLOCK + "\nTail.\n", encoding="utf-8")
    assert init_block.remove_block(shared).action == "removed"
    assert shared.read_text(encoding="utf-8") == "# Notes\n\nTail.\n"
    assert init_block.remove_block(shared).action == "absent"
    alone = tmp_path / "ONLY.md"
    alone.write_text(BLOCK, encoding="utf-8")
    assert init_block.remove_block(alone).action == "deleted"
    assert not alone.exists()
    assert init_block.remove_block(tmp_path / "missing.md").action == "absent"


@pytest.mark.parametrize(
    "text",
    [
        f"{init_block.BLOCK_START}\nno end\n",
        f"no start\n{init_block.BLOCK_END}\n",
        f"{init_block.BLOCK_END}\n{init_block.BLOCK_START}\n",
        BLOCK + BLOCK,
    ],
    ids=["start-only", "end-only", "reversed", "two-blocks"],
)
def test_damaged_markers_are_refused_untouched(tmp_path: Path, text: str) -> None:
    """A file with damaged markers is never rewritten, by init or by --remove."""

    target = tmp_path / "AGENTS.md"
    target.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="markers are damaged"):
        init_block.install_block(target, BLOCK)
    with pytest.raises(ValueError):
        init_block.remove_block(target)
    assert target.read_text(encoding="utf-8") == text


def test_cli_init_defaults_to_agents_md_and_reports_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``torchlens agent init`` writes AGENTS.md here; --file repeats; --remove undoes."""

    monkeypatch.chdir(tmp_path)
    assert cli.main(["agent", "init", "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["files"] == [{"path": "AGENTS.md", "action": "created"}]
    assert payload["data"]["index"] == str(init_block.index_path())
    assert str(init_block.index_path()) in (tmp_path / "AGENTS.md").read_text(encoding="utf-8")

    args = ["agent", "init", "--json", "--file", "CLAUDE.md", "--file", "AGENTS.md"]
    assert cli.main(args) == cli.EXIT_OK
    rows = json.loads(capsys.readouterr().out)["data"]["files"]
    assert [row["action"] for row in rows] == ["created", "unchanged"]

    assert cli.main([*args, "--remove"]) == cli.EXIT_OK
    rows = json.loads(capsys.readouterr().out)["data"]["files"]
    assert [row["action"] for row in rows] == ["deleted", "deleted"]


def test_cli_damaged_markers_exit_as_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refused file exits 4 with the reason on stderr; a bare ``agent`` is a usage error."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text(init_block.BLOCK_START + "\n", encoding="utf-8")
    assert cli.main(["agent", "init"]) == cli.EXIT_REFUSAL
    assert "markers are damaged" in capsys.readouterr().err
    assert cli.main(["agent"]) == cli.EXIT_USAGE
