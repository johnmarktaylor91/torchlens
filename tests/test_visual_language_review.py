"""Review mode of the visual language deck: a branch rendered against its base, compared by DOT.

The end-to-end test builds a two-commit repository whose generator and ``torchlens`` are
stand-ins (the review drives any tree's own generator through its command line), so it
proves the mechanism without drawing the real deck: each ref comes from ``git archive``,
the working tree never moves, the strip is drawn over each side's own ``torchlens``, and a
changed, an unchanged and an added slide are each reported as such.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.visual_language.review import (
    review,
)

GENERATOR = """
import json, sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
import torchlens
command, out = sys.argv[1], Path(sys.argv[3])
spec = json.loads((root / "scripts" / f"{command}.json").read_text())
(out / "raw").mkdir(parents=True, exist_ok=True)
for slide_id, body in spec.items():
    body = body.replace("COLOR", torchlens.COLOR)
    (out / "raw" / f"{slide_id}-a.dot").write_text("digraph { " + body + " }")
(out / "slides.json").write_text(json.dumps([{"id": s, "status": "ok"} for s in spec]))
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, color: str, slides: dict[str, str], message: str) -> str:
    (repo / "torchlens").mkdir(exist_ok=True)
    (repo / "scripts").mkdir(exist_ok=True)
    (repo / "torchlens" / "__init__.py").write_text(f"COLOR = {color!r}\n")
    (repo / "scripts" / "render_visual_language_deck.py").write_text(GENERATOR)
    (repo / "scripts" / "deck.json").write_text(json.dumps(slides))
    strip = {"sentinel-01-anatomy": "n [color=COLOR]", "sentinel-02-plain": "m -> k"}
    (repo / "scripts" / "sentinels.json").write_text(json.dumps(strip))
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def test_review_reports_changed_unchanged_and_added_slides(tmp_path: Path) -> None:
    """A renderer change shows on the slide it touches and on the strip; nothing else moves."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    base_slides = {
        "same": "grad_fn_140112 -> b [label=<t=53.41 us>]",
        "fill": "x [fillcolor=COLOR]",
    }
    base = _commit(repo, "red", base_slides, "base")
    head_slides = {
        # The same graph drawn by another process: a new autograd id and a new timing.
        "same": "grad_fn_977311 -> b [label=<t=61.02 us>]",
        "fill": "x [fillcolor=COLOR]",
        "added": "y -> z",
    }
    head = _commit(repo, "blue", head_slides, "head")
    _git(repo, "checkout", "-q", base)

    result = review(repo, base, head, tmp_path / "out")

    assert result["changed"] == ["fill"]
    assert result["unchanged"] == ["same"]
    assert result["added"] == ["added"]
    assert result["removed"] == []
    assert result["sentinel"]["changed"] == ["sentinel-01-anatomy"]
    assert result["sentinel"]["unchanged"] == ["sentinel-02-plain"]
    # The base strip drew the base's own torchlens, from a tree of its own.
    assert Path(result["base"]["torchlens"]).is_relative_to(tmp_path / "out" / "base")
    assert json.loads((tmp_path / "out" / "changes.json").read_text())["added"] == ["added"]
    assert (tmp_path / "out" / "changes.pdf").read_bytes()[:5] == b"%PDF-"
    # Every difference the comparison ignored is named, with its reason, in the result.
    ignored = {
        rule for side in result["normalized"].values() for hits in side.values() for rule in hits
    }
    assert ignored == {"grad_fn ids", "measured durations"}
    assert ignored <= set(result["normalization"])
    # The working tree stayed where it was.
    assert _git(repo, "rev-parse", "HEAD") == base
    assert _git(repo, "status", "--porcelain") == ""
