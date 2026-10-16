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
    NORMALIZE_RULES,
    SlideRender,
    compare,
    normalize,
    review,
    rule_reasons,
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


def test_normalization_keeps_every_real_difference() -> None:
    """Renumbered autograd ids keep topology, and a changed colour or count is never hidden."""

    one, _ = normalize("grad_fn_1 -> grad_fn_2; grad_fn_2 -> grad_fn_3")
    two, _ = normalize("grad_fn_7 -> grad_fn_9; grad_fn_9 -> grad_fn_8")
    rewired, _ = normalize("grad_fn_7 -> grad_fn_9; grad_fn_7 -> grad_fn_8")
    assert one == two
    assert one != rewired
    assert normalize('x [fillcolor="#FF0000"]')[0] != normalize('x [fillcolor="#00FF00"]')[0]
    assert normalize("(1, 3), 12 B")[0] != normalize("(1, 3), 16 B")[0]
    assert normalize("t=53.41 us")[0] == normalize("t=8.2 ms")[0]


def test_each_rule_names_why_it_differs_by_construction() -> None:
    """Every ignored difference carries its reason, so the PDF cover can state it."""

    assert all(name and why for name, why, _rule in NORMALIZE_RULES)
    _text, timed = normalize(
        "/build/side t=5 us (54% of total) at 0x7f00deadbeef footer "
        "commit abcdef1, built 2026-10-16 grad_fn_9",
        roots=("/build/side",),
    )
    assert set(timed) <= set(rule_reasons())
    assert set(timed) == set(rule_reasons())
    text, hits = normalize("/build/side/a.py at 0x7f00deadbeef", roots=("/build/side",))
    assert text == "<tree>/a.py at 0x<addr>"
    assert hits == {"tree paths": 1, "object addresses": 1}


def test_time_shares_count_as_noise_only_beside_measured_times() -> None:
    """A share of measured time is noise; a share with no timing beside it is a real number."""

    assert normalize("t=5 us (54% of total)")[0] == normalize("t=7 us (51% of total)")[0]
    assert normalize("(54% of total)")[0] != normalize("(51% of total)")[0]


def test_a_slide_failing_on_both_sides_is_named_failed() -> None:
    """Two failed builds of one slide compare equal, so the failure is reported on its own."""

    failed = {"x": SlideRender("x", status="failed")}
    result = compare((["x"], failed), (["x"], {"x": SlideRender("x", status="failed")}))
    assert result["unchanged"] == ["x"]
    assert result["failed"] == {"x": {"base": "failed", "head": "failed"}}
