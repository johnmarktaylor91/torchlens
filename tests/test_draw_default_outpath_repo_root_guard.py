"""Regression guard: a default-outpath draw() call must never write into the repo root.

xdist race (nightly fast tier, 2026-10-03): ``Trace.draw(vis_save_only=True)``
with no explicit ``vis_outpath=`` resolves the literal default basename
(``"modelgraph"``) against the process's current working directory. Several
xdist workers sharing the repo root as cwd then wrote and read the same file
at once ("dot: can't open .../modelgraph"). About 21 test files call
``draw(vis_save_only=True)`` this way.

The fix is the autouse ``_draw_default_outpath_in_tmp_path`` fixture in
``tests/conftest.py``, which redirects ``draw``'s (and ``draw_backward`` /
``draw_combined`` / ``render_dagua_graph``'s) default output path into the
test's own ``tmp_path``. This test exercises the real default codepath --
the same call shape the exposed files use -- and fails if that fixture ever
regresses (is removed, narrowed, or stops covering ``draw``) and lets a
render land outside ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import example_models
import torch

import torchlens as tl

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_draw_default_outpath_never_lands_in_repo_root(tmp_path: Path) -> None:
    """``draw(vis_save_only=True)`` with no ``vis_outpath`` must render under tmp_path."""

    trace = tl.trace(example_models.SimpleFF(), torch.randn(3, 4))
    stray_candidates = [
        REPO_ROOT / "modelgraph",
        REPO_ROOT / "modelgraph.pdf",
        REPO_ROOT / "modelgraph.dot",
    ]
    try:
        trace.draw(vis_save_only=True)

        for candidate in stray_candidates:
            assert not candidate.exists(), (
                f"draw() with no vis_outpath wrote {candidate} into the repo root "
                "instead of the per-test tmp_path -- the "
                "_draw_default_outpath_in_tmp_path conftest fixture regressed."
            )

        rendered = tmp_path / "modelgraph.pdf"
        assert rendered.exists(), (
            f"draw()'s redirected default output was not found at {rendered}; "
            "the conftest fixture's default-path rewrite did not take effect."
        )
    finally:
        for candidate in stray_candidates:
            candidate.unlink(missing_ok=True)
