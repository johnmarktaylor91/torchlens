"""Tests for the multi_trace visualization layer (show_bundle_graph).

Covers the 15 cases from the Phase 2 spec. Fixtures are inlined rather
than shared via conftest to keep this file self-contained -- it mirrors
the pattern used in ``test_multi_trace.py``.

Where the test only needs to verify that styling differs between modes
we inspect the generated DOT source via the ``return_dot=True`` test
hook on ``show_bundle_graph``. Where the test verifies actual file
output, we render to a temp directory and check size + extension.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.multi_trace import TraceBundle, bundle, show_bundle_graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _LinearNet(nn.Module):
    """Small static model used for shared-topology bundles."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


class _BranchNet(nn.Module):
    """Conditionally branching model used for divergent-topology bundles."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.mean() > 0:
            return torch.relu(x)
        return torch.sigmoid(x)


def _shared_bundle(n_traces: int = 3, batch: int = 2) -> TraceBundle:
    """Build a shared-topology bundle from N forward passes of _LinearNet."""

    model = _LinearNet()
    traces = [tl.log_forward_pass(model, torch.rand(batch, 4)) for _ in range(n_traces)]
    return TraceBundle(traces)


def _divergent_bundle() -> TraceBundle:
    """Build a divergent-topology bundle from a branching model."""

    model = _BranchNet()
    ml_pos = tl.log_forward_pass(model, torch.ones(2, 4))
    ml_neg = tl.log_forward_pass(model, -torch.ones(2, 4))
    return TraceBundle([ml_pos, ml_neg], names=["pos", "neg"])


def _grouped_bundle() -> TraceBundle:
    """Bundle with non-empty groups, suitable for group_color mode."""

    model = _LinearNet()
    traces = [tl.log_forward_pass(model, torch.rand(2, 4)) for _ in range(4)]
    names = ["a1", "a2", "b1", "b2"]
    groups = {"alphas": ["a1", "a2"], "betas": ["b1", "b2"]}
    return TraceBundle(traces, names=names, groups=groups)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_nonempty(path: Path) -> bool:
    """Return True if ``path`` exists and has size > 0."""

    return path.exists() and path.stat().st_size > 0


def _find_rendered(parent: Path, stem: str, ext: str) -> Path:
    """Locate the rendered file by stem/extension under ``parent``."""

    direct = parent / f"{stem}.{ext}"
    if direct.exists():
        return direct
    matches: List[Path] = list(parent.glob(f"{stem}.{ext}"))
    if matches:
        return matches[0]
    raise AssertionError(
        f"No file matching '{stem}.{ext}' found under {parent}; "
        f"contents: {sorted(p.name for p in parent.iterdir())}"
    )


# ---------------------------------------------------------------------------
# 1-10: API behaviour + mode coverage
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_show_bundle_graph_smoke(tmp_path: Path) -> None:
    """Shared-topology bundle, mode='auto', writes a non-empty PDF."""

    b = _shared_bundle()
    out = tmp_path / "bundle"
    show_bundle_graph(b, vis_outpath=str(out), mode="auto", save_only=True)
    rendered = _find_rendered(tmp_path, "bundle", "pdf")
    assert _file_nonempty(rendered), f"expected non-empty file at {rendered}"


def test_show_bundle_graph_divergence_mode(tmp_path: Path) -> None:
    """Explicit mode='divergence' on a shared-topology bundle writes output."""

    b = _shared_bundle()
    out = tmp_path / "bundle_div"
    show_bundle_graph(
        b,
        vis_outpath=str(out),
        mode="divergence",
        save_only=True,
    )
    rendered = _find_rendered(tmp_path, "bundle_div", "pdf")
    assert _file_nonempty(rendered)


def test_show_bundle_graph_swarm_mode(tmp_path: Path) -> None:
    """Divergent-topology bundle in swarm mode writes output."""

    b = _divergent_bundle()
    out = tmp_path / "bundle_swarm"
    show_bundle_graph(
        b,
        vis_outpath=str(out),
        mode="swarm",
        save_only=True,
    )
    rendered = _find_rendered(tmp_path, "bundle_swarm", "pdf")
    assert _file_nonempty(rendered)


def test_show_bundle_graph_group_color_mode(tmp_path: Path) -> None:
    """Bundle with groups + mode='group_color' writes output."""

    b = _grouped_bundle()
    out = tmp_path / "bundle_groups"
    show_bundle_graph(
        b,
        vis_outpath=str(out),
        mode="group_color",
        save_only=True,
    )
    rendered = _find_rendered(tmp_path, "bundle_groups", "pdf")
    assert _file_nonempty(rendered)


def test_show_bundle_graph_group_color_no_groups_errors() -> None:
    """mode='group_color' on a bundle with no groups raises ValueError."""

    b = _shared_bundle()
    with pytest.raises(ValueError, match="bundle.groups"):
        show_bundle_graph(b, mode="group_color", save_only=True, return_dot=True)


def test_show_bundle_graph_auto_mode_picks_divergence() -> None:
    """Shared-topology bundle + mode='auto' resolves to divergence styling.

    Inspect the DOT caption (which records the resolved mode) plus a
    palette colour cue: divergence uses the Reds palette, of which the
    final stop ``#4a000a`` is a unique signature relative to viridis.
    """

    b = _shared_bundle()
    src = show_bundle_graph(b, mode="auto", return_dot=True)
    assert src is not None
    # Caption records the auto resolution.
    assert "auto -&gt; divergence" in src or "auto -> divergence" in src
    # Reds palette in use somewhere.
    assert any(stop in src for stop in ("#fff5f0", "#fee0d2", "#fcbba1"))


def test_show_bundle_graph_auto_mode_picks_swarm() -> None:
    """Divergent-topology bundle + mode='auto' resolves to swarm styling."""

    b = _divergent_bundle()
    src = show_bundle_graph(b, mode="auto", return_dot=True)
    assert src is not None
    assert "auto -&gt; swarm" in src or "auto -> swarm" in src
    # Viridis palette signature -- ``#440154`` is unique to viridis stops.
    assert "#440154" in src or "#fde725" in src


def test_bundle_show_method(tmp_path: Path) -> None:
    """``bundle.show()`` produces equivalent output to the top-level helper."""

    b = _shared_bundle()
    out_a = tmp_path / "via_method"
    out_b = tmp_path / "via_func"

    b.show(vis_outpath=str(out_a), mode="divergence", save_only=True)
    show_bundle_graph(b, vis_outpath=str(out_b), mode="divergence", save_only=True)

    file_a = _find_rendered(tmp_path, "via_method", "pdf")
    file_b = _find_rendered(tmp_path, "via_func", "pdf")
    assert _file_nonempty(file_a)
    assert _file_nonempty(file_b)
    # Both DOT sources should be identical -- the method is a thin wrapper.
    src_a = b.show(vis_outpath=None, mode="divergence", return_dot=True)
    src_b = show_bundle_graph(b, mode="divergence", return_dot=True)
    assert src_a == src_b


def test_show_bundle_graph_format_png(tmp_path: Path) -> None:
    """vis_format='png' produces a .png file."""

    b = _shared_bundle()
    out = tmp_path / "bundle_png"
    show_bundle_graph(
        b,
        vis_outpath=str(out),
        mode="divergence",
        vis_format="png",
        save_only=True,
    )
    rendered = _find_rendered(tmp_path, "bundle_png", "png")
    assert _file_nonempty(rendered)


def test_show_bundle_graph_format_svg(tmp_path: Path) -> None:
    """vis_format='svg' produces a .svg file."""

    b = _shared_bundle()
    out = tmp_path / "bundle_svg"
    show_bundle_graph(
        b,
        vis_outpath=str(out),
        mode="divergence",
        vis_format="svg",
        save_only=True,
    )
    rendered = _find_rendered(tmp_path, "bundle_svg", "svg")
    assert _file_nonempty(rendered)


# ---------------------------------------------------------------------------
# 11-15: regression + kwargs + IO
# ---------------------------------------------------------------------------


def test_show_model_graph_unchanged(tmp_path: Path) -> None:
    """Regression: tl.show_model_graph(modellog) still produces a file."""

    model = _LinearNet()
    out = tmp_path / "single_trace"
    tl.show_model_graph(
        model,
        torch.rand(2, 4),
        vis_outpath=str(out),
        vis_save_only=True,
        vis_fileformat="pdf",
    )
    rendered = _find_rendered(tmp_path, "single_trace", "pdf")
    assert _file_nonempty(rendered)


def test_swarm_show_coverage_kwarg() -> None:
    """``show_coverage=True`` in swarm mode writes coverage labels into DOT."""

    b = _divergent_bundle()
    src_with = show_bundle_graph(b, mode="swarm", show_coverage=True, return_dot=True)
    src_without = show_bundle_graph(b, mode="swarm", show_coverage=False, return_dot=True)
    assert src_with is not None and src_without is not None
    assert "coverage:" in src_with
    assert "coverage:" not in src_without


def test_metric_kwarg_for_divergence() -> None:
    """metric='relative_l2' produces different DOT than the default cosine."""

    b = _shared_bundle()
    src_default = show_bundle_graph(b, mode="divergence", return_dot=True)
    src_l2 = show_bundle_graph(b, mode="divergence", metric="relative_l2", return_dot=True)
    assert src_default is not None and src_l2 is not None
    # The per-node distance values flow into the rendered ``div: X.XX``
    # rows, so swapping metrics SHOULD change the source unless the
    # metrics happen to agree exactly (vanishingly unlikely on random
    # initialisation).
    assert src_default != src_l2


def test_show_bundle_graph_with_default_outpath(tmp_path: Path) -> None:
    """No vis_outpath -> writes to ``bundle_graph.<format>`` in cwd."""

    import os

    cwd = Path(os.getcwd())
    saved = list(cwd.glob("bundle_graph.pdf"))
    # Move to tmp_path so cleanup is automatic and we don't pollute repo.
    os.chdir(tmp_path)
    try:
        b = _shared_bundle()
        show_bundle_graph(b, mode="divergence", save_only=True)
        rendered = tmp_path / "bundle_graph.pdf"
        assert _file_nonempty(rendered)
    finally:
        os.chdir(cwd)
    # Sanity: we haven't accidentally written into the repo root.
    after = list(cwd.glob("bundle_graph.pdf"))
    assert after == saved, f"unexpected new bundle_graph.pdf in cwd: {after} vs {saved}"


def test_directory_creation(tmp_path: Path) -> None:
    """vis_outpath='subdir/bundle.pdf' creates the subdir if needed."""

    b = _shared_bundle()
    nested = tmp_path / "deep" / "nest"
    assert not nested.exists()
    out = nested / "bundle"
    show_bundle_graph(
        b,
        vis_outpath=str(out),
        mode="divergence",
        save_only=True,
    )
    rendered = _find_rendered(nested, "bundle", "pdf")
    assert _file_nonempty(rendered)


# ---------------------------------------------------------------------------
# Bonus: a couple of internal-consistency checks discovered while writing
# the spec'd 15 -- helpful for catching regressions later but cheap to run.
# ---------------------------------------------------------------------------


def test_invalid_mode_raises() -> None:
    """An unknown mode literal raises ValueError before doing any work."""

    b = _shared_bundle()
    with pytest.raises(ValueError, match="mode must be one of"):
        show_bundle_graph(b, mode="rainbow", return_dot=True)  # type: ignore[arg-type]


def test_too_many_groups_raises() -> None:
    """More groups than the categorical palette supports raises ValueError."""

    model = _LinearNet()
    traces = [tl.log_forward_pass(model, torch.rand(2, 4)) for _ in range(11)]
    names = [f"t{i}" for i in range(11)]
    groups = {f"g{i}": [names[i]] for i in range(11)}
    b = TraceBundle(traces, names=names, groups=groups)
    with pytest.raises(ValueError, match="up to 10 groups"):
        show_bundle_graph(b, mode="group_color", return_dot=True)
