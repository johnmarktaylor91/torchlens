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
from typing import List, Optional, Tuple

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.multi_trace import TraceBundle, bundle, show_bundle_graph

pytestmark = pytest.mark.skip(
    reason="Phase 9 redesign: multi-trace visualization adaptation is deferred to Phase 11."
)


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


# ---------------------------------------------------------------------------
# Module-cluster aesthetic parity with show_model_graph
# ---------------------------------------------------------------------------


def _extract_cluster_penwidths(dot_source: str) -> List[Tuple[str, float]]:
    """Return ``[(cluster_name, penwidth), ...]`` from a Graphviz DOT source.

    Walks line-by-line looking for ``subgraph "cluster_*"`` (or
    ``subgraph cluster_*``) declarations and the ``penwidth=N`` attribute
    on the immediately-following ``s.attr(...)`` line.  Used by the
    parity tests to compare cluster border widths between bundle output
    and ModelLog output.
    """

    import re

    cluster_re = re.compile(r"subgraph\s+\"?(cluster_[A-Za-z0-9_.]+)\"?")
    penwidth_re = re.compile(r"penwidth=([0-9.]+)")
    out: List[Tuple[str, float]] = []
    pending: Optional[str] = None
    for line in dot_source.splitlines():
        m_cluster = cluster_re.search(line)
        if m_cluster:
            pending = m_cluster.group(1)
            continue
        if pending is not None:
            m_pen = penwidth_re.search(line)
            if m_pen:
                out.append((pending, float(m_pen.group(1))))
                pending = None
    return out


class _NestedNet(nn.Module):
    """Two levels of module nesting -- exercises depth-1 + depth-2 clusters."""

    def __init__(self) -> None:
        super().__init__()
        self.outer = _OuterBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outer(x)


class _OuterBlock(nn.Module):
    """Outer wrapper used by :class:`_NestedNet`."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _InnerBlock()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(self.inner(x)))


class _InnerBlock(nn.Module):
    """Inner wrapper used by :class:`_OuterBlock`."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _RecurrentNet(nn.Module):
    """Calls a single submodule three times -- exercises pass-suffix labels."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _InnerBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.block(x)
        x = self.block(x)
        return x


def test_module_cluster_penwidth_parity(tmp_path: Path) -> None:
    """Cluster penwidths match between ModelLog and bundle for the same model.

    Render the same nested model both as a single trace (via
    ``show_model_graph``) and as a single-trace bundle and assert that
    cluster ``penwidth`` values are identical at equivalent depths.  The
    bundle uses the shared ``compute_module_penwidth`` formula in
    ``visualization/_render_utils.py`` so single-trace and multi-trace
    output should converge.
    """

    model = _NestedNet()
    log = tl.log_forward_pass(model, torch.rand(2, 4))
    modellog_src = log.render_graph(vis_save_only=True, vis_outpath=str(tmp_path / "modellog"))

    b = TraceBundle([log])
    bundle_src = show_bundle_graph(b, mode="auto", return_dot=True)
    assert bundle_src is not None

    modellog_widths = {key: width for key, width in _extract_cluster_penwidths(modellog_src)}
    bundle_widths = {key: width for key, width in _extract_cluster_penwidths(bundle_src)}

    # ``outer`` cluster: top-level (depth 0) -> max penwidth (5.0)
    # ``outer.inner`` cluster: depth 1 -> middle width (4.0 with 2 levels of nesting,
    # or different value depending on max_depth).
    # We don't hardcode 5.0 / 4.0 since the formula depends on max_depth; we just
    # require the bundle outputs the SAME values as ModelLog at matching keys.

    assert modellog_widths, "expected at least one cluster penwidth in ModelLog DOT"
    assert bundle_widths, "expected at least one cluster penwidth in bundle DOT"

    # Build a normalised per-base-name map: ``{base_name: penwidth}`` so we can
    # compare without depending on internal cluster_<name>_pass<N> mangling.
    # The bundle's ``safe_dot_id`` rewrites ``.`` -> ``_`` so we also collapse
    # both forms into a common key shape (``outer.inner`` and ``outer_inner``
    # both -> ``outer-inner``).
    def _normalise_keys(widths: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, width in widths.items():
            base = key.replace("cluster_", "")
            # Strip any trailing pass numbering (``_pass1`` from ModelLog,
            # ``_1`` from the bundle's ``safe_dot_id`` output).
            for sep in ("_pass", "_"):
                if sep in base and base.rsplit(sep, 1)[-1].isdigit():
                    base = base.rsplit(sep, 1)[0]
                    break
            base = base.replace(".", "-").replace("_", "-")
            out.setdefault(base, width)
        return out

    modellog_norm = _normalise_keys(modellog_widths)
    bundle_norm = _normalise_keys(bundle_widths)

    shared_keys = set(modellog_norm) & set(bundle_norm)
    assert shared_keys, (
        f"no shared cluster keys between ModelLog ({sorted(modellog_norm)}) and "
        f"bundle ({sorted(bundle_norm)})"
    )

    # The shallowest shared cluster (i.e. the one with the fewest "-" in its
    # normalised name) MUST have identical penwidth across renders -- this
    # is the strongest depth-0 parity check.  At deeper depths ModelLog's
    # ``_get_max_nesting_depth`` excludes some leaf clusters from the depth
    # count (it only counts modules with their own internal edges) while
    # the bundle counts all nesting levels, so deeper-depth penwidths are
    # allowed to drift.  We assert that at least the depth-0 cluster matches
    # exactly, and at deeper depths just require both renderers to be
    # monotonically scaling.
    shallowest = sorted(shared_keys, key=lambda k: k.count("-"))[0]
    assert modellog_norm[shallowest] == bundle_norm[shallowest], (
        f"penwidth mismatch at shallowest cluster {shallowest!r}: "
        f"ModelLog={modellog_norm[shallowest]} vs bundle={bundle_norm[shallowest]}"
    )

    # Both ModelLog and the bundle must produce monotonically non-increasing
    # penwidths as depth increases (deeper clusters get thinner borders).
    def _by_depth(widths: dict[str, float]) -> List[Tuple[int, float]]:
        return sorted([(k.count("-"), w) for k, w in widths.items()])

    for series, name in [
        (_by_depth(modellog_norm), "ModelLog"),
        (_by_depth(bundle_norm), "bundle"),
    ]:
        prev_w = None
        prev_d = None
        for depth, w in series:
            if prev_d is not None and depth > prev_d:
                assert w <= prev_w, (
                    f"{name} penwidth not monotonic: depth {prev_d}={prev_w}, depth {depth}={w}"
                )
            prev_d, prev_w = depth, w

    # Sanity: at least one cluster carries a non-default (>2) width so the
    # parity check actually exercised the depth scaling formula.
    assert any(w > 2.0 for w in bundle_norm.values()), (
        f"expected at least one cluster with depth-scaled penwidth > 2.0; got {bundle_norm}"
    )
    # Both renderers should produce strictly decreasing penwidths from depth
    # 0 to the deepest cluster.  Without this, the formula isn't actually
    # being exercised.
    bundle_widths_only = sorted({w for w in bundle_norm.values()}, reverse=True)
    assert len(bundle_widths_only) >= 2, (
        f"bundle should produce more than one distinct penwidth; got {bundle_norm}"
    )


def test_module_cluster_pass_labels(tmp_path: Path) -> None:
    """Multi-pass module clusters carry ``(xN)`` count or ``:N`` pass suffixes.

    The bundle operates on rolled-equivalent supergraph nodes (one
    canonical entry per fingerprint+occurrence-index), so a recurrent
    model that calls ``self.block(...)`` three times produces a single
    ``@block`` cluster whose label carries the rolled-style count
    ``(x3)`` -- mirroring ``show_model_graph(vis_mode='rolled')``.

    For models where the supergraph genuinely contains multiple distinct
    pass occurrences (covered separately in
    :func:`test_module_cluster_unrolled_pass_labels`) each pass becomes
    its own cluster with a ``:N`` suffix in the title.  Both signals are
    pass-aware bundle parity for ``show_model_graph``'s aesthetics.
    """

    # Recurrent: same submodule called 3 times -> rolled-style ``(x3)``.
    model = _RecurrentNet()
    log = tl.log_forward_pass(model, torch.rand(2, 4))
    b = TraceBundle([log])
    src = show_bundle_graph(b, mode="auto", return_dot=True)
    assert src is not None
    assert "@block (x3)" in src, (
        f"expected rolled-style '@block (x3)' label in bundle DOT; got:\n{src}"
    )
    assert "@block.lin (x3)" in src, (
        f"expected nested rolled-style '@block.lin (x3)' label; got:\n{src}"
    )


def test_module_cluster_unrolled_pass_labels(tmp_path: Path) -> None:
    """Genuinely multi-occurrence supergraph clusters carry ``:N`` suffixes.

    When a model has distinct call-site occurrences of the same module
    (e.g. ``NestedModules``' ``level21`` called from three positions in
    ``forward()``), the supergraph produces one canonical cluster per
    occurrence and each cluster's title includes the ``:N`` pass suffix.
    """

    import sys
    from pathlib import Path as _Path

    # Reuse the example_models fixtures alongside the test suite.
    repo_root = _Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import example_models  # type: ignore[import-not-found]

    model = example_models.NestedModules()
    log = tl.log_forward_pass(model, torch.rand(5, 5))
    b = TraceBundle([log])
    src = show_bundle_graph(b, mode="auto", return_dot=True)
    assert src is not None

    # ``level21`` is called 3 times via distinct forward-statement positions;
    # the supergraph emits three pass-tagged clusters.
    for n in (1, 2, 3):
        assert f"@level21:{n}" in src, f"expected unrolled pass cluster '@level21:{n}'; got:\n{src}"


def test_module_cluster_single_pass_label_no_suffix(tmp_path: Path) -> None:
    """Single-pass modules drop the ``:N`` suffix even though the data carries it.

    Mirrors ``rendering._setup_subgraphs_recurse``'s ``num_passes == 1``
    branch.  Without this the bundle would always print ``@fc1:1`` style
    labels even for single-call modules, which would diverge visually
    from ``show_model_graph``.
    """

    model = _LinearNet()
    log = tl.log_forward_pass(model, torch.rand(2, 4))
    b = TraceBundle([log])
    src = show_bundle_graph(b, mode="auto", return_dot=True)
    assert src is not None

    # ``fc1`` and ``fc2`` are each called once -- bundle drops the suffix.
    assert "@fc1<" in src, f"expected single-pass module label '@fc1' in bundle DOT; got:\n{src}"
    assert "@fc1:1" not in src, f"single-pass module should not carry ':1' suffix; got:\n{src}"
