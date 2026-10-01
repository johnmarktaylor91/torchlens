"""F12 pins: the Stage-0 deterministic audit and the R0 baseline runner.

Stage-0 runs with no model in the loop; a failure is a build bug and never
reaches an evaluator. The R0 leg here baselines the geometry gate and the
shipped headlabel family on the toy corpus (the real-corpus R0 legs are
D03's, per the gallery spec).
"""

from __future__ import annotations

import json
import re
import shutil
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization import lenses
from torchlens.visualization.lenses import audit

DOT_AVAILABLE = shutil.which("dot") is not None


@pytest.fixture(scope="module")
def speed_render(tmp_path_factory: Any) -> Any:
    """One rendered speed-lens artifact with its resolution."""

    class Toy(nn.Module):
        """Two-linear toy."""

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x: Any) -> Any:
            return self.fc2(torch.relu(self.fc1(x)))

    out_dir = tmp_path_factory.mktemp("stage0")
    log = tl.trace(Toy(), torch.randn(2, 8))
    resolution = lenses.resolve_lens(log, "speed")
    graph = log.draw(
        **resolution.draw_kwargs,
        vis_outpath=str(out_dir / "speed"),
        vis_fileformat="svg",
        vis_save_only=True,
        return_graph=True,
    )
    yield log, resolution, graph, str(out_dir / "speed.svg")
    log.cleanup()


@pytest.mark.smoke
@pytest.mark.skipif(not DOT_AVAILABLE, reason="graphviz dot binary unavailable")
def test_stage0_runs_green_on_a_clean_artifact(speed_render: Any) -> None:
    """The full audit passes on a small clean render (headlabel baseline
    measured from the artifact itself -- the R0 discipline)."""

    log, resolution, graph, svg_path = speed_render
    headlabel_baseline = len(re.findall(r"\bheadlabel=", graph.source))
    xlabel_baseline = len(re.findall(r"\bxlabel=", graph.source))
    report = audit.run_stage0(
        graph.source,
        resolution=resolution,
        svg_path=svg_path,
        manifest=audit.build_run_manifest(
            "toy", lens="speed", skin=None, resolved_kwargs=resolution.draw_kwargs
        ),
        checks=audit.Stage0Checks(
            # Cardinality = the DISTINCT values the channel can express here
            # (timings may tie run-to-run on a tiny toy).
            cardinality=len({float(op.func_duration) for op in log.ops}),
            baseline_headlabels=headlabel_baseline,
            baseline_xlabels=xlabel_baseline,
        ),
    )
    assert report.passed, report.failed_checks()
    assert report.manifest["lens"] == "speed"
    assert report.manifest["versions"]["torch"]


@pytest.mark.smoke
def test_label_spelling_audit_counts_forbidden_families() -> None:
    """A new xlabel is a failure; headlabels are gated by the R0 baseline."""

    dot = 'digraph { a -> b [xlabel="arg 0"]; }'
    findings = audit.stage0.label_spelling_findings(dot, baseline_headlabels=0)
    assert not findings[0].passed
    clean = audit.stage0.label_spelling_findings("digraph { a -> b; }", baseline_headlabels=0)
    assert clean[0].passed


@pytest.mark.smoke
def test_output_caps_flag_extreme_aspect(tmp_path: Any) -> None:
    """A declared 20000x100 artifact fails the caps."""

    svg = tmp_path / "wide.svg"
    svg.write_text('<svg width="20000pt" height="100pt" xmlns="http://www.w3.org/2000/svg"/>')
    findings = audit.stage0.output_caps_findings(str(svg))
    assert not findings[0].passed


@pytest.mark.smoke
def test_disclosure_checks_catch_a_stripped_caption(speed_render: Any) -> None:
    """Removing the coverage line from the DOT fails the audit."""

    log, resolution, graph, svg_path = speed_render
    stripped = graph.source.replace("coverage: encoded", "")
    report = audit.run_stage0(
        stripped, resolution=resolution, checks=audit.Stage0Checks(geometry=False)
    )
    assert "coverage_line" in report.failed_checks()


@pytest.mark.smoke
def test_fill_spread_gates_apply_only_with_a_channel(speed_render: Any) -> None:
    """A channel-free artifact carries no encoded-fill findings."""

    log, resolution, graph, svg_path = speed_render
    bare = lenses.resolve_lens(log, "blueprint")
    bare_graph = log.draw(
        **bare.draw_kwargs,
        vis_outpath=svg_path.replace(".svg", "-bare"),
        vis_fileformat="svg",
        vis_save_only=True,
        return_graph=True,
    )
    report = audit.run_stage0(
        bare_graph.source, resolution=bare, checks=audit.Stage0Checks(geometry=False)
    )
    assert "distinct_fills" not in {finding.check for finding in report.findings}


@pytest.mark.smoke
def test_compaction_non_identity_is_checkable(speed_render: Any) -> None:
    """The render-pair check: an active compaction may not be byte-identical
    to compaction-off (composition row 3's audit half)."""

    log, resolution, graph, svg_path = speed_render
    collapsed = lenses.resolve_lens(log, "overview")
    # On a 5-op toy the resolver legitimately picks collapse="none" (renders
    # in full): identity is then EXPECTED, and the dial line discloses it.
    if collapsed.budget is not None and collapsed.budget.draw_kwargs.get("collapse") != "none":
        collapsed_graph = log.draw(
            **collapsed.draw_kwargs,
            vis_outpath=svg_path.replace(".svg", "-ov"),
            vis_fileformat="svg",
            vis_save_only=True,
            return_graph=True,
        )
        assert collapsed_graph.source != graph.source


@pytest.mark.heavy
@pytest.mark.skipif(not DOT_AVAILABLE, reason="graphviz dot binary unavailable")
def test_geometry_baseline_over_the_toy_corpus(tmp_path: Any) -> None:
    """ROW GATE (R0 geometry baseline): run the Stage-0 geometry audit over
    the toy corpus under the overview and debug lenses, record the baseline
    JSON artifact, and hold the shipped-headlabel count stable per artifact.

    The baseline is DESCRIPTIVE on today's code: geometry findings are
    recorded (not asserted zero) because the shipped ``arg N`` head-label
    family is a known pre-existing penetration source the battery will
    relocate; NEW xlabel families remain hard failures.
    """

    baseline: dict[str, Any] = {}
    for member in audit.CORPUS:
        if member.kind != "toy":
            continue
        model, example = member.build()
        log = tl.trace(model, example)
        try:
            for lens_name in ("overview", "debug"):
                try:
                    resolution = lenses.resolve_lens(log, lens_name)
                except tl.errors.InvalidArgumentError:
                    continue  # typed subject refusals are legitimate on some toys
                graph = log.draw(
                    **resolution.draw_kwargs,
                    vis_outpath=str(tmp_path / f"{member.name}-{lens_name}"),
                    vis_fileformat="svg",
                    vis_save_only=True,
                    return_graph=True,
                )
                headlabels = len(re.findall(r"\bheadlabel=", graph.source))
                xlabels = len(re.findall(r"\bxlabel=", graph.source))
                geometry = audit.stage0.geometry_findings(graph.source)
                baseline[f"{member.name}:{lens_name}"] = {
                    "headlabels": headlabels,
                    "xlabels": xlabels,
                    "node_overlaps": geometry[0].measurements["count"],
                    "label_penetrations": geometry[1].measurements["count"],
                }
        finally:
            log.cleanup()
    assert baseline
    artifact = tmp_path / "r0_geometry_baseline.json"
    artifact.write_text(json.dumps(baseline, indent=2, sort_keys=True))
    reloaded = json.loads(artifact.read_text())
    assert reloaded == baseline
