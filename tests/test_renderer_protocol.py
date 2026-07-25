"""Contract tests for the trace-free RenderIR renderer boundary."""

from pathlib import Path

import pytest

from torchlens.visualization import _render_nodes
from torchlens.visualization.render_ir import RenderIR, RenderIRDotStatement, RenderIRNode
from torchlens.visualization.renderers import (
    GraphvizRenderer,
    RendererCapabilities,
    UnsupportedRendererCapabilityError,
)
from torchlens.visualization.request import RenderContext, RenderTarget


def _raise_legacy_entrypoint(*args: object, **kwargs: object) -> None:
    """Fail immediately if a retired smart-emission entrypoint is reached."""

    del args, kwargs
    raise AssertionError("retired smart-emission entrypoint reached")


def _trace_free_ir() -> RenderIR:
    """Build a self-contained RenderIR without constructing a Trace."""

    return RenderIR(
        context=RenderContext(),
        nodes=(
            RenderIRNode(
                name="inputpass1",
                kind="boundary",
                owner_cluster=None,
                source_label="input:1",
            ),
        ),
        edges=(),
        regions=(),
        dot_statements=(
            RenderIRDotStatement(
                kind="node",
                attrs=(
                    ("name", "inputpass1"),
                    ("label", "input:1"),
                    ("shape", "oval"),
                ),
            ),
        ),
    )


def test_graphviz_renderer_renders_ir_without_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove backend execution cannot fall back to the legacy Trace walker."""

    monkeypatch.setattr(_render_nodes, "_add_node_to_graphviz", _raise_legacy_entrypoint)
    target = RenderTarget(
        outpath=str(tmp_path / "trace_free"),
        fileformat="svg",
        save_only=True,
        viewer=False,
    )

    report = GraphvizRenderer().render(_trace_free_ir(), target)

    assert "inputpass1" in report.source
    assert report.output_path is not None
    assert report.output_path.is_file()


def test_missing_renderer_capability_fails_explicitly() -> None:
    """Reject unsupported semantics instead of silently approximating them."""

    capabilities = RendererCapabilities(layout_execution=False)
    required = RendererCapabilities(layout_execution=True)

    with pytest.raises(UnsupportedRendererCapabilityError, match="layout_execution"):
        capabilities.require(required, "minimal")


def test_retired_forward_dot_entrypoints_are_absent() -> None:
    """Keep deleted recorder/replay entrypoints from silently returning."""

    from torchlens.visualization import _render_common, _render_dot

    assert not hasattr(_render_common, "ForwardDotIR")
    assert not hasattr(_render_common, "_ForwardDotRecorder")
    assert not hasattr(_render_dot, "_replay_forward_dot_calls")
