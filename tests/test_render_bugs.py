"""Regression tests for Graphviz render failure handling."""

from __future__ import annotations

import dataclasses
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
import graphviz
from torch import nn

import torchlens as tl
from torchlens.data_classes.trace import Trace
from torchlens.visualization._rank_layout_internal import layout as rank_layout
from torchlens.visualization._render_dot import _strip_render_extension
from torchlens.visualization._render_utils import render_dot_to_file
from torchlens.visualization.collapse_plan import RenderContext
from torchlens.visualization.render_ir import build_render_ir
from torchlens.visualization.rendering import GraphvizRenderError


class _TinyRenderModel(nn.Module):
    """Small model that produces forward and backward graph nodes."""

    def __init__(self) -> None:
        """Initialize submodules."""
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return torch.relu(self.linear(x)).sum()


class _LargeChainRenderModel(nn.Module):
    """Model with enough repeated ops to exercise large PDF page geometry."""

    def __init__(self, width: int = 4, depth: int = 48) -> None:
        """Initialize a deterministic linear chain."""

        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain with one activation per layer."""

        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class _NestedTorchOpModel(nn.Module):
    """Model with two non-module ops inside one nested module scope."""

    class _Inner(nn.Module):
        """Nested module whose internal op edge exposes cluster selection."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run two differentiable torch ops in the same nested scope."""

            return torch.sigmoid(torch.relu(x))

    class _Block(nn.Module):
        """Outer module wrapping the nested op scope."""

        def __init__(self) -> None:
            """Initialize the inner module."""

            super().__init__()
            self.inner = _NestedTorchOpModel._Inner()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the nested module."""

            return self.inner(x)

    def __init__(self) -> None:
        """Initialize nested modules."""

        super().__init__()
        self.block = self._Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.block(x).sum()


class _SpecialCharDictOutputModel(nn.Module):
    """Return a two-leaf dict output with an HTML-special-character key."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the model."""

        return {"loss & aux": x + 1, "b": x + 2}


class _LSTMCellSeq(nn.Module):
    """Loop over an LSTMCell whose call returns two tensors."""

    def __init__(self) -> None:
        """Initialize the recurrent cell."""

        super().__init__()
        self.cell = nn.LSTMCell(6, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run four recurrent cell calls."""

        h = torch.zeros(x.shape[1], 5)
        c = torch.zeros(x.shape[1], 5)
        outputs = []
        for step in range(x.shape[0]):
            h, c = self.cell(x[step], (h, c))
            outputs.append(h)
        return torch.stack(outputs)


@pytest.fixture
def forward_trace() -> Trace:
    """Return a tiny forward Trace."""

    return tl.trace(_TinyRenderModel(), torch.randn(2, 3, requires_grad=True))


@pytest.fixture
def backward_trace() -> Trace:
    """Return a tiny Trace with backward metadata."""

    trace = tl.trace(_TinyRenderModel(), torch.randn(2, 3, requires_grad=True), save_grads="all")
    trace.log_backward(trace[trace.output_layers[0]].out)
    return trace


def test_skip_fn_omits_unrolled_skipped_node(tmp_path: Path) -> None:
    """Skipped unrolled nodes should not be emitted as detached DOT nodes."""

    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))

    dot = trace.draw(
        skip_fn=lambda layer: layer.func_name == "relu",
        vis_outpath=str(tmp_path / "skip_relu"),
        vis_save_only=True,
        vis_fileformat="dot",
        order_siblings=False,
    )

    assert "relu_1_2" not in dot


def test_render_ir_honors_skip_fn_without_run_folds() -> None:
    """Render IR node/edge topology follows skip-spliced drawing topology."""

    trace = tl.trace(_TinyRenderModel(), torch.randn(2, 3))

    def skip_relu(layer: Any) -> bool:
        """Skip relu layers."""

        return getattr(layer, "layer_type", None) == "relu"

    render_ir = build_render_ir(
        trace,
        collapse_fn=None,
        run_folds=None,
        context=RenderContext(skip_fn=skip_relu),
    )

    node_labels = {node.source_label for node in render_ir.nodes}
    edge_originals = {
        label for edge in render_ir.edges for label in edge.source_originals + edge.target_originals
    }
    assert "relu_1_2" not in node_labels
    assert "relu_1_2" not in edge_originals
    assert ("linear_1_1",) in {edge.source_originals for edge in render_ir.edges}
    assert ("sum_1_3",) in {edge.target_originals for edge in render_ir.edges}


def test_render_extension_stripping_is_case_insensitive_and_shared() -> None:
    """Graphviz outpath normalization strips known extensions once."""

    assert _strip_render_extension("/tmp/model.PDF") == "/tmp/model"
    assert _strip_render_extension("/tmp/model.SVG") == "/tmp/model"
    assert _strip_render_extension("/tmp/model.dot") == "/tmp/model"


def test_hidden_buffer_update_node_is_not_rendered(tmp_path: Path) -> None:
    """Buffer-only update ops hidden by buffer visibility should not render."""

    model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8)).train()
    trace = tl.trace(model, torch.randn(4, 8))

    dot = trace.draw(
        vis_outpath=str(tmp_path / "batchnorm_hidden_buffers"),
        vis_save_only=True,
        vis_fileformat="dot",
        order_siblings=False,
    )

    assert "add_1_2" not in dot
    assert "batchnorm_1_3" in dot


def test_lstmcell_rolled_count_uses_calls_not_outputs(tmp_path: Path) -> None:
    """Rolled multi-output module ops should count calls in the ``(xN)`` badge."""

    trace = tl.trace(_LSTMCellSeq(), torch.randn(4, 1, 6))

    dot = trace.draw(
        vis_mode="rolled",
        vis_outpath=str(tmp_path / "lstmcell_rolled"),
        vis_save_only=True,
        vis_fileformat="dot",
        order_siblings=False,
    )

    assert "lstmcell_1_4 (x4)" in dot
    assert "lstmcell_1_4 (x8)" not in dot


def test_dark_theme_themes_caption_and_parameter_nodes(tmp_path: Path) -> None:
    """Dark theme should not leave graph captions or parameter nodes dark-on-dark."""

    trace = tl.trace(nn.Linear(4, 2), torch.randn(1, 4))

    dot = trace.draw(
        vis_theme="dark",
        vis_outpath=str(tmp_path / "dark"),
        vis_save_only=True,
        vis_fileformat="dot",
        order_siblings=False,
    )

    assert "FONT COLOR='#F9FAFB'" in dot
    assert 'fillcolor="#374151"' in dot


def test_rank_layout_embeds_code_panel(tmp_path: Path) -> None:
    """Rank layout should keep code panel content instead of dropping it."""

    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(1, 4))

    dot = trace.draw(
        vis_node_placement="rank",
        code_panel=lambda _model: "def forward(self, x):\n    return self[1](self[0](x))",
        vis_outpath=str(tmp_path / "rank_code_panel"),
        vis_save_only=True,
        vis_fileformat="svg",
        order_siblings=False,
    )

    assert "cluster_torchlens_code_panel" in dot
    assert "Source code" in dot


def test_shared_render_timeout_preserves_reported_dot_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared bundle render helper keeps DOT source after a timeout warning."""

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    outpath = tmp_path / "timeout_graph"
    dot = graphviz.Digraph()
    dot.node("a")

    with pytest.warns(UserWarning, match="DOT source saved"):
        source = render_dot_to_file(dot, str(outpath), "svg", True, timeout_seconds=0)

    assert source.startswith("digraph")
    assert outpath.exists()
    assert "a" in outpath.read_text(encoding="utf-8")


def test_rank_layout_failure_preserves_dot_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank layout keeps its generated DOT source when neato fails."""

    def fail_neato(*args: Any, **kwargs: Any) -> None:
        """Simulate a rank-layout render failure after DOT is written."""

        del args, kwargs
        raise RuntimeError("forced neato failure")

    monkeypatch.setattr(rank_layout, "_run_neato_with_fallbacks", fail_neato)
    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(1, 4))
    outpath = tmp_path / "rank_failed"
    try:
        with pytest.raises(RuntimeError, match="forced neato failure"):
            trace.draw(
                vis_node_placement="rank",
                vis_outpath=str(outpath),
                vis_save_only=True,
                vis_fileformat="svg",
                order_siblings=False,
            )
    finally:
        trace.cleanup()

    dot_path = outpath.with_suffix(".dot")
    assert dot_path.exists()
    assert "digraph" in dot_path.read_text(encoding="utf-8")


def _raise_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Simulate a Graphviz timeout from ``subprocess.run``."""

    raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))


def _raise_called_process_error(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Simulate a Graphviz process failure from ``subprocess.run``."""

    raise subprocess.CalledProcessError(returncode=1, cmd=args[0], stderr=b"graphviz failed")


def _write_zero_byte_output(args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Simulate a successful Graphviz run that leaves an empty output file."""

    del kwargs
    output_flag_index = args.index("-o")
    Path(args[output_flag_index + 1]).write_bytes(b"")
    return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


def test_forward_render_timeout_raises_typed_error(
    forward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward rendering raises a typed error when Graphviz times out."""

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    with pytest.raises(GraphvizRenderError, match="timed out.*lowering dpi.*direct SVG.*node cap"):
        forward_trace.draw(
            vis_outpath=str(tmp_path / "forward"),
            vis_save_only=True,
            vis_fileformat="svg",
            order_siblings=False,
        )


def test_backward_render_timeout_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward rendering raises a typed error when Graphviz times out."""

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    with pytest.raises(GraphvizRenderError, match="timed out.*lowering dpi.*direct SVG.*node cap"):
        backward_trace.draw_backward(
            vis_outpath=str(tmp_path / "backward"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_combined_render_timeout_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined rendering raises a typed error when Graphviz times out."""

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    with pytest.raises(GraphvizRenderError, match="timed out.*lowering dpi.*direct SVG.*node cap"):
        backward_trace.draw_combined(
            vis_outpath=str(tmp_path / "combined"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_forward_zero_byte_render_raises_typed_error(
    forward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward rendering raises when Graphviz reports success with an empty file."""

    monkeypatch.setattr(subprocess, "run", _write_zero_byte_output)

    with pytest.raises(GraphvizRenderError, match="zero-byte.*lowering dpi.*direct SVG.*node cap"):
        forward_trace.draw(
            vis_outpath=str(tmp_path / "forward_empty"),
            vis_save_only=True,
            vis_fileformat="svg",
            order_siblings=False,
        )


def test_backward_zero_byte_render_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward rendering raises when Graphviz reports success with an empty file."""

    monkeypatch.setattr(subprocess, "run", _write_zero_byte_output)

    with pytest.raises(GraphvizRenderError, match="zero-byte.*lowering dpi.*direct SVG.*node cap"):
        backward_trace.draw_backward(
            vis_outpath=str(tmp_path / "backward_empty"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_combined_zero_byte_render_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined rendering raises when Graphviz reports success with an empty file."""

    monkeypatch.setattr(subprocess, "run", _write_zero_byte_output)

    with pytest.raises(GraphvizRenderError, match="zero-byte.*lowering dpi.*direct SVG.*node cap"):
        backward_trace.draw_combined(
            vis_outpath=str(tmp_path / "combined_empty"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_forward_called_process_error_raises_typed_error(
    forward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward rendering raises a typed error when Graphviz exits unsuccessfully."""

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)

    with pytest.raises(GraphvizRenderError, match="Graphviz failed.*graphviz failed"):
        forward_trace.draw(
            vis_outpath=str(tmp_path / "forward_failed"),
            vis_save_only=True,
            vis_fileformat="svg",
            order_siblings=False,
        )


def test_backward_called_process_error_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward rendering raises a typed error when Graphviz exits unsuccessfully."""

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)

    with pytest.raises(GraphvizRenderError, match="Graphviz failed.*graphviz failed"):
        backward_trace.draw_backward(
            vis_outpath=str(tmp_path / "backward_failed"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_combined_called_process_error_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined rendering raises a typed error when Graphviz exits unsuccessfully."""

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)

    with pytest.raises(GraphvizRenderError, match="Graphviz failed.*graphviz failed"):
        backward_trace.draw_combined(
            vis_outpath=str(tmp_path / "combined_failed"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_combined_render_keeps_dot_source_on_graphviz_failure(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined rendering preserves DOT source when Graphviz fails."""

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)
    dot_path = tmp_path / "combined_failed"

    with pytest.raises(GraphvizRenderError, match="DOT source was saved"):
        backward_trace.draw_combined(
            vis_outpath=str(tmp_path / "combined_failed"),
            vis_save_only=True,
            vis_fileformat="svg",
        )

    assert dot_path.exists()
    assert "combined forward/backward graph" in dot_path.read_text()


def test_grad_edges_use_preserved_edge_cluster_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward render passes grad edges the same LCA cluster as dataflow edges."""

    import torchlens.visualization._render_edges as _render_edges

    modules_by_edge: dict[tuple[str, str], str | int] = {}
    original_add_grad_edge = _render_edges._add_grad_edge

    def capture_add_grad_edge(
        self: Trace,
        parent_layer: object,
        child_layer: object,
        edge_style: str,
        module: str | int,
        module_edge_dict: dict[str, object],
        graphviz_graph: object,
        overrides: object,
    ) -> None:
        """Capture grad-edge cluster keys before delegating to the real implementation."""

        modules_by_edge[
            (
                str(getattr(parent_layer, "func_name", "")),
                str(getattr(child_layer, "func_name", "")),
            )
        ] = module
        original_add_grad_edge(
            self,
            parent_layer,
            child_layer,
            edge_style,
            module,
            module_edge_dict,  # type: ignore[arg-type]
            graphviz_graph,  # type: ignore[arg-type]
            overrides,  # type: ignore[arg-type]
        )

    # Patch in the CALLER's module namespace: _render_edges holds its own binding
    # of _add_grad_edge (star-imported from _render_leaf), so patching the
    # rendering facade re-export would not intercept the call.
    monkeypatch.setattr(_render_edges, "_add_grad_edge", capture_add_grad_edge)
    trace = tl.trace(
        _NestedTorchOpModel(),
        torch.randn(2, 3, requires_grad=True),
        save_grads="all",
    )
    try:
        trace.log_backward(trace[trace.output_layers[0]].out)
        trace.draw(
            vis_outpath=str(tmp_path / "nested_grad"),
            vis_save_only=True,
            vis_fileformat="svg",
            order_siblings=False,
        )
    finally:
        trace.cleanup()

    assert modules_by_edge[("relu", "sigmoid")] == "block.inner:1"


def test_container_edge_label_escapes_html_special_dict_key(tmp_path: Path) -> None:
    """Container edge labels HTML-escape special characters from dict keys.

    Regression test for commit ee5c1bcc: ``_html_container_edge_label`` (and
    sibling ``_html_edge_label``/``_html_combined_recurrence_label``)
    previously interpolated raw container-path text into a Graphviz
    HTML-like edge label. A ``DictKey``/``HFKey`` output-dict key containing
    ``&``, ``<``, or ``>`` is a realistic shape -- container path components
    render as ``str(component.key)`` (``_container_component_role`` in
    ``_render_leaf.py``) -- and previously broke Graphviz's HTML-like label
    parser, raising ``GraphvizRenderError`` on ``draw()``. Now the text is
    escaped before interpolation. Renders to an actual SVG file (not just
    the returned DOT source string) so the fix is verified end-to-end
    through the real Graphviz binary, not just at the string-building layer.
    """

    trace = tl.trace(
        _SpecialCharDictOutputModel(),
        torch.ones(2),
        capture_container_structure=True,
    )
    outpath = tmp_path / "container_edge_special_char"
    try:
        # Must NOT raise GraphvizRenderError -- the raw "&" in the dict key
        # previously broke the HTML-like label parser mid-render.
        dot = trace.draw(
            show_containers="nodes",
            vis_outpath=str(outpath),
            vis_save_only=True,
            vis_fileformat="svg",
            order_siblings=False,
        )
    finally:
        trace.cleanup()

    assert "loss &amp; aux" in dot
    svg_path = outpath.with_suffix(".svg")
    assert svg_path.exists()
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "loss &amp; aux" in svg_text


def test_large_composed_pdf_contains_visible_graph_region(tmp_path: Path) -> None:
    """Large composed PDF renders graph contents inside the page bounds."""

    fitz = pytest.importorskip("fitz")
    model = _LargeChainRenderModel().eval()
    trace = tl.trace(model, torch.randn(1, 4))
    pdf_path = tmp_path / "large_composed.pdf"
    try:
        trace.draw(
            vis_outpath=str(pdf_path.with_suffix("")),
            vis_save_only=True,
            vis_fileformat="pdf",
            code_panel=lambda _model: "def forward(self, x):\n    return x",
            order_siblings=False,
        )
    finally:
        trace.cleanup()

    document = fitz.open(pdf_path)
    try:
        page = document[0]
        page_rect = page.rect
        content_rect = fitz.Rect()
        for block in page.get_text("blocks"):
            content_rect |= fitz.Rect(block[:4])
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect is not None:
                content_rect |= rect
        assert not content_rect.is_empty
        assert page_rect.intersects(content_rect)
        assert content_rect.get_area() > 0.01 * page_rect.get_area()
    finally:
        document.close()


class _TwoInputSubModel(nn.Module):
    """Non-commutative op with two distinct parents, for arg-label tests."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return ``x - y``.

        Parameters
        ----------
        x:
            First input tensor.
        y:
            Second input tensor.

        Returns
        -------
        torch.Tensor
            Difference of the two inputs.
        """
        return torch.sub(x, y)


def test_rank_layout_escapes_html_special_chars_in_arg_edge_use_labels(
    tmp_path: Path,
) -> None:
    """Rank-engine arg-position labels must escape HTML specials.

    ``_add_arg_label`` (the rank-layout engine's argument-position labeler,
    ``torchlens/visualization/_rank_layout_internal/layout.py``) builds its
    label text from ``render_edge.argument_label``, which for ``Op`` children
    is derived from ``edge_uses[].arg_path`` (see
    ``_edge_use_argument_label``/``_arg_path_label_value`` in
    ``_render_flow.py``). Real capture can populate ``arg_path`` with
    container-key text pulled from a nested dict/list container argument
    (``DictKey``/``HFKey`` components carry the raw key string verbatim).
    Before the fix, that text was interpolated into the rank engine's
    Graphviz HTML-like edge label unescaped, so a key containing ``<``,
    ``>``, or ``&`` raised a real ``neato`` parse failure
    (``RuntimeError: neato rendering failed ... not well-formed (invalid
    token)``) -- confirmed by reverting the fix and re-running this exact
    scenario. This mirrors the parallel fix already applied to the dot
    engine's ``_label_node_arguments_if_needed`` in ``_render_edges.py``.
    """

    trace = tl.trace(_TwoInputSubModel(), (torch.randn(3), torch.randn(3)))
    try:
        sub_op = trace["sub_1_1:1"]
        # Simulate a container-key edge use (e.g. a DictKey/HFKey component
        # whose ``.key`` carries arbitrary output-dict text) landing in
        # ``arg_path`` for the first positional edge use, matching the real
        # data shape produced by container-argument capture.
        mutated_edge_uses = tuple(
            dataclasses.replace(record, arg_path=("loss & aux <script>",))
            if record.arg_kind == "positional" and record.arg_path == (0,)
            else record
            for record in sub_op.edge_uses
        )
        sub_op._edge_uses = mutated_edge_uses  # type: ignore[attr-defined]

        dot = trace.draw(
            vis_node_placement="rank",
            vis_outpath=str(tmp_path / "rank_html_escape"),
            vis_save_only=True,
            vis_fileformat="svg",
            order_siblings=False,
        )
    finally:
        trace.cleanup()

    assert "arg loss &amp; aux &lt;script&gt;" in dot
    assert "loss & aux <script>" not in dot


def test_rank_layout_model_class_name_html_specials_render_cleanly(
    tmp_path: Path,
) -> None:
    """Rank-engine graph caption must escape HTML specials in the class name.

    Defense-in-depth companion to the arg-label fix: ``model_class_name`` is
    interpolated into the top-level graph caption
    (``_render_dot.py:build_and_render_graph``) and the backward/combined
    graph captions (``_render_entrypoints.py``). A class name containing
    ``<``, ``>``, or ``&`` (dynamically constructed via ``type(...)``) must
    not break the Graphviz HTML-like label parser.
    """

    model_cls = type("Loss&Aux<Model>", (nn.Module,), {"forward": lambda self, x: x.relu()})
    trace = tl.trace(model_cls(), torch.randn(3))
    try:
        dot = trace.draw(
            vis_node_placement="rank",
            vis_outpath=str(tmp_path / "rank_class_name_escape"),
            vis_save_only=True,
            vis_fileformat="svg",
            order_siblings=False,
        )
    finally:
        trace.cleanup()

    assert "Loss&amp;Aux&lt;Model&gt;" in dot
    assert "Loss&Aux<Model>" not in dot
