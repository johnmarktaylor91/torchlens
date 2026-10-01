"""Regression tests for Graphviz render failure handling."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import graphviz
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.trace import Trace
from torchlens.visualization import _render_utils
from torchlens.visualization._rank_layout_internal import layout as rank_layout
from torchlens.visualization._render_common import GraphvizRenderError
from torchlens.visualization._render_dot import _strip_render_extension
from torchlens.visualization._render_utils import render_dot_to_file
from torchlens.visualization.collapse_plan import RenderContext
from torchlens.visualization.render_ir import build_render_ir


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


class _ModuleDictSpecialKeyModel(nn.Module):
    """Model whose ``nn.ModuleDict`` key contains an HTML-special character."""

    def __init__(self) -> None:
        """Initialize a ModuleDict keyed by a string containing ``&``."""

        super().__init__()
        self.heads = nn.ModuleDict(
            {"score & rank": nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ``"score & rank"`` branch."""

        return self.heads["score & rank"](x)


class _ModuleDictAngleAmpersandKeyModel(nn.Module):
    """Model whose ``nn.ModuleDict`` key mixes ``<``, ``>``, and ``&``."""

    def __init__(self) -> None:
        """Initialize a ModuleDict keyed by a string with mixed DOT-illegal chars."""

        super().__init__()
        self.heads = nn.ModuleDict(
            {"a<b>&c": nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ``"a<b>&c"`` branch."""

        return self.heads["a<b>&c"](x)


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

    trace = tl.trace(
        _TinyRenderModel(),
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
    )
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


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux-only")
def test_bounded_render_child_dies_with_hard_parent_death(tmp_path: Path) -> None:
    """SIGKILL of the parent must not leave a bounded render child running.

    The spawn seam's group teardown only runs in parent exception handlers,
    so hard parent death left the session-leading child alive with no
    watchdog (b6 R40). The pdeathsig binding closes exactly that hole.
    """

    import os
    import signal as signal_module
    import time

    parent_script = (
        "import os, subprocess, sys, threading, time\n"
        "from torchlens.visualization._render_utils import run_bounded_subprocess\n"
        "t = threading.Thread(\n"
        "    target=lambda: run_bounded_subprocess(['sleep', '30'], timeout=60),\n"
        "    daemon=True,\n"
        ")\n"
        "t.start()\n"
        "child = None\n"
        "for _ in range(200):\n"
        "    listing = subprocess.run(\n"
        "        ['ps', '-o', 'pid=,comm=', '--ppid', str(os.getpid())],\n"
        "        capture_output=True, text=True,\n"
        "    ).stdout\n"
        "    for line in listing.splitlines():\n"
        "        pid_text, _, comm = line.strip().partition(' ')\n"
        "        if comm.strip() == 'sleep':\n"
        "            child = int(pid_text)\n"
        "            break\n"
        "    if child is not None:\n"
        "        break\n"
        "    time.sleep(0.05)\n"
        "print(child, flush=True)\n"
        "time.sleep(60)\n"
    )
    worktree_root = str(Path(tl.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": worktree_root}
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_script],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        line = parent.stdout.readline().strip()  # type: ignore[union-attr]
        assert line and line != "None", "parent never spawned the bounded child"
        child_pid = int(line)
        os.kill(parent.pid, signal_module.SIGKILL)
        parent.wait(timeout=5)
        deadline = time.monotonic() + 3.0
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.05)
        if alive:
            os.kill(child_pid, signal_module.SIGKILL)
        assert not alive, "bounded render child survived hard parent death"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)


def test_missing_graphviz_binary_refuses_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing Graphviz binary must refuse typed, not leak FileNotFoundError.

    The spawn seam guarded CalledProcessError/TimeoutExpired but not the
    exec failure itself, so a PATH without ``dot`` escaped ``draw()`` as a
    raw ``FileNotFoundError: 'dot'`` naming neither Graphviz nor the remedy
    (b8 R65).
    """

    from torchlens.visualization._render_common import GraphvizUnavailableError

    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    empty_bin = tmp_path / "emptybin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    with pytest.raises(GraphvizUnavailableError) as exc_info:
        trace.draw(
            vis_outpath=str(tmp_path / "graph"),
            vis_save_only=True,
            order_siblings=False,
        )
    assert exc_info.value.fields["code"] == "graphviz_binary_unavailable"
    assert "graphviz" in exc_info.value.fields["remedy"].lower()
    assert exc_info.value.fields["executable"]


def test_render_ir_honors_skip_fn_without_repeat_folds() -> None:
    """Render IR node/edge topology follows skip-spliced drawing topology."""

    trace = tl.trace(_TinyRenderModel(), torch.randn(2, 3))

    def skip_relu(layer: Any) -> bool:
        """Skip relu layers."""

        return getattr(layer, "layer_type", None) == "relu"

    render_ir = build_render_ir(
        trace,
        collapse_fn=None,
        repeat_folds=None,
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
    """Rank layout should keep code panel content instead of dropping it.

    The model stays bound to a local: a callable ``code_panel`` documents that
    the ORIGINAL model must still be alive at draw time (the Trace holds only
    a weakref), so an anonymous inline model only ever rendered by
    cycle-collection timing luck.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(model, torch.randn(1, 4))

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

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _raise_timeout)
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
    """Simulate a Graphviz timeout from the bounded render runner."""

    raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))


def _raise_called_process_error(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Simulate a Graphviz process failure from the bounded render runner."""

    raise subprocess.CalledProcessError(returncode=1, cmd=args[0], stderr=b"graphviz failed")


def _write_zero_byte_output(args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Simulate a successful Graphviz run that leaves an empty output file."""

    del kwargs
    output_flag_index = args.index("-o")
    Path(args[output_flag_index + 1]).write_bytes(b"")
    return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


def test_draw_bool_flag_kwarg_refuses_non_bool_typed(
    forward_trace: Trace,
    tmp_path: Path,
) -> None:
    """Bool-typed draw kwargs refuse strings with the stable code (R64-F3).

    ``order_siblings='no'`` used to be silently truthy — OFF spelled as a
    string meant ON.
    """

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError, match="order_siblings") as excinfo:
        forward_trace.draw(
            vis_outpath=str(tmp_path / "bool_flag"),
            vis_save_only=True,
            order_siblings="no",  # type: ignore[arg-type]
        )
    assert excinfo.value.fields["code"] == "visualization_bool_option_invalid"


def test_draw_show_containers_vocabulary_refuses_typed(
    forward_trace: Trace,
    tmp_path: Path,
) -> None:
    """``show_containers`` outside its closed vocabulary refuses typed."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError, match="show_containers") as excinfo:
        forward_trace.draw(
            vis_outpath=str(tmp_path / "containers_vocab"),
            vis_save_only=True,
            show_containers="everything",  # type: ignore[arg-type]
        )
    assert excinfo.value.fields["code"] == "visualization_show_containers_invalid"


def test_forward_render_timeout_raises_typed_error(
    forward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward rendering raises a typed error when Graphviz times out."""

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _raise_timeout)

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

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _raise_timeout)

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

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _raise_timeout)

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

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _write_zero_byte_output)

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

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _write_zero_byte_output)

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

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _write_zero_byte_output)

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

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _raise_called_process_error)

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

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _raise_called_process_error)

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

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _raise_called_process_error)

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

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _raise_called_process_error)
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
        capture=tl.options.CaptureOptions(save_grads="all"),
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


def test_module_cluster_title_escapes_html_special_moduledict_key(tmp_path: Path) -> None:
    """Module cluster titles HTML-escape special characters from module addresses.

    Regression test for the module-address cluster-title bug: an
    ``nn.ModuleDict`` key containing ``&`` (e.g. ``"score & rank"``) becomes
    part of the module address used as the cluster title in both render
    engines. ``_setup_subgraphs_recurse`` (``_render_dot.py``) previously
    passed ``title_already_escaped=True`` on the false assumption that Trace
    subgraph titles never contain HTML specials, and the rank-layout
    engine's ``_write_cluster`` (``_rank_layout_internal/layout.py``) built
    its ``cluster_label`` with zero escaping at all. Raw ``&`` in a Graphviz
    HTML-like cluster label breaks the parser, raising
    ``GraphvizRenderError`` on ``draw()`` -- reproduced on the DEFAULT (dot)
    render engine with DEFAULT settings by reverting the fix and re-running
    this exact scenario. Renders to an actual SVG file through the real
    Graphviz binary to verify end-to-end, not just at the string-building
    layer.
    """

    trace = tl.trace(_ModuleDictSpecialKeyModel(), torch.randn(2, 3))
    outpath = tmp_path / "moduledict_special_char"
    try:
        # Must NOT raise GraphvizRenderError -- the raw "&" in the
        # ModuleDict key previously broke the HTML-like label parser
        # mid-render of the "heads.score & rank" module cluster.
        dot = trace.draw(
            vis_outpath=str(outpath),
            vis_save_only=True,
            vis_fileformat="svg",
            order_siblings=False,
        )
    finally:
        trace.cleanup()

    assert "score &amp; rank" in dot
    assert "score & rank<" not in dot  # raw ampersand must not survive
    svg_path = outpath.with_suffix(".svg")
    assert svg_path.exists()
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "score &amp; rank" in svg_text


def test_rank_engine_cluster_subgraph_id_quotes_moduledict_special_key(
    tmp_path: Path,
) -> None:
    """Rank-layout ``_write_cluster`` quotes the subgraph identifier, not just the label.

    Regression test for a gap left by the F4 fix (``bfe71593``): that commit
    HTML-escaped the cluster *label* text (``cluster_label`` in
    ``_write_cluster``) but the raw DOT *subgraph identifier* built two lines
    earlier (``subgraph cluster_{safe} {``) was spliced straight into the DOT
    source with no sanitization at all -- unlike every other raw-DOT
    identifier in this file (node names, edge tail/head names), which route
    through the ``_dot_id()`` quoting helper. An ``nn.ModuleDict`` key
    containing ``<``, ``>``, and ``&`` together produces an unquoted
    subgraph name like ``subgraph cluster_heads_a<b>&c_pass1 {``, which
    ``neato`` rejects with a raw ``syntax error near '>'`` -- a real crash on
    the rank engine, reachable by default whenever ``vis_node_placement=
    "auto"`` promotes to rank for a large enough graph.

    Exercises the *explicit* rank engine (``vis_node_placement="rank"``),
    which the shipped ``test_module_cluster_title_escapes_html_special_
    moduledict_key`` test above never does (it only calls ``draw()`` with
    default -- i.e. dot-engine -- settings), plus the default dot engine for
    parity.
    """

    for engine_kwargs in ({}, {"vis_node_placement": "rank"}):
        trace = tl.trace(_ModuleDictAngleAmpersandKeyModel(), torch.randn(2, 3))
        outpath = tmp_path / f"moduledict_angle_amp{'_rank' if engine_kwargs else '_dot'}"
        try:
            # Must not raise: neither a GraphvizRenderError (dot engine) nor a
            # RuntimeError from a failed `neato` subprocess (rank engine).
            dot = trace.draw(
                vis_outpath=str(outpath),
                vis_save_only=True,
                vis_fileformat="svg",
                order_siblings=False,
                **engine_kwargs,
            )
        finally:
            trace.cleanup()

        # The label is HTML-escaped (F4 fix, still holding)...
        assert "a&lt;b&gt;&amp;c" in dot
        # ...and the subgraph identifier itself must be quoted so the raw
        # `<`/`>`/`&` never appear unescaped/unquoted in the DOT source as a
        # bare identifier.
        assert "subgraph cluster_heads_a<b>&c" not in dot
        svg_path = outpath.with_suffix(".svg")
        assert svg_path.exists()


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
        capture=tl.options.CaptureOptions(capture_container_structure=True),
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


def _attach_probe_image(trace: Trace) -> Path:
    """Write one PNG into the trace's visualizer scratch root and return it."""

    from PIL import Image

    from torchlens.utils.display import ensure_trace_visualizer_dir

    root = ensure_trace_visualizer_dir(trace)
    image_path = root / "probe.png"
    Image.new("RGB", (12, 12), "red").save(image_path)
    return image_path


def test_saved_dot_source_free_of_per_run_temp_imagepath(tmp_path: Path) -> None:
    """User-saved DOT must not bake the per-run mkdtemp visualizer path.

    T9 (grind-p3, LOW) red pin: node images live in a ``tempfile.mkdtemp``
    scratch dir that is removed when the trace is garbage collected. Baking
    that dir into the saved source as a graph-level ``imagepath`` made every
    user-kept DOT unrenderable after the trace died. The image root now
    reaches Graphviz as the render subprocess working directory only; the
    saved source keeps stable relative image refs.
    """

    trace = tl.trace(_TinyRenderModel(), torch.randn(1, 3))
    try:
        image_path = _attach_probe_image(trace)

        def spec_fn(layer_log: Any, default_spec: Any) -> Any:
            if "relu" in str(layer_log.layer_label):
                default_spec.image = str(image_path)
            return default_spec

        outpath = tmp_path / "temp_free"
        trace.draw(
            node_spec_fn=spec_fn,
            vis_fileformat="dot",
            vis_save_only=True,
            vis_outpath=str(outpath),
        )
        source = (tmp_path / "temp_free.dot").read_text()
    finally:
        trace.cleanup()

    assert 'image="probe.png"' in source, "probe image ref missing; test rig broke"
    assert "torchlens_visualizers_" not in source, (
        "saved DOT bakes the per-run mkdtemp visualizer path"
    )
    assert "imagepath" not in source


def test_node_image_renders_without_in_source_imagepath(tmp_path: Path) -> None:
    """Node images still resolve (inlined into SVG) with no in-source root.

    Counterpart guard for the T9 imagepath removal: dropping the in-source
    root must not silently break image rendering — the SVG pipeline inlines
    the probe image as a data URI, which requires Graphviz (running in the
    scratch root) and the inliner to both find it.
    """

    trace = tl.trace(_TinyRenderModel(), torch.randn(1, 3))
    try:
        image_path = _attach_probe_image(trace)

        def spec_fn(layer_log: Any, default_spec: Any) -> Any:
            if "relu" in str(layer_log.layer_label):
                default_spec.image = str(image_path)
            return default_spec

        outpath = tmp_path / "image_inlined"
        trace.draw(
            node_spec_fn=spec_fn,
            vis_fileformat="svg",
            vis_save_only=True,
            vis_outpath=str(outpath),
        )
        svg = (tmp_path / "image_inlined.svg").read_text()
    finally:
        trace.cleanup()

    assert "data:image/png;base64" in svg, (
        "probe node image was not inlined; Graphviz could not resolve the "
        "relative image ref without an in-source imagepath"
    )


def test_final_viewer_child_is_reaped_without_another_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LAST viewer of a process is reaped asynchronously (r3 R40 carried).

    The registry alone reaped only at the NEXT launch, so one draw() that
    opened a viewer left one zombie for the process lifetime. Drive the real
    _open_file_quietly with a stub opener and require the child to be waited
    on and released with NO later launch.
    """

    import os
    import time

    from torchlens.visualization import _render_utils

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    opener_name = "open" if sys.platform == "darwin" else "xdg-open"
    stub = stub_dir / opener_name
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)

    target = tmp_path / "artifact.pdf"
    target.write_text("stub")
    assert _render_utils._open_file_quietly(str(target)) is True
    assert len(_render_utils._VIEWER_PROCS) <= 1

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and _render_utils._VIEWER_PROCS:
        time.sleep(0.05)
    # RED before the fix: the exited child stayed registered (and unreaped,
    # i.e. a zombie) until another viewer launch.
    assert _render_utils._VIEWER_PROCS == []


def test_tooltip_reprs_mask_memory_addresses() -> None:
    """DOT tooltip reprs must never embed live memory addresses.

    r19 (b6-fable carried LOW): a default-repr object in a decoded-output
    mapping, or a custom Sequence batch container with the default
    ``object.__repr__``, leaked ``0x...`` addresses into the DOT bytes,
    making otherwise-identical renders nondeterministic across processes.
    """

    import re

    from torchlens.visualization import _render_nodes

    address = re.compile(r"0x[0-9a-fA-F]{4,}")

    class _Opaque:
        """Object with the default address-bearing repr."""

    mapping_attrs = _render_nodes._render_raw_output({"key": _Opaque()})
    assert mapping_attrs is not None
    assert not address.search(mapping_attrs["tooltip"])

    class _StrBatch(Sequence):
        """Sequence of strings with the default address-bearing repr."""

        def __init__(self, items: list[str]) -> None:
            self._items = items

        def __getitem__(self, index: int) -> str:
            return self._items[index]

        def __len__(self) -> int:
            return len(self._items)

    trace = tl.trace(nn.Identity(), torch.randn(2, 4))
    batch_attrs = _render_nodes._render_raw_input(
        trace, _StrBatch(["alpha", "beta"]), batch_render="all"
    )
    assert batch_attrs is not None
    assert not address.search(batch_attrs["tooltip"])


class _TwoOutInner(nn.Module):
    """Module returning TWO tensors, both consumed by one exterior op."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.lin(x)
        return torch.relu(y), torch.tanh(y)


class _TwoOutOuter(nn.Module):
    """Consumes both inner outputs in a single exterior add."""

    def __init__(self) -> None:
        super().__init__()
        self.two = _TwoOutInner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.two(x)
        return a + b


def test_collapsed_module_boundary_edge_discloses_multiplicity(tmp_path: Path) -> None:
    """Distinct dataflow edges merged by collapse must disclose multiplicity.

    r19 (b6-fable carried LOW, 3rd round): a collapsed module returning two
    tensors, both consumed by one exterior ``add``, rendered as ONE
    unlabeled edge — the boundary-crossing distinct-dataflow edges were
    visually deduped with no disclosure. The merged edge must carry an
    ``x2`` label.
    """

    trace = tl.trace(_TwoOutOuter(), torch.randn(2, 4))
    outpath = tmp_path / "twoout"
    trace.draw(
        vis_call_depth=1,
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(outpath),
    )
    source = (tmp_path / "twoout.dot").read_text()
    boundary_edge_lines = [
        index for index, line in enumerate(source.splitlines()) if "twopass1 -> add" in line
    ]
    assert len(boundary_edge_lines) == 1
    lines = source.splitlines()
    start = boundary_edge_lines[0]
    edge_stanza = "\n".join(lines[start : start + 6])
    assert "x2" in edge_stanza, (
        "two distinct dataflow edges merged into one rendered edge with no multiplicity disclosure"
    )


def test_downstream_intervening_inference_builds_reverse_edges_once() -> None:
    """The reverse-edge map is built once per downstream inference (R52-3).

    ``_infer_intervening_module_downstream`` built the O(E) reverse autograd
    edge map to seed the BFS, then ``_infer_intervening_module_bfs`` rebuilt
    the identical map -- two full ``trace.grad_fns`` sweeps per intervening
    grad_fn. The prebuilt map is now passed through, so the grad_fn table is
    swept exactly once.
    """

    from types import SimpleNamespace

    from torchlens.visualization._render_leaf import _infer_intervening_module_downstream

    class _CountingGradFns(list):
        """Grad-fn table that counts full sweeps."""

        iter_calls = 0

        def __iter__(self) -> Any:
            type(self).iter_calls += 1
            return super().__iter__()

    grad_fns = _CountingGradFns(
        SimpleNamespace(grad_fn_object_id=i, next_grad_fn_ids=[i + 1], op=None) for i in range(5)
    )
    trace = SimpleNamespace(
        grad_fns=grad_fns,
        grad_fn_logs={fn.grad_fn_object_id: fn for fn in grad_fns},
    )
    handle = SimpleNamespace(grad_fn_object_id=3, next_grad_fn_ids=[4], op=None)

    _CountingGradFns.iter_calls = 0
    result = _infer_intervening_module_downstream(trace, handle)  # type: ignore[arg-type]

    assert result is None  # no module-anchored grad_fn in the stub chain
    assert _CountingGradFns.iter_calls == 1, (
        f"downstream inference swept trace.grad_fns {_CountingGradFns.iter_calls} "
        "times -- the duplicate reverse-edge build is back"
    )


def test_code_panel_tooltip_shows_basename_not_absolute_path() -> None:
    """The visible source-link tooltip must not leak the absolute host path.

    Hunt-6 R62-2: ``draw(code_panel=True)`` embedded the absolute
    (username-bearing) source path in BOTH the ``vscode://file`` HREF and the
    visible tooltip. The HREF keeps the absolute path -- local editor
    clickability is the deliberate feature, disclosed in limitations.md --
    but the tooltip now shows the basename only.
    """

    from types import SimpleNamespace

    from torchlens.visualization.code_panel import _source_text_to_html_rows

    source_text = SimpleNamespace(
        file_path="/home/canary_user_zq81/models/canary_src.py",
        line_number=21,
    )
    rows = _source_text_to_html_rows(source_text, ["def forward(self, x):"])  # type: ignore[arg-type]
    link_row = rows[0]

    assert "vscode://file//home/canary_user_zq81/models/canary_src.py:21" in link_row
    tooltip = link_row.split("TOOLTIP='", 1)[1].split("'", 1)[0]
    assert "canary_src.py" in tooltip
    assert "canary_user_zq81" not in tooltip
    assert "/home/" not in tooltip
