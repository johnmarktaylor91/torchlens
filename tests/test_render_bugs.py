"""Regression tests for Graphviz render failure handling."""

from __future__ import annotations

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
from torchlens._data_classes.trace import Trace
from torchlens._visualization import _render_utils
from torchlens._visualization._rank_layout_internal import layout as rank_layout
from torchlens._visualization._render_common import GraphvizRenderError, GraphvizUnavailableError
from torchlens._visualization._render_utils import render_dot_to_file


class _TinyRenderModel(nn.Module):
    """Small model that produces forward and backward graph nodes."""

    def __init__(self) -> None:
        """Initialize submodules."""
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return torch.relu(self.linear(x)).sum()


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
    trace.backward(trace[trace.output_layers[0]].out)
    return trace


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
        "from torchlens._visualization._render_utils import run_bounded_subprocess\n"
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

    from torchlens._visualization._render_common import GraphvizUnavailableError

    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    empty_bin = tmp_path / "emptybin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    with pytest.raises(GraphvizUnavailableError) as exc_info:
        trace.draw(
            path=str(tmp_path / "graph"),
            order_siblings=False,
        )
    assert exc_info.value.fields["code"] == "graphviz_binary_unavailable"
    assert "graphviz" in exc_info.value.fields["remedy"].lower()
    assert exc_info.value.fields["executable"]


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
                layout="rank",
                path=str(outpath),
                file_format="svg",
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


def test_draw_show_containers_vocabulary_refuses_typed(
    forward_trace: Trace,
    tmp_path: Path,
) -> None:
    """``show_containers`` outside its closed vocabulary refuses typed."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError, match="show_containers") as excinfo:
        forward_trace.draw(
            path=str(tmp_path / "containers_vocab"),
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
            path=str(tmp_path / "forward"),
            file_format="svg",
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
        backward_trace.draw(
            graph="backward",
            path=str(tmp_path / "backward"),
            file_format="svg",
        )


def test_combined_render_timeout_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined rendering raises a typed error when Graphviz times out."""

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _raise_timeout)

    with pytest.raises(GraphvizRenderError, match="timed out.*lowering dpi.*direct SVG.*node cap"):
        backward_trace.draw(
            graph="combined",
            path=str(tmp_path / "combined"),
            file_format="svg",
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
            path=str(tmp_path / "forward_empty"),
            file_format="svg",
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
        backward_trace.draw(
            graph="backward",
            path=str(tmp_path / "backward_empty"),
            file_format="svg",
        )


def test_combined_zero_byte_render_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined rendering raises when Graphviz reports success with an empty file."""

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _write_zero_byte_output)

    with pytest.raises(GraphvizRenderError, match="zero-byte.*lowering dpi.*direct SVG.*node cap"):
        backward_trace.draw(
            graph="combined",
            path=str(tmp_path / "combined_empty"),
            file_format="svg",
        )


def test_backward_called_process_error_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward rendering raises a typed error when Graphviz exits unsuccessfully."""

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _raise_called_process_error)

    with pytest.raises(GraphvizRenderError, match="Graphviz failed.*graphviz failed"):
        backward_trace.draw(
            graph="backward",
            path=str(tmp_path / "backward_failed"),
            file_format="svg",
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
        backward_trace.draw(
            graph="combined",
            path=str(tmp_path / "combined_failed"),
            file_format="svg",
        )

    assert dot_path.exists()
    assert "combined forward/backward graph" in dot_path.read_text()


def test_grad_edges_use_preserved_edge_cluster_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward render passes grad edges the same LCA cluster as dataflow edges."""

    import torchlens._visualization._render_edges as _render_edges

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
        **endpoint_names: str,
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
            **endpoint_names,
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
        trace.backward(trace[trace.output_layers[0]].out)
        trace.draw(
            path=str(tmp_path / "nested_grad"),
            file_format="svg",
            order_siblings=False,
        )
    finally:
        trace.cleanup()

    assert modules_by_edge[("relu", "sigmoid")] == "block.inner:1"


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
            layout="rank",
            path=str(tmp_path / "rank_class_name_escape"),
            file_format="svg",
            order_siblings=False,
        ).dot
    finally:
        trace.cleanup()

    assert "Loss&amp;Aux&lt;Model&gt;" in dot
    assert "Loss&Aux<Model>" not in dot


def _attach_probe_image(trace: Trace) -> Path:
    """Write one PNG into the trace's visualizer scratch root and return it."""

    from PIL import Image

    from torchlens._utils.display import ensure_trace_visualizer_dir

    root = ensure_trace_visualizer_dir(trace)
    image_path = root / "probe.png"
    Image.new("RGB", (12, 12), "red").save(image_path)
    return image_path


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
            file_format="svg",
            path=str(outpath),
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

    from torchlens._visualization import _render_utils

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

    from torchlens._visualization import _render_nodes

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


def test_downstream_intervening_inference_builds_reverse_edges_once() -> None:
    """The reverse-edge map is built once per downstream inference (R52-3).

    ``_infer_intervening_module_downstream`` built the O(E) reverse autograd
    edge map to seed the BFS, then ``_infer_intervening_module_bfs`` rebuilt
    the identical map -- two full ``trace.autograd.grad_fns`` sweeps per intervening
    grad_fn. The prebuilt map is now passed through, so the grad_fn table is
    swept exactly once.
    """

    from types import SimpleNamespace

    from torchlens._visualization._render_leaf import _infer_intervening_module_downstream

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
        autograd=SimpleNamespace(grad_fns=grad_fns),
        _grad_fn_logs={fn.grad_fn_object_id: fn for fn in grad_fns},
    )
    handle = SimpleNamespace(grad_fn_object_id=3, next_grad_fn_ids=[4], op=None)

    _CountingGradFns.iter_calls = 0
    result = _infer_intervening_module_downstream(trace, handle)  # type: ignore[arg-type]

    assert result is None  # no module-anchored grad_fn in the stub chain
    assert _CountingGradFns.iter_calls == 1, (
        f"downstream inference swept trace.autograd.grad_fns {_CountingGradFns.iter_calls} "
        "times -- the duplicate reverse-edge build is back"
    )


def test_rank_layout_without_neato_plugin_names_the_plugin_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Graphviz without its neato plugin raises the typed install error.

    Graphviz 14 packaging ships neato's layout engine as a separate package,
    so the binary runs but exits 1 naming the missing engine; the error must
    carry that remedy instead of a bare neato failure.
    """

    def neato_without_plugin(**kwargs: Any) -> subprocess.CompletedProcess:
        """Return what Graphviz 14's neato prints without its layout plugin."""

        return subprocess.CompletedProcess(
            args=["neato"],
            returncode=1,
            stdout="",
            stderr='There is no layout engine support for "neato"\nUse one of: dot\n',
        )

    monkeypatch.setattr(rank_layout, "_run_neato", neato_without_plugin)
    with pytest.raises(GraphvizUnavailableError) as exc_info:
        rank_layout._run_neato_with_fallbacks(
            rendered_path=str(tmp_path / "graph.svg"),
            source_path=str(tmp_path / "graph"),
            vis_fileformat="svg",
            spline_mode="true",
            render_timeout=30,
        )
    assert isinstance(exc_info.value, RuntimeError)
    assert exc_info.value.fields["code"] == "graphviz_binary_unavailable"
    assert "libgvplugin-neato-layout8" in exc_info.value.fields["remedy"]
    assert "layout='dot'" in exc_info.value.fields["remedy"]
    assert 'no layout engine support for "neato"' in str(exc_info.value)


def test_large_composed_pdf_contains_visible_graph_region(tmp_path: Path) -> None:
    """Large composed PDF renders graph contents inside the page bounds."""

    fitz = pytest.importorskip("fitz")
    model = _LargeChainRenderModel().eval()
    trace = tl.trace(model, torch.randn(1, 4))
    pdf_path = tmp_path / "large_composed.pdf"
    try:
        trace.draw(
            path=str(pdf_path.with_suffix("")),
            file_format="pdf",
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
