"""Capture each fixture and draw each panel with the exact TorchLens call the slide names.

One capture per panel, serial, never in threads. Each panel writes ``<stem>.svg`` (the real
render), ``<stem>.dot`` (the DOT source TorchLens returned) and ``<stem>.json`` (Graphviz
``-Tjson`` of that DOT, the attributes the key selectors read). Callable tokens in the
slide table (``"@name"``) resolve here.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import graphviz
import torch

import torchlens as tl
from scripts.visual_language.models import FIXTURES
from scripts.visual_language.slides import CONVENTIONS, Panel, Slide
from torchlens.visualization import lenses
from torchlens.visualization._encoding import EncodingChannelRequest


def _badge(op: Any, spec: Any) -> Any:
    """A user node_spec_fn: gold fill on the tanh op only."""

    if "tanh" in str(getattr(op, "layer_label", "")):
        return spec.replace(fillcolor="#FFE9A8")
    return None


def _boxbadge(module: Any, spec: Any) -> Any:
    """A user collapsed_node_spec_fn: gold fill on every collapsed module."""

    return spec.replace(fillcolor="#FFE9A8")


class _ToTensor:
    """A named input transform (a plain lambda would print an address that changes per run)."""

    def __call__(self, images: Any) -> torch.Tensor:
        return torch.ones(len(images), 3, 8, 8)

    def __repr__(self) -> str:
        return "to_tensor"


class _Decode:
    """Label and score rows for the first image (named, like ``_ToTensor``)."""

    def __call__(self, logits: torch.Tensor) -> list[tuple[str, float]]:
        probs = logits.softmax(-1)[0]
        return [(name, round(float(p), 2)) for name, p in zip(("red", "green", "blue"), probs)]

    def __repr__(self) -> str:
        return "decode"


CALLABLES: Mapping[str, Callable[[], Any]] = {
    "@skip_reshape": lambda: lambda layer: layer.layer_type == "reshape",
    "@exclude_reshapes": lambda: lenses.DisplayFilter(exclude="reshapes"),
    "@bytes_linear": lambda: EncodingChannelRequest(source="bytes", transform="linear"),
    "@bytes_rank": lambda: EncodingChannelRequest(source="bytes", transform="rank"),
    "@bytes_log": lambda: EncodingChannelRequest(source="bytes", transform="log"),
    "@score_map": lambda: {"mul_1_1": 0.8},
    "@unsaved_abs": lambda: lambda op: "abs" not in str(op.func_name),
    "@badge": lambda: _badge,
    "@boxbadge": lambda: _boxbadge,
    "@to_tensor": _ToTensor,
    "@decode": _Decode,
}

INTERVENTIONS: Mapping[str, Callable[[], Any]] = {
    "zero_tanh": lambda: tl.when(tl.func("tanh"), tl.zero_ablate()),
}


def _resolve(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("@"):
        return CALLABLES[value]()
    return value


def _fork_edit(trace: Any) -> Any:
    fork = trace.fork()
    fork.do(tl.units("linear_1_1:1", [(0, 0)]).resolve(fork), tl.zero_ablate())
    return fork


def _log_backward(trace: Any) -> Any:
    trace.log_backward(trace[trace.output_layers[0]].out)
    return trace


def _higher_order(trace: Any, inputs: Any) -> Any:
    loss = trace[trace.output_layers[0]].out
    first = torch.autograd.grad(loss, inputs, create_graph=True, retain_graph=True)[0]
    # backward(), not grad(): the second pass accumulates into the leaf, so accum is drawn.
    first.sum().backward(retain_graph=True)
    return trace


def capture(panel: Panel) -> tuple[Any, Any]:
    """Build the fixture and capture it with the panel's capture options."""

    model, inputs = FIXTURES[panel.fixture]()
    kwargs: dict[str, Any] = {}
    if panel.capture:
        options = {k: _resolve(v) for k, v in panel.capture.items()}
        kwargs["capture"] = tl.options.CaptureOptions(**options)
    if panel.intervene:
        kwargs["intervene"] = INTERVENTIONS[panel.intervene]()
    trace = tl.trace(model, inputs, **kwargs)
    if panel.prep == "log_backward":
        trace = _log_backward(trace)
    elif panel.prep == "higher_order":
        trace = _higher_order(trace, inputs)
    return trace, inputs


def draw_kwargs(panel: Panel) -> dict[str, Any]:
    """The panel's draw arguments: conventions first, then the panel's own (which win)."""

    merged: dict[str, Any] = dict(CONVENTIONS) if panel.conventions else {}
    if panel.call in ("draw_backward", "draw_combined"):
        merged.pop("direction", None)
        merged.pop("font_size", None)
        merged.pop("collapse", None)
        merged.pop("node_label_fields", None)
        merged["vis_direction"] = "leftright"
    if panel.call == "lens":
        # A lens chooses its own rows; the deck's two-row convention would override it.
        merged.pop("node_label_fields", None)
    if panel.call in ("surgery_diff",):
        merged = {}
    merged.update({k: _resolve(v) for k, v in panel.kwargs.items()})
    return merged


def draw(panel: Panel, trace: Any, outpath: Path) -> str:
    """Run the panel's draw call; return the DOT source."""

    kwargs = draw_kwargs(panel)
    kwargs["vis_outpath"] = str(outpath)
    if panel.call == "draw":
        return str(trace.draw(**kwargs))
    if panel.call == "draw_backward":
        return str(trace.draw_backward(**kwargs))
    if panel.call == "draw_combined":
        return str(trace.draw_combined(**kwargs))
    if panel.call == "lens":
        lens = kwargs.pop("lens")
        return str(lenses.draw_with_lens(trace, lens, **kwargs))
    if panel.call == "surgery":
        from torchlens.visualization.surgery_visuals import render_surgery

        fork = _fork_edit(trace)
        graph = render_surgery(fork, **kwargs)
        return str(getattr(graph, "source", graph))
    if panel.call == "surgery_diff":
        from torchlens.visualization import surgery_diff

        fork = _fork_edit(trace)
        from torchlens.visualization._surgery_diff import _build_diff_dot

        diff = surgery_diff(fork, trace)
        diff.draw(str(outpath), vis_fileformat="svg", vis_save_only=True)
        # draw() writes only the SVG and the census; the DOT it rendered is rebuilt here.
        return str(_build_diff_dot(diff, theme="torchlens").source)
    raise ValueError(f"unknown call {panel.call!r}")


def _svg_path(outpath: Path) -> Path:
    for candidate in (outpath.with_suffix(".svg"), Path(f"{outpath}.svg")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no SVG written for {outpath}")


def render_panel(slide: Slide, panel: Panel, out: Path) -> dict[str, Any]:
    """Capture and draw one panel; write SVG, DOT and JSON; return a record."""

    stem = f"{slide.id}-{panel.name}"
    outpath = out / stem
    record: dict[str, Any] = {"slide": slide.id, "panel": panel.name, "stem": stem}
    caught = io.StringIO()
    with warnings.catch_warnings(record=True) as seen, contextlib.redirect_stdout(caught):
        warnings.simplefilter("always")
        trace, _inputs = capture(panel)
        if panel.kwargs.get("return_graph"):
            graph = trace.draw(**{**draw_kwargs(panel), "vis_outpath": str(outpath)})
            record["returned"] = type(graph).__name__
            dot_source = graph.source
            graph.render(str(outpath), format="svg", cleanup=True)
        else:
            dot_source = draw(panel, trace, outpath)
    record["warnings"] = sorted({f"{w.category.__name__}: {w.message}"[:300] for w in seen})
    (out / f"{stem}.dot").write_text(dot_source)
    fmt = str(draw_kwargs(panel).get("vis_fileformat", "svg"))
    if fmt == "png":
        png = outpath.with_suffix(".png") if outpath.with_suffix(".png").exists() else None
        png = png or Path(f"{outpath}.png")
        record["png_bytes"] = png.stat().st_size if png.exists() else 0
        record["png_size"] = _png_size(png) if png.exists() else None
        svg_text = graphviz.Source(dot_source).pipe(format="svg").decode()
        (out / f"{stem}.svg").write_text(svg_text)
    else:
        svg = _svg_path(outpath)
        if svg != out / f"{stem}.svg":
            shutil.move(str(svg), out / f"{stem}.svg")
    json_text = graphviz.Source(dot_source).pipe(format="json").decode()
    (out / f"{stem}.json").write_text(json_text)
    return record


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[16:24]
    return int.from_bytes(data[:4], "big"), int.from_bytes(data[4:], "big")


def probe_export(raw: Path) -> dict[str, str]:
    """Measure the export controls on ``Flow``: raster sizes, the returned object, refusals."""

    import inspect
    import re

    from torchlens.data_classes._trace_viz import TraceVisualizationMixin

    defaults = inspect.signature(TraceVisualizationMixin.draw).parameters
    model, inputs = FIXTURES["Flow"]()
    trace = tl.trace(model, inputs)
    facts: dict[str, str] = {}
    base = {"direction": "leftright", "vis_save_only": True}
    for dpi in (96, 192):
        path = raw / f"export-png-{dpi}"
        trace.draw(vis_outpath=str(path), vis_fileformat="png", dpi=dpi, **base)
        png = path.with_suffix(".png") if path.with_suffix(".png").exists() else Path(f"{path}.png")
        width, height = _png_size(png)
        facts[f'vis_fileformat="png", dpi={dpi}'] = f"{width} x {height} pixels"
    svg_path = raw / "export-svg"
    trace.draw(vis_outpath=str(svg_path), vis_fileformat="svg", **base)
    svg_file = svg_path.with_suffix(".svg") if svg_path.with_suffix(".svg").exists() else None
    svg_file = svg_file or Path(f"{svg_path}.svg")
    view = re.search(r'viewBox="([^"]+)"', svg_file.read_text())
    facts['vis_fileformat="svg"'] = (
        f"vector, viewBox {view.group(1) if view else '?'}; dpi has no effect"
    )
    graph = trace.draw(return_graph=True, vis_outpath=str(raw / "export-graph"), **base)
    facts["return_graph=True"] = f"returns a {type(graph).__module__}.{type(graph).__name__} object"
    facts["vis_save_only"] = (
        f"default {defaults['vis_save_only'].default!r}: True writes without opening a viewer"
    )
    facts["vis_outpath"] = f"default {defaults['vis_outpath'].default!r} (the file stem)"
    facts["vis_fileformat"] = f"default {defaults['vis_fileformat'].default!r}"
    facts['view="none"'] = f"draws nothing and returns {trace.draw(view='none')!r}"
    try:
        trace.draw(view="sideways", vis_outpath=str(raw / "export-refused"), **base)
        facts['view="sideways"'] = "accepted (unexpected)"
    except Exception as exc:  # the refusal is the measurement
        first = str(exc).strip().splitlines()[0][:150]
        facts['view="sideways"'] = f"refused before any render: {type(exc).__name__}: {first}"
    return facts
