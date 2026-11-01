"""Door epilogue helpers for the FINALIZE stage.

Moved verbatim from ``torchlens._user_funcs`` by the capture seam.
"""

import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..._data_classes.trace import (
    Trace,
)
from ..._utils.display import ensure_trace_visualizer_dir
from ...intervention.resolver import _resolve_sites

if TYPE_CHECKING:
    from ..stages import CaptureRequest, ForwardResult


def _render_layer_visualizers(
    trace: Trace,
    layer_visualizers: dict[Any, Callable[..., Any]],
) -> None:
    """Render configured per-layer visualizer thumbnails after capture.

    Parameters
    ----------
    trace:
        Completed trace whose layer outs may be rendered.
    layer_visualizers:
        Mapping from TorchLens site selectors to visualizer callables.
    """

    output_dir = ensure_trace_visualizer_dir(trace)
    visualizer_dir = output_dir / "visualizers"
    visualizer_dir.mkdir(parents=True, exist_ok=True)
    max_fanout = max(1, len(trace.layer_list))

    for selector, visualizer in layer_visualizers.items():
        try:
            selected_ops = tuple(_resolve_sites(trace, selector, max_fanout=max_fanout))
        except Exception as exc:
            warnings.warn(
                f"Skipping layer visualizer for selector {selector!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for op in selected_ops:
            _render_one_layer_visualizer(visualizer_dir, op, visualizer)


def _render_one_layer_visualizer(
    visualizer_dir: Path,
    op: Any,
    visualizer: Callable[..., Any],
) -> None:
    """Render one op visualizer and store its output path on the op.

    Parameters
    ----------
    visualizer_dir:
        Directory that receives rendered files.
    op:
        Layer operation to render.
    visualizer:
        Callable accepting ``(tensor, *, layer_label=None)``.
    """

    if not bool(getattr(op, "has_saved_activation", False)) or getattr(op, "out", None) is None:
        return
    try:
        rendered = visualizer(op.out, layer_label=getattr(op, "layer_label", None))
        if rendered is None:
            return
        safe_label = _safe_visualizer_filename(str(op.layer_label))
        if isinstance(rendered, str):
            output_path = visualizer_dir / f"{safe_label}.html"
            output_path.write_text(rendered, encoding="utf-8")
        else:
            output_path = visualizer_dir / f"{safe_label}.png"
            rendered.save(output_path)
        op.visualizer_path = str(output_path)
    except Exception as exc:
        warnings.warn(
            f"Layer visualizer failed for {getattr(op, 'layer_label', '<unknown>')}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _safe_visualizer_filename(label: str) -> str:
    """Return a filesystem-safe visualizer basename for a layer label.

    Parameters
    ----------
    label:
        Layer label to encode.

    Returns
    -------
    str
        Safe filename stem.
    """

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label.replace(":", "pass")).strip("_") or "layer"


def finish_trace(result: ForwardResult, request: CaptureRequest) -> Trace:
    """Return the finished trace of one door call.

    Seam entry: the door epilogue still runs inline at the end of
    ``torchlens._capture.prepare.door._trace_torch_model``; the FINALIZE lane
    moves it behind this name.

    Parameters
    ----------
    result:
        Forward result of the run.
    request:
        The request that started the run.

    Returns
    -------
    Trace
        The finished trace.
    """
    del request
    return result.trace
