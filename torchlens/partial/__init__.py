"""Partial capture helpers for failed TorchLens forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from torchlens.data_classes.layer_pass_log import LayerPassLog
    from torchlens.data_classes.model_log import ModelLog


@dataclass(frozen=True)
class PartialModelLog:
    """Thin wrapper around raw capture state from a failed forward pass.

    Parameters
    ----------
    model_log:
        Partially populated ``ModelLog`` whose raw layer entries were captured
        before the exception.
    original_exception:
        Exception raised by the failed capture.
    """

    model_log: ModelLog
    original_exception: BaseException

    @classmethod
    def from_model_log(cls, model_log: ModelLog, exception: BaseException) -> "PartialModelLog":
        """Build a partial log wrapper from a failed capture's internal state.

        Parameters
        ----------
        model_log:
            ``ModelLog`` instance active when capture failed.
        exception:
            Original exception raised during capture.

        Returns
        -------
        PartialModelLog
            Wrapper exposing minimal inspection and graph rendering helpers.
        """

        return cls(model_log=model_log, original_exception=exception)

    @property
    def raw_layers(self) -> tuple[LayerPassLog, ...]:
        """Return raw layer entries captured before failure.

        Returns
        -------
        tuple[LayerPassLog, ...]
            Raw layer pass logs in capture order.
        """

        return tuple(getattr(self.model_log, "_raw_layer_dict", {}).values())

    def first_nonfinite(self) -> str:
        """Return a text summary of the first raw non-finite tensor.

        Returns
        -------
        str
            Human-readable summary with layer, operation, shape, dtype, and parents.
        """

        for layer in self.raw_layers:
            activation = getattr(layer, "activation", None)
            candidate = activation if isinstance(activation, torch.Tensor) else None
            if candidate is None or candidate.numel() == 0:
                continue
            try:
                has_nonfinite = bool((~torch.isfinite(candidate.detach())).any().item())
            except (RuntimeError, TypeError):
                continue
            if has_nonfinite:
                parents = ", ".join(getattr(layer, "parent_layers", None) or []) or "none"
                return (
                    "First non-finite captured tensor is in "
                    f"layer {getattr(layer, 'tensor_label_raw', 'unknown')} "
                    f"(op {getattr(layer, 'func_name', 'unknown')}), "
                    f"shape={getattr(layer, 'tensor_shape', None)}, "
                    f"dtype={getattr(layer, 'tensor_dtype', None)}, parents={parents}."
                )
        fields = getattr(self.original_exception, "fields", {})
        if "layer" in fields:
            return (
                "First non-finite captured tensor is in "
                f"layer {fields.get('layer')} (op {fields.get('op')}), "
                f"shape={fields.get('shape')}, dtype={fields.get('dtype')}, "
                f"parents={fields.get('parents', [])}."
            )
        return "No non-finite tensor values found in partial capture."

    def render_graph(self, vis_outpath: str | None = None, **_: Any) -> str:
        """Render the failed capture as minimal Graphviz DOT source.

        Parameters
        ----------
        vis_outpath:
            Accepted for API symmetry; no file is written by this minimal renderer.
        **_:
            Ignored rendering keyword arguments accepted for compatibility.

        Returns
        -------
        str
            DOT source for the raw operations captured before failure.
        """

        lines = [
            "digraph torchlens_partial {",
            '  graph [label="TorchLens partial capture", labelloc=t];',
            '  node [shape=box, style="rounded"];',
        ]
        for layer in self.raw_layers:
            raw_label = _raw_label(layer)
            shape = getattr(layer, "tensor_shape", None)
            dtype = getattr(layer, "tensor_dtype", None)
            func_name = getattr(layer, "func_name", "unknown")
            label = f"{raw_label}\\nop={func_name}\\nshape={shape}\\ndtype={dtype}"
            lines.append(f'  "{_dot_escape(raw_label)}" [label="{_dot_escape(label)}"];')
            for parent in getattr(layer, "parent_layers", []) or []:
                lines.append(f'  "{_dot_escape(str(parent))}" -> "{_dot_escape(raw_label)}";')
        failure_label = _failure_label(self.original_exception)
        lines.append(f'  "__failure__" [shape=note, label="{_dot_escape(failure_label)}"];')
        if self.raw_layers:
            lines.append(f'  "{_dot_escape(_raw_label(self.raw_layers[-1]))}" -> "__failure__";')
        lines.append("}")
        return "\n".join(lines)

    def show(self, method: Literal["graph", "repr", "html"] = "graph", **kwargs: Any) -> str:
        """Display this partial log using a small set of inspection modes.

        Parameters
        ----------
        method:
            ``"graph"`` returns DOT, ``"repr"`` returns ``repr(self)``, and
            ``"html"`` returns a compact HTML fragment.
        **kwargs:
            Forwarded to ``render_graph`` for graph mode.

        Returns
        -------
        str
            Rendered graph, representation, or HTML fragment.
        """

        if method == "graph":
            return self.render_graph(**kwargs)
        if method == "repr":
            return repr(self)
        if method == "html":
            return self._repr_html_()
        raise ValueError("method must be 'graph', 'repr', or 'html'.")

    def _repr_html_(self) -> str:
        """Return a compact notebook HTML representation.

        Returns
        -------
        str
            HTML fragment summarizing the failed capture.
        """

        summary = escape(self.first_nonfinite())
        error = escape(str(self.original_exception))
        return (
            "<div><b>PartialModelLog</b>"
            f"<div>raw_layers={len(self.raw_layers)}</div>"
            f"<div>{summary}</div>"
            f"<div>error={error}</div></div>"
        )

    def __repr__(self) -> str:
        """Return a concise partial capture representation.

        Returns
        -------
        str
            Debug representation with raw layer count and original error type.
        """

        return (
            f"PartialModelLog(raw_layers={len(self.raw_layers)}, "
            f"error={type(self.original_exception).__name__})"
        )


def from_failed_capture(exception: BaseException) -> PartialModelLog:
    """Return the partial log attached to a failed TorchLens capture exception.

    Parameters
    ----------
    exception:
        Exception raised by ``torchlens.log_forward_pass``.

    Returns
    -------
    PartialModelLog
        Partial capture wrapper attached as ``exception.partial_log``.

    Raises
    ------
    ValueError
        If the exception does not carry TorchLens partial capture state.
    """

    partial_log = getattr(exception, "partial_log", None)
    if isinstance(partial_log, PartialModelLog):
        return partial_log
    raise ValueError("exception does not contain a TorchLens partial capture")


def _raw_label(layer: LayerPassLog) -> str:
    """Return the raw label for a captured layer entry.

    Parameters
    ----------
    layer:
        Raw layer pass log.

    Returns
    -------
    str
        Raw tensor label, falling back to the raw layer label.
    """

    return str(getattr(layer, "tensor_label_raw", getattr(layer, "layer_label_raw", "unknown")))


def _dot_escape(value: str) -> str:
    """Escape a value for a Graphviz DOT string literal.

    Parameters
    ----------
    value:
        String value to escape.

    Returns
    -------
    str
        Escaped string.
    """

    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _failure_label(exception: BaseException) -> str:
    """Return a concise failure label for DOT output.

    Parameters
    ----------
    exception:
        Original capture exception.

    Returns
    -------
    str
        Failure label including the exception type and message.
    """

    return f"{type(exception).__name__}: {exception}"


__all__ = ["PartialModelLog", "from_failed_capture"]
