"""Seam adapters: cell density + echo observer (F10; lovely item 10).

Three consumers read the ONE TensorStats record without re-deriving the
grammar (D1: no formatter touches a tensor; a number may never exist only
in HTML):

- ``summary_cell`` -- the CELL density for the summary panel's value
  columns (F08 renders it verbatim; the seam test pins byte equality with
  the line renderer's distribution zone).
- ``treescope_card_fields`` -- the semantic fixture the treescope/notebook
  card renderers (F16) consume: named fields, honesty tokens, and the
  ascii/unicode sparkline pair, so an HTML card can never invent a number.
- ``EchoObserver`` / ``echo`` -- the live line density mounted on the
  shipped ``hooks=``/observer surface: one core line per firing, address
  envelope prefixed, computed entirely under ``pause_logging`` (capture
  hooks wrap EVERY tensor read).

Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from ._stats_render import render_core_line, sparkline
from ._tensor_stats import TensorStats, tensor_stats


def summary_cell(stats: TensorStats, *, max_width: int = 40, style: str = "ascii") -> str:
    """Render the bounded CELL density for one record (summary-panel seam).

    The cell is the core line's distribution zone under a hard width bound:
    the D14 retention order applies (extrema/health survive; bytes, ``n=``,
    and the sparkline drop first), so a table column never wraps.

    Parameters
    ----------
    stats:
        The frozen record (the caller computed it ONCE; a cell render must
        never be a compute trigger).
    max_width:
        Column budget in characters.
    style:
        ``"ascii"`` (canonical) or ``"unicode"``.

    Returns
    -------
    str
        The cell text, byte-identical to the line renderer's output at the
        same width (one grammar, two densities).
    """

    return render_core_line(stats, style=style, max_width=max_width)


def treescope_card_fields(stats: TensorStats) -> dict[str, Any]:
    """Semantic fixture for HTML card renderers (treescope seam).

    Returns named FACTS, not markup: the card renderer may lay them out
    freely but can never show a number this record does not carry (the
    matrix rule: a number may never exist only in HTML).

    Parameters
    ----------
    stats:
        The frozen record.

    Returns
    -------
    dict[str, Any]
        ``line_ascii``/``line_unicode`` (full core, both encodings),
        ``sparkline_ascii``/``sparkline_unicode`` (or None below the D13
        minimum), scalar facts, and per-family evidence tokens.
    """

    def evidence_token(evidence: Any) -> str:
        """One family's policy disclosure token."""

        if evidence.policy == "sampled" and evidence.sample_size:
            return f"sampled {evidence.sample_size}/{evidence.population}"
        if evidence.policy == "unavailable":
            return f"unavailable({evidence.reason})"
        return "exact"

    counts = stats.histogram_counts
    return {
        "line_ascii": render_core_line(stats, style="ascii"),
        "line_unicode": render_core_line(stats, style="unicode"),
        "sparkline_ascii": None if counts is None else sparkline(counts, style="ascii"),
        "sparkline_unicode": None if counts is None else sparkline(counts, style="unicode"),
        "dtype": stats.dtype,
        "shape": stats.shape,
        "device": stats.device,
        "numel": stats.numel,
        "nbytes": stats.nbytes,
        "finite_min": stats.finite_min,
        "finite_max": stats.finite_max,
        "mean": stats.mean,
        "sd": stats.sd,
        "zero_count": stats.zero_count,
        "nan_count": stats.nan_count,
        "posinf_count": stats.posinf_count,
        "neginf_count": stats.neginf_count,
        "magnitude_basis": stats.magnitude_basis,
        "evidence": {
            "nonfinite": evidence_token(stats.nonfinite_evidence),
            "extrema": evidence_token(stats.extrema_evidence),
            "mean": evidence_token(stats.mean_evidence),
            "sd": evidence_token(stats.sd_evidence),
            "histogram": evidence_token(stats.histogram_evidence),
        },
    }


@dataclass
class EchoObserver:
    """Live stats-line echo mounted on the shipped observer surface.

    Mounts exactly like a tap
    (``capture=CaptureOptions(intervention_ready=True, hooks=echo_obs)``,
    the ``site``/``direction``/``__call__(out, *, hook)`` protocol) and
    returns the observed value unchanged. Each firing appends one
    ``address -> core`` line to ``lines`` and forwards it to ``sink`` when
    given. Stats computation and rendering run under ``pause_logging`` so
    the echo can never contaminate the capture with its own torch ops.
    """

    site: Any = None
    direction: Literal["forward", "backward", "both"] = "forward"
    sink: Callable[[str], None] | None = None
    style: Literal["ascii", "unicode"] = "ascii"
    lines: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Bounded identity card: never dump the accumulated lines (D31)."""

        return (
            f"EchoObserver(site={self.site!r}, {len(self.lines)} lines echoed, style={self.style})"
        )

    def __call__(self, out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        """Record one firing; the observed value passes through unchanged."""

        from .._state import pause_logging
        from ..utils.fail_open import fail_open

        with pause_logging():
            address = self._address(hook)
            # Never break a forward from an echo: degrade to the typed token.
            core = fail_open(
                lambda: render_core_line(tensor_stats(out, identity=address), style=self.style),
                lambda error: f"(echo unavailable: {type(error).__name__})",
            )
            line = f"{address} -> {core}" if address else core
            self.lines.append(line)
            if self.sink is not None:
                self.sink(line)
        return out

    @staticmethod
    def _address(hook: Any) -> str | None:
        """Best-effort site address from the hook context."""

        layer_log = getattr(hook, "layer_log", None)
        if layer_log is None:
            return None
        if hasattr(layer_log, "get"):
            return layer_log.get("label") or layer_log.get("layer_label")
        return getattr(layer_log, "label", None) or getattr(layer_log, "layer_label", None)


def echo(
    site: Any,
    *,
    sink: Callable[[str], None] | None = None,
    direction: Literal["forward", "backward", "both"] = "forward",
    style: Literal["ascii", "unicode"] = "ascii",
) -> EchoObserver:
    """Create an echo observer (the live line density) for one site.

    Mounts on the shipped hook surface exactly like ``tl.tap``::

        obs = tl.stats.echo(tl.func("relu"), sink=print)
        tl.trace(model, x,
                 capture=tl.options.CaptureOptions(
                     intervention_ready=True, hooks=obs))

    Parameters
    ----------
    site:
        Selector-like site to observe (any ``tap()``-accepted selector).
    sink:
        Optional per-line consumer (e.g. ``print`` or a logger method);
        lines always also accumulate on ``observer.lines``.
    direction:
        Hook direction, as on ``tl.tap``.
    style:
        Rendering style for the echoed cores.

    Returns
    -------
    EchoObserver
        Callable observer for the ``hooks=`` surface.
    """

    return EchoObserver(site=site, direction=direction, sink=sink, style=style)
