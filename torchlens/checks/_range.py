"""Live semantic-range and liveness probes over the PUBLIC phase seam (D14).

The value-probing ``capture.hooks`` path is public and adds ZERO ops to a
trace (measured 13/13/13, BatchNorm included), so what was missing is only
the register: named findings, dedup, severity/action, lifecycle, report
integration. These probes are implemented 100% over that public seam --
plain ``fn(out, *, hook)`` callables handed to
``CaptureOptions(hooks=probe.hook_plan())`` -- and that implementation IS
the seam's CI conformance test: if a probe ever needs private capture
state, the seam is too weak and gets fixed (memo D14).

Liveness records ONLY zero-fraction / running-max facts and points at
``tl.dead``: its multi-sample upper-bound epistemics stay the ONLY dead-unit
verdict, and no finding here ever contains the word "dead".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .. import _state
from ..observability._chassis import EventStream, ObserverEvent
from ._errors import CheckConfigError
from ._records import CheckFinding, severity_sorted

__tl_layer__ = "L5"


def _fired_label(hook_ctx: Any, fallback: str) -> str:
    """Read the fired site's label from the public hook context."""

    layer_log = getattr(hook_ctx, "layer_log", None)
    if isinstance(layer_log, Mapping):
        for key in ("layer_label", "label", "layer_label_w_pass"):
            value = layer_log.get(key)
            if value:
                return str(value)
    return fallback


class RangeProbe:
    """Opt-in live semantic range check per site (memo D14).

    Parameters
    ----------
    bounds:
        Site target -> ``(low, high)`` declared semantic bounds (either
        side ``None``). Site targets are anything the public ``hooks=``
        door accepts (labels, selectors); bounds are the USER'S semantics,
        never a dtype fact -- dtype headroom is ``dtype_range_audit``'s
        different, already-covered question.
    event_stream:
        Optional stream to publish per-fire verdict events into.
    """

    def __init__(
        self,
        bounds: Mapping[Any, tuple[float | None, float | None]],
        *,
        event_stream: EventStream | None = None,
    ) -> None:
        if not bounds:
            raise CheckConfigError(
                "RangeProbe needs at least one site -> (low, high) entry.",
                code="check_bounds_invalid",
                remedy="Pass bounds={tl.func('relu'): (0.0, 6.0)}-style live-selector entries.",
            )
        for site, pair in bounds.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise CheckConfigError(
                    f"bounds for site {site!r} must be a (low, high) tuple.",
                    code="check_bounds_invalid",
                    remedy="Use (low, high) with None for an open side.",
                )
            low, high = pair
            if low is not None and high is not None and low > high:
                raise CheckConfigError(
                    f"bounds for site {site!r} have low={low!r} > high={high!r}.",
                    code="check_bounds_invalid",
                    remedy="Swap the bounds so low <= high.",
                )
        self._bounds = dict(bounds)
        self._events = event_stream
        self._findings: list[CheckFinding] = []
        self._observed: dict[str, int] = {}

    def hook_plan(self) -> dict[Any, Any]:
        """Return the ``hooks=`` mapping for ``CaptureOptions`` (the seam)."""

        return {
            site: self._make_hook(site, low, high) for site, (low, high) in self._bounds.items()
        }

    def _make_hook(self, site: Any, low: float | None, high: float | None) -> Any:
        """Build one site's range hook (public ``fn(out, *, hook)`` shape)."""

        def _range_hook(out: Any, *, hook: Any) -> Any:
            """Check one fired value against this site's declared bounds."""

            if isinstance(out, torch.Tensor):
                label = _fired_label(hook, str(site))
                with torch.no_grad(), _state.pause_logging():
                    values = out.detach()
                    n_below = int((values < low).sum()) if low is not None else 0
                    n_above = int((values > high).sum()) if high is not None else 0
                self._observe(label, (low, high), (n_below, n_above), out_numel=values.numel())
            return out

        return _range_hook

    def _observe(
        self,
        label: str,
        bounds: tuple[float | None, float | None],
        counts: tuple[int, int],
        *,
        out_numel: int,
    ) -> None:
        """Record one fire's counts; finding only on violation."""

        low, high = bounds
        n_below, n_above = counts
        self._observed[label] = self._observed.get(label, 0) + 1
        violated = (n_below + n_above) > 0
        if self._events is not None:
            self._events.publish(
                ObserverEvent(
                    key=f"checks/range/{label}",
                    kind="verdict",
                    verdict="violated" if violated else "within_bounds",
                    axis_provenance="hook_only",
                )
            )
        if not violated:
            return
        self._findings.append(
            CheckFinding(
                check="activation_range",
                code="activation_range_violated",
                severity="warning",
                action="collect",
                message=(
                    f"{label} violated declared range ({low}, {high}): "
                    f"{n_below} element(s) below, {n_above} above, of "
                    f"{out_numel}."
                ),
                names=(label,),
                evidence="capture_hook",
                values={"n_below": float(n_below), "n_above": float(n_above)},
                remedy="Bounds are the caller's semantic declaration; inspect the named site",
                follow_up="tl.threshold(within=...) on the captured trace",
            )
        )

    @property
    def findings(self) -> tuple[CheckFinding, ...]:
        """Severity-ordered range findings collected so far."""

        return severity_sorted(self._findings)

    @property
    def observed_fires(self) -> dict[str, int]:
        """Per-label fire counts (coverage: zero fires is visible, not silent)."""

        return dict(self._observed)


class LivenessProbe:
    """Liveness STATISTIC recorder -- never a verdict (memo D14).

    Records per-site zero-fraction and running-max across fires. Findings
    never contain the word "dead": ``tl.dead(samples=...)`` keeps the only
    dead-unit verdict with its >=2-sample upper-bound epistemics; this probe
    is the cheap live measurement that tells you WHERE to point it.
    """

    def __init__(self, sites: Any, *, event_stream: EventStream | None = None) -> None:
        sites = list(sites)
        if not sites:
            raise CheckConfigError(
                "LivenessProbe needs at least one site target.",
                code="check_bounds_invalid",
                remedy="Pass sites=[tl.func('relu')]-style live-selector targets.",
            )
        self._sites = sites
        self._events = event_stream
        self._zero_fractions: dict[str, list[float]] = {}
        self._running_max: dict[str, float] = {}

    def hook_plan(self) -> dict[Any, Any]:
        """Return the ``hooks=`` mapping for ``CaptureOptions`` (the seam)."""

        return {site: self._make_hook(site) for site in self._sites}

    def _make_hook(self, site: Any) -> Any:
        """Build one site's liveness hook (public ``fn(out, *, hook)`` shape)."""

        def _liveness_hook(out: Any, *, hook: Any) -> Any:
            """Record one fired value's zero-fraction and running max."""

            if isinstance(out, torch.Tensor) and out.numel():
                label = _fired_label(hook, str(site))
                with torch.no_grad(), _state.pause_logging():
                    values = out.detach()
                    zero_fraction = float((values == 0).sum()) / values.numel()
                    running_max = float(values.abs().max())
                self._zero_fractions.setdefault(label, []).append(zero_fraction)
                self._running_max[label] = max(self._running_max.get(label, 0.0), running_max)
                if self._events is not None:
                    self._events.publish(
                        ObserverEvent(
                            key=f"checks/liveness/{label}",
                            kind="scalar",
                            value=zero_fraction,
                            axis_provenance="hook_only",
                        )
                    )
            return out

        return _liveness_hook

    def facts(self) -> dict[str, dict[str, float]]:
        """Per-site liveness statistics recorded so far.

        Statistics, not verdicts: a high zero fraction on ONE capture is a
        single-sample fact (``tl.sign(site, 'zero')`` is the single-capture
        spelling); the multi-sample verdict belongs to
        ``tl.dead(samples=...)``.
        """

        return {
            label: {
                "fires": float(len(fractions)),
                "mean_zero_fraction": sum(fractions) / len(fractions),
                "last_zero_fraction": fractions[-1],
                "running_abs_max": self._running_max.get(label, 0.0),
            }
            for label, fractions in self._zero_fractions.items()
        }


__all__ = ["LivenessProbe", "RangeProbe"]
