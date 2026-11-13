"""Capture-time honesty disclosures for the MLX backend.

Two conditions leave an MLX capture looking complete when it is not: a model
holding ``mx.compile`` or traced-transform attributes whose interiors are never
logged, and an intervention selector that matched no site. Each one is
disclosed with a ``UserWarning`` (the compiled-attribute case also marks the
trace unverified). A ``stop_after`` selector that never fired refuses typed
(``stop_after_never_fired``), as on torch; an unfired callable is recorded.
The backend module calls these from ``capture_trace``; the warning stack level
points at that method's caller, as it did inline.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..._capture._stop_after_policy import settle_unfired_stop_after

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


def _disclose_compiled_attributes(trace: Trace, compiled_attribute_paths: Sequence[str]) -> None:
    """Warn about, and mark unverified, a capture with compiled attributes.

    Parameters
    ----------
    trace
        Trace being captured.
    compiled_attribute_paths
        Attribute paths holding ``mx.compile`` or traced-transform callables.
    """

    if compiled_attribute_paths:
        # Conservative ceiling, not a refusal: the attribute may never be
        # called, but a call would bypass wrapper capture (cold trace) or
        # replay a cached graph (warm), so honesty cannot depend on it.
        names = ", ".join(compiled_attribute_paths)
        warnings.warn(
            "MLX model holds mx.compile/traced-transform attribute(s) "
            f"({names}); their interiors are not logged, so this capture "
            "is marked capture_verified=False.",
            UserWarning,
            stacklevel=3,
        )
        trace._provenance_verified = False
        trace._provenance_verification_reason = "mlx_compiled_attribute_not_logged"


def _disclose_unfired_selectors(
    trace: Trace,
    intervention_plan: object | None,
    stop_site: object | None,
    resolved_stop: object | None,
    *,
    halted: bool,
) -> None:
    """Warn when an intervention selector matched no site; refuse an unfired stop_after.

    Parameters
    ----------
    trace
        Trace just captured; its ``_mlx_intervene_fired`` flag is consumed.
    intervention_plan
        Resolved intervention plan, or ``None`` when none was requested.
    stop_site
        The requested ``stop_after`` site (a selector or a callable), or
        ``None`` when none was requested.
    resolved_stop
        The selector the site resolved to: the site itself for a selector, a
        ``tl.where`` wrapper for a callable.
    halted
        Whether the forward stopped at the ``stop_after`` selector.
    """

    # Zero-match disclosure (preview half of the torch-side fix): a
    # selector matching nothing produced a trace byte-identical to
    # plain capture with no warning and no counter anywhere.
    if intervention_plan is not None and not trace.__dict__.pop("_mlx_intervene_fired", False):
        warnings.warn(
            "Capture-time intervention selector matched zero sites on the mlx "
            "forward; no intervention fired (check the selector's op/module name).",
            UserWarning,
            stacklevel=3,
        )
    if not halted:
        # Same never-fired policy as torch: an unfired selector refuses typed
        # (``stop_after_never_fired``), never a silent full forward; an unfired
        # callable is exploratory and is recorded in the trace annotations.
        settle_unfired_stop_after(trace, stop_site, selector_shaped=stop_site is resolved_stop)
