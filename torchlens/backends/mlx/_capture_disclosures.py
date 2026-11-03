"""Capture-time honesty disclosures for the MLX preview backend.

Two conditions leave an MLX capture looking complete when it is not: a model
holding ``mx.compile`` or traced-transform attributes whose interiors are never
logged, and an intervention or halt selector that matched no site. Each one is
disclosed with a ``UserWarning`` (the compiled-attribute case also marks the
trace unverified). The backend module calls these from ``capture_trace``; the
warning stack level points at that method's caller, as it did inline.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

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
    halt_selector: object | None,
    *,
    halted: bool,
) -> None:
    """Warn when a capture-time intervention or halt selector matched no site.

    Parameters
    ----------
    trace
        Trace just captured; its ``_mlx_intervene_fired`` flag is consumed.
    intervention_plan
        Resolved intervention plan, or ``None`` when none was requested.
    halt_selector
        Resolved halt selector, or ``None`` when none was requested.
    halted
        Whether the forward stopped at the halt selector.
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
    if halt_selector is not None and not halted:
        warnings.warn(
            "Capture-time halt selector matched zero sites on the mlx forward; "
            "the capture ran the full forward and completed without halting.",
            UserWarning,
            stacklevel=3,
        )
