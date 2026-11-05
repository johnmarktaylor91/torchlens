"""Layer-sequence and out-payload readers behind the nonfinite scan kinds."""

from __future__ import annotations

from typing import Any

from .._utils._record_group_table import record_member


def _trace_out(layer: Any) -> Any:
    """Read a layer's out payload directly, letting unsaved reads raise."""

    return getattr(layer, "out", None)


def _saved_out(layer: Any) -> Any:
    """Return a layer's saved output payload, or ``None`` when unavailable.

    The report anomaly scans only reason about *saved* activation payloads. On a
    selective-save trace most layers retain no payload, and reading ``.out`` on
    such an op raises ``ValueError`` (``"... was not saved; no saved payload is
    available"``) -- a per-pass ``ValueError`` that a plain ``getattr(layer,
    "out", None)`` cannot swallow, so both the JSON and prose reports previously
    crashed on the ordinary predicate-save trace shape instead of honoring their
    documented ``unknown``/scoped-clean contract. Gate on the saved-payload flag
    first, then read the property inside the known-unavailable boundary so an
    unsaved op is honestly skipped rather than aborting the whole report.

    Parameters
    ----------
    layer:
        A per-pass operation/layer entry from ``log._layer_list``.

    Returns
    -------
    Any
        The saved output tensor when a payload was retained, else ``None``.
    """

    if not bool(getattr(layer, "has_saved_activation", False)):
        return None
    if record_member(layer, "out_ref", None) is not None:
        slot = getattr(layer, "_slot", None)
        if callable(slot):
            try:
                resident = slot("out")
            except (AttributeError, KeyError, TypeError):
                resident = None
            if resident is None:
                # Repr/report surfaces are metadata queries. A disk-backed value
                # remains explicitly unexamined until the user requests it.
                return None
            return resident
    try:
        return getattr(layer, "out", None)
    except ValueError:
        return None


def _layer_list(log: Any) -> Any:
    """Return the finalized layer sequence of a log."""

    return log._layer_list or []


def _raw_layers(log: Any) -> Any:
    """Return the raw pre-postprocessing layer sequence of a partial capture."""

    return getattr(log, "raw_layers", ()) or ()
