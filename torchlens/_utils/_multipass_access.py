"""Multi-pass-safe attribute access for aggregate (recurrent) ``Layer`` objects.

Shared helper for the whole hardening sprint. A recurrent model rolls its ``N``
executed passes into one aggregate :class:`~torchlens._data_classes.layer.Layer`.
Reading a *per-pass* field (``func_duration``, ``out``, ``grad``, ``label``,
``is_module_output``, ``interventions``, ...) on that aggregate raises a
DELIBERATE ``ValueError`` tripwire (``Layer._single_pass_or_error``) that directs
the caller to a specific ``layer.ops[i]`` pass. The tripwire uses ``ValueError``
-- NOT ``AttributeError`` -- precisely so a naive ``getattr(layer, attr, default)``
cannot silently swallow it and fabricate the default: a wrong ``func_duration=0.0``
/ ``is_module_output=False`` would be exactly the silent-wrongness the tripwire
exists to prevent.

The bug class this module fixes: visualization / debug / report call sites that
did ``getattr(layer_or_op, attr, default)`` -- which swallows only
``AttributeError`` -- so on a recurrent model the deliberate ``ValueError`` escaped
and crashed a public entrypoint (``draw_combined``, ``summary(mode=...)``,
``Layer.show()``, ``preview_fastlog``, node overlays, ...).

THE ONE DOCUMENTED PATTERN (round-12 flagged divergent per-site ``try``/``except``
as a defect). Every site that reads a possibly-per-pass attribute off a
Layer-or-Op routes through :func:`get_multipass_attr`, then decides EXPLICITLY at
the call site: surface the correct aggregate (``total_*``) or per-pass value,
degrade to an honest ``None`` / "n/a", or raise the site's OWN typed error. Never
a silent wrong default; never a bare ``ValueError`` leak.

Reused verbatim by the A4 debug/report fixers -- import from here, do not
re-derive. ``torchlens._utils`` is a neutral low-level package (stdlib/torch only),
so ``visualization``, ``debug``, and ``report`` all import it without a backwards
layering dependency.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MISSING",
    "RAISE",
    "MultiPassAmbiguityError",
    "get_multipass_attr",
    "is_multipass_layer",
]

# "no default supplied" sentinel -- distinct from a caller-supplied ``None`` default.
MISSING: Any = object()
# ``multipass=`` sentinel -- re-raise the ambiguity as ``MultiPassAmbiguityError``.
RAISE: Any = object()


class MultiPassAmbiguityError(ValueError):
    """A per-pass ``Layer`` attribute was read on an aggregate multi-pass (recurrent) Layer.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep
    catching it, while letting a call site catch *exactly* the multi-pass
    ambiguity and convert it into a site-specific, documented error (for example
    "layer is recurrent; select a pass: ``log['x:1']``"). Never let a bare
    ``ValueError`` from the tripwire leak out of a public entrypoint, and never
    swallow it into a wrong default.
    """


def is_multipass_layer(obj: Any) -> bool:
    """Return ``True`` iff ``obj`` is an aggregate recurrent ``Layer`` (``num_passes > 1``).

    This is the exact object whose per-pass attribute access trips the
    ``Layer._single_pass_or_error`` ``ValueError`` guard. A single per-pass ``Op``
    is NOT multi-pass here even though it proxies ``num_passes`` from its parent
    Layer -- an ``Op`` reads its own pass value without tripping the guard -- so we
    key on the concrete ``Layer`` type name (dependency-free; avoids an
    ``utils -> data_classes`` import cycle), NOT on ``num_passes`` alone.
    """

    if type(obj).__name__ != "Layer":
        return False
    try:
        return int(getattr(obj, "num_passes", 1) or 1) > 1
    except (TypeError, ValueError):
        return False


def get_multipass_attr(
    obj: Any,
    attr: str,
    default: Any = MISSING,
    *,
    multipass: Any = RAISE,
) -> Any:
    """Read ``attr`` from ``obj`` without ever leaking the multi-pass ``ValueError`` tripwire.

    Behaviour
    ---------
    * ``Op`` / single-pass ``Layer`` / any non-Layer: behaves like ``getattr``.
      Returns the value; a genuinely-missing attribute returns ``default`` when one
      is supplied, else re-raises ``AttributeError`` (like builtin ``getattr``).
    * Aggregate multi-pass (recurrent) ``Layer`` reading a per-pass attribute: the
      tripwire ``ValueError`` fires. This helper NEVER routes that into ``default``
      (that would silently fabricate a wrong per-pass value). Instead:

      - ``multipass=RAISE`` (default) -> raise :class:`MultiPassAmbiguityError` so
        the caller converts it into a typed, documented error.
      - ``multipass=<value>`` -> return ``<value>`` as an EXPLICIT, honest
        "not-a-single-value" marker (e.g. ``None`` for an overlay that renders
        "n/a"). The caller chose this; it is not a silent default.
    * A ``ValueError`` raised for any OTHER reason (not the multi-pass tripwire)
      always propagates unchanged -- genuine errors are never swallowed.

    Parameters
    ----------
    obj:
        A ``Layer``, ``Op``, or arbitrary object.
    attr:
        Attribute name to read.
    default:
        Returned for a genuinely-absent attribute (``AttributeError``). If omitted,
        an absent attribute re-raises ``AttributeError``. NEVER used for the
        multi-pass ambiguity case.
    multipass:
        Policy for an aggregate multi-pass Layer: ``RAISE`` (default) or an explicit
        honest sentinel value to return.
    """

    try:
        return getattr(obj, attr)
    except AttributeError:
        if default is MISSING:
            raise
        return default
    except ValueError as exc:
        if is_multipass_layer(obj):
            if multipass is RAISE:
                raise MultiPassAmbiguityError(str(exc)) from exc
            return multipass
        raise
