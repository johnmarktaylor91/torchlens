"""Frozen predicate runtime extension point (the S4 seam contract).

This module is the ONE documented extension surface through which downstream
predicate consumers (aten-layer predicates, ``until=`` / run-time ``save=``)
accept user predicates for the capture-lifecycle slots. It ships in two
merges: m1 (``PredicateProtocol`` + ``coerce_predicate`` over raw callables)
and m2 (``register_predicate`` + name acceptance inside ``coerce_predicate``).

Contract of record: ``docs/reference/predicate_runtime.md``. Every public
spelling here is DOCUMENTED-UNSTABLE pending its naming-session ratification
(provisional-name protocol): it may rename without a deprecation
shim.

Scope (frozen):

* ``PredicateProtocol`` covers the CAPTURE-LIFECYCLE ``save``/``halt``/
  ``until`` slots ONLY, over the concrete :class:`~torchlens.ir.predicate.
  RecordContext`. The ``intervene=`` slot (live-proxy subject, wider decision
  domain) and the grad slot (``GradRecordContext`` subject) are OUTSIDE this
  protocol and outside the registry.
* ``coerce_predicate`` is the single coercion door. Raw callables — including
  ``BaseSelector`` instances, ``followed_by`` composites, and
  ``.selector``-bearing closures — are returned BY IDENTITY, never wrapped:
  every shipped capture-layer introspection point sees exactly the object the
  user supplied. Registered NAMES return a slot-aware enforcing wrapper.
* Registration never mutates the user's object and never stamps any attribute
  consulted by the restricted loader (``utils._callable_safety``): a pickled
  or persisted reference to a registered predicate still refuses typed.
* The registry is INERT at m2 merge time: no shipped surface accepts a
  registered NAME until the consuming lanes adopt name acceptance in their
  own merges.
"""

from __future__ import annotations

import itertools
import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol

from .predicate import RecordContext, RetroactiveCaptureDecision

if TYPE_CHECKING:
    from ..fastlog.options import PredicateDecision

__all__ = ["PredicateProtocol", "coerce_predicate", "register_predicate"]

#: Closed slot vocabulary (S2-owned): the capture-lifecycle predicate slots.
#: ``save`` normalizes through the shipped decision table; ``halt`` is
#: bool-only EXCLUSIVE stop (the satisfying record is NOT captured); ``until``
#: is bool-only INCLUSIVE stop (the satisfying record IS captured, then the
#: run stops). Consumers own their dispatch; this module owns the vocabulary.
PredicateSlot = Literal["save", "halt", "until"]
_PREDICATE_SLOTS: tuple[str, ...] = ("save", "halt", "until")

#: Bool-only slots: the enforcing wrapper narrows registered-name returns.
_BOOL_ONLY_SLOTS: frozenset[str] = frozenset({"halt", "until"})


class PredicateProtocol(Protocol):
    """Frozen callable signature for capture-lifecycle predicate slots.

    Exactly the shipped contract: ONE positional :class:`RecordContext` — the
    concrete class, not a generic subject — and no kwargs. Covers the
    ``save``/``halt``/``until`` slots only. Caller-facing clauses (idempotence
    under alias retry, short-circuit composition, in-flight ``ctx.label``
    spelling, value-dependent fields may raise) are normative in
    ``docs/reference/predicate_runtime.md``.
    """

    def __call__(self, ctx: RecordContext) -> PredicateDecision:
        """Evaluate the predicate for one chronological capture event."""
        ...


# ---------------------------------------------------------------------------
# Registry state (m2). Builtin names are NEVER replaceable; the split exists
# for future built-ins and is empty at S4. Registration is setup-time; the
# lock guards the registry dict only.
# ---------------------------------------------------------------------------

_BUILTIN_PREDICATES: dict[str, Callable[[RecordContext], Any]] = {}
_USER_PREDICATES: dict[str, tuple[Callable[[RecordContext], Any], int]] = {}
_REGISTRY_LOCK = threading.Lock()
_VERSION_COUNTER = itertools.count(1)


def _carries_capture_introspection(value: object) -> bool:
    """Return whether the capture layer structurally introspects ``value``.

    ``BaseSelector`` trees (including ``followed_by`` composites) and any
    object carrying a ``.selector`` attribute get capture-time meaning from
    introspection the registry deliberately does not forward, so registering
    them would silently change their behavior.
    """

    if getattr(value, "selector", None) is not None:
        return True
    from ..intervention.selectors import BaseSelector

    return isinstance(value, BaseSelector)


def register_predicate(
    name: str, *, replace: bool = False
) -> Callable[[Callable[[RecordContext], Any]], Callable[[RecordContext], Any]]:
    """Register a plain predicate callable under ``name`` (decorator).

    The decorated callable is returned truly unchanged: no attribute is
    written onto it, and in particular no attribute consulted by the
    restricted loader is set. The registry stores an internal
    ``name -> (callable, version)`` table; the same callable registered under
    two names yields two registrations with distinct cache keys.

    Parameters
    ----------
    name:
        Registry name later accepted by :func:`coerce_predicate`.
    replace:
        Allow replacing an existing USER registration (bumps the version
        counter, invalidating cache keys minted for the old registration).
        Builtin names are never replaceable.

    Raises
    ------
    ValueError
        For a malformed name, a non-callable, or an introspection-bearing
        object (``BaseSelector`` trees, ``followed_by`` composites,
        ``.selector``-bearing objects — register the underlying plain
        predicate instead).
    PredicateError
        ``code="predicate_name_conflict"`` for a duplicate user name without
        ``replace=True``, or any attempt to shadow/replace a builtin name.
    """

    if not isinstance(name, str) or not name:
        raise ValueError(
            "register_predicate name must be a non-empty str. "
            f"Got {name!r}. Remedy: pass the registry name the predicate should be "
            "coerced under."
        )

    def decorator(fn: Callable[[RecordContext], Any]) -> Callable[[RecordContext], Any]:
        """Store ``fn`` in the registry table and return it unchanged."""

        if not callable(fn):
            raise ValueError(
                "register_predicate requires a callable satisfying PredicateProtocol "
                f"(one positional RecordContext). Got {type(fn).__name__}."
            )
        if _carries_capture_introspection(fn):
            raise ValueError(
                "register_predicate refuses selector objects the capture layer "
                "introspects structurally (BaseSelector trees, followed_by "
                "composites, .selector-bearing callables): registering them would "
                "silently drop that introspection surface. Remedy: register the "
                "underlying plain predicate callable instead, or pass the selector "
                "directly (raw callables never need registration)."
            )
        from ..fastlog.exceptions import PredicateError

        with _REGISTRY_LOCK:
            if name in _BUILTIN_PREDICATES:
                raise PredicateError(
                    f"predicate name {name!r} is a TorchLens builtin and is never "
                    "replaceable. Remedy: choose a different registry name.",
                    code="predicate_name_conflict",
                )
            if name in _USER_PREDICATES and not replace:
                raise PredicateError(
                    f"predicate name {name!r} is already registered. Remedy: pass "
                    "replace=True to replace the existing registration, or choose a "
                    "different name.",
                    code="predicate_name_conflict",
                )
            _USER_PREDICATES[name] = (fn, next(_VERSION_COUNTER))
        return fn

    return decorator


def _lookup_registered(name: str) -> tuple[Callable[[RecordContext], Any], int] | None:
    """Return the ``(callable, version)`` registration for ``name``, if any.

    Internal: the ONLY registry exit is :func:`coerce_predicate`, which wraps.
    """

    with _REGISTRY_LOCK:
        builtin = _BUILTIN_PREDICATES.get(name)
        if builtin is not None:
            return (builtin, 0)
        return _USER_PREDICATES.get(name)


def _make_registered_wrapper(
    name: str,
    fn: Callable[[RecordContext], Any],
    version: int,
    slot: str,
) -> Callable[[RecordContext], Any]:
    """Return the slot-aware enforcing wrapper for one registered name.

    Binding is COERCE-TIME: the wrapper closes over the resolved
    ``(callable, version)`` pair, so a later ``replace=True`` re-registration
    invalidates caches through the NEXT coercion's bumped key while an
    already-bound wrapper keeps its consistent pair for the run.

    The wrapper deliberately presents as a PLAIN callable (no ``.selector``
    forwarding, not a ``BaseSelector``): registration refuses
    introspection-bearing objects, so a wrapped registered predicate has no
    retroactive path to lose, and plain-callable conservative treatment
    (alias retry with documented double invocation) is intended.
    """

    if slot in _BOOL_ONLY_SLOTS:

        def wrapper(ctx: RecordContext) -> Any:
            """Enforce the bool-only return domain of this slot on ``fn``."""

            result = fn(ctx)
            if not isinstance(result, bool):
                from ..fastlog.exceptions import PredicateError

                raise PredicateError(
                    f"registered predicate {name!r} in the {slot!r} slot must return "
                    f"bool; got {type(result).__name__}. The {slot!r} slot's accepted "
                    "return domain is bool ONLY.",
                    result=result,
                    code="predicate_return_invalid",
                    reason=f"{slot}_slot_bool_only",
                )
            return result

    else:

        def wrapper(ctx: RecordContext) -> Any:
            """Refuse RAW-CALLABLE-ONLY retroactive returns from a registered name."""

            result = fn(ctx)
            if isinstance(result, RetroactiveCaptureDecision):
                from ..fastlog.exceptions import PredicateError

                raise PredicateError(
                    f"registered predicate {name!r} returned a "
                    "RetroactiveCaptureDecision, which is RAW-CALLABLE-ONLY behavior: "
                    "the followed_by machinery that gives retroactive decisions "
                    "meaning is builtin-gated, so a registered predicate's return "
                    "domain is narrowed to bool | CaptureSpec | None. Remedy: pass "
                    "the predicate as a raw callable instead of a registered name.",
                    result=result,
                    code="predicate_return_invalid",
                    reason="registered_retroactive_unsupported",
                )
            return result

    wrapper.__name__ = f"registered_predicate_{name}"
    wrapper.__qualname__ = wrapper.__name__
    # Read unmodified by the shipped cache helper
    # (_trace_selector_helpers._predicate_cache_key).
    wrapper.__torchlens_cache_key__ = ("registered", name, version)  # type: ignore[attr-defined]
    return wrapper


def coerce_predicate(value: Any, *, slot: PredicateSlot) -> Callable[[RecordContext], Any]:
    """Coerce a raw callable or registered name for one capture-lifecycle slot.

    THE single documented coercion point predicate consumers call. The value
    domain is CLOSED: exactly ``{raw callable, registered-name str}``.

    * RAW CALLABLES ARE RETURNED UNWRAPPED (identity): the object leaving this
      function ``is`` the object that entered, so every shipped capture-layer
      introspection point (``followed_by`` support, ``BaseSelector`` alias
      retry, cache keys) sees exactly what it sees today. ``BaseSelector``
      instances ARE raw callables and are accepted on this branch — a selector
      that reaches this door gets shipped capture-predicate ``keep_op``
      semantics for that same object, never silent misbehavior. Consumers
      dispatch selector/label forms to their own selector paths BEFORE calling
      this function; a str here is ALWAYS read as a registry name.
    * REGISTERED NAMES return the slot-aware enforcing wrapper (narrowed
      return domain; ``__torchlens_cache_key__ = ("registered", name,
      version)``).

    Parameters
    ----------
    value:
        Raw predicate callable or registered name.
    slot:
        Consuming slot, one of the closed vocabulary ``save | halt | until``.

    Raises
    ------
    ValueError
        Unknown slot, or a non-callable non-str value (house-style setup-time
        refusals).
    PredicateError
        ``code="predicate_unregistered"`` for a name miss.
    """

    if slot not in _PREDICATE_SLOTS:
        raise ValueError(
            f"unknown predicate slot {slot!r}: the closed slot vocabulary is "
            "'save' | 'halt' | 'until'. The intervene and grad slots are outside "
            "the S4 predicate contract (see docs/reference/predicate_runtime.md)."
        )
    if callable(value):
        return value
    if isinstance(value, str):
        entry = _lookup_registered(value)
        if entry is None:
            from ..fastlog.exceptions import PredicateError

            raise PredicateError(
                f"no predicate registered under name {value!r}. Remedy: register it "
                "with @register_predicate(name) first, or pass the predicate "
                "callable directly.",
                code="predicate_unregistered",
            )
        fn, version = entry
        return _make_registered_wrapper(value, fn, version, slot)
    raise ValueError(
        "coerce_predicate accepts exactly a raw predicate callable or a "
        f"registered-name str; got {type(value).__name__}. Remedy: pass the "
        "callable itself, or the name it was registered under."
    )


def _reset_for_tests() -> None:
    """Clear USER registrations. TEST-ONLY: refuses outside pytest.

    Mirrors the ``activate_prerelease_fields`` precedent: test isolation only,
    never a production spelling. Builtin registrations are untouched.
    """

    if "PYTEST_CURRENT_TEST" not in os.environ:
        raise RuntimeError(
            "_reset_for_tests() is a TEST-ONLY registry reset and refuses to run outside pytest."
        )
    with _REGISTRY_LOCK:
        _USER_PREDICATES.clear()
