"""Input-tree bounds, refusals and the inert instance-state inspector.

The basis half of the model-input boundary walk: the shared nesting ceiling, the typed
refusals every walker raises, and the inert instance-state enumeration. ``_input_walk``
(the engine-side walker) re-exports these names; ``_utils`` helpers import them from here
so the basis never imports upward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._errors import InvalidArgumentError

__all__ = [
    "INPUT_TREE_MAX_DEPTH",
    "MAX_INPUT_CONTAINER_DEPTH",
    "InstanceStateInspection",
    "inspect_instance_state",
    "raise_input_tree_cycle_refusal",
    "raise_input_tree_depth_refusal",
    "raise_input_tree_namedtuple_refusal",
    "raise_input_tree_stack_refusal",
]


MAX_INPUT_CONTAINER_DEPTH: int = 200

INPUT_TREE_MAX_DEPTH: int = MAX_INPUT_CONTAINER_DEPTH
"""The shared model-input boundary nesting ceiling (r-b4 R27-1).

Mirrors the artifact-side literal-decode ceiling
(``_runnable_execution._MAX_DECODE_NESTING_DEPTH`` / the ``_io`` parse quartet):
200 sits far above any real input nesting and well below the interpreter's
default recursion crash depth (the live walkers burn ~2-3 frames per level, so
an unbounded walk died with a raw ``RecursionError`` at ~350 user levels).
Every capture-entry input walker (``walk_input_boundary``,
``snapshot_input_boundary``, ``backends.default_specs._simple_leaves``,
``utils.arg_handling.copy_arg_tree``) enforces this ONE ceiling with a typed
refusal instead of an untyped stdlib crash.
"""


def raise_input_tree_depth_refusal(*, depth: int) -> None:
    """Raise the typed over-depth input-boundary refusal (r-b4 R27-1).

    Parameters
    ----------
    depth:
        Nesting depth at which the ceiling was crossed.
    """

    raise InvalidArgumentError(
        "Model-input tree nesting exceeds the supported input-boundary depth "
        f"ceiling ({INPUT_TREE_MAX_DEPTH}).",
        code="input_tree_depth_exceeded",
        remedy=(
            "Flatten the nested input containers (or unwrap the deep wrapper "
            "object) before tracing; the ceiling matches the portable-artifact "
            "nesting bound."
        ),
        depth=depth,
    )


def raise_input_tree_stack_refusal(cause: BaseException | None = None) -> None:
    """Raise the typed stack-budget input-boundary refusal (grind-p3 T11.4).

    The nesting ceiling is a DEPTH bound, but the actual failure class is
    STACK BUDGET: the live walkers burn ~2-3 interpreter frames per level, so
    a LEGAL input (nesting <= the ceiling) still died in a raw
    ``RecursionError`` whenever the caller entered capture with most of the
    stack already consumed (deep user recursion, constrained
    ``sys.setrecursionlimit``). Every walker entry converts that exhaustion
    into this typed refusal instead of an untyped stdlib crash.

    Parameters
    ----------
    cause:
        The caught ``RecursionError``, chained as ``__cause__``.
    """

    error = InvalidArgumentError(
        "Walking the model-input tree exhausted the Python stack budget before "
        f"reaching the depth ceiling ({INPUT_TREE_MAX_DEPTH}): capture was "
        "entered with most of the interpreter stack already consumed.",
        code="input_tree_stack_exhausted",
        remedy=(
            "Call the capture entry point from a shallower call stack, or raise "
            "sys.setrecursionlimit() to leave headroom for the bounded input walk."
        ),
    )
    raise error from cause


def raise_input_tree_cycle_refusal(*, kind: str) -> None:
    """Raise the typed cyclic-container input-boundary refusal (r-b4 R27-1).

    Parameters
    ----------
    kind:
        Container kind (closed vocabulary) at which the cycle closed.
    """

    raise InvalidArgumentError(
        f"Model-input tree contains a self-referential {kind} container "
        "(a container reachable from itself).",
        code="input_tree_cycle",
        remedy="Remove the container reference cycle from the model input.",
        kind=kind,
    )


def raise_input_tree_namedtuple_refusal(*, declared: int, physical: int) -> None:
    """Raise the typed non-total-namedtuple-schema input refusal (B3R4-R12-1).

    A tuple subclass DECLARING a ``_fields`` schema that does not account for
    the physical tuple (a malformed non-tuple-of-str ``_fields``, or a declared
    arity differing from the physical arity) used to descend only the declared
    names -- NOTHING for a malformed schema -- so every remaining tensor leaf
    silently vanished from the capture-side walkers: the trace had no input
    node and the gap was misattributed to a stale-reference escape.

    Parameters
    ----------
    declared:
        Number of usable declared field names (0 for a malformed schema).
    physical:
        Concrete builtin tuple arity.
    """

    raise InvalidArgumentError(
        "Model-input tree contains a tuple subclass declaring a namedtuple "
        "`_fields` schema that does not account for the physical tuple "
        f"({declared} usable declared field(s) vs {physical} physical "
        "element(s); a non-tuple-of-str `_fields` declaration counts as 0).",
        code="input_namedtuple_schema_not_total",
        remedy=(
            "Fix the container's `_fields` declaration (a tuple of one str per "
            "positional element) or pass a plain tuple/list instead."
        ),
        declared_fields=declared,
        physical_arity=physical,
    )


@dataclass(frozen=True, slots=True)
class InstanceStateInspection:
    """One inert instance-state inspection result (r71 C).

    ``complete`` is ``True`` only when the enumeration is PROVABLY total inertly;
    ``reason`` names the closed fail-closed cause otherwise. ``uncertainty is NEVER
    an empty set``: a blinded inspection reports ``complete=False`` with a non-empty
    reason, never ``names=frozenset()`` masquerading as "no extra state".
    """

    names: frozenset[str]
    complete: bool
    reason: str | None


def _raw_mro_attr(value: Any, attr: str) -> Any:
    """Resolve one attribute through raw class-dict MRO lookup (never ``getattr``).

    Returns the WINNING descriptor/object from ``type(value).__mro__`` without
    invoking any descriptor protocol, ``__getattribute__``, or ``__getattr__`` -- so
    a property/custom-descriptor shadow is OBSERVED, never executed.
    """

    for cls in type(value).__mro__:
        if attr in cls.__dict__:
            return cls.__dict__[attr]
    return None


def inspect_instance_state(value: Any) -> InstanceStateInspection:
    """Inertly enumerate the SET instance-state NAMES on ``value``, fail-closed (r71 C).

    THE single instance-state inspector for declared-schema containers. It NEVER routes the instance ``__dict__`` through ``getattr`` (r70 hon1-F1: a property/descriptor-shadowed
    ``__dict__`` executes user code returning attacker-chosen ``{}``, turning
    "untrustworthy" into "proved empty"). Instead:

    * The instance ``__dict__`` descriptor is resolved by RAW MRO lookup. Absence is
      valid (a genuinely slotted type). If present it must be EXACTLY
      ``types.GetSetDescriptorType`` (the standard instance-dict getset); it is
      invoked DIRECTLY and the result must be ``type(result) is dict``. A property,
      custom/replaced descriptor, dict-subclass/non-dict result, or an exception is
      INCOMPLETE (``instance_state_uninspectable``).
    * The all-MRO genuine ``MemberDescriptorType`` slot scan is unchanged (unset
      slots absent, mangled private names, falsey values counted -- inert).

    PRESENCE is state (a ``None``-valued extra counts). Returns ``complete=True`` with
    the actual storage names only when the enumeration is inertly total.
    """

    items = _inspect_instance_state_items(value)
    if items is None:
        return InstanceStateInspection(frozenset(), False, "instance_state_uninspectable")
    return InstanceStateInspection(frozenset(items), True, None)


def _inspect_instance_state_items(value: Any) -> dict[str, Any] | None:
    """Inertly enumerate SET instance-state ``name -> value`` pairs, or ``None``.

    The value-bearing spine under :func:`inspect_instance_state`: same inert
    channels (raw-MRO ``__dict__`` getset invoked directly, all-MRO genuine
    member-descriptor slots), same fail-closed rules -- ``None`` means the
    enumeration cannot be inertly proven total (a blinded inspection is never
    an empty mapping masquerading as "no extra state").
    """

    import types as _types

    items: dict[str, Any] = {}
    dict_descriptor = _raw_mro_attr(value, "__dict__")
    if dict_descriptor is not None:
        if type(dict_descriptor) is not _types.GetSetDescriptorType:
            # A property or non-standard descriptor shadows __dict__: reading it would
            # execute user code and could report attacker-chosen state. Fail closed.
            return None
        try:
            instance_dict = dict_descriptor.__get__(value, type(value))
        except Exception:
            return None
        if type(instance_dict) is not dict:
            # A dict SUBCLASS (or non-dict) result could carry a custom __iter__ /
            # __contains__ that hides keys: not inertly trustworthy.
            return None
        for key in instance_dict:
            items[str(key)] = instance_dict[key]
    for cls in type(value).__mro__:
        raw_slots = cls.__dict__.get("__slots__")
        if raw_slots is None:
            continue
        slot_names = [raw_slots] if isinstance(raw_slots, str) else [str(s) for s in raw_slots]
        for raw_name in slot_names:
            if raw_name in {"__dict__", "__weakref__"}:
                continue
            name = raw_name
            if raw_name.startswith("__") and not raw_name.endswith("__"):
                # CPython mangles private slot names against the DECLARING class.
                stripped = cls.__name__.lstrip("_")
                if stripped:
                    name = f"_{stripped}{raw_name}"
            descriptor = cls.__dict__.get(name)
            if not isinstance(descriptor, _types.MemberDescriptorType):
                # A shadowed/replaced slot descriptor makes the slot storage
                # unreachable by ordinary attribute access; reading anything else
                # here could execute user code, so the name is skipped (inert).
                continue
            try:
                slot_value = descriptor.__get__(value, cls)
            except AttributeError:
                continue  # unset slot: absent
            items[name] = slot_value
    return items
