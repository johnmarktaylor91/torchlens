"""Narrow class-method shadow filter for portable ``__setstate__`` loads.

A portable data class rehydrates from a pickled ``state`` dict via
``self.__dict__.update(state)``. Round 54 ``sec_3`` showed that an attacker who
plants a ``state`` key shadowing a class-owned method (``_internal_set``) turns
the instance attribute into an execution boundary: ``rehydrate`` later
read-then-calls it. The primary close resolves that setter off the CLASS
(``rehydrate._assign_rehydrated_field``); this module is the defense-in-depth
enabler filter -- it refuses, at ``__setstate__`` time, any incoming state key
that shadows a class-owned *plain method / non-descriptor callable*.

The filter is deliberately NARROW. The portable dataclasses expose their real
serialized fields as ``@property`` / slot descriptors, so a *broad* "shadows any
class attribute" filter would reject legitimate state by construction (round-51
over-catch: 336 real ``@property`` collisions). Refusing only callable *methods*
had zero legitimate collisions across all portable classes (probe C2) while still
catching ``_internal_set``/``save``/``run``/``draw`` impersonation.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Mapping
from typing import Any

from ..errors import TorchLensError

# The class-owned callable *method* surface: a state key resolving to one of these
# on the class MRO can never be a legitimate serialized field value, so it is an
# attacker shadow. Descriptors (``property``/``member_descriptor``/getset) are
# EXCLUDED -- real fields are exposed through them.
_METHOD_SHADOW_TYPES: tuple[type, ...] = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    staticmethod,
    classmethod,
)


class PortableStateKeyError(TorchLensError, ValueError):
    """A portable ``__setstate__`` received a key shadowing a class-owned method.

    Raised as a load-integrity tripwire (round 54 ``sec_3``): the state dict of an
    attacker ``.tlspec`` carried a key whose name resolves to a plain method on the
    portable class, which rehydrate would otherwise be able to read-then-call.
    """

    def __init__(self, cls: type, shadowed: list[str]) -> None:
        super().__init__(
            f"Portable {cls.__name__} state carries key(s) {sorted(shadowed)!r} that shadow "
            "class-owned method(s); refusing the load (attacker-planted callable shadow)."
        )
        self.cls = cls
        self.shadowed = tuple(sorted(shadowed))


def _is_descriptor(attr: Any) -> bool:
    """Return whether ``attr`` implements the descriptor protocol on its type."""

    attr_type = type(attr)
    return (
        hasattr(attr_type, "__get__")
        or hasattr(attr_type, "__set__")
        or hasattr(attr_type, "__delete__")
    )


def _key_shadows_class_method(cls: type, key: str) -> bool:
    """Return whether ``key`` resolves to a class-owned plain method on ``cls``.

    Uses ``inspect.getattr_static`` so the class MRO is walked WITHOUT triggering
    any descriptor ``__get__`` (execution-free). A key absent from the class, or
    resolving to a ``property`` / data descriptor / plain value, is a legitimate
    field and returns ``False``.
    """

    try:
        attr = inspect.getattr_static(cls, key)
    except AttributeError:
        return False
    if isinstance(attr, _METHOD_SHADOW_TYPES):
        return True
    # A plain callable class attribute that is NOT itself a type and NOT a
    # descriptor (e.g. a bare callable object) is also a method-like shadow; a
    # ``property`` is a descriptor and is therefore NOT caught here.
    return callable(attr) and not isinstance(attr, type) and not _is_descriptor(attr)


def refuse_callable_shadowing_state_keys(cls: type, state: Mapping[str, Any]) -> None:
    """Refuse a portable ``state`` dict that shadows any class-owned method of ``cls``.

    Call immediately before ``self.__dict__.update(state)`` in every portable
    ``__setstate__``. Narrow by design (probe C2: 0 legitimate collisions), it
    closes the read-then-call enabler class for every non-slotted portable class at
    once without touching the ``@property``-backed real state keys.
    """

    shadowed = [
        key for key in state if isinstance(key, str) and _key_shadows_class_method(cls, key)
    ]
    if shadowed:
        raise PortableStateKeyError(cls, shadowed)
