"""Output-container contract for ``torch.distributions`` objects.

A ``Distribution`` returned by a model or module is walked as a structured
container. Each tensor-valued instance attribute is one output leaf at an
attribute path (``.loc``, ``.scale``), the family's own parameters first in
``arg_constraints`` order; each nested ``Distribution`` attribute
(``Independent.base_dist``, ``Beta._dirichlet``) is a child container. Every
other instance attribute must be a portable literal (shapes, flags, counts)
and is recorded in the spec; an identity alias of a recorded tensor
(``Categorical._param`` is ``logits``) is recorded as an alias.

Reconstruction allocates the class through its inert ``__new__`` and restores
the recorded instance dictionary. No ``__init__`` runs, so there is no
argument validation and no recomputation: the rebuilt object holds exactly the
recorded tensors. An instance attribute that cannot be recorded (a transform
list, a parameter, a callable) makes the output lossy, and the reason names
it.

The spellings a user can see (the spec kind and the lossy reason prefix) live
here and only here.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import Any, Final

import torch

DISTRIBUTION_CONTAINER_KIND: Final = "distribution"
"""``ContainerSpec.kind`` for a ``torch.distributions`` output."""

DISTRIBUTION_STATE_UNRECORDABLE: Final = "distribution_state_unrecordable"
"""Lossy reason prefix: ``distribution_state_unrecordable:<Type>.<attribute>``."""

_LITERAL_TYPES: Final = (bool, int, float, str, type(None), torch.Size, torch.dtype, torch.device)

_STATE_KEY: Final = "state"
_ALIASES_KEY: Final = "aliases"


@dataclass(frozen=True)
class DistributionLayout:
    """How one live distribution instance maps onto the container contract.

    Attributes
    ----------
    fields:
        Attribute names recorded as children, in walk order (tensor leaves and
        nested distributions).
    state:
        ``(name, literal)`` pairs for the non-tensor instance attributes.
    aliases:
        ``(alias, target)`` pairs: ``alias`` holds the same tensor object as
        the recorded field ``target``.
    lossy_reason:
        Empty when the rebuild is exact, otherwise the first unrecordable
        attribute as ``distribution_state_unrecordable:<Type>.<name>``.
    """

    fields: tuple[str, ...]
    state: tuple[tuple[str, Any], ...]
    aliases: tuple[tuple[str, str], ...]
    lossy_reason: str

    def aux_data(self) -> tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]:
        """Return the portable ``ContainerSpec.aux_data`` for this layout."""

        return ((_STATE_KEY, self.state), (_ALIASES_KEY, self.aliases))


def _distribution_base() -> type[Any] | None:
    """Return ``torch.distributions.Distribution`` when torch has loaded it."""

    module = sys.modules.get("torch.distributions.distribution")
    base = getattr(module, "Distribution", None) if module is not None else None
    return base if isinstance(base, type) else None


def is_distribution(value: Any) -> bool:
    """Return whether ``value`` is a ``torch.distributions.Distribution`` instance."""

    base = _distribution_base()
    return base is not None and not isinstance(value, type) and isinstance(value, base)


def is_distribution_type(container_type: Any) -> bool:
    """Return whether ``container_type`` is a ``Distribution`` subclass."""

    base = _distribution_base()
    return (
        base is not None and isinstance(container_type, type) and issubclass(container_type, base)
    )


def _parameter_names(container_type: type[Any]) -> tuple[str, ...]:
    """Return the family's declared parameter names, read statically from the class.

    ``arg_constraints`` is a plain class dictionary on the standard families; a
    property spelling (or anything else) yields no declared order, and the
    instance attributes are then recorded in their own order.
    """

    for klass in container_type.__mro__:
        declared = klass.__dict__.get("arg_constraints")
        if declared is not None:
            return (
                tuple(name for name in declared if isinstance(name, str))
                if isinstance(declared, dict)
                else ()
            )
    return ()


def _is_portable_literal(value: Any) -> bool:
    """Return whether ``value`` is a literal (or a plain tuple of them) a spec can carry.

    Exact types only: a subclass could carry behaviour the spec does not record.
    ``ContinuousBernoulli._lims`` is a tuple of two floats.
    """

    if type(value) is tuple:
        return all(_is_portable_literal(item) for item in value)
    return type(value) in _LITERAL_TYPES


def _unrecordable(value: Any, name: str) -> str:
    """Spell the lossy reason for one unrecordable attribute."""

    return f"{DISTRIBUTION_STATE_UNRECORDABLE}:{type(value).__qualname__}.{name}"


def distribution_layout(value: Any) -> DistributionLayout:
    """Classify every instance attribute of a live distribution.

    Parameters
    ----------
    value:
        A ``torch.distributions.Distribution`` instance.

    Returns
    -------
    DistributionLayout
        The recorded fields, literal state, aliases and the lossy reason.
    """

    from ._container_type_probes import _fields_are_inert_settable, _type_has_safe_new

    instance_dict = getattr(value, "__dict__", None)
    if not isinstance(instance_dict, dict):
        return DistributionLayout((), (), (), _unrecordable(value, "__dict__"))
    declared = _parameter_names(type(value))
    ordered = [name for name in declared if name in instance_dict]
    ordered += [name for name in instance_dict if name not in declared]
    fields: list[str] = []
    state: list[tuple[str, Any]] = []
    aliases: list[tuple[str, str]] = []
    recorded: dict[int, str] = {}
    lossy_reason = ""
    for name in ordered:
        item = instance_dict[name]
        if not isinstance(name, str) or (name.startswith("__") and name.endswith("__")):
            lossy_reason = lossy_reason or _unrecordable(value, str(name))
        elif isinstance(item, torch.Tensor) and not isinstance(item, torch.nn.Parameter):
            if id(item) in recorded:
                aliases.append((name, recorded[id(item)]))
            else:
                recorded[id(item)] = name
                fields.append(name)
        elif is_distribution(item):
            fields.append(name)
        elif _is_portable_literal(item):
            state.append((name, item))
        else:
            lossy_reason = lossy_reason or _unrecordable(value, name)
    names = (*fields, *(name for name, _ in state), *(alias for alias, _ in aliases))
    container_type = type(value)
    if not lossy_reason and not (
        _type_has_safe_new(container_type) and _fields_are_inert_settable(container_type, names)
    ):
        lossy_reason = _unrecordable(value, "__new__")
    return DistributionLayout(tuple(fields), tuple(state), tuple(aliases), lossy_reason)


def distribution_children(value: Any) -> list[tuple[str, Any]]:
    """Return ``(attribute name, child)`` pairs for a live distribution, in walk order."""

    instance_dict = vars(value)
    return [(name, instance_dict[name]) for name in distribution_layout(value).fields]


def _parse_aux_data(
    aux_data: Any, fields: tuple[str, ...]
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...], str]:
    """Parse a recorded distribution ``aux_data`` into its state, aliases and a problem.

    The spec is portable, attacker-influenceable data, so every name must be a
    plain non-dunder string, every state value a portable literal and every alias
    target a recorded field. The problem is empty when the data is admissible.
    """

    try:
        sections = dict(aux_data)
        state = dict(tuple(sections[_STATE_KEY]))
        aliases = tuple((alias, target) for alias, target in tuple(sections[_ALIASES_KEY]))
    except (TypeError, ValueError, KeyError):
        return {}, (), "its aux_data is malformed"
    names = [*state, *(alias for alias, _ in aliases)]
    for name in names:
        if not isinstance(name, str) or (name.startswith("__") and name.endswith("__")):
            return state, aliases, f"it names an inadmissible attribute {name!r}"
    for name, item in state.items():
        if not _is_portable_literal(item):
            return state, aliases, f"its state {name!r} is not a portable literal"
    for alias, target in aliases:
        if target not in fields:
            return state, aliases, f"its alias {alias!r} names unrecorded field {target!r}"
    return state, aliases, ""


def rebuild_distribution(
    container_type: type[Any] | None,
    spec: Any,
    field_values: dict[str, Any],
) -> Any:
    """Rebuild a distribution from its recorded fields without running ``__init__``.

    Parameters
    ----------
    container_type:
        The resolved, admissible ``Distribution`` subclass, or ``None`` when the
        recorded type is not loaded (there is no plain stand-in, so that refuses).
    spec:
        The distribution ``ContainerSpec`` (its fields, type reference and
        ``aux_data``).
    field_values:
        Rebuilt children by attribute name.

    Returns
    -------
    Any
        A new instance whose instance dictionary equals the recorded one.

    Raises
    ------
    ContainerReconstructionError
        If the type is not loaded or the recorded data is not admissible.
    """

    from ._container_type_probes import _inert_new
    from .container import ContainerReconstructionError

    state, aliases, problem = _parse_aux_data(spec.aux_data, tuple(spec.fields))
    if container_type is None:
        problem = f"its type {spec.type_module}.{spec.type_qualname} is not loaded"
    if problem or container_type is None:
        raise ContainerReconstructionError(
            f"Cannot rebuild the recorded distribution output: {problem}.",
            code="container_spec_inadmissible",
        )
    rebuilt = _inert_new(container_type)
    instance_dict = rebuilt.__dict__
    instance_dict.update(state)
    instance_dict.update(field_values)
    for alias, target in aliases:
        instance_dict[alias] = field_values[target]
    return rebuilt


def distribution_type_rebuild_is_lossy(container_type: type[Any], spec: Any) -> bool:
    """Recompute, from the resolved type at load, whether the inert rebuild is lossy.

    A forged ``lossy_reconstruction=False`` cannot bless a type whose ``__new__`` is
    not inert, whose recorded names are data descriptors, which has no instance
    dictionary, or whose metaclass defines its own ``__call__``.
    """

    from .container import (
        _fields_are_inert_settable,
        _metaclass_defines_foreign_call,
        _type_has_safe_new,
    )

    state, aliases, problem = _parse_aux_data(spec.aux_data, tuple(spec.fields))
    if problem:
        return True
    names = (*spec.fields, *state, *(alias for alias, _ in aliases))
    return not (
        is_distribution_type(container_type)
        and _type_has_safe_new(container_type)
        and _fields_are_inert_settable(container_type, names)
        and not _metaclass_defines_foreign_call(container_type)
    )


def replace_distribution_field(value: Any, name: str, new_value: Any) -> Any:
    """Return a copy of a live distribution with one recorded field replaced.

    Aliases of the replaced tensor (``Categorical._param``) follow it, so the
    copy stays internally consistent. The original object is not mutated.
    Callers pass a name from :func:`distribution_children`.
    """

    layout = distribution_layout(value)
    copied = copy.copy(value)
    copied.__dict__[name] = new_value
    for alias, target in layout.aliases:
        if target == name:
            copied.__dict__[alias] = new_value
    return copied


__all__ = (
    "DISTRIBUTION_CONTAINER_KIND",
    "DISTRIBUTION_STATE_UNRECORDABLE",
    "DistributionLayout",
    "distribution_children",
    "distribution_layout",
    "distribution_type_rebuild_is_lossy",
    "is_distribution",
    "is_distribution_type",
    "rebuild_distribution",
    "replace_distribution_field",
)
