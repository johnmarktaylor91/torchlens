"""``coerce_transform`` — the ONE door every transform slot goes through (memo B1).

Accepts ``None`` | raw unary callable | registered name | frozen spec |
ordered sequence | per-site Mapping, following the ratified predicate-door
pattern: one door means one audit shape. Bare strings resolve only to
zero-parameter deterministic presets; strings are never import paths or a
parameter mini-language. Nested sequences flatten; UNORDERED containers
refuse (T-C8: composition is ordered and recorded).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._context import ContextTransform
from ._errors import TransformContractError
from ._pipeline import OpaqueStep, TransformPipeline
from ._registry import lookup_transform
from ._spec import TransformSpec, freeze_params

__tl_layer__ = "L4"

__all__ = ["DEFAULT", "chain", "coerce_transform", "coerce_transform_mapping"]


class _DefaultKey:
    """Sentinel keying the fallback chain in a per-site transform Mapping."""

    def __repr__(self) -> str:
        """Return the stable sentinel spelling.

        Returns
        -------
        str
            ``"tl.transforms.DEFAULT"``.
        """

        return "tl.transforms.DEFAULT"


#: Mapping key selecting the fallback chain for sites without their own row.
DEFAULT = _DefaultKey()

#: Containers with no defined iteration order — refused, never guessed (T-C8).
_UNORDERED_TYPES = (set, frozenset)


def _coerce_step(value: Any) -> TransformSpec | OpaqueStep:
    """Coerce one non-sequence chain element to a step.

    Parameters
    ----------
    value:
        A spec, registered name, or callable.

    Returns
    -------
    TransformSpec | OpaqueStep
        The coerced step.
    """

    if isinstance(value, TransformSpec):
        # Re-normalize through the registered definition so a hand-built
        # spec cannot smuggle unvalidated params into a chain record.
        definition = lookup_transform(value.name)
        if definition.version != value.version:
            raise TransformContractError(
                f"TransformSpec {value.name!r} was built against algorithm "
                f"version {value.version} but the installed registration is "
                f"version {definition.version} (T-C13: never a silent "
                "substitution).",
                code="transform_version_mismatch",
                remedy=(
                    "install the matching transform version, or rebuild the "
                    "spec against the installed one"
                ),
                name=value.name,
                spec_version=value.version,
                registered_version=definition.version,
            )
        return TransformSpec(
            name=value.name,
            version=value.version,
            params=freeze_params(definition.normalize_params(value.params_dict())),
            seed=value.seed,
            seed_source=value.seed_source,
        )
    if isinstance(value, str):
        definition = lookup_transform(value)
        if not definition.zero_param_preset:
            raise TransformContractError(
                f"Transform name {value!r} is registered but takes parameters; "
                "bare strings resolve only to zero-parameter deterministic "
                "presets (strings are never a parameter mini-language).",
                code="transform_name_not_preset",
                remedy=(f"call the constructor instead, e.g. tl.transforms.{value}(...)"),
                name=value,
            )
        return TransformSpec(
            name=definition.name,
            version=definition.version,
            params=freeze_params(definition.normalize_params({})),
        )
    if isinstance(value, ContextTransform):
        return OpaqueStep(
            module=getattr(value, "__module__", "") or "",
            qualname=getattr(value, "__qualname__", getattr(type(value), "__qualname__", "")),
            context_capable=True,
            fn=value,
        )
    if callable(value):
        return OpaqueStep(
            module=getattr(value, "__module__", "") or "",
            qualname=getattr(value, "__qualname__", getattr(type(value), "__qualname__", "")),
            context_capable=False,
            fn=value,
        )
    raise TransformContractError(
        f"Transform value of type {type(value).__name__} is not coercible; the "
        "door accepts None, a unary callable, a registered name, a frozen "
        "TransformSpec, an ordered sequence of these, or a per-site Mapping.",
        code="transform_coercion_invalid",
        remedy=(
            "pass a callable, a tl.transforms constructor result, a registered "
            "name, or an ordered list of them"
        ),
        value_type=type(value).__name__,
    )


def _flatten_steps(value: Any, out: list[TransformSpec | OpaqueStep]) -> None:
    """Flatten nested ordered sequences into ``out`` (T-C8).

    Parameters
    ----------
    value:
        Chain element or (nested) ordered sequence of elements.
    out:
        Accumulator receiving coerced steps in order.
    """

    if isinstance(value, _UNORDERED_TYPES):
        raise TransformContractError(
            f"Transform chains must be ORDERED; got a {type(value).__name__} "
            "(composition is applied left to right and recorded as one chain, "
            "T-C8 — an unordered container has no left or right).",
            code="transform_coercion_invalid",
            remedy="pass a list or tuple instead of a set",
            value_type=type(value).__name__,
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _flatten_steps(item, out)
        return
    out.append(_coerce_step(value))


def coerce_transform(value: Any) -> TransformPipeline | None:
    """Coerce one transform-slot value into a pipeline (the single door).

    Parameters
    ----------
    value:
        ``None`` | callable | registered name | :class:`TransformSpec` |
        :class:`TransformPipeline` | ordered (possibly nested) sequence.
        Per-site Mappings go through :func:`coerce_transform_mapping`.

    Returns
    -------
    TransformPipeline | None
        The ordered chain, or ``None`` when no transform is configured.

    Raises
    ------
    TransformContractError
        ``transform_coercion_invalid`` on uncoercible values or unordered
        containers; ``transform_name_unknown`` / ``transform_name_not_preset``
        on string resolution failures.
    """

    if value is None:
        return None
    if isinstance(value, TransformPipeline):
        return value
    if isinstance(value, Mapping):
        raise TransformContractError(
            "A per-site transform Mapping cannot coerce to one chain; the "
            "engine resolves Mappings per output key BEFORE coercion.",
            code="transform_coercion_invalid",
            remedy="pass the Mapping to the extraction slot, or one chain here",
            value_type=type(value).__name__,
        )
    steps: list[TransformSpec | OpaqueStep] = []
    _flatten_steps(value, steps)
    return TransformPipeline(steps=tuple(steps))


def coerce_transform_mapping(
    value: Mapping[Any, Any], output_keys: Sequence[str]
) -> dict[str, TransformPipeline | None]:
    """Resolve a per-site transform Mapping per output key, then coerce.

    Parameters
    ----------
    value:
        Mapping from output key (or :data:`DEFAULT`) to a transform-slot
        value.
    output_keys:
        The run's output keys, in plan order.

    Returns
    -------
    dict[str, TransformPipeline | None]
        One coerced chain (or ``None``) per output key. Keys without a row
        fall back to the :data:`DEFAULT` row when present, else ``None``.

    Raises
    ------
    TransformContractError
        ``transform_mapping_key_unknown`` when a Mapping key names no output
        key (a typo'd site name silently untransformed is a fails-open bug).
    """

    known = set(output_keys)
    unknown = sorted(
        str(key) for key in value if not isinstance(key, _DefaultKey) and key not in known
    )
    if unknown:
        raise TransformContractError(
            f"Per-site transform Mapping names unknown output keys {unknown}; "
            f"this run's output keys are {sorted(known)}.",
            code="transform_mapping_key_unknown",
            remedy=(
                "key the Mapping by the run's output keys (and "
                "tl.transforms.DEFAULT for the fallback chain)"
            ),
            unknown_keys=unknown,
            output_keys=sorted(known),
        )
    default_value: Any = None
    for key, row in value.items():
        if isinstance(key, _DefaultKey):
            default_value = row
    resolved: dict[str, TransformPipeline | None] = {}
    for key in output_keys:
        resolved[key] = coerce_transform(value.get(key, default_value))
    return resolved


def chain(*items: Any) -> TransformPipeline:
    """Build an ordered chain: ``chain(a, b, c)`` == coercing ``[a, b, c]``.

    Parameters
    ----------
    *items:
        Chain elements (specs, names, callables, nested ordered sequences).

    Returns
    -------
    TransformPipeline
        The ordered chain (memo decision 3: both spellings normalize to the
        same object).
    """

    steps: list[TransformSpec | OpaqueStep] = []
    _flatten_steps(list(items), steps)
    return TransformPipeline(steps=tuple(steps))
