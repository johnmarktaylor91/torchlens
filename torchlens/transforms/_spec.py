"""Frozen TransformSpec + plan/apply protocol + canonical JSON (memo B1).

A built-in transform is a frozen :class:`TransformSpec` — canonical JSON of
``(name, numerics-visible version, normalized params, explicit seed where
stochastic)`` — with ``plan(input_spec, ctx)`` the runtime validates BEFORE a
shard publishes and a pure ``apply(tensor, ctx)``. All five measured
fails-open defects in the v1 slot were the engine trusting an opaque
callable; a validated plan turns each one into a typed refusal.

The canonical JSON of a chain of these records is the RESUME IDENTITY: any
numerics-visible change mismatches (transforms memo decision 17).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from ._context import TransformContext
from ._errors import TransformContractError

__tl_layer__ = "L4"

__all__ = [
    "PlannedStep",
    "TensorSpec",
    "TransformDefinition",
    "TransformSpec",
    "canonical_json",
]

#: Closed seed-source vocabulary (memo decision 4): the realized seed is
#: ALWAYS recorded together with where it came from.
_SEED_SOURCES: tuple[str, ...] = ("explicit", "library_default")


def freeze_params(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Freeze a normalized param mapping into the hashable spec form.

    Lists freeze to tuples (recursively) so a spec rebuilt from a JSON record
    equals the spec the factory built — canonical JSON serializes both back
    to the same arrays.

    Parameters
    ----------
    params:
        Normalized parameter mapping.

    Returns
    -------
    tuple[tuple[str, Any], ...]
        Sorted, hashable ``(key, value)`` pairs.
    """

    def _freeze(value: Any) -> Any:
        """Recursively freeze lists/tuples to tuples; pass scalars through."""

        if isinstance(value, (list, tuple)):
            return tuple(_freeze(item) for item in value)
        return value

    return tuple(sorted((key, _freeze(value)) for key, value in params.items()))


def canonical_json(value: Any) -> str:
    """Serialize a JSON-portable value to its one canonical byte form.

    Sorted keys, minimal separators, ASCII-only, NaN/Infinity refused: two
    equal records always produce byte-identical strings, so string equality
    of canonical JSON IS record equality (the resume-identity comparison).

    Parameters
    ----------
    value:
        JSON-portable value (scalars, lists, string-keyed dicts).

    Returns
    -------
    str
        Canonical JSON string.

    Raises
    ------
    TransformContractError
        ``transform_params_invalid`` when the value is not JSON-portable.
    """

    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise TransformContractError(
            f"Value is not canonical-JSON-portable: {exc}.",
            code="transform_params_invalid",
            remedy=(
                "restrict transform params to JSON scalars, lists, and "
                "string-keyed mappings (no tensors, no NaN/Infinity)"
            ),
        ) from exc


@dataclass(frozen=True)
class TensorSpec:
    """Shape/dtype description of a batch tensor at a chain position.

    Attributes
    ----------
    shape:
        Full shape INCLUDING the leading stimulus axis; ``None`` marks an
        unknown extent (the batch extent is normally ``None`` — the plan is
        batch-composition-independent, T-C10).
    dtype:
        ``str(torch.dtype)`` spelling, e.g. ``"torch.float32"``.
    """

    shape: tuple[int | None, ...]
    dtype: str

    @staticmethod
    def of(tensor: torch.Tensor, *, generalize_batch: bool = True) -> TensorSpec:
        """Describe a concrete tensor, optionally generalizing the batch extent.

        Parameters
        ----------
        tensor:
            Tensor with the stimulus axis leading.
        generalize_batch:
            When true (default), record the batch extent as ``None`` so the
            plan holds for every batch size.

        Returns
        -------
        TensorSpec
            The spec describing ``tensor``.
        """

        shape: tuple[int | None, ...] = tuple(int(s) for s in tensor.shape)
        if generalize_batch and shape:
            shape = (None, *shape[1:])
        return TensorSpec(shape=shape, dtype=str(tensor.dtype))


@dataclass(frozen=True)
class PlannedStep:
    """The validated prediction one chain step makes before any shard publishes.

    Attributes
    ----------
    name:
        Transform name (registry entry id), or the opaque disclosure name.
    version:
        Numerics-visible algorithm version of the step.
    output:
        Predicted output :class:`TensorSpec` (T-C6: runtime output must match
        or the shard is refused BEFORE publication).
    stream_safe:
        Whether the step can run at emission time (T-C11).
    may_alias:
        Whether the output may share storage with the input — a first-class
        planning fact; no peak-memory claim follows from shape reduction
        alone (T-C11).
    context_capable:
        Whether the step declared context capability (T-C1).
    """

    name: str
    version: int
    output: TensorSpec
    stream_safe: bool
    may_alias: bool
    context_capable: bool


@dataclass(frozen=True)
class TransformDefinition:
    """Registry unit binding a transform name+version to its behavior.

    Attributes
    ----------
    name:
        Stable transform name (the registry entry id).
    version:
        Numerics-visible algorithm version; any numeric change bumps it
        (T-C13: a construction change is never a silent substitution).
    normalize_params:
        Callable validating and normalizing raw constructor params into the
        canonical JSON-portable dict recorded in the spec.
    plan_fn:
        ``(spec, input_spec, ctx) -> PlannedStep`` shape/dtype predictor;
        refuses typed instead of guessing.
    apply_fn:
        ``(spec, tensor, ctx) -> tensor`` pure application; row count and
        order preserved (T-C2).
    context_capable:
        Whether ``apply_fn`` consumes the context (T-C1 declaration).
    stream_safe:
        Whether the transform can run at emission time (T-C11).
    zero_param_preset:
        Whether a BARE STRING may coerce to this transform with all-default
        params (memo decision 2: strings are never import paths or a
        parameter mini-language).
    stochastic:
        Whether the transform draws randomness; stochastic transforms must
        record an explicit realized seed + seed_source (T-C4).
    """

    name: str
    version: int
    normalize_params: Callable[[Mapping[str, Any]], dict[str, Any]]
    plan_fn: Callable[[TransformSpec, TensorSpec, TransformContext | None], PlannedStep]
    apply_fn: Callable[[TransformSpec, torch.Tensor, TransformContext | None], torch.Tensor]
    context_capable: bool = False
    stream_safe: bool = True
    zero_param_preset: bool = False
    stochastic: bool = False


@dataclass(frozen=True)
class TransformSpec:
    """One frozen, declared transform step (the B1 protocol object).

    Attributes
    ----------
    name:
        Registered transform name.
    version:
        Numerics-visible algorithm version the spec was built against.
    params:
        Normalized params as a sorted tuple of ``(key, value)`` pairs
        (kept hashable; :meth:`params_dict` rebuilds the mapping).
    seed:
        Realized seed for stochastic transforms; ``None`` for deterministic.
    seed_source:
        ``"explicit" | "library_default"`` for stochastic transforms;
        ``None`` for deterministic (T-C4: always recorded, never implied).
    """

    name: str
    version: int
    params: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    seed: int | None = None
    seed_source: str | None = None

    def __post_init__(self) -> None:
        """Validate the seed/seed-source pairing invariant (T-C4)."""

        if (self.seed is None) != (self.seed_source is None):
            raise TransformContractError(
                f"TransformSpec {self.name!r} pairs seed={self.seed!r} with "
                f"seed_source={self.seed_source!r}; a realized seed and its "
                "source are recorded together or not at all.",
                code="transform_params_invalid",
                remedy="pass both seed and seed_source, or neither",
                name=self.name,
            )
        if self.seed_source is not None and self.seed_source not in _SEED_SOURCES:
            raise TransformContractError(
                f"TransformSpec {self.name!r} carries seed_source="
                f"{self.seed_source!r}; the closed vocabulary is {_SEED_SOURCES}.",
                code="transform_params_invalid",
                remedy="record seed_source as 'explicit' or 'library_default'",
                name=self.name,
                seed_source=self.seed_source,
            )

    def params_dict(self) -> dict[str, Any]:
        """Return the normalized params as a plain dict.

        Returns
        -------
        dict[str, Any]
            Normalized parameter mapping.
        """

        return dict(self.params)

    def canonical_record(self) -> dict[str, Any]:
        """Return the canonical static-chain record of this step (memo 3.17).

        Returns
        -------
        dict[str, Any]
            ``{kind, name, version, params, seed, seed_source}`` — the
            portable step row embedded in ``tl_transform_pipeline_v1``.
        """

        return {
            "kind": "spec",
            "name": self.name,
            "version": self.version,
            "params": self.params_dict(),
            "seed": self.seed,
            "seed_source": self.seed_source,
        }

    def canonical_json(self) -> str:
        """Return this step's canonical JSON (byte-stable identity).

        Returns
        -------
        str
            Canonical JSON of :meth:`canonical_record`.
        """

        return canonical_json(self.canonical_record())

    def _definition(self) -> TransformDefinition:
        """Resolve this spec's registered definition or refuse typed.

        Returns
        -------
        TransformDefinition
            The registry entry for ``(name)`` with a matching version.
        """

        from ._registry import lookup_transform

        definition = lookup_transform(self.name)
        if definition.version != self.version:
            raise TransformContractError(
                f"TransformSpec {self.name!r} was built against algorithm "
                f"version {self.version} but the installed registration is "
                f"version {definition.version}; a version change is "
                "numerics-visible and never silently substituted (T-C13).",
                code="transform_version_mismatch",
                remedy=(
                    "install the matching transform version, or rebuild the "
                    "spec against the installed one"
                ),
                name=self.name,
                spec_version=self.version,
                registered_version=definition.version,
            )
        return definition

    def plan(self, input_spec: TensorSpec, ctx: TransformContext | None = None) -> PlannedStep:
        """Predict and validate this step's output for ``input_spec``.

        Parameters
        ----------
        input_spec:
            Description of the incoming batch tensor.
        ctx:
            Optional context spec; ``None`` is always legal.

        Returns
        -------
        PlannedStep
            The validated output prediction.
        """

        return self._definition().plan_fn(self, input_spec, ctx)

    def apply(self, tensor: torch.Tensor, ctx: TransformContext | None = None) -> torch.Tensor:
        """Apply this step to one batch tensor (pure; T-C2 row-preserving).

        Parameters
        ----------
        tensor:
            Batch tensor with the stimulus axis leading.
        ctx:
            Optional context; ``None`` is always legal.

        Returns
        -------
        torch.Tensor
            The transformed tensor.
        """

        return self._definition().apply_fn(self, tensor, ctx)
