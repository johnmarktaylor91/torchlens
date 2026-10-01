"""Ordered transform chains + the portable ``tl_transform_pipeline_v1`` record (memo B2).

Chains are ordered sequences planned and applied LEFT TO RIGHT, one numeric
record (T-C8). The canonical JSON of the static chain is the RESUME IDENTITY:
any numerics-visible change mismatches (memo 3.17). Opaque callables are
accepted and identified (``identification_only``); the strict opaque-resume
rule (memo decision 14) makes cross-process resume with an opaque step a
typed refusal — enforced by the consuming engine, disclosed here through
:attr:`TransformPipeline.resume_verifiable`.

Artifact loading NEVER imports or executes code: :func:`pipeline_from_record`
rebuilds builtins and registered names through the registry; an opaque record
rehydrates as an unresolvable step that refuses application typed.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ._context import TransformContext, wants_context
from ._errors import TransformContractError
from ._registry import lookup_transform
from ._spec import PlannedStep, TensorSpec, TransformSpec, canonical_json, freeze_params

__tl_layer__ = "L4"

__all__ = [
    "PIPELINE_SCHEMA",
    "OpaqueStep",
    "TransformPipeline",
    "pipeline_from_record",
    "pipeline_record",
]

#: Portable sub-schema id for the static pipeline record (memo 3.17).
PIPELINE_SCHEMA = "tl_transform_pipeline_v1"


@dataclass(frozen=True)
class OpaqueStep:
    """A raw user callable accepted into a chain with identification only.

    Attributes
    ----------
    module:
        ``__module__`` disclosure of the callable (or its type).
    qualname:
        ``__qualname__`` disclosure of the callable (or its type).
    context_capable:
        Whether the callable DECLARED context capability
        (:class:`~torchlens.transforms.ContextTransform`); never sniffed.
    fn:
        The live callable, or ``None`` when rehydrated from an artifact
        record (identification cannot resurrect code; applying an
        unresolved opaque step refuses typed).
    """

    module: str
    qualname: str
    context_capable: bool
    fn: Any = None

    def canonical_record(self) -> dict[str, Any]:
        """Return the opaque step's static-chain record.

        Returns
        -------
        dict[str, Any]
            ``{kind, module, qualname, identity}`` disclosure row.
        """

        return {
            "kind": "opaque",
            "module": self.module,
            "qualname": self.qualname,
            "identity": "identification_only",
        }

    def apply(self, tensor: torch.Tensor, ctx: TransformContext | None) -> torch.Tensor:
        """Invoke the callable (unary unless context-capable by declaration).

        Parameters
        ----------
        tensor:
            Batch tensor with the stimulus axis leading.
        ctx:
            Optional context; passed only to declared context-capable steps.

        Returns
        -------
        torch.Tensor
            The callable's result.

        Raises
        ------
        TransformContractError
            ``transform_opaque_unresolvable`` when the step was rehydrated
            from a record and holds no code.
        """

        if self.fn is None:
            raise TransformContractError(
                f"Opaque transform step {self.module}.{self.qualname} was "
                "rehydrated from an artifact record; identification is a "
                "disclosure, not code, so it cannot be applied.",
                code="transform_opaque_unresolvable",
                remedy=(
                    "register the transform under a versioned name "
                    "(torchlens.transforms.register_transform) and rebuild "
                    "the chain from names, or pass the live callable"
                ),
                module=self.module,
                qualname=self.qualname,
            )
        if self.context_capable:
            return self.fn(tensor, ctx)
        return self.fn(tensor)


@dataclass(frozen=True)
class TransformPipeline:
    """One ordered chain of transform steps (the single numeric record).

    Attributes
    ----------
    steps:
        Ordered steps: frozen :class:`TransformSpec` rows and/or
        :class:`OpaqueStep` disclosures. Applied left to right.
    """

    steps: tuple[TransformSpec | OpaqueStep, ...]

    @property
    def resume_verifiable(self) -> bool:
        """Whether the static chain fully reconstructs this pipeline.

        Returns
        -------
        bool
            ``False`` iff any step is opaque (identification-only); the
            engine's strict opaque-resume rule keys on this.
        """

        return all(isinstance(step, TransformSpec) for step in self.steps)

    def canonical_chain(self) -> str:
        """Return the canonical JSON of the static chain (the resume identity).

        Returns
        -------
        str
            Canonical JSON array of step records; byte equality IS chain
            equality, and any numerics-visible change mismatches.
        """

        return canonical_json([step.canonical_record() for step in self.steps])

    def plan(
        self, input_spec: TensorSpec, ctx: TransformContext | None = None
    ) -> tuple[PlannedStep | None, ...]:
        """Plan the chain left to right (T-C6/T-C8).

        Parameters
        ----------
        input_spec:
            Description of the tensor entering the chain.
        ctx:
            Optional context spec; ``None`` always legal.

        Returns
        -------
        tuple[PlannedStep | None, ...]
            One entry per step: a validated prediction for spec steps,
            ``None`` for opaque steps (an opaque callable has no plan; the
            engine validates its observed output per shard instead of
            trusting batch one forever). Once an opaque step breaks the
            spec chain, every LATER entry is also ``None`` — planning a
            spec step against a fabricated post-opaque shape would be a
            guess, and a guessed plan is worse than no plan.
        """

        planned: list[PlannedStep | None] = []
        current: TensorSpec | None = input_spec
        for step in self.steps:
            if isinstance(step, TransformSpec) and current is not None:
                plan = step.plan(current, ctx)
                planned.append(plan)
                current = plan.output
            else:
                planned.append(None)
                current = None
        return tuple(planned)

    def apply(self, tensor: torch.Tensor, ctx: TransformContext | None = None) -> torch.Tensor:
        """Apply the chain left to right with declared ctx dispatch (T-C1).

        Parameters
        ----------
        tensor:
            Batch tensor with the stimulus axis leading.
        ctx:
            Optional context; ``None`` always legal.

        Returns
        -------
        torch.Tensor
            The chained result.
        """

        result = tensor
        for step in self.steps:
            result = step.apply(result, ctx)
        return result


def pipeline_record(pipeline: TransformPipeline | None) -> dict[str, Any] | None:
    """Build the portable ``tl_transform_pipeline_v1`` static record.

    Parameters
    ----------
    pipeline:
        The chain, or ``None`` when no transform is configured.

    Returns
    -------
    dict[str, Any] | None
        The embedded sub-schema record (static chain + resume identity +
        verifiability), or ``None``.
    """

    if pipeline is None:
        return None
    return {
        "schema": PIPELINE_SCHEMA,
        "steps": [step.canonical_record() for step in pipeline.steps],
        "canonical_chain": pipeline.canonical_chain(),
        "resume_verifiable": pipeline.resume_verifiable,
    }


def pipeline_from_record(record: Mapping[str, Any]) -> TransformPipeline:
    """Rehydrate a pipeline from its ``tl_transform_pipeline_v1`` record.

    Explicit rehydration rebuilds builtins and registered names through the
    registry; a custom name requires a MATCHING versioned registration.
    Opaque rows rehydrate as unresolvable disclosures. No code is imported
    or executed.

    Parameters
    ----------
    record:
        The persisted sub-schema record.

    Returns
    -------
    TransformPipeline
        The rehydrated chain (spec steps live; opaque steps disclosure-only).

    Raises
    ------
    TransformContractError
        ``transform_pipeline_record_invalid`` on a malformed record;
        ``transform_name_unknown`` / ``transform_version_mismatch`` when a
        recorded name has no matching installed registration.
    """

    if not isinstance(record, Mapping) or record.get("schema") != PIPELINE_SCHEMA:
        raise TransformContractError(
            f"Transform pipeline record does not carry schema {PIPELINE_SCHEMA!r} "
            f"(found {record.get('schema') if isinstance(record, Mapping) else type(record).__name__!r}).",
            code="transform_pipeline_record_invalid",
            remedy="pass the manifest's recorded transform_pipeline block unmodified",
        )
    steps: list[TransformSpec | OpaqueStep] = []
    raw_steps = record.get("steps")
    if not isinstance(raw_steps, (list, tuple)):
        raise TransformContractError(
            "Transform pipeline record has no ordered steps array.",
            code="transform_pipeline_record_invalid",
            remedy="pass the manifest's recorded transform_pipeline block unmodified",
        )
    for row in raw_steps:
        if not isinstance(row, Mapping):
            raise TransformContractError(
                f"Transform pipeline step row is not a mapping: {row!r}.",
                code="transform_pipeline_record_invalid",
                remedy="pass the manifest's recorded transform_pipeline block unmodified",
            )
        kind = row.get("kind")
        if kind == "opaque":
            steps.append(
                OpaqueStep(
                    module=str(row.get("module", "")),
                    qualname=str(row.get("qualname", "")),
                    context_capable=False,
                    fn=None,
                )
            )
            continue
        if kind != "spec":
            raise TransformContractError(
                f"Transform pipeline step kind {kind!r} is not 'spec' or 'opaque'.",
                code="transform_pipeline_record_invalid",
                remedy="pass the manifest's recorded transform_pipeline block unmodified",
                kind=kind,
            )
        name = str(row.get("name", ""))
        definition = lookup_transform(name)
        params = row.get("params") or {}
        if not isinstance(params, Mapping):
            raise TransformContractError(
                f"Transform pipeline step {name!r} carries non-mapping params.",
                code="transform_pipeline_record_invalid",
                remedy="pass the manifest's recorded transform_pipeline block unmodified",
                name=name,
            )
        seed = row.get("seed")
        seed_source = row.get("seed_source")
        spec = TransformSpec(
            name=name,
            version=int(row.get("version", definition.version)),
            params=freeze_params(definition.normalize_params(params)),
            seed=None if seed is None else int(seed),
            seed_source=None if seed_source is None else str(seed_source),
        )
        steps.append(spec)
    return TransformPipeline(steps=tuple(steps))


def apply_dispatch(step: Any, tensor: torch.Tensor, ctx: TransformContext | None) -> torch.Tensor:
    """Invoke one raw callable with ctx dispatch BY DECLARATION (P2).

    Parameters
    ----------
    step:
        A raw unary callable or a declared
        :class:`~torchlens.transforms.ContextTransform`.
    tensor:
        Batch tensor.
    ctx:
        Optional context.

    Returns
    -------
    torch.Tensor
        The callable's result. Dispatch is one isinstance check — never
        ``inspect.signature`` (15 measured counterexamples).
    """

    if wants_context(step):
        return step(tensor, ctx)
    return step(tensor)
