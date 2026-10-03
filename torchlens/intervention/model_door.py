"""The ONE audited model-argument funnel (F01; surgery memo 3.3 item 10, OP2).

TorchLens's own model doors (``tl.trace``, ``tl.record``, ``tl.validate``,
``tl.release_model``, and ``spec.bind`` itself) never silently treat a bound
intervention executor as a plain model. Whether they refuse with the canonical
spelling printed (arm a), or normalize through this one shared audited funnel
(arm b), is open parameter OP2 (FORK-2, unruled): BOTH arms are built behind
ONE switch and both stay tested until it is ruled. Under either arm the outcome
is identical audit and artifact identity, and no door can quietly capture the
UNMODIFIED model.

Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from .._errors import ArgumentTypeError, InvalidArgumentError

DoorPolicy = Literal["refuse", "normalize"]

#: The complete inventory of public TorchLens doors that accept a model
#: operand. ``True`` marks doors wired through :func:`resolve_model_operand`
#: today; ``False`` marks doors whose modules sit outside lane F01's fence
#: (their wiring is a declared remainder, not a silent gap). The
#: funnel-coverage test (``tests/test_surgery_live_bind.py``) asserts every
#: wired door routes its model operand through this one funnel.
MODEL_DOOR_INVENTORY: dict[str, bool] = {
    "trace": True,
    "release_model": True,
    "bind": True,
    "record": False,
    "validate": False,
    "compat.report": False,
    "extract_dataset": False,
    "autoroute": False,
}

_DOOR_POLICY: ContextVar[DoorPolicy] = ContextVar("tl_model_door_policy", default="refuse")

_POLICIES: frozenset[str] = frozenset({"refuse", "normalize"})


def model_door_policy() -> DoorPolicy:
    """Return the active OP2 model-door policy (``"refuse"`` is the default)."""

    return _DOOR_POLICY.get()


@contextmanager
def door_policy(policy: str) -> Iterator[None]:
    """Scoped OP2 door-policy switch (both arms tested until ruled).

    Parameters
    ----------
    policy:
        ``"refuse"`` (arm a: refuse-with-teaching, printing the canonical
        base-model + spec spelling) or ``"normalize"`` (arm b: normalize the
        binding through the funnel into ``base_model`` + ``intervene=spec``).

    Raises
    ------
    InvalidArgumentError
        ``model_door_policy_invalid`` for a policy outside the closed set.
    """

    if policy not in _POLICIES:
        raise InvalidArgumentError(
            f"model door policy must be 'refuse' or 'normalize', got {policy!r}",
            code="model_door_policy_invalid",
            remedy="pass 'refuse' (OP2 arm a) or 'normalize' (OP2 arm b)",
            argument="policy",
        )
    token = _DOOR_POLICY.set(policy)  # type: ignore[arg-type]
    try:
        yield
    finally:
        _DOOR_POLICY.reset(token)


@dataclass(frozen=True)
class DoorResolution:
    """One funnel verdict for one model operand at one door.

    Parameters
    ----------
    model:
        The model the door should actually operate on (the base model when a
        binding was normalized).
    spec:
        The immutable ``InterventionSpec`` extracted from a normalized
        binding, else ``None``.
    normalized:
        Whether a binding was unwrapped (arm b fired).
    door:
        The door name the operand arrived at (audit disclosure).
    """

    model: Any
    spec: Any = None
    normalized: bool = False
    door: str = ""


def _is_binding(model: Any) -> bool:
    """Return whether ``model`` is a bound intervention executor.

    Checked structurally (module-qualified type name) so the funnel never
    imports the binding engine just to classify a plain model.
    """

    cls = type(model)
    return cls.__name__ == "BoundInterventionExecutor" and cls.__module__.startswith(
        "torchlens.intervention"
    )


def resolve_model_operand(model: Any, *, door: str, intervene: Any = None) -> DoorResolution:
    """Route one model operand through the ONE audited model-door funnel.

    Every wired door calls this before touching its model operand. Plain
    models pass through untouched. A bound intervention executor resolves by
    the active OP2 policy: arm (a) refuses with the canonical spelling
    printed; arm (b) returns the base model plus the binding's spec so the
    door composes ``intervene=spec`` -- producing audit and artifact identity
    identical to the canonical spelling.

    Parameters
    ----------
    model:
        The door's model operand.
    door:
        Door name for the refusal message and audit disclosure.
    intervene:
        The door's own ``intervene=`` operand, when it has one; a normalized
        binding may not silently collide with an explicit spec.

    Raises
    ------
    ArgumentTypeError
        ``model_door_binding_refused`` under arm (a).
    InvalidArgumentError
        ``model_door_intervene_conflict`` when arm (b) would have to merge
        the binding's spec with an explicit ``intervene=`` operand.
    """

    if not _is_binding(model):
        return DoorResolution(model=model, door=door)
    if model_door_policy() == "refuse":
        raise ArgumentTypeError(
            f"tl.{door} received a bound intervention executor, not a model; "
            "a binding is a capture-free live lane and doors never silently "
            "treat it as a plain model. The canonical spelling is "
            f"tl.{door}(binding.base_model, ..., intervene=binding.spec)",
            code="model_door_binding_refused",
            remedy="pass binding.base_model as the model and binding.spec as "
            "intervene=, or run the capture-free lane by calling the binding",
            argument="model",
        )
    if intervene is not None:
        raise InvalidArgumentError(
            f"tl.{door} received BOTH a bound intervention executor and an "
            "explicit intervene= spec; the funnel never merges two spec "
            "sources silently",
            code="model_door_intervene_conflict",
            remedy="merge the specs explicitly (binding.spec & other_spec) "
            "and pass one spec, or pass the plain base model",
            argument="intervene",
        )
    return DoorResolution(
        model=model.base_model,
        spec=model.spec,
        normalized=True,
        door=door,
    )


def resolve_trace_operands(model: Any, intervene: Any) -> tuple[Any, Any]:
    """Resolve ``tl.trace``'s ``(model, intervene)`` pair through the funnel.

    The one-line trace-door spelling: it runs BEFORE the quickstart input
    ladder so a bound executor never reaches the ``nn.Module``-typed ladder
    unresolved, and it returns the operands the entry consumes directly --
    no residual locals for the entry's ``locals().copy()`` payload builders
    to pick up. Arm (a) refuses inside :func:`resolve_model_operand`; arm
    (b) normalizes to ``(base model, binding spec)``; a plain model passes
    through with both operands untouched.
    """

    resolution = resolve_model_operand(model, door="trace", intervene=intervene)
    if resolution.normalized:
        return resolution.model, resolution.spec
    return model, intervene


def resolve_released_model(model: Any) -> Any:
    """Resolve ``tl.release_model``'s operand through the funnel.

    The one-line release-door spelling: arm (a) refuses inside
    :func:`resolve_model_operand`; arm (b) returns the binding's base model
    so release operates on the model that was actually prepared.
    """

    resolution = resolve_model_operand(model, door="release_model")
    return resolution.model if resolution.normalized else model


__all__ = [
    "MODEL_DOOR_INVENTORY",
    "DoorPolicy",
    "DoorResolution",
    "door_policy",
    "model_door_policy",
    "resolve_model_operand",
    "resolve_released_model",
    "resolve_trace_operands",
]
