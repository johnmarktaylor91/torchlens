"""TransformContext + axis-role declarations (transforms memo B4; contract T-C1/T-C5).

Axis semantics are EVIDENCE-BASED; rank is never evidence. Roles come from
the closed evidence vocabulary ``declared | op_contract | unknown`` with
``inferred`` RESERVED (a guessed layout is a wrong-number bug by definition).
Capability dispatch is by explicit DECLARATION, never ``inspect.signature``:
every ``nn.Module`` and C-implemented torch op reports ``(*args, **kwargs)``
and would be falsely called with ctx, while keyword-only-ctx partials report
unary and would silently run mask-blind — both failure directions measured on
15 real callables (transforms memo decision 15). One isinstance check fixes
it: :func:`wants_context` tests :class:`ContextTransform` and nothing else.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ._errors import TransformContractError

__tl_layer__ = "L4"

__all__ = [
    "BTD",
    "NCHW",
    "ROLE_EVIDENCE",
    "ContextTransform",
    "RoleDeclaration",
    "TransformContext",
    "axis_for_role",
    "wants_context",
    "with_context",
]

#: Closed axis-role evidence vocabulary (transforms memo decision 11).
#: ``inferred`` is RESERVED: rank-based guessing is never admissible evidence.
ROLE_EVIDENCE: tuple[str, ...] = ("declared", "op_contract", "unknown")


@dataclass(frozen=True)
class RoleDeclaration:
    """An ordered per-axis role assertion with its evidence source.

    Attributes
    ----------
    axes:
        One role name per tensor axis, in axis order (axis 0 is the stimulus
        axis and its role is conventionally ``"batch"``).
    evidence:
        Member of :data:`ROLE_EVIDENCE`. ``"declared"`` records a caller
        assertion; ``"op_contract"`` is the v1.5 trace-derived producer.
    """

    axes: tuple[str, ...]
    evidence: str = "declared"

    def __post_init__(self) -> None:
        """Validate the closed evidence vocabulary and role-name shape."""

        if self.evidence not in ROLE_EVIDENCE:
            raise TransformContractError(
                f"Axis-role evidence {self.evidence!r} is not in the closed "
                f"vocabulary {ROLE_EVIDENCE} ('inferred' is reserved: rank is "
                "never evidence for axis semantics).",
                code="transform_role_evidence_invalid",
                remedy=(
                    "declare roles with evidence='declared' (a recorded caller "
                    "assertion) or leave roles undeclared"
                ),
                evidence=self.evidence,
            )
        if not self.axes or not all(isinstance(role, str) and role for role in self.axes):
            raise TransformContractError(
                f"Axis roles must be a non-empty tuple of non-empty strings; got {self.axes!r}.",
                code="transform_role_declaration_invalid",
                remedy="pass one role name per tensor axis, e.g. ('batch', 'token', 'feature')",
                axes=self.axes,
            )


#: Prebuilt vision declaration: batch x channel x height x width.
NCHW = RoleDeclaration(axes=("batch", "channel", "height", "width"))

#: Prebuilt transformer declaration: batch x token x feature.
BTD = RoleDeclaration(axes=("batch", "token", "feature"))


def axis_for_role(roles: RoleDeclaration | None, rank: int, role: str) -> int:
    """Resolve a semantic role to a concrete axis or refuse with both remedies.

    Parameters
    ----------
    roles:
        The declaration attached to the tensor, or ``None`` when nothing was
        declared (semantic presets must then refuse, never guess).
    rank:
        Rank of the tensor being addressed.
    role:
        Semantic role to resolve (e.g. ``"token"``).

    Returns
    -------
    int
        The axis index carrying ``role``.

    Raises
    ------
    TransformContractError
        ``transform_axis_roles_unavailable`` when no declaration covers the
        role at this rank; the teaching message names both remedies.
    """

    if roles is not None and len(roles.axes) == rank and role in roles.axes:
        return roles.axes.index(role)
    raise TransformContractError(
        f"No recorded axis role resolves {role!r} for a rank-{rank} tensor "
        f"(declared roles: {None if roles is None else roles.axes!r}); "
        "TorchLens never guesses axis semantics from rank.",
        code="transform_axis_roles_unavailable",
        remedy=(
            "declare axis roles (e.g. TransformContext(roles=tl.transforms.BTD)) "
            "or use the explicit-axis primitive (reduce/take_index/unit_norm "
            "with axis=)"
        ),
        role=role,
        rank=rank,
        declared=None if roles is None else list(roles.axes),
    )


@dataclass(frozen=True)
class TransformContext:
    """Frozen per-invocation context handed to context-capable transforms.

    ``None`` is accepted FOREVER wherever a context is consumed: role-free
    kernels ignore the context entirely, and every built-in must behave
    identically under ``ctx=None`` and an empty context.

    Attributes
    ----------
    roles:
        Optional axis-role declaration for the tensor being transformed.
    mask:
        Optional validity mask (e.g. an attention mask) aligned with the
        stimulus axis; consumed by mask-aware pooling built-ins.
    site_label:
        Output key / site label the tensor was extracted under, for
        per-site diagnostics in refusal text.
    workspace:
        Optional planner byte budget carried from v1 of this dataclass;
        ``None`` means no declared budget.
    """

    roles: RoleDeclaration | None = None
    mask: torch.Tensor | None = None
    site_label: str | None = None
    workspace: int | None = None


class ContextTransform:
    """Base class DECLARING context capability for a user transform (T-C1).

    Subclass (or wrap a callable with :func:`with_context`) to receive the
    :class:`TransformContext` alongside each tensor. Raw callables stay
    unary; capability dispatch is one isinstance check on this class, never
    ``inspect.signature`` (15 measured counterexamples, memo decision 15).
    """

    #: Declaration consumed by :func:`wants_context`; never sniffed.
    context_capable: bool = True

    def __call__(self, tensor: torch.Tensor, ctx: TransformContext | None) -> torch.Tensor:
        """Apply the transform to one batch tensor under an optional context.

        Parameters
        ----------
        tensor:
            Batch tensor with the stimulus axis leading.
        ctx:
            Per-invocation context; ``None`` is always legal.

        Returns
        -------
        torch.Tensor
            Transformed tensor with the stimulus axis preserved (T-C2).
        """

        raise NotImplementedError


class _WrappedContextTransform(ContextTransform):
    """Adapter minted by :func:`with_context` around a ``(tensor, ctx)`` callable."""

    def __init__(self, fn: Callable[[torch.Tensor, TransformContext | None], torch.Tensor]):
        """Wrap ``fn`` as an explicitly context-capable transform.

        Parameters
        ----------
        fn:
            Callable taking ``(tensor, ctx)`` and returning the transformed
            tensor.
        """

        self._fn = fn
        self.__module__ = getattr(fn, "__module__", type(self).__module__)
        self.__qualname__ = getattr(
            fn, "__qualname__", getattr(type(fn), "__qualname__", "with_context")
        )

    def __call__(self, tensor: torch.Tensor, ctx: TransformContext | None) -> torch.Tensor:
        """Delegate to the wrapped callable with the context attached.

        Parameters
        ----------
        tensor:
            Batch tensor with the stimulus axis leading.
        ctx:
            Per-invocation context; forwarded verbatim (``None`` legal).

        Returns
        -------
        torch.Tensor
            The wrapped callable's result.
        """

        return self._fn(tensor, ctx)


def with_context(
    fn: Callable[[torch.Tensor, TransformContext | None], torch.Tensor],
) -> ContextTransform:
    """Declare a two-argument callable context-capable (the T-C1 wrapper door).

    Parameters
    ----------
    fn:
        Callable taking ``(tensor, ctx)``.

    Returns
    -------
    ContextTransform
        A wrapper the dispatch layer recognizes by isinstance, never by
        signature inspection.

    Raises
    ------
    TransformContractError
        ``transform_coercion_invalid`` when ``fn`` is not callable.
    """

    if not callable(fn):
        raise TransformContractError(
            f"with_context() needs a callable taking (tensor, ctx); got {type(fn).__name__}.",
            code="transform_coercion_invalid",
            remedy="pass a callable of two arguments: (tensor, ctx)",
            value_type=type(fn).__name__,
        )
    return _WrappedContextTransform(fn)


def wants_context(obj: Any) -> bool:
    """Return whether ``obj`` DECLARED context capability (never sniffed).

    Parameters
    ----------
    obj:
        Candidate transform callable.

    Returns
    -------
    bool
        ``True`` iff ``obj`` is a :class:`ContextTransform`. Everything else
        — including partials with keyword-only ctx parameters and C-implemented
        ops reporting ``(*args, **kwargs)`` — is called unary.
    """

    return isinstance(obj, ContextTransform)
