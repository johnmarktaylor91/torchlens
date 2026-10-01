"""``tl.compose``: the explicit value algebra for chained edits (edits memo D20-D22).

One composed edit chains VALUES at one site: ``compose(a, b, c)(x) == c(b(a(x)))``
(left-to-right, matching the hook plan's declared ``composition="left_to_right"``
and the order sequential edits already apply in). The panel named and
constrained semantics that already existed -- and the spec door's
``spec_rules_overlap`` refusal points here as the explicit spelling for two
edits at one site.

The D20 laws implemented:

- FLATTENED at construction: associativity is syntactic AND audit-visible
  (``compose(a, compose(b, c))`` and ``compose(a, b, c)`` are one identity).
- Zero leaves refuse; non-commutative and documented so; kind-homogeneous
  (never a forward/backward mix); no fusion in v1.
- ONE fire, ONE final scatter, ONE FireRecord, with the ordered leaf names on
  the helper identity and per-leaf failures naming ``compose[i]``.
- Flags AND-combine (``batch_independent``/``compatible_with_append``);
  portability is the WEAKEST leaf; a leaf failure aborts the whole fire
  (atomic -- the engine never sees a half-applied chain).

The D21 refusals: mixed forward/backward kinds; ANY unseeded stochastic leaf
(the honest-draw law admits no unacknowledged RNG into a chain); ANY
shape-changing leaf in v1 (the conditional next-leaf-declares-contract rule is
the documented relaxation -- operationally identical today, since no
HelperSpec declares an input contract).

The D22 three-layer boundary: compose chains VALUES at one site;
selection-batch ``do([(sel, edit), ...])`` carries MANY rules in ONE
transaction; the immutable ``InterventionSpec`` carries many rules as the
DURABLE artifact. Cross-site compose is refused by construction (a composed
edit is one edit at whatever site it is attached to).

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .._errors import InvalidArgumentError
from .errors import HookValueError
from .hooks import HookContext
from .types import HelperPortability, HelperSpec

__all__ = ["compose"]

#: Portability lattice: the composed spec takes the WEAKEST leaf (D20).
_PORTABILITY_RANK: dict[str, int] = {"builtin": 0, "import_ref": 1, "opaque_audit": 2}


def _flatten(edits: tuple[Any, ...]) -> list[HelperSpec]:
    """Flatten nested compose specs into the ordered leaf list (D20)."""

    leaves: list[HelperSpec] = []
    for edit in edits:
        if isinstance(edit, HelperSpec) and edit.helper_name == "compose":
            leaves.extend(spec for spec in edit.args if isinstance(spec, HelperSpec))
        else:
            leaves.append(edit)
    return leaves


def _validate_leaves(leaves: list[HelperSpec]) -> None:
    """Apply the D20/D21 construction refusals over the flattened leaves."""

    if not leaves:
        raise InvalidArgumentError(
            "compose() needs at least one edit leaf",
            code="compose_empty",
            remedy="pass the edits to chain, left to right: compose(a, b, ...)",
            argument="edits",
        )
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, HelperSpec):
            raise InvalidArgumentError(
                f"compose leaf {index} is {type(leaf).__name__}; v1 composes "
                "declared helper specs only (a bare callable's kind, flags, and "
                "seed posture are undeclarable)",
                code="compose_leaf_invalid",
                remedy="wrap the transform as a helper (or attach the callable "
                "separately through attach_hooks)",
                argument="edits",
            )
        if leaf.factory is None:
            raise InvalidArgumentError(
                f"compose[{index}] ({leaf.helper_name!r}) carries no runtime "
                "factory (an audit-only spec cannot fire)",
                code="compose_leaf_invalid",
                remedy="compose executable helper specs only",
                argument="edits",
            )
        kwargs = dict(leaf.kwargs)
        if "seed" in kwargs and kwargs["seed"] is None:
            raise InvalidArgumentError(
                f"compose[{index}] ({leaf.helper_name!r}) is stochastic and "
                "UNSEEDED; the honest-draw law admits no unacknowledged RNG "
                "into a chain (an unseeded leaf is silently re-rolled by later "
                "unrelated edits)",
                code="compose_unseeded_stochastic",
                remedy=f"pass an explicit seed to {leaf.helper_name} (or "
                "seed='auto' on the derived-seed family)",
                argument="edits",
            )
        if kwargs.get("force_shape_change"):
            raise InvalidArgumentError(
                f"compose[{index}] ({leaf.helper_name!r}) declares a shape "
                "change; v1 refuses ANY shape-changing leaf (the "
                "next-leaf-declares-contract rule is the documented relaxation)",
                code="compose_shape_change_unsupported",
                remedy="drop force_shape_change= inside a chain, or apply the "
                "shape-changing edit alone",
                argument="edits",
            )
    kinds = {leaf.kind for leaf in leaves}
    if kinds != {"forward"}:
        names = ", ".join(
            f"compose[{i}]={leaf.helper_name}:{leaf.kind}" for i, leaf in enumerate(leaves)
        )
        raise InvalidArgumentError(
            f"compose chains forward VALUE edits only in v1 ({names}); backward "
            "hook signatures vary by grad kind, so a backward chain would fire "
            "wrong call shapes silently -- the exact defect class this family "
            "exists to kill",
            code="compose_kind_mismatch",
            remedy="chain forward helpers only; attach backward helpers as "
            "separate rules (tl.when per site)",
            argument="edits",
        )


def compose(*edits: Any) -> HelperSpec:
    """Chain edits left-to-right into ONE composed edit at one site.

    ``compose(a, b, c)(x) == c(b(a(x)))``. Non-commutative by design
    (``compose(scale, clamp) != compose(clamp, scale)``); granularity mixing
    is deliberately LEGAL (``compose(permute_batch, noise)`` is a standard
    degradation control -- each leaf is a total tensor-to-tensor function,
    and a tool that second-guesses legal math teaches users to distrust it).

    DOCUMENTED-UNSTABLE spelling pending naming-session ratification.

    Parameters
    ----------
    *edits:
        Helper specs to chain, applied left to right. Nested ``compose``
        results flatten at construction (associativity is syntactic).

    Returns
    -------
    HelperSpec
        One composed helper: one fire, one scatter, one FireRecord, with the
        ordered leaf identities on the spec (``args``) and the AND-combined
        flags.
    """

    leaves = _flatten(tuple(edits))
    _validate_leaves(leaves)

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime chain hook (leaf hooks instantiated in order)."""

        leaf_hooks = [leaf.factory() for leaf in leaves if leaf.factory is not None]

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Apply every leaf in order; a leaf failure aborts the whole fire."""

            current = out
            for index, leaf_hook in enumerate(leaf_hooks):
                # Thread the composed-leaf path for the derived-seed law (D6):
                # two stochastic leaves inside one compose cannot collide.
                hook.ctx["compose_leaf_path"] = (index,)
                try:
                    current = leaf_hook(current, hook=hook)
                except Exception as exc:
                    if isinstance(exc, HookValueError):
                        raise
                    raise HookValueError(
                        f"compose[{index}] ({leaves[index].helper_name!r}) failed: "
                        f"{exc}. Remedy: fix the named leaf; a leaf failure aborts "
                        "the whole composed fire (atomic, never half-applied)",
                        code="compose_leaf_failed",
                        leaf_index=index,
                        leaf_name=leaves[index].helper_name,
                    ) from exc
                if not isinstance(current, torch.Tensor):
                    raise HookValueError(
                        f"compose[{index}] ({leaves[index].helper_name!r}) "
                        f"returned {type(current).__name__}; every leaf must "
                        "return a tensor for the next leaf to consume. Remedy: "
                        "make the named leaf return a tensor",
                        code="compose_leaf_failed",
                        leaf_index=index,
                        leaf_name=leaves[index].helper_name,
                    )
            hook.ctx.pop("compose_leaf_path", None)
            return current

        return _hook

    portability: HelperPortability = "builtin"
    for leaf in leaves:
        if _PORTABILITY_RANK[leaf.portability] > _PORTABILITY_RANK[portability]:
            portability = leaf.portability

    from .helpers import _helper_spec

    return _helper_spec(
        "compose",
        args=tuple(leaves),
        factory=factory,
        portability=portability,
        batch_independent=all(leaf.batch_independent for leaf in leaves),
        compatible_with_append=all(leaf.compatible_with_append for leaf in leaves),
        metadata={
            "compose_leaves": "|".join(leaf.helper_name for leaf in leaves),
            "compose_leaf_count": len(leaves),
            # Row-coherent leaves keep their mask-equivariance obligation
            # through the chain (the D23 check reads these facts).
            **(
                {
                    "row_coherent": True,
                    "batch_axis": next(
                        dict(leaf.metadata).get("batch_axis")
                        for leaf in leaves
                        if dict(leaf.metadata).get("row_coherent")
                    ),
                }
                if any(dict(leaf.metadata).get("row_coherent") for leaf in leaves)
                else {}
            ),
            **(
                {"batch_coherent": True}
                if any(dict(leaf.metadata).get("batch_coherent") for leaf in leaves)
                else {}
            ),
        },
    )
