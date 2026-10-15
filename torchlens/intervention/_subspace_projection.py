"""Projection-valued subspace edits: ``do(subspace(...), edit)`` edits the component.

A :func:`torchlens.subspace` selection names a basis ``Q`` (``k`` rows over
one feature axis of the site). Under ``do()`` the edit acts on the
coordinates ``c = Q x`` (one coordinate per basis row) and writes back
``x + Q^T (edit(c) - c)``: the component inside the span changes and the
orthogonal complement is untouched. ``resolve()`` keeps reporting the
basis's support set, which is where the edit can land.

Basis: weights at or below the producer's ``tol=`` are dropped (they are
outside the resolved support). An orthonormal basis (rows orthonormal to
``1e-10``) is then used exactly as given, so the coordinates are the
caller's own. Any other full-rank basis is
orthonormalised by order- and sign-preserving Gram-Schmidt (reduced QR with a
positive diagonal), so for one direction ``v`` the coordinate is the
component along ``v / |v|``; the audit record says so. A rank-deficient basis
refuses ``subspace_basis_invalid``.

Each edit verb has one of three meanings here:

* coordinate verbs run on ``c`` itself (``zero_ablate``, ``scale``,
  ``clamp``, ``noise``, ``mean_ablate()``, ``mean_fill`` without ``ref=``,
  ``scramble_elements`` without ``source=``, ``add`` of a scalar,
  ``splice_module(module, input="out")`` and any hook callable, which
  receives the coordinates);
* value verbs compute a value in activation space and the edit takes that
  value's component (``patch_from``, the interchange; ``swap_with``,
  ``replace_with``, ``add`` of a tensor, ``steer``, ``permute_batch``,
  ``resample_rows_from``, ``mean_from``, ``mean_fill(ref)``);
* every other verb refuses ``subspace_edit_unsupported`` with the spelling
  that works.

DOCUMENTED-UNSTABLE semantics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from .._errors import InvalidArgumentError
from ..selection import ResolvedSelection, Selection, SiteEntry, _Combinator, _lift
from ..selection_subspace import _SubspaceTerm
from ._subspace_edits import _orthonormal_rows
from .types import Edit

__all__ = ["PROJECTION_NOTE", "SubspaceProjection", "subspace_projection_of"]

#: Marker stamped into every projected site's audit ``source``: the do()
#: audit record states which meaning ran.
PROJECTION_NOTE = (
    " [projection: do() edits the component along the basis span; the "
    "orthogonal complement is unchanged]"
)
_ORTHONORMALISED_NOTE = " [basis orthonormalised: coordinates are along the Gram-Schmidt rows]"
_ORTHONORMAL_TOL = 1e-10
_GUIDE = "see the subspace section of docs/intervention_explainers.md"

_COORDINATE_VERBS = frozenset({"zero_ablate", "scale", "clamp", "noise"})
_VALUE_VERBS = frozenset(
    {
        "patch_from",
        "swap_with",
        "replace_with",
        "steer",
        "permute_batch",
        "resample_rows_from",
        "mean_from",
    }
)
_NESTED_PROJECTIONS = frozenset(
    {"project_onto", "project_off", "set_direction_mean", "subspace_ablate", "subspace_interchange"}
)


def _unsupported(verb: str, why: str, remedy: str) -> InvalidArgumentError:
    """Build the contracted refusal for a verb with no coordinate meaning."""

    return InvalidArgumentError(
        f"do(subspace(...), {verb}) has no meaning on the subspace coordinates: {why}",
        code="subspace_edit_unsupported",
        remedy=f"{remedy} ({_GUIDE})",
        argument="edit",
        verb=verb,
    )


def _refusal_for(verb: str) -> InvalidArgumentError:
    """Return the refusal (with its working spelling) for an unsupported verb."""

    if verb == "set_direction_mean":
        return _unsupported(
            verb,
            "it is itself a projection edit, so nesting it in a subspace selection "
            "would project twice",
            "set the component to the population mean with "
            "fork.do(tl.subspace(site, basis, origin=...), tl.intervention.mean_from(ref))",
        )
    if verb in _NESTED_PROJECTIONS:
        return _unsupported(
            verb,
            "it is itself a projection edit, so nesting it in a subspace selection "
            "would project twice",
            "edit the component with a plain verb, e.g. "
            "fork.do(tl.subspace(site, basis, origin=...), tl.zero_ablate()), or apply "
            f"tl.intervention.{verb}(...) to the site label without a subspace",
        )
    if verb == "compose":
        return _unsupported(
            verb,
            "a composed edit mixes verbs with different coordinate meanings",
            "apply the edits one at a time: fork.do(selection, first) then "
            "fork.do(selection, second)",
        )
    if verb == "at_positions":
        return _unsupported(
            verb,
            "positions are chosen by the selection, not by the edit",
            "restrict positions through within=, e.g. tl.subspace(tl.units(site, mask), "
            "basis, origin=...) with mask True on every feature of the chosen positions",
        )
    if verb in {"mean_ablate", "scramble_elements"}:
        return _unsupported(
            verb,
            "source= is an activation-space value whose elements are not coordinates",
            f"drop source= (tl.{verb}() acts on the component itself), or write a "
            "source's component with tl.replace_with(source_tensor)",
        )
    if verb == "splice_module":
        return _unsupported(
            verb,
            "input='in' runs the module on the call inputs, which have no coordinates",
            "pass input='out' so the module maps the coordinates [..., k] to [..., k]",
        )
    return _unsupported(
        verb,
        "the verb is not one of the coordinate or value verbs listed for subspace edits",
        "pass a hook callable, which receives the coordinates: "
        "fork.do(tl.subspace(site, basis, origin=...), lambda c, *, hook: c * 0)",
    )


def _verb_meaning(edit: Any) -> str:
    """Classify an edit as ``"coordinates"`` or ``"value"``, refusing the rest."""

    if not isinstance(edit, Edit):
        return "coordinates"
    verb = edit.helper_name
    kwargs = dict(edit.kwargs)
    if edit.kind != "forward" or edit.direction == "backward":
        raise _unsupported(
            verb,
            "backward-direction verbs edit gradients, not the forward component",
            "attach the gradient verb at capture time: tl.trace(model, x, "
            "intervene=tl.when(tl.module(name), tl.grad_zero()))",
        )
    if verb in _COORDINATE_VERBS:
        return "coordinates"
    if verb in _VALUE_VERBS:
        return "value"
    if verb == "add":
        return "value" if isinstance(edit.args[0], torch.Tensor) else "coordinates"
    if verb == "mean_fill":
        return "value" if "population_identity" in kwargs else "coordinates"
    if verb in {"mean_ablate", "scramble_elements"} and edit.args and edit.args[0] is None:
        return "coordinates"
    if verb == "splice_module" and kwargs.get("input") == "out":
        return "coordinates"
    raise _refusal_for(verb)


def _move_coords(value: torch.Tensor, rows: torch.Tensor, axis: int) -> torch.Tensor:
    """Return ``value``'s coordinates on ``rows`` with ``axis`` resized to ``k``."""

    moved = torch.movedim(value, axis, -1)
    coords = moved @ rows.to(device=value.device, dtype=value.dtype).T
    return torch.movedim(coords, -1, axis)


def _span_vector(coords: torch.Tensor, rows: torch.Tensor, axis: int) -> torch.Tensor:
    """Map coordinates back to activation space: ``sum_j coords_j * rows_j``."""

    moved = torch.movedim(coords, axis, -1)
    vector = moved @ rows.to(device=coords.device, dtype=coords.dtype)
    return torch.movedim(vector, -1, axis)


def _checked(result: Any, expected: torch.Tensor, verb: str, what: str) -> torch.Tensor:
    """Refuse an edit result whose shape or dtype differs from its input."""

    if not isinstance(result, torch.Tensor) or tuple(result.shape) != tuple(expected.shape):
        shape = tuple(result.shape) if isinstance(result, torch.Tensor) else type(result).__name__
        raise InvalidArgumentError(
            f"subspace edit {verb!r} returned {shape!r} for {what} of shape "
            f"{tuple(expected.shape)!r}; a subspace edit keeps the shape it receives",
            code="subspace_edit_unsupported",
            remedy=f"return a tensor shaped like the {what} ({_GUIDE})",
            argument="edit",
            verb=verb,
        )
    return result.to(dtype=expected.dtype)


@dataclasses.dataclass(frozen=True)
class SubspaceProjection:
    """The basis a projection-valued ``do()`` edit acts through (one per target).

    ``rows`` is the orthonormal float64 ``[k, d]`` basis actually used,
    ``dim`` the bound site axis, ``orthonormalised`` whether the caller's
    basis had to be orthonormalised.
    """

    rows: torch.Tensor
    dim: int
    sha256: str
    orthonormalised: bool

    @classmethod
    def from_term(cls, node: _SubspaceTerm) -> SubspaceProjection:
        """Build the projection for one subspace term (orthonormalising if needed)."""

        # Weights at or below tol= are outside the resolved support, so the
        # edit acts through the basis restricted to that support.
        support = node.basis.abs() > node.provenance.tol
        basis = node.basis * support
        gram = basis @ basis.T
        identity = torch.eye(int(basis.shape[0]), dtype=basis.dtype)
        if bool(torch.allclose(gram, identity, rtol=0.0, atol=_ORTHONORMAL_TOL)):
            return cls(basis, node.provenance.dim, node.provenance.sha256, False)
        rows = _orthonormal_rows(basis, "subspace")
        # Sign-preserving Gram-Schmidt: row j keeps a positive inner product
        # with the caller's row j, so a one-direction coordinate keeps its sign.
        signs = torch.sign(torch.sum(rows * basis, dim=1, keepdim=True))
        rows = rows * signs * support.any(dim=0)
        return cls(rows, node.provenance.dim, node.provenance.sha256, True)

    @property
    def note(self) -> str:
        """Return the audit-source marker for this projection."""

        return PROJECTION_NOTE + (_ORTHONORMALISED_NOTE if self.orthonormalised else "")

    def check_population(self, entry: SiteEntry, site_label: str) -> None:
        """Refuse a population that keeps only part of the support along the axis."""

        dense = entry._mask._dense_ro()
        axis = self.dim % dense.ndim
        support = (self.rows.abs() > 0).any(dim=0)
        along = torch.movedim(dense, axis, -1)[..., support]
        if bool((along.any(dim=-1) & ~along.all(dim=-1)).any()):
            raise InvalidArgumentError(
                f"the subspace population at {site_label!r} selects only part of the basis "
                f"support along axis {axis}; editing a component needs every supported "
                "feature of each chosen position",
                code="subspace_projection_partial_axis",
                remedy=(
                    "restrict positions with a within= mask that is True on every feature "
                    f"of each chosen position, e.g. tl.units(site, mask) ({_GUIDE})"
                ),
                argument="within",
                site=site_label,
            )

    def derive(self, edit: Any, entry: SiteEntry, site_label: str) -> Any:
        """Return ``edit`` rewritten to act on this site's subspace component."""

        self.check_population(entry, site_label)
        meaning = _verb_meaning(edit)
        verb = edit.helper_name if isinstance(edit, Edit) else "hook callable"
        if isinstance(edit, Edit):
            if edit.factory is None:
                raise _refusal_for(verb)
            inner_factory: Callable[[], Any] = edit.factory
        else:

            def inner_factory() -> Any:
                """Adapt a bare hook callable to the factory protocol."""

                return edit

        factory = self._factory(inner_factory, meaning, verb)
        disclosure = (
            ("subspace_projection_sha256", self.sha256),
            ("subspace_edit_meaning", meaning),
        )
        if isinstance(edit, Edit):
            return dataclasses.replace(
                edit,
                factory=factory,
                # The rewritten hook is session-time: its persisted form must
                # never rebuild as the plain verb over the whole site.
                portability="opaque_audit",
                metadata=tuple(edit.metadata) + disclosure,
            )
        return factory()

    def _factory(self, inner_factory: Callable[[], Any], meaning: str, verb: str) -> Any:
        """Wrap one inner hook factory with the coordinate write-back."""

        rows, dim = self.rows, self.dim

        def factory() -> Callable[..., torch.Tensor]:
            """Instantiate the inner hook and wrap it with the projection."""

            inner = inner_factory()

            def _hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
                """Edit the in-span component of ``out``; keep its complement."""

                axis = dim % out.ndim
                coords = _move_coords(out, rows, axis)
                if meaning == "coordinates":
                    edited = _checked(inner(coords, hook=hook), coords, verb, "coordinates")
                    delta = edited - coords
                else:
                    value = _checked(inner(out, hook=hook), out, verb, "site value")
                    delta = _move_coords(value, rows, axis) - coords
                return out + _span_vector(delta, rows, axis)

            return _hook

        return factory


def _contains_subspace(node: Any) -> bool:
    """Return whether a selection AST holds a subspace term below its root."""

    if isinstance(node, _SubspaceTerm):
        return True
    if isinstance(node, _Combinator):
        return any(_contains_subspace(operand) for operand in node.operands)
    return False


def subspace_projection_of(selection_like: Any) -> SubspaceProjection | None:
    """Return the projection a ``do()`` target asks for, or ``None`` for a set target.

    A bare ``tl.subspace(...)`` query is projection-valued. A subspace inside
    set algebra (``| & - ~``) or already resolved has no single basis to edit
    through and refuses ``subspace_projection_composed``.
    """

    lifted = _lift(selection_like)
    if isinstance(lifted, Selection):
        if isinstance(lifted._node, _SubspaceTerm):
            return SubspaceProjection.from_term(lifted._node)
        if _contains_subspace(lifted._node):
            raise InvalidArgumentError(
                "do() received a subspace selection combined with set operators; a "
                "subspace edit acts on the component along one basis, which set "
                "algebra does not define",
                code="subspace_projection_composed",
                remedy=(
                    "pass the subspace on its own and restrict it with within=, e.g. "
                    f"tl.subspace(tl.units(site, mask), basis, origin=...) ({_GUIDE})"
                ),
                argument="target",
            )
    if isinstance(lifted, ResolvedSelection) and any(
        entry.provenance.source.startswith("subspace(origin=") for entry in lifted
    ):
        raise InvalidArgumentError(
            "do() received a RESOLVED subspace selection; resolving keeps the support "
            "set but drops the basis the edit acts through",
            code="subspace_projection_composed",
            remedy=(
                "pass the unresolved tl.subspace(...) query to do(); to edit the support "
                "set itself, pass tl.units(site, resolved[0].mask) "
                f"({_GUIDE})"
            ),
            argument="target",
        )
    return None
