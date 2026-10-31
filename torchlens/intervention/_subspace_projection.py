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

Each edit verb has one of four meanings here:

* coordinate verbs run on ``c`` itself (``zero_ablate``, ``scale``,
  ``clamp``, ``noise``, ``mean_ablate()``, ``mean_fill`` without ``ref=``,
  ``scramble_elements`` without ``source=``, ``add`` of a scalar,
  ``splice_module(module, input="out")`` and any hook callable, which
  receives the coordinates);
* value verbs compute a value in activation space and the edit takes that
  value's component (``patch_from``, the interchange; ``swap_with``,
  ``replace_with``, ``add`` of a tensor, ``steer``, ``permute_batch``,
  ``resample_rows_from``, ``mean_from``, ``mean_fill(ref)``);
* source verbs map the full site value ``x`` to new coordinates
  ``[..., k]`` (``tl.intervention.affine_source(weight, bias=b)``, so
  ``x + Q^T (W x + b - Q x)``, the LoReFT form);
* every other verb refuses ``subspace_edit_unsupported`` with the spelling
  that works.

Portability: a projected builtin verb is itself a builtin edit named
``subspace_edit`` whose stored arguments are the inner verb and the basis
actually applied (the float64 orthonormal rows, saved as a tensor sidecar),
the bound axis, the rows' sha256, and the basis provenance (``origin``,
``method``, the caller's basis digest and dtype). A saved spec therefore
reloads as the same projection, never as the plain verb over the whole site,
and a malformed, tampered or non-orthonormal stored basis refuses at load.
This is how a rotation learned elsewhere (DAS, ReFT) travels: its first
``k`` rows are the basis. A projected edit whose inner verb is not builtin
(``patch_from``, population verbs, hook callables) stays ``opaque_audit``.
A subspace restricted by a ``within=`` element mask (one position, as LoReFT
acts) stores that mask and its sha256 too, so it also saves portably; any
other masked edit keeps its own portability.

DOCUMENTED-UNSTABLE semantics.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable
from typing import Any

import torch

from .._errors import InvalidArgumentError
from ..selection import Selection, Selector, SiteEntry, _Combinator, _lift, _Mask
from ..selection_subspace import _basis_digest, _SubspaceTerm
from ._affine_source import AFFINE_SOURCE_NAME, SourceCall, check_source_shape
from ._subspace_edits import _orthonormal_rows
from .types import Edit

__all__ = [
    "PROJECTION_NOTE",
    "SubspaceProjection",
    "masked_subspace_edit",
    "subspace_projection_of",
]

#: Builtin helper name of a projected verb in a saved spec.
SUBSPACE_EDIT_NAME = "subspace_edit"

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


def _unsupported(
    verb: str, why: str, remedy: str, *, code: str = "subspace_edit_unsupported"
) -> InvalidArgumentError:
    """Build the contracted refusal for a verb with no coordinate meaning.

    ``code`` is always ``subspace_edit_unsupported``; raise sites spell it so
    the refusal census sees a coded site.
    """

    return InvalidArgumentError(
        f"do(subspace(...), {verb}) has no meaning on the subspace coordinates: {why}",
        code=code,
        remedy=f"{remedy} ({_GUIDE})",
        argument="edit",
        verb=verb,
    )


_NESTED_WHY = (
    "it is itself a projection edit, so nesting it in a subspace selection would project twice"
)
_GENERIC_REFUSAL = (
    "the verb is not one of the coordinate or value verbs listed for subspace edits",
    "pass a hook callable, which receives the coordinates: "
    "fork.do(tl.subspace(site, basis, origin=...), lambda c, *, hook: c * 0)",
)
# verb -> (why it has no coordinate meaning, working spelling); ``{verb}`` is filled in.
_REFUSALS: dict[str, tuple[str, str]] = {
    "set_direction_mean": (
        _NESTED_WHY,
        "set the component to the population mean with "
        "fork.do(tl.subspace(site, basis, origin=...), tl.intervention.mean_from(ref))",
    ),
    **dict.fromkeys(
        sorted(_NESTED_PROJECTIONS - {"set_direction_mean"}),
        (
            _NESTED_WHY,
            "edit the component with a plain verb, e.g. "
            "fork.do(tl.subspace(site, basis, origin=...), tl.zero_ablate()), or apply "
            "tl.intervention.{verb}(...) to the site label without a subspace",
        ),
    ),
    "compose": (
        "a composed edit mixes verbs with different coordinate meanings",
        "apply the edits one at a time: fork.do(selection, first) then fork.do(selection, second)",
    ),
    "at_positions": (
        "positions are chosen by the selection, not by the edit",
        "restrict positions through within=, e.g. tl.subspace(tl.units(site, mask), "
        "basis, origin=...) with mask True on every feature of the chosen positions",
    ),
    **dict.fromkeys(
        ("mean_ablate", "scramble_elements"),
        (
            "source= is an activation-space value whose elements are not coordinates",
            "drop source= (tl.{verb}() acts on the component itself), or write a "
            "source's component with tl.replace_with(source_tensor)",
        ),
    ),
    "splice_module": (
        "input='in' runs the module on the call inputs, which have no coordinates",
        "pass input='out' so the module maps the coordinates [..., k] to [..., k]",
    ),
}


def _refusal_for(verb: str, *, code: str = "subspace_edit_unsupported") -> InvalidArgumentError:
    """Return the refusal (with its working spelling) for an unsupported verb."""

    del code  # spelled at the raise site for the refusal census; always the same code
    why, remedy = _REFUSALS.get(verb, _GENERIC_REFUSAL)
    return _unsupported(verb, why, remedy.format(verb=verb))


# Verbs whose coordinate meaning depends on their arguments: "coordinates",
# "value", or None (refused with the verb's working spelling).
_CONDITIONAL_MEANINGS: dict[str, Callable[[Edit, dict[str, Any]], str | None]] = {
    "add": lambda edit, _kw: "value" if isinstance(edit.args[0], torch.Tensor) else "coordinates",
    "mean_fill": lambda _edit, kw: "value" if "population_identity" in kw else "coordinates",
    "mean_ablate": lambda edit, _kw: "coordinates" if edit.args and edit.args[0] is None else None,
    "scramble_elements": lambda edit, _kw: (
        "coordinates" if edit.args and edit.args[0] is None else None
    ),
    "splice_module": lambda _edit, kw: "coordinates" if kw.get("input") == "out" else None,
}


def _verb_meaning(edit: Any) -> str:
    """Classify an edit as ``"coordinates"``, ``"value"`` or ``"source"``, refusing the rest."""

    if not isinstance(edit, Edit):
        return "coordinates"
    verb = edit.helper_name
    if edit.kind != "forward" or edit.direction == "backward":
        raise _unsupported(
            verb,
            "backward-direction verbs edit gradients, not the forward component",
            "attach the gradient verb at capture time: tl.trace(model, x, "
            "intervene=tl.when(tl.module(name), tl.grad_zero()))",
            code="subspace_edit_unsupported",
        )
    if verb == AFFINE_SOURCE_NAME:
        return "source"
    if verb in _COORDINATE_VERBS or verb in _VALUE_VERBS:
        return "coordinates" if verb in _COORDINATE_VERBS else "value"
    classify = _CONDITIONAL_MEANINGS.get(verb)
    meaning = None if classify is None else classify(edit, dict(edit.kwargs))
    if meaning is None:
        raise _refusal_for(verb, code="subspace_edit_unsupported")
    return meaning


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
    ``dim`` the bound site axis, ``sha256`` the provenance digest of the
    caller's basis, ``orthonormalised`` whether the caller's basis had to be
    orthonormalised; ``origin``, ``method`` and ``source_dtype`` are the
    basis provenance a saved spec keeps.
    """

    rows: torch.Tensor
    dim: int
    sha256: str
    orthonormalised: bool
    origin: str = ""
    method: str | None = None
    source_dtype: str = "float64"

    @classmethod
    def from_term(cls, node: _SubspaceTerm) -> SubspaceProjection:
        """Build the projection for one subspace term (orthonormalising if needed)."""

        # Weights at or below tol= are outside the resolved support, so the
        # edit acts through the basis restricted to that support.
        support = node.basis.abs() > node.provenance.tol
        basis = node.basis * support
        gram = basis @ basis.T
        identity = torch.eye(int(basis.shape[0]), dtype=basis.dtype)
        orthonormal = bool(torch.allclose(gram, identity, rtol=0.0, atol=_ORTHONORMAL_TOL))
        rows = basis
        if not orthonormal:
            rows = _orthonormal_rows(basis, "subspace")
            # Sign-preserving Gram-Schmidt: row j keeps a positive inner product
            # with the caller's row j, so a one-direction coordinate keeps its sign.
            signs = torch.sign(torch.sum(rows * basis, dim=1, keepdim=True))
            rows = rows * signs * support.any(dim=0)
        provenance = node.provenance
        return cls(
            rows,
            provenance.dim,
            provenance.sha256,
            not orthonormal,
            origin=provenance.origin,
            method=provenance.method,
            source_dtype=provenance.source_dtype,
        )

    @property
    def note(self) -> str:
        """Return the audit-source marker for this projection."""

        return PROJECTION_NOTE + (_ORTHONORMALISED_NOTE if self.orthonormalised else "")

    def check_population(self, entry: SiteEntry, site_label: str) -> None:
        """Refuse a population that keeps only part of the support along the axis."""

        dense = entry._mask._dense_ro()
        if self.partial_along_support(dense):
            axis = self.dim % dense.ndim
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

    def partial_along_support(self, dense: torch.Tensor) -> bool:
        """Return whether a mask keeps only part of the basis support at some position."""

        axis = self.dim % dense.ndim
        support = (self.rows.abs() > 0).any(dim=0)
        along = torch.movedim(dense, axis, -1)[..., support]
        return bool((along.any(dim=-1) & ~along.all(dim=-1)).any())

    def _inner_factory(self, edit: Edit, meaning: str) -> Callable[[], Any]:
        """Return the verb's hook factory (a source verb's weight must match these rows)."""

        if meaning == "source":
            check_source_shape(edit, self.rows)
        if edit.factory is None:
            raise _refusal_for(edit.helper_name, code="subspace_edit_unsupported")
        return edit.factory

    def derive(self, edit: Any, entry: SiteEntry, site_label: str) -> Any:
        """Return ``edit`` rewritten to act on this site's subspace component."""

        self.check_population(entry, site_label)
        meaning = _verb_meaning(edit)
        verb = edit.helper_name if isinstance(edit, Edit) else "hook callable"
        if isinstance(edit, Edit):
            inner_factory: Callable[[], Any] = self._inner_factory(edit, meaning)
        else:

            def inner_factory() -> Any:
                """Adapt a bare hook callable to the factory protocol."""

                return edit

        factory = self._factory(inner_factory, meaning, verb)
        disclosure = self._disclosure(meaning)
        if isinstance(edit, Edit) and edit.portability == "builtin":
            return self._portable_edit(edit, factory, disclosure)
        if isinstance(edit, Edit):
            return dataclasses.replace(
                edit,
                factory=factory,
                # A non-builtin verb is session-time: its persisted form must
                # never rebuild as the plain verb over the whole site.
                portability="opaque_audit",
                metadata=tuple(edit.metadata) + disclosure,
            )
        return factory()

    def wrap(self, edit: Edit) -> Edit:
        """Return the builtin ``subspace_edit`` for a builtin verb (the reload path)."""

        meaning = _verb_meaning(edit)
        factory = self._factory(self._inner_factory(edit, meaning), meaning, edit.helper_name)
        return self._portable_edit(edit, factory, self._disclosure(meaning))

    def _disclosure(self, meaning: str) -> tuple[tuple[str, Any], ...]:
        """Return the metadata a projected edit carries about its basis."""

        return (
            ("subspace_projection_sha256", self.sha256),
            ("subspace_edit_meaning", meaning),
        )

    def stored_kwargs(self) -> tuple[tuple[str, Any], ...]:
        """Return the constructor keywords a saved ``subspace_edit`` carries."""

        return (
            ("basis", self.rows),
            ("axis", self.dim),
            ("basis_sha256", _basis_digest(self.rows)),
            ("origin", self.origin),
            ("method", self.method),
            ("origin_sha256", self.sha256),
            ("source_dtype", self.source_dtype),
            ("orthonormalised", self.orthonormalised),
        )

    def _portable_edit(
        self, edit: Edit, factory: Any, disclosure: tuple[tuple[str, Any], ...]
    ) -> Edit:
        """Return the builtin ``subspace_edit`` that saves and reloads as this projection."""

        # replace() keeps the verb's direction and batch flags on the wrapper.
        return dataclasses.replace(
            edit,
            helper_name=SUBSPACE_EDIT_NAME,
            args=(edit,),
            kwargs=self.stored_kwargs(),
            factory=factory,
            portability="builtin",
            metadata=tuple(edit.metadata) + disclosure,
        )

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
                elif meaning == "source":
                    source = inner(torch.movedim(out, axis, -1), hook=SourceCall(hook))
                    moved = torch.movedim(source, -1, axis)
                    delta = _checked(moved, coords, verb, "coordinates") - coords
                else:
                    value = _checked(inner(out, hook=hook), out, verb, "site value")
                    delta = _move_coords(value, rows, axis) - coords
                return out + _span_vector(delta, rows, axis)

            return _hook

        return factory


def _mask_digest(mask: torch.Tensor) -> str:
    """Return the sha256 of a bool element mask (shape header plus bytes)."""

    digest = hashlib.sha256()
    digest.update(repr(tuple(mask.shape)).encode())
    digest.update(mask.to(torch.uint8).contiguous().numpy().tobytes())
    return digest.hexdigest()


def masked_subspace_edit(edit: Edit, mask: _Mask, site_label: str) -> Edit:
    """Return a builtin ``subspace_edit`` restricted to ``mask``, still portable.

    The stored keywords gain ``mask`` (the dense bool element mask) and
    ``mask_sha256``, and the hook scatters only the masked elements, exactly as
    the session-time masked edit does, so a reload applies bit-identically.
    """

    from .masked_edit import _masked_factory

    if edit.factory is None:
        raise _refusal_for(edit.helper_name, code="subspace_edit_unsupported")
    dense = mask.to_dense().cpu().contiguous()
    stored = dict(edit.kwargs)
    if isinstance(stored.get("mask"), torch.Tensor):
        # A reloaded masked edit under another mask: the session scatters through
        # both, which is the intersection, so that is the one mask stored.
        if tuple(stored["mask"].shape) != tuple(dense.shape):
            raise _refusal_for(edit.helper_name, code="subspace_edit_unsupported")
        dense = dense & stored["mask"]
    kwargs = tuple(item for item in edit.kwargs if item[0] not in ("mask", "mask_sha256"))
    return dataclasses.replace(
        edit,
        kwargs=kwargs + (("mask", dense), ("mask_sha256", _mask_digest(dense))),
        factory=_masked_factory(edit.factory, mask, site_label),
    )


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
    if isinstance(lifted, Selector):
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
    if isinstance(lifted, Selection) and any(
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
