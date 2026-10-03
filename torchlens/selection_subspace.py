"""Subspace selection producer: directions in activation space (L6 producer wave).

Selects the elements a DIRECTION (or small subspace) in activation space
lives on — the spelling for probe directions, steering vectors, PCA
components, and SAE feature directions. One producer,
:func:`subspace`, returning a :class:`~torchlens.selection.Selection` query
that composes with the full ``| & - ~`` algebra and resolves explicitly
against one trace.

SET, NOT PROJECTION (the normative boundary): a Selection denotes a
(touched-site family, selected-element set) pair, masks are exact AS SETS,
and ``do()`` applies edits elementwise — so this producer resolves to the
basis's SUPPORT SET: the elements whose coordinate carries weight
``|w| > tol`` in at least one basis vector. The WEIGHTS themselves are
deliberately NOT carried on the selection (a weighted mask would be a new
denotation level: boolean operators are set algebra while subspaces compose
linearly, the provenance lattice speaks subset relations, and the do()
engine's edit-then-scatter contract is elementwise substitution, not a
linear operator). A dense direction therefore resolves to the WHOLE feature
axis — ``do(subspace(...), zero_ablate())`` ablates every supported
element, NEVER "just the direction's component". Projection-valued
selections/edits are a named design fork, not a promise.

BASIS PROVENANCE IS MANDATORY: a direction without a recorded origin is an
unreproducible result. ``origin=`` is a REQUIRED non-empty description of
where the basis came from (a trained probe, a PCA fit, an SAE decoder row,
a hand-specified vector); the producer canonicalizes the basis to float64
and stamps origin, optional ``method=``, geometry, and the basis's sha256
content digest into every resolved entry's ``provenance.source`` — so the
record rides ``do()`` audit records and any downstream disclosure. The
frozen :class:`BasisProvenance` record is the programmatic face.

DIMENSION HONESTY: the basis indexes ONE named axis of each population
site's output space (``dim=``, default ``-1`` — the trailing feature axis
convention; channel directions on conv outputs are ``dim=1``). A site whose
extent along that axis differs from the basis dimension refuses typed
(``selection_unresolvable`` / ``basis_dim_mismatch``) — a 768-dim direction
is never silently broadcast, truncated, or padded onto a 512-wide layer.

Support is a fact about the basis and the site GEOMETRY, never the values:
resolution reads no payloads, so unsaved sites resolve fine (contrast the
value producers). Masks are exact as sets about the stated criterion
(``provenance.relation="exact"``); population restriction composes through
the normative JOIN table.

Every spelling here ships DOCUMENTED-UNSTABLE pending naming-session
ratification (provisional-name protocol). ACT-kind only:
PARAM/EDGE populations refuse ``selection_kind_incompatible`` (a parameter-
space direction — e.g. a task vector — is a named possibility, not a
promise).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch

from .selection import (
    ResolvedSelection,
    Selection,
    SiteEntry,
    _unresolvable,
    register_term_resolver,
)
from .selection_values import (
    _entry_with_mask,
    _lift_within,
    _resolve_population,
    _validate_real_number,
)

__all__ = [
    "BasisProvenance",
    "subspace",
]


@dataclass(frozen=True)
class BasisProvenance:
    """Frozen reproducibility record for one subspace basis.

    ``origin`` is the user-declared account of where the basis came from
    (required, non-empty — the honest half of a direction). ``method`` is an
    optional free-form family tag (``'probe'`` / ``'pca'`` / ``'sae'`` /
    ``'manual'`` ...; deliberately not a closed vocabulary). ``sha256`` is
    the content digest of the canonical float64 basis bytes plus its shape
    header, so an identical claim can be re-checked bit-exactly.
    ``n_vectors`` x ``space_dim`` is the canonical basis geometry; ``dim``
    and ``tol`` are the axis binding and support threshold the producer was
    built with.
    """

    origin: str
    method: str | None
    sha256: str
    n_vectors: int
    space_dim: int
    dim: int
    tol: float

    def summary(self) -> str:
        """Return the compact one-line disclosure used in provenance sources."""

        method_part = "" if self.method is None else f", method={self.method!r}"
        return (
            f"subspace(origin={self.origin!r}{method_part}, k={self.n_vectors}, "
            f"d={self.space_dim}, dim={self.dim}, tol={self.tol}, "
            f"basis_sha256={self.sha256})"
        )


@dataclass(frozen=True)
class _SubspaceTerm:
    """AST leaf for the subspace producer (canonical basis + provenance)."""

    within: Any
    basis: torch.Tensor = field(compare=False)  # canonical float64 [k, d], never handed out
    provenance: BasisProvenance = field(compare=False)

    def __repr__(self) -> str:
        """Return the compact constructor-shaped disclosure (never the tensor)."""

        return f"{self.provenance.summary()[:-1]}, within={self.within!r})"


def _canonicalize_basis(basis: Any) -> torch.Tensor:
    """Validate and canonicalize the basis to a float64 ``[k, d]`` tensor."""

    if not isinstance(basis, torch.Tensor):
        raise ValueError(
            f"subspace `basis` must be a torch.Tensor of shape [d] or [k, d]; "
            f"got {type(basis).__name__}."
        )
    if basis.is_complex() or basis.dtype == torch.bool:
        raise ValueError(
            f"subspace `basis` must have a real numeric dtype (a direction in "
            f"activation space); got {basis.dtype}. Complex and boolean bases "
            "are refused rather than reinterpreted."
        )
    if basis.ndim == 1:
        basis = basis.unsqueeze(0)
    if basis.ndim != 2:
        raise ValueError(
            f"subspace `basis` must be one direction [d] or a stack of "
            f"directions [k, d]; got a {basis.ndim}-d tensor of shape "
            f"{tuple(basis.shape)!r}."
        )
    if basis.shape[0] < 1 or basis.shape[1] < 1:
        raise ValueError(f"subspace `basis` must be non-empty; got shape {tuple(basis.shape)!r}.")
    canonical = basis.detach().to(torch.float64).cpu().contiguous().clone()
    if not torch.isfinite(canonical).all():
        raise ValueError(
            "subspace `basis` contains NaN/Inf weights: a direction with "
            "non-finite coordinates claims nothing. Clean the basis before "
            "minting a selection from it."
        )
    return canonical


def _basis_digest(canonical: torch.Tensor) -> str:
    """Return the sha256 content digest of the canonical basis (shape-bound)."""

    digest = hashlib.sha256()
    digest.update(repr(tuple(canonical.shape)).encode())
    digest.update(canonical.numpy().tobytes())
    return digest.hexdigest()


#: Provenance-source marker for a basis whose support is the ENTIRE bound
#: axis. It rides every resolved entry's ``provenance.source`` (and therefore
#: every ``do()`` audit record), and ``do()`` reads it to fire the
#: point-of-use disclosure: editing a dense direction's support set is
#: FULL-AXIS ablation under the documented set semantics, never a projection
#: along the direction -- the wrong-intent case the walkthrough flagged.
DENSE_SUPPORT_NOTE = (
    " [dense direction: support is the ENTIRE axis -- do() edits every "
    "element of the axis (set semantics), never the component along the "
    "direction]"
)


def _support_vector(canonical: torch.Tensor, tol: float) -> torch.Tensor:
    """Return the union support over the basis rows (bool ``[d]``)."""

    return (canonical.abs() > tol).any(dim=0)


def _resolve_subspace_term(node: _SubspaceTerm, trace: Any) -> ResolvedSelection:
    """Resolve one subspace term to its support set over the population.

    Pure geometry: no payload is read, so unsaved sites resolve. Every
    population site must carry the basis dimension on the bound axis — a
    mismatch refuses ``basis_dim_mismatch``, never broadcasts or truncates.
    """

    record = node.provenance
    population = _resolve_population(node.within, trace)
    support = _support_vector(node.basis, record.tol)
    source = record.summary()
    if support.numel() > 1 and bool(support.all()):
        source += DENSE_SUPPORT_NOTE
    entries: list[SiteEntry] = []
    for entry in population:
        shape = entry.shape
        axis = record.dim if record.dim >= 0 else len(shape) + record.dim
        if axis < 0 or axis >= len(shape):
            raise _unresolvable(
                "basis_dim_mismatch",
                f"subspace binds basis axis dim={record.dim}, but site "
                f"{entry.site_key!r} has only {len(shape)} output axes "
                f"(shape {shape!r}). Name an axis of the site's output space.",
                site=repr(entry.site_key),
                dim=record.dim,
                site_ndim=len(shape),
            )
        if shape[axis] != record.space_dim:
            raise _unresolvable(
                "basis_dim_mismatch",
                f"subspace basis {record.origin!r} is {record.space_dim}-dimensional, "
                f"but site {entry.site_key!r} has extent {shape[axis]} on axis "
                f"{axis} (shape {shape!r}). A direction is minted for one "
                "representation space: it is never broadcast, truncated, or "
                "padded onto another. Restrict `within=` to sites with a "
                f"{record.space_dim}-wide axis, or bind a different `dim=`.",
                site=repr(entry.site_key),
                basis_dim=record.space_dim,
                site_extent=shape[axis],
                axis=axis,
            )
        view = [1] * len(shape)
        view[axis] = record.space_dim
        dense = support.reshape(view).expand(shape) & entry._mask._dense_ro()
        entries.append(_entry_with_mask(entry, dense, "exact", source))
    return ResolvedSelection(trace, "ACT", entries)


def subspace(
    within: Any,
    basis: torch.Tensor,
    *,
    origin: str,
    method: str | None = None,
    dim: int = -1,
    tol: float = 0.0,
) -> Selection:
    """Select the elements a direction (or subspace) in activation space lives on.

    ``basis`` is one direction ``[d]`` or a stack ``[k, d]`` (probe
    directions, steering vectors, PCA components, SAE decoder rows),
    canonicalized to float64. The resolved mask is the basis's SUPPORT SET —
    every element whose coordinate on the bound axis carries weight
    ``|w| > tol`` in at least one basis vector — expanded across the site's
    other axes. This is a SET claim, not a projection: the weights are
    disclosed (via the provenance record) but never carried on the
    selection, and a dense direction supports the whole axis, so
    ``do(subspace(...), edit)`` edits every supported element, never "the
    component along the direction".

    ``within`` is REQUIRED (a direction is minted for a specific
    representation space): a site label, ``Op``/``Layer`` handle, or any ACT
    selection; a bare layer label on a multi-pass layer is the all-passes
    Layer spelling, and every pass is dimension-checked. ``origin=`` is
    REQUIRED and non-empty — where the basis came from (a direction without
    a recorded origin is an unreproducible result); ``method=`` optionally
    tags the family (``'probe'`` / ``'pca'`` / ``'sae'`` / ``'manual'``).
    Origin, method, geometry, and the basis's sha256 content digest ride
    every resolved entry's ``provenance.source`` (and therefore ``do()``
    audit records). ``dim=`` names the site output axis the basis indexes
    (default ``-1``, the trailing feature axis; conv channel directions are
    ``dim=1``); a site whose extent there differs from ``d`` refuses
    ``selection_unresolvable`` / ``basis_dim_mismatch`` — never a silent
    broadcast or truncation. A basis row entirely at-or-below ``tol``
    refuses at construction (a named direction that contributes no support
    is vacuous). Resolution reads geometry only (unsaved sites resolve);
    ``provenance.relation`` is ``exact``. PARAM/EDGE populations refuse
    ``selection_kind_incompatible``. DOCUMENTED-UNSTABLE spelling.
    """

    if within is None:
        raise ValueError(
            "subspace `within` is required: a direction is minted for a "
            "specific representation space, so name the site(s) it lives on "
            "(a site label, Op/Layer handle, or ACT selection)."
        )
    if not isinstance(origin, str) or not origin.strip():
        raise ValueError(
            "subspace `origin` is required and must be a non-empty string "
            "recording where the basis came from (a trained probe, a PCA fit, "
            "an SAE decoder row, a hand-specified vector): a direction "
            "without a recorded origin is an unreproducible result."
        )
    if method is not None and (not isinstance(method, str) or not method.strip()):
        raise ValueError(f"subspace `method` must be None or a non-empty string; got {method!r}.")
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValueError(f"subspace `dim` must be an int axis index; got {dim!r}.")
    tol = _validate_real_number(tol, "tol", "subspace")
    if tol < 0:
        raise ValueError(f"subspace `tol` must be non-negative; got {tol!r}.")
    canonical = _canonicalize_basis(basis)
    row_max = canonical.abs().max(dim=1).values
    if (row_max <= tol).any():
        silent = int((row_max <= tol).sum().item())
        raise ValueError(
            f"subspace `basis` has {silent} direction(s) with every weight "
            f"<= tol ({tol}): a direction that contributes no support is "
            "vacuous by construction. Drop the zero/sub-tol rows or lower "
            "`tol`."
        )
    record = BasisProvenance(
        origin=origin,
        method=method,
        sha256=_basis_digest(canonical),
        n_vectors=int(canonical.shape[0]),
        space_dim=int(canonical.shape[1]),
        dim=dim,
        tol=tol,
    )
    return Selection(
        _SubspaceTerm(
            within=_lift_within(within, "subspace"),
            basis=canonical,
            provenance=record,
        ),
        kind="ACT",
    )


register_term_resolver(_SubspaceTerm, _resolve_subspace_term)
