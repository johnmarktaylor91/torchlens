"""Reload a saved ``subspace_edit``: the portable form of a projection-valued edit.

``do(tl.subspace(site, basis, origin=...), verb)`` with a builtin verb stages a
builtin edit named ``subspace_edit`` (see ``_subspace_projection``). A saved
spec stores its inner verb, the float64 orthonormal rows actually applied (a
tensor sidecar), the bound axis, the rows' sha256 and the basis provenance. The
loader rebuilds builtin edits by calling their constructor with the stored
arguments; this module is that constructor. It checks every stored field before
it builds anything: a malformed field refuses ``subspace_spec_invalid``, rows
whose bytes no longer match their digest refuse ``subspace_spec_basis_tampered``,
and rows that are not orthonormal refuse ``subspace_spec_basis_not_orthonormal``.
A rotation learned elsewhere (DAS, ReFT) travels this way: its first ``k`` rows
are the basis. An edit restricted by a ``within=`` element mask (LoReFT's one
position) also stores the dense bool ``mask`` and its ``mask_sha256``; a
malformed mask refuses ``subspace_spec_invalid`` and a mask whose bytes no
longer match its digest refuses ``subspace_spec_mask_tampered``. Any stored
keyword this loader does not know refuses ``subspace_spec_invalid``, so a spec
is never rebuilt with part of its meaning dropped. The digest check detects corruption and hand edits of the rows;
it is not a signature, so a forger who also recomputes the digest is not caught
(the spec is trusted data, like every builtin helper argument).

DOCUMENTED-UNSTABLE persisted form.
"""

from __future__ import annotations

from typing import Any

import torch

from .._errors import InvalidArgumentError
from .._selection import _Mask
from .._selection_subspace import _basis_digest
from ._subspace_projection import (
    _ORTHONORMAL_TOL,
    SubspaceProjection,
    _mask_digest,
    masked_subspace_edit,
)
from .types import Edit

__all__ = ["rebuild_subspace_edit"]

_RESAVE = (
    "re-save the edit from the session that built it: "
    "fork.do(tl.subspace(site, basis, origin=...), edit) then "
    "tl.io.save_intervention(fork, path, level='portable') "
    "(see the subspace section of docs/intervention_explainers.md)"
)


def _invalid(field: str, why: str, *, code: str = "subspace_spec_invalid") -> InvalidArgumentError:
    """Return the refusal for one malformed stored field.

    ``code`` is spelled at each raise site so the refusal census sees it.
    """

    return InvalidArgumentError(
        f"the saved subspace edit's {field!r} field is malformed: {why}, so the spec "
        "cannot rebuild the projection it recorded",
        code=code,
        remedy=_RESAVE,
        argument=field,
    )


def _is_text(value: Any) -> bool:
    """Return whether ``value`` is a non-empty string."""

    return isinstance(value, str) and bool(value.strip())


def _checked_basis(basis: Any) -> torch.Tensor:
    """Return the stored rows as a contiguous CPU tensor, refusing a malformed one."""

    if not isinstance(basis, torch.Tensor) or basis.ndim != 2 or 0 in basis.shape:
        shape = tuple(basis.shape) if isinstance(basis, torch.Tensor) else type(basis).__name__
        raise _invalid(
            "basis", f"expected a [k, d] tensor, got {shape!r}", code="subspace_spec_invalid"
        )
    if basis.layout != torch.strided or basis.device.type == "meta":
        raise _invalid(
            "basis",
            f"expected dense rows with data, got {basis.layout} on {basis.device}",
            code="subspace_spec_invalid",
        )
    if basis.dtype != torch.float64:
        raise _invalid(
            "basis", f"expected float64 rows, got {basis.dtype}", code="subspace_spec_invalid"
        )
    if not bool(torch.isfinite(basis).all()):
        raise _invalid("basis", "the rows contain NaN or Inf", code="subspace_spec_invalid")
    return basis.detach().cpu().contiguous()


def _checked_fields(fields: dict[str, Any]) -> None:
    """Refuse a stored scalar field of the wrong type."""

    axis = fields["axis"]
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise _invalid("axis", f"expected an int axis, got {axis!r}", code="subspace_spec_invalid")
    for name in ("basis_sha256", "origin", "origin_sha256", "source_dtype"):
        if not _is_text(fields[name]):
            raise _invalid(
                name,
                f"expected a non-empty string, got {fields[name]!r}",
                code="subspace_spec_invalid",
            )
    if fields["method"] is not None and not _is_text(fields["method"]):
        raise _invalid(
            "method",
            f"expected None or a non-empty string, got {fields['method']!r}",
            code="subspace_spec_invalid",
        )
    if not isinstance(fields["orthonormalised"], bool):
        raise _invalid(
            "orthonormalised",
            f"expected a bool, got {fields['orthonormalised']!r}",
            code="subspace_spec_invalid",
        )


def _checked_rows(rows: torch.Tensor, basis_sha256: str) -> None:
    """Refuse rows whose bytes changed since saving, or that are not orthonormal."""

    if _basis_digest(rows) != basis_sha256:
        raise InvalidArgumentError(
            "the saved subspace edit's basis no longer matches its recorded sha256 "
            f"{basis_sha256[:16]}...: the rows were changed after saving, so the "
            "projection they would apply is not the one recorded",
            code="subspace_spec_basis_tampered",
            remedy=_RESAVE,
            argument="basis",
        )
    gram = rows @ rows.T
    identity = torch.eye(int(rows.shape[0]), dtype=rows.dtype)
    if not bool(torch.allclose(gram, identity, rtol=0.0, atol=_ORTHONORMAL_TOL)):
        error = float((gram - identity).abs().max())
        raise InvalidArgumentError(
            "the saved subspace edit's basis rows are not orthonormal (largest Gram "
            f"error {error:.3g}, tolerance {_ORTHONORMAL_TOL:g}); a saved projection "
            "always stores the orthonormal rows it applied",
            code="subspace_spec_basis_not_orthonormal",
            remedy=_RESAVE,
            argument="basis",
        )


def _checked_mask(mask: Any, digest: Any, projection: SubspaceProjection) -> _Mask:
    """Return the stored element mask, refusing a malformed or tampered one."""

    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim < 1:
        kind = f"{mask.dtype} tensor" if isinstance(mask, torch.Tensor) else type(mask).__name__
        raise _invalid("mask", f"expected a bool tensor, got {kind}", code="subspace_spec_invalid")
    if mask.layout != torch.strided or mask.device.type == "meta":
        raise _invalid(
            "mask",
            f"expected a dense mask with data, got {mask.layout} on {mask.device}",
            code="subspace_spec_invalid",
        )
    axis = projection.dim % mask.ndim if -mask.ndim <= projection.dim < mask.ndim else None
    width = int(projection.rows.shape[1])
    if axis is None or int(mask.shape[axis]) != width:
        raise _invalid(
            "mask",
            f"its shape {tuple(mask.shape)!r} has no axis {projection.dim} of the basis "
            f"width {width}",
            code="subspace_spec_invalid",
        )
    if not _is_text(digest):
        raise _invalid(
            "mask_sha256",
            f"expected a non-empty string beside the mask, got {digest!r}",
            code="subspace_spec_invalid",
        )
    dense = mask.detach().cpu().contiguous()
    if _mask_digest(dense) != digest:
        raise InvalidArgumentError(
            "the saved subspace edit's mask no longer matches its recorded sha256 "
            f"{str(digest)[:16]}...: the mask was changed after saving, so the "
            "positions it would edit are not the ones recorded",
            code="subspace_spec_mask_tampered",
            remedy=_RESAVE,
            argument="mask",
        )
    if projection.partial_along_support(dense):
        raise _invalid(
            "mask",
            "it keeps only part of the basis support at some position, which a saved "
            "subspace edit never stores",
            code="subspace_spec_invalid",
        )
    return _Mask(shape=tuple(dense.shape), form="dense", dense=dense)


_REQUIRED = ("basis", "axis", "basis_sha256", "origin")
_OPTIONAL: dict[str, Any] = {
    "method": None,
    "origin_sha256": None,
    "source_dtype": "float64",
    "orthonormalised": False,
    "mask": None,
    "mask_sha256": None,
}


def rebuild_subspace_edit(edit: Any, **stored: Any) -> Edit:
    """Rebuild a saved ``subspace_edit`` as the projection it recorded.

    Parameters
    ----------
    edit:
        The inner builtin verb, already rebuilt by the loader.
    **stored:
        The saved keywords: ``basis`` (the float64 orthonormal ``[k, d]`` rows
        the edit applied), ``axis`` (the bound site axis), ``basis_sha256``
        (digest of ``basis``: shape header plus float64 bytes), ``origin`` and
        optionally ``method`` (the ``tl.subspace`` provenance), ``origin_sha256``
        (digest of the caller's basis; default ``basis_sha256``),
        ``source_dtype`` (dtype of the caller's basis; default ``"float64"``),
        ``orthonormalised`` (default ``False``), and ``mask`` with
        ``mask_sha256`` (the ``within=`` element mask; default ``None``, the
        whole site).

    Returns
    -------
    Edit
        The builtin ``subspace_edit`` whose hook edits the component along the
        stored rows, exactly as the saved session's edit did.

    Raises
    ------
    InvalidArgumentError
        ``subspace_spec_invalid``, ``subspace_spec_basis_tampered``,
        ``subspace_spec_basis_not_orthonormal`` or ``subspace_spec_mask_tampered``.
    """

    if not isinstance(edit, Edit) or edit.portability != "builtin":
        raise _invalid(
            "edit",
            f"expected a builtin edit verb, got {type(edit).__name__}",
            code="subspace_spec_invalid",
        )
    unknown = sorted(set(stored) - set(_REQUIRED) - set(_OPTIONAL))
    missing = [name for name in _REQUIRED if name not in stored]
    if unknown or missing:
        field = (unknown or missing)[0]
        raise _invalid(
            field,
            f"unknown fields {unknown}, missing fields {missing}",
            code="subspace_spec_invalid",
        )
    fields = {**_OPTIONAL, **stored}
    if fields["origin_sha256"] is None:
        fields["origin_sha256"] = fields["basis_sha256"]
    rows = _checked_basis(fields["basis"])
    _checked_fields(fields)
    _checked_rows(rows, str(fields["basis_sha256"]))
    projection = SubspaceProjection(
        rows,
        int(fields["axis"]),
        str(fields["origin_sha256"]),
        bool(fields["orthonormalised"]),
        origin=str(fields["origin"]),
        method=fields["method"],
        source_dtype=str(fields["source_dtype"]),
    )
    if fields["mask"] is None:
        if fields["mask_sha256"] is not None:
            raise _invalid(
                "mask_sha256",
                "a mask digest was stored without its mask",
                code="subspace_spec_invalid",
            )
        return projection.wrap(edit)
    mask = _checked_mask(fields["mask"], fields["mask_sha256"], projection)
    return masked_subspace_edit(projection.wrap(edit), mask, "the saved subspace edit's site")
