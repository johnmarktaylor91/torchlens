"""Engine-owned masked-edit derivation for ``do(selection, edit)``.

The scatter contract lives here: a whole-site mask returns the edit's
behavior unchanged (recipe disclosure stamped); element masks wrap the hook
factory with the EDIT-THEN-SCATTER step on a fresh tensor (stored capture
truth is never written through). Consumers: the ACT selection do-plan
(``torchlens._selection.do_plan.build_selection_do_plan``) and the edge/param
substitution doors. Moved out of ``torchlens/_selection/selection.py`` under the R43
file-size ratchet; semantics unchanged.

A builtin verb under an element mask stays portable: it stores the dense bool
``mask`` and its ``mask_sha256`` among its keywords (the mask travels in the
tensor sidecar). On load :func:`checked_arguments` takes the two keys off and
checks them before the verb's constructor runs, and :func:`with_stored_mask`
wraps the rebuilt verb with the same scatter, so a reload applies
bit-identically. A malformed, partial or tampered stored mask is reported as an
``intervention_helper_arguments_invalid`` problem. The digest detects corruption
and hand edits; it is not a signature. DOCUMENTED-UNSTABLE persisted form.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import torch

from .._errors import InvalidArgumentError
from .._selection.selection import Site, _apply_invalid, _Mask
from .types import Edit

__all__ = [
    "MASK_KEYS",
    "ArgumentProblem",
    "_derive_masked_edit",
    "_masked_factory",
    "_validate_edited",
    "checked_arguments",
    "masked_builtin_edit",
    "split_stored_mask",
    "with_stored_mask",
]


def _row_equivariance_facts(
    edit: Edit, entry: Site, site_label: str
) -> tuple[tuple[str, Any], ...]:
    """Apply the mask-equivariance law to a row-coherent element-masked edit.

    The v1 check is DRAW-INDEPENDENT (adjudicated over SOL's per-permutation
    rule, which would make legality depend on the realized seed and
    intermittently fail a seed sweep): every batch row must share the same
    non-batch mask. A shared token span across rows is a legitimate positional
    resampling ablation and passes; per-row-differing masks refuse typed,
    teaching the equal-mask-group remedy. The returned facts (mask digest +
    row-equivalence summary) ride the derived spec's persisted metadata NOW so
    the deferred mask-visible draw grouping needs no schema change.

    Parameters
    ----------
    edit:
        The row-coherent helper spec (metadata ``row_coherent`` +
        ``batch_axis``).
    entry:
        Resolved site entry carrying the element mask.
    site_label:
        Site label for refusal messages.

    Returns
    -------
    tuple[tuple[str, Any], ...]
        Metadata disclosure pairs for the derived spec.
    """

    metadata = dict(edit.metadata)
    axis = metadata.get("batch_axis")
    dense = entry._mask._dense_ro()
    if not isinstance(axis, int) or not -dense.ndim <= axis < dense.ndim:
        raise InvalidArgumentError(
            f"row-coherent edit {edit.name!r} declares batch axis "
            f"{axis!r}, which is out of range for site {site_label!r}'s mask "
            f"of rank {dense.ndim} (shape {tuple(dense.shape)!r})",
            code="sampling_geometry_mismatch",
            remedy="pass the batch axis that exists on this site's geometry",
            argument="axis",
        )
    normalized_axis = axis % dense.ndim
    first = dense.select(normalized_axis, 0)
    rows_equal = all(
        torch.equal(dense.select(normalized_axis, index), first)
        for index in range(1, dense.shape[normalized_axis])
    )
    digest = hashlib.sha256(
        dense.detach().to("cpu", torch.uint8).contiguous().numpy().tobytes()
    ).hexdigest()[:16]
    if not rows_equal:
        raise InvalidArgumentError(
            f"row-coherent edit {edit.name!r} at {site_label!r} received "
            "a partial mask whose selected elements DIFFER across batch rows: a "
            "whole-row exchange under a per-row mask would silently mix donor "
            "and subject values row by row. The v1 legality check is "
            "draw-independent (all rows must share one non-batch mask), so a "
            "seed sweep can never flip it",
            code="mask_not_row_equivariant",
            remedy="select the same positions in every row (an all-rows span "
            "such as tl.units masks broadcast over the batch axis), or split "
            "the batch into equal-mask groups and apply the edit per group",
            argument="selection",
        )
    return (
        ("selection_mask_digest", digest),
        ("selection_mask_rows_equal", True),
    )


def _validate_edited(edited: Any, out: torch.Tensor, site_label: str) -> torch.Tensor:
    """Validate the edit output against the site output (no-broadcast v1)."""

    if not isinstance(edited, torch.Tensor):
        raise _apply_invalid(
            "not_maskable",
            f"element-masked edit at {site_label!r} produced a non-tensor "
            f"({type(edited).__name__}); masked edits require tensor outputs.",
            site=site_label,
        )
    if tuple(edited.shape) != tuple(out.shape):
        try:
            torch.broadcast_shapes(tuple(edited.shape), tuple(out.shape))
            broadcastable = True
        except RuntimeError:
            broadcastable = False
        raise _apply_invalid(
            "broadcast" if broadcastable else "shape",
            f"edit output shape {tuple(edited.shape)!r} does not match site "
            f"{site_label!r} output shape {tuple(out.shape)!r}; no broadcasting in v1.",
            site=site_label,
        )
    if edited.dtype != out.dtype:
        raise _apply_invalid(
            "dtype",
            f"edit output dtype {edited.dtype} does not match site {site_label!r} "
            f"output dtype {out.dtype}.",
            site=site_label,
        )
    if edited.device != out.device:
        raise _apply_invalid(
            "device",
            f"edit output device {edited.device} does not match site {site_label!r} "
            f"output device {out.device}.",
            site=site_label,
        )
    return edited


def _masked_factory(inner_factory: Any, mask: _Mask, site_label: str) -> Any:
    """Wrap a helper hook factory with the engine-owned scatter step."""

    def factory() -> Any:
        """Instantiate the inner hook and wrap it with the mask scatter."""

        inner = inner_factory()

        def _masked_hook(out: Any, *, ctx: Any) -> Any:
            """Run the inner edit, then scatter only masked elements into ``out``."""

            if not isinstance(out, torch.Tensor):
                raise _apply_invalid(
                    "not_maskable",
                    f"site {site_label!r} produced a non-tensor output at apply "
                    "time; element-masked edits address single-tensor outputs only.",
                    site=site_label,
                )
            if tuple(out.shape) != mask.shape:
                raise _apply_invalid(
                    "shape",
                    f"site {site_label!r} output shape {tuple(out.shape)!r} does not "
                    f"match the selection's recorded index space {mask.shape!r}.",
                    site=site_label,
                )
            edited = inner(out, ctx=ctx)
            edited = _validate_edited(edited, out, site_label)
            dense = mask._dense_ro().to(out.device)
            # EDIT-THEN-SCATTER on a fresh tensor; stored capture truth is
            # never written through.
            return torch.where(dense, edited, out)

        return _masked_hook

    return factory


#: The keywords a masked builtin edit stores beside the verb's own.
MASK_KEYS = ("mask", "mask_sha256")


def _mask_digest(mask: torch.Tensor) -> str:
    """Return the sha256 of a bool element mask (shape header plus bytes)."""

    digest = hashlib.sha256()
    digest.update(repr(tuple(mask.shape)).encode())
    digest.update(mask.to(torch.uint8).contiguous().numpy().tobytes())
    return digest.hexdigest()


def masked_builtin_edit(edit: Edit, mask: _Mask, site_label: str) -> Edit:
    """Return a builtin verb restricted to ``mask``, still portable.

    The stored keywords gain the dense bool ``mask`` and its ``mask_sha256``
    (savers write the mask through the tensor sidecar; :func:`with_stored_mask`
    rebuilds it on load). A reloaded masked verb staged under another mask scatters
    through both, their intersection, so that is the one mask stored. Masks of
    different shapes cannot be stored as one: that edit is ``opaque_audit`` and
    the stored mask's scatter refuses at apply time.

    Parameters
    ----------
    edit:
        Builtin verb with a runtime factory.
    mask:
        The resolved site's element mask.
    site_label:
        Site label for apply-time refusals.

    Returns
    -------
    Edit
        The masked verb with a scatter-wrapped factory.
    """

    factory = _masked_factory(edit.factory, mask, site_label)
    dense = mask.to_dense().cpu().contiguous()
    stored = dict(edit.kwargs).get("mask")
    if isinstance(stored, torch.Tensor):
        if tuple(stored.shape) != tuple(dense.shape):
            return dataclasses.replace(edit, factory=factory, portability="opaque_audit")
        dense = dense & stored
    kwargs = tuple(item for item in edit.kwargs if item[0] not in MASK_KEYS)
    kwargs += (("mask", dense), ("mask_sha256", _mask_digest(dense)))
    return dataclasses.replace(edit, kwargs=kwargs, factory=factory)


def _derive_masked_edit(edit: Any, entry: Site, digest: str, site_label: str) -> Any:
    """Derive the per-site edit spec/hook under the mask contract.

    A whole-site mask short-circuits the scatter and returns the edit's
    behavior unchanged (only the recipe disclosure is stamped). Element
    masks wrap the hook factory with the engine scatter. A builtin verb
    stores the dense mask and its sha256 among its own keywords and stays
    portable (a projected ``subspace_edit`` through its own
    payload); any other edit is ``opaque_audit``. The recipe rides the
    ``selection_recipe`` family.
    """

    from ._subspace_projection import SUBSPACE_EDIT_NAME, masked_subspace_edit

    recipe = {
        "resolve_digest": digest,
        "site_key": repr(entry.key),
        "relation": entry.provenance.relation,
        "selected": entry.num_selected,
        "source": entry.provenance.source,
    }
    if isinstance(edit, Edit):
        disclosure = (
            ("selection_digest", digest),
            ("selection_site", repr(entry.key)),
            ("selection_relation", entry.provenance.relation),
        )
        if entry._mask.form == "whole":
            return dataclasses.replace(
                edit,
                metadata=tuple(edit.metadata) + disclosure,
                selection_recipe=recipe,
            )
        if edit.factory is None:
            raise _apply_invalid(
                "not_maskable",
                f"edit {edit.name!r} has no runtime factory to mask.",
                site=site_label,
            )
        if dict(edit.metadata).get("row_coherent"):
            disclosure = disclosure + _row_equivariance_facts(edit, entry, site_label)
        if edit.portability == "builtin":
            # A builtin verb stores its mask, so it stays portable.
            builder = (
                masked_subspace_edit if edit.name == SUBSPACE_EDIT_NAME else masked_builtin_edit
            )
            masked = builder(edit, entry._mask, site_label)
        else:
            # Any other edit cannot be rebuilt from its persisted form alone.
            factory = _masked_factory(edit.factory, entry._mask, site_label)
            masked = dataclasses.replace(edit, factory=factory, portability="opaque_audit")
        return dataclasses.replace(
            masked, metadata=tuple(edit.metadata) + disclosure, selection_recipe=recipe
        )
    if callable(edit):
        if entry._mask.form == "whole":
            return edit

        def _plain_factory() -> Any:
            """Adapt a bare hook callable to the masked-factory protocol."""

            def _adapter(out: Any, *, ctx: Any) -> Any:
                """Forward to the user's hook callable unchanged."""

                return edit(out, ctx=ctx)

            return _adapter

        return _masked_factory(_plain_factory, entry._mask, site_label)()
    raise ValueError(
        "do(selection, edit) requires an Edit/Edit, a hook callable, or a "
        f"replacement tensor; got {type(edit).__name__}."
    )


_RESAVE = (
    "re-save the edit from the session that built it: fork.do(tl.units(site, mask), edit) "
    "then tl.io.save_intervention(fork, path, level='portable')"
)
_REBUILD = (
    "load the spec with the torchlens version that saved it, or rebuild the "
    "edit with the current constructor and save it again"
)


class ArgumentProblem(NamedTuple):
    """Why a saved builtin edit's arguments do not rebuild it.

    The caller raises it as ``intervention_helper_arguments_invalid``.
    """

    argument: str
    message: str
    remedy: str
    cause: BaseException | None = None


def checked_arguments(
    name: str, constructor: Callable[..., Edit], args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[Mapping[str, Any], _Mask | None, ArgumentProblem | None]:
    """Split a saved builtin edit's stored mask off and check what is left binds.

    Parameters
    ----------
    name:
        Builtin helper name.
    constructor:
        The helper's public constructor.
    args, kwargs:
        The saved arguments.

    Returns
    -------
    tuple[Mapping[str, Any], _Mask | None, ArgumentProblem | None]
        The verb's own keywords, the checked stored mask (``None`` when none is
        stored) and the first problem found (``None`` when the arguments rebuild).
    """

    own, mask, problem = split_stored_mask(name, kwargs)
    if problem is not None:
        return own, mask, problem
    try:
        inspect.signature(constructor).bind(*args, **own)
    except TypeError as exc:
        message = (
            f"The saved intervention calls builtin helper {name!r} with arguments its "
            f"constructor does not take ({exc}); the spec was saved by a torchlens whose "
            "helper keywords differ"
        )
        return own, mask, ArgumentProblem("kwargs", message, _REBUILD, exc)
    return own, mask, None


def split_stored_mask(
    name: str, kwargs: Mapping[str, Any]
) -> tuple[Mapping[str, Any], _Mask | None, ArgumentProblem | None]:
    """Take a stored element mask off a saved builtin edit's keywords.

    ``subspace_edit`` keeps its keywords: its own constructor checks its mask.
    Returns the verb's own keywords, the checked mask (``None`` when no mask is
    stored or it is malformed) and the problem with a malformed, partial or
    tampered stored mask.
    """

    from ._subspace_projection import SUBSPACE_EDIT_NAME

    if name == SUBSPACE_EDIT_NAME or not any(key in kwargs for key in MASK_KEYS):
        return kwargs, None, None
    own = {key: value for key, value in kwargs.items() if key not in MASK_KEYS}
    mask, problem = _checked_mask(name, kwargs.get("mask"), kwargs.get("mask_sha256"))
    return own, mask, problem


def with_stored_mask(edit: Edit, mask: _Mask | None) -> Edit:
    """Wrap a rebuilt builtin verb with its stored mask (``None`` returns it unchanged)."""

    if mask is None:
        return edit
    return masked_builtin_edit(edit, mask, f"the saved {edit.name} edit's site")


def _problem(name: str, field: str, why: str) -> tuple[None, ArgumentProblem]:
    """Return the problem with one malformed stored mask field."""

    message = (
        f"The saved intervention's masked {name!r} edit has a malformed {field!r} keyword: "
        f"{why}, so the positions it would edit are not the ones recorded"
    )
    return None, ArgumentProblem(field, message, _RESAVE)


def _checked_mask(name: str, mask: Any, digest: Any) -> tuple[_Mask | None, ArgumentProblem | None]:
    """Return the stored element mask, or the problem with a malformed, partial or tampered one."""

    if mask is None:
        return _problem(name, "mask", "a mask digest was stored without its mask")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim < 1:
        kind = f"{mask.dtype} tensor" if isinstance(mask, torch.Tensor) else type(mask).__name__
        return _problem(name, "mask", f"expected a bool tensor, got {kind}")
    if mask.layout != torch.strided or mask.device.type == "meta":
        return _problem(name, "mask", f"expected a dense mask with data, got {mask.layout}")
    if not isinstance(digest, str) or not digest:
        return _problem(name, "mask_sha256", f"expected a non-empty string, got {digest!r}")
    dense = mask.detach().cpu().contiguous()
    if _mask_digest(dense) != digest:
        return _problem(name, "mask", f"it no longer matches its recorded sha256 {digest[:16]}...")
    return _Mask(shape=tuple(dense.shape), form="dense", dense=dense), None
