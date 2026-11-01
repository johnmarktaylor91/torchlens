"""Audited wrapper boundaries and input sites."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import torch
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from ... import _state
from ._tl import (
    get_tensor_label,
)
from .escape_detection import (
    ExpectedOriginalToken,
)

__all__ = (
    "_in_replacement_hook_frame",
    "_is_expected_opaque_dispatch",
    "completeness_scope_for_wrapper",
    "record_runnable_input_storage_sites",
    "_classify_input_storage_alias",
    "_input_base_tensor",
    "_record_input_metadata_read_at_site",
    "_record_input_metadata_read",
    "_observe_input_metadata_read",
)


def _in_replacement_hook_frame() -> bool:
    """Return whether a genuine raw replacement hook is executing above the dispatch.

    A genuine output-replacement ``register_forward_hook`` runs the user hook inside
    TorchLens's own ``wrapped_hook`` frame (``model_prep._instrumented_forward_hook``).
    Every aten dispatch emitted while that torchlens-owned frame is live is genuine
    replacement construction -- either a raw-aten call (unowned) or a python-wrapped
    call whose op is orphaned out of the final trace because its only consumer is the
    untraceable replacement tensor. This exact, per-event signal lets the completeness
    census excuse ONLY the untraceable dispatch attributable to a real replacement
    while STILL failing on any unrelated silent drop, which fires OUTSIDE a
    replacement hook.

    Returns
    -------
    bool
        ``True`` only when a torchlens replacement-hook frame is live on the stack.
    """

    frame: Any = sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        if code.co_name == _REPLACEMENT_HOOK_FUNC:
            try:
                if Path(code.co_filename).resolve() == _REPLACEMENT_HOOK_FILE:
                    return True
            except (OSError, RuntimeError, ValueError):
                pass
        frame = frame.f_back
    return False


def _is_expected_opaque_dispatch(operator: str, owner: ExpectedOriginalToken) -> bool:
    """Return whether one owned dispatch exactly matches an audited boundary.

    Parameters
    ----------
    operator:
        Stable dispatcher operator name.
    owner:
        Exact wrapper token active for the dispatch.

    Returns
    -------
    bool
        ``True`` for an exact wrapper/operator row or a wrapper-wide metadata boundary.
    """

    return any(
        row.wrapper_name == owner.wrapper_name
        and (row.operator is None or row.operator == operator)
        for row in AUDITED_COMPLETENESS_BOUNDARIES
    )


def completeness_scope_for_wrapper(
    wrapper_name: str,
) -> Literal["owned", "expected_opaque"]:
    """Return the exact audited census scope for a wrapper edge.

    Parameters
    ----------
    wrapper_name:
        Stable wrapper edge name.

    Returns
    -------
    Literal["owned", "expected_opaque"]
        Audited scope; unknown wrappers always remain owned and fail closed.
    """

    return "expected_opaque" if wrapper_name in _EXPECTED_OPAQUE_WRAPPERS else "owned"


def record_runnable_input_storage_sites(
    trace: Any, tensor_leaves: list[tuple[torch.Tensor, Any]]
) -> None:
    """Index model-input TENSOR leaves by BASE-storage identity for alias-read witnessing (r31).

    The object-identity map (``_runnable_input_tensor_sites``) misses a metadata read routed
    through a ``.data`` / ``.detach()`` alias (or any derived view) of an input leaf: the alias
    is a distinct Python object sharing the leaf's STORAGE but neither the leaf object nor a
    ``_base``-linked view. This companion map lets :func:`_classify_input_storage_alias`
    attribute such a read by storage identity + geometry. Storage pointers and geometry are
    read under the internal-scalar-read marker so the live ``untyped_storage`` / ``data_ptr`` /
    ``stride`` / ``storage_offset`` patches treat them as TorchLens-internal (no spurious
    escape record, no fail-closed data_ptr trip). Runs only for runnable captures; stores no
    tensors.
    """

    if not tensor_leaves:
        return
    # INV-2 annotation (r37): this map keys candidate input-leaf sites by storage
    # POINTER for attribution-only lookups (a ``.data``/view metadata read resolves
    # to its leaf). A pointer miss fails CLOSED (no attribution -> the fail-closed
    # nets keep the run honest), never proves disjointness, so identity keying is
    # sound without the absolute-interval engine.
    storage_sites: dict[int, list[Any]] = {}
    # r73 F1: LABEL-keyed capture-layout map for the input-DERIVED activation layout
    # net. Input source tensors are logged (and labeled) BEFORE this indexer runs, so
    # each leaf's raw label resolves an ``OpEvent.input_ancestors`` member back to its
    # boundary site plus the leaf's capture-time stride tuple -- the layout basis a
    # derived-intermediate layout read depends on.
    label_layouts: dict[str, tuple[Any, tuple[int, ...]]] = {}
    # ``pause_logging`` suppresses OP CAPTURE (``storage_offset`` / ``untyped_storage`` are
    # torch-function-wrapped and would otherwise be logged as spurious ops, shifting call ids);
    # ``internal_scalar_read`` marks the reads internal for the escape census / metadata patches.
    with _state.pause_logging(), internal_scalar_read():
        for tensor, site in tensor_leaves:
            try:
                ptr = tensor.untyped_storage().data_ptr()
                geometry = (
                    tuple(tensor.shape),
                    tuple(int(v) for v in tensor.stride()),
                    int(tensor.storage_offset()),
                )
                # r33 F5: the LEAF's conj/neg dispatch bits. A conj/neg VIEW of an input leaf
                # shares its storage AND geometry but flips these bits; recording the leaf's
                # true bits here lets the classifier reject such a same-geometry view instead
                # of misattributing its ``is_conj``/``is_neg`` read as a leaf fact.
                conj_neg = (bool(tensor.is_conj()), bool(tensor.is_neg()))
            except (RuntimeError, AttributeError, TypeError, ValueError, NotImplementedError):
                continue
            storage_sites.setdefault(ptr, []).append((site, *geometry, *conj_neg))
            label = get_tensor_label(tensor)
            if isinstance(label, str):
                label_layouts[label] = (site, geometry[1])
    if storage_sites:
        _RUNNABLE_INPUT_STORAGE_SITES[trace] = storage_sites
    if label_layouts:
        trace._runnable.input_label_layouts = label_layouts


def _classify_input_storage_alias(
    trace: Any, source: torch.Tensor
) -> tuple[str | None, Any, tuple[bool, bool] | None]:
    """Classify ``source`` against the input-leaf storage map (r31, holes A/C).

    Returns ``(_ALIAS_EQUIVALENT, site, leaf_conj_neg)`` when ``source`` shares an input leaf's
    base storage with IDENTICAL geometry (a ``.data`` / ``.detach()`` alias -- a metadata read on
    it equals a direct leaf read), ``(_ALIAS_DERIVED_VIEW, site, None)`` when it shares the
    storage with DIFFERENT geometry (a derived view the replay never re-derives), or
    ``(None, None, None)`` when it does not alias any input leaf's storage (a genuine unrelated
    activation). ``leaf_conj_neg`` is the matched leaf's ``(is_conj, is_neg)`` bits so the caller
    can reject a same-geometry conj/neg view (r33 F5). Storage-pointer/geometry reads run under
    the internal marker so the live patches stay pass-through and cannot recurse back into
    observation.
    """

    storage_sites = _RUNNABLE_INPUT_STORAGE_SITES.get(trace)
    if not storage_sites:
        return (None, None, None)
    # ``pause_logging`` so the ``untyped_storage`` / ``storage_offset`` reads below are not
    # captured as spurious ops mid-forward; ``internal_scalar_read`` keeps them off the census.
    with _state.pause_logging(), internal_scalar_read():
        try:
            ptr = source.untyped_storage().data_ptr()
        except (RuntimeError, AttributeError, TypeError, NotImplementedError):
            return (None, None, None)
        candidates = storage_sites.get(ptr)
        if not candidates:
            return (None, None, None)
        try:
            geometry: Any = (
                tuple(source.shape),
                tuple(int(v) for v in source.stride()),
                int(source.storage_offset()),
            )
        except (RuntimeError, TypeError, ValueError):
            geometry = None
    for site, size, stride, offset, leaf_conj, leaf_neg in candidates:
        if geometry is not None and geometry == (size, stride, offset):
            return (_ALIAS_EQUIVALENT, site, (leaf_conj, leaf_neg))
    return (_ALIAS_DERIVED_VIEW, candidates[0][0], None)


def _input_base_tensor(source: torch.Tensor) -> torch.Tensor | None:
    """Return ``source._base`` read under the internal marker (r31).

    ``_base`` is a witnessed getset PROPERTY replaced by a recording descriptor during a
    runnable forward; reading it here for the view-linkage check must go under the
    internal-scalar-read marker so the recording getter treats it as a TorchLens-internal read
    (no recursion back into observation).
    """

    with _state.pause_logging(), internal_scalar_read():
        try:
            base = source._base
        except (RuntimeError, AttributeError):
            return None
    return base if isinstance(base, torch.Tensor) else None


def _record_input_metadata_read_at_site(trace: Any, site: Any, name: str, value: Any) -> None:
    """Record one metadata-read fact against a resolved MODEL-INPUT leaf site.

    Facts accumulate per input site into a runtime-only Trace stash the runnable producer
    serializes as declared witness facts. A repeated read of the same predicate overwrites --
    tensor metadata is stable across one forward, so the values are identical unless an in-place
    layout change occurred, in which case the LAST observed value is the one nearest the branch.
    """

    facts = trace._runnable.input_metadata_reads
    site_facts = facts.setdefault(site, {})
    site_facts[name] = value


def _record_input_metadata_read(trace: Any, source: torch.Tensor, name: str, value: Any) -> None:
    """Record one metadata-read fact against the model-input leaf that IS ``source`` (by identity).

    The receiver is attributed via the object-identity map recorded at capture start
    (``_record_runnable_input_tensor_sites``); a read on any other tensor records nothing here
    (storage-alias attribution is handled by :func:`_observe_input_metadata_read`).
    """

    sites = trace._runnable.input_tensor_sites
    if not sites:
        return
    site = sites.get(id(source))
    if site is None:
        return
    _record_input_metadata_read_at_site(trace, site, name, value)


def _observe_input_metadata_read(trace: Any, source: torch.Tensor, name: str, value: Any) -> None:
    """Attribute one metadata read to a model-input leaf, an alias, or a fail-closed view (r31).

    Cases, in order:

    * The receiver IS a model-input leaf (object identity) -> record a re-checkable
      (site, predicate, value) fact.
    * LEAF-ONLY AUTOGRAD (``requires_grad`` / ``grad_fn``): TorchLens's own per-op bookkeeping
      reads these on input-derived views while logging is enabled (verified), indistinguishable
      from a user view read, so a non-leaf read is IGNORED (leaf-only; documented residual).
    * VIEW-FAIL AUTOGRAD / structural (``is_leaf`` / ``retains_grad`` / ``_base`` / ``_is_view``):
      a read on a DERIVED VIEW of an input leaf (``retains_grad`` on a non-leaf view, r31 hole C)
      is attributed by the CHEAP ``_base``-in-sites linkage and fails closed -- the view's state
      is not re-derivable from the runtime leaf. A ``.data`` / ``.detach()`` storage-alias
      (``_base`` None) is IGNORED (CONSTANT/detached, input-independent, no hole). These four are
      NEVER read internally on an input view (verified), so a ``_base`` match is a genuine user
      Python view read -- the framework-vs-user discriminator is the linkage itself.
    * ALIAS-SAFE family (layout methods + ``is_conj`` / ``is_neg`` / ``is_inference`` /
      ``is_pinned`` / ``is_shared`` / ``is_coalesced``): attributed by STORAGE IDENTITY. A read
      on a ``.data`` / ``.detach()`` storage-alias with IDENTICAL geometry (r31 hole A) records
      the leaf fact -- its value provably equals a direct leaf read. A storage-alias with
      DIFFERENT geometry (a derived view, ``x.t().is_contiguous()``) fails closed. Only
      Python-level reads reach this patch; torch's internal C++ layout reads bypass it, so a
      layout-oblivious model records nothing.
    * Anything else (a genuinely new activation not aliasing an input) -> ignore.
    """

    sites = trace._runnable.input_tensor_sites
    if not sites:
        return
    if id(source) in sites:
        _record_input_metadata_read(trace, source, name, value)
        return
    if name in _INPUT_METADATA_LEAF_ONLY_AUTOGRAD_NAMES:
        # ``requires_grad`` / ``grad_fn`` are read by TorchLens's own per-op bookkeeping on
        # input-derived views; witnessed on the LEAF only (see the set docstring).
        return
    if name in _INPUT_METADATA_VIEW_FAIL_AUTOGRAD_NAMES:
        base = _input_base_tensor(source)
        if base is not None and id(base) in sites:
            _INPUT_METADATA_VIEW_READ.add(trace)
        return
    if name in _INPUT_METADATA_ALIAS_SAFE_NAMES:
        kind, site, leaf_conj_neg = _classify_input_storage_alias(trace, source)
        if kind == _ALIAS_EQUIVALENT:
            # r33 F5 (over-trigger fix): a conj/neg VIEW shares an input leaf's storage AND
            # geometry but FLIPS the conj/neg dispatch bit. Geometry alone would record the
            # view's ``is_conj=True`` as a LEAF fact, which the RAW runtime leaf (``is_conj``
            # False) then contradicts -> a forced FALSE divergence on the ORIGINAL input
            # (complex models permanently diverged, r31 regression). For ``is_conj``/``is_neg``
            # the observed bit must EQUAL the leaf's; a same-geometry bit MISMATCH is a genuine
            # derived (conj/neg) view and fails closed rather than misrecording a leaf fact.
            if name in _INPUT_METADATA_CONJ_NEG_NAMES and leaf_conj_neg is not None:
                leaf_bit = leaf_conj_neg[0] if name == "is_conj" else leaf_conj_neg[1]
                if bool(value) != bool(leaf_bit):
                    _INPUT_METADATA_VIEW_READ.add(trace)
                    return
            _record_input_metadata_read_at_site(trace, site, name, value)
        elif kind == _ALIAS_DERIVED_VIEW:
            _INPUT_METADATA_VIEW_READ.add(trace)
        else:
            base = _input_base_tensor(source)
            if base is not None and id(base) in sites:
                _INPUT_METADATA_VIEW_READ.add(trace)
            elif name in _INPUT_METADATA_LAYOUT_NAMES:
                # r73 F1: a layout read on a genuinely NEW activation (fresh storage,
                # no input alias/view linkage). Memory format PROPAGATES through
                # elementwise ops, so if the receiver's value DAG roots at a model
                # input, the read steers on the runtime input's layout -- attribute
                # by traced ancestry (see ``INPUT_DERIVED_LAYOUT_FACT_NAME``).
                _observe_input_derived_layout_read(trace, source)


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._completeness_metadata import (  # noqa: E402
    _observe_input_derived_layout_read,
)
from ._completeness_origins import (  # noqa: E402
    internal_scalar_read,
)
from .completeness_witness import (  # noqa: E402
    _ALIAS_DERIVED_VIEW,
    _ALIAS_EQUIVALENT,
    _EXPECTED_OPAQUE_WRAPPERS,
    _INPUT_METADATA_ALIAS_SAFE_NAMES,
    _INPUT_METADATA_CONJ_NEG_NAMES,
    _INPUT_METADATA_LAYOUT_NAMES,
    _INPUT_METADATA_LEAF_ONLY_AUTOGRAD_NAMES,
    _INPUT_METADATA_VIEW_FAIL_AUTOGRAD_NAMES,
    _INPUT_METADATA_VIEW_READ,
    _REPLACEMENT_HOOK_FILE,
    _REPLACEMENT_HOOK_FUNC,
    _RUNNABLE_INPUT_STORAGE_SITES,
    AUDITED_COMPLETENESS_BOUNDARIES,
)
