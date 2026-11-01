"""Host-escape state and caller authorization."""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from typing import Any

import torch
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from ... import _state
from .escape_detection import (
    expected_original_call,
    mark_expected_original_accounted,
)

__all__ = (
    "input_metadata_view_read",
    "host_escape_source_labels",
    "host_escape_state_source_names",
    "host_escape_has_unattributable_bool",
    "host_escape_has_unattributable_opaque",
    "host_escape_state_source_labels",
    "host_escape_bool_source_labels",
    "host_escape_bool_consumer_locations",
    "host_escape_label_leaf_origins",
    "_record_escape_label_fallback",
    "pruned_rng_control_source_labels",
    "record_pruned_rng_control_source",
    "alias_mutation_candidate_labels",
    "record_alias_mutation_candidate",
    "pruned_alias_mutation_source_labels",
    "record_pruned_alias_mutation_source",
    "data_alias_mutation_detected",
    "record_data_alias_mutation",
    "host_escape_has_mutable_writeback",
    "host_escape_has_raw_pointer",
    "host_escape_has_cross_thread_captured_tensor",
    "_register_authorized_caller_namespace",
    "_caller_frame_is_torchlens_internal",
    "_raw_storage_ptr_no_observe",
)


def input_metadata_view_read(trace: Any) -> bool:
    """Return whether a metadata predicate was read on an input-derived view (r29-C1, F5)."""

    return trace in _INPUT_METADATA_VIEW_READ


def host_escape_source_labels(trace: Any) -> frozenset[str]:
    """Return the recorded raw escape-source op labels for one trace."""

    labels = _HOST_ESCAPE_SOURCE_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def host_escape_state_source_names(trace: Any) -> frozenset[str]:
    """Return the ``state_dict`` names of every registered-state escape source."""

    names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
    return frozenset(names) if names else frozenset()


def host_escape_has_unattributable_bool(trace: Any) -> bool:
    """Return whether an unwitnessable (pruned, unlabelled) bool escape was seen."""

    return trace in _HOST_ESCAPE_UNATTRIBUTABLE_BOOL


def host_escape_has_unattributable_opaque(trace: Any) -> bool:
    """Return whether an unwitnessable census-invisible escape (``.tolist``/``.numpy``) was seen."""

    return trace in _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE


def host_escape_state_source_labels(trace: Any) -> frozenset[str]:
    """Return the raw escape-source labels whose source is a registered param/buffer."""

    labels = _HOST_ESCAPE_STATE_SOURCE_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def host_escape_bool_source_labels(trace: Any) -> frozenset[str]:
    """Return the raw escape-source labels whose source is a bool control predicate."""

    labels = _HOST_ESCAPE_BOOL_SOURCE_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def host_escape_bool_consumer_locations(trace: Any) -> dict[str, tuple[tuple[str, int], ...]]:
    """Return user source locations that consumed each captured bool tensor."""

    locations = _HOST_ESCAPE_BOOL_CONSUMER_LOCATIONS.get(trace)
    if not locations:
        return {}
    return {label: tuple(entries) for label, entries in locations.items()}


def host_escape_label_leaf_origins(
    trace: Any,
) -> Mapping[str, tuple[frozenset[str], frozenset[str]] | None]:
    """Return the per-raw-label leaf-origin fallback basis for one trace."""

    return dict(_HOST_ESCAPE_LABEL_LEAF_ORIGINS.get(trace, {}))


def _record_escape_label_fallback(trace: Any, raw_label: str, source: torch.Tensor) -> None:
    """Record the leaf-origin fallback basis for one labelled escape source.

    ``None`` (fail-closed marker) wins over any positive entry on collision: if the
    same label escapes twice and either occurrence is unresolvable, the fallback is
    unusable for that label.
    """

    with _state.pause_logging(), internal_scalar_read():
        leaf = _operand_leaf_origins(trace, source)
    if _ORIGIN_UNKNOWN in leaf or _ORIGIN_RNG in leaf:
        entry: tuple[frozenset[str], frozenset[str]] | None = None
    else:
        entry = (
            frozenset(
                origin[len(_ORIGIN_LABEL_PREFIX) :]
                for origin in leaf
                if origin.startswith(_ORIGIN_LABEL_PREFIX)
            ),
            frozenset(
                origin[len(_ORIGIN_STATE_PREFIX) :]
                for origin in leaf
                if origin.startswith(_ORIGIN_STATE_PREFIX)
            ),
        )
    table = _HOST_ESCAPE_LABEL_LEAF_ORIGINS.get(trace)
    if table is None:
        table = {}
        _HOST_ESCAPE_LABEL_LEAF_ORIGINS[trace] = table
    if raw_label in table and table[raw_label] is None:
        return  # fail-closed marker sticks
    if entry is None:
        table[raw_label] = None
        return
    previous = table.get(raw_label)
    if previous is None:
        table[raw_label] = entry
    else:
        table[raw_label] = (previous[0] | entry[0], previous[1] | entry[1])


def pruned_rng_control_source_labels(trace: Any) -> frozenset[str]:
    """Return raw labels of pruned torch-RNG ops that steered control flow."""

    labels = _PRUNED_RNG_CONTROL_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def record_pruned_rng_control_source(trace: Any, label: str) -> None:
    """Record one pruned torch-RNG op whose result drove a control decision."""

    labels = _PRUNED_RNG_CONTROL_LABELS.get(trace)
    if labels is None:
        labels = set()
        _PRUNED_RNG_CONTROL_LABELS[trace] = labels
    labels.add(label)


def alias_mutation_candidate_labels(trace: Any) -> frozenset[str]:
    """Return raw labels of in-place ops that mutate an unlabelled (invisible-alias) target."""

    labels = _ALIAS_MUTATION_CANDIDATE_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def record_alias_mutation_candidate(trace: Any, label: str) -> None:
    """Record one in-place op whose mutation target carries no resolvable capture label."""

    labels = _ALIAS_MUTATION_CANDIDATE_LABELS.get(trace)
    if labels is None:
        labels = set()
        _ALIAS_MUTATION_CANDIDATE_LABELS[trace] = labels
    labels.add(label)


def pruned_alias_mutation_source_labels(trace: Any) -> frozenset[str]:
    """Return raw labels of unlabelled-alias in-place ops that were orphan-pruned."""

    labels = _PRUNED_ALIAS_MUTATION_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def record_pruned_alias_mutation_source(trace: Any, label: str) -> None:
    """Record one orphan-pruned in-place op that mutated an unlabelled (invisible) alias."""

    labels = _PRUNED_ALIAS_MUTATION_LABELS.get(trace)
    if labels is None:
        labels = set()
        _PRUNED_ALIAS_MUTATION_LABELS[trace] = labels
    labels.add(label)


def data_alias_mutation_detected(trace: Any) -> bool:
    """Return whether capture observed a write through a ``Tensor.data`` alias.

    Parameters
    ----------
    trace : Any
        Capture trace to inspect.

    Returns
    -------
    bool
        ``True`` when a successful receiver mutation targeted a data-alias
        tensor or storage-sharing view derived from one.
    """

    return trace in _DATA_ALIAS_MUTATION_TRACES


def record_data_alias_mutation(trace: Any) -> None:
    """Record a successful write through a ``Tensor.data`` alias.

    Parameters
    ----------
    trace : Any
        Active capture trace whose runnable proof must be ceilinged.
    """

    _DATA_ALIAS_MUTATION_TRACES.add(trace)


def host_escape_has_mutable_writeback(trace: Any) -> bool:
    """Return whether a host write-back through a mutable zero-copy alias was detected."""

    return trace in _HOST_ESCAPE_MUTABLE_WRITEBACK


def host_escape_has_raw_pointer(trace: Any) -> bool:
    """Return whether a raw ``Tensor.data_ptr()`` pointer escaped to the host (r15-H1)."""

    return trace in _HOST_ESCAPE_RAW_POINTER


def host_escape_has_cross_thread_captured_tensor(trace: Any) -> bool:
    """Return whether a non-owner thread touched a captured tensor during the window (r43)."""

    return trace in _HOST_ESCAPE_CROSS_THREAD_CAPTURED


def _register_authorized_caller_namespace(namespace: Mapping[str, Any], module_file: str) -> None:
    """Collect a module namespace's own code objects into the witness roster.

    Only code objects whose ``co_filename`` is ``module_file`` register (read
    at import time from real code objects, before user code can interpose),
    so imported foreign helpers and decorator-wrapper code from other modules
    never widen the roster. Nested code constants (closures, comprehensions)
    and ``__wrapped__`` chains are followed so a decorated or nested internal
    caller still authenticates.
    """

    # Imported here, not from this module's bottom block: this runs during the
    # completeness_witness module's import (its roster collection), which can happen
    # while this module is still executing its bottom imports.
    from .completeness_witness import (
        _AUTHORIZED_INTERNAL_CALLER_CODE,
        _AUTHORIZED_INTERNAL_CALLER_CODE_IDS,
    )

    def _collect(code: types.CodeType) -> None:
        """Register one module-local code object and its nested code constants."""

        if code.co_filename != module_file or id(code) in _AUTHORIZED_INTERNAL_CALLER_CODE_IDS:
            return
        _AUTHORIZED_INTERNAL_CALLER_CODE.append(code)
        _AUTHORIZED_INTERNAL_CALLER_CODE_IDS.add(id(code))
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                _collect(const)

    def _walk(value: Any, seen: set[int]) -> None:
        """Recurse module attributes to find every locally defined callable."""

        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        if isinstance(value, property):
            for accessor in (value.fget, value.fset, value.fdel):
                if accessor is not None:
                    _walk(accessor, seen)
            return
        if isinstance(value, types.FunctionType):
            _collect(value.__code__)
            wrapped = getattr(value, "__wrapped__", None)
            if wrapped is not None:
                _walk(wrapped, seen)
            return
        if isinstance(value, type):
            for member in vars(value).values():
                _walk(member, seen)

    seen: set[int] = set()
    for value in namespace.values():
        _walk(value, seen)


def _caller_frame_is_torchlens_internal(depth: int = 2) -> bool:
    """Return whether the frame ``depth`` levels up is executing TorchLens's own code.

    The witness's detector authorization is bound to TorchLens's OWN calling
    frames, never to a helper's identity: user model code that imports and
    calls an internal helper must not inherit the authorization (its raw
    reads then run bare and the shadow detector convicts them normally).
    Authentication is code-object IDENTITY against the import-time roster --
    a frame qualifies only when its ``f_code`` IS one of the roster's own
    code objects. Frame metadata (``f_globals['__name__']``,
    ``co_filename``) is forgeable by ``exec``/``compile`` from user code and
    is deliberately never consulted.

    Parameters
    ----------
    depth:
        Stack depth of the frame to authenticate, counted from this
        function's own frame (``2`` = the immediate caller of the helper
        that invoked this check).
    """

    try:
        caller = sys._getframe(depth)
    except ValueError:
        return False
    return id(caller.f_code) in _AUTHORIZED_INTERNAL_CALLER_CODE_IDS


def _raw_storage_ptr_no_observe(tensor: Any) -> int | None:
    """Return a tensor's untyped-storage data pointer via the true originals, ptr 0 -> None (r43).

    GIL-atomic, wrapper-free, side-effect-free: it never fires an escape observer, a dispatch,
    or a logging toggle, so it is safe to call from ANY thread. ``0`` (a meta / storageless
    tensor) normalizes to ``None`` so distinct storageless tensors never alias one synthetic
    pointer.

    Detector authorization is granted only to TorchLens-internal callers: the
    frame-bound tokens below exist so the WITNESS's own instrumentation reads
    never self-trip the shadow detector. User code that imports and calls this
    helper does not inherit that authorization -- its reads run bare through
    the true originals and the detector observes and convicts them as the
    unwrapped raw reaches they are.
    """

    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        if _state._escape_detector_mode != "off" and _caller_frame_is_torchlens_internal():
            # The witness's own raw-original reads are instrumentation, not an
            # escaped user op: authorize each one through the detector's typed
            # per-call accounting so the shadow detector never reports
            # TorchLens's own frame (a false ceiling that degraded otherwise
            # verified armed captures with user-directed remediation text no
            # user action could clear). The token is FRAME-BOUND to this exact
            # call, so a genuine raw untyped_storage reach anywhere else still
            # trips the detector — the exemption cannot widen.
            with expected_original_call(
                _ORIG_TENSORBASE_UNTYPED_STORAGE,
                "completeness_witness:internal_storage_ptr",
                census_scope="expected_opaque",
            ) as storage_token:
                storage = _ORIG_TENSORBASE_UNTYPED_STORAGE(tensor)
            mark_expected_original_accounted(storage_token, captured=False)
            with expected_original_call(
                _ORIG_UNTYPED_STORAGE_DATA_PTR,
                "completeness_witness:internal_storage_ptr",
                census_scope="expected_opaque",
            ) as ptr_token:
                ptr = _ORIG_UNTYPED_STORAGE_DATA_PTR(storage)
            mark_expected_original_accounted(ptr_token, captured=False)
        else:
            storage = _ORIG_TENSORBASE_UNTYPED_STORAGE(tensor)
            ptr = _ORIG_UNTYPED_STORAGE_DATA_PTR(storage)
    except (RuntimeError, TypeError, NotImplementedError, AttributeError):
        return None
    return int(ptr) if ptr else None


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._completeness_origins import (  # noqa: E402
    _operand_leaf_origins,
    internal_scalar_read,
)
from .completeness_witness import (  # noqa: E402
    _ALIAS_MUTATION_CANDIDATE_LABELS,
    _AUTHORIZED_INTERNAL_CALLER_CODE_IDS,
    _DATA_ALIAS_MUTATION_TRACES,
    _HOST_ESCAPE_BOOL_CONSUMER_LOCATIONS,
    _HOST_ESCAPE_BOOL_SOURCE_LABELS,
    _HOST_ESCAPE_CROSS_THREAD_CAPTURED,
    _HOST_ESCAPE_LABEL_LEAF_ORIGINS,
    _HOST_ESCAPE_MUTABLE_WRITEBACK,
    _HOST_ESCAPE_RAW_POINTER,
    _HOST_ESCAPE_SOURCE_LABELS,
    _HOST_ESCAPE_STATE_SOURCE_LABELS,
    _HOST_ESCAPE_STATE_SOURCE_NAMES,
    _HOST_ESCAPE_UNATTRIBUTABLE_BOOL,
    _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE,
    _INPUT_METADATA_VIEW_READ,
    _ORIG_TENSORBASE_UNTYPED_STORAGE,
    _ORIG_UNTYPED_STORAGE_DATA_PTR,
    _ORIGIN_LABEL_PREFIX,
    _ORIGIN_RNG,
    _ORIGIN_STATE_PREFIX,
    _ORIGIN_UNKNOWN,
    _PRUNED_ALIAS_MUTATION_LABELS,
    _PRUNED_RNG_CONTROL_LABELS,
)
