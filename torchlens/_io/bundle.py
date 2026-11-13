"""Portable directory-bundle save/load helpers for TorchLens model logs.

This module owns the high-level bundle lifecycle for TorchLens portable I/O:
save a completed ``Trace`` into a directory bundle, load that bundle back
eagerly or lazily, and clean up interrupted ``.tmp.*`` directories left behind
by partial saves. The bundle format is intentionally a plain directory with
``manifest.json``, ``metadata.pkl``, and one ``safetensors`` file per blob.
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import json
import os
import pickle
import platform
import shutil
import sys
import uuid
import warnings
import weakref
from collections.abc import Collection, Mapping, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast, overload

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file

from torchlens.backends.registry import BackendPayloadUnsupportedError

from .._capture._episode_ledger_anchors import reanchor_loaded_episode_evidence
from .._data_classes._state_adapter import state_items
from .._data_classes._trace_missing_attrs import is_live_trace
from .._data_classes.module import ModuleAccessor
from .._data_classes.trace import Trace
from .._errors import InvalidArgumentError
from .._runnable.state import (
    bind_embedded_nonpersistent_buffers,
    bind_embedded_trace_state,
    runnable_tensor_byte_digest,
)
from .._utils._record_group_table import record_member, record_storage_attr
from .._utils.display import user_stacklevel
from .._utils.hashing import sha256_file
from .._vocab._package_version import __version__ as TORCHLENS_VERSION
from ..backends.registry import BackendSpec, get_backend_spec
from ..errors import RunnablePreflightError, TorchLensWarning
from . import (
    TLSPEC_VERSION,
    ArtifactError,
    BlobRef,
    FieldPolicy,
    PayloadLoadHints,
    _format3_runnable_bind as runnable_bind,
    _json,
    raise_if_manifest_below_floor,
    refuse_bundle_exists,
    refuse_missing_blobs,
)
from ._artifact_anchors import check_manifest_metadata_anchors, persisted_anchor_facts
from ._durability import (
    _directories_content_equal,
    _restore_backup,
    _restrict_mode,
    _restrict_tree_modes,
    fsync_dir,
    fsync_tree,
)
from ._extra_files import _warn_extra_files
from ._load_validators import MODEL_FREE_TLSPEC_KINDS, load_model_free_tlspec
from ._pickle_prescan import OpcodeCeilingExceeded
from ._provenance_git import _git_commit_hash
from ._root_manifest import NOT_ROUTED, load_by_root_manifest
from ._safe_unpickle import SafeBundleUnpickler
from ._save_gates import (
    _RUNNABLE_ACTIVATION_KIND,
    _RUNNABLE_NONPERSISTENT_BUFFER_KIND,
    _RUNNABLE_WEIGHT_KIND,
    _refuse_edge_intervened_save,
    _refuse_shard_local_erasure,
    _runnable_write_context,
    _save_bundle_via_container_door,
    _stamped_descriptor,
)
from ._save_subject import resolve_save_subject
from ._skipped_blobs import _apply_skipped_blobs_to_scrubbed_state
from ._staging import (
    PARTIAL_SENTINEL,
    mark_partial as _mark_partial,
    refuse_stale_load,
    sibling_staging_dirs,
)
from ._tlspec_identity import _preflight_schema_v2_runtime
from .format3 import (
    contract as format3_contract,
    reader as format3_reader,
    runnable as format3_runnable,
)
from .format3.payloads import (
    NUMPY_CODEC,
    TORCH_CODEC,
    PayloadStore,
    host_tensor,
    seal_tensor_entries,
    submit_tensor_payload,
    tensor_reference,
)
from .format3.recipe import write_trace_recipe
from .format3.restore import (
    _stamp_loaded_context,
    loaded_persistent_buffer_universe,
    true_up_record,
)
from .format3.route import Format3LoadRequest, is_format3_root, load_format3
from .format3.writer import (
    EnvelopeSpec,
    TraceTables,
    preflight_reload,
    settle_live_side,
    write_trace_envelope,
)
from .format_errors import DENSE_REMEDY, OpaqueCallableError
from .lazy import LazyActivationRef, _read_materialization_manifest
from .manifest import Manifest, Provenance, TensorEntry, enforce_version_policy
from .paths import (
    _make_tmp_bundle_path,
    reanchor_visualizer_paths as _reanchor_visualizer_paths,
    reject_symlink_path as _reject_symlink_path,
    resolve_bundle_blob_path,
    resolve_bundle_blobs_dir,
)
from .payload_codec import (
    PayloadCodec,
    get_payload_codec,
    numpy_to_transport_tensor,
)
from .prerelease import effective_policy
from .rehydrate import rehydrate_trace
from .runnable_load import attach_sparse_run_readiness
from .save_policy import _refuse_incoherent_structure_only_write, _resolve_include_flags
from .scrub import BlobSpec, scrub_for_save
from .state_keys import invalidate_static_class_attr_cache
from .tensor_policy import Ok
from .tlspec import _TlSpecWriter, coerce_tlspec_save_level

if TYPE_CHECKING:
    from ..bundle import Bundle
    from ..intervention._spec import Recipe
    from ..intervention._types import InterventionSpec

# Coarse anti-DoS ceiling on ``metadata.pkl``, for parity with the JSON
# boundary's byte ceiling (512 MiB, ``_json._MAX_JSON_BYTES``). Structural trace
# metadata (tensor payloads live in separate safetensors blobs) never approaches
# this, so the cap only refuses an absurd artifact; it is deliberately generous
# to avoid refusing a real save.
_MAX_METADATA_PKL_BYTES = 512 * 1024**2

# THE single authority for the load-provenance transient family: every
# attribute _io/bundle attaches to a LOADED trace that must never persist into
# a re-save. One conceptual family used to have four declaration stories and a
# function-local strip tuple as its only complete inventory; the strip pass in
# ``scrub_for_save``'s caller and the exemption ledger in
# ``data_classes/_trace_components.py`` both key off this constant's members.
LOAD_PROVENANCE_TRANSIENT_ATTRS: tuple[str, ...] = (
    "_loaded_from_bundle",
    "_source_bundle_manifest_sha256",
    "_source_bundle_path",
    "_source_bundle_created_at",
    "_source_bundle_provenance",
    "_source_bundle_model_fingerprint",
    "payload_load_status",
    "_validation_replay_status",
)
# Object-count ceiling for ``metadata.pkl``: the byte cap alone does not
# bound allocation -- a pickle of tiny values expands its byte size into RSS
# BEFORE any structural check can refuse it. This is the same lesson the JSON
# boundary's ``_MAX_JSON_NODES`` prescan already encodes, carried to the sibling
# pickle boundary. The unpickler counts opcodes with a tight byte reader (no object
# allocation) with an early stop, so its own worst case is bounded CPU, never
# unbounded memory.
#
# CALIBRATION: the original 64M value was derived
# from a ~5x bytes->RSS expansion assumption, but the measured worst case for
# dict-of-tiny-dict payloads is 22-26x -- about 70 BYTES OF RSS PER OPCODE --
# so a within-ceiling hostile artifact still projected to ~2.8 GiB of RSS and
# minutes of CPU inside ``tl.load()``. The ceiling is now derived from that
# measured per-opcode cost against a ~1.6 GiB worst-case allocation budget:
# 24M opcodes x ~70 B/opcode ~= 1.6 GiB, with in-ceiling wall time under a
# minute. Still far above any honest save: structural metadata at this opcode
# count would be a multi-hundred-MB pickle, an order of magnitude beyond the
# largest real traces (tensor payloads live in separate safetensors blobs).
_MAX_METADATA_PKL_OPCODES = 24_000_000
# Count opcodes only on files large enough for the ceiling to be reachable: below this
# the count would tax every real load for a ceiling no stream that size can hit.
_METADATA_PKL_PRESCAN_BYTES = 8 * 1024**2
_BLOB_TENSOR_KEY = "data"


class _RenameAwareUnpickler(SafeBundleUnpickler):
    """Restricted unpickler for untrusted portable bundle metadata.

    A loaded ``.tlspec`` bundle is UNTRUSTED input, so ``metadata.pkl`` is read
    with the default-deny :class:`SafeBundleUnpickler` class allowlist (which
    closes a load-time ``__reduce__`` / ``os.system`` RCE). A global that only a pre-2.38 release
    wrote refuses with the one pre-2.38 error; nothing is renamed on the way in (the class name
    predates that and stays).
    """

    def __init__(
        self,
        file: Any,
        *,
        trust_custom_callables: bool = False,
        allowed_custom_callable_modules: Collection[str] | None = None,
        max_opcodes: int | None = None,
        count_opcodes_above: int = 0,
    ) -> None:
        """Initialize the restricted unpickler.

        Parameters
        ----------
        file:
            Binary file object positioned at the start of a pickle stream.
        trust_custom_callables:
            Whether to import+resolve foreign custom callables (default deny).
        allowed_custom_callable_modules:
            Optional narrow allowlist of foreign modules whose callables may load.
        max_opcodes:
            Opcode-count ceiling enforced before the load (``None`` leaves it off).
        count_opcodes_above:
            Byte size above which the ceiling is counted (below it, unreachable).
        """

        super().__init__(
            file,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
            max_opcodes=max_opcodes,
            count_opcodes_above=count_opcodes_above,
        )

    def load(self) -> Any:
        """Unpickle inside the governed-artifact-load contract window.

        Nested record ``__setstate__`` calls run within this frame, so the
        unknown-field partition (state contract, MEMO 3.3) arms here: governed
        artifact bytes refuse unknown fields typed, while plain session
        pickling of live records (which legitimately carries user extras the
        save path refuses to persist) stays outside the contract.
        """

        from .state_contract import governed_artifact_load

        with governed_artifact_load():
            return super().load()


@dataclass(frozen=True)
class _FastCopySpec:
    """One lazily-backed direct tensor field that can be copied into a new bundle.

    Parameters
    ----------
    blob_id:
        New blob id allocated for the destination bundle.
    kind:
        Logical tensor kind recorded in the destination manifest.
    label:
        Human-readable label stored alongside the destination manifest entry.
    source_ref:
        Lazy source blob reference from the current in-memory ``Trace``.
    """

    blob_id: str
    kind: str
    label: str
    source_ref: LazyActivationRef


def save(
    trace: Trace,
    path: str | Path,
    *,
    level: str = "portable",
    include_outs: bool | None = None,
    include_grads: bool | None = None,
    include_saved_args: bool | None = None,
    include_rng_states: bool | None = None,
    include_weights: bool = False,
    include_activations: bool = False,
    include_source: bool = True,
    include_custom_attributes: bool = True,
    include_buffer_values: bool | None = None,
    strict: bool = True,
    overwrite: bool = False,
) -> None:
    """Persist a ``Trace`` into a portable TorchLens directory bundle.

    Parameters
    ----------
    trace:
        Completed model log to save. A ``Bundle`` delegates to the container
        door (``Bundle.save``): members gate and persist individually, and
        per-trace payload options refuse typed
        (``bundle_save_option_unsupported``).
    path:
        Output bundle directory path.
    level:
        Public ``.tlspec`` save level: ``"audit"``,
        ``"executable_with_callables"``, ``"portable"``, or ``"runnable"``.
    include_outs:
        Whether outs should be saved as blobs. Omitted (``None``) resolves to
        the level default (``True`` except at the payload-free ``audit`` /
        ``runnable`` levels); an EXPLICIT value contradicting a level
        requirement refuses typed (``save_payload_level_conflict``) instead
        of being silently overridden.
    include_grads:
        Whether grads should be saved as blobs. Same omitted/explicit
        resolution as ``include_outs``.
    include_saved_args:
        Whether captured args/kwargs and related tensor payloads should be
        saved. Omitted resolves to ``False`` except at
        ``level="executable_with_callables"``, which REQUIRES these payloads
        to re-execute and refuses an explicit ``False`` (historically it
        silently shipped raw input tensors against that explicit opt-out).
    include_rng_states:
        Whether per-layer RNG state tensors should be saved. Same
        omitted/explicit resolution as ``include_saved_args``.
    include_weights:
        Whether a runnable save should bundle the full capture-time
        ``state_dict``: all named parameters and persistent buffers. This
        state-only payload contains no model object, gradients, RNG state,
        callables, or per-call snapshots. Valid only with ``level="runnable"``.
    include_activations:
        Whether a runnable save should archive exactly the ``save=``-selected
        ``out``/``transformed_out`` payloads for offline inspection and eligible
        original-input, real-state numeric attestation. The payloads never seed execution.
    include_source:
        Whether the captured model source code is embedded in the bundle
        (default ``True``). The source blob powers the ``draw(code_panel=...)``
        source panels, so it is kept by default. A ``.tlspec`` is the portable,
        shareable format, so this defaults ``True`` embeds the model's verbatim
        class / ``__init__`` / ``forward`` source, per-call ``code_context``
        source lines, and captured docstrings. Set ``include_source=False`` to
        strip all of that from a shared bundle; source panels on the reloaded
        trace then degrade to a "source not embedded" placeholder instead of
        rendering code. Regardless of this flag, absolute source paths
        (``$HOME``, OS username, site-packages / capturing-script layout) are
        always reduced to a bare basename, so no host filesystem PII is ever
        embedded. Applies at every save ``level``.
    include_custom_attributes:
        Whether harvested public module instance attributes
        (``Module.custom_attributes`` — every public, non-callable attribute a
        module instance holds, captured verbatim at prep time) are persisted
        (default ``True``, the historical behavior). These are arbitrary user
        values, so tokens, paths, or config blobs stored as module attributes
        ride along in a shared bundle; set ``False`` to drop the whole channel.
        Values are never rewritten or partially scrubbed. Every save discloses
        the channel in ``manifest.json`` under ``custom_attributes_disclosure``
        (module count + top-level key names). Sparse runnable cores always drop
        the field regardless of this flag.
    include_buffer_values:
        Whether captured pre-forward buffer values are persisted (omitted
        resolves to ``True``, the historical behavior). When a forward pass
        overwrites a registered buffer (BatchNorm running statistics, step
        counters, caches), TorchLens records the value the buffer held BEFORE
        the forward in ``Trace._buffer_initial_values``, and every save level —
        audit included — shipped those tensors verbatim with no opt-out.
        Buffer values are training-data-derived state, so set ``False`` to
        drop the whole channel from the artifact; values are never rewritten
        or partially scrubbed. Every save discloses the channel in
        ``manifest.json`` under ``buffer_values_disclosure`` (buffer count +
        buffer names). Sparse runnable cores always drop the field regardless
        of this flag (used non-persistent buffers ship separately there as the
        REQUIRED, independently disclosed ``runnable_nonpersistent_buffer_v1``
        family). Structure-only saves resolve the omitted default to
        ``False`` (a weights-free artifact must not carry
        training-derived buffer values) and refuse an explicit ``True``.
    strict:
        Whether unsupported tensors should abort the save instead of being skipped.
    overwrite:
        Whether an existing bundle at ``path`` may be replaced.

    Raises
    ------
    ArtifactError
        If the bundle cannot be created or contains unsupported state.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> import torchlens as tl
    >>> model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    >>> x = torch.randn(2, 4)
    >>> trace = tl.trace(model, x)
    >>> trace.save("demo_bundle", overwrite=True)
    >>> loaded = tl.load("demo_bundle")
    >>> loaded["linear_1_1"].out.shape
    torch.Size([2, 3])

    Warnings
    --------
    Portable bundles contain a pickle file, but the default load path decodes
    it through a restricted, default-deny unpickler: only allowlisted
    constructors resolve, and foreign ``custom`` callable modules are never
    imported unless explicitly trusted via ``trust_custom_callables=True`` or
    ``allowed_custom_callable_modules=...``. The exposure is that residual
    trust model, not ordinary arbitrary-code execution: opting into trusted
    custom-callable execution imports and runs module top-level code, so only
    grant it to artifacts whose provenance you trust.
    """

    # Presenters refuse typed (MergedTrace, TraceSlice) and a Recording cooks
    # into the Trace it stands for; see ``torchlens._io._save_subject``.
    trace = resolve_save_subject(trace)
    # A Bundle is a CONTAINER product, never a Trace: applying the per-trace
    # N1 gate below to it read the container's empty outcome sidecar as
    # UNKNOWN and false-refused EVERY bundle tl.Bundle() built (with the
    # hand-built-object warning) while bundle.save() worked -- the third
    # instance of the two-answers outcome-authority disease.
    # Trace.save delegates to the ONE container door before any per-trace gate;
    # each member's own save applies the N1 gate PER MEMBER inside the
    # container write (a container-level status is a fold reported
    # beside results, never a gate on a member it does not describe). Same
    # sys.modules shape as the presenter refusals above: a Bundle can only
    # exist after its module was imported. This DELEGATION precedes the
    # pinned save-entry refusal order without reordering it -- the branches
    # dispatch on disjoint types.
    bundle_module = sys.modules.get("torchlens.bundle")
    if bundle_module is not None and isinstance(trace, bundle_module.Bundle):
        _save_bundle_via_container_door(
            trace,
            path,
            level=level,
            overwrite=overwrite,
            trace_only_options={
                "include_outs": include_outs,
                "include_grads": include_grads,
                "include_saved_args": include_saved_args,
                "include_rng_states": include_rng_states,
                "include_weights": include_weights,
                "include_activations": include_activations,
                "include_source": include_source,
                "include_custom_attributes": include_custom_attributes,
                "include_buffer_values": include_buffer_values,
                "strict": strict,
            },
        )
        return
    # N1: the settled capture outcome gates every export. FAILED, aborted, and
    # UNKNOWN captures never produce a portable artifact (the historical
    # ungated pass-through of failed partials was the hole this closes);
    # HALTED and legacy UNATTESTED re-export stay open. This runs BEFORE the
    # poison gate because a failed PartialTrace / failed Recording has no
    # ``_runnable`` sparse-run state, so refuse_poisoned_trace crashed with a
    # bare AttributeError before this typed N1 refusal could fire -- the exact
    # shape the MergedTrace typed-refusal-first check above already closed.
    from .._capture.outcome import CaptureOutcomeError, require_capture_capability

    require_capture_capability(trace, "save_analysis")
    # A structure-only trace refuses persistence wholesale until the
    # wave-3 coordinated bump lands the marker + load-validation rows (the
    # registrar's test-only activation switch is the one sanctioned
    # exit-gate round-trip; every switch-on write stamps the fail-closed
    # pre-release marker).
    from .._capture.structure_only import require_structure_only_capability

    require_structure_only_capability(trace, "save_analysis_artifact")
    # THE v7 PERSISTENCE BOUNDARY for edge-intervened traces,
    # keyed on TWO conjuncts — tier-(ii) edge-substitution entries PRESENT
    # AND the pre-release switch INACTIVE (the state in which the gated
    # fields scrub DROP, i.e. every state v7 production code can reach). No
    # occurrence-granular carrier survives a v7 save at ANY level, so an
    # edge-intervened trace is SESSION-ONLY in production until the wave-3
    # coordinated bump flips the registered rows. Runs at the save entry
    # AHEAD of level dispatch/validation, so on trace-carrying artifacts
    # this refusal PRECEDES artifact_save_level_unsupported. SAVE-ENTRY
    # REFUSAL ORDER (owners disjoint, all fail closed — do not silently
    # reorder another owner's refusal): (1) MergedTrace presenter refusal,
    # (2) capture-outcome gate, (3) structure-only chokepoint,
    # (4) this edge boundary.
    _refuse_edge_intervened_save(trace)
    # (5) shard-local erasure prevention: appended AFTER the pinned
    # four-refusal order above (owners disjoint; nothing reordered). Same
    # two-conjunct key shape as the edge boundary.
    _refuse_shard_local_erasure(trace)
    # (6) Injections: the injected family persists at analysis
    # level (synthesized op rows appended by injection_codec below); the
    # RUNNABLE sparse core cannot carry it, so a runnable save of a logged
    # trace refuses typed rather than silently dropping recorded
    # computation. Appended after (5); owners disjoint, nothing reordered.
    # The records-presence guard keeps level coercion at its historical
    # raise site for every unlogged trace (refusal order unchanged there).
    from ..intervention import _injection

    if (_injection.peek_injection_state(trace) or {}).get("records"):
        _injection.refuse_injection_logged_runnable_save(trace, coerce_tlspec_save_level(level))
    # A PartialTrace is a failed-capture inspection wrapper, never a savable
    # product (its _FIELD_POLICY declares both fields session-time DROP). Every
    # SHIPPED wrapper settles FAILED and refuses through the gate above; this
    # closes the hand-built wrapper-around-a-settled-trace residual, which
    # before the outcome delegation fail-closed only by ACCIDENT
    # (the gate read the wrapper's empty ``__dict__`` as UNKNOWN) and would
    # now otherwise crash untyped deeper in the save machinery. Same
    # sys.modules shape as the MergedTrace refusal: a PartialTrace can only
    # exist after its module was imported.
    partial_module = sys.modules.get("torchlens._partial")
    if partial_module is not None and isinstance(trace, partial_module.PartialTrace):
        from .._capture.outcome import outcome_for

        settled = outcome_for(trace)
        raise CaptureOutcomeError(
            "trace.save() does not support PartialTrace wrappers: a partial "
            "capture is an inspection product with no persistable schema. "
            "Inspect it live (draw, audit, first_nonfinite), or re-capture "
            "successfully and save the resulting Trace.",
            code="N1",
            capability="save_analysis",
            status="unknown" if settled is None else settled.status.value,
        )
    from .._runnable import refuse_poisoned_trace

    # The poison gate only applies to products that carry sparse-run state; a
    # PartialTrace / Recording has none, so there is nothing to refuse.
    if getattr(trace, "_runnable", None) is not None:
        refuse_poisoned_trace(trace, "export")
    save_level = coerce_tlspec_save_level(level)
    if save_level == "runnable":
        require_structure_only_capability(trace, "save_runnable")
        # N4: a halted capture records a PREFIX of the forward, and the sparse
        # runnable contract requires the complete taken path. Refusal surfaces
        # through the runnable error vocabulary.
        try:
            require_capture_capability(trace, "save_runnable")
        except CaptureOutcomeError as outcome_exc:
            if outcome_exc.fields.get("code") == "N4":
                from .._runnable import RunnableErrorCode

                raise RunnablePreflightError(
                    "Runnable save refuses a HALTED capture: the recorded graph "
                    "is an honest prefix of the forward, not the complete taken "
                    "path the sparse runnable contract requires. Re-capture "
                    "without stop_after= to produce a runnable artifact.",
                    code=RunnableErrorCode.HALTED_CAPTURE_NOT_RUNNABLE.value,
                ) from outcome_exc
            raise
    (
        include_outs,
        include_grads,
        include_saved_args,
        include_rng_states,
        include_buffer_values,
    ) = _resolve_include_flags(
        trace,
        save_level=save_level,
        include_outs=include_outs,
        include_grads=include_grads,
        include_saved_args=include_saved_args,
        include_rng_states=include_rng_states,
        include_buffer_values=include_buffer_values,
    )
    sparse_run_descriptor = None
    weight_blob_specs: list[BlobSpec] = []
    nonpersistent_buffer_blob_specs: list[BlobSpec] = []
    activation_blob_specs: list[BlobSpec] = []
    if include_weights and save_level != "runnable":
        raise InvalidArgumentError(
            f"include_weights=True cannot be used with save level {save_level!r}",
            code="save_payload_level_conflict",
            remedy="set level='runnable' or set include_weights=False",
            arguments=("include_weights", "level"),
        )
    if include_activations and save_level != "runnable":
        raise InvalidArgumentError(
            f"include_activations=True cannot be used with save level {save_level!r}",
            code="save_payload_level_conflict",
            remedy="set level='runnable' or set include_activations=False",
            arguments=("include_activations", "level"),
        )
    if save_level == "runnable":
        (
            sparse_run_descriptor,
            nonpersistent_buffer_blob_specs,
            weight_blob_specs,
            activation_blob_specs,
        ) = _stamped_descriptor(
            trace, include_weights=include_weights, include_activations=include_activations
        )
    backend_name = str(trace._provenance_backend if is_live_trace(trace) else "torch")
    backend_spec = get_backend_spec(backend_name)
    _reject_audit_only_materialized_payload_save(
        backend_name=backend_name,
        backend_spec=backend_spec,
        save_level=save_level,
        include_outs=include_outs,
        include_grads=include_grads,
        include_saved_args=include_saved_args,
        include_rng_states=include_rng_states,
    )

    bundle_path = Path(path)
    _reject_symlink_path(bundle_path, context="save target")
    _validate_activation_transform_outputs(trace, include_outs=include_outs)

    backup_path: Path | None = None
    tmp_path = _make_tmp_bundle_path(bundle_path)
    try:
        if bundle_path.exists() and not overwrite:
            refuse_bundle_exists(bundle_path, overwritable=True)

        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.mkdir()
        (tmp_path / "blobs").mkdir()
        # Bundle directories are 0700 from the start (no second principal can swap
        # a blob in a group-writable dir); the written files are 0600 below.
        _restrict_mode(tmp_path, 0o700)
        _restrict_mode(tmp_path / "blobs", 0o700)

        settle_live_side(trace)
        record_pairs: list[tuple[Any, Any]] = []
        scrubbed_state, blob_specs, scrub_unsupported_tensors = _scrub_trace_for_bundle(
            trace,
            record_pairs=record_pairs,
            include_outs=include_outs,
            include_grads=include_grads,
            include_saved_args=include_saved_args,
            include_rng_states=include_rng_states,
            include_source=include_source,
            include_custom_attributes=include_custom_attributes,
            include_buffer_values=include_buffer_values,
            sparse_runnable=sparse_run_descriptor is not None,
        )
        if sparse_run_descriptor is not None:
            from .runnable import assert_sparse_core_has_no_tensor_payload

            scrubbed_state["_buffer_initial_values"] = {}
            assert_sparse_core_has_no_tensor_payload(scrubbed_state)
            if blob_specs:
                raise ArtifactError(
                    "Sparse runnable scrub produced tensor blob specs; refusing runnable label."
                )
        _refuse_incoherent_structure_only_write(scrubbed_state)
        _apply_visualization_save_policy(
            trace,
            scrubbed_state=scrubbed_state,
            bundle_path=bundle_path,
            tmp_path=tmp_path,
        )
        _raise_for_unmaterialized_nested_blob_refs(
            scrubbed_state,
            allowed_blob_ids={blob_spec.blob_id for blob_spec in blob_specs},
        )
        fast_copy_specs = _attach_fast_copy_specs(
            trace,
            scrubbed_state=scrubbed_state,
            blob_specs=blob_specs,
            include_outs=include_outs,
            include_grads=include_grads,
        )
        # append one synthesized op row per recorded injected op (the
        # injection_provenance slot verbatim) AFTER every 1:1
        # live/scrubbed pairing consumer and BEFORE the blob-write loop, so
        # injected payload specs write like any other blob.
        if sparse_run_descriptor is None:
            from .injection_codec import append_injected_op_rows

            append_injected_op_rows(
                trace,
                scrubbed_state,
                blob_specs,
                include_outs=include_outs,
                backend_name=backend_name,
            )
        if not backend_spec.capabilities.payload_materialization and (
            blob_specs or fast_copy_specs
        ):
            raise BackendPayloadUnsupportedError(
                f"Backend {backend_name!r} .tlspec payloads are audit-only in this runtime. "
                "The scrubbed trace still contains materialized tensor blobs, so it cannot "
                "be saved without a backend payload codec."
            )

        payload_counter = [0]
        payload_store = PayloadStore(tmp_path, counter=payload_counter)
        tensor_refs: dict[str, dict[str, Any]] = {}
        tensor_entries: list[TensorEntry] = []
        unsupported_tensors: list[dict[str, str]] = list(scrub_unsupported_tensors)
        skipped_blob_ids: set[str] = set()

        for blob_spec in blob_specs:
            codec = get_payload_codec(blob_spec.logical_backend)
            decision = codec.validate_for_save(blob_spec.value, strict=True)
            if isinstance(decision, Ok):
                tensor_entries.append(
                    _write_payload_blob(
                        store=payload_store,
                        tensor_refs=tensor_refs,
                        blob_spec=blob_spec,
                        codec=codec,
                    )
                )
                continue
            error_text = (
                "Unsupported tensor for bundle save at "
                f"{blob_spec.label} ({blob_spec.kind}): {decision.text}"
            )
            if blob_spec.logical_backend != "torch":
                raise BackendPayloadUnsupportedError(error_text, code="metadata_value_unportable")
            raise ArtifactError(error_text, code="metadata_value_unportable", remedy=DENSE_REMEDY)

        source_manifest_cache: dict[Path, dict[str, TensorEntry]] = {}
        for fast_copy_spec in fast_copy_specs:
            manifest_index = _load_and_verify_fast_copy_source(
                trace,
                fast_copy_spec.source_ref.source_bundle_path,
                cache=source_manifest_cache,
            )
            tensor_entries.append(
                _fast_copy_tensor_blob(
                    store=payload_store,
                    tensor_refs=tensor_refs,
                    fast_copy_spec=fast_copy_spec,
                    manifest_index=manifest_index,
                )
            )

        # io.1: a staged intervention spec persists as intervention.json (its
        # tensors join this artifact's payload store before it closes).
        intervention_entry, recipe_disclosures = write_trace_recipe(
            tmp_path, trace, store=payload_store, save_level=save_level
        )
        runnable_entry = None
        if sparse_run_descriptor is not None:
            runnable_entry = format3_runnable.write_runnable(
                tmp_path,
                context=_runnable_write_context(
                    (
                        sparse_run_descriptor,
                        nonpersistent_buffer_blob_specs,
                        weight_blob_specs,
                        activation_blob_specs,
                    ),
                    payload_counter,
                ),
            )
            payload_store.writer(TORCH_CODEC)
        payload_store.close()
        tensor_entries = seal_tensor_entries(tmp_path, tensor_entries)

        if skipped_blob_ids:
            _apply_skipped_blobs_to_scrubbed_state(scrubbed_state, skipped_blob_ids)

        persisted_rows = scrubbed_state.get("_layer_list")
        manifest = _build_manifest(
            trace=trace,
            tensor_entries=tensor_entries,
            unsupported_tensors=unsupported_tensors,
            include_source=include_source,
            disclosures=_SaveDisclosures(
                custom_attributes=(
                    custom_attributes_disclosure := _custom_attributes_disclosure(
                        trace,
                        included=include_custom_attributes and sparse_run_descriptor is None,
                    )
                ),
                buffer_values=(
                    buffer_values_disclosure := _buffer_values_disclosure(
                        trace,
                        included=include_buffer_values and sparse_run_descriptor is None,
                    )
                ),
                persisted_op_rows=(
                    len(persisted_rows) if isinstance(persisted_rows, list) else None
                ),
            ),
        )
        _warn_custom_attribute_embedding(custom_attributes_disclosure)
        _warn_buffer_value_embedding(buffer_values_disclosure)
        # 3.0 format: typed JSON record tables beside the safetensors blobs, no
        # pickle at any level; the staged directory is reloaded once before publish.
        base_manifest = _TlSpecWriter.build_trace_manifest(
            trace=trace,
            legacy_manifest=manifest,
            save_level=save_level,
            scrubbed_state=scrubbed_state,
        )
        if intervention_entry is not None:
            base_manifest["intervention"] = intervention_entry
        if runnable_entry is not None:
            base_manifest["runnable"] = runnable_entry.manifest
        tables = TraceTables(
            scrubbed_state,
            tensor_refs,
            record_pairs=[*record_pairs, (scrubbed_state, trace)],
            include_source=bool(include_source),
        )
        envelope = EnvelopeSpec(
            "trace",
            save_level,
            payload_policy={
                "include_outs": bool(include_outs),
                "include_grads": bool(include_grads),
                "include_saved_args": bool(include_saved_args),
                "include_rng_states": bool(include_rng_states),
                "include_buffer_values": bool(include_buffer_values),
                "include_custom_attributes": bool(include_custom_attributes),
                "structure_only": scrubbed_state.get("_options_structure_only") is True,
            },
            codecs=payload_store.codecs,
            disclosures=recipe_disclosures,
        )
        write_trace_envelope(
            tmp_path, base_manifest=base_manifest, tables=tables, envelope=envelope
        )
        _restrict_tree_modes(tmp_path)
        preflight_reload(tmp_path)

        # Durability before publish: fsync every written blob/sidecar and the
        # bundle directories so a power/OS crash after the rename below cannot
        # publish a bundle holding zero-length or partial files.
        fsync_tree(tmp_path)
        # Move the existing target aside only now, immediately before the swap
        # (as ``_TlSpecWriter.write_bundle`` does), so readers never see it vanish.
        if bundle_path.exists():
            if not overwrite:
                refuse_bundle_exists(bundle_path, overwritable=True)
            backup_path = _make_backup_path(bundle_path)
            bundle_path.rename(backup_path)
        tmp_path.rename(bundle_path)
        # Make the rename itself durable before declaring the save complete.
        fsync_dir(bundle_path.parent)
        if backup_path is not None:
            try:
                _remove_path(backup_path)
            except OSError:
                # The replacement is already installed atomically. A stale backup
                # is recoverable cleanup debris, not a failed save.
                pass
    except (ArtifactError, OpaqueCallableError) as exc:
        _run_save_recovery(tmp_path, backup_path, bundle_path, primary=exc)
        raise
    except BackendPayloadUnsupportedError as exc:
        _run_save_recovery(tmp_path, backup_path, bundle_path, primary=exc)
        raise
    except (ImportError, OSError, TypeError, ValueError, pickle.PickleError) as exc:
        # ``TypeError`` is here because serializers raise it bare for live
        # resources; every branch marks the staging dir PARTIAL and restores.
        _run_save_recovery(tmp_path, backup_path, bundle_path, primary=exc)
        raise ArtifactError(
            f"Failed to save bundle at {bundle_path}: {type(exc).__name__}: {exc}. "
            "Remedy: the staging directory was marked PARTIAL (sweepable by "
            "cleanup) and any pre-overwrite bundle was restored; fix the named "
            "cause and re-save.",
            code="bundle_save_failed",
            cause_type=type(exc).__name__,
        ) from exc
    except BaseException as exc:
        # Catch-all recovery contract: for any exception, known or not, mark the
        # staging dir PARTIAL (sweepable by ``cleanup_tmp()``) and restore the
        # pre-overwrite backup. ``BaseException`` also covers interrupts and
        # exits, which re-raise unwrapped; ordinary exceptions are wrapped.
        _run_save_recovery(tmp_path, backup_path, bundle_path, primary=exc)
        if isinstance(exc, Exception):
            raise ArtifactError(
                f"Failed to save bundle at {bundle_path}: {type(exc).__name__}: {exc}. "
                "Remedy: the staging directory was marked PARTIAL (sweepable by "
                "cleanup) and any pre-overwrite bundle was restored; fix the named "
                "cause and re-save.",
                code="bundle_save_failed",
                cause_type=type(exc).__name__,
            ) from exc
        raise


def _reject_audit_only_materialized_payload_save(
    *,
    backend_name: str,
    backend_spec: BackendSpec,
    save_level: str,
    include_outs: bool,
    include_grads: bool,
    include_saved_args: bool,
    include_rng_states: bool,
) -> None:
    """Reject materialized payload saves before audit-only scrub blobification.

    Parameters
    ----------
    backend_name:
        Backend identifier recorded on the trace.
    backend_spec:
        Registered backend capability spec.
    save_level:
        Public ``.tlspec`` save level after coercion.
    include_outs:
        Whether activation outputs would be materialized.
    include_grads:
        Whether gradient outputs would be materialized.
    include_saved_args:
        Whether captured args/kwargs would be materialized.
    include_rng_states:
        Whether RNG states would be materialized.

    Raises
    ------
    BackendPayloadUnsupportedError
        If the requested save would require payload materialization for a
        backend whose public serialization contract is audit-only.
    """

    payload_requested = include_outs or include_grads or include_saved_args or include_rng_states
    if (
        backend_name == "mlx"
        or save_level == "audit"
        or backend_spec.capabilities.payload_materialization
        or not payload_requested
    ):
        return
    raise BackendPayloadUnsupportedError(
        f"Backend {backend_name!r} .tlspec payloads are audit-only in this runtime. "
        "Portable saves with materialized payloads require a backend payload codec; "
        "use level='audit' to save metadata without materialized payloads."
    )


def _load_216_intervention_spec(
    bundle_path: Path,
    tlspec_format: str,
    trust_custom_callables: bool,
    allowed_custom_callable_modules: Any,
) -> Any:
    """Load (or refuse typed) a directory the detector classifies as a 2.16 intervention spec."""

    from ..intervention._save import load_intervention_spec

    # Typed boundary: a trace manifest whose ``tlspec_version``
    # key was lost still carries ``kind``, so format detection
    # classifies it as a v2.16 intervention spec and the spec loader
    # then surfaced a RAW FileNotFoundError for the spec.json the
    # artifact never had. Incoherent format markers are a corrupt
    # artifact and must refuse typed at the dispatch.
    if bundle_path.is_dir() and not (bundle_path / "spec.json").exists():
        raise ArtifactError(
            f"Artifact at {bundle_path} classifies as {tlspec_format} "
            "(its manifest carries 'kind' without 'tlspec_version') but "
            "contains no spec.json — the format markers are incoherent, "
            "which means the manifest lost required keys or the "
            "artifact is corrupt. Remedy: restore the manifest's "
            "tlspec_version key or re-save the artifact from its "
            "source trace.",
            code="tlspec_format_markers_incoherent",
        )
    return load_intervention_spec(
        bundle_path,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )


def _attach_run_readiness(trace: Trace, sparse_run: Any) -> None:
    """Attach the sparse-run readiness report (the loader's one late import of runnable_load)."""

    attach_sparse_run_readiness(trace, sparse_run)


# ---------------------------------------------------------------------------
# 3.0 saved format: the rehydration side of the reader doors
# ---------------------------------------------------------------------------
# ``format3/reader.py`` sits below the record classes; the steps needing ``rehydrate_trace``,
# the accessors, ``Bundle`` and the intervention types live here and are installed into the
# reader's doors at import (``format3_reader.install_load_hooks`` below).


def _finish_format3_trace(
    artifact: format3_reader.DecodedArtifact,
    *,
    lazy: bool,
    map_location: str | torch.device,
    materialize_nested: bool,
    payload_hints: PayloadLoadHints | None,
) -> Trace:
    """Rehydrate one decoded 3.0 trace artifact: tensors, accessors, provenance, recipe."""

    root, manifest = artifact.root, artifact.manifest
    if (
        manifest.get("backend", format3_contract.DEFAULT_BACKEND)
        != format3_contract.DEFAULT_BACKEND
    ):
        # The runtime fingerprint stays fail-closed (a tf trace loads only on its TF release).
        _preflight_schema_v2_runtime(manifest, backend_name=manifest["backend"])
    state = artifact.trace_state(ModuleAccessor)
    has_blobs = (root / format3_contract.BLOBS_DIR).is_dir()
    blobs_dir = resolve_bundle_blobs_dir(root) if has_blobs else None
    trace = rehydrate_trace(
        state,
        artifact.manifest_for_rehydrate(),
        root,
        lazy=lazy,
        map_location=map_location,
        materialize_nested=materialize_nested,
        payload_hints=payload_hints,
        resolved_blobs_dir=blobs_dir,
        trace=artifact.objects["trace"][0],
    )
    # The rehydrate path re-runs the trace state hook; put back what its fill changed.
    true_up_record(trace, artifact.tables["trace"][0], "trace")
    reanchor_loaded_episode_evidence(trace)
    _reanchor_visualizer_paths(trace, root)
    _stamp_loaded_context(trace, manifest, root, Provenance.from_dict, sha256_file)
    # Contract section 7: the runnable component (descriptor and families under ``runnable/``,
    # listed in ``manifest["runnable"]``); an analysis artifact attaches analysis-only readiness.
    runnable_bind.attach_runnable(
        trace,
        root,
        manifest.get("runnable"),
        manifest.get("files") or {},
        map_location=map_location,
    )
    # A runnable resave declares its persistent-buffer slots from the capture-time universe;
    # without a live model the validated descriptor is that universe's source.
    if trace._runnable.persistent_buffer_universe is None:
        trace._runnable.persistent_buffer_universe = loaded_persistent_buffer_universe(
            trace._runnable.descriptor
        )
    if manifest.get("disclosures") is not None:
        format3_reader.attach_disclosure_advisory(trace, manifest["disclosures"])
    if manifest.get("intervention") is not None:
        trace._intervention_spec = _finish_format3_recipe(root, trace)
    return trace


def _finish_format3_bundle(
    members: list[tuple[str, Trace]],
    *,
    metadata: Mapping[str, Any],
    root: Path,
    tlspec_version: Any,
    unknown_relations: Literal["opaque", "refuse"] = "opaque",
) -> Bundle:
    """Assemble a 3.0 bundle from its members and ``relations.json`` (bare unknown keys refuse
    ``bundle_section_unknown``; ``unknown_relations`` is the policy for unknown relation kinds)."""

    from .bundle_container import assemble_bundle

    bundle = assemble_bundle(
        dict(members),
        dict(metadata),
        tlspec_version=tlspec_version,
        unknown_relations=unknown_relations,
    )
    setattr(bundle, "_source_bundle_path", root)
    return bundle


def _finish_format3_recipe(root: Path, trace: Trace | None = None) -> InterventionSpec:
    """Stage the recipe ``intervention.json`` describes (design section 8): the intervention
    package rebuilds built-in edits (facet writes rebound on ``trace``); a callable placeholder
    becomes an audit-only ``Edit`` listed in ``metadata["recipe_placeholders"]`` (rerun refuses)."""

    from ..intervention._save import _load_trace_recipe_spec

    return _load_trace_recipe_spec(root, trace)


format3_reader.install_load_hooks(
    trace=_finish_format3_trace, bundle=_finish_format3_bundle, recipe=_finish_format3_recipe
)
runnable_bind.install_runnable_binders(
    runnable_bind.RunnableBinders(
        attach_readiness=attach_sparse_run_readiness,
        bind_state=bind_embedded_trace_state,
        bind_buffers=bind_embedded_nonpersistent_buffers,
        byte_digest=runnable_tensor_byte_digest,
    )
)


@overload
def load(
    path: str | Path,
    *,
    lazy: Literal[False] = False,
    map_location: str | torch.device = "cpu",
    materialize_nested: bool = True,
    payload_hints: PayloadLoadHints | None = None,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
    unknown_relations: Literal["opaque", "refuse"] = "opaque",
    _bundle_visited: frozenset[Path] | None = None,
) -> Trace | Bundle | InterventionSpec | Recipe:
    """Load a ``.tlspec`` object with eager tensor materialization.

    Parameters
    ----------
    path:
        ``.tlspec`` directory path.
    lazy:
        Eager-loading overload marker.
    map_location:
        Target device for eager tensor materialization.
    materialize_nested:
        Whether nested blob refs should be materialized.
    payload_hints:
        Optional backend payload hints used during materialization.

    Returns
    -------
    Trace | Bundle | InterventionSpec | Recipe
        Rehydrated object selected by the bundle manifest.
    """
    ...


@overload
def load(
    path: str | Path,
    *,
    lazy: Literal[True],
    map_location: str | torch.device = "cpu",
    materialize_nested: bool = True,
    payload_hints: PayloadLoadHints | None = None,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
    unknown_relations: Literal["opaque", "refuse"] = "opaque",
    _bundle_visited: frozenset[Path] | None = None,
) -> Trace | Bundle | InterventionSpec | Recipe:
    """Load a ``.tlspec`` object while leaving direct tensors lazy.

    Parameters
    ----------
    path:
        ``.tlspec`` directory path.
    lazy:
        Lazy-loading overload marker.
    map_location:
        Target device for deferred tensor materialization.
    materialize_nested:
        Whether nested blob refs should be materialized.
    payload_hints:
        Optional backend payload hints used during materialization.

    Returns
    -------
    Trace | Bundle | InterventionSpec | Recipe
        Rehydrated object selected by the bundle manifest.
    """
    ...


#: Root files only a 2.x artifact carries; a directory holding one keeps the 2.x
#: path until that path is deleted. ``spec.json`` is the v2.16 intervention root.
#: Detector answers the 3.0 reader owns: the 3.0 format, the tlspec-stamped 2.x shape (it
#: refuses by stamp or root listing there) and manifests no 2.x family claims.


def load(
    path: str | Path,
    *,
    lazy: bool = False,
    map_location: str | torch.device = "cpu",
    materialize_nested: bool = True,
    payload_hints: PayloadLoadHints | None = None,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
    unknown_relations: Literal["opaque", "refuse"] = "opaque",
    _bundle_visited: frozenset[Path] | None = None,
) -> Trace | Bundle | InterventionSpec | Recipe:
    """Load a TorchLens ``.tlspec`` object polymorphically.

    Parameters
    ----------
    path:
        ``.tlspec`` directory path.
    lazy:
        Whether direct out/grad blobs should remain lazy placeholders. For a
        saved ``Bundle`` this, ``map_location`` and ``materialize_nested``
        reach every member.
    map_location:
        Target device for eager tensor materialization.
    materialize_nested:
        Whether nested blob refs in captured args and RNG states should be
        materialized when ``lazy=True``.
    payload_hints:
        Optional backend payload hints used during materialization. This is
        separate from ``map_location``; JAX sharding hints must be passed here.
    trust_custom_callables:
        Explicit permission to import custom callables from an intervention
        spec when no allowlist is supplied. Enable only for trusted specs.
    allowed_custom_callable_modules:
        Optional allowlist of custom callable module names. When supplied,
        custom imports must be listed even if ``trust_custom_callables=True``.
    unknown_relations:
        Bundle loader-doctrine policy for member-relation rows of unknown
        NAMESPACED kinds (the preserve-and-disclose amendment):
        ``"opaque"`` (default) loads them preserved, disclosed, unchecked,
        and never executed; ``"refuse"`` restores the strict typed refusal.
        Bare unknown kinds refuse under both policies.

    Returns
    -------
    Trace | Bundle | InterventionSpec | Recipe
        Rehydrated object selected by ``manifest.kind`` for unified files, or
        by legacy format detection for older files.

    Raises
    ------
    ArtifactError
        If the bundle is invalid, corrupt, or incompatible with this runtime.

    Warnings
    --------
    Portable bundles contain a pickle file, but the default load path decodes
    it through a restricted, default-deny unpickler: only allowlisted
    constructors resolve, and foreign ``custom`` callable modules are never
    imported unless explicitly trusted via ``trust_custom_callables=True`` or
    ``allowed_custom_callable_modules=...``. The exposure is that residual
    trust model, not ordinary arbitrary-code execution: opting into trusted
    custom-callable execution imports and runs module top-level code, so only
    grant it to artifacts whose provenance you trust.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> trace = tl.trace(model, x)
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = pathlib.Path(tmp) / "demo.tlspec"
    ...     trace.save(path)
    ...     loaded = tl.load(path)
    ...     same = loaded.layer_labels == trace.layer_labels
    >>> same
    True
    """

    # Load boundary: force every memoized class-owned static lookup to re-validate
    # its class-definition fingerprint before this artifact is decoded.
    invalidate_static_class_attr_cache()
    bundle_path = Path(path)
    refuse_stale_load(bundle_path, member=_bundle_visited is not None)
    if bundle_path.is_dir():
        # Version-gated merged-directory branch: the merged root manifest is a
        # different discriminated object from a trace bundle's manifest (it
        # carries no tensor table), so it routes before Manifest parsing.
        # Runtimes without this branch refuse the format typed at the closed
        # bundle_format vocabulary check.
        routed = load_by_root_manifest(bundle_path)
        if routed is not NOT_ROUTED:
            return routed
        from ..io import _pre3_observed, detect_tlspec_format
        from .format_contract import _guard_pre3_format

        tlspec_format = detect_tlspec_format(bundle_path)
        _guard_pre3_format(tlspec_format, bundle_path, lambda: _pre3_observed(bundle_path))
        if is_format3_root(bundle_path, tlspec_format):
            # 3.0 saved format: record tables plus safetensors chunks, never the pickle path.
            return load_format3(
                bundle_path,
                Format3LoadRequest(
                    lazy=lazy,
                    map_location=map_location,
                    materialize_nested=materialize_nested,
                    payload_hints=payload_hints,
                    unknown_relations=unknown_relations,
                ),
            )

    if bundle_path.is_dir() and (bundle_path / "spec.json").exists():
        from ..intervention._save import load_intervention_spec

        return load_intervention_spec(
            bundle_path,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )
    # Stable ``fields["code"]`` on the six distinct tl.load front-door causes so a caller can
    # branch on the cause instead of parsing one string: a symlinked load path, an unreadable
    # manifest, a non-object manifest root, a metadata-integrity refusal (tagged at the
    # unpickle boundary below), a below-floor version, and a generic bundle-load failure.
    _reject_symlink_path(bundle_path, context="bundle path", code="load_path_symlink_rejected")
    manifest_path = bundle_path / "manifest.json"
    metadata_path = bundle_path / "metadata.pkl"
    blobs_path = bundle_path / "blobs"
    _reject_symlink_path(manifest_path, context="manifest", code="load_path_symlink_rejected")
    _reject_symlink_path(metadata_path, context="metadata", code="load_path_symlink_rejected")
    _reject_symlink_path(blobs_path, context="blobs directory", code="load_path_symlink_rejected")

    try:
        manifest = Manifest.read(manifest_path)
    except ArtifactError:
        raise
    return _load_trace_payload(
        bundle_path,
        manifest,
        lazy=lazy,
        map_location=map_location,
        materialize_nested=materialize_nested,
        payload_hints=payload_hints,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )


def _load_trace_payload(
    bundle_path: Path,
    manifest: Manifest,
    *,
    lazy: bool,
    map_location: str | torch.device,
    materialize_nested: bool,
    payload_hints: PayloadLoadHints | None,
    sparse_run: Mapping[str, Any] | None = None,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> Trace | Bundle | InterventionSpec | Recipe:
    """Load a portable Trace payload after manifest dispatch.

    Parameters
    ----------
    bundle_path:
        Directory containing ``manifest.json``, ``metadata.pkl``, and blobs.
    manifest:
        Parsed portable manifest.
    lazy:
        Whether direct out/grad blobs should remain lazy placeholders.
    map_location:
        Target device for eager tensor materialization.
    materialize_nested:
        Whether nested blob refs should be materialized when ``lazy=True``.
    payload_hints:
        Optional backend payload hints used during materialization.
    sparse_run:
        Structurally validated public sparse descriptor, when present.

    Returns
    -------
    Trace
        Rehydrated model log.
    """

    manifest_path = bundle_path / "manifest.json"
    metadata_path = bundle_path / "metadata.pkl"
    blobs_path = bundle_path / "blobs"
    python_major_mismatch = False
    try:
        enforce_version_policy(manifest)
        resolved_blobs_dir = resolve_bundle_blobs_dir(bundle_path)
        _validate_manifest_blob_paths(manifest, bundle_path, resolved_blobs_dir)
        _check_unknown_blob_entries(manifest, blobs_path)
        if not lazy:
            _eager_verify_blob_payloads(
                manifest,
                bundle_path,
                map_location,
                resolved_blobs_dir,
            )

        python_major_mismatch = _python_major_mismatch(manifest)
        with metadata_path.open("rb") as handle:
            # Coarse allocation guard for parity with the JSON boundary:
            # every JSON read enforces a byte ceiling, but ``metadata.pkl`` had
            # none. This bounds an absurd on-disk pickle before it is streamed into
            # the unpickler; the SafeBundleUnpickler allowlist remains the actual
            # code-execution defense, and pickle expansion bombs are out of scope
            # for a file-size cap. The ceiling is generous so no real structural
            # metadata is refused.
            metadata_size = os.fstat(handle.fileno()).st_size
            if metadata_size > _MAX_METADATA_PKL_BYTES:
                raise ArtifactError(
                    f"Bundle metadata {metadata_path} is {metadata_size} bytes, above "
                    f"the {_MAX_METADATA_PKL_BYTES}-byte ceiling; refusing to load a "
                    "structurally implausible artifact."
                )
            scrubbed_state = _metadata_unpickler(
                handle,
                trust_custom_callables=trust_custom_callables,
                allowed_custom_callable_modules=allowed_custom_callable_modules,
            ).load()
        if not isinstance(scrubbed_state, dict):
            # Without this guard a corrupt/hostile payload escapes
            # tl.load() as a raw stdlib TypeError/ValueError from the downstream
            # dict() walk, naming a "dictionary update sequence element".
            raise ArtifactError(
                f"Bundle metadata at {metadata_path} is not a metadata mapping "
                f"(got {type(scrubbed_state).__name__}); the artifact is corrupt or "
                "hand-edited. Remedy: re-save the trace with trace.save().",
                code="metadata_payload_not_a_mapping",
            )
        # AUD-CODE 3.0b: snapshot the row count from the bytes as written;
        # rehydration mutates the persisted layer_list in place.
        persisted_rows = persisted_anchor_facts(scrubbed_state)
    except ArtifactError:
        raise
    except OpcodeCeilingExceeded:
        raise _opcode_ceiling_error(metadata_path, code="metadata_object_count_exceeded") from None
    except (pickle.UnpicklingError, EOFError) as exc:
        hint = ""
        if python_major_mismatch:
            hint = (
                f" Bundle was written with python_version={manifest.python_version} but runtime is "
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
            )
        # A guarded-unpickler denylist refusal or corrupt/truncated pickle stream
        # is a bundle-INTEGRITY signal, not environmental drift: it is tagged with
        # a stable code so a downstream loader (e.g. the merged-artifact rank-core
        # loader) can refuse it as tamper rather than laundering it into
        # the "no longer parses on this runtime" degradation channel.
        raise ArtifactError(
            f"Failed to load bundle metadata from {metadata_path}.{hint}",
            code="bundle_metadata_integrity_refused",
        ) from exc
    except (OSError, AttributeError, ImportError, TypeError, ValueError) as exc:
        # Generic bundle-load failure (torch/codec drift, missing dep): the sixth
        # tl.load front-door cause gets its own stable code.
        raise ArtifactError(
            f"Failed to load bundle at {bundle_path}.", code="bundle_load_failed"
        ) from exc

    try:
        trace = rehydrate_trace(
            scrubbed_state,
            manifest,
            bundle_path,
            lazy=lazy,
            map_location=map_location,
            materialize_nested=materialize_nested,
            payload_hints=payload_hints,
            resolved_blobs_dir=resolved_blobs_dir,
        )
    except RecursionError as exc:
        # A hostile scrubbed-state object graph deeply nested
        # enough to blow the C stack during the rehydrate walk must surface as a
        # typed artifact error, not an uncaught RecursionError that escapes
        # ``tl.load(path)``. Pairs with the load-walk depth ceiling.
        raise ArtifactError(
            f"Failed to load bundle at {bundle_path}: metadata nesting exceeded the "
            "interpreter recursion limit during rehydration."
        ) from exc
    # AUD-CODE 3.0b: the manifest's row count must agree with the pickled root
    # state. Runs AFTER rehydrate so the owner-specific persisted-claim refusals
    # (injection codec, site keys, graph structure, ...) keep precedence.
    check_manifest_metadata_anchors(manifest, persisted_rows, bundle_path)
    # Inside ``Trace.__setstate__`` the root
    # output was an unmaterialized blob handle, so the episode ledger's
    # evidence anchor had nothing to compare. Payloads are attached now; the
    # load itself refutes a ledger re-minted over another execution.
    reanchor_loaded_episode_evidence(trace)
    _reanchor_visualizer_paths(trace, bundle_path)
    setattr(trace, "_loaded_from_bundle", True)
    setattr(trace, "_source_bundle_manifest_sha256", sha256_file(manifest_path))
    setattr(trace, "_source_bundle_path", bundle_path)
    setattr(trace, "_source_bundle_created_at", manifest.created_at)
    setattr(trace, "_source_bundle_provenance", manifest.provenance)
    _attach_run_readiness(trace, sparse_run)
    _bind_embedded_nonpersistent_buffer_payload(
        trace,
        manifest=manifest,
        bundle_path=bundle_path,
        map_location=map_location,
        resolved_blobs_dir=resolved_blobs_dir,
    )
    _bind_embedded_weight_payload(
        trace,
        manifest=manifest,
        bundle_path=bundle_path,
        map_location=map_location,
        resolved_blobs_dir=resolved_blobs_dir,
    )
    _bind_archived_activation_payload(
        trace,
        manifest=manifest,
        bundle_path=bundle_path,
        map_location=map_location,
        resolved_blobs_dir=resolved_blobs_dir,
    )
    return trace


def _runnable_payload_disposition(trace: Trace, entries: tuple[Any, ...]) -> str:
    """Return ``"bind"`` / ``"skip"`` / ``"error"`` for one runnable payload family.

    THE single structural disposition every payload binder (weights, non-persistent buffers,
    archived activations, and any future execution-only family) consults before decoding a blob,
    so ONE typed-analysis degradation rule covers EVERY typed descriptor-parse refusal -- not just
    ``context_field_invalid`` or the legacy v1 carve-out. Three-way:

    * ``"bind"`` -- a parsed sparse descriptor exists; validate and bind normally.
    * ``"skip"`` -- a runnable descriptor was PRESENT but refused at parse (``context_field_invalid``,
      legacy v1, or any typed parse refusal), so the trace loaded ANALYSIS-ONLY
      (``provider == LOADED_SPARSE``, ``descriptor is None``, readiness UNAVAILABLE with the typed
      diagnostic). Its payload blobs stay unbound (no supported descriptor validates them) and the
      typed readiness diagnostic survives -- the load must NOT hard-fail on the payload binder
      (an ``include_weights`` artifact with a tampered context field
      raised an untyped IO error pointing at intact weights and lost the typed diagnostic).
    * ``"error"`` -- a genuine analysis artifact (``provider == LOADED_ANALYSIS`` / no runnable
      descriptor) carrying STRAY runnable payload blobs is a real inconsistency; hard-fail.
    """

    from .._runnable import RunProvider

    if trace._sparse_run_descriptor is not None:
        return "bind"
    readiness = trace._runnable.readiness
    if getattr(readiness, "provider", None) is RunProvider.LOADED_SPARSE:
        # A refused/legacy sparse descriptor degrades to analysis-only; skip binding.
        return "skip"
    # No runnable descriptor at all: stray runnable blobs on a true analysis artifact hard-fail.
    return "error" if entries else "skip"


def _bind_embedded_nonpersistent_buffer_payload(
    trace: Trace,
    *,
    manifest: Manifest | runnable_bind.RunnablePayloadIndex,
    bundle_path: Path,
    map_location: str | torch.device,
    resolved_blobs_dir: Path | None = None,
) -> None:
    """Decode and bind mandatory captured non-persistent buffer values.

    Parameters
    ----------
    trace:
        Rehydrated Trace with its sparse descriptor and readiness attached.
    manifest:
        Parsed tensor manifest containing the dedicated buffer entries.
    bundle_path:
        Artifact root used to resolve blob paths safely.
    map_location:
        Device selected for loaded buffer tensors.
    resolved_blobs_dir:
        Canonical blob containment root for this load operation.

    Raises
    ------
    ArtifactError
        If descriptor declarations, blob membership, or checksums disagree.
    StateBindingError
        If decoded values violate the non-persistent buffer slot contract.
    """

    if resolved_blobs_dir is None:
        resolved_blobs_dir = resolve_bundle_blobs_dir(bundle_path)
    descriptor = trace._sparse_run_descriptor
    entries = tuple(
        entry for entry in manifest.tensors if entry.kind == _RUNNABLE_NONPERSISTENT_BUFFER_KIND
    )
    disposition = _runnable_payload_disposition(trace, entries)
    if disposition == "skip":
        # Parse-refused / legacy analysis-only degradation: payload blobs stay unbound and the
        # typed readiness diagnostic survives.
        return
    if disposition == "error":
        raise ArtifactError(
            "Non-persistent buffer payloads require a parsed sparse runnable descriptor."
        )
    assert descriptor is not None  # disposition == "bind"
    declared = descriptor.payload_layers.nonpersistent_buffers
    if not declared.present:
        if entries:
            raise ArtifactError(
                "Runnable non-persistent buffer blobs are present while their payload "
                "declaration is false."
            )
        return
    if declared.schema != "runnable_nonpersistent_buffer_v1":
        raise ArtifactError(
            f"Unsupported runnable non-persistent buffer payload schema {declared.schema!r}."
        )
    embedded: dict[str, torch.Tensor] = {}
    for entry in entries:
        if entry.label in embedded:
            raise ArtifactError(
                f"Runnable non-persistent buffer payload repeats canonical name {entry.label!r}."
            )
        blob_path = resolve_bundle_blob_path(
            bundle_path,
            entry.relative_path,
            resolved_blobs_dir=resolved_blobs_dir,
        )
        embedded[entry.label] = runnable_bind.read_runnable_payload(blob_path, entry, map_location)

    bind_embedded_nonpersistent_buffers(trace, embedded)


def _bind_embedded_weight_payload(
    trace: Trace,
    *,
    manifest: Manifest | runnable_bind.RunnablePayloadIndex,
    bundle_path: Path,
    map_location: str | torch.device,
    resolved_blobs_dir: Path | None = None,
) -> None:
    """Decode and strictly bind the optional runnable state-dict blob family.

    Parameters
    ----------
    trace:
        Rehydrated Trace with its sparse descriptor and readiness attached.
    manifest:
        Parsed tensor manifest containing the separately named weight entries.
    bundle_path:
        Artifact root used to resolve blob paths safely.
    map_location:
        Device selected for loaded state tensors.
    resolved_blobs_dir:
        Canonical blob containment root for this load operation.

    Raises
    ------
    ArtifactError
        If descriptor flags and blob-family membership disagree or a blob is
        corrupt.
    StateBindingError
        If the decoded state violates the shared strict binding contract.
    """

    if resolved_blobs_dir is None:
        resolved_blobs_dir = resolve_bundle_blobs_dir(bundle_path)
    descriptor = trace._sparse_run_descriptor
    weight_entries = tuple(
        entry for entry in manifest.tensors if entry.kind == _RUNNABLE_WEIGHT_KIND
    )
    disposition = _runnable_payload_disposition(trace, weight_entries)
    if disposition == "skip":
        # an ``include_weights`` artifact whose sparse descriptor was refused at
        # parse (e.g. ``context_field_invalid``) loads ANALYSIS-ONLY; the intact weight blobs
        # stay unbound and the typed diagnostic survives -- never an untyped hard IO error.
        return
    if disposition == "error":
        raise ArtifactError("Weight payload blobs require a parsed sparse runnable descriptor.")
    assert descriptor is not None  # disposition == "bind"
    declared = descriptor.payload_layers.weights
    if not declared.present:
        if weight_entries:
            raise ArtifactError(
                "Runnable weight blobs are present while payload_layers.weights.present is false."
            )
        return
    if declared.schema != "state_dict_v1":
        raise ArtifactError(f"Unsupported runnable weight payload schema {declared.schema!r}.")
    embedded: dict[str, torch.Tensor] = {}
    for entry in weight_entries:
        if entry.label in embedded:
            raise ArtifactError(
                f"Runnable weight payload repeats canonical state name {entry.label!r}."
            )
        blob_path = resolve_bundle_blob_path(
            bundle_path,
            entry.relative_path,
            resolved_blobs_dir=resolved_blobs_dir,
        )
        embedded[entry.label] = runnable_bind.read_runnable_payload(blob_path, entry, map_location)

    bind_embedded_trace_state(trace, embedded)


def _bind_archived_activation_payload(
    trace: Trace,
    *,
    manifest: Manifest | runnable_bind.RunnablePayloadIndex,
    bundle_path: Path,
    map_location: str | torch.device,
    resolved_blobs_dir: Path | None = None,
) -> None:
    """Load the inspection-only selected-activation family without seeding execution.

    Parameters
    ----------
    trace:
        Rehydrated Trace receiving the separate archive inspection mapping.
    manifest:
        Parsed tensor manifest containing activation-family entries.
    bundle_path:
        Artifact root used to resolve blob paths safely.
    map_location:
        Device selected for loaded activation tensors.
    resolved_blobs_dir:
        Canonical blob containment root for this load operation.

    Raises
    ------
    ArtifactError
        If descriptor membership, blob entries, or file checksums disagree.
    """

    from .._runnable import ActivationPayloadLayerDescriptor, ArchivedActivation

    if resolved_blobs_dir is None:
        resolved_blobs_dir = resolve_bundle_blobs_dir(bundle_path)
    descriptor = trace._sparse_run_descriptor
    activation_entries = {
        entry.blob_id: entry
        for entry in manifest.tensors
        if entry.kind == _RUNNABLE_ACTIVATION_KIND
    }
    disposition = _runnable_payload_disposition(trace, tuple(activation_entries.values()))
    if disposition == "skip":
        # Parse-refused / legacy analysis-only degradation: archived activation
        # blobs stay unbound and the typed readiness diagnostic survives.
        return
    if disposition == "error":
        raise ArtifactError("Activation payload blobs require a parsed sparse runnable descriptor.")
    assert descriptor is not None  # disposition == "bind"
    declared = descriptor.payload_layers.activations
    if not declared.present:
        if activation_entries:
            raise ArtifactError(
                "Runnable activation blobs are present while their payload flag is false."
            )
        trace._runnable.archived_activations = {}
        return
    if not isinstance(declared, ActivationPayloadLayerDescriptor):
        raise ArtifactError("Runnable activation payload metadata is incomplete.")
    from .._runnable import RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION

    if declared.schema != RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION:
        raise ArtifactError(f"Unsupported runnable activation payload schema {declared.schema!r}.")
    archived: dict[str, ArchivedActivation] = {}
    for member in declared.members:
        entry = activation_entries.get(member.blob_id)
        if entry is None:
            raise ArtifactError(
                f"Runnable activation member {member.blob_id!r} has no tensor entry."
            )
        blob_path = resolve_bundle_blob_path(
            bundle_path,
            entry.relative_path,
            resolved_blobs_dir=resolved_blobs_dir,
        )
        tensor = runnable_bind.read_runnable_payload(blob_path, entry, map_location)
        # The file ``sha256`` above attests the BLOB against the MANIFEST, but the
        # member's ``byte_digest`` -- the value republished on the public
        # ``ArchivedActivation.byte_digest`` beside ``.value`` -- was never checked
        # against what actually loaded. Editing one member digest (blobs untouched) loaded
        # silently and produced a record that CONTRADICTED ITSELF on the inspection
        # surface. Verify the loaded tensor against its own declared digest so the
        # published pair is always internally consistent.
        observed_byte_digest = runnable_tensor_byte_digest(tensor)
        if observed_byte_digest != member.byte_digest:
            raise ArtifactError(
                "Archived activation byte-digest mismatch for "
                f"blob_id={entry.blob_id} (slot_id={member.slot_id!r}, field={member.field!r}): "
                f"declared {member.byte_digest!r} but the loaded tensor digests to "
                f"{observed_byte_digest!r}."
            )
        archive_key = f"{member.slot_id}:{member.field}"
        if archive_key in archived:
            raise ArtifactError(f"Runnable activation payload repeats {archive_key!r}.")
        archived[archive_key] = ArchivedActivation(
            slot_id=member.slot_id,
            call_id=member.call_id,
            op_label=member.op_label,
            field=member.field,
            byte_digest=member.byte_digest,
            value=tensor,
        )
    if set(activation_entries) != {member.blob_id for member in declared.members}:
        raise ArtifactError("Runnable activation blobs and declared membership disagree.")
    trace._runnable.archived_activations = archived


def _load_unified_tlspec(
    bundle_path: Path,
    *,
    lazy: bool,
    map_location: str | torch.device,
    materialize_nested: bool,
    payload_hints: PayloadLoadHints | None,
    trust_custom_callables: bool,
    allowed_custom_callable_modules: Collection[str] | None,
    bundle_visited: frozenset[Path] | None = None,
    unknown_relations: Literal["opaque", "refuse"] = "opaque",
) -> Trace | Bundle | InterventionSpec | Recipe:
    """Load a unified ``.tlspec`` bundle by manifest kind.

    Parameters
    ----------
    bundle_path:
        Directory containing a unified ``manifest.json``.
    lazy:
        Whether direct out/grad blobs should remain lazy placeholders.
    map_location:
        Target device for eager tensor materialization.
    materialize_nested:
        Whether nested blob refs should be materialized when ``lazy=True``.
    payload_hints:
        Optional backend payload hints used during materialization.
    trust_custom_callables:
        Whether arbitrary custom callable imports are trusted when no allowlist
        is supplied.
    allowed_custom_callable_modules:
        Optional custom callable module allowlist.

    Returns
    -------
    Trace | Bundle | InterventionSpec | Recipe
        Loaded object selected by unified manifest kind.

    Raises
    ------
    ArtifactError
        If the unified kind is unsupported in this runtime, or if the bundle
        path or one of its well-known members is a symlink.
    """

    _reject_symlink_path(bundle_path, context="bundle path")
    _reject_symlink_path(bundle_path / "manifest.json", context="manifest")
    _reject_symlink_path(bundle_path / "metadata.pkl", context="metadata")
    _reject_symlink_path(bundle_path / "blobs", context="blobs directory")
    manifest = _read_manifest_object(bundle_path / "manifest.json")
    # `kind` predates the "bundle"/"intervention" discriminator on some older unified
    # manifests (ledgered legacy-manifest tolerance, test_tlspec_parse_fuzz.py::_OPTIONAL_KEYS):
    # an absent `kind` means a plain trace, the only kind that existed before the
    # discriminator was introduced, never an unsupported-kind refusal.
    kind = manifest.get("kind", "trace")
    if isinstance(kind, str) and kind in MODEL_FREE_TLSPEC_KINDS:
        return load_model_free_tlspec(
            kind,
            bundle_path,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )
    if kind == "trace":
        _preflight_unified_trace_manifest(manifest, bundle_path=bundle_path)
        parsed_manifest = _manifest_for_unified_trace_load(manifest)
        loaded_trace = _load_trace_payload(
            bundle_path,
            parsed_manifest,
            lazy=lazy,
            map_location=map_location,
            materialize_nested=materialize_nested,
            payload_hints=payload_hints,
            sparse_run=cast(Mapping[str, Any] | None, manifest.get("run")),
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )
        model_fingerprint = manifest.get("model_fingerprint")
        if isinstance(model_fingerprint, dict):
            setattr(loaded_trace, "_source_bundle_model_fingerprint", model_fingerprint)
        return loaded_trace
    if kind == "bundle":
        from .bundle_container import BundleLoadContext, _load_unified_bundle

        member_options = {
            "lazy": lazy,
            "map_location": map_location,
            "materialize_nested": materialize_nested,
        }
        return _load_unified_bundle(
            bundle_path,
            bundle_visited=bundle_visited,
            unknown_relations=unknown_relations,
            context=BundleLoadContext(member_options, manifest.get("tlspec_version")),
        )
    raise ArtifactError(f"Unsupported unified tlspec kind={kind!r}.")


def _preflight_unified_trace_manifest(
    manifest: dict[str, Any],
    *,
    bundle_path: Path,
) -> None:
    """Inspect backend-aware trace manifest fields before torch manifest parsing.

    Parameters
    ----------
    manifest:
        Raw unified manifest object.
    bundle_path:
        Root bundle directory containing the declared tensor blobs.

    Raises
    ------
    ArtifactTooOldError
        If the manifest predates the tlspec_version 6 rehydration floor
        (first written by released torchlens 2.31.0). Checked before schema validation so a pre-floor
        artifact (which also fails the current schema) refuses with the
        floor named instead of a missing-field error.
    ArtifactError
        If the manifest schema version is unsupported or inconsistent.
    BackendPayloadUnsupportedError
        If a non-torch audit-only manifest cannot be materialized by this runtime.
    """

    from ..validation import validate_tlspec

    raw_version = manifest.get("tlspec_version")
    raise_if_manifest_below_floor(raw_version, str(bundle_path))

    try:
        validate_tlspec(bundle_path, allow_unsupported_runnable_versions=True)
    except ValueError as exc:
        # Never launder an already-typed refusal: re-wrapping a
        # ValueError-lineage TorchLens error here stripped its stable code and
        # structured fields into one content-free message.
        if isinstance(getattr(exc, "fields", None), dict) and exc.fields.get("code"):  # type: ignore[attr-defined]
            raise
        raise ArtifactError(f"Invalid unified trace manifest: {exc}") from exc

    schema_version = manifest.get("schema_version", 1)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ArtifactError("Unified trace manifest schema_version must be an integer.")
    if schema_version == 1:
        # Schema v1 is torch-only (see ``_manifest_for_unified_trace_load``), so a
        # per-entry logical_backend that is not torch would force materialization
        # to import a foreign codec runtime. Fail closed before the body index.
        _preflight_torch_entry_logical_backends(manifest, backend_name="torch")
        _preflight_unified_trace_body_index(manifest, bundle_path=bundle_path)
        return
    if schema_version != 2:
        raise ArtifactError(
            f"Unsupported unified trace manifest schema_version={schema_version}; "
            "this runtime supports schema versions 1 and 2."
        )

    backend_name = manifest.get("backend")
    if not isinstance(backend_name, str) or backend_name == "":
        raise ArtifactError("Manifest schema v2 trace requires a non-empty backend field.")
    spec = get_backend_spec(backend_name)
    if schema_version not in spec.serialization_policy.manifest_schema_versions:
        raise ArtifactError(
            f"Backend {backend_name!r} does not support manifest schema_version={schema_version}."
        )
    if not isinstance(manifest.get("backend_runtime"), dict):
        raise ArtifactError("Manifest schema v2 trace requires object backend_runtime.")
    if not isinstance(manifest.get("payload_policy"), dict):
        raise ArtifactError("Manifest schema v2 trace requires object payload_policy.")
    if str(spec.name) == "torch":
        if (
            not isinstance(manifest.get("torch_version"), str)
            or manifest.get("torch_version") == ""
        ):
            raise ArtifactError("Torch manifest schema v2 requires non-empty torch_version.")
        # A torch bundle early-returns past ``_preflight_schema_v2_payload_codecs``,
        # yet materialization still honors each entry's ``logical_backend``. Assert
        # every entry stays torch so a torch-declared bundle cannot smuggle a
        # non-torch codec import past preflight.
        _preflight_torch_entry_logical_backends(manifest, backend_name=backend_name)
        _preflight_unified_trace_body_index(manifest, bundle_path=bundle_path)
        return

    _preflight_schema_v2_runtime(manifest, backend_name=backend_name)
    payload_policy = manifest["payload_policy"]
    materializes = bool(payload_policy.get("materialization_supported", False))
    if materializes:
        _preflight_schema_v2_payload_codecs(manifest, backend_name=backend_name)
    _preflight_unified_trace_body_index(manifest, bundle_path=bundle_path)


def _preflight_unified_trace_body_index(
    manifest: dict[str, Any],
    *,
    bundle_path: Path,
) -> None:
    """Cross-check the public body index against operative tensor entries.

    Parameters
    ----------
    manifest:
        Raw unified trace manifest.
    bundle_path:
        Root bundle directory containing persisted blobs.

    Raises
    ------
    ArtifactError
        If the body index is desynchronized from tensor entries or files.
    """

    body_index = manifest.get("body_index")
    tensors = manifest.get("tensors")
    if not isinstance(body_index, list):
        raise ArtifactError("Unified trace manifest body_index must be a list.")
    if not isinstance(tensors, list):
        raise ArtifactError("Unified trace manifest tensors must be a list.")
    payload_policy = manifest.get("payload_policy")
    if manifest.get("body_format") == "audit_only" and isinstance(payload_policy, dict):
        if payload_policy.get("materialization_supported") is False:
            return
    if len(body_index) != len(tensors):
        raise ArtifactError(
            "Unified trace manifest body_index length does not match tensor entries."
        )

    missing_blob_ids: list[str] = []
    # The explicit length guard above already refused any mismatch, so these
    # pair exactly.
    for index, (body_entry, tensor_entry) in enumerate(zip(body_index, tensors, strict=True)):
        if not isinstance(body_entry, dict) or not isinstance(tensor_entry, dict):
            raise ArtifactError(
                f"Unified trace body_index/tensors entry {index} must be an object."
            )
        body_filename = body_entry.get("filename")
        tensor_filename = tensor_entry.get("relative_path")
        if not isinstance(tensor_filename, str) or tensor_filename == "":
            raise ArtifactError(
                f"Unified trace tensors[{index}].relative_path must be a non-empty string."
            )
        blob_path = resolve_bundle_blob_path(bundle_path, tensor_filename)
        _reject_symlink_path(blob_path, context="tensor blob")
        if body_filename != tensor_filename:
            raise ArtifactError(
                f"Unified trace body_index[{index}].filename does not match "
                "the corresponding tensor relative_path."
            )
        if not blob_path.is_file():
            missing_blob_ids.append(str(tensor_entry.get("blob_id", "<unknown>")))
    if missing_blob_ids:
        refuse_missing_blobs(missing_blob_ids)


def _preflight_torch_entry_logical_backends(
    manifest: dict[str, Any],
    *,
    backend_name: str,
) -> None:
    """Fail closed for torch bundles that declare a non-torch entry backend.

    Torch trace bundles (schema v1, and schema v2 with ``backend == "torch"``)
    early-return past ``_preflight_schema_v2_payload_codecs``, but blob
    materialization (``materialize_transport_tensor``) still keys off each
    manifest entry's ``logical_backend``. A per-entry ``logical_backend`` that is
    not ``backend_name`` would therefore force a non-torch payload codec import
    (and, if the runtime is present, decode) during a DEFAULT ``tl.load`` on a
    torch-declared bundle. Reject any such entry before the body index.

    Parameters
    ----------
    manifest:
        Raw unified torch trace manifest.
    backend_name:
        Declared torch backend name (``"torch"``).

    Raises
    ------
    ArtifactError
        If ``body_index`` or ``tensors`` is malformed, or if any entry declares
        a ``logical_backend`` other than ``backend_name``.
    """

    for section_name in ("body_index", "tensors"):
        entries = manifest.get(section_name, [])
        if not isinstance(entries, list):
            raise ArtifactError(f"Torch trace manifest {section_name} must be a list.")
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                raise ArtifactError(f"Torch trace manifest {section_name} entries must be objects.")
            logical_backend = raw_entry.get("logical_backend", backend_name)
            if logical_backend is None:
                continue
            if not isinstance(logical_backend, str) or logical_backend == "":
                raise ArtifactError(
                    f"Torch trace manifest {section_name} entry logical_backend must be a "
                    "non-empty string."
                )
            if logical_backend != backend_name:
                raise ArtifactError(
                    f"Torch trace manifest declares {section_name} entry "
                    f"logical_backend={logical_backend!r}; a torch bundle may not select a "
                    "non-torch payload codec."
                )


def _preflight_schema_v2_payload_codecs(
    manifest: dict[str, Any],
    *,
    backend_name: str,
) -> None:
    """Fail closed for schema-v2 materialized payloads with unknown codecs.

    Parameters
    ----------
    manifest:
        Raw unified trace manifest.
    backend_name:
        Logical backend declared by the trace manifest.

    Raises
    ------
    ArtifactError
        If body-index or tensors codec fields are malformed, or if the codec
        registry fails unexpectedly during preflight.
    BackendPayloadUnsupportedError
        If a declared materialized body or tensor entry uses an unsupported
        codec.
    """

    body_index = manifest.get("body_index", [])
    if not isinstance(body_index, list):
        raise ArtifactError("Manifest schema v2 trace body_index must be a list.")
    tensors = manifest.get("tensors", [])
    if not isinstance(tensors, list):
        raise ArtifactError("Manifest schema v2 trace tensors must be a list.")
    # ``body_index`` is the public mirror, but the load path materializes
    # out/grad body blobs from the ``tensors`` entries. Both sections must
    # declare a supported codec so a desynchronized manifest cannot smuggle an
    # unvalidated codec past preflight into blob materialization.
    for section_name, entries in (("body_index", body_index), ("tensors", tensors)):
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                raise ArtifactError(f"Manifest schema v2 {section_name} entries must be objects.")
            _preflight_schema_v2_entry_codec(
                raw_entry,
                backend_name=backend_name,
                section_name=section_name,
            )


def _preflight_schema_v2_entry_codec(
    raw_entry: dict[str, Any],
    *,
    backend_name: str,
    section_name: str,
) -> None:
    """Validate one materialized schema-v2 manifest entry's payload codec.

    Parameters
    ----------
    raw_entry:
        Raw ``body_index`` or ``tensors`` manifest entry.
    backend_name:
        Logical backend declared by the trace manifest.
    section_name:
        Manifest section owning the entry, used in error messages.

    Raises
    ------
    ArtifactError
        If the entry codec fields are malformed or codec resolution fails
        unexpectedly.
    BackendPayloadUnsupportedError
        If the entry declares an unsupported codec.
    """

    logical_backend = raw_entry.get("logical_backend", backend_name)
    codec_name = raw_entry.get("codec")
    if not isinstance(logical_backend, str) or logical_backend == "":
        raise ArtifactError(
            f"Manifest {section_name} entry logical_backend must be a non-empty string."
        )
    if logical_backend != backend_name:
        raise ArtifactError(
            f"Manifest backend {backend_name!r} conflicts with {section_name} entry "
            f"logical_backend={logical_backend!r}."
        )
    if not isinstance(codec_name, str) or codec_name == "":
        raise ArtifactError(
            f"Materialized schema v2 {section_name} entries require a codec string."
        )
    try:
        codec = get_payload_codec(logical_backend)
        supported_codec_name = str(codec.codec_name)
    except Exception as exc:
        raise ArtifactError(
            f"Failed to resolve the payload codec for backend {logical_backend!r} "
            "during manifest preflight."
        ) from exc
    if supported_codec_name == "none" or supported_codec_name != codec_name:
        raise BackendPayloadUnsupportedError(
            f"Manifest schema v2 trace for backend {backend_name!r} declares unsupported "
            f"payload codec {codec_name!r}."
        )


def _manifest_for_unified_trace_load(manifest: dict[str, Any]) -> Manifest:
    """Build the legacy payload manifest used by trace rehydration.

    Parameters
    ----------
    manifest:
        Raw unified trace manifest.

    Returns
    -------
    Manifest
        Manifest object for the metadata/body payload loader.
    """

    if manifest.get("schema_version", 1) != 2 or manifest.get("backend") == "torch":
        parsed_manifest = Manifest.from_dict(manifest)
        object.__setattr__(parsed_manifest, "_tl_logical_backend", "torch")
        object.__setattr__(parsed_manifest, "_tl_payload_materialization_supported", True)
        return parsed_manifest

    payload_manifest = dict(manifest)
    payload_manifest["torch_version"] = torch.__version__
    parsed_manifest = Manifest.from_dict(payload_manifest)
    payload_policy = manifest.get("payload_policy", {})
    materializes = isinstance(payload_policy, dict) and bool(
        payload_policy.get("materialization_supported", False)
    )
    object.__setattr__(parsed_manifest, "_tl_logical_backend", manifest.get("backend"))
    object.__setattr__(parsed_manifest, "_tl_payload_materialization_supported", materializes)
    return parsed_manifest


def _read_manifest_object(path: Path) -> dict[str, Any]:
    """Read one manifest as a JSON object without schema validation.

    Parameters
    ----------
    path:
        Manifest file path.

    Returns
    -------
    dict[str, Any]
        Decoded manifest object.

    Raises
    ------
    ArtifactError
        If the manifest cannot be read as a JSON object.
    """

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = _json.load_bounded(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"Failed to read manifest at {path}: {type(exc).__name__}: {exc}.",
            code="manifest_unreadable",
        ) from exc
    if not isinstance(data, dict):
        raise ArtifactError("Manifest root must be a JSON object.", code="manifest_not_json_object")
    return data


def cleanup_tmp(path: str | Path, *, force: bool = False) -> list[Path]:
    """Remove leftover sibling temp/backup bundle directories for one target path.

    Also sweeps orphaned ``.bak.<uuid>`` directories left behind when
    ``save(overwrite=True)`` fails and the best-effort ``_restore_backup()``
    step that normally renames the backup back onto ``bundle_path`` itself
    fails too (e.g. a second, independent I/O failure). If ``bundle_path``
    is missing, the ``.bak.*`` dir holds the only surviving copy of the
    pre-overwrite bundle, so it is restored back onto ``bundle_path``
    (recovering the data) instead of deleted. If ``bundle_path`` already
    exists and a candidate ``.bak.*`` is byte-for-byte identical to it
    (e.g. it was just restored there by an earlier candidate in this same
    sweep, or ``save()``'s post-success backup cleanup failed after a fully
    successful overwrite), the duplicate is provably redundant and is
    always removed. Otherwise the candidate's contents are NOT provably
    redundant -- it may be a genuinely distinct backup from an unrelated
    incident -- so it is only removed when ``force=True`` is passed
    (mirroring the ``.tmp.*`` sweep's non-``PARTIAL`` gating below); by
    default it is left in place with a warning to avoid silent data loss.

    Parameters
    ----------
    path:
        The **target bundle path itself** (e.g. ``"demo_bundle"``, the same
        path you pass to ``save(path, ...)``/``load(path)``) -- NOT its
        containing directory. Sibling ``.tmp.*``/``.bak.*`` candidates are
        found by globbing ``f"{Path(path).name}.tmp.*"`` /
        ``f"{Path(path).name}.bak.*"`` inside ``Path(path).parent``, so
        passing the parent directory instead (expecting "clean up everything
        inside this directory" semantics) silently matches nothing -- no
        error, no warning, zero directories removed.
    force:
        Whether temp dirs without a ``PARTIAL`` sentinel, and backup dirs
        that are not provably redundant, should also be removed.

    Returns
    -------
    list[Path]
        Removed temp directory paths, plus any restored backup paths (now
        living at ``bundle_path``).

    Raises
    ------
    ArtifactError
        If the requested target path or candidate temp/backup dirs are symlinks.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     target = pathlib.Path(tmp) / "demo.tlspec"
    ...     partial = pathlib.Path(tmp) / "demo.tlspec.tmp.partial"
    ...     partial.mkdir()
    ...     _ = (partial / "PARTIAL").write_text("", encoding="ascii")
    ...     removed = [p.name for p in tl.io.cleanup_tmp(target)]
    >>> removed
    ['demo.tlspec.tmp.partial']
    """

    bundle_path = Path(path)
    _reject_symlink_path(bundle_path, context="cleanup target")
    removed: list[Path] = []
    # the bundle basename is DATA, not a pattern. Escaping it stops glob
    # metacharacters in a legal filename (e.g. ``job*``) from widening the sweep --
    # and, crucially, the backup RESTORATION below -- to sibling bundles.
    escaped_name = glob.escape(bundle_path.name)
    for candidate in sibling_staging_dirs(bundle_path):
        if not candidate.is_dir():
            continue
        if force or (candidate / PARTIAL_SENTINEL).exists():
            shutil.rmtree(candidate)
            removed.append(candidate)
            continue
        warnings.warn(
            f"Leaving non-partial temp directory {candidate} in place; pass force=True to remove it.",
            TorchLensWarning,
            stacklevel=2,
        )

    bak_pattern = f"{escaped_name}.bak.*"
    for candidate in bundle_path.parent.glob(bak_pattern):
        if candidate.is_symlink():
            raise ArtifactError(f"Refusing to clean symlink backup directory {candidate}.")
        if not candidate.is_dir():
            continue
        if not bundle_path.exists():
            _restore_backup(candidate, bundle_path, warn_on_failure=False)
            if not candidate.exists():
                removed.append(bundle_path)
            else:
                warnings.warn(
                    f"Leaving orphaned backup directory {candidate} in place; "
                    "restoring it onto the missing bundle path failed.",
                    TorchLensWarning,
                    stacklevel=2,
                )
            continue
        if _directories_content_equal(candidate, bundle_path):
            shutil.rmtree(candidate)
            removed.append(candidate)
            continue
        if force:
            shutil.rmtree(candidate)
            removed.append(candidate)
            warnings.warn(
                f"Force-removed backup directory {candidate} whose contents differ "
                f"from the live bundle at {bundle_path}; it was not provably redundant.",
                TorchLensWarning,
                stacklevel=2,
            )
            continue
        warnings.warn(
            f"Leaving backup directory {candidate} in place; its contents differ from "
            f"the live bundle at {bundle_path} and it is not provably redundant. "
            "Pass force=True to remove it anyway.",
            TorchLensWarning,
            stacklevel=2,
        )
    return removed


def _scrub_trace_for_bundle(
    trace: Trace,
    *,
    include_outs: bool,
    include_grads: bool,
    include_saved_args: bool,
    include_rng_states: bool,
    include_source: bool = True,
    include_custom_attributes: bool = True,
    include_buffer_values: bool = True,
    sparse_runnable: bool = False,
    record_pairs: list[tuple[Any, Any]] | None = None,
) -> tuple[dict[str, Any], list[BlobSpec], list[dict[str, str]]]:
    """Scrub a model log while excluding transient load-only private attrs.

    Parameters
    ----------
    trace:
        Model log being saved.
    include_outs, include_grads, include_saved_args, include_rng_states:
        Whether outs, grads, nested captured args and RNG states are blobified.
    include_source:
        Whether captured model source text and docstrings are embedded; absolute
        source paths are relativized to basenames regardless.
    include_custom_attributes, include_buffer_values:
        Whether module instance attributes and pre-forward buffer values persist.
    sparse_runnable:
        Whether all sparse-core tensor payload families must be dropped.
    record_pairs:
        Filled with ``(scrubbed copy, live original)`` pairs (``scrub_for_save``).

    Returns
    -------
    tuple[dict[str, Any], list[BlobSpec], list[dict[str, str]]]
        Scrubbed metadata, blob specs, and unsupported tensor audit records.
    """

    backend_name = str(trace._provenance_backend if is_live_trace(trace) else "torch")
    payload_materialization = get_backend_spec(backend_name).capabilities.payload_materialization
    transient_attrs = {}
    for attr_name in LOAD_PROVENANCE_TRANSIENT_ATTRS:
        if hasattr(trace, attr_name):
            transient_attrs[attr_name] = getattr(trace, attr_name)
            delattr(trace, attr_name)
    try:
        return scrub_for_save(
            trace,
            include_outs=include_outs,
            include_grads=include_grads,
            include_saved_args=include_saved_args,
            include_rng_states=include_rng_states,
            include_source=include_source,
            include_custom_attributes=include_custom_attributes,
            include_buffer_values=include_buffer_values,
            sparse_runnable=sparse_runnable,
            backend_name=backend_name,
            payload_materialization=payload_materialization,
            record_pairs=record_pairs,
        )
    finally:
        for attr_name, attr_value in transient_attrs.items():
            setattr(trace, attr_name, attr_value)


def _apply_visualization_save_policy(
    trace: Trace,
    *,
    scrubbed_state: dict[str, Any],
    bundle_path: Path,
    tmp_path: Path,
) -> None:
    """Copy or clear rendered visualizer paths in scrubbed bundle metadata.

    Parameters
    ----------
    trace:
        Live trace being saved.
    scrubbed_state:
        Scrubbed state that will be pickled.
    bundle_path:
        Final bundle path.
    tmp_path:
        Temporary bundle directory.
    """

    scrubbed_layers = scrubbed_state.get("_layer_list")
    if not isinstance(scrubbed_layers, list):
        return
    if not bool(getattr(trace, "save_visualizations", False)):
        for layer in scrubbed_layers:
            if hasattr(layer, "visualizer_path"):
                layer.visualizer_path = None
        return

    visualizer_dir = tmp_path / "visualizers"
    # scrubbed_layers is a 1:1 scrub of trace._layer_list; pairing a truncated
    # prefix would silently skip visualizer copies for the tail layers.
    for index, (live_layer, scrubbed_layer) in enumerate(
        zip(trace._layer_list, scrubbed_layers, strict=True)
    ):
        source_path_value = getattr(live_layer, "visualizer_path", None)
        if not isinstance(source_path_value, str):
            continue
        source_path = Path(source_path_value)
        if not source_path.is_file():
            scrubbed_layer.visualizer_path = None
            continue
        visualizer_dir.mkdir(parents=True, exist_ok=True)
        # parity: the sidecar tree must not escape the bundle's permission
        # tightening -- copy2 preserves the scratch file's umask-derived mode and
        # would leak on cp -a/tar or a relaxed bundle root.
        _restrict_mode(visualizer_dir, 0o700)
        destination_name = f"{index:05d}_{source_path.name}"
        destination_path = visualizer_dir / destination_name
        shutil.copy2(source_path, destination_path)
        _restrict_mode(destination_path, 0o600)
        # Persist a bundle-RELATIVE path: the absolute final path embeds
        # $HOME/username, contradicting the scrub's basename-only PII policy, and
        # load re-anchors from the basename anyway (_reanchor_visualizer_paths),
        # so the absolute directory was never used -- only leaked.
        scrubbed_layer.visualizer_path = f"visualizers/{destination_name}"


def _write_payload_blob(
    *,
    store: PayloadStore,
    tensor_refs: dict[str, dict[str, Any]],
    blob_spec: BlobSpec,
    codec: PayloadCodec,
) -> TensorEntry:
    """Queue one codec-supported payload into the artifact's chunk files.

    Parameters
    ----------
    store, tensor_refs:
        Chunk store of the staged artifact; blob id to tensor reference fields
        (this payload's entry is added).
    blob_spec, codec:
        Logical payload selected during scrub and its backend payload codec.

    Returns
    -------
    TensorEntry
        Manifest tensor entry for the queued payload.
    """

    if blob_spec.logical_backend == "torch" and isinstance(blob_spec.value, torch.Tensor):
        return TensorEntry(
            **submit_tensor_payload(store=store, tensor_refs=tensor_refs, blob_spec=blob_spec)
        )

    encoded = codec.to_numpy(blob_spec.value)
    transport_tensor = numpy_to_transport_tensor(encoded.array).contiguous()
    blob, key = store.submit(host_tensor(transport_tensor), codec=NUMPY_CODEC)
    manifest_fields = codec.manifest_fields(blob_spec.value, encoded)
    codec_fields = {"id": NUMPY_CODEC, **manifest_fields}
    # ``device`` is the transport tensor's; the backend device rides the hints (logical_device).
    reference = tensor_reference(transport_tensor, blob=blob, key=key, codec=codec_fields)
    tensor_refs[blob_spec.blob_id] = reference
    return TensorEntry(
        blob_id=blob_spec.blob_id,
        kind=blob_spec.kind,
        label=blob_spec.label,
        relative_path=blob,
        backend="safetensors",
        shape=[int(dim) for dim in transport_tensor.shape],
        dtype=str(transport_tensor.dtype).replace("torch.", ""),
        device_at_save=encoded.logical_device,
        layout=str(transport_tensor.layout).replace("torch.", ""),
        bytes=int(transport_tensor.numel() * transport_tensor.element_size()),
        sha256="",
        requires_grad=False,
        **manifest_fields,
    )


def _attach_fast_copy_specs(
    trace: Trace,
    *,
    scrubbed_state: dict[str, Any],
    blob_specs: list[BlobSpec],
    include_outs: bool,
    include_grads: bool,
) -> list[_FastCopySpec]:
    """Attach direct-field blob refs for lazily-backed tensors that can be fast-copied.

    Parameters
    ----------
    trace:
        Live model log being saved.
    scrubbed_state:
        Scrubbed metadata state returned by ``scrub_for_save``.
    blob_specs:
        Existing eager-write blob specs.
    include_outs:
        Whether out fields should be persisted.
    include_grads:
        Whether grad fields should be persisted.

    Returns
    -------
    list[_FastCopySpec]
        Lazily-backed direct tensor fields that should be copied from their source
        bundles into the new bundle.
    """

    scrubbed_layers = scrubbed_state.get("_layer_list")
    if not isinstance(scrubbed_layers, list):
        return []

    used_blob_ids = {blob_spec.blob_id for blob_spec in blob_specs}
    fast_copy_specs: list[_FastCopySpec] = []
    # 1:1 scrub of trace._layer_list -- a truncated pairing would silently drop
    # fast-copy specs for the tail layers, losing their payloads in the copy.
    for live_layer, scrubbed_layer in zip(trace._layer_list, scrubbed_layers, strict=True):
        if include_outs:
            fast_copy_spec = _maybe_make_fast_copy_spec(
                live_layer=live_layer,
                scrubbed_layer=scrubbed_layer,
                tensor_field="out",
                ref_field="out_ref",
                kind="out",
                has_field="has_saved_activation",
                used_blob_ids=used_blob_ids,
            )
            if fast_copy_spec is not None:
                fast_copy_specs.append(fast_copy_spec)
        if include_grads:
            fast_copy_spec = _maybe_make_fast_copy_spec(
                live_layer=live_layer,
                scrubbed_layer=scrubbed_layer,
                tensor_field="grad",
                ref_field="grad_ref",
                kind="grad",
                has_field="has_grad",
                used_blob_ids=used_blob_ids,
            )
            if fast_copy_spec is not None:
                fast_copy_specs.append(fast_copy_spec)
    return fast_copy_specs


def _maybe_make_fast_copy_spec(
    *,
    live_layer: Any,
    scrubbed_layer: Any,
    tensor_field: str,
    ref_field: str,
    kind: str,
    has_field: str,
    used_blob_ids: set[str],
) -> _FastCopySpec | None:
    """Create one fast-copy spec for a lazily-backed direct tensor field.

    Parameters
    ----------
    live_layer:
        Live ``Op`` instance being saved.
    scrubbed_layer:
        Scrubbed ``Op`` counterpart that will be pickled.
    tensor_field:
        Direct tensor field name, for example ``"out"``.
    ref_field:
        Matching lazy-ref field name.
    kind:
        Logical manifest tensor kind.
    has_field:
        Boolean field indicating whether the tensor is expected to exist.
    used_blob_ids:
        Set of blob ids already allocated for the destination bundle.

    Returns
    -------
    _FastCopySpec | None
        Fast-copy spec when the field should be copied from its source bundle,
        otherwise ``None``.
    """

    if not bool(record_member(live_layer, has_field, False)):
        return None
    scrubbed_state = dict(state_items(scrubbed_layer))
    if scrubbed_state.get(tensor_field) is not None:
        return None
    live_field_value = dict(state_items(live_layer)).get(tensor_field)
    if live_field_value is not None:
        return None

    source_ref = record_member(live_layer, ref_field, None)
    if not isinstance(source_ref, LazyActivationRef):
        return None

    blob_id = _next_bundle_blob_id(used_blob_ids)
    blob_ref = BlobRef(blob_id=blob_id, kind=kind)
    setattr(scrubbed_layer, record_storage_attr(scrubbed_layer, tensor_field), blob_ref)
    return _FastCopySpec(
        blob_id=blob_id,
        kind=kind,
        label=str(getattr(live_layer, "_streaming_label")),
        source_ref=source_ref,
    )


def _next_bundle_blob_id(used_blob_ids: set[str]) -> str:
    """Allocate the next monotonically increasing bundle blob id.

    Parameters
    ----------
    used_blob_ids:
        Set of blob ids already reserved for the destination bundle.

    Returns
    -------
    str
        Newly-allocated zero-padded blob id.
    """

    next_id = max((int(blob_id) for blob_id in used_blob_ids), default=0) + 1
    blob_id = f"{next_id:010d}"
    used_blob_ids.add(blob_id)
    return blob_id


def _load_and_verify_fast_copy_source(
    trace: Trace,
    source_bundle_path: Path,
    *,
    cache: dict[Path, dict[str, TensorEntry]],
) -> dict[str, TensorEntry]:
    """Load and drift-verify one source bundle used for lazy fast-copy.

    Parameters
    ----------
    trace:
        Model log being resaved.
    source_bundle_path:
        Source bundle directory referenced by one lazy tensor ref.
    cache:
        Per-save cache keyed by source bundle path.

    Returns
    -------
    dict[str, TensorEntry]
        Source manifest entries indexed by blob id.

    Raises
    ------
    ArtifactError
        If the source manifest is missing or has changed since the refs were loaded.
    """

    if source_bundle_path in cache:
        return cache[source_bundle_path]

    manifest_path = source_bundle_path / "manifest.json"
    _reject_symlink_path(manifest_path, context="source manifest")
    if not manifest_path.exists():
        raise ArtifactError(f"Source bundle manifest not found at {manifest_path}.")

    expected_manifest_sha256 = getattr(trace, "_source_bundle_manifest_sha256", None)
    if not isinstance(expected_manifest_sha256, str):
        raise ArtifactError(
            "source bundle manifest fingerprint is unavailable; materialize refs and retry"
        )
    observed_manifest_sha256 = sha256_file(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise ArtifactError(
            "source bundle manifest has changed since load; materialize refs and retry"
        )

    manifest = _read_materialization_manifest(source_bundle_path)
    manifest_index = {row["blob_id"]: TensorEntry.from_dict(row) for row in manifest["tensors"]}
    cache[source_bundle_path] = manifest_index
    return manifest_index


def _fast_copy_tensor_blob(
    *,
    store: PayloadStore,
    tensor_refs: dict[str, dict[str, Any]],
    fast_copy_spec: _FastCopySpec,
    manifest_index: dict[str, TensorEntry],
) -> TensorEntry:
    """Carry one lazily-backed tensor from its source artifact into the new one.

    The source blob is digest-checked, read by key and queued into the new artifact's chunk
    files as an entry of its own (its reference takes the source device after the store answers).

    Parameters
    ----------
    store, tensor_refs:
        Chunk store of the staged artifact; blob id to tensor reference fields.
    fast_copy_spec, manifest_index:
        Fast-copy instruction; source manifest entries indexed by blob id.

    Returns
    -------
    TensorEntry
        Destination manifest entry for the copied blob.
    """

    store, source_entry = store.unshared(), manifest_index.get(fast_copy_spec.source_ref.blob_id)
    if source_entry is None:
        raise ArtifactError(f"Manifest is missing blob_id={fast_copy_spec.source_ref.blob_id}.")

    source_blob_path = resolve_bundle_blob_path(
        fast_copy_spec.source_ref.source_bundle_path,
        source_entry.relative_path,
    )
    _reject_symlink_path(source_blob_path, context="source blob")
    if not source_blob_path.exists():
        raise ArtifactError(f"Tensor blob not found at {source_blob_path}.")

    observed_sha256 = sha256_file(source_blob_path)
    if observed_sha256 != fast_copy_spec.source_ref.expected_sha256:
        raise ArtifactError(
            f"blob at {source_blob_path} sha256 mismatch; expected "
            f"{fast_copy_spec.source_ref.expected_sha256} got {observed_sha256}"
        )

    source_value = fast_copy_spec.source_ref.materialize(map_location="cpu")
    backend = source_entry.logical_backend or "torch"
    spec = BlobSpec(
        fast_copy_spec.blob_id, source_value, fast_copy_spec.kind, fast_copy_spec.label, backend
    )
    entry = _write_payload_blob(
        store=store, tensor_refs=tensor_refs, blob_spec=spec, codec=get_payload_codec(backend)
    )
    tensor_refs[fast_copy_spec.blob_id]["device"] = source_entry.device_at_save
    return dataclasses.replace(entry, device_at_save=source_entry.device_at_save)


_CUSTOM_ATTRIBUTES_DISCLOSURE_KEY_CAP = 100
"""Bound on the number of distinct top-level key names a disclosure records."""


# Structural constructor echoes harvested off stock nn.Module types; their key
# names carry no user secrets, so they do not by themselves trigger the
# save-time embedding warning below.
_BORING_CUSTOM_ATTRIBUTE_KEYS = frozenset(
    {
        "add_zero_attn",
        "affine",
        "batch_first",
        "bias",
        "bidirectional",
        "ceil_mode",
        "count_include_pad",
        "d_model",
        "dilation",
        "dim_feedforward",
        "dropout",
        "elementwise_affine",
        "embed_dim",
        "embedding_dim",
        "end_dim",
        "eps",
        "groups",
        "hidden_size",
        "in_channels",
        "in_features",
        "inplace",
        "input_size",
        "kdim",
        "kernel_size",
        "max_norm",
        "momentum",
        "nhead",
        "norm_type",
        "normalized_shape",
        "num_embeddings",
        "num_features",
        "num_heads",
        "num_layers",
        "out_channels",
        "out_features",
        "output_padding",
        "p",
        "padding",
        "padding_idx",
        "padding_mode",
        "return_indices",
        "scale_grad_by_freq",
        "sparse",
        "start_dim",
        "stride",
        "track_running_stats",
        "vdim",
    }
)
_CUSTOM_ATTRIBUTE_WARNING_KEY_PREVIEW = 8


def _warn_custom_attribute_embedding(disclosure: Mapping[str, Any]) -> None:
    """Tell the SAVER that module attributes are shipping in the artifact.

    The manifest disclosure lands INSIDE the file the user is about to hand
    out -- the one reader guaranteed not to see it is the person saving. A
    model carrying ``self.hf_token = os.environ["HF_TOKEN"]`` previously saved
    with zero terminal output; the token shipped silently. One warning at save
    time, attributed to the user's save call, changes no default and no
    persisted byte.
    """

    if not disclosure.get("included"):
        return
    interesting = [
        key
        for key in disclosure.get("top_level_keys", ())
        if key not in _BORING_CUSTOM_ATTRIBUTE_KEYS
    ]
    if not interesting:
        return
    preview = ", ".join(interesting[:_CUSTOM_ATTRIBUTE_WARNING_KEY_PREVIEW])
    if len(interesting) > _CUSTOM_ATTRIBUTE_WARNING_KEY_PREVIEW:
        preview += ", ..."
    warnings.warn(
        f"This save embeds {len(interesting)} custom module attribute(s) verbatim "
        f"in the artifact ({preview}). Review them before sharing the bundle; pass "
        "include_custom_attributes=False to withhold the values.",
        TorchLensWarning,
        stacklevel=user_stacklevel(),
    )


def _custom_attributes_disclosure(trace: Trace, *, included: bool) -> dict[str, Any]:
    """Summarize the harvested module-attribute channel for the manifest.

    Save-time disclosure: portable bundles persist every
    public, non-callable module instance attribute verbatim, so the manifest
    names the channel — whether it shipped, how many modules carry attributes,
    and the (bounded, sorted) union of top-level key names. Key NAMES only;
    values are never inspected or rewritten here.

    Parameters
    ----------
    trace:
        Source model log.
    included:
        Whether the save actually persisted the channel
        (``include_custom_attributes`` and not a sparse runnable core).

    Returns
    -------
    dict[str, Any]
        JSON-ready disclosure mapping.
    """

    module_count = 0
    top_level_keys: set[str] = set()
    for module in getattr(trace, "modules", ()) or ():
        attrs = getattr(module, "custom_attributes", None)
        if not isinstance(attrs, Mapping) or not attrs:
            continue
        module_count += 1
        for key in attrs:
            top_level_keys.add(str(key))
    truncated = len(top_level_keys) > _CUSTOM_ATTRIBUTES_DISCLOSURE_KEY_CAP
    return {
        "included": bool(included),
        "module_count": module_count,
        "top_level_keys": sorted(top_level_keys)[:_CUSTOM_ATTRIBUTES_DISCLOSURE_KEY_CAP],
        "top_level_keys_truncated": truncated,
    }


def _buffer_values_disclosure(trace: Trace, *, included: bool) -> dict[str, Any]:
    """Summarize the captured pre-forward buffer-value channel for the manifest.

    Save-time disclosure mirroring ``_custom_attributes_disclosure`` (R62
    buffer extension): every save level embedded the pre-forward value of each
    forward-overwritten registered buffer (``Trace._buffer_initial_values``)
    verbatim, with no flag, no warning, and no manifest row — training-data-
    derived state (running statistics, counters) shipped invisibly even at
    ``level="audit"``. The manifest names the channel: whether it shipped, how
    many buffers carry captured values, and the (bounded, sorted) buffer
    names. Names only; values are never inspected or rewritten here.

    Parameters
    ----------
    trace:
        Source model log.
    included:
        Whether the save actually persisted the channel
        (``include_buffer_values`` and not a sparse runnable core).

    Returns
    -------
    dict[str, Any]
        JSON-ready disclosure mapping.
    """

    captured = getattr(trace, "_buffer_initial_values", {}) or {}
    buffer_names = sorted(str(name) for name in captured)
    truncated = len(buffer_names) > _CUSTOM_ATTRIBUTES_DISCLOSURE_KEY_CAP
    return {
        "included": bool(included),
        "buffer_count": len(buffer_names),
        "buffer_names": buffer_names[:_CUSTOM_ATTRIBUTES_DISCLOSURE_KEY_CAP],
        "buffer_names_truncated": truncated,
    }


def _warn_buffer_value_embedding(disclosure: Mapping[str, Any]) -> None:
    """Tell the SAVER that pre-forward buffer values ship in the artifact.

    Same rationale as ``_warn_custom_attribute_embedding``: the manifest
    disclosure lands inside the file being handed out, so the saver is the one
    reader guaranteed not to see it. Fires only when the forward actually
    overwrote a registered buffer (the common eval-mode capture has an empty
    channel and stays silent).
    """

    if not disclosure.get("included"):
        return
    buffer_names = list(disclosure.get("buffer_names", ()))
    if not buffer_names:
        return
    preview = ", ".join(buffer_names[:_CUSTOM_ATTRIBUTE_WARNING_KEY_PREVIEW])
    if len(buffer_names) > _CUSTOM_ATTRIBUTE_WARNING_KEY_PREVIEW:
        preview += ", ..."
    warnings.warn(
        f"This save embeds the pre-forward value(s) of {len(buffer_names)} "
        f"forward-overwritten buffer(s) verbatim in the artifact ({preview}). "
        "Buffer values are training-data-derived state; pass "
        "include_buffer_values=False to withhold them.",
        TorchLensWarning,
        stacklevel=user_stacklevel(),
    )


class _SaveDisclosures(NamedTuple):
    """Save-time manifest disclosures harvested from the scrubbed state.

    ``custom_attributes`` / ``buffer_values`` are the two harvested-channel
    payloads. ``persisted_op_rows`` is the PERSISTED op-row count (the scrubbed
    ``layer_list``, which carries injected-op rows the live ``layer_list``
    never holds); the loader anchors ``manifest.n_layers`` against
    ``metadata.pkl`` (AUD-CODE 3.0b). ``None`` falls back to the live count.
    """

    custom_attributes: dict[str, Any] | None
    buffer_values: dict[str, Any] | None
    persisted_op_rows: int | None = None


def _build_manifest(
    *,
    trace: Trace,
    tensor_entries: list[TensorEntry],
    unsupported_tensors: list[dict[str, str]],
    include_source: bool = True,
    disclosures: _SaveDisclosures | None = None,
) -> Manifest:
    """Create a manifest instance for a finished bundle save.

    Parameters
    ----------
    trace:
        Source model log.
    tensor_entries:
        Persisted tensor entries.
    unsupported_tensors:
        Unsupported tensor records accumulated under ``strict=False``.
    include_source:
        When ``False`` the environment-provenance git commit hash is omitted, so
        ``include_source=False`` also drops the cwd repo's HEAD commit.
    disclosures:
        Save-time disclosures of the harvested module-attribute channel, the
        captured pre-forward buffer-value channel, and the persisted op-row count.

    Returns
    -------
    Manifest
        Fully-populated bundle manifest.
    """

    n_out_blobs = sum(1 for entry in tensor_entries if entry.kind == "out")
    n_grad_blobs = sum(1 for entry in tensor_entries if entry.kind == "grad")
    n_auxiliary_blobs = len(tensor_entries) - n_out_blobs - n_grad_blobs
    n_layers = disclosures.persisted_op_rows if disclosures else None
    return Manifest(
        tlspec_version=TLSPEC_VERSION,
        torchlens_version=TORCHLENS_VERSION,
        torch_version=torch.__version__,
        python_version=(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        platform=f"{platform.system().lower()}-{platform.machine().lower()}",
        created_at=datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        ),
        bundle_format="directory",
        n_layers=len(trace._layer_list) if n_layers is None else n_layers,
        n_out_blobs=n_out_blobs,
        n_grad_blobs=n_grad_blobs,
        n_auxiliary_blobs=n_auxiliary_blobs,
        tensors=tensor_entries,
        unsupported_tensors=unsupported_tensors,
        provenance=_collect_provenance(trace, include_source=include_source),
        custom_attributes_disclosure=disclosures.custom_attributes if disclosures else None,
        buffer_values_disclosure=disclosures.buffer_values if disclosures else None,
    )


def _collect_provenance(trace: Trace, *, include_source: bool = True) -> Provenance:
    """Collect a best-effort, bounded provenance certificate for one save.

    Parameters
    ----------
    trace:
        Source trace whose already-recorded capture facts should be certified.
    include_source:
        When ``False`` the cwd repo's HEAD commit hash is omitted: the
        manifest otherwise embedded the git commit of whatever repository contained
        the working directory at save time, with no opt-out. Tying it to the
        existing source-embedding flag lets ``include_source=False`` drop this
        environment detail along with the source it already withholds.

    Returns
    -------
    Provenance
        Provenance certificate. Individual unavailable facts are empty or ``None``.
    """

    from .. import fingerprint as trace_hash

    # A loaded trace carries its source bundle's provenance verbatim so a
    # plain resave preserves the CAPTURE-time certificate. That passthrough
    # must not bypass the include_source gate: with include_source=False the
    # fresh-save path below omits the git commit, so re-emitting a loaded
    # certificate here both leaked it and let a hostile bundle's forged
    # provenance propagate into resaves this host appears to attest. With
    # include_source=False, recompute honestly from the loaded trace instead.
    source_provenance = getattr(trace, "_source_bundle_provenance", None)
    if include_source and isinstance(source_provenance, Provenance):
        return source_provenance

    devices = sorted(
        {
            str(device)
            for op in trace._layer_list
            if (device := getattr(op, "device_ref", None)) is not None
        }
    )
    ambient = trace._runnable.capture_ambient
    default_dtype = ambient.get("default_dtype") if isinstance(ambient, Mapping) else None
    autocast_facts: list[dict[str, Any]] = []
    seen_autocast: set[str] = set()
    for op in trace._layer_list:
        state = record_member(op, "func_autocast_state", None)
        if not isinstance(state, Mapping) or not state:
            continue
        normalized = _json_ready_provenance_value(state)
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        if encoded not in seen_autocast:
            seen_autocast.add(encoded)
            autocast_facts.append(cast(dict[str, Any], normalized))

    # Could-not-compute must stay
    # distinguishable from does-not-apply. A digest failure used to silently
    # omit the engine (absence read as "engine not present") and the two
    # hashes degraded to None (indistinguishable from "no inputs"), so a
    # consumer comparing provenance across artifacts never learned the hash
    # machinery failed. Failures now record an explicit string-only
    # "unavailable:<ExceptionName>" sentinel in the same slot.
    rng_digests: dict[str, str] = {}
    rng_states = getattr(trace, "_pre_forward_rng_states", None)
    if isinstance(rng_states, Mapping):
        for engine, state in rng_states.items():
            try:
                rng_digests[str(engine)] = trace_hash.content(state)
            except Exception as digest_error:
                rng_digests[str(engine)] = f"unavailable:{type(digest_error).__name__}"

    input_hash: str | None = None
    try:
        inputs = [input_op.out for input_op in trace.input_ops]
        if inputs:
            input_hash = trace_hash.content(inputs)
    except Exception as input_error:
        input_hash = f"unavailable:{type(input_error).__name__}"

    model_structure_hash: str | None = None
    try:
        model_structure_hash = trace_hash.trace(trace)
    except Exception as structure_error:
        model_structure_hash = f"unavailable:{type(structure_error).__name__}"

    return Provenance(
        provenance_version=1,
        capture_devices=devices,
        dtype_policy={
            "default_dtype": None if default_dtype is None else str(default_dtype),
            "observed_autocast": autocast_facts,
        },
        rng_state_digests=rng_digests,
        input_hash=input_hash,
        model_structure_hash=model_structure_hash,
        git_commit_hash=_git_commit_hash(_torchlens_package_dir()) if include_source else None,
    )


def _json_ready_provenance_value(value: Any) -> Any:
    """Convert recorded provenance facts into bounded JSON-ready values.

    Parameters
    ----------
    value:
        Recorded value to normalize.

    Returns
    -------
    Any
        JSON-ready scalar or nested container.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready_provenance_value(item) for key, item in value.items()}
    if isinstance(value, Set):
        normalized = [_json_ready_provenance_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, Collection) and not isinstance(value, str | bytes | bytearray):
        return [_json_ready_provenance_value(item) for item in value]
    return str(value)


def _torchlens_package_dir() -> Path:
    """Return the torchlens package directory (provenance root for the git probe)."""

    return Path(__file__).resolve().parent.parent


def _validate_activation_transform_outputs(
    trace: Trace,
    *,
    include_outs: bool,
) -> None:
    """Reject portable saves when out postprocessing produced non-tensors.

    Parameters
    ----------
    trace:
        Model log being saved.
    include_outs:
        Whether out fields will be saved.

    Raises
    ------
    ArtifactError
        If a saved out is not a plain tensor.
    """

    if not include_outs or not is_live_trace(trace):
        return
    if trace._options_activation_transform is None:
        return
    for layer in trace._layer_list:
        if not getattr(layer, "has_saved_activation", False):
            continue
        transformed_out = record_member(layer, "transformed_out", None)
        if transformed_out is None:
            continue
        if not isinstance(transformed_out, torch.Tensor):
            raise ArtifactError(
                "Portable bundle save requires activation_transform outputs to be torch.Tensor "
                f"instances, but layer {layer.layer_label} produced "
                f"{type(transformed_out).__name__}."
            )


def _raise_for_unmaterialized_nested_blob_refs(
    value: Any,
    *,
    allowed_blob_ids: set[str],
) -> None:
    """Reject resave attempts that still contain nested lazy ``BlobRef`` objects.

    Parameters
    ----------
    value:
        Scrubbed portable state about to be written.
    allowed_blob_ids:
        Blob ids created during the current scrub pass. Any surviving nested
        ``BlobRef`` outside this set came from a prior lazy load and must be
        materialized before resave.

    Raises
    ------
    ArtifactError
        If any nested portable blob refs remain in blob-recursive fields.
    """

    if _contains_nested_blob_refs(value, set(), allowed_blob_ids):
        raise ArtifactError(
            "Trace contains unmaterialized nested blob references. "
            "Call torchlens.io.materialize_nested(trace) before saving."
        )


_NESTED_BLOB_SCALARS = (str, int, float, bool, type(None), torch.dtype, torch.device)

_NESTED_BLOB_SCALAR = 0
_NESTED_BLOB_REF = 1
_NESTED_BLOB_DICT = 2
_NESTED_BLOB_SEQUENCE = 3
_NESTED_BLOB_OBJECT = 4

# WEAK type keys so a notebook-cell / factory-made class used once as a nested
# blob value is not pinned (with its __globals__) for the process lifetime
# matches the weak caches elsewhere in _io.
_NESTED_BLOB_KINDS: weakref.WeakKeyDictionary[type, int] = weakref.WeakKeyDictionary()


def _nested_blob_node_kind(value_type: type) -> int:
    """Classify one node type for the nested-``BlobRef`` walk, memoized per type.

    Same branch order as the ``isinstance`` chain it replaces. ``frozenset`` is
    deliberately NOT a container kind here, matching the original chain.
    """

    if issubclass(value_type, _NESTED_BLOB_SCALARS):
        kind = _NESTED_BLOB_SCALAR
    elif issubclass(value_type, BlobRef):
        kind = _NESTED_BLOB_REF
    elif issubclass(value_type, dict):
        kind = _NESTED_BLOB_DICT
    elif issubclass(value_type, (list, tuple, set)):
        kind = _NESTED_BLOB_SEQUENCE
    else:
        kind = _NESTED_BLOB_OBJECT
    _NESTED_BLOB_KINDS[value_type] = kind
    return kind


def _contains_nested_blob_refs(
    value: Any,
    seen: set[int],
    allowed_blob_ids: set[str],
) -> bool:
    """Return whether any blob-recursive field still contains live ``BlobRef`` values.

    Parameters
    ----------
    value:
        Object graph node to inspect.
    seen:
        Identity set used to avoid infinite recursion on shared objects.

    Returns
    -------
    bool
        ``True`` when a nested ``BlobRef`` is still present in a blob-recursive field.
    """

    # ``OrderedDict``/``defaultdict`` were three identical ``.values()`` branches and
    # ``list``/``tuple``/``set`` three identical direct-iteration branches; they are
    # collapsed, the ``any(genexpr)`` wrappers are plain loops (this walk visits
    # hundreds of thousands of nodes per save and paid a generator frame per
    # container), and the branch chain itself is one cached type lookup.
    value_type = type(value)
    kind = _NESTED_BLOB_KINDS.get(value_type)
    if kind is None:
        kind = _nested_blob_node_kind(value_type)
    if kind == _NESTED_BLOB_SCALAR:
        return False
    if kind == _NESTED_BLOB_REF:
        return value.blob_id not in allowed_blob_ids
    if kind == _NESTED_BLOB_DICT:
        for item in value.values():
            if _contains_nested_blob_refs(item, seen, allowed_blob_ids):
                return True
        return False
    if kind == _NESTED_BLOB_SEQUENCE:
        return any(_contains_nested_blob_refs(item, seen, allowed_blob_ids) for item in value)

    spec = getattr(value_type, "_PORTABLE_STATE_SPEC", None)
    if spec is None:
        return False

    obj_id = id(value)
    if obj_id in seen:
        return False
    seen.add(obj_id)

    for field_name, field_value in state_items(value):
        policy = spec.get(field_name)
        if policy is not None:
            # Registered pre-release fields are guarded under their
            # switched-on persisting policy: a declared-DROP field that
            # persisted BLOB_RECURSIVE under the switch can still carry
            # nested lazy BlobRefs into a resave.
            policy = effective_policy(type(value), field_name, policy)
        if policy == FieldPolicy.BLOB_RECURSIVE and _container_contains_blob_ref(
            field_value, allowed_blob_ids
        ):
            return True
        if policy == FieldPolicy.KEEP and _contains_nested_blob_refs(
            field_value, seen, allowed_blob_ids
        ):
            return True
    return False


def _container_contains_blob_ref(value: Any, allowed_blob_ids: set[str]) -> bool:
    """Return whether a nested container still contains a ``BlobRef`` leaf.

    Parameters
    ----------
    value:
        Nested container or leaf value to inspect.

    Returns
    -------
    bool
        ``True`` when any descendant is a ``BlobRef``.
    """

    if isinstance(value, BlobRef):
        return value.blob_id not in allowed_blob_ids
    if isinstance(value, dict):  # covers OrderedDict / defaultdict
        return any(_container_contains_blob_ref(item, allowed_blob_ids) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_container_contains_blob_ref(item, allowed_blob_ids) for item in value)
    return False


def _check_unknown_blob_entries(manifest: Manifest, blobs_path: Path) -> None:
    """Warn when ``blobs/`` contains files that are not referenced by the manifest.

    Parameters
    ----------
    manifest:
        Parsed bundle manifest.
    blobs_path:
        Bundle ``blobs/`` directory path.
    """

    expected_names = {Path(entry.relative_path).name for entry in manifest.tensors}
    actual_names: set[str] = set()
    for child in blobs_path.iterdir():
        if child.is_symlink():
            raise ArtifactError(f"Refusing to load symlinked blob path {child}.")
        actual_names.add(child.name)
    extra_names = sorted(actual_names - expected_names)
    if not extra_names:
        return

    expected_sha256s = {entry.sha256 for entry in manifest.tensors}
    colliding_extra_names: list[str] = []
    for extra_name in extra_names:
        extra_path = blobs_path / extra_name
        extra_sha256 = sha256_file(extra_path)
        if extra_sha256 in expected_sha256s:
            colliding_extra_names.append(extra_name)

    if colliding_extra_names:
        raise ArtifactError(
            "Bundle contains unreferenced blob files whose sha256 matches a manifest entry: "
            f"{', '.join(colliding_extra_names)}."
        )

    if extra_names:
        _warn_extra_files(
            f"Bundle contains unreferenced extra files in blobs/: {', '.join(extra_names)}.",
            stacklevel=2,
        )


def _validate_manifest_blob_paths(
    manifest: Manifest,
    bundle_path: Path,
    resolved_blobs_dir: Path | None = None,
) -> None:
    """Ensure every manifest tensor entry points at a real non-symlink blob file.

    Parameters
    ----------
    manifest:
        Parsed bundle manifest.
    bundle_path:
        Bundle directory root.
    resolved_blobs_dir:
        Canonical blob containment root for this load operation.

    Raises
    ------
    ArtifactError
        If any referenced blob is missing or symlinked.
    """

    if resolved_blobs_dir is None:
        resolved_blobs_dir = resolve_bundle_blobs_dir(bundle_path)
    missing_blob_ids: list[str] = []
    for entry in manifest.tensors:
        blob_path = resolve_bundle_blob_path(
            bundle_path,
            entry.relative_path,
            resolved_blobs_dir=resolved_blobs_dir,
        )
        if blob_path.is_symlink():
            raise ArtifactError(f"Refusing to load symlinked blob path {blob_path}.")
        if not blob_path.exists():
            missing_blob_ids.append(entry.blob_id)
    if missing_blob_ids:
        refuse_missing_blobs(missing_blob_ids)


def _eager_verify_blob_payloads(
    manifest: Manifest,
    bundle_path: Path,
    map_location: str | torch.device,
    resolved_blobs_dir: Path | None = None,
) -> None:
    """Eagerly checksum and decode every blob for ``lazy=False`` loads.

    Parameters
    ----------
    manifest:
        Parsed bundle manifest.
    bundle_path:
        Bundle directory root.
    map_location:
        Device passed through to ``safetensors`` decoding.
    resolved_blobs_dir:
        Canonical blob containment root for this load operation.
    """

    if resolved_blobs_dir is None:
        resolved_blobs_dir = resolve_bundle_blobs_dir(bundle_path)
    for entry in manifest.tensors:
        blob_path = resolve_bundle_blob_path(
            bundle_path,
            entry.relative_path,
            resolved_blobs_dir=resolved_blobs_dir,
        )
        observed_sha256 = sha256_file(blob_path)
        if observed_sha256 != entry.sha256:
            raise ArtifactError(f"Checksum mismatch for blob_id={entry.blob_id} at {blob_path}.")
        tensor_map = _load_safetensors_file(blob_path, map_location)
        if _BLOB_TENSOR_KEY not in tensor_map:
            raise ArtifactError(
                f"Blob {blob_path} does not contain the expected {_BLOB_TENSOR_KEY!r} tensor entry."
            )


def _metadata_unpickler(
    handle: Any,
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> _RenameAwareUnpickler:
    """Return the metadata unpickler with the opcode ceiling the artifact channel enforces."""

    return _RenameAwareUnpickler(
        handle,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
        max_opcodes=_MAX_METADATA_PKL_OPCODES,
        count_opcodes_above=_METADATA_PKL_PRESCAN_BYTES,
    )


_OPCODE_CEILING_CODE = "metadata_object_count_exceeded"


def _opcode_ceiling_error(metadata_path: Path, *, code: str) -> ArtifactError:
    """Return the refusal for a metadata pickle above the opcode allocation ceiling.

    The byte ceiling alone does not bound allocation: a pickle packed
    with tiny values expands many times its byte size into RSS before any
    structural check can refuse it. The unpickler's pre-scan counts opcodes
    before anything is built; this names the refusal on the artifact channel.
    """

    return ArtifactError(
        f"Bundle metadata {metadata_path} exceeds the "
        f"{_MAX_METADATA_PKL_OPCODES}-opcode allocation ceiling; refusing "
        "to load a structurally implausible artifact.",
        code=code,
    )


def _python_major_mismatch(manifest: Manifest) -> bool:
    """Return whether the manifest's Python major version differs from runtime.

    Parameters
    ----------
    manifest:
        Parsed bundle manifest.

    Returns
    -------
    bool
        True when manifest and runtime major versions differ.
    """

    try:
        return int(manifest.python_version.split(".", maxsplit=1)[0]) != sys.version_info.major
    except ValueError:
        return False


def _load_safetensors_file(
    blob_path: Path,
    map_location: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Load one safetensors blob with a TorchLens-specific install hint.

    Parameters
    ----------
    blob_path:
        Blob file path to load.
    map_location:
        Target device for decoded tensors.

    Returns
    -------
    dict[str, torch.Tensor]
        Loaded tensor mapping from the safetensors file.

    Raises
    ------
    ArtifactError
        If the safetensors backend is unavailable.
    """

    try:
        return load_file(blob_path, device=str(map_location))
    except ImportError as exc:
        raise ArtifactError(
            "Portable bundle load requires the safetensors backend. Install safetensors>=0.4."
        ) from exc
    except (OSError, SafetensorError, ValueError) as exc:
        raise ArtifactError(f"Failed to read safetensors blob at {blob_path}.") from exc


def _make_backup_path(bundle_path: Path) -> Path:
    """Create the sibling backup path used during overwrite replacement.

    Parameters
    ----------
    bundle_path:
        Final bundle directory path.

    Returns
    -------
    Path
        Backup directory path.
    """

    return bundle_path.parent / f"{bundle_path.name}.bak.{uuid.uuid4().hex}"


def _run_save_recovery(
    tmp_path: Path,
    backup_path: Path | None,
    bundle_path: Path,
    *,
    primary: BaseException,
) -> None:
    """Run the failed-save bookkeeping without ever masking ``primary``.

    The PARTIAL mark and the backup restore are the two contracts the save
    handlers exist to guarantee, but running them unguarded meant a
    rollback-time failure (e.g. ENOSPC while writing the sentinel) replaced
    the primary exception -- a Ctrl-C was reported as an ordinary I/O error --
    AND skipped the backup restore, stranding the pre-overwrite bundle under
    its ``.bak.<uuid>`` name. Each step is independently best-effort;
    recovery failures are disclosed via warning (and ``add_note`` where the
    runtime has it), never raised over the primary.
    """

    def _disclose(step: str, failure: BaseException) -> None:
        """Note and warn that recovery ``step`` failed, never raising over ``primary``."""

        detail = (
            f"bundle-save recovery step '{step}' itself failed "
            f"({type(failure).__name__}: {failure}); the primary error is re-raised "
            f"unchanged. Recovery debris may remain next to {bundle_path}."
        )
        note = getattr(primary, "add_note", None)  # py3.11+; 3.10 floor lacks it
        if callable(note):
            with contextlib.suppress(Exception):
                note(detail)
        with contextlib.suppress(Exception):
            warnings.warn(detail, TorchLensWarning, stacklevel=3)

    try:
        _mark_partial(tmp_path, reason=type(primary).__name__)
    except Exception as failure:
        _disclose("mark-partial", failure)
    try:
        if backup_path is not None and not bundle_path.exists() and backup_path.exists():
            _restore_backup(backup_path, bundle_path)
    except Exception as failure:
        _disclose("backup-restore", failure)


def _remove_path(path: Path) -> None:
    """Remove a backup path after a successful overwrite replacement.

    Parameters
    ----------
    path:
        Path to remove.
    """

    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
