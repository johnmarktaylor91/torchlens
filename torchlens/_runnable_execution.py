"""Transactional execution providers for the unified :meth:`Trace.run` surface."""

# ruff: noqa: F401

from __future__ import annotations

import dataclasses
import math
import struct
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence, Set as AbstractSet
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from itertools import count
from typing import Any, cast

import numpy as np
import torch

from . import (
    _runnable_attestation as _runnable_attestation,
    _runnable_call_arguments as _runnable_call_arguments,
    _runnable_call_outputs as _runnable_call_outputs,
    _runnable_input_aliases as _runnable_input_aliases,
    _runnable_input_metadata as _runnable_input_metadata,
    _runnable_input_sites as _runnable_input_sites,
    _runnable_output_contracts as _runnable_output_contracts,
    _runnable_path_faithfulness as _runnable_path_faithfulness,
    _runnable_providers as _runnable_providers,
    _runnable_state_context as _runnable_state_context,
    _runnable_transaction as _runnable_transaction,
    _runnable_verification as _runnable_verification,
    _runnable_witness_contracts as _runnable_witness_contracts,
    _state,
)
from ._io._torch_symbols import torch_attr
from ._runnable_state import (
    _INPUT_STRUCTURE_SITE_PREFIX,
    _OUTPUT_COUNT_FLOOR,
    _STATE_METADATA_FACT_SITE_PREFIX,
    PreparedRunnableState,
    RunResourceCeiling,
    _allocation_budget_bytes,
    _guarded_defensive_materialize,
    prepare_runnable_state,
    runnable_tensor_byte_digest,
    state_metadata_full_violations,
    validate_run_seed,
)
from ._split_rebind import (
    rebind_contextmanager as _rebind_contextmanager,
    rebind_function as _rebind_function,
    rebind_lru_cache as _rebind_lru_cache,
)
from .errors import (
    NumericAttestationError,
    PathDivergenceError,
    ReattachError,
    RunCapabilityUnavailableError,
    RunPreconditionError,
    RuntimeSignatureDriftError,
)
from .intervention.replay import _CallConeNode, _walk_call_cone
from .ir.container import (
    CONTAINER_KIND_CAPABILITIES,
    ContainerReconstructionError,
    ContainerSpec,
    _reconstruction_would_substitute_plain,
    namedtuple_type_can_carry_instance_state,
    rebuild_container_from_spec,
    reconstruction_is_lossy_by_type,
    resolve_container_type,
)
from .runnable import (
    NONDETERMINISTIC_SOURCE_VOCABULARY,
    ActivationPayloadLayerDescriptor,
    ActivationPayloadMember,
    CallableRegistryEntry,
    ContractCheck,
    ControlWitness,
    ControlWitnessKind,
    DivergencePolicy,
    InputAttestationFingerprint,
    LiteralAtom,
    LiteralAtomKind,
    LiteralMapping,
    LiteralSequence,
    LiteralSequenceKind,
    LiteralSlice,
    LiteralTorchSymbol,
    LiteralTupleKey,
    NonTensorLiteral,
    NumericAttestationStatus,
    PathFaithfulness,
    ReadinessReport,
    ReadinessStatus,
    RunnableCallDescriptor,
    RunnableDiagnostic,
    RunnableErrorCode,
    RunProvider,
    RunReport,
    RunResult,
    SparseRunDescriptor,
    StateSlotRole,
    StateSource,
    TensorSlotDescriptor,
    TensorSlotRole,
    WitnessCompleteness,
    WitnessGapKind,
    derived_witness_completeness,
    is_mode_sensitive_qualname,
    mark_trace_path_status,
)
from .utils._torch_compat import tensor_has_named_dims, tensor_version_or_none
from .utils.alias_footprint import touched_bytes_relation
from .utils.rng import (
    aten_qualname_is_seeded_rng,
    deterministic_fill_governs,
    qualname_is_uninit_growth_resize,
    qualname_is_uninit_size_gated_alloc,
    qualname_is_uninit_total_writer,
    qualname_is_uninitialized_alloc,
    restore_host_rng,
    snapshot_host_rng,
    uninit_new_call_is_size_form,
)

_RUN_FORK_COUNTER = count(1)


_MODEL_INPUT_LITERAL_SITE_PREFIX = "model_input_literal:"
"""``site_label`` prefix marking a witnessed non-tensor model-input leaf."""

_MODEL_INPUT_LITERAL_FACT_KEY = "model_input_literal"
"""Discriminator key present in every non-tensor model-input leaf fact."""


_MODEL_INPUT_METADATA_SITE_PREFIX = "model_input_metadata:"
"""``site_label`` prefix marking a witnessed model-input metadata-predicate read."""

_MODEL_INPUT_METADATA_FACT_KEY = "model_input_metadata"
"""Discriminator key present in every model-input metadata-predicate fact."""

_INPUT_DERIVED_LAYOUT_FACT_NAME = "derived_layout_read"
"""Synthetic envelope fact: a layout-trio read on an activation whose value DAG roots at the
owning input site (r73 F1). Carries the leaf's capture-time stride tuple. Consumed by
:func:`_input_derived_layout_stale` as a run-time UNVERIFIABLE ceiling -- NEVER compared in
:func:`_input_metadata_contract_checks`: a changed input layout does not PROVE the derived
intermediate's layout differs (a ``reshape`` in between can canonicalize it), so the honest
verdict is "cannot verify", not an observed DIVERGED contradiction (r35 three-state rule).
Producer mirror: ``torchlens.backends.torch.completeness_witness.INPUT_DERIVED_LAYOUT_FACT_NAME``."""


_VIEW_OP_QUALNAMES = frozenset(
    {
        # Storage-sharing view ops whose output aliases an input tensor's storage (r29-C3, F2).
        # Deliberately OVER-broad: some entries here also cover copy variants -- that only
        # widens the fail-closed gate, which fires ONLY when runtime inputs actually alias, so
        # a false positive here can never over-trigger an ordinary (non-aliased-input) run.
        "__getitem__",
        "getitem",
        "select",
        "slice",
        "narrow",
        "narrow_copy",
        "view",
        "view_as",
        "_view",
        "reshape",
        "reshape_as",
        "transpose",
        "t",
        "permute",
        "squeeze",
        "unsqueeze",
        "expand",
        "expand_as",
        "broadcast_to",
        "flatten",
        "unflatten",
        "ravel",
        "unfold",
        "diagonal",
        "diagonal_scatter",
        "movedim",
        "moveaxis",
        "swapaxes",
        "swapdims",
        "detach",
        "as_strided",
        "split",
        "split_with_sizes",
        "chunk",
        "tensor_split",
        "hsplit",
        "vsplit",
        "dsplit",
        "unbind",
        "real",
        "imag",
        "view_as_real",
        "view_as_complex",
        "adjoint",
        "alias",
        "contiguous",
        "indices",
        "values",
    }
)
"""Torch op qualnames whose output can SHARE STORAGE with a tensor input (r29-C3, F2)."""


# r37 INV-2: the alias/overlap proof engine lives in ``utils.tensor_utils`` --
# absolute, device-scoped byte intervals, three-valued relation, pure-integer
# enumeration. The former local implementation keyed "same memory" on
# ``untyped_storage().data_ptr()`` EQUALITY, which mis-proved disjointness for
# overlapping views of one external buffer (``torch.from_numpy(arr[:6])`` vs
# ``arr[2:8]`` -- distinct torch storages, genuinely shared host memory; hon1_1).
# No local reimplementation or pointer-equality shortcut may reappear here.


_FINGERPRINT_ALIGNMENT_MODULUS = 16
"""Data-pointer alignment-class modulus for the physical input fingerprint.

H_B_RESOLUTION R2: fp16 CUDA conv kernel selection/reduction order is keyed to the
data pointer's alignment class (a misaligned offset view vs an aligned buffer), so
the fingerprint records ``data_ptr() % 16`` of the value that actually seeds
execution and attestation goes ``not_applicable`` on any mismatch instead of
false-tripping the byte tripwire. The modulus is the vector-width class boundary:
every fresh torch allocation (capture and replay both execute fresh clones) is at
least 16-byte aligned, so equal-basis executions always agree, while an offset
view that changes the executed alignment class is caught.
"""


# r55 C3 -> r57 C3 (free_1 HIGH / sec_1 MED / corr_2 HIGH): op-agnostic
# allocation-bomb preflight, with the size-driving-op ALLOWLIST DELETED.
#
# The op-execution literal-argument path is the seam r53 free_1 missed: a hostile
# ``.tlspec`` can edit one plaintext ``manifest.json`` numeric literal so a
# taken-path call (``torch.zeros(10**12)``/``F.pad(x, [10**12])``/
# ``hann_window(10**11)``/``interpolate(scale_factor=1e12)``) drives a
# multi-terabyte allocation and OOM-kills the default
# ``tl.load(path).run(inputs)`` victim. The recorded-output-slot bound
# (``_preflight_run_allocation`` at run-prep) catches the *self-consistent* tamper
# (literal AND output slot both huge) but NOT the literal-only tamper (output slot
# left honest/small), and per-arg ``.to("meta")`` never forces a *factory* op (no
# tensor operand) onto meta.
#
# r55 gated the projection behind a hand-maintained ``_SIZE_DRIVING_QUALNAME_TAILS``
# allowlist; that allowlist re-opened the class at every op it forgot (r56 found
# ``pad``/``constant_pad_nd``/``fold``/``*_window``/``tril_indices``/``one_hot``
# missing, plus a float ``interpolate(scale_factor=...)`` slipping the integer-only
# sub-gate). The projection *mechanism* was ALREADY op-agnostic; only the GATE was
# enumerated. r57 DELETES the allowlist entirely: ``FakeTensorMode`` projects the
# output shape/bytes of EVERY taken-path call carrying a non-bool numeric literal
# WITHOUT allocating (fake tensors never allocate -- factory ops included), and a
# projected over-budget NEW allocation is refused BEFORE the real allocator runs.
# "Size-relevance" is decided STRUCTURALLY by the projection, never guessed from an
# op-name family list. Pure views / input-returning / in-place ops are excluded
# structurally too (their fake output storage aliases a fake input storage, so they
# contribute zero NEW bytes). It is the primary layer; the run-prep
# recorded-output bound is layer 2, and both fail OPEN on a projection that raises.


class _ProjectionCountExceeded(Exception):
    """Private sentinel: a projection realized more fake outputs than the ceiling (r59).

    Raised from ``_CountBoundedFakeTensorMode.__torch_dispatch__`` DURING fanout
    construction (before the full fake tree exists), so a huge output-count literal can
    never self-DoS the projection. It is caught BEFORE ``_preflight_call_allocation``'s
    generic ``except Exception`` fail-open and converted to a typed refusal -- an honest
    op never realizes more than ``max(recorded * 8, 4096)`` outputs, so a breach is a
    hard faithfulness divergence, not a missing fake implementation. Deliberately NOT a
    ``RuntimeError`` subclass so it is caught by its own clause ahead of the
    allocation-classed / fail-open handling.
    """


class _UnreadableProjectedOutput(Exception):
    """Private sentinel: a projected output's allocation size could not be read.

    Raised from ``_new_allocation_bytes`` when ``numel()``/``element_size()``/
    ``device`` raises on a projected output tensor (grind-r5 b7 R22, sol
    HIGH): the old ``except Exception: continue`` charged the unknown
    allocation as ZERO bytes, contradicting the helper's own fail-closed
    contract ("an output whose storage identity cannot be read is CHARGED")
    and letting the real call allocate before any backstop. Converted by the
    caller to a typed ``op_allocation_preflight`` refusal. Deliberately NOT a
    ``RuntimeError`` subclass, for the same catch-ordering reason as
    ``_ProjectionCountExceeded``.
    """


# Allocator-death signatures on a raw ``RuntimeError`` message (r59 section 2.4). A
# projection or real call that dies of allocation cannot fail OPEN: the real op's
# identical prelude would die the same way. These convert to a typed refusal.
_ALLOCATOR_SIGNATURES = (
    "std::bad_alloc",
    "DefaultCPUAllocator",
    "can't allocate memory",
    "CUDA out of memory",
    "CUDACachingAllocator",
)


_UNBOUND_STATE_ESCAPE_SITE_PREFIX = "unbound_state_escape:"
"""``site_label`` prefix marking a witnessed unbound state (buffer/param) escape."""

_UNBOUND_STATE_ESCAPE_FACT_KEY = "unbound_state_escape"
"""Discriminator key present in every unbound-state escape fact."""

# Matmul-family qualname tails whose BLAS backends pick a reduction order that
# depends on tensor memory layout and grad/inference dispatch context. Capture
# records these ops under autograd; the run path recomputes them under no-grad
# ``pause_logging()`` isolation, so their outputs can legitimately differ from
# the archived bytes by ~1 dtype ULP without any corruption. This set is
# deliberately NARROW -- only the reduction kernels proven layout/grad-sensitive
# -- so the byte-exact tripwire stays armed for every other op.
_LAYOUT_SENSITIVE_BLAS_QUALNAMES: frozenset[str] = frozenset(
    {
        "linear",
        "matmul",
        "mm",
        "bmm",
        "mv",
        "dot",
        "vdot",
        "inner",
        "outer",
        "ger",
        "addmm",
        "addbmm",
        "baddbmm",
        "addmv",
        "addr",
        "einsum",
        "tensordot",
    }
)


_UNINIT_SOURCE_KIND = "uninitialized_alloc"
_SEEDED_SOURCE_KIND = "seeded_rng"
_HOST_RNG_SOURCE_KIND = "host_rng"

_UNINIT_SANITIZER_NAMESPACES = frozenset({"torch", "torch.Tensor", "torch.nn.functional"})
"""Trusted namespaces whose ``out=`` convention is a total destination overwrite."""


_MODULE_TRAINING_MODE_SITE_PREFIX = "module_training_mode:"
"""``site_label`` prefix marking the declared capture-time per-module train/eval mode."""


_MAX_DECODE_NESTING_DEPTH = 200
"""Defensive run-side literal-decode nesting ceiling (r55 C4).

W3's load-side parser (``_io/runnable_load._parse_literal``) already bounds
descriptor nesting to the SAME ceiling at parse, so a legitimately-loaded
descriptor never approaches this here. This is the belt-and-suspenders guard on
the run-side decoder: an over-depth decode raises ``ValueError`` -- routed to the
same typed refusal surface as any other malformed literal -- rather than an
uncaught ``RecursionError``. 200 sits far above any real literal nesting and
below the interpreter's default recursion crash depth.
"""


# POSITIVE allowlist of the torch symbolic constant *types* a forward op
# legitimately takes as a literal argument. A loaded bundle is untrusted input,
# so decoding an arbitrary ``torch`` attribute by name would otherwise admit whole
# submodules (``torch.serialization`` / ``torch.os``) or any other non-callable
# attribute. These are the only symbolic literals the encoder ever emits
# (``_torch_symbol_qualname``): dtype / layout / memory_format instances, plus
# qscheme and the ``torch.Size`` type -- everything else is denied by construction.
_ALLOWED_TORCH_SYMBOL_TYPES: tuple[type, ...] = (
    torch.dtype,
    torch.layout,
    torch.memory_format,
    torch.qscheme,
)


__all__ = ["raise_analysis_run_unavailable", "run_live_trace", "run_loaded_sparse_trace"]

# Private implementation slices; public names remain owned by this module.
run_loaded_sparse_trace = _rebind_function(_runnable_providers.run_loaded_sparse_trace, globals())
_seeded_fork_devices = _rebind_function(_runnable_providers._seeded_fork_devices, globals())
_seed_run_generators = _rebind_function(_runnable_providers._seed_run_generators, globals())
_host_rng_unreproduced = _rebind_function(_runnable_providers._host_rng_unreproduced, globals())
_finalize_provider_run = _rebind_function(_runnable_providers._finalize_provider_run, globals())
_execute_loaded_sparse_transaction = _rebind_function(
    _runnable_transaction._execute_loaded_sparse_transaction, globals()
)
run_live_trace = _rebind_function(_runnable_transaction.run_live_trace, globals())
_restore_declared_state_or_mark = _rebind_function(
    _runnable_transaction._restore_declared_state_or_mark, globals()
)
# These names resolve in THIS module's globals at runtime inside rebound
# functions (same mechanism as _INPUT_CHECK_UNAVAILABLE).
_RunUntilPlan = _runnable_transaction._RunUntilPlan
from .runnable import (  # noqa: E402
    RUN_TRUNCATION_CAUSE_VOCABULARY,
    RUN_TRUNCATION_REGIME_VOCABULARY,
    RunTruncation,
)

_resolve_run_until_plan = _rebind_function(_runnable_transaction._resolve_run_until_plan, globals())
_validated_until_tokens = _rebind_function(_runnable_transaction._validated_until_tokens, globals())
_resolve_until_site = _rebind_function(_runnable_transaction._resolve_until_site, globals())
_until_stop_raw_index = _rebind_function(_runnable_transaction._until_stop_raw_index, globals())
_until_execution_partition = _rebind_function(
    _runnable_transaction._until_execution_partition, globals()
)
_until_dependency_closure = _rebind_function(
    _runnable_transaction._until_dependency_closure, globals()
)
_run_truncation_record = _rebind_function(_runnable_transaction._run_truncation_record, globals())
_loaded_until_cut = _rebind_function(_runnable_transaction._loaded_until_cut, globals())
_LiveRunOptions = _runnable_transaction._LiveRunOptions
_LiveFinalizeContext = _runnable_transaction._LiveFinalizeContext
_require_live_source_model = _rebind_function(
    _runnable_transaction._require_live_source_model, globals()
)
_attempt_live_forward = _rebind_function(_runnable_transaction._attempt_live_forward, globals())
_live_graph_change_check = _rebind_function(
    _runnable_transaction._live_graph_change_check, globals()
)
_settle_live_graph_change = _rebind_function(
    _runnable_transaction._settle_live_graph_change, globals()
)
_split_live_inputs = _rebind_function(_runnable_transaction._split_live_inputs, globals())
_install_live_until_latch = _rebind_function(
    _runnable_transaction._install_live_until_latch, globals()
)
_live_nondeterministic_sources = _rebind_function(
    _runnable_transaction._live_nondeterministic_sources, globals()
)
_live_readiness_report = _rebind_function(_runnable_transaction._live_readiness_report, globals())
_finalize_truncated_live_run = _rebind_function(
    _runnable_transaction._finalize_truncated_live_run, globals()
)
_finalize_full_live_run = _rebind_function(_runnable_transaction._finalize_full_live_run, globals())
_live_runtime_input_leaves = _rebind_function(
    _runnable_transaction._live_runtime_input_leaves, globals()
)
# The classifier-unavailable sentinel must resolve in THIS module's globals:
# the rebound function body looks its name up here (R22-2).
_INPUT_CHECK_UNAVAILABLE = _runnable_input_sites._INPUT_CHECK_UNAVAILABLE
_first_failed_live_input_check = _rebind_function(
    _runnable_input_sites._first_failed_live_input_check, globals()
)
raise_analysis_run_unavailable = _rebind_function(
    _runnable_input_sites.raise_analysis_run_unavailable, globals()
)
_require_loaded_sparse_provider = _rebind_function(
    _runnable_input_sites._require_loaded_sparse_provider, globals()
)
_model_input_literal_facts = _rebind_function(
    _runnable_input_sites._model_input_literal_facts, globals()
)
_is_model_input_literal_witness = _rebind_function(
    _runnable_input_sites._is_model_input_literal_witness, globals()
)
_model_input_metadata_facts = _rebind_function(
    _runnable_input_sites._model_input_metadata_facts, globals()
)
_is_model_input_metadata_witness = _rebind_function(
    _runnable_input_sites._is_model_input_metadata_witness, globals()
)
_runtime_input_metadata_value = _rebind_function(
    _runnable_input_sites._runtime_input_metadata_value, globals()
)
_fact_path_tuple_or_none = _rebind_function(
    _runnable_input_sites._fact_path_tuple_or_none, globals()
)
_input_metadata_contract_checks = _rebind_function(
    _runnable_input_sites._input_metadata_contract_checks, globals()
)
_input_derived_layout_stale = _rebind_function(
    _runnable_input_sites._input_derived_layout_stale, globals()
)
_inventory_site_positions = _rebind_function(
    _runnable_input_sites._inventory_site_positions, globals()
)
_model_input_arity_positions = _rebind_function(
    _runnable_input_sites._model_input_arity_positions, globals()
)
_runtime_top_level_positions = _rebind_function(
    _runnable_input_sites._runtime_top_level_positions, globals()
)
_top_level_input_site_contract_checks = _rebind_function(
    _runnable_input_metadata._top_level_input_site_contract_checks, globals()
)
_literal_leaf_equal = _rebind_function(_runnable_input_metadata._literal_leaf_equal, globals())
_input_literal_contract_checks = _rebind_function(
    _runnable_input_metadata._input_literal_contract_checks, globals()
)
_bind_runtime_inputs = _rebind_function(_runnable_input_metadata._bind_runtime_inputs, globals())
_runtime_mirror_clone = _rebind_function(_runnable_input_metadata._runtime_mirror_clone, globals())
_model_input_version_closure = _rebind_function(
    _runnable_input_metadata._model_input_version_closure, globals()
)
_model_input_storage_closure = _rebind_function(
    _runnable_input_metadata._model_input_storage_closure, globals()
)
_touched_bytes_relation = _rebind_function(
    _runnable_input_metadata._touched_bytes_relation, globals()
)
_input_alias_topology_checks = _rebind_function(
    _runnable_input_aliases._input_alias_topology_checks, globals()
)
_clone_state_values = _rebind_function(_runnable_input_aliases._clone_state_values, globals())
_snapshot_input_byte_digests = _rebind_function(
    _runnable_input_aliases._snapshot_input_byte_digests, globals()
)
_snapshot_input_fingerprints = _rebind_function(
    _runnable_input_aliases._snapshot_input_fingerprints, globals()
)
build_input_attestation_fingerprint = _rebind_function(
    _runnable_input_aliases.build_input_attestation_fingerprint, globals()
)
_positions_are_mixed = _rebind_function(_runnable_input_aliases._positions_are_mixed, globals())
_split_mixed_inputs = _rebind_function(_runnable_input_aliases._split_mixed_inputs, globals())
_input_site_value = _rebind_function(_runnable_input_aliases._input_site_value, globals())
_type_strict_path = _rebind_function(_runnable_input_aliases._type_strict_path, globals())
_input_tree_contract_checks = _rebind_function(
    _runnable_input_aliases._input_tree_contract_checks, globals()
)
_runtime_nontensor_leaf_paths = _rebind_function(
    _runnable_input_aliases._runtime_nontensor_leaf_paths, globals()
)
_input_nontensor_tree_contract_checks = _rebind_function(
    _runnable_input_aliases._input_nontensor_tree_contract_checks, globals()
)
_state_contract_checks = _rebind_function(_runnable_state_context._state_contract_checks, globals())
_allowed_state_roles = _rebind_function(_runnable_state_context._allowed_state_roles, globals())
_pre_call_contract_checks = _rebind_function(
    _runnable_state_context._pre_call_contract_checks, globals()
)
_context_unavailable_error = _rebind_function(
    _runnable_state_context._context_unavailable_error, globals()
)
_ambient_execution_context_restored = _rebind_contextmanager(
    _runnable_state_context._ambient_execution_context_restored, globals()
)
_call_execution_context_entered = _rebind_contextmanager(
    _runnable_state_context._call_execution_context_entered, globals()
)
_is_allocator_death = _rebind_function(_runnable_state_context._is_allocator_death, globals())
_fake_tensor_mode_class = _rebind_lru_cache(
    _runnable_state_context._fake_tensor_mode_class, globals(), maxsize=1
)
_count_fake_tensor_leaves = _rebind_function(
    _runnable_state_context._count_fake_tensor_leaves, globals()
)
_count_bounded_fake_tensor_mode_class = _rebind_lru_cache(
    _runnable_state_context._count_bounded_fake_tensor_mode_class, globals(), maxsize=1
)
_has_numeric_literal = _rebind_function(_runnable_call_arguments._has_numeric_literal, globals())
_has_tensor_operand = _rebind_function(_runnable_call_arguments._has_tensor_operand, globals())
_projection_required_by_arguments = _rebind_function(
    _runnable_call_arguments._projection_required_by_arguments, globals()
)
_tree_to_fake = _rebind_function(_runnable_call_arguments._tree_to_fake, globals())
_fake_tensor_storage_id = _rebind_function(
    _runnable_call_arguments._fake_tensor_storage_id, globals()
)
_input_storage_ids = _rebind_function(_runnable_call_arguments._input_storage_ids, globals())
_new_allocation_bytes = _rebind_function(_runnable_call_arguments._new_allocation_bytes, globals())
_preflight_call_allocation = _rebind_function(
    _runnable_call_arguments._preflight_call_allocation, globals()
)
_execute_sparse_call = _rebind_function(_runnable_call_arguments._execute_sparse_call, globals())
_populate_source_slots = _rebind_function(
    _runnable_call_arguments._populate_source_slots, globals()
)
_write_argument = _rebind_function(_runnable_call_arguments._write_argument, globals())
_write_path = _rebind_function(_runnable_call_arguments._write_path, globals())
_resolve_setter_output = _rebind_function(
    _runnable_call_arguments._resolve_setter_output, globals()
)
_bind_call_outputs = _rebind_function(_runnable_call_outputs._bind_call_outputs, globals())
_mutation_contract_checks = _rebind_function(
    _runnable_call_outputs._mutation_contract_checks, globals()
)
_mutation_target_slot_id = _rebind_function(
    _runnable_call_outputs._mutation_target_slot_id, globals()
)
_out_argument_slot_id = _rebind_function(_runnable_call_outputs._out_argument_slot_id, globals())
_reconstruct_output = _rebind_function(_runnable_call_outputs._reconstruct_output, globals())
_output_container_spec = _rebind_function(_runnable_call_outputs._output_container_spec, globals())
_output_not_reproduced = _rebind_function(_runnable_call_outputs._output_not_reproduced, globals())
_container_spec_reconstruction_lossy = _rebind_function(
    _runnable_call_outputs._container_spec_reconstruction_lossy, globals()
)
_spec_node_reconstruction_lossy = _rebind_function(
    _runnable_call_outputs._spec_node_reconstruction_lossy, globals()
)
_fresh_bare_tensor_root = _rebind_function(
    _runnable_call_outputs._fresh_bare_tensor_root, globals()
)
_reconstruct_live_output = _rebind_function(
    _runnable_output_contracts._reconstruct_live_output, globals()
)
_diagnose_live_output = _rebind_function(
    _runnable_output_contracts._diagnose_live_output, globals()
)
_container_from_paths = _rebind_function(
    _runnable_output_contracts._container_from_paths, globals()
)
_write_output_path = _rebind_function(_runnable_output_contracts._write_output_path, globals())
_call_witness_checks = _rebind_function(_runnable_output_contracts._call_witness_checks, globals())
_post_execution_contract_checks = _rebind_function(
    _runnable_output_contracts._post_execution_contract_checks, globals()
)
_conditional_arm_check = _rebind_function(
    _runnable_output_contracts._conditional_arm_check, globals()
)
_input_structure_positions = _rebind_function(
    _runnable_output_contracts._input_structure_positions, globals()
)
_input_structure_witness_check = _rebind_function(
    _runnable_output_contracts._input_structure_witness_check, globals()
)
_structure_witness_check = _rebind_function(
    _runnable_output_contracts._structure_witness_check, globals()
)
_runtime_input_for_structure_witness = _rebind_function(
    _runnable_output_contracts._runtime_input_for_structure_witness, globals()
)
_raw_runtime_output = _rebind_function(_runnable_output_contracts._raw_runtime_output, globals())
_registered_flatten_children = _rebind_function(
    _runnable_output_contracts._registered_flatten_children, globals()
)
_codec_component = _rebind_function(_runnable_output_contracts._codec_component, globals())
_tensor_leaf_paths = _rebind_function(_runnable_witness_contracts._tensor_leaf_paths, globals())
_canonicalize_structseq_output_paths = _rebind_function(
    _runnable_witness_contracts._canonicalize_structseq_output_paths, globals()
)
_canonicalize_structseq_output_path = _rebind_function(
    _runnable_witness_contracts._canonicalize_structseq_output_path, globals()
)
_recorded_structseq_output_type_matches = _rebind_function(
    _runnable_witness_contracts._recorded_structseq_output_type_matches, globals()
)
_call_has_recorded_torch_structseq_output = _rebind_function(
    _runnable_witness_contracts._call_has_recorded_torch_structseq_output, globals()
)
_container_leaf_paths = _rebind_function(
    _runnable_witness_contracts._container_leaf_paths, globals()
)
_container_kind = _rebind_function(_runnable_witness_contracts._container_kind, globals())
_is_hf_model_output = _rebind_function(_runnable_witness_contracts._is_hf_model_output, globals())
_container_field_names = _rebind_function(
    _runnable_witness_contracts._container_field_names, globals()
)
_torch_structseq_field_names = _rebind_function(
    _runnable_witness_contracts._torch_structseq_field_names, globals()
)
_scalar_literal_equal = _rebind_function(
    _runnable_witness_contracts._scalar_literal_equal, globals()
)
_tensor_derived_scalar_witness_slot_ids = _rebind_function(
    _runnable_witness_contracts._tensor_derived_scalar_witness_slot_ids, globals()
)
_tensor_derived_scalar_stale = _rebind_function(
    _runnable_witness_contracts._tensor_derived_scalar_stale, globals()
)
_is_unbound_state_escape_witness = _rebind_function(
    _runnable_witness_contracts._is_unbound_state_escape_witness, globals()
)
_is_state_metadata_fact_witness = _rebind_function(
    _runnable_witness_contracts._is_state_metadata_fact_witness, globals()
)
_unbound_state_escape_stale = _rebind_function(
    _runnable_witness_contracts._unbound_state_escape_stale, globals()
)
_path_faithfulness = _rebind_function(_runnable_path_faithfulness._path_faithfulness, globals())
_run_report = _rebind_function(_runnable_path_faithfulness._run_report, globals())
_numeric_attestation_check = _rebind_function(
    _runnable_path_faithfulness._numeric_attestation_check, globals()
)
_member_producer_is_layout_sensitive_blas = _rebind_function(
    _runnable_path_faithfulness._member_producer_is_layout_sensitive_blas, globals()
)
_within_layout_reduction_tolerance = _rebind_function(
    _runnable_path_faithfulness._within_layout_reduction_tolerance, globals()
)
_is_benign_layout_nonreproducible = _rebind_function(
    _runnable_path_faithfulness._is_benign_layout_nonreproducible, globals()
)
_is_benign_downstream_nonreproducible = _rebind_function(
    _runnable_attestation._is_benign_downstream_nonreproducible, globals()
)
_has_journaled_buffer_activation_member = _rebind_function(
    _runnable_attestation._has_journaled_buffer_activation_member, globals()
)
_has_out_mutated_activation_member = _rebind_function(
    _runnable_attestation._has_out_mutated_activation_member, globals()
)
_raw_activation_slot_ids = _rebind_function(
    _runnable_attestation._raw_activation_slot_ids, globals()
)
_descriptor_has_seeded_rng = _rebind_function(
    _runnable_attestation._descriptor_has_seeded_rng, globals()
)
_descriptor_has_nondeterministic_rng = _rebind_function(
    _runnable_attestation._descriptor_has_nondeterministic_rng, globals()
)
_decoded_positional_literals = _rebind_function(
    _runnable_attestation._decoded_positional_literals, globals()
)
_nondeterministic_value_sources = _rebind_function(
    _runnable_attestation._nondeterministic_value_sources, globals()
)
_uninit_taint_reaches = _rebind_function(_runnable_attestation._uninit_taint_reaches, globals())
_control_witness_source_slot_ids = _rebind_function(
    _runnable_attestation._control_witness_source_slot_ids, globals()
)
_declared_nondeterministic_sources = _rebind_function(
    _runnable_attestation._declared_nondeterministic_sources, globals()
)
_is_mode_sensitive_qualname = _rebind_function(
    _runnable_attestation._is_mode_sensitive_qualname, globals()
)
_descriptor_has_mode_sensitive_op = _rebind_function(
    _runnable_attestation._descriptor_has_mode_sensitive_op, globals()
)
_descriptor_declares_training_mode = _rebind_function(
    _runnable_attestation._descriptor_declares_training_mode, globals()
)
_mode_sensitive_op_unwitnessed = _rebind_function(
    _runnable_attestation._mode_sensitive_op_unwitnessed, globals()
)
_call_consumes_seeded_rng = _rebind_function(
    _runnable_verification._call_consumes_seeded_rng, globals()
)
_is_dropout_qualname = _rebind_function(_runnable_verification._is_dropout_qualname, globals())
_dropout_call_draws_rng = _rebind_function(
    _runnable_verification._dropout_call_draws_rng, globals()
)
_named_literal_values = _rebind_function(_runnable_verification._named_literal_values, globals())
_lacks_recorded_original_input_eligibility = _rebind_function(
    _runnable_verification._lacks_recorded_original_input_eligibility, globals()
)
_attestation_inputs_match = _rebind_function(
    _runnable_verification._attestation_inputs_match, globals()
)
_attestation_state_matches = _rebind_function(
    _runnable_verification._attestation_state_matches, globals()
)
_raise_numeric_attestation_failure = _rebind_function(
    _runnable_verification._raise_numeric_attestation_failure, globals()
)
_contract_check = _rebind_function(_runnable_verification._contract_check, globals())
_first_failed_contract = _rebind_function(_runnable_verification._first_failed_contract, globals())
_raise_failed_contract_as_divergence = _rebind_function(
    _runnable_verification._raise_failed_contract_as_divergence, globals()
)
_raise_first_divergence = _rebind_function(
    _runnable_verification._raise_first_divergence, globals()
)
_raise_monotonic_divergence = _rebind_function(
    _runnable_verification._raise_monotonic_divergence, globals()
)
_decode_literal = _rebind_function(_runnable_verification._decode_literal, globals())
_decode_nonfinite_float_literal = _rebind_function(
    _runnable_verification._decode_nonfinite_float_literal, globals()
)
_decode_torch_symbol = _rebind_function(_runnable_verification._decode_torch_symbol, globals())
_field_getattr = _rebind_function(_runnable_verification._field_getattr, globals())
_value_at_path = _rebind_function(_runnable_verification._value_at_path, globals())
_input_error = _rebind_function(_runnable_verification._input_error, globals())
_op_for_label = _rebind_function(_runnable_verification._op_for_label, globals())
_op_for_slot = _rebind_function(_runnable_verification._op_for_slot, globals())
_run_fork_name = _rebind_function(_runnable_verification._run_fork_name, globals())

_split_namespace = {name: value for name, value in globals().items() if not name.startswith("__")}
for _split_module in (
    _runnable_providers,
    _runnable_transaction,
    _runnable_input_sites,
    _runnable_input_metadata,
    _runnable_input_aliases,
    _runnable_state_context,
    _runnable_call_arguments,
    _runnable_call_outputs,
    _runnable_output_contracts,
    _runnable_witness_contracts,
    _runnable_path_faithfulness,
    _runnable_attestation,
    _runnable_verification,
):
    _split_module.__dict__.update(_split_namespace)
