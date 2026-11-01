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

from torchlens._runnable import ContractCheck, ReadinessReport, RunnableDiagnostic, RunReport

from . import _state
from ._io._torch_symbols import torch_attr
from ._ir.container import (
    CONTAINER_KIND_CAPABILITIES,
    ContainerReconstructionError,
    ContainerSpec,
    _reconstruction_would_substitute_plain,
    namedtuple_type_can_carry_instance_state,
    rebuild_container_from_spec,
    reconstruction_is_lossy_by_type,
    resolve_container_type,
)
from ._runnable import (
    NONDETERMINISTIC_SOURCE_VOCABULARY,
    ActivationPayloadLayerDescriptor,
    ActivationPayloadMember,
    CallableRegistryEntry,
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
    ReadinessStatus,
    RunnableCallDescriptor,
    RunnableErrorCode,
    RunProvider,
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
from ._runnable_retention import _check_unretained_extent, _fail_open_unless_admitted
from ._runnable_state import (
    _INPUT_STRUCTURE_SITE_PREFIX,
    _OUTPUT_COUNT_FLOOR,
    _STATE_METADATA_FACT_SITE_PREFIX,
    PreparedRunnableState,
    RunResourceCeiling,
    _allocation_budget_bytes,
    _guarded_defensive_materialize,
    _slot_device,
    _torch_dtype,
    prepare_runnable_state,
    runnable_tensor_byte_digest,
    state_metadata_full_violations,
    validate_run_seed,
)
from ._utils._torch_compat import tensor_has_named_dims, tensor_version_or_none
from ._utils.alias_footprint import touched_bytes_relation
from ._utils.rng import (
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
from .errors import (
    PathDivergenceError,
    ReattachError,
    ReplayValueMismatchError,
    RunCapabilityUnavailableError,
    RunPreconditionError,
    RuntimeSignatureDriftError,
)
from .intervention.replay import _CallConeNode, _walk_call_cone

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


def refuse_truncated_rerun(trace: Any) -> None:
    """Refuse ``run()`` on a TRUNCATED (``until=``) run product, on every provider.

    L4 3.3.2: the poison bit alone does not refuse a re-run; the transaction's
    inherited-status leg would silently settle UNVERIFIABLE instead, which is
    exactly the silence the capability table forbids.
    """

    if trace.__dict__.get("_run_truncation_skipped_raw_labels") is None:
        return
    raise RunCapabilityUnavailableError(
        "This Trace is a TRUNCATED run product (until=): it covers only "
        "the executed prefix, so re-running it would replay a cut "
        "forward. Run the ORIGINAL source trace instead (with or "
        "without until=).",
        code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
        detection_stage="truncated_result_rerun",
    )


__all__ = [
    "raise_analysis_run_unavailable",
    "refuse_truncated_rerun",
    "run_live_trace",
    "run_loaded_sparse_trace",
]

# Late imports (after this module's own definitions): the split modules import this
# module's classes and constants back, so they must exist before the splits load.
from ._runnable import (  # noqa: E402
    RUN_TRUNCATION_CAUSE_VOCABULARY,
    RUN_TRUNCATION_REGIME_VOCABULARY,
    RunTruncation,
)
from ._runnable_input_sites import (  # noqa: E402
    raise_analysis_run_unavailable,
)
from ._runnable_providers import (  # noqa: E402
    run_loaded_sparse_trace,
)
from ._runnable_transaction import (  # noqa: E402
    run_live_trace,
)
