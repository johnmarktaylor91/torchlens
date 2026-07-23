"""Non-executing state binding and allocation for sparse runnable traces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
import math
from types import MappingProxyType
from typing import Any
import warnings

import torch

from . import _state
from .errors import RunCapabilityUnavailableError, RunPreconditionError, StateBindingError
from .utils._torch_symbols import torch_attr
from .runnable import (
    CANONICAL_INITIALIZER_BY_ROLE,
    RUNNABLE_INITIALIZER_POLICY_VERSION,
    ControlWitnessKind,
    InitializerPolicy,
    RunnableDiagnostic,
    RunnableErrorCode,
    SparseRunDescriptor,
    StateSlotRole,
    StateSource,
    TensorSlotDescriptor,
    TensorSlotRole,
)


@dataclass(frozen=True, slots=True)
class PreparedRunnableState:
    """Run-preflight state values prepared without executing graph operations."""

    slot_values: Mapping[str, torch.Tensor]
    state_source: StateSource
    initializer_policy_version: str | None
    seed: int | None
    random_filled_slot_ids: tuple[str, ...]


TORCH_MANUAL_SEED_MIN = -0x8000_0000_0000_0000
"""Smallest seed torch ``Generator.manual_seed`` accepts (int64 min)."""

TORCH_MANUAL_SEED_MAX = 0xFFFF_FFFF_FFFF_FFFF
"""Largest seed torch ``Generator.manual_seed`` accepts (uint64 max)."""


def validate_run_seed(seed: Any) -> int | None:
    """Validate a user-supplied run seed before any generator work.

    The r77 run door rejected non-``int`` seeds, but two values still escaped
    to raw torch ``RuntimeError`` (r78, hon1 + Sol): ``bool`` (an ``int``
    subclass, so it passed the type check but ``Generator.manual_seed``
    rejects it) and an ``int`` outside torch's accepted
    ``[-0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF]`` long range (pybind
    overflow). This is the ONE canonical guard for every ``manual_seed`` site
    on the run path -- the run door plus the executor/state-initializer
    mirrors -- matching the capture-seed convention
    (``isinstance(seed, int) and not isinstance(seed, bool)``).

    Parameters
    ----------
    seed:
        User-supplied seed value, or ``None`` for unseeded runs.

    Returns
    -------
    int | None
        The validated seed unchanged.

    Raises
    ------
    RunPreconditionError
        Typed ``context_field_invalid`` refusal for a non-int (including
        ``bool``) or out-of-range seed. Transactional: raised before any
        generator state is touched.
    """

    if seed is None:
        return None
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise RunPreconditionError(
            f"run(seed=...) requires an int or None, got {type(seed).__name__}.",
            code=RunnableErrorCode.CONTEXT_FIELD_INVALID.value,
        )
    if not TORCH_MANUAL_SEED_MIN <= seed <= TORCH_MANUAL_SEED_MAX:
        raise RunPreconditionError(
            "run(seed=...) is outside torch's accepted manual_seed range "
            f"[{TORCH_MANUAL_SEED_MIN}, {TORCH_MANUAL_SEED_MAX}], got {seed}.",
            code=RunnableErrorCode.CONTEXT_FIELD_INVALID.value,
        )
    return seed


def snapshot_capture_state(model: object) -> Mapping[str, torch.Tensor] | None:
    """Clone a model's persistent state at the capture execution boundary.

    Parameters
    ----------
    model:
        Live model about to execute the captured forward pass.

    Returns
    -------
    Mapping[str, torch.Tensor] | None
        Detached tensor clones keyed by canonical ``state_dict`` name, or
        ``None`` when the model cannot provide a tensor-only state mapping.

    Notes
    -----
    The clones deliberately retain their original device. Runnable descriptor
    validation records the capture device contract, while serialization later
    performs its ordinary device-neutral transport conversion.
    """

    state_dict_method = getattr(model, "state_dict", None)
    if not callable(state_dict_method):
        return None
    try:
        state = state_dict_method()
    except Exception:
        return None
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        return None
    with _state.pause_logging():
        return MappingProxyType({name: value.detach().clone() for name, value in state.items()})


def snapshot_persistent_buffer_universe(model: object) -> dict[str, dict[str, Any]] | None:
    """Record the persistent-buffer NAME universe and geometry at the capture boundary (r77 F2).

    :func:`snapshot_capture_state` deliberately returns ``None`` for a model whose
    ``state_dict()`` carries ANY non-tensor value (``get_extra_state()``, packed or
    quantized entries) -- the embedded-state comparison basis is missing, and that
    ceiling stays. But the DECLARED persistent-buffer slot universe must not shrink
    with it: a dead-model save whose r75 fallback silently returned dropped
    never-forward-used persistent buffers (BatchNorm's ``num_batches_tracked``) from
    the declared universe, so an honest tensor-only ``load_state_dict`` refused with
    ``state_unexpected_key`` while the live lane bound fine. This record derives the
    SAME name set the live save computes -- ``state_dict()`` names that are buffers
    (present in ``named_buffers``, absent from ``named_parameters``) with tensor
    values -- plus the per-slot geometry the value-free slot drafts need, so the dead
    lane declares identically. No tensor values are retained.

    Returns
    -------
    dict[str, dict[str, Any]] | None
        ``name -> {"shape", "dtype", "device"}`` for every persistent buffer. ``{}``
        when the model exposes no state accessors (a stateless callable positively
        has an empty persistent-buffer universe). ``None`` when ``state_dict()``
        itself fails or is not a mapping -- an UNKNOWN universe the producer must
        refuse loudly, never silently under-declare.
    """

    state_dict_method = getattr(model, "state_dict", None)
    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    if not (callable(state_dict_method) and callable(named_parameters) and callable(named_buffers)):
        return {}
    try:
        state = state_dict_method()
        parameter_names = {str(name) for name, _value in named_parameters(remove_duplicate=False)}
        buffer_names = {str(name) for name, _value in named_buffers(remove_duplicate=False)}
    except Exception:
        return None
    if not isinstance(state, Mapping):
        return None
    with _state.pause_logging():
        universe: dict[str, dict[str, Any]] = {}
        for name, value in state.items():
            if not isinstance(name, str) or name in parameter_names or name not in buffer_names:
                continue
            if not isinstance(value, torch.Tensor):
                continue
            universe[name] = {
                "shape": tuple(int(dim) for dim in value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    return universe


_ADMITTED_STATE_CLASS_CATEGORIES = frozenset({"tensor", "parameter"})
"""Exact-type state admission (r63 C1): ``torch.Tensor`` and ``torch.nn.Parameter`` only.

A ``__torch_function__``/property-overriding tensor SUBCLASS could lie about (or count) every
metadata read below, so classification reads ``type()`` FIRST and performs ZERO tensor-metadata
reads on an unadmitted class.
"""

_STATE_METADATA_BIND_SCOPE: tuple[tuple[str, Any], ...] = (
    ("layout", "torch.strided"),
    ("is_nested", False),
    ("is_quantized", False),
    ("has_named_dims", False),
)
"""LOAD-SURVIVING signature dims (r63 C1 bind gate) and their canonical values.

Inclusion principle: a dim belongs here iff its source-side value SURVIVES oracle-1's default
``load_state_dict(strict=True, assign=False)`` copy into a canonical destination -- either by
steering the fresh oracle (named dims propagate into an unnamed destination) or by making the
default copy RAISE (sparse/mkldnn layouts, nested, quantized sources into a dense destination).
Physical dims (stride/contiguity/memory-format, storage_offset, conj/neg bits) are EXCLUDED
here because default copy NORMALIZES them away: a channels-last or non-contiguous USER source
into a canonical captured slot binds and runs exactly like PyTorch's own ``load_state_dict``.
"""

_STATE_METADATA_PHYSICAL_SCOPE: tuple[tuple[str, Any], ...] = (
    ("is_rowmajor_contiguous", True),
    ("stride_is_default", True),
    ("storage_offset_is_zero", True),
    ("is_conj", False),
    ("is_neg", False),
    # r65 Cluster X: the full input-net-parity dims. Every one is transport/staging-
    # normalized -- the canonical staging clone produces a fresh, non-inference,
    # non-view, leaf, grad-free, tightly-allocated tensor -- so the
    # UNCONDITIONAL staged-runtime tripwire stays sound (pinned by the staging
    # canonicality self-check tests, CPU and CUDA). ``is_shared``/``is_pinned`` are
    # deliberately ABSENT (r67 C3): they are OBSERVED-VALUE read kinds -- the honest
    # producer predicate is the user's one actual accessor return against the
    # device-defined staged canonical (``_state_placement_canonical``), never a
    # pre-forward signature stamp (free-F4: the CUDA-init proof-by-absence stamped
    # canonical False on genuinely pinned XPU/MPS/externally-registered memory). The
    # staged runtime tripwire checks them with real reads on TorchLens's OWN staged
    # clones (``_staged_placement_violations``).
    ("is_inference", False),
    ("is_view", False),
    ("is_leaf", True),
    ("retains_grad", False),
    ("output_nr_is_zero", True),
    ("grad_is_none", True),
    ("storage_nbytes_is_tight", True),
)
"""Destination-owned PHYSICAL signature dims (r63 C1; r65 full mirror) and their canonicals.

These are normalized by transport (the snapshot clone compacts ``storage_offset``,
materializes conj/neg, drops sharing/pinning/inference-ness/view-ness/autograd state, and
allocates tight storage; safetensors re-lays stride) and by the canonical staging clone, so a
recorded path that READ one of them on captured state rests on an unwitnessable constructor
assumption whenever the captured value was non-canonical. They are enforced ESCAPE-GATED at the
producer (only when the dim was actually read on that slot) and unconditionally by the staged
runtime tripwire (staging guarantees canonical form). ``requires_grad`` (+ ``grad_fn``
presence) is deliberately ABSENT: it is a DECLARED-STATE FACT (r65 F-1 ruling) staging
REPRODUCES, never a canonicality dim -- escape-gating it would refuse every frozen model.
"""

_STATE_METADATA_READ_REQUIRED_DIMS: Mapping[str, tuple[str, Any]] = MappingProxyType(
    {
        "contiguous_default": ("is_rowmajor_contiguous", True),
        "stride_exact": ("stride_is_default", True),
        "storage_offset": ("storage_offset_is_zero", True),
        "is_conj": ("is_conj", False),
        "is_neg": ("is_neg", False),
        # r65 Cluster X rows (kinds emitted by the STATE_METADATA_MIRROR dispatch in
        # ``torchlens.backends.torch.completeness_witness``). ``is_shared``/``is_pinned``
        # are ABSENT (r67 C3): observed-value kinds validate through
        # ``_STATE_METADATA_OBSERVED_PLACEMENT_KINDS``, never a signature dim.
        "is_inference": ("is_inference", False),
        "is_view": ("is_view", False),
        "is_leaf": ("is_leaf", True),
        "retains_grad": ("retains_grad", False),
        "output_nr": ("output_nr_is_zero", True),
        "grad_presence": ("grad_is_none", True),
        # r64 F3: ``untyped_storage().nbytes()`` off a state receiver pins the BASE storage
        # byte count; reproducible iff the captured storage was exactly
        # ``numel * element_size`` (a larger-base offset-0-contiguous view is NOT).
        "storage_nbytes": ("storage_nbytes_is_tight", True),
    }
)
"""Witnessed state-metadata READ kinds -> the signature dim each read exposes (r63 C1, r65 X).

``contiguous_default`` (a bare ``is_contiguous()``) is reproducible iff the captured slot was
row-major contiguous; ``stride_exact`` (a ``stride()`` read, ``is_contiguous`` probed with an
explicit ``memory_format=``, or a zero-copy view export -- ``numpy()`` / ``__array__`` /
``__dlpack__`` / ``__cuda_array_interface__`` / ``to_dlpack``, which pin the exact layout with
no accessor call) pins the full stride tuple, reproducible iff the captured stride equals the
canonical dense stride of the shape; the remaining kinds map one-to-one onto their dims. An
UNKNOWN future kind resolves to no entry and fails closed at the consumer. ``_version`` is
deliberately ABSENT: it is a ``refuse_on_any_read`` policy row (r67 C4), never a canonical dim.
``is_shared``/``is_pinned`` are ALSO absent (r67 C3): observed-value kinds, below.
"""


_STATE_METADATA_OBSERVED_PLACEMENT_KINDS = frozenset({"is_shared", "is_pinned"})
"""OBSERVED-VALUE placement read kinds (r67 C3): validated against the user's ONE actual
accessor return, never a pre-forward signature stamp and never a speculative TorchLens
re-read. The producer predicate is device-defined (:func:`_state_placement_canonical`),
evaluated from the slot's inert pre-clone ``device_type`` stamp."""


def _state_placement_canonical(kind: str, device_type: Any) -> bool | None:
    """Return the device-defined staged/oracle placement value for one observed kind.

    ``is_pinned``: a staged/oracle-1 destination is never page-locked (staging clones and
    default ``load_state_dict`` copies allocate ordinary memory) -- canonical ``False`` on
    every device. ``is_shared``: the ``share_memory_()`` CPU transport backing is dropped by
    staging (canonical ``False`` on CPU), while off-CPU ``is_shared()`` is a DEVICE CONSTANT
    ``True`` (device memory is inherently "shared" in torch's sense) that a staged clone on
    the slot device reproduces (r65 regap ruling). Unknown device -> ``None`` (fail closed).
    """

    if not isinstance(device_type, str) or not device_type:
        return None
    if kind == "is_pinned":
        return False
    if kind == "is_shared":
        return device_type != "cpu"
    return None


ORACLE_POLICY_ORACLE_CANONICAL = "oracle_canonical"
"""Policy: the read exposes a dim whose value on the ORACLE-1 destination (fresh instance +
default ``load_state_dict(strict=True, assign=False)`` copy) is a provable device/layout
observation predicate; the producer refuses the save iff the captured pre-clone value departs
that canonical (the escape gate), and the staged runtime tripwire enforces it unconditionally."""

ORACLE_POLICY_DECLARED_REPRODUCED = "declared_reproduced"
"""Policy: the read records a DECLARED-STATE FACT staging REPRODUCES (r65 F-1 ``requires_grad``
/ ``grad_fn`` presence) -- never a canonicality dim, never an escape-gated refusal except a
fact staging provably cannot reproduce."""

ORACLE_POLICY_STRUCTURALLY_COVERED = "structurally_covered"
"""Policy: provably covered by another gate (e.g. ``is_coalesced`` raises on the dense strided
state the layout dim already pins); nothing is recorded."""

ORACLE_POLICY_REFUSE_ON_ANY_READ = "refuse_on_any_read"
"""Policy: NO artifact-independent canonical value exists on the oracle-1 destination, so ANY
attributed read refuses the runnable save (``state_metadata_mismatch``). r67 C4: ``_version``
is the charter member -- oracle-1's default copy increments constructor-owned counters
(``0 -> 1`` for plain slots, ``1 -> 2`` for initialized modules), so no static expected scalar
is honest and no engineered staging form may manufacture one."""

_ORACLE_POLICY_CLASSES = frozenset(
    {
        ORACLE_POLICY_ORACLE_CANONICAL,
        ORACLE_POLICY_DECLARED_REPRODUCED,
        ORACLE_POLICY_STRUCTURALLY_COVERED,
        ORACLE_POLICY_REFUSE_ON_ANY_READ,
    }
)
"""The CLOSED oracle-policy vocabulary: every state-metadata row is exactly one of these."""

STATE_METADATA_ORACLE_POLICY: Mapping[str, str] = MappingProxyType(
    {
        # -- escape-gated read kinds whose canonical is the oracle-1 post-copy observation --
        "contiguous_default": ORACLE_POLICY_ORACLE_CANONICAL,
        "stride_exact": ORACLE_POLICY_ORACLE_CANONICAL,
        "storage_offset": ORACLE_POLICY_ORACLE_CANONICAL,
        "is_conj": ORACLE_POLICY_ORACLE_CANONICAL,
        "is_neg": ORACLE_POLICY_ORACLE_CANONICAL,
        # r67 C3: observed-value placement rows -- still oracle_canonical (their canonical
        # IS the device-defined oracle/staging predicate), but validated against the user's
        # ONE actual accessor return (``_STATE_METADATA_OBSERVED_PLACEMENT_KINDS``), never a
        # pre-forward signature stamp.
        "is_shared": ORACLE_POLICY_ORACLE_CANONICAL,
        "is_pinned": ORACLE_POLICY_ORACLE_CANONICAL,
        "is_inference": ORACLE_POLICY_ORACLE_CANONICAL,
        "is_view": ORACLE_POLICY_ORACLE_CANONICAL,
        "is_leaf": ORACLE_POLICY_ORACLE_CANONICAL,
        "retains_grad": ORACLE_POLICY_ORACLE_CANONICAL,
        "output_nr": ORACLE_POLICY_ORACLE_CANONICAL,
        "grad_presence": ORACLE_POLICY_ORACLE_CANONICAL,
        "storage_nbytes": ORACLE_POLICY_ORACLE_CANONICAL,
        # -- constructor-history-dependent counter: refuse EVERY attributed read (r67 C4) --
        "_version": ORACLE_POLICY_REFUSE_ON_ANY_READ,
        # -- declared-state facts (r65 F-1) --
        "requires_grad": ORACLE_POLICY_DECLARED_REPRODUCED,
        "grad_fn": ORACLE_POLICY_DECLARED_REPRODUCED,
        # -- structural rows --
        "is_coalesced": ORACLE_POLICY_STRUCTURALLY_COVERED,
    }
)
"""THE authoritative oracle-policy classification for state-metadata rows (r67 C4).

Keys are the state read-KIND vocabulary (escape-gated rows), the declared fact names, and the
structural accessor names -- one row per state-metadata surface member. Producer checks
(:func:`state_metadata_read_violations`), signatures, staging tripwires, and diagnostics
consume the SAME rows: an ``oracle_canonical`` row resolves through
``_STATE_METADATA_READ_REQUIRED_DIMS`` to its signature dim; a ``refuse_on_any_read`` row
refuses regardless of any signature; ``declared_reproduced`` rows ride the fact ledger;
``structurally_covered`` rows record nothing. The oracle-1 parity matrix
(tests/test_tlspec_runnable_r65_state_metadata_parity.py) machine-checks that every
``oracle_canonical`` dim equals what oracle-1's default copy into a fresh instance actually
produces, and that no static expected scalar exists without an oracle probe recipe.
"""


def _default_dense_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return torch's canonical row-major contiguous stride for ``shape``.

    Matches ``torch.empty(shape).stride()`` including size-0/size-1 dims (running product of
    ``max(size, 1)`` right-to-left), which is exactly the stride every canonical staging clone
    and transport-decoded state tensor exhibits.
    """

    strides: list[int] = []
    acc = 1
    for size in reversed(shape):
        strides.append(acc)
        acc *= max(int(size), 1)
    return tuple(reversed(strides))


def _state_metadata_signature(value: Any) -> dict[str, Any]:
    """Compute THE state-tensor metadata/layout signature (r63 C1 -- the ONE helper).

    Every state-metadata comparison choke (bind gate, producer preflight, runtime tripwire)
    consumes this signature; future physical dims are added ONLY here, never as scattered
    per-attribute checks. Safety ordering is load-bearing:

    * ``type()`` is read FIRST; an unadmitted class (tensor subclass, non-tensor) short-circuits
      with ZERO tensor-metadata reads (a hostile ``__torch_function__`` subclass observes
      nothing).
    * A nested or non-strided value short-circuits the name/physical reads (several raise
      there).
    * Every metadata read is exception-guarded to ``None`` -- an unreadable dim is UNKNOWN and
      every consumer treats it as non-canonical (fail closed), never as a pass.
    """

    signature: dict[str, Any] = {
        "class_category": "non_tensor",
        "device_type": None,
        "layout": None,
        "is_nested": None,
        "is_quantized": None,
        "has_named_dims": None,
        "is_rowmajor_contiguous": None,
        "stride_is_default": None,
        "storage_offset_is_zero": None,
        "is_conj": None,
        "is_neg": None,
        "is_inference": None,
        "is_view": None,
        "is_leaf": None,
        "retains_grad": None,
        "output_nr_is_zero": None,
        "grad_is_none": None,
        "storage_nbytes_is_tight": None,
        "requires_grad": None,
    }
    cls = type(value)
    if cls is torch.nn.Parameter:
        signature["class_category"] = "parameter"
    elif cls is torch.Tensor:
        signature["class_category"] = "tensor"
    elif isinstance(value, torch.Tensor):
        signature["class_category"] = f"subclass:{cls.__module__}.{cls.__qualname__}"
        return signature
    else:
        return signature
    try:
        # Inert placement context (r67 C3): the device type feeds the observed-value
        # placement predicate; reading ``.device`` has no side effects on any backend.
        signature["device_type"] = str(value.device.type)
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        signature["is_nested"] = bool(value.is_nested)
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        signature["layout"] = str(value.layout)
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        signature["is_quantized"] = bool(value.is_quantized)
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    if signature["is_nested"] is not False or signature["layout"] != "torch.strided":
        # Nested / non-strided / unreadable: name+physical reads may raise and could never
        # be canonical anyway -- leave them UNKNOWN (fail closed at every consumer).
        return signature
    try:
        signature["has_named_dims"] = bool(value.has_names())
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        signature["is_rowmajor_contiguous"] = bool(value.is_contiguous())
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        stride = tuple(int(v) for v in value.stride())
        shape = tuple(int(v) for v in value.shape)
        signature["stride_is_default"] = stride == _default_dense_stride(shape)
    except (RuntimeError, TypeError, ValueError, AttributeError, NotImplementedError):
        pass
    try:
        signature["storage_offset_is_zero"] = int(value.storage_offset()) == 0
    except (RuntimeError, TypeError, ValueError, AttributeError, NotImplementedError):
        pass
    try:
        signature["is_conj"] = bool(value.is_conj())
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        signature["is_neg"] = bool(value.is_neg())
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    # -- r65 Cluster X dims (each exception-guarded to None: unreadable == UNKNOWN, and
    # every consumer treats UNKNOWN as non-canonical -- fail closed, never a pass).
    # ``is_shared``/``is_pinned`` are deliberately NOT stamped here (r67 C3): the honest
    # authority for those reads is the user's ONE actual accessor return (the observation
    # ledger), and a pre-forward speculative stamp is both forbidden (a TorchLens-added
    # ``is_pinned()`` could lazily initialize an accelerator as a capture side effect on
    # pre-guard-era torch) and dishonest (free-F4: the CUDA-init proof-by-absence stamped
    # canonical False on genuinely pinned XPU/MPS/externally-registered memory). --
    try:
        signature["is_inference"] = bool(value.is_inference())
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        signature["is_view"] = bool(value._is_view())
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        signature["is_leaf"] = bool(value.is_leaf)
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        # r71 E1: the capture-time autograd trainable bit, stamped PRE-CLONE from the
        # LIVE tensor -- the source of the TOTALIZED declared ``requires_grad`` fact
        # (state_dict transport detaches, so only this pre-clone read is capture truth).
        signature["requires_grad"] = bool(value.requires_grad)
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        signature["retains_grad"] = bool(value.retains_grad)
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        signature["output_nr_is_zero"] = int(value.output_nr) == 0
    except (RuntimeError, TypeError, ValueError, AttributeError, NotImplementedError):
        pass
    try:
        # r67 C6: ALWAYS the actual ``.grad`` read, under local warning suppression (the
        # non-leaf access warning) -- torch 2.8 allows assigning ``.grad`` on a non-leaf,
        # so the former "a non-leaf without retains_grad structurally carries no grad"
        # shortcut asserted a fact it never read (free-F5). Failure stays UNKNOWN.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signature["grad_is_none"] = value.grad is None
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        pass
    try:
        # Stamped PRE-CLONE (``snapshot_capture_state_signatures`` runs before the
        # normalizing snapshot clone), which is exactly why the r64 F3 large-base view is
        # catchable: offset 0, contiguous, default stride, yet base storage larger than
        # ``numel * element_size``.
        signature["storage_nbytes_is_tight"] = int(value.untyped_storage().nbytes()) == int(
            value.numel()
        ) * int(value.element_size())
    except (RuntimeError, TypeError, ValueError, AttributeError, NotImplementedError):
        pass
    return signature


def _signature_scope_violations(
    signature: Mapping[str, Any],
    scope: tuple[tuple[str, Any], ...],
) -> list[tuple[str, Any, Any]]:
    """Return ``(dim, expected, actual)`` rows where ``signature`` departs canonical ``scope``.

    An unadmitted ``class_category`` is itself the single violation (no other dim of an
    unadmitted class was read, so none can be trusted). ``None`` (unreadable) never equals a
    canonical expectation: fail closed.
    """

    category = signature.get("class_category")
    if category not in _ADMITTED_STATE_CLASS_CATEGORIES:
        return [("class_category", "tensor|parameter", category)]
    return [
        (dim, expected, signature.get(dim))
        for dim, expected in scope
        if signature.get(dim) != expected
    ]


def state_metadata_bind_violations(value: Any) -> list[tuple[str, Any, Any]]:
    """Return load-surviving-subset violations for one supplied/embedded state tensor."""

    return _signature_scope_violations(_state_metadata_signature(value), _STATE_METADATA_BIND_SCOPE)


def _staged_placement_violations(value: Any) -> list[tuple[str, Any, Any]]:
    """Return staged-clone placement violations via REAL run-time reads (r67 C3 tripwire leg).

    The staged runtime tripwire keeps enforcing ``is_shared``/``is_pinned`` on TorchLens's
    OWN staged clones -- these are run-time self-checks on TorchLens-constructed tensors,
    not capture-time speculative reads of the user's slot. Canonicals are the same
    device-defined predicate the producer applies to observations
    (:func:`_state_placement_canonical`); an unreadable value fails closed.
    """

    violations: list[tuple[str, Any, Any]] = []
    try:
        device_type: Any = str(value.device.type)
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        device_type = None
    for kind, reader in (("is_shared", "is_shared"), ("is_pinned", "is_pinned")):
        expected = _state_placement_canonical(kind, device_type)
        try:
            observed: Any = bool(getattr(value, reader)())
        except (RuntimeError, TypeError, AttributeError, NotImplementedError):
            observed = None
        if expected is None or observed != expected:
            violations.append((kind, expected, observed))
    return violations


def state_metadata_full_violations(value: Any) -> list[tuple[str, Any, Any]]:
    """Return full-signature violations for one STAGED runtime state tensor (tripwire scope)."""

    signature = _state_metadata_signature(value)
    violations = _signature_scope_violations(
        signature,
        _STATE_METADATA_BIND_SCOPE + _STATE_METADATA_PHYSICAL_SCOPE,
    )
    if signature.get("class_category") in _ADMITTED_STATE_CLASS_CATEGORIES:
        violations.extend(_staged_placement_violations(value))
    return violations


def state_metadata_read_violations(
    signature: Mapping[str, Any] | None,
    read_kinds: Iterable[str],
    observations: Mapping[str, Any] | None = None,
) -> list[tuple[str, str, Any]]:
    """Return ``(read_kind, dim, actual)`` rows a witnessed metadata read cannot reproduce.

    ``signature`` is the PRE-CLONE capture-time signature stamped for the slot; ``None`` (an
    absent stamp) fails closed for every read kind, as does an unknown kind or an unreadable
    dim -- a metadata read whose captured value cannot be PROVEN canonical can never settle
    ``verified`` after transport normalization. ``observations`` carries the ACTUAL values
    returned by the slot's placement accessor calls (r67 C3): an observed-value kind
    validates the user's one real return against the device-defined staged canonical --
    a missing, unknown (``None``), or non-canonical observation refuses.
    """

    observed_map: Mapping[str, Any] = observations if isinstance(observations, Mapping) else {}
    violations: list[tuple[str, str, Any]] = []
    for kind in sorted(set(read_kinds)):
        # r67 C4: a ``refuse_on_any_read`` policy row refuses REGARDLESS of the captured
        # signature -- no artifact-independent oracle-1 canonical exists for the dim, so no
        # captured value can make the read reproducible (``_version``: oracle-1's default
        # copy perturbs constructor-owned counters, 0 -> 1 plain / 1 -> 2 initialized).
        if STATE_METADATA_ORACLE_POLICY.get(kind) == ORACLE_POLICY_REFUSE_ON_ANY_READ:
            violations.append((kind, "<refuse_on_any_read>", None))
            continue
        if kind in _STATE_METADATA_OBSERVED_PLACEMENT_KINDS:
            device_type = signature.get("device_type") if isinstance(signature, Mapping) else None
            canonical = _state_placement_canonical(kind, device_type)
            observed = observed_map.get(kind)
            if canonical is None or observed is None or bool(observed) != canonical:
                violations.append((kind, "<observed_placement>", observed))
            continue
        required = _STATE_METADATA_READ_REQUIRED_DIMS.get(kind)
        if required is None:
            violations.append((kind, "<unknown_read_kind>", None))
            continue
        dim, expected = required
        actual = signature.get(dim) if isinstance(signature, Mapping) else None
        if actual != expected:
            violations.append((kind, dim, actual))
    if isinstance(signature, Mapping):
        category = signature.get("class_category")
        if category not in _ADMITTED_STATE_CLASS_CATEGORIES:
            violations.append(("<class_admission>", "class_category", category))
    return violations


def snapshot_capture_state_signatures(model: object) -> dict[str, dict[str, Any]] | None:
    """Stamp per-slot state metadata signatures from the LIVE model tensors (r63 C1).

    MUST run BEFORE :func:`snapshot_capture_state`'s clones: the clone itself normalizes
    ``storage_offset`` and materializes conj/neg, so a post-clone signature is blind to two of
    the four lossy physical dims. Walks named parameters AND named buffers
    (``remove_duplicate=False``) so non-persistent buffers -- absent from ``state_dict()`` but
    part of the declared state -- are covered under the same dotted addresses the escape ledger
    records. Metadata reads only; no tensor values are retained.
    """

    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    if not callable(named_parameters) or not callable(named_buffers):
        return None
    try:
        entries: list[tuple[str, Any]] = [
            (str(name), value) for name, value in named_parameters(remove_duplicate=False)
        ]
        entries.extend((str(name), value) for name, value in named_buffers(remove_duplicate=False))
    except Exception:
        return None
    with _state.pause_logging():
        return {
            name: _state_metadata_signature(value)
            for name, value in entries
            if isinstance(value, torch.Tensor)
        }


def snapshot_state_alias_topology(model: object) -> Mapping[str, Any] | None:
    """Capture the live bound-state alias topology BEFORE cloning erases it (r37 corr2-4).

    Walks every named parameter and buffer (``remove_duplicate=False``) of the LIVE
    model and classifies each pair through the shared absolute-byte relation engine:

    * repeated live Python object identity -> a shared ``alias_group`` (one staged
      allocation reproduces ``a is b`` semantics -- tied weights, double-registered
      buffers);
    * DISTINCT objects whose touched bytes ``overlap`` (or are ``unknown``, including
      an unprovable footprint) -> a save-time REFUSAL record: the v2 schema has no
      backing-storage/view recipe, so serializing the pair as independent values
      would silently change in-place propagation semantics on replay (the corr2-4
      false-VERIFIED class);
    * proved-``disjoint`` pairs (including disjoint views of one storage) serialize
      independently.

    An interval sweep per device address space bounds the pairwise work. Runs under
    ``pause_logging`` so geometry reads are never captured. Returns ``None`` when the
    model exposes no named state accessors.
    """

    from .utils.tensor_utils import tensor_byte_footprint, touched_bytes_relation

    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    if not callable(named_parameters) or not callable(named_buffers):
        return None
    try:
        entries: list[tuple[str, torch.Tensor]] = [
            (str(name), value)
            for name, value in named_parameters(remove_duplicate=False)
            if isinstance(value, torch.Tensor)
        ]
        entries.extend(
            (str(name), value)
            for name, value in named_buffers(remove_duplicate=False)
            if isinstance(value, torch.Tensor)
        )
    except Exception:
        return None
    groups: dict[str, str] = {}
    names_by_object: dict[int, list[str]] = {}
    tensor_by_name: dict[str, torch.Tensor] = {}
    for name, value in entries:
        names_by_object.setdefault(id(value), []).append(name)
        tensor_by_name.setdefault(name, value)
    for names in names_by_object.values():
        if len(names) > 1:
            group_id = f"state_alias:{min(names)}"
            for name in names:
                groups[name] = group_id
    refusals: list[tuple[str, str, str]] = []
    with _state.pause_logging():
        footprints: list[tuple[str, Any]] = []
        for name, value in tensor_by_name.items():
            footprint = tensor_byte_footprint(value)
            if footprint is None:
                # An unprovable footprint cannot participate in any disjointness
                # proof: fail closed against every other state name (INV-2).
                refusals.append((name, "<unprovable footprint>", "unknown"))
                continue
            if footprint.numel == 0:
                continue
            footprints.append((name, footprint))
        # Interval sweep per device address space: only interval-intersecting pairs
        # need the exact relation (identity pairs were already grouped above).
        footprints.sort(key=lambda item: (item[1].device_key, item[1].start_byte))
        active: list[tuple[str, Any]] = []
        for name, footprint in footprints:
            still_active: list[tuple[str, Any]] = []
            for other_name, other in active:
                if (
                    other.device_key == footprint.device_key
                    and other.end_byte > footprint.start_byte
                ):
                    still_active.append((other_name, other))
            active = still_active
            for other_name, _other in active:
                left_tensor = tensor_by_name[name]
                right_tensor = tensor_by_name[other_name]
                if left_tensor is right_tensor:
                    continue  # identity: covered by an alias group
                relation = touched_bytes_relation(left_tensor, right_tensor)
                if relation != "disjoint":
                    refusals.append((other_name, name, relation))
            active.append((name, footprint))
    # Plain containers (not MappingProxyType): the snapshot rides on the Trace
    # ``__dict__`` and must survive pickle/deepcopy like its sibling capture-state.
    return {"groups": groups, "refusals": tuple(refusals)}


def load_trace_state_dict(trace: Any, sd: Mapping[str, Any]) -> None:
    """Validate and atomically stage a user state mapping on a sparse Trace.

    Parameters
    ----------
    trace:
        Trace receiving transient run state.
    sd:
        Canonically named parameter and persistent-buffer tensors.

    Raises
    ------
    StateBindingError
        If the Trace is not sparse-runnable or any strict slot contract fails.
    RunCapabilityUnavailableError
        If a staging clone's logical byte size exceeds the live device budget
        (``clone_allocation_preflight``, r61 corr_2) -- refused BEFORE the clone
        allocates, like the staging device failures this stage already types.
    """

    staged = _validate_state_mapping(trace, sd)
    readiness = trace.__dict__.get("_runnable_readiness")
    updated_readiness = readiness
    if readiness is not None and hasattr(readiness, "state_sources_available"):
        sources = tuple(
            source
            for source in readiness.state_sources_available
            if source is not StateSource.USER_STATE_DICT
        )
        updated_readiness = replace(
            readiness,
            state_sources_available=(StateSource.USER_STATE_DICT, *sources),
        )
    trace.__dict__["_runnable_staged_user_state"] = staged
    if updated_readiness is not readiness:
        trace.__dict__["_runnable_readiness"] = updated_readiness


def bind_embedded_trace_state(trace: Any, sd: Mapping[str, Any]) -> None:
    """Validate and atomically bind an embedded capture-time state mapping.

    Parameters
    ----------
    trace:
        Loaded sparse Trace receiving the optional weight payload.
    sd:
        Canonically named parameter and persistent-buffer tensors decoded from
        the runnable artifact.

    Raises
    ------
    StateBindingError
        If any embedded entry violates the ordinary strict state contract.
    RunCapabilityUnavailableError
        If a bind clone's byte size exceeds the live device budget
        (``clone_allocation_preflight``, r61 corr_2 defense-in-depth: decoded
        blobs are dense, so this bounds the transient doubling).
    """

    _validate_state_mapping(trace, sd)
    trace.__dict__["_runnable_embedded_state"] = MappingProxyType(
        {
            name: _staged_state_clone(value, state_dict_name=name)
            for name, value in sd.items()
            if isinstance(name, str) and isinstance(value, torch.Tensor)
        }
    )


def bind_embedded_nonpersistent_buffers(trace: Any, buffers: Mapping[str, Any]) -> None:
    """Validate and bind captured values for used non-persistent buffers.

    Parameters
    ----------
    trace:
        Loaded sparse Trace receiving the mandatory buffer payload.
    buffers:
        Registered buffer names mapped to capture-time tensor values.

    Raises
    ------
    StateBindingError
        If names, shapes, dtypes, or roles violate the non-persistent slots.
    """

    descriptor = _require_descriptor(trace)
    validate_nonpersistent_buffer_mapping_for_descriptor(descriptor, buffers)
    trace.__dict__["_runnable_embedded_nonpersistent_buffers"] = {
        name: _staged_state_clone(value, state_dict_name=name)
        for name, value in buffers.items()
        if isinstance(name, str) and isinstance(value, torch.Tensor)
    }


_STATE_METADATA_FACT_SITE_PREFIX = "state_metadata:"
"""``site_label`` prefix of a persisted DECLARED-STATE metadata fact witness (r65 F-1)."""

_STATE_METADATA_FACT_KEY = "state_metadata"
"""Discriminator key present in every declared state-metadata fact."""


def recorded_state_metadata_facts(descriptor: SparseRunDescriptor) -> dict[str, dict[str, bool]]:
    """Decode the per-state-name DECLARED metadata facts persisted in a descriptor (r65 F-1).

    Returns ``{state_dict_name: {fact_name: bool}}`` for every well-formed
    ``state_metadata:<name>`` SHAPE_STRUCTURE_FACT witness. Malformed entries are refused at
    PARSE time (closed fact-name vocabulary, ``context_field_invalid``), so a descriptor that
    reaches run preparation carries only validated facts; anything unexpected here decodes to
    nothing (fail closed into "no fact", which stages the detached default).
    """

    from ._runnable_execution import _decode_literal

    facts: dict[str, dict[str, bool]] = {}
    for witness in descriptor.control_witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        if not witness.site_label.startswith(_STATE_METADATA_FACT_SITE_PREFIX):
            continue
        try:
            decoded = _decode_literal(witness.observed_value)
        except Exception:
            continue
        if not isinstance(decoded, Mapping) or decoded.get(_STATE_METADATA_FACT_KEY) is not True:
            continue
        name = decoded.get("state")
        fact_map = decoded.get("facts")
        if not isinstance(name, str) or not isinstance(fact_map, Mapping):
            continue
        facts[name] = {
            str(fact): bool(fact_value)
            for fact, fact_value in fact_map.items()
            if isinstance(fact_value, bool)
        }
    return facts


def _apply_state_metadata_facts(
    descriptor: SparseRunDescriptor, prepared: PreparedRunnableState
) -> PreparedRunnableState:
    """Reproduce the TOTALIZED per-slot ``requires_grad`` bit on staged state (r71 E1).

    r71 totalizes the r65 F-1 declared fact over the whole declared state-name
    universe: EVERY staged slot receives its capture-time autograd trainable bit from
    the OWNER-RECORD ``StateSlotBinding.captured_requires_grad`` (``state_dict()``
    detaches, so the bit is transport-lost). The recorded bit ALWAYS wins:
    user-supplied ``load_state_dict`` tensors carrying a different ``requires_grad``
    still stage the recorded value -- capture truth, strictly more oracle-1-faithful
    than the detached default and aligned with the LOCKED r65 F-1 declared-fact
    ruling. ``grad_fn`` presence needs no application: True refuses at save AND parse
    (no staged leaf can carry a grad_fn) and False is what every staged clone already
    exhibits.

    The r69 staging belt is re-pointed at the PARSE-DERIVED required set (r71): the
    witness-derived facts must equal the binding-derived facts EXACTLY -- impossible
    to violate post-parse; a descriptor that somehow reached staging with a deficit
    fails closed typed instead of silently omitting a declared bit (the
    free-F1-secondary strip lane).
    """

    facts = recorded_state_metadata_facts(descriptor)
    # r71: the PARSE-DERIVED required set comes from the OWNING state bindings
    # (never the inventory mirror or the witness stream): one (name, fact) pair per
    # declared state name x the closed two-fact vocabulary.
    binding_facts: dict[str, dict[str, bool]] = {}
    for slot in descriptor.tensor_slots:
        binding = slot.state_binding
        if binding is not None:
            binding_facts.setdefault(
                binding.state_dict_name,
                {
                    "grad_fn": bool(binding.captured_grad_fn),
                    "requires_grad": bool(binding.captured_requires_grad),
                },
            )
    if facts != binding_facts:
        applied_identities = sorted(
            f"{name}::{fact_name}" for name, fact_map in facts.items() for fact_name in fact_map
        )
        required_identities = sorted(
            f"{name}::{fact_name}"
            for name, fact_map in binding_facts.items()
            for fact_name in fact_map
        )
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.CONTEXT_FIELD_INVALID,
                    "Declared state-metadata facts do not equal the parse-derived "
                    f"required set (witnessed {applied_identities[:8]!r}, required "
                    f"{required_identities[:8]!r}); staging refuses rather than "
                    "silently omitting a declared bit.",
                    detection_stage="state_metadata_fact_staging",
                    details=(("reason", "state_metadata_inventory_mismatch"),),
                ),
            )
        )
    if not binding_facts:
        return prepared
    name_by_slot: dict[str, str] = {}
    for slot in descriptor.tensor_slots:
        if slot.state_binding is not None:
            name_by_slot[slot.slot_id] = slot.state_binding.state_dict_name
    for slot_id, value in prepared.slot_values.items():
        name = name_by_slot.get(slot_id)
        recorded = binding_facts.get(name, {}).get("requires_grad") if name is not None else None
        if recorded is None or bool(value.requires_grad) == recorded:
            continue
        try:
            value.requires_grad_(recorded)
        except RuntimeError as exc:
            # Unreachable for producer-validated artifacts (a ``requires_grad=True`` fact on
            # a non-differentiable slot refuses at save); a tampered artifact fails typed
            # here rather than running with an unreproduced declared fact.
            raise _binding_error(
                (
                    _diagnostic(
                        RunnableErrorCode.STATE_METADATA_MISMATCH,
                        f"Recorded state-metadata fact requires_grad={recorded!r} for state "
                        f"{name!r} cannot be applied to the staged slot {slot_id!r}: {exc}. "
                        "The declared capture-time trainable bit cannot be reproduced, so "
                        "the run is refused (state_metadata_mismatch).",
                        detection_stage="state_metadata_fact_staging",
                        details=(
                            ("reason", "state_metadata_mismatch"),
                            ("state_dict_name", str(name)),
                            ("slot_id", slot_id),
                        ),
                    ),
                )
            ) from exc
    return prepared


def prepare_runnable_state(trace: Any, seed: int | None = None) -> PreparedRunnableState:
    """Resolve and allocate all parameter/buffer slots without executing the DAG.

    Parameters
    ----------
    trace:
        Loaded sparse Trace whose descriptor supplies state-slot contracts.
    seed:
        Optional isolated initializer seed. ``None`` uses normal runtime RNG.

    Returns
    -------
    PreparedRunnableState
        Run-local slot values and honest source/initializer reporting.

    Raises
    ------
    StateBindingError
        If the descriptor or selected state source violates a slot contract.
    """

    descriptor = _require_descriptor(trace)
    # r55 free_1: bound every recorded op-output allocation BEFORE the DAG runs, on
    # every state source (staged/embedded/random-init), so a tampered self-consistent
    # descriptor cannot drive an out-of-budget op allocation and OOM-kill the host.
    _preflight_run_allocation((), _recorded_output_slots(descriptor))
    # r59: aggregate retention floor -- the SUM of guaranteed-retained op-output clone
    # bytes per device provably exceeding the budget is a guaranteed mid-replay OOM the
    # per-slot bound above cannot see (honest, individually-feasible slots that together
    # exhaust the host). Refused typed here, before the DAG runs. Zero false refusals:
    # the floor under-counts true retention and the budget never under-estimates.
    _preflight_retention_floor(descriptor)
    nonpersistent_buffers = _prepared_nonpersistent_buffers(trace, descriptor)

    def _staged(prepared: PreparedRunnableState) -> PreparedRunnableState:
        staged = replace(
            prepared,
            slot_values=stage_state_to_slot_devices(descriptor, prepared.slot_values),
        )
        # r65 F-1: reproduce the recorded per-slot ``requires_grad`` bit AFTER device
        # staging, on every state source (user-staged, embedded, random-init) -- the
        # recorded declared fact always wins over the source tensor's transport-lost bit.
        return _apply_state_metadata_facts(descriptor, staged)

    user_state = trace.__dict__.get("_runnable_staged_user_state")
    if isinstance(user_state, Mapping):
        return _staged(
            _with_nonpersistent_buffers(
                _prepared_bound_state(user_state, StateSource.USER_STATE_DICT, seed),
                nonpersistent_buffers,
            )
        )

    embedded_state = trace.__dict__.get("_runnable_embedded_state")
    if embedded_state is not None:
        if not isinstance(embedded_state, Mapping):
            raise _binding_error(
                (
                    _diagnostic(
                        RunnableErrorCode.STATE_ROLE_MISMATCH,
                        "Embedded state hook does not contain a state mapping.",
                        detection_stage="state_embedded_hook",
                    ),
                )
            )
        validated = _validate_state_mapping(trace, embedded_state)
        return _staged(
            _with_nonpersistent_buffers(
                _prepared_bound_state(validated, StateSource.EMBEDDED_CAPTURE_STATE, seed),
                nonpersistent_buffers,
            )
        )
    if descriptor.payload_layers.weights.present:
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.STATE_MISSING_KEY,
                    "Descriptor declares embedded capture state, but its Stage 7 hook is empty.",
                    detection_stage="state_embedded_hook",
                ),
            )
        )

    slot_values, random_slot_ids = _initialize_state_slots(descriptor, seed)
    return _staged(
        _with_nonpersistent_buffers(
            PreparedRunnableState(
                slot_values=MappingProxyType(slot_values),
                state_source=StateSource.RANDOM_INITIALIZATION,
                initializer_policy_version=RUNNABLE_INITIALIZER_POLICY_VERSION,
                seed=seed,
                random_filled_slot_ids=random_slot_ids,
            ),
            nonpersistent_buffers,
        )
    )


def _prepared_bound_state(
    state: Mapping[str, torch.Tensor],
    source: StateSource,
    seed: int | None,
) -> PreparedRunnableState:
    """Build a preparation record for a previously validated state source."""

    return PreparedRunnableState(
        slot_values=MappingProxyType(dict(state)),
        state_source=source,
        initializer_policy_version=None,
        seed=seed,
        random_filled_slot_ids=(),
    )


def _with_nonpersistent_buffers(
    prepared: PreparedRunnableState,
    buffers: Mapping[str, torch.Tensor],
) -> PreparedRunnableState:
    """Merge capture-embedded non-persistent buffers into prepared run state."""

    values = dict(prepared.slot_values)
    values.update(buffers)
    return replace(prepared, slot_values=MappingProxyType(values))


def _prepared_nonpersistent_buffers(
    trace: Any,
    descriptor: SparseRunDescriptor,
) -> Mapping[str, torch.Tensor]:
    """Return validated slot-keyed capture values for non-persistent buffers."""

    slots = _nonpersistent_buffer_slots(descriptor)
    declared = descriptor.payload_layers.nonpersistent_buffers
    embedded = trace.__dict__.get("_runnable_embedded_nonpersistent_buffers")
    if not slots:
        return MappingProxyType({})
    if (
        not declared.present
        or declared.schema != "runnable_nonpersistent_buffer_v1"
        or not isinstance(embedded, Mapping)
    ):
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.STATE_MISSING_KEY,
                    "Used non-persistent buffers require their capture-embedded payload.",
                    detection_stage="state_nonpersistent_buffer_hook",
                ),
            )
        )
    return validate_nonpersistent_buffer_mapping_for_descriptor(descriptor, embedded)


def _validate_state_mapping(trace: Any, sd: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Validate one strict mapping and return detached slot-keyed values."""

    descriptor = _require_descriptor(trace)
    return validate_state_mapping_for_descriptor(descriptor, sd)


def validate_state_mapping_for_descriptor(
    descriptor: SparseRunDescriptor,
    sd: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Validate a strict state mapping against an explicit sparse descriptor.

    Parameters
    ----------
    descriptor:
        Runnable descriptor supplying canonical state slot contracts.
    sd:
        Mapping of canonical state names to tensor values.

    Returns
    -------
    Mapping[str, torch.Tensor]
        Detached, slot-keyed values suitable for one runnable execution.

    Raises
    ------
    StateBindingError
        If names, roles, aliases, shapes, or dtypes violate the contract.
    """

    return _validate_named_slot_mapping(_persistent_state_slots(descriptor), sd)


def validate_nonpersistent_buffer_mapping_for_descriptor(
    descriptor: SparseRunDescriptor,
    buffers: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Validate captured non-persistent buffers against their runnable slots.

    Parameters
    ----------
    descriptor:
        Runnable descriptor supplying non-persistent buffer contracts.
    buffers:
        Mapping of registered buffer names to captured tensor values.

    Returns
    -------
    Mapping[str, torch.Tensor]
        Detached, slot-keyed buffer values suitable for execution.

    Raises
    ------
    StateBindingError
        If names, roles, aliases, shapes, or dtypes violate the contract.
    """

    return _validate_named_slot_mapping(_nonpersistent_buffer_slots(descriptor), buffers)


def _validate_named_slot_mapping(
    state_slots: tuple[TensorSlotDescriptor, ...],
    values: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Validate one exact canonical-name mapping for a selected slot set."""

    if not isinstance(values, Mapping):
        raise TypeError("state values must be a mapping of canonical names to tensors.")

    slots_by_name: dict[str, list[TensorSlotDescriptor]] = defaultdict(list)
    for slot in state_slots:
        assert slot.state_binding is not None
        slots_by_name[slot.state_binding.state_dict_name].append(slot)

    diagnostics: list[RunnableDiagnostic] = []
    supplied_names = {name for name in values if isinstance(name, str)}
    expected_names = set(slots_by_name)
    for name in sorted(expected_names - supplied_names):
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.STATE_MISSING_KEY,
                f"State mapping is missing canonical key {name!r}.",
                detection_stage="state_name_binding",
                details=(("state_dict_name", name),),
            )
        )
    for key in values:
        if not isinstance(key, str) or key not in expected_names:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.STATE_UNEXPECTED_KEY,
                    f"State mapping contains unexpected key {key!r}.",
                    detection_stage="state_name_binding",
                    details=(("state_dict_name", repr(key)),),
                )
            )

    values_by_name: dict[str, torch.Tensor] = {}
    for name in sorted(expected_names & supplied_names):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.STATE_DTYPE_MISMATCH,
                    f"State value for {name!r} is not a tensor.",
                    detection_stage="state_tensor_contract",
                    details=(("state_dict_name", name), ("actual_type", type(value).__name__)),
                )
            )
            continue
        # r63 C1 bind gate: exact-class admission + the LOAD-SURVIVING metadata subset,
        # checked BEFORE any shape/dtype/alias read. A refused value never enters
        # ``values_by_name``, so no later validation stage reads metadata on it (an
        # unadmitted subclass observes zero reads; a nested value whose ``.shape``
        # raises is refused typed here instead of crashing the shape check).
        bind_violations = state_metadata_bind_violations(value)
        if bind_violations:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.STATE_METADATA_MISMATCH,
                    f"State tensor {name!r} carries metadata that cannot bind to a "
                    "canonical dense slot under default load_state_dict copy semantics "
                    f"(violations: {bind_violations!r}).",
                    detection_stage="state_tensor_contract",
                    details=(
                        ("state_dict_name", name),
                        ("violations", repr(bind_violations)),
                    ),
                )
            )
            continue
        values_by_name[name] = value
        for slot in slots_by_name[name]:
            diagnostics.extend(_slot_contract_diagnostics(slot, value))

    diagnostics.extend(_alias_value_diagnostics(state_slots, values_by_name))
    if diagnostics:
        raise _binding_error(tuple(diagnostics))

    staged: dict[str, torch.Tensor] = {}
    shared_by_alias: dict[str, torch.Tensor] = {}
    shared_by_name: dict[str, torch.Tensor] = {}
    for slot in sorted(state_slots, key=lambda item: item.slot_id):
        binding = slot.state_binding
        assert binding is not None
        group_key = binding.alias_group
        if group_key is not None and group_key in shared_by_alias:
            staged[slot.slot_id] = shared_by_alias[group_key]
            continue
        if binding.state_dict_name in shared_by_name:
            value = shared_by_name[binding.state_dict_name]
        else:
            # r61 corr_2: the staging clone materializes the LOGICAL extent of a
            # supplied value (the strict binder accepts an expanded view whose
            # storage is tiny), and user/embedded state never enters the run-prep
            # representative sum -- so this is the load-bearing byte bound on the
            # state re-materialization chain. r63 C1: the clone stages through the
            # canonical destination (default load_state_dict copy semantics).
            value = _staged_state_clone(
                values_by_name[binding.state_dict_name],
                slot_id=slot.slot_id,
                state_dict_name=binding.state_dict_name,
            )
            shared_by_name[binding.state_dict_name] = value
        if group_key is not None:
            shared_by_alias[group_key] = value
        staged[slot.slot_id] = value
    return MappingProxyType(staged)


def _slot_contract_diagnostics(
    slot: TensorSlotDescriptor,
    value: torch.Tensor,
) -> list[RunnableDiagnostic]:
    """Return strict name-derived and tensor-contract diagnostics for one slot."""

    binding = slot.state_binding
    assert binding is not None
    diagnostics: list[RunnableDiagnostic] = []
    name = binding.state_dict_name
    inferred_module, inferred_role = _name_contract(name, slot.role)
    if binding.module_path != inferred_module:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.STATE_MODULE_PATH_MISMATCH,
                f"Recorded module path for {name!r} disagrees with its canonical name.",
                detection_stage="state_module_path_validation",
                details=(
                    ("state_dict_name", name),
                    ("recorded_module_path", binding.module_path),
                    ("canonical_module_path", inferred_module),
                ),
            )
        )
    if binding.semantic_role not in inferred_role:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.STATE_ROLE_MISMATCH,
                f"Recorded semantic role for {name!r} disagrees with its canonical name.",
                detection_stage="state_role_validation",
                details=(
                    ("state_dict_name", name),
                    ("recorded_role", binding.semantic_role.value),
                    ("allowed_roles", ",".join(sorted(role.value for role in inferred_role))),
                ),
            )
        )
    if tuple(value.shape) != slot.shape:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.STATE_SHAPE_MISMATCH,
                f"State tensor {name!r} has shape {tuple(value.shape)}, expected {slot.shape}.",
                detection_stage="state_tensor_contract",
                details=(
                    ("state_dict_name", name),
                    ("expected_shape", repr(slot.shape)),
                    ("actual_shape", repr(tuple(value.shape))),
                ),
            )
        )
    if str(value.dtype) != slot.dtype:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.STATE_DTYPE_MISMATCH,
                f"State tensor {name!r} has dtype {value.dtype}, expected {slot.dtype}.",
                detection_stage="state_tensor_contract",
                details=(
                    ("state_dict_name", name),
                    ("expected_dtype", slot.dtype),
                    ("actual_dtype", str(value.dtype)),
                ),
            )
        )
    return diagnostics


def _name_contract(
    state_dict_name: str,
    slot_role: TensorSlotRole,
) -> tuple[str, frozenset[StateSlotRole]]:
    """Infer the module-path and allowed semantic roles from a canonical state name."""

    module_path, separator, leaf_name = state_dict_name.rpartition(".")
    canonical_module = module_path if separator else "self"
    if leaf_name == "weight":
        roles = frozenset({StateSlotRole.WEIGHT, StateSlotRole.NORM_SCALE})
    elif leaf_name == "bias":
        roles = frozenset({StateSlotRole.BIAS, StateSlotRole.NORM_OFFSET})
    elif leaf_name == "running_mean":
        roles = frozenset({StateSlotRole.RUNNING_MEAN})
    elif leaf_name == "running_var":
        roles = frozenset({StateSlotRole.RUNNING_VAR})
    elif leaf_name in {"num_batches_tracked", "counter"}:
        roles = frozenset({StateSlotRole.COUNTER})
    elif slot_role is TensorSlotRole.BUFFER:
        roles = frozenset({StateSlotRole.GENERIC_BUFFER})
    else:
        roles = frozenset({StateSlotRole.WEIGHT})
    return canonical_module, roles


def _alias_values_coherent(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Return whether two same-shape/dtype alias-group values are byte-coherent.

    r35 corr2_2: coherence is the STORAGE-IDENTITY fast path (the tied-parameter
    case -- one allocation exposed under two canonical names) or frozen
    LOGICAL-BYTE equality via the same representation as
    ``runnable_tensor_byte_digest``. Never float ``torch.equal`` (NaN payloads
    falsely conflict), never ``allclose``, never ``equal_nan=True`` (a
    different-NaN-payload pair must still conflict): NaN payloads, complex
    components, signed zero, infinities, and int/bool bytes are all exact.
    """

    try:
        if (
            first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
            and first.storage_offset() == second.storage_offset()
            and tuple(first.stride()) == tuple(second.stride())
            and tuple(first.shape) == tuple(second.shape)
        ):
            return True
    except (RuntimeError, AttributeError, TypeError, NotImplementedError):
        pass
    try:
        return runnable_tensor_byte_digest(first) == runnable_tensor_byte_digest(second)
    except Exception:
        # An undigestable exotic value cannot be proven coherent: conflict
        # (fail closed, atomic typed rejection at the caller).
        return False


def _alias_value_diagnostics(
    slots: tuple[TensorSlotDescriptor, ...],
    values_by_name: Mapping[str, torch.Tensor],
) -> list[RunnableDiagnostic]:
    """Return diagnostics when named entries in an alias group are not coherent."""

    slots_by_alias: dict[str, list[TensorSlotDescriptor]] = defaultdict(list)
    for slot in slots:
        binding = slot.state_binding
        assert binding is not None
        if binding.alias_group is not None:
            slots_by_alias[binding.alias_group].append(slot)
    diagnostics: list[RunnableDiagnostic] = []
    for alias_group, members in sorted(slots_by_alias.items()):
        named_values = [
            (slot.state_binding.state_dict_name, values_by_name[slot.state_binding.state_dict_name])
            for slot in members
            if slot.state_binding is not None
            and slot.state_binding.state_dict_name in values_by_name
        ]
        if len(named_values) < 2:
            continue
        first_name, first_value = named_values[0]
        for name, value in named_values[1:]:
            if (
                first_value.shape != value.shape
                or first_value.dtype != value.dtype
                or not _alias_values_coherent(first_value, value)
            ):
                diagnostics.append(
                    _diagnostic(
                        RunnableErrorCode.STATE_ALIAS_CONFLICT,
                        f"Alias group {alias_group!r} has conflicting named state values.",
                        detection_stage="state_alias_validation",
                        details=(
                            ("alias_group", alias_group),
                            ("first_state_dict_name", first_name),
                            ("conflicting_state_dict_name", name),
                        ),
                    )
                )
                break
    return diagnostics


def _initialize_state_slots(
    descriptor: SparseRunDescriptor,
    seed: int | None,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Allocate every state slot using the frozen role initializer table."""

    state_slots = _persistent_state_slots(descriptor)
    groups: dict[str, list[TensorSlotDescriptor]] = defaultdict(list)
    for slot in state_slots:
        binding = slot.state_binding
        assert binding is not None
        group = binding.alias_group or f"name:{binding.state_dict_name}"
        groups[group].append(slot)

    ordered_groups = [
        sorted(members, key=lambda item: item.slot_id)
        for members in sorted(
            groups.values(), key=lambda items: min(item.slot_id for item in items)
        )
    ]
    # r53 free_1: refuse an infeasible total BEFORE the first allocation, once
    # per alias group (aliases share one allocation).
    _preflight_random_init_allocation([ordered[0] for ordered in ordered_groups])

    values: dict[str, torch.Tensor] = {}
    generator_by_device: dict[str, torch.Generator] = {}
    for ordered in ordered_groups:
        _validate_alias_allocation_contract(ordered)
        representative = ordered[0]
        generator = _generator_for_slot(representative, seed, generator_by_device)
        value = _initialize_slot(representative, generator)
        for member in ordered:
            values[member.slot_id] = value
    random_slot_ids = tuple(sorted(slot.slot_id for slot in state_slots))
    return values, random_slot_ids


_STATIC_ALLOCATION_BUDGET_BYTES = 1 << 40
"""Last-resort 1 TiB defense ceiling when no live budget probe is available.

Kills every 10^12+-byte allocation bomb while touching no real model: the
largest legitimate single random-init state slot on record (a 70B-class
embedding) is three orders of magnitude below it.
"""


def _host_memory_budget_bytes() -> int | None:
    """Return available host memory plus free swap, or ``None`` when unprobeable."""

    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available) + int(psutil.swap_memory().free)
        except Exception:  # pragma: no cover - defensive probe fallback
            pass
    try:
        fields: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    fields[name.strip()] = int(parts[0]) * 1024
        if "MemAvailable" in fields:
            return fields["MemAvailable"] + fields.get("SwapFree", 0)
    except OSError:  # pragma: no cover - non-Linux hosts without /proc
        pass
    return None


def _allocation_budget_bytes(device: torch.device) -> int:
    """Return a NEVER-under-estimating allocation budget for one device.

    The budget deliberately over-estimates when uncertain (free device memory
    PLUS the caching allocator's reusable reserve for CUDA; available + swap
    for host-backed devices; the static 1 TiB ceiling otherwise), so the
    preflight can only refuse requests that could never have succeeded --
    a refusal here is strictly better than the OOM-kill it replaces, and a
    legitimate large-model slot is never refused (no static per-slot cap).
    """

    if device.type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info(device)
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            return int(free) + max(0, int(reserved) - int(allocated))
        except Exception:  # pragma: no cover - unprobeable CUDA runtime
            return _STATIC_ALLOCATION_BUDGET_BYTES
    if device.type in {"cpu", "mps"}:
        budget = _host_memory_budget_bytes()
        if budget is not None:
            return budget
    return _STATIC_ALLOCATION_BUDGET_BYTES


_OUTPUT_COUNT_MARGIN = 8
"""Realized-output headroom multiplier over the recorded count (r59).

An honest replay produces EXACTLY the recorded number of output tensors per call, so
realized == recorded and the margin is pure headroom. It is load-bearing only at HIGH
recorded arity: a legit ``unbind`` recording 2048 outputs needs a 16384 ceiling. A
smaller multiple would false-refuse a legitimate high-arity op; never drop it.
"""

_OUTPUT_COUNT_FLOOR = 4096
"""Absolute realized-output floor (r59, LOAD-BEARING -- do not simplify to ``rec * MARGIN``).

A legit LOW-arity op DECOMPOSES into many fake intermediates during projection -- measured
up to 55 fakes per 1 recorded output (``interpolate`` = 55, ``batch_norm`` = 28,
``einsum`` = 12, ``layer_norm`` = 11, ``addmm`` = 9). The floor carries those honest
decompositions so the count gate never over-refuses them. It must NEVER be reduced to
``recorded * MARGIN``: that would refuse ``interpolate``/``batch_norm``/... on a
single-output call.
"""


@contextmanager
def _guarded_defensive_materialize() -> Iterator[None]:
    """NEUTRAL ambient for every PRE-EXECUTION defensive materialization (r67 C5).

    Oracle 1 constructs/loads state and receives inputs BEFORE its forward context, so
    TorchLens's defensive materializations -- the staging clone (incl. re-layout and
    cross-device ``.to()`` transfer), the second state clone, the runtime input mirror,
    and random-state allocation/fill -- run under neutral ``torch.inference_mode(False)``
    plus ``torch.enable_grad()``:

    * never the CALLER's ambient -- a user calling ``.run()`` / binding state inside
      ``torch.inference_mode()`` or ``torch.no_grad()`` must not mint inference-mode
      clones that trip the staged-state tripwire (corr1-2's false
      ``PathDivergenceError(state_metadata_mismatch is_inference)``) or strip the mirror
      of attestation eligibility on an otherwise-exact run;
    * never the RECORDED ambient -- recorded ambient/per-call contexts govern sparse
      EXECUTION only, entered around the transaction after binding/staging.

    Exact caller restoration is the context managers' contract (both restore prior state
    on exit, success or raise). NARROW by design: mid-transaction op/witness/attestation
    snapshots (:meth:`RunResourceCeiling.guarded_clone` from execution sites) stay OUTSIDE
    this helper -- they legitimately run under recorded execution semantics, so neither
    ``guarded_clone`` nor :func:`_byte_guarded_clone` is globally neutralized. The
    bidirectional source-scan immunizer owns this boundary.
    """

    with torch.inference_mode(False), torch.enable_grad():
        yield


def _byte_guarded_clone(
    value: torch.Tensor,
    *,
    call_id: str | None = None,
    slot_id: str | None = None,
    affected_op_labels: Sequence[str] = (),
    state_dict_name: str | None = None,
) -> torch.Tensor:
    """Byte-guard ONE TorchLens-owned re-materialization clone before it allocates.

    THE single byte-guard core (r61 corr_2): ``numel * itemsize`` (pure integer math,
    no allocation) is compared to the SAME never-under-estimating per-device budget
    run-prep uses; a clone that could never fit is refused typed at
    ``clone_allocation_preflight`` BEFORE the materializing ``.detach().clone()``. The
    clone's ``numel()`` is the LOGICAL numel of an expanded view -- exactly what
    ``.clone()`` materializes -- so a small-storage/huge-logical view is bounded here.
    Every TorchLens-owned replay/staging re-materialization routes through this core:
    op-output/attestation/witness/reconstruct snapshots (via
    :meth:`RunResourceCeiling.guarded_clone`), the accepted runtime input mirror, the
    binder staging clone (user ``load_state_dict``), the embedded-state and
    non-persistent-buffer bind clones, and the run-time state clone. An honest clone
    equals an honest recorded/declared slot that already passed the run-prep bound, so
    a faithful run is never refused here. The capture-side save-time snapshot is
    deliberately NOT routed: it clones the live model's own values, so no
    artifact-driven amplification exists there.
    """

    requested = int(value.numel()) * int(value.element_size())
    device = value.device
    available = _allocation_budget_bytes(device)
    if requested > available:
        subject = (
            f"state entry {state_dict_name!r}"
            if state_dict_name is not None
            else f"sparse call {call_id!r} output slot {slot_id!r}"
        )
        extra: dict[str, Any] = {}
        if state_dict_name is not None:
            extra["state_dict_name"] = state_dict_name
        raise RunCapabilityUnavailableError(
            f"Re-materializing {subject} requires a {requested}-byte clone on device "
            f"{str(device)!r}, but only {available} bytes are available on this host. "
            "A recorded view or supplied value whose logical size cannot fit (honest "
            "small storage, inflated logical extent) cannot be materialized into an "
            "out-of-budget clone; the descriptor or supplied state may be tampered.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="clone_allocation_preflight",
            call_id=call_id,
            slot_id=slot_id,
            device=str(device),
            required_bytes=requested,
            available_bytes=available,
            affected_op_labels=tuple(affected_op_labels),
            **extra,
        )
    return value.detach().clone()


def _staged_state_clone(
    value: torch.Tensor,
    *,
    slot_id: str | None = None,
    state_dict_name: str | None = None,
) -> torch.Tensor:
    """Stage ONE state tensor through the canonical destination (r63 C1 Part 3).

    THE single state staging helper: every user ``load_state_dict``, embedded
    ``state_dict_v1``, and non-persistent-buffer bind clone routes here (source tripwire:
    no state-binding path may call bare :func:`_byte_guarded_clone` / ``.detach().clone()``
    directly). It reproduces oracle-1's DEFAULT ``load_state_dict(strict=True,
    assign=False)`` copy semantics: the staged allocation is always the canonical physical
    form (row-major-contiguous default stride, zero storage offset, materialized conj/neg)
    regardless of the source's physical form, exactly like a copy into a canonical fresh
    destination. ``.detach().clone()`` already compacts the offset and materializes lazy
    bits; stride is the one physical dim ``preserve_format`` keeps, so it is re-laid only
    when non-canonical (an unreadable stride fails closed into the re-lay).
    """

    # r67 C5: materialize under the ONE neutral defensive-materialization ambient
    # (``inference_mode(False)`` + ``enable_grad``) so a user calling ``tl.load(...).run()``
    # (or binding state) INSIDE an inference_mode/no_grad region cannot mint inference-mode
    # staged clones and trip the ``is_inference`` runtime tripwire dim on TorchLens's own
    # staging output (supersedes the r65 staging hardening (a) inference-only guard).
    with _guarded_defensive_materialize():
        clone = _byte_guarded_clone(value, slot_id=slot_id, state_dict_name=state_dict_name)
        try:
            canonical = tuple(int(v) for v in clone.stride()) == _default_dense_stride(
                tuple(int(v) for v in clone.shape)
            )
        except (RuntimeError, TypeError, ValueError, NotImplementedError):
            canonical = False
        if not canonical:
            clone = clone.clone(memory_format=torch.contiguous_format)
        # r67 C4: the r65 "staging hardening (b)" extra clone that engineered ``_version == 0``
        # is REMOVED with the ``version_is_zero`` dim itself: ``_version`` is a
        # ``refuse_on_any_read`` policy row (oracle-1's default copy leaves 1 or 2, never a
        # stable scalar), so no staged form needs to -- or may -- manufacture a version value.
    return clone


class RunResourceCeiling:
    """Aggregate replay resource admission for one loaded-sparse transaction (r59).

    Bounds two axes the per-op byte budget (:func:`_allocation_budget_bytes`) is
    structurally blind to, closing the allocation-DoS class as a WHOLE rather than
    op-by-op (each prior round found a new op-shape seam the byte budget missed):

    * output COUNT -- the number of output tensors a projection/bind may realize,
      bounded per call and aggregately by ``max(recorded * 8, 4096)`` and refused
      typed at ``op_output_count_preflight``. The front line is a count-instrumented
      ``FakeTensorMode`` in ``_preflight_call_allocation`` (refuses DURING fanout,
      before the full fake OR real tree exists, so a huge N cannot self-DoS the
      projection); :meth:`charge_realized_outputs` is the bind-time backstop for any
      projection-skipped path (data-dependent fail-open, or no size source -- r61).
    * re-materialization BYTES -- every TorchLens-owned sparse tensor snapshot clone
      (op-output snapshots, the accepted runtime input mirror, state staging and bind
      clones, and the run-time state clone) is byte-guarded against the live per-device
      budget BEFORE it allocates (:meth:`guarded_clone` / :func:`_byte_guarded_clone` ->
      ``clone_allocation_preflight``). An honest clone equals an honest recorded slot
      that already passed run-prep, so the guard is provably non-regressive; a tampered
      huge view (honest small recorded slot, inflated size literal) is refused before
      the clone materializes it.

    The third aggregate axis (whole-run RETAINED bytes) is a run-prep floor in
    :func:`_preflight_retention_floor`, co-located because it shares the same budget.
    """

    __slots__ = ("_expected_output_count", "_realized_output_count")

    def __init__(self, descriptor: SparseRunDescriptor) -> None:
        self._expected_output_count = sum(len(call.output_slot_ids) for call in descriptor.calls)
        self._realized_output_count = 0

    def aggregate_output_count_ceiling(self) -> int:
        """Aggregate realized-leaf ceiling ``max(sum recorded * MARGIN, FLOOR)``."""

        return max(self._expected_output_count * _OUTPUT_COUNT_MARGIN, _OUTPUT_COUNT_FLOOR)

    def per_call_output_count_ceiling(self, call: Any) -> int:
        """Per-call realized-leaf ceiling ``max(recorded * MARGIN, FLOOR)``."""

        recorded = len(getattr(call, "output_slot_ids", ()) or ())
        return max(recorded * _OUTPUT_COUNT_MARGIN, _OUTPUT_COUNT_FLOOR)

    def charge_realized_outputs(self, call: Any, count: int) -> None:
        """Accrue realized output leaves; refuse typed past the aggregate ceiling.

        The bind-time backstop for gate 1: a projection-skipped call (data-dependent
        fail-open, or one carrying no size source -- neither a tensor operand nor a
        numeric literal, r61) cannot smuggle an unbounded realized-output tree past
        the front-line projection. Honest realized equals recorded, so the aggregate
        margin is pure headroom and this never fires on a faithful run.
        """

        self._realized_output_count += count
        ceiling = self.aggregate_output_count_ceiling()
        if self._realized_output_count > ceiling:
            raise RunCapabilityUnavailableError(
                f"Sparse replay realized {self._realized_output_count} output tensors, "
                f"exceeding the aggregate ceiling {ceiling} (recorded "
                f"{self._expected_output_count}). A tampered descriptor cannot drive an "
                "unbounded output-count allocation on the default run path.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="op_output_count_preflight",
                call_id=getattr(call, "call_id", None),
                affected_op_labels=tuple(getattr(call, "op_labels", ())),
                charged_output_count=self._realized_output_count,
                output_count_ceiling=ceiling,
            )

    def guarded_clone(
        self,
        value: torch.Tensor,
        *,
        call_id: str | None,
        slot_id: str | None,
        affected_op_labels: Sequence[str] = (),
        mirror_requires_grad: bool = False,
    ) -> torch.Tensor:
        """Byte-guard one sparse tensor snapshot clone before it allocates (gate 3).

        Delegates to :func:`_byte_guarded_clone` -- ``numel * itemsize`` (pure integer
        math, no allocation) compared to the SAME never-under-estimating per-device
        budget run-prep uses; a clone that could never fit is refused typed at
        ``clone_allocation_preflight`` BEFORE the materializing ``.clone()`` and before
        any shape-mismatch check. This closes the view-exclusion / bind-clone
        composition (r58 free_2): the projection correctly charges a view zero NEW
        bytes, and this bounds the framework's own re-materialization of that view at
        every snapshot site -- op outputs, the accepted runtime input mirror, and the
        run-time state clone (r61 corr_2). ``mirror_requires_grad=True`` restores the
        source leaf's ``requires_grad`` on the clone where legal (the r37 corr2-7
        runtime-mirror rule); state clones keep their recorded-``trainable`` rule at
        the call site instead. An honest clone equals an honest recorded slot that
        already passed the run-prep bound, so a faithful run is never refused here.
        """

        clone = _byte_guarded_clone(
            value,
            call_id=call_id,
            slot_id=slot_id,
            affected_op_labels=affected_op_labels,
        )
        if mirror_requires_grad and bool(value.requires_grad) and not clone.requires_grad:
            try:
                clone.requires_grad_(True)
            except RuntimeError:
                # Non-differentiable dtype cannot require grad; the source flag could
                # not have been set either, so this is unreachable in practice --
                # degrade to the detached clone rather than aborting the bind.
                pass
        return clone


_OP_OUTPUT_SLOT_ROLES = frozenset({TensorSlotRole.INTERMEDIATE, TensorSlotRole.OUTPUT})
"""Slot roles whose tensors are ALLOCATED by an op during replay (r55 free_1).

Parameters/buffers are declared state (covered by the summed state pass); model
inputs are user-supplied; constants are embedded payloads. Only INTERMEDIATE and
OUTPUT slots carry a recorded op-output shape the replay path allocates, so they
are the recorded-output allocation bound.
"""


def _recorded_output_slots(descriptor: SparseRunDescriptor) -> tuple[TensorSlotDescriptor, ...]:
    """Return the op-produced output slots whose replay allocation is size-driving."""

    return tuple(slot for slot in descriptor.tensor_slots if slot.role in _OP_OUTPUT_SLOT_ROLES)


def _preflight_run_allocation(
    representatives: Sequence[TensorSlotDescriptor],
    output_slots: Iterable[TensorSlotDescriptor] = (),
) -> None:
    """Refuse an infeasible run allocation BEFORE any op executes (r53 free_1 / r55 free_1).

    Two per-device passes against the never-under-estimating live budget
    (:func:`_allocation_budget_bytes`), so a refusal can only fire on a request
    that could never have been satisfied on this host -- it replaces the raw
    allocator OOM/OOM-kill a hostile descriptor integer would otherwise cause, and
    never over-triggers a legitimate large-but-feasible model:

    * ``representatives`` -- the random role-init state slots (one per alias group)
      are SUMMED per device, because they are all staged at once
      (``state_allocation_preflight``).
    * ``output_slots`` -- each recorded op-output slot is compared INDIVIDUALLY,
      because replay peak memory is one call's output, never the whole-graph sum;
      this is the recorded-output allocation bound that catches a self-consistent
      ``arange(10**12)``-scale artifact before the DAG allocates
      (``op_allocation_preflight``).

    W2's per-call ``FakeTensorMode`` projection preflight composes with this: it is
    the primary op-agnostic projection, this is the run-prep fallback bound.
    """

    from .errors import RunCapabilityUnavailableError

    required: dict[str, int] = defaultdict(int)
    slot_ids_by_device: dict[str, list[str]] = defaultdict(list)
    devices: dict[str, torch.device] = {}
    for slot in representatives:
        device = _slot_device(slot)
        key = str(device)
        devices[key] = device
        required[key] += math.prod(slot.shape) * _torch_dtype(slot.dtype).itemsize
        slot_ids_by_device[key].append(slot.slot_id)
    for key, requested in sorted(required.items()):
        available = _allocation_budget_bytes(devices[key])
        if requested > available:
            sample = ",".join(sorted(slot_ids_by_device[key])[:8])
            raise RunCapabilityUnavailableError(
                f"Random state initialization for device {key!r} requires "
                f"{requested} bytes, but only {available} bytes are available "
                f"on this host (slots: {sample}). The declared state cannot be "
                "allocated here; stage smaller state or run on a larger host.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="state_allocation_preflight",
                device=key,
                required_bytes=requested,
                available_bytes=available,
            )

    for slot in output_slots:
        device = _slot_device(slot)
        requested = math.prod(slot.shape) * _torch_dtype(slot.dtype).itemsize
        available = _allocation_budget_bytes(device)
        if requested > available:
            raise RunCapabilityUnavailableError(
                f"Recorded op-output slot {slot.slot_id!r} for device {str(device)!r} "
                f"requires {requested} bytes, but only {available} bytes are available "
                "on this host. The recorded output cannot be allocated here; the "
                "descriptor may be tampered or the host is too small.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="op_allocation_preflight",
                device=str(device),
                required_bytes=requested,
                available_bytes=available,
            )


def _preflight_retention_floor(descriptor: SparseRunDescriptor) -> None:
    """Refuse a run whose GUARANTEED retained clone bytes provably exceed the budget (r59).

    Loaded-sparse replay retains a materialized clone of every taken-path op output for
    the life of the returned trace (one ``Op.out`` clone per ``(call, output_slot_id)``
    at bind, plus one reconstruct clone per model-``output`` slot). The SUM of those
    recorded op-output bytes is therefore a guaranteed LOWER BOUND on replay memory --
    a floor. A descriptor whose per-device floor exceeds the never-under-estimating
    live budget could never complete on this host: every honest per-op allocation may
    pass the per-slot bound (:func:`_preflight_run_allocation`) and every clone may pass
    its per-clone byte guard, yet their cumulative retention guarantees a mid-replay
    allocator death (r58 free_1/free_2 accumulation seam: e.g. 40 honest 4 GB output
    slots = 160 GB floor > a 105 GB host budget). This refuses that typed at
    ``run_retention_preflight`` before any call executes.

    The floor UNDER-counts true retention (it ignores live ``slot_values`` entries, raw
    ``call_outputs``, attestation/witness snapshots) and the budget never under-estimates,
    so ``floor > budget`` proves a guaranteed OOM -- a refusal can NEVER hit a completable
    run (zero false refusals). A legit deep model whose cumulative retention is within
    budget (a deep-adds chain of small outputs) is untouched.
    """

    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    floor: dict[str, int] = defaultdict(int)
    devices: dict[str, torch.device] = {}

    def _charge(slot: TensorSlotDescriptor) -> None:
        device = _slot_device(slot)
        key = str(device)
        devices[key] = device
        floor[key] += math.prod(slot.shape) * _torch_dtype(slot.dtype).itemsize

    # One retained ``Op.out`` clone per taken-path op label (bind, ``:2957``): alias/
    # version slots are NOT in ``output_slot_ids`` so each producing op label charges
    # exactly once, mirroring ``_bind_call_outputs``'s ``output_slot_ids`` zip.
    for call in descriptor.calls:
        for slot_id in call.output_slot_ids:
            slot = slots.get(slot_id)
            if slot is not None:
                _charge(slot)
    # One additional retained reconstruct clone per model-output slot (``:3161``).
    for slot in descriptor.tensor_slots:
        if slot.role is TensorSlotRole.OUTPUT:
            _charge(slot)

    for key, requested in sorted(floor.items()):
        available = _allocation_budget_bytes(devices[key])
        if requested > available:
            raise RunCapabilityUnavailableError(
                f"Loaded-sparse replay retains at least {requested} bytes of op-output "
                f"clones on device {key!r} for the run's lifetime, but only {available} "
                "bytes are available on this host. The cumulative retained state exceeds "
                "the budget and the run is guaranteed to fail mid-replay; the descriptor "
                "may be tampered or the host is too small.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="run_retention_preflight",
                device=key,
                required_bytes=requested,
                available_bytes=available,
            )


def _preflight_random_init_allocation(
    representatives: Sequence[TensorSlotDescriptor],
) -> None:
    """Backward-compatible spelling for the state-only preflight (r53 free_1).

    Retained so existing callers keep the summed random role-init behavior;
    :func:`_preflight_run_allocation` is the generalized entry that also bounds
    recorded op-output slots.
    """

    _preflight_run_allocation(representatives)


def _validate_alias_allocation_contract(members: list[TensorSlotDescriptor]) -> None:
    """Require all members of one allocation group to share an initializer contract."""

    first = members[0]
    first_binding = first.state_binding
    assert first_binding is not None
    first_policy = CANONICAL_INITIALIZER_BY_ROLE[first_binding.semantic_role]
    for member in members[1:]:
        binding = member.state_binding
        assert binding is not None
        policy = CANONICAL_INITIALIZER_BY_ROLE[binding.semantic_role]
        if (
            member.shape != first.shape
            or member.dtype != first.dtype
            or member.device_type != first.device_type
            or member.device_index != first.device_index
            or policy is not first_policy
        ):
            raise _binding_error(
                (
                    _diagnostic(
                        RunnableErrorCode.STATE_ALIAS_CONFLICT,
                        "Alias group has incompatible allocation contracts.",
                        detection_stage="state_random_alias_preflight",
                        details=(("slot_ids", ",".join(item.slot_id for item in members)),),
                    ),
                )
            )


def _generator_for_slot(
    slot: TensorSlotDescriptor,
    seed: int | None,
    generators: dict[str, torch.Generator],
) -> torch.Generator | None:
    """Return one isolated per-device generator when an explicit seed is supplied."""

    if seed is None:
        return None
    # r79 seed-door mirror: no path may reach raw ``manual_seed`` with a
    # bool/out-of-range seed even if it bypassed the run door.
    validate_run_seed(seed)
    device = _slot_device(slot)
    key = str(device)
    if key not in generators:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        generators[key] = generator
    return generators[key]


def _kaiming_fan_in(shape: tuple[int, ...]) -> int:
    """Return the frozen N1-a fan-in for a nonempty Kaiming-initialized shape."""

    return math.prod(shape[1:]) if len(shape) >= 2 else max(1, shape[0])


def initializer_contract_diagnostics(
    slot: TensorSlotDescriptor,
) -> tuple[RunnableDiagnostic, ...]:
    """Validate the ``torchlens_role_init_v2`` totality contract for one slot.

    Shared by producer preflight AND the runtime initializer (defense in depth,
    corr2_3): a legal ``numel() == 0`` slot is total for every role (allocated
    and returned with ZERO generator consumption), and every nonempty Kaiming
    slot must have finite positive ``fan_in``. An unsupported contract fails
    typed here, never by division or backend sampling at run time.
    """

    binding = slot.state_binding
    if binding is None:
        return ()
    policy = CANONICAL_INITIALIZER_BY_ROLE[binding.semantic_role]
    if policy is not InitializerPolicy.KAIMING_NORMAL:
        return ()
    if math.prod(slot.shape) == 0:
        # Degenerate-total: nothing to sample.
        return ()
    if not slot.shape:
        return (
            _diagnostic(
                RunnableErrorCode.STATE_DTYPE_MISMATCH,
                f"Kaiming initialization is unsupported for scalar slot {slot.slot_id!r}.",
                detection_stage="state_random_initializer",
                details=(("slot_id", slot.slot_id), ("dtype", slot.dtype)),
            ),
        )
    fan_in = _kaiming_fan_in(slot.shape)
    if fan_in <= 0:
        return (
            _diagnostic(
                RunnableErrorCode.STATE_DTYPE_MISMATCH,
                f"Kaiming initialization requires finite positive fan_in for slot "
                f"{slot.slot_id!r}; got {fan_in}.",
                detection_stage="state_random_initializer",
                details=(("slot_id", slot.slot_id), ("shape", repr(slot.shape))),
            ),
        )
    return ()


def _initialize_slot(
    slot: TensorSlotDescriptor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Allocate and fill one representative slot under ``torchlens_role_init_v2``.

    r67 C5: the WHOLE allocation+fill is a defensive materialization -- neutral ambient,
    so a caller inference_mode/no_grad region cannot mint inference-mode slots or alter
    the fill semantics.
    """

    with _guarded_defensive_materialize():
        return _initialize_slot_neutral(slot, generator)


def _initialize_slot_neutral(
    slot: TensorSlotDescriptor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Allocate and fill one representative slot (inside the neutral ambient)."""

    binding = slot.state_binding
    assert binding is not None
    dtype = _torch_dtype(slot.dtype)
    device = _slot_device(slot)
    policy = CANONICAL_INITIALIZER_BY_ROLE[binding.semantic_role]
    _validate_initializer_dtype(slot, dtype)
    try:
        value = torch.empty(slot.shape, dtype=dtype, device=device)
    except (RuntimeError, MemoryError) as exc:
        # r53 free_1 belt: the allocation preflight refuses infeasible totals
        # before this point; a RESIDUAL allocator failure (budget shifted under
        # us, fragmentation) still surfaces as the SAME typed diagnostic, never
        # a raw allocator traceback.
        from .errors import RunCapabilityUnavailableError

        raise RunCapabilityUnavailableError(
            f"Random state initialization could not allocate slot {slot.slot_id!r} "
            f"(shape={slot.shape!r}, dtype={slot.dtype!r}, device={device}): {exc}",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="state_allocation_preflight",
            slot_id=slot.slot_id,
        ) from exc
    if value.numel() == 0:
        # v2 degenerate totality: a legal empty slot returns its allocation
        # immediately -- no sampling, PROVABLY zero generator consumption.
        return value
    if policy is InitializerPolicy.ZEROS:
        return value.zero_()
    if policy is InitializerPolicy.ONES:
        return value.fill_(1)
    contract = initializer_contract_diagnostics(slot)
    if contract:
        raise _binding_error(contract)
    if not dtype.is_floating_point or not slot.shape:
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.STATE_DTYPE_MISMATCH,
                    f"Kaiming initialization is unsupported for slot {slot.slot_id!r}.",
                    detection_stage="state_random_initializer",
                    details=(("slot_id", slot.slot_id), ("dtype", slot.dtype)),
                ),
            )
        )
    fan_in = _kaiming_fan_in(slot.shape)
    return value.normal_(mean=0.0, std=math.sqrt(2.0 / fan_in), generator=generator)


def _validate_initializer_dtype(slot: TensorSlotDescriptor, dtype: torch.dtype) -> None:
    """Reject dtype/semantic-role combinations outside frozen N1-a."""

    binding = slot.state_binding
    assert binding is not None
    floating_roles = {
        StateSlotRole.WEIGHT,
        StateSlotRole.BIAS,
        StateSlotRole.NORM_SCALE,
        StateSlotRole.NORM_OFFSET,
        StateSlotRole.RUNNING_MEAN,
        StateSlotRole.RUNNING_VAR,
    }
    integral_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    compatible = (
        dtype.is_floating_point
        if binding.semantic_role in floating_roles
        else binding.semantic_role is not StateSlotRole.COUNTER or dtype in integral_dtypes
    )
    if compatible:
        return
    raise _binding_error(
        (
            _diagnostic(
                RunnableErrorCode.STATE_DTYPE_MISMATCH,
                f"State role {binding.semantic_role.value!r} is incompatible with {slot.dtype!r}.",
                detection_stage="state_random_initializer",
                details=(
                    ("slot_id", slot.slot_id),
                    ("semantic_role", binding.semantic_role.value),
                    ("dtype", slot.dtype),
                ),
            ),
        )
    )


def _torch_dtype(dtype_name: str) -> torch.dtype:
    """Resolve one recorded public torch dtype without evaluating artifact text."""

    name = dtype_name.removeprefix("torch.")
    # r47 secD_1: resolve through the sanctioned ``torch_attr`` helper so an attacker slot-dtype
    # (``"onnx"`` / ``"_dynamo"`` / ``"has_cuda"``) on the ``.run()`` random-init path reads
    # ``torch.__dict__`` directly and NEVER fires ``torch.__getattr__`` (no lazy submodule import,
    # no deprecated-attr shim, no raw ImportError) before the ``isinstance(torch.dtype)`` gate.
    value = torch_attr(name)
    if not isinstance(value, torch.dtype):
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.STATE_DTYPE_MISMATCH,
                    f"Recorded state dtype {dtype_name!r} is unsupported.",
                    detection_stage="state_random_dtype",
                    details=(("dtype", dtype_name),),
                ),
            )
        )
    return value


def _slot_device_compatible(value: torch.Tensor, slot: TensorSlotDescriptor) -> bool:
    """Return whether a tensor already satisfies a slot's recorded device contract."""

    if value.device.type != slot.device_type:
        return False
    return (
        slot.device_index is None
        or value.device.index is None
        or value.device.index == slot.device_index
    )


def stage_state_to_slot_devices(
    descriptor: SparseRunDescriptor,
    values: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Stage every bound state value to its recorded slot device, atomically (r37 R5).

    LAZY placement: blobs stay transport-placed (``map_location`` semantics) through
    load and analysis; THIS single helper -- the sole execution-placement authority
    -- runs once at run preparation for embedded capture state, staged user state,
    and required non-persistent buffers alike. Transfers happen once per shared
    value (alias groups keep one allocation and their shared identity), preserve
    dtype (already validated) and ``requires_grad``, and publish NOTHING on failure:
    a device this runtime cannot allocate raises the typed
    ``run_capability_unavailable`` refusal before any callable resolves an input.
    """

    from .errors import RunCapabilityUnavailableError

    slots_by_id = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    staged: dict[str, torch.Tensor] = {}
    moved_by_identity: dict[int, torch.Tensor] = {}
    for slot_id, value in values.items():
        slot = slots_by_id.get(slot_id)
        if slot is None or _slot_device_compatible(value, slot):
            staged[slot_id] = value
            continue
        cached = moved_by_identity.get(id(value))
        if cached is None:
            try:
                # r67 C5: the cross-device staging transfer is a defensive
                # materialization -- neutral ambient, never the caller's (a ``.to()``
                # inside a caller inference_mode region would mint an inference tensor).
                with _state.pause_logging(), _guarded_defensive_materialize():
                    cached = value.to(_slot_device(slot))
            except (RuntimeError, AssertionError) as exc:
                raise RunCapabilityUnavailableError(
                    f"State slot {slot_id!r} requires device "
                    f"{slot.device_type}"
                    + (f":{slot.device_index}" if slot.device_index is not None else "")
                    + f", which this runtime cannot stage: {exc}",
                    code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                ) from exc
            if value.requires_grad and not cached.requires_grad:
                cached.requires_grad_(True)
            moved_by_identity[id(value)] = cached
        staged[slot_id] = cached
    return MappingProxyType(staged)


def _slot_device(slot: TensorSlotDescriptor) -> torch.device:
    """Build the recorded allocation device for one state slot."""

    if slot.device_index is None:
        return torch.device(slot.device_type)
    return torch.device(slot.device_type, slot.device_index)


def _state_slots(descriptor: SparseRunDescriptor) -> tuple[TensorSlotDescriptor, ...]:
    """Return all state slots and reject parameter/buffer slots without bindings."""

    state_roles = {TensorSlotRole.PARAMETER, TensorSlotRole.BUFFER}
    missing = tuple(
        slot.slot_id
        for slot in descriptor.tensor_slots
        if slot.role in state_roles and slot.state_binding is None
    )
    if missing:
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.MISSING_TENSOR_SLOT,
                    "Parameter or buffer slot is missing its state binding contract.",
                    detection_stage="state_slot_preflight",
                    details=(("slot_ids", ",".join(missing)),),
                ),
            )
        )
    return tuple(
        slot
        for slot in descriptor.tensor_slots
        if slot.role in state_roles and slot.state_binding is not None
    )


def _persistent_state_slots(
    descriptor: SparseRunDescriptor,
) -> tuple[TensorSlotDescriptor, ...]:
    """Return only slots belonging to the canonical persistent state mapping."""

    return tuple(
        slot
        for slot in _state_slots(descriptor)
        if slot.state_binding is not None and slot.state_binding.persistent
    )


def _nonpersistent_buffer_slots(
    descriptor: SparseRunDescriptor,
) -> tuple[TensorSlotDescriptor, ...]:
    """Return registered buffer slots intentionally excluded from ``state_dict``."""

    return tuple(
        slot
        for slot in _state_slots(descriptor)
        if slot.role is TensorSlotRole.BUFFER
        and slot.state_binding is not None
        and not slot.state_binding.persistent
    )


def _require_descriptor(trace: Any) -> SparseRunDescriptor:
    """Return a sparse descriptor or raise a structured binding error."""

    descriptor = trace.__dict__.get("_runnable_descriptor")
    if not isinstance(descriptor, SparseRunDescriptor):
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
                    "State binding requires a loaded sparse runnable Trace.",
                    detection_stage="state_descriptor_presence",
                ),
            )
        )
    return descriptor


def _binding_error(diagnostics: tuple[RunnableDiagnostic, ...]) -> StateBindingError:
    """Build one structured strict state-binding exception."""

    codes = tuple(diagnostic.code.value for diagnostic in diagnostics)
    return StateBindingError(
        f"Strict state binding failed with {len(diagnostics)} diagnostic(s): {', '.join(codes)}.",
        diagnostics=diagnostics,
        codes=codes,
    )


def _diagnostic(
    code: RunnableErrorCode,
    message: str,
    *,
    detection_stage: str,
    details: tuple[tuple[str, str], ...] = (),
) -> RunnableDiagnostic:
    """Build one state-binding diagnostic in the frozen shared shape."""

    return RunnableDiagnostic(
        code=code,
        message=message,
        registry_id=None,
        affected_op_labels=(),
        recorded_runtime=None,
        current_runtime=str(torch.__version__),
        detection_stage=detection_stage,
        resolver_provenance=None,
        analysis_load_available=True,
        details=details,
    )


def runnable_tensor_byte_digest(value: torch.Tensor) -> str:
    """Return a SHA-256 digest of one tensor's exact logical value.

    Parameters
    ----------
    value:
        Dense tensor whose capture/runtime bytes should be attested.

    Returns
    -------
    str
        Lowercase SHA-256 hexadecimal digest of dtype, shape, and contiguous
        CPU bytes.

    Raises
    ------
    RunPreconditionError
        If the tensor is not a plain strided dense tensor (r35 I5 defense in
        depth: the digest helper itself refuses exotic layouts typed instead of
        surfacing a raw backend error mid-materialization).
    """

    if (
        value.layout is not torch.strided
        or bool(getattr(value, "is_meta", False))
        or value.is_nested
        or bool(getattr(value, "is_quantized", False))
        or any(name is not None for name in (value.names or ()))
    ):
        from .errors import RunPreconditionError

        raise RunPreconditionError(
            "Byte digesting requires a plain strided dense tensor; got layout "
            f"{value.layout} (meta={bool(getattr(value, 'is_meta', False))}, "
            f"nested={bool(value.is_nested)}, "
            f"quantized={bool(getattr(value, 'is_quantized', False))}).",
            code=RunnableErrorCode.INPUT_TREE_MISMATCH.value,
        )
    with _state.pause_logging():
        cpu_value = value.detach().cpu().contiguous()
        payload = cpu_value.reshape(-1).view(torch.uint8).numpy().tobytes()
        logical_prefix = f"{cpu_value.dtype}|{tuple(cpu_value.shape)}|".encode("utf-8")
    return sha256(logical_prefix + payload).hexdigest()


__all__ = [
    "PreparedRunnableState",
    "load_trace_state_dict",
    "prepare_runnable_state",
    "runnable_tensor_byte_digest",
]
