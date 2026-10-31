"""Cross-thread and scalar escape observation."""

from __future__ import annotations

import functools
import threading
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch._ops as _torch_ops  # r47 hon2_1: enumerate the ``torch.ops.*`` __call__ classes
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from ... import _state
from ..._utils._callable_safety import private_c_forward_op_module_names
from ._completeness_types import _WitnessState
from ._tl import (
    get_buffer_address,
    get_tensor_meta,
    session_meta_is_anchored,
)

if TYPE_CHECKING:
    from .completeness_witness import (
        _ACTIVE_WITNESS_STATE,
        _CAPTURED_STORAGE_PTRS,
        _DISABLE_MODE_SITE_CATEGORIES,
        _DISPATCH_TENSOR_ORIGINS,
        _HOST_ESCAPE_CROSS_THREAD_CAPTURED,
        _HOST_ESCAPE_OBSERVER_FAILED,
        _RUNNABLE_INPUT_STORAGE_SITES,
        _internal_read_state,
        _iter_tensors_deep,
        _raw_storage_ptr_no_observe,
        _record_escape_source_tensor,
    )

__all__ = (
    "_nonowner_ptr_is_captured",
    "_nonowner_touch_is_captured",
    "_nonowner_escape_observe",
    "observe_nonowner_operands",
    "_torch_ops_call_classes",
    "_make_nonowner_ops_call",
    "_private_c_forward_op_modules",
    "_private_c_module_callables",
    "_make_nonowner_private_c_callable",
    "string_escape_is_owner_thread",
    "host_escape_observer_install_failed",
    "record_host_string_escape_source",
    "audit_disable_current_modes_sites",
    "_internal_read_active",
)


def _nonowner_ptr_is_captured(state: _WitnessState, ptr: int) -> bool:
    """Return whether a storage pointer belongs to a captured input / param / activation (r43)."""

    trace = state.trace
    param_addresses = getattr(trace, "_param_storage_addresses", None)
    if param_addresses and ptr in param_addresses:
        return True
    input_sites = _RUNNABLE_INPUT_STORAGE_SITES.get(trace)
    if input_sites is not None and ptr in input_sites:
        return True
    # Activation storage identity is LIVENESS-VERIFIED: the ptr matches only when a captured
    # producing tensor is still alive AND still occupies this exact address (so the touched
    # tensor genuinely aliases it). A freed-then-reused address has only dead weakrefs -> no match
    # (no over-trigger on a benign own-tensor allocation that inherited a stale address).
    captured = _CAPTURED_STORAGE_PTRS.get(trace)
    if captured is not None:
        for producer_ref in captured.get(ptr, ()):
            producer = producer_ref()
            if producer is not None and _raw_storage_ptr_no_observe(producer) == ptr:
                return True
    return False


def _nonowner_touch_is_captured(state: _WitnessState, tensor: Any) -> bool:
    """Return whether a non-owner thread's touched tensor is a CAPTURED tensor (r43).

    Captured membership (all reads GIL-atomic; NO torch op, NO ``pause_logging``, NO observer
    recursion): a capture label OR a registered param/buffer state address OR a dispatch-origin
    ledger hit (a previously-registered owner-derived alias) OR STORAGE IDENTITY -- the tensor's
    true-original storage pointer is a captured input-leaf, parameter, or activation pointer. A
    benign own-tensor read (no label, no state, no ledger entry, unrelated storage) returns
    ``False`` and never ceilings the capture.
    """

    if not isinstance(tensor, torch.Tensor):
        return False
    trace = state.trace
    meta = get_tensor_meta(tensor)
    # This path must remain observer-free: get_tensor_label() also validates the
    # live storage, which calls the patched untyped_storage() host-escape surface.
    # On a non-owner thread that wrapper re-enters this predicate indefinitely.
    # The current-session object anchor is sufficient for captured membership:
    # even a storage-rebound captured object must still trip the cross-thread
    # ceiling, while a stale label from an earlier capture remains rejected.
    if meta is not None and isinstance(meta.label_raw, str) and session_meta_is_anchored(meta):
        return True
    if meta is not None and getattr(meta, "address", None) is not None:
        return True
    if get_buffer_address(tensor) is not None:
        return True
    registry = _DISPATCH_TENSOR_ORIGINS.get(trace)
    if registry is not None and registry.get(tensor) is not None:
        return True
    ptr = _raw_storage_ptr_no_observe(tensor)
    return bool(ptr is not None and _nonowner_ptr_is_captured(state, ptr))


def _nonowner_escape_observe(state: _WitnessState, tensor: Any) -> None:
    """Ceiling the capture if a non-owner thread touched a captured tensor (r43).

    The ONE non-owner belt action: NO origin resolution, NO ``pause_logging``, NO precise
    witness -- a captured-tensor touch off-owner is outside the single-owner-thread replay
    model and simply marks the cross-thread ceiling. Must be called only when the caller has
    confirmed ``not owner`` and ``state.belt_armed``.
    """

    if _nonowner_touch_is_captured(state, tensor):
        _HOST_ESCAPE_CROSS_THREAD_CAPTURED.add(state.trace)


def observe_nonowner_operands(args: tuple[Any, ...], kwargs: dict[str, Any] | None) -> None:
    """Ceiling the runnable capture when a NON-owner thread CONSUMES a captured operand (r45 hon2_1).

    The r43 cross-thread belt patches tensor METHODS, so it only recognizes a non-owner thread's
    tensor->host escape when the escaped tensor's OBJECT IDENTITY is captured / owner-registered
    (a capture label, a registered state address, a dispatch-origin ledger hit, or captured
    storage identity). A tensor DERIVED on the worker from a captured input (``(gate * 2).sum()``,
    ``gate.clone()``, ``gate + 0``, ``gate @ w``, ``torch.cat([gate], 0)`` ...) has FRESH storage
    that the OWNER-thread-only census / dispatch-origin ledger never registered, so its later value
    escape was unwitnessed -> false ``VERIFIED`` on a changed input a fresh live run would branch
    differently on (the r44 hon2_1 finding).

    Every Python-visible torch/Tensor op flows through the GLOBAL torch-function wrapper (a
    process-wide monkeypatch, unlike the thread-local aten census / dispatch mode). This observer
    runs on the wrapper's NON-owner fast path and ceilings the artifact the FIRST time a non-owner
    thread runs ANY torch op that consumes a captured tensor as an OPERAND -- op-agnostic, so it
    covers the whole worker-derivation class by construction (no derived-product registry: ceiling
    at the first consumption makes deeper-chain and escape-time provenance moot; the derived tensor
    does not even exist yet).

    Fail-CLOSED (r45 Fork C): any operand-inspection error during an armed capture ceilings the
    trace -- an inspection failure on a non-owner op cannot be read as "no captured touch"
    (validation is a tripwire). Benign-worker-safe: a non-owner thread operating only on tensors it
    created INDEPENDENTLY of the capture matches no captured-membership signal and stays
    ``VERIFIED``. The wrapper caller has already confirmed ``_state._nonowner_belt_armed`` and
    non-owner identity, so the disarmed global hot path pays only a single bool read.

    Parameters
    ----------
    args:
        The positional operands of the wrapped torch call (``*args``).
    kwargs:
        The keyword operands of the wrapped torch call (``**kwargs``), or ``None``.
    """

    state = _ACTIVE_WITNESS_STATE
    if state is None or not state.belt_armed:
        return
    if _state._active_trace is not state.trace:
        return
    if threading.get_ident() == state.owner_thread_id:
        return
    try:
        for container in (args, kwargs):
            if container is None:
                continue
            for operand in _iter_tensors_deep(container):
                if _nonowner_touch_is_captured(state, operand):
                    _HOST_ESCAPE_CROSS_THREAD_CAPTURED.add(state.trace)
                    return
    except Exception:
        # An operand-inspection failure on a non-owner op during an armed capture cannot prove
        # "no captured touch": fail closed (ceiling), never silently pass.
        _HOST_ESCAPE_CROSS_THREAD_CAPTURED.add(state.trace)


def _torch_ops_call_classes() -> tuple[type, ...]:
    """Feature-detect every ``torch._ops`` class that defines its OWN ``__call__`` (r47 hon2_1).

    The r45 hon2_1 non-owner operand observer runs on the GLOBAL torch-FUNCTION wrapper, but the
    ``torch.ops.*`` (aten / higher-order / TorchBind) surface bypasses that wrapper entirely: a
    worker thread deriving from / reading a captured tensor via ``torch.ops.aten.mul.Tensor(...)``,
    ``torch.ops.aten.sum.default(...)``, ``torch.ops.aten._local_scalar_dense(...)``, ... never
    hits the wrapper and the aten dispatch census is thread-LOCAL (a ``TorchDispatchMode`` cannot
    see a non-owner thread), so its captured-operand consumption went unwitnessed -> false
    ``VERIFIED`` on a diverging changed input (the r46 hon2_1 finding).

    Every Python-visible ``torch.ops.*`` call flows through the ``__call__`` of a small set of
    ``torch._ops`` classes (``OpOverloadPacket`` / ``OpOverload`` / ``TorchBindOpOverload`` /
    ``HigherOrderOperator`` (+ the abstract ``OperatorBase``)). This scans STRUCTURALLY -- every
    ``torch._ops`` class object defining its own callable ``__call__`` -- rather than importing the
    names, which is version-robust across the declared torch floor->ceiling (2.1 -> 2.12+):
    ``TorchBindOpOverload`` only appeared ~2.4, so a by-name import would ``ImportError`` on older
    torch. A future torch that adds a call class is auto-covered by shape; a class whose ``__call__``
    is removed silently drops out (fail-closed install downgrades the capture, never a silent hole).

    Wrapping the WHOLE set including the abstract ``OperatorBase`` never double-observes: a concrete
    subclass's ``__call__`` does NOT chain to ``super().__call__``, so a single ``aten.mul.Tensor``
    call fires the observer exactly once (probed on torch 2.8).
    """

    out: list[type] = []
    for attr in dir(_torch_ops):
        obj = getattr(_torch_ops, attr, None)
        if isinstance(obj, type) and callable(obj.__dict__.get("__call__")):
            out.append(obj)
    return tuple(out)


def _make_nonowner_ops_call(original: Any) -> Any:
    """Wrap a ``torch._ops`` class ``__call__`` with the non-owner captured-operand observer (r47).

    The patched ``__call__`` short-circuits on a SINGLE bool read (``_nonowner_belt_armed``) so the
    disarmed steady state pays ~nothing, and a real eager OWNER forward hits the Python
    ``torch.ops.*.__call__`` path ZERO times (C++ dispatch; probed), so the armed owner window adds
    ~no overhead. When the belt is armed, a runnable capture is active, and the caller is a
    NON-owner thread, it routes the operands through the SAME storage-identity captured-membership
    test (:func:`observe_nonowner_operands`, fail-closed internally) BEFORE delegating to the
    original ``__call__``. The worker op is NEVER logged into the owner trace.
    """

    @functools.wraps(original)
    def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Observe non-owner ``torch._ops`` calls before delegating.

        Parameters
        ----------
        self:
            ``torch._ops`` receiver.
        *args:
            Positional operands passed to ``original``.
        **kwargs:
            Keyword operands passed to ``original``.

        Returns
        -------
        Any
            Result from ``original``.
        """
        if (
            _state._nonowner_belt_armed
            and _state._active_trace is not None
            and _state._active_owner_thread_id != threading.get_ident()
        ):
            observe_nonowner_operands(args, kwargs)
        return original(self, *args, **kwargs)

    _patched.__tl_nonowner_ops_observer__ = True  # type: ignore[attr-defined]
    return _patched


def _private_c_forward_op_modules() -> tuple[Any, ...]:
    """Resolve the patchable module-typed private-C forward-op modules (r49 hon2_1).

    Structurally enumerated from the canonical forward-op module authority
    (:func:`torchlens._utils._callable_safety.private_c_forward_op_module_names` -> the
    ``torch._C._*`` entries of ``_ALLOWED_FORWARD_OP_MODULES``), resolved on the RUNNING torch
    and filtered to ``types.ModuleType`` so:

    * a torch lacking one (``_sparse`` / ``_nested`` on an older build) degrades gracefully
      (skip-if-absent), and
    * the class-typed, read-only / non-Python-patchable holders (``_VariableFunctions`` /
      ``_TensorBase``) are EXCLUDED (their setattr raises -- accepted residual).

    On torch 2.8 this yields exactly ``{_nn, _special, _fft, _linalg, _sparse, _nested}``. A
    future private-C op module added to the curated set is auto-covered.
    """

    modules: list[Any] = []
    for name in private_c_forward_op_module_names():
        obj: Any = torch
        resolved = True
        for part in name.split(".")[1:]:  # skip the leading "torch"
            obj = getattr(obj, part, None)
            if obj is None:
                resolved = False
                break
        if resolved and isinstance(obj, types.ModuleType):
            modules.append(obj)
    return tuple(modules)


def _private_c_module_callables() -> tuple[tuple[Any, str, Any], ...]:
    """Return ``(module, attr, original)`` for every module-level callable of the patchable
    private-C forward-op modules (r49 hon2_1).

    Dunder module metadata (``__loader__`` / ``__spec__`` / ...) is skipped; every remaining
    module-level callable (the ~225 ``torch._C._{nn,special,fft,linalg,sparse,nested}`` free
    functions) is a patch target so the belt is surface-complete for the whole module, not a
    known-alias subset.
    """

    out: list[tuple[Any, str, Any]] = []
    for module in _private_c_forward_op_modules():
        for attr in dir(module):
            if attr.startswith("__"):
                continue
            value = getattr(module, attr, None)
            if callable(value):
                out.append((module, attr, value))
    return tuple(out)


def _make_nonowner_private_c_callable(original: Any) -> Any:
    """Wrap a private-C module FREE function with the non-owner captured-operand observer (r49).

    Twin of :func:`_make_nonowner_ops_call` for MODULE-level free functions (no ``self``
    receiver): private-C ops (``torch._C._nn.gelu(gate)``) are a THIRD op surface -- they bypass
    BOTH the global torch-FUNCTION wrapper (no ``__torch_function__``) AND the ``torch._ops.*``
    class patch (they dispatch their inner aten op down in C++), so a non-owner worker consuming
    a captured operand through one went unwitnessed -> false ``VERIFIED`` (the r48 hon2_1
    finding). Same three-term armed/owner short-circuit (disarmed steady state pays one bool
    read; an OWNER-thread forward never reaches ``observe_nonowner_operands``) and the same
    fail-closed operand test.
    """

    @functools.wraps(original)
    def _patched(*args: Any, **kwargs: Any) -> Any:
        """Observe non-owner private-C free-function calls before delegating.

        Parameters
        ----------
        *args:
            Positional operands passed to ``original``.
        **kwargs:
            Keyword operands passed to ``original``.

        Returns
        -------
        Any
            Result from ``original``.
        """
        if (
            _state._nonowner_belt_armed
            and _state._active_trace is not None
            and _state._active_owner_thread_id != threading.get_ident()
        ):
            observe_nonowner_operands(args, kwargs)
        return original(*args, **kwargs)

    _patched.__tl_nonowner_ops_observer__ = True  # type: ignore[attr-defined]
    return _patched


def string_escape_is_owner_thread(trace: Any) -> bool:
    """Return whether the current thread is the capture owner for the string hook (r43).

    Consumed by the wrappers.py ``__repr__``/``__str__``/``_str`` interception: the OWNER
    thread keeps ``print_override`` (which formats under a global ``pause_logging``); a
    NON-OWNER thread must NEVER flip that global toggle mid-forward (the hon2_4 crash), so it
    calls the original torch string function unchanged. With no active runnable witness state
    the legacy owner behavior is preserved (``True``).
    """

    state = _ACTIVE_WITNESS_STATE
    if state is None or state.trace is not trace:
        return True
    return threading.get_ident() == state.owner_thread_id


def host_escape_observer_install_failed(trace: Any) -> bool:
    """Return whether a required tensor->host value observer failed to install/restore (r39)."""

    return trace in _HOST_ESCAPE_OBSERVER_FAILED


def record_host_string_escape_source(trace: Any, tensor: Any) -> None:
    """Record a tensor->host VALUE escape via string formatting (r39 hon2_1).

    TorchLens intercepts ``__repr__`` / ``__str__`` / ``_str`` on a captured tensor and formats
    it internally under ``pause_logging()`` (``print_override`` -> ``.detach().cpu().numpy()``),
    which extracts the tensor's VALUES into the returned string -- a genuine tensor->host value
    escape the user can fold back into control flow (the string NaN guard). Because that
    extraction runs under PAUSED logging, the ordinary ``.numpy()`` / ``.item()`` escape
    observers are blind to it (they gate on ``_state._logging_enabled``), so the print
    interception records the SOURCE tensor here through the SAME attribution ladder.

    NOTE (r39): the reconciled plan's E6 measurement of str/repr transitivity was taken on RAW
    torch, where ``str()`` crosses patched ``item``/``tolist``. Inside a live capture TorchLens
    intercepts the string path itself, so the fix lives at the interception, not in a
    ``__repr__``/``__str__`` patch -- the escape still lands UNVERIFIABLE, consistently with how
    every other value-extraction spelling (``.numpy()`` / ``.tolist()``) ceilings a changed run.

    Gated by the runnable-capture escape-observation flag and skipped for TorchLens's own
    marked internal reads (``internal_scalar_read``). A no-string forward records nothing.
    """

    if not getattr(trace, "intervention_ready", False):
        return
    if not isinstance(tensor, torch.Tensor):
        return
    if _internal_read_active():
        return
    # r43 hon2_4: route the string hook through the SAME owner-vs-non-owner rule as every
    # other belt observer. The OWNER thread keeps the precise attribution ladder. A NON-OWNER
    # thread applies the captured-tensor predicate (never origin resolution -- that flips the
    # global ``pause_logging`` toggle, the hon2_4 crash path): a captured-tensor stringification
    # ceilings, a benign OWN-tensor ``str()`` records nothing (no over-trigger).
    state = _ACTIVE_WITNESS_STATE
    if (
        state is not None
        and state.trace is trace
        and threading.get_ident() != state.owner_thread_id
    ):
        if state.belt_armed:
            _nonowner_escape_observe(state, tensor)
        return
    _record_escape_source_tensor(trace, tensor)


def audit_disable_current_modes_sites() -> dict[str, tuple[str, ...]]:
    """Snapshot-audit torch's ``_disable_current_modes`` sites (r39 advisory-but-armed immunizer).

    The mode-independent method/module belt (:data:`HOST_VALUE_ESCAPE_METHODS` /
    :data:`HOST_VALUE_ESCAPE_MODULE_FUNCS`) closes the census blind spot regardless of WHICH
    torch region pops the dispatch modes, so this audit is NOT load-bearing -- it is an armed
    snapshot. It enumerates the ``_disable_current_modes`` sites in the installed torch and
    classifies each by its top-level containing module against
    :data:`_DISABLE_MODE_SITE_CATEGORIES`. A site whose category is unknown is ``unclassified``,
    turning the coverage meta-test RED so a human confirms the new region introduces no
    value-escape spelling the belt misses.

    Returns
    -------
    dict[str, tuple[str, ...]]
        ``{"classified": (...), "unclassified": (...)}`` module paths (sorted).
    """

    classified: set[str] = set()
    unclassified: set[str] = set()
    try:
        torch_root = Path(torch.__file__).resolve().parent
    except Exception:  # pragma: no cover - torch always has a file
        return {"classified": (), "unclassified": ()}
    for path in torch_root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeError):  # pragma: no cover - unreadable file
            continue
        if "_disable_current_modes(" not in text:
            continue
        relative = path.relative_to(torch_root)
        top = relative.parts[0]
        category = top[:-3] if top.endswith(".py") else top
        (classified if category in _DISABLE_MODE_SITE_CATEGORIES else unclassified).add(
            str(relative)
        )
    return {
        "classified": tuple(sorted(classified)),
        "unclassified": tuple(sorted(unclassified)),
    }


def _internal_read_active() -> bool:
    """Return whether an explicit TorchLens internal-scalar-read marker is live.

    The marker is set by construction ONLY around TorchLens's own capture-internal
    scalar/comparison reads (see :func:`internal_scalar_read`). It is a per-thread depth
    counter so nested internal reads compose correctly.
    """

    return getattr(_internal_read_state, "depth", 0) > 0
