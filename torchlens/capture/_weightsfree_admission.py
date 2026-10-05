"""Scoped meta admission for weights-free capture (W2, weightsfree memo D2).

Meta state/inputs are admitted IF AND ONLY IF structure-only is in force:
``admit_meta`` derives once from resolved ``CaptureOptions.structure_only``
and threads to all SEVEN variant-gate call sites; rerun, backward, and
fastlog thread a permanently closed regime. An unscoped carve-out silently
captures an unbannered value-free trace with full save/replay reach
(measured) — worse than the refusal it replaces.

This module owns the admission record, the weak-keyed session registries
(the ledger pattern of ``completeness_witness``; a Trace object is never
mutated with unregistered attributes), the D20 meta identity self-test, and
the D22 settlement invariants. The admission decision itself runs inside
:func:`torchlens._robustness.check_model_and_input_variants` so entry-order
guarantees (wrapper -> lazy -> distributed -> variants) are unchanged.

Every spelling is DOCUMENTED-UNSTABLE pending naming-session/S2
ratification; every refusal code here is S2-gated.
"""

from __future__ import annotations

import contextlib
import threading
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final

import torch

__all__ = [
    "MetaAdmissionRecord",
    "admission_record_for",
    "enforce_settlement_invariants",
    "register_admission",
    "self_test_meta_identity",
    "stamp_wrap_generation",
    "wrap_generation_of",
]


@dataclass(frozen=True)
class MetaAdmissionRecord:
    """Frozen facts of one admitted weights-free (meta) capture entry.

    Consumed at settlement: an admitted meta offense whose finished trace
    lacks the structure-only marker fails closed (memo sec 4.2), and the
    evidence envelope (memo sec 5) is built from these fields.
    """

    substrate: str  # "meta"
    factory_device_policy: str  # "torchlens_owned"
    ambient_mode_present: bool
    wrap_generation: int
    meta_input_paths: tuple[str, ...]
    meta_state_names: tuple[str, ...]


# Weak-keyed session registries (never persisted; a loaded trace is not
# admitted — its envelope carries the persisted disclosure instead).
_ADMISSIONS: weakref.WeakKeyDictionary[Any, MetaAdmissionRecord] = weakref.WeakKeyDictionary()
_WRAP_GENERATIONS: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()


def register_admission(trace: Any, record: MetaAdmissionRecord) -> None:
    """Bind one admission record to the capture's Trace for this session."""

    _ADMISSIONS[trace] = record


def admission_record_for(trace: Any) -> MetaAdmissionRecord | None:
    """The admission record of a live admitted capture, or ``None``."""

    return _ADMISSIONS.get(trace)


def stamp_wrap_generation(trace: Any, generation: int) -> None:
    """Stamp the capture-time wrap generation (W1-ORD) on a live trace."""

    _WRAP_GENERATIONS[trace] = generation


def wrap_generation_of(trace: Any) -> int | None:
    """The session wrap-generation stamp of a live trace, or ``None``.

    ``None`` on loaded traces: generations are process-scoped counters and
    are never comparable across processes (the persisted envelope carries the
    structure side's generation as DISCLOSURE, not as a joinable key).
    """

    return _WRAP_GENERATIONS.get(trace)


# ---------------------------------------------------------------------------
# D20: meta-safe identity, feature-detected or refuse
# ---------------------------------------------------------------------------

_IDENTITY_SELF_TEST_RESULT: str | None = None


def self_test_meta_identity() -> None:
    """Verify a trustworthy storage-identity primitive exists on meta (D20).

    No meta path may key identity on ``data_ptr() == 0``. Storage ``_cdata``
    is measured to distinguish sibling meta allocations and preserve view
    sharing, but the primitive is PRIVATE, so production feature-detects and
    self-tests it once per process; if no trustworthy primitive exists for
    the running torch, meta admission refuses typed rather than guessing.

    Raises
    ------
    torchlens._errors.WeightsfreeIntegrityError
        With code ``structure_only_meta_identity_unavailable`` when the
        running torch offers no primitive that distinguishes sibling meta
        allocations while preserving view identity.
    """

    global _IDENTITY_SELF_TEST_RESULT
    if _IDENTITY_SELF_TEST_RESULT == "ok":
        return
    if _IDENTITY_SELF_TEST_RESULT is None:
        _IDENTITY_SELF_TEST_RESULT = _run_identity_self_test()
    if _IDENTITY_SELF_TEST_RESULT == "ok":
        return
    from .._errors import WeightsfreeIntegrityError

    raise WeightsfreeIntegrityError(
        "Weights-free admission requires a storage-identity primitive that "
        "distinguishes sibling meta allocations (every meta tensor reports "
        "data_ptr()==0, so pointer identity would alias-collapse unrelated "
        f"tensors); the self-test on this torch build reported: "
        f"{_IDENTITY_SELF_TEST_RESULT}. TorchLens refuses admission rather "
        "than guessing identity. Remedy: capture on a real device "
        "(materialized weights), or use a torch build whose meta storages "
        "expose distinguishable identities.",
        code="structure_only_meta_identity_unavailable",
        self_test=_IDENTITY_SELF_TEST_RESULT,
    )


def _run_identity_self_test() -> str:
    """One-per-process probe: can meta storages be told apart and aliased?"""

    try:
        first = torch.empty(2, device="meta")
        second = torch.empty(2, device="meta")
        view = first.view(1, 2)
        first_key = meta_storage_key(first)
        second_key = meta_storage_key(second)
        view_key = meta_storage_key(view)
    except Exception as exc:  # noqa: BLE001 — probe failure = refuse typed
        return f"identity probe raised {type(exc).__name__}: {exc}"
    if first_key is None:
        return "untyped_storage()._cdata is unavailable on this torch build"
    if first_key == second_key:
        return "sibling meta allocations report identical storage identity"
    if view_key != first_key:
        return "a view does not share its base's storage identity"
    return "ok"


def meta_storage_key(tensor: torch.Tensor) -> int | None:
    """A storage-identity key valid on the meta substrate, or ``None``.

    ``untyped_storage()._cdata`` distinguishes sibling meta allocations and
    is shared by views of one allocation — exactly the identity the aliasing
    and buffer-tracking paths need where ``data_ptr()`` reads 0 for every
    meta tensor.
    """

    try:
        return int(tensor.untyped_storage()._cdata)
    except Exception:  # noqa: BLE001 — feature detection, never authority
        return None


# ---------------------------------------------------------------------------
# D22: settlement invariants (the net, not the fix)
# ---------------------------------------------------------------------------

_SETTLEMENT_REMEDY: Final[str] = (
    "this is a TorchLens capture bug, not a user error: re-run with a real "
    "capture and report the invariant name upstream"
)


def enforce_settlement_invariants(trace: Any) -> None:
    """Enforce the D22 settlement invariants on a finished admitted capture.

    Before a weights-free capture settles COMPLETE: declared parameter
    counts reconcile across the live totals and module records; no value
    payload is populated; the structure-only marker survived; and no
    false host-escape/witness state arose from unavailable meta values.
    Motivated by measured self-contradictions (``trace.num_params == 0``
    beside ``modules['self'].num_params == 30``); funded as the NET that
    catches defect mechanism number six.

    Raises
    ------
    torchlens._errors.WeightsfreeIntegrityError
        With code ``structure_only_settlement_incoherent`` naming the first
        violated invariant. A tripwire, never an exemption surface.
    """

    record = admission_record_for(trace)
    if record is None:
        return
    violation = _first_settlement_violation(trace)
    if violation is None:
        return
    from .._errors import WeightsfreeIntegrityError

    raise WeightsfreeIntegrityError(
        f"Weights-free settlement invariant violated: {violation}. An "
        "admitted meta capture may not settle with incoherent structural "
        f"accounting. Remedy: {_SETTLEMENT_REMEDY}.",
        code="structure_only_settlement_incoherent",
        invariant=violation,
    )


def _first_settlement_violation(trace: Any) -> str | None:
    """The first violated D22 invariant, or ``None`` when coherent."""

    if not bool(getattr(trace, "structure_only", False)):
        return "admitted meta capture lost its structure_only marker (fails closed)"
    if bool(getattr(trace, "capture_verified", False)):
        return "capture_verified is True on a value-free capture"
    for layer in getattr(trace, "layer_list", ()):
        if getattr(layer, "out", None) is not None:
            return f"value payload populated at {getattr(layer, 'label', '?')!r}"
    total = getattr(trace, "num_params", None)
    module_totals = _module_param_total(trace)
    if total is not None and module_totals is not None and total != module_totals:
        return (
            f"declared parameter totals disagree: trace.num_params={total} "
            f"vs root module accounting {module_totals}"
        )
    from ..backends.torch.completeness_witness import _HOST_ESCAPE_MUTABLE_WRITEBACK

    if trace in _HOST_ESCAPE_MUTABLE_WRITEBACK:
        return (
            "an opaque host-write witness flag was raised by the substrate "
            "(no value comparison can observe a write weights-free)"
        )
    return None


def _module_param_total(trace: Any) -> int | None:
    """Root-module declared parameter total, or ``None`` when unavailable."""

    modules = getattr(trace, "modules", None)
    if not modules:
        return None
    try:
        root = modules["self"] if "self" in modules else next(iter(modules.values()))
    except Exception:  # noqa: BLE001 — accounting is diagnostic input only
        return None
    total = getattr(root, "num_params", None)
    return int(total) if total is not None else None


# ---------------------------------------------------------------------------
# The admitted forward scope (armed around the ONE captured user forward)
# ---------------------------------------------------------------------------


class _PendingAdmission(threading.local):
    """Thread-local hand-off slot from the entry gate to the forward scope."""

    def __init__(self) -> None:
        self.record: MetaAdmissionRecord | None = None


# Module-level mutable state: thread-local hand-off, armed by the capture
# entry (user_funcs) around the one capture driver call and read by the
# forward scope; inventoried by tests/test_global_state_inventory.py.
_PENDING = _PendingAdmission()

# Live admitted-meta captures (consumed by the W1 transparency gate on the
# wrapper hot path; a WeakSet so a dropped trace disarms automatically).
_META_ACTIVE: weakref.WeakSet[Any] = weakref.WeakSet()


@contextlib.contextmanager
def pending_admission(record: MetaAdmissionRecord | None) -> Iterator[None]:
    """Arm the entry-gate admission record for the capture driver call.

    ``None`` is a legal no-op arm (ordinary and real-substrate captures).
    The slot is cleared on exit even on ``BaseException`` so a failed capture
    never leaks admission into a later one. The rescue re-run inside the
    scope reads the SAME record: admission facts are entry facts.
    """

    previous = _PENDING.record
    _PENDING.record = record
    try:
        yield
    finally:
        _PENDING.record = previous


def weightsfree_meta_active(trace: Any) -> bool:
    """Whether ``trace`` is a LIVE admitted-meta capture (W1 gate check)."""

    return trace in _META_ACTIVE


@contextlib.contextmanager
def weightsfree_forward_scope(trace: Any) -> Iterator[None]:
    """Arm the admitted-meta machinery around ONE captured user forward.

    No-op unless the entry gate admitted a meta substrate for this capture
    (the pending slot holds the record). When armed: binds the admission
    record to the trace, activates the W1 transparency gate, arms the owned
    factory-device slot (W1-CTX), maps the exact null-autocast call to
    ``nullcontext`` (W1-AC), and absorbs any caller-active ``DeviceContext``
    for the duration of the forward (D19). Everything restores on
    ``BaseException``.

    Before the slot arms, torch's lazy ``torch._dynamo`` import is forced
    (logging paused, no device context active). Meta kernels for ops such as
    ``aten.addmm`` are Python decompositions wrapped in
    ``torch._compile._disable_dynamo``, which imports ``torch._dynamo`` on
    first call. Left to fire inside the forward, that import's module bodies
    (``populate_builtin_to_tensor_fn_map``: ``torch.ones(1)`` then unary ops)
    got the slot's meta device, so their ops re-entered a meta decomposition
    and reached ``_disable_dynamo`` again on the half-initialised module
    (``AttributeError: partially initialized module 'torch._dynamo'``, torch
    2.7). The slot owns the USER forward's factories, never torch's own
    imports, and an admitted meta forward imports ``torch._dynamo`` anyway,
    so the warm moves an import that was coming rather than adding one.
    """

    record = _PENDING.record
    if record is None:
        yield
        return
    import torch as _torch

    from ..backends.torch._weightsfree_ctx import factory_device_scope
    from ..utils._torch_compat import warm_lazy_torch_imports

    register_admission(trace, record)
    _META_ACTIVE.add(trace)
    try:
        with _absorbed_ambient_device_context():
            warm_lazy_torch_imports()
            with factory_device_scope(_torch.device("meta")), _null_autocast_scope():
                yield
    finally:
        _META_ACTIVE.discard(trace)


@contextlib.contextmanager
def _null_autocast_scope() -> Iterator[None]:
    """W1-AC: map EXACTLY (meta device AND enabled=False) to ``nullcontext``.

    Semantically null: disabled autocast performs no dtype coercion and no
    numerics exist on meta — yet torch validates the device string even for
    ``enabled=False`` (measured on torch 2.13:
    ``RuntimeError: unsupported scalarType``), which blocks the flagship
    llama family on the declared transformers 4.x band (defect L8).
    ``enabled=True`` on meta still delegates to torch and raises. Scoped to
    the admitted forward only; the shipped ``torch.autocast`` object is
    restored on exit. The replacement is a SUBCLASS of the shipped class
    (``isinstance``/``issubclass``/``inspect.isclass``/decorator use hold
    inside the scope). Disclosed table condition: code introspecting
    ``torch.is_autocast_enabled()`` inside the shimmed scope sees outer
    state (no measured model does this).
    """

    import torch as _torch

    original = _torch.autocast
    shim = _null_on_meta_autocast_class(original)
    setattr(_torch, "autocast", shim)  # noqa: B010 — scoped shim swap (a CLASS, see below)
    try:
        yield
    finally:
        setattr(_torch, "autocast", original)  # noqa: B010 — restore the shipped class


def _null_on_meta_autocast_class(original: type[Any]) -> type[Any]:
    """Build the scoped ``torch.autocast`` replacement as a SUBCLASS.

    The shim must stay a class deriving from the shipped ``torch.autocast``:
    user code that introspects the scope (``isinstance(ctx, torch.autocast)``,
    ``issubclass``, ``inspect.isclass(torch.autocast)``, decorator use
    ``@torch.autocast(...)``) must see the same kind of object it does outside
    the admitted forward. The historical FUNCTION swap broke every such check
    inside the scope (AUD-CODE 4.3). Exactly the null call -- ``meta`` device
    AND ``enabled=False`` -- skips the base initializer (which validates the
    device string and raises on meta) and enters/exits as a no-op; every
    other call constructs and behaves as the shipped class.
    """

    class _NullOnMetaAutocast(original):
        """``torch.autocast`` whose (meta, enabled=False) call is a null scope."""

        _tl_null: bool

        def __init__(self, device_type: Any, *args: Any, **kwargs: Any) -> None:
            enabled = kwargs.get("enabled", True)
            if len(args) >= 2:
                enabled = args[1]
            if str(device_type) == "meta" and enabled is False:
                self._tl_null = True
                return
            self._tl_null = False
            super().__init__(device_type, *args, **kwargs)

        def __enter__(self) -> Any:
            if self._tl_null:
                return self
            return super().__enter__()

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
            if self._tl_null:
                return None
            return super().__exit__(exc_type, exc_val, exc_tb)

    _NullOnMetaAutocast.__name__ = original.__name__
    _NullOnMetaAutocast.__qualname__ = original.__qualname__
    _NullOnMetaAutocast.__module__ = original.__module__
    return _NullOnMetaAutocast


@contextlib.contextmanager
def _absorbed_ambient_device_context() -> Iterator[None]:
    """D19: no torch ``DeviceContext`` mode survives into an admitted forward.

    Where the mode-stack surgery primitives exist (feature-detected
    ``HAS_TORCH_FUNCTION_STACK_SURGERY``), every caller-active
    ``DeviceContext`` is popped for the duration of the forward and the
    exact original stack interleaving is restored on exit — its catch-all
    ``__torch_function__``
    re-entry respells every dunder op (defect L3) while TorchLens's own
    injection already covers factory placement. Where the primitives are
    missing, a caller-active context refuses typed with the exit-the-context
    remedy (SOL's fallback rule).
    """

    from ..backends.torch.wrappers import _get_active_device
    from ..utils._torch_compat import (
        get_device_context_type,
        get_torch_function_stack_surgery,
    )

    if _get_active_device() is None:
        yield
        return
    surgery = get_torch_function_stack_surgery()
    if surgery is None:
        from .._errors import WeightsfreeIntegrityError

        raise WeightsfreeIntegrityError(
            "A torch device context (with torch.device(...)) is active at an "
            "admitted weights-free capture, and this torch build offers no "
            "mode-stack primitives to absorb it. Its catch-all torch-function "
            "re-entry would respell every dunder op and corrupt the public "
            "digest. Remedy: exit the construction context before capture "
            "(build the model inside `with torch.device('meta')`, capture "
            "OUTSIDE it).",
            code="structure_only_ambient_device_context",
        )
    len_stack, pop_stack, push_stack = surgery
    # A missing DeviceContext class is fail-neutral here: nothing on the
    # stack matches, so every mode is kept in place (the compat accessor
    # discloses the degradation through HAS_DEVICE_CONTEXT_DISPATCH).
    device_context_type: type[Any] = get_device_context_type() or type(None)
    # Snapshot the ENTIRE original stack, bottom to top, then re-push only the
    # non-device modes in their original relative order for the forward.
    original_top_down: list[Any] = []
    while len_stack() > 0:
        original_top_down.append(pop_stack())
    original_bottom_up = list(reversed(original_top_down))
    kept = [mode for mode in original_bottom_up if not isinstance(mode, device_context_type)]
    for mode in kept:
        push_stack(mode)
    try:
        yield
    finally:
        # Restore the EXACT original interleaving (AUD-CODE 4.3: the historical
        # rebuild pushed every popped DeviceContext on TOP of the kept modes,
        # losing the caller's ordering). Whatever the forward left on the
        # stack above the kept modes (an unbalanced user push) is preserved
        # on top of the restored original stack rather than dropped.
        current_top_down: list[Any] = []
        while len_stack() > 0:
            current_top_down.append(pop_stack())
        current_bottom_up = list(reversed(current_top_down))
        kept_ids = {id(mode) for mode in kept}
        leftover = [mode for mode in current_bottom_up if id(mode) not in kept_ids]
        for mode in original_bottom_up:
            push_stack(mode)
        for mode in leftover:
            push_stack(mode)
