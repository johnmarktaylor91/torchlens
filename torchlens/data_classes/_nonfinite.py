"""Shared non-finite activation scan behind a revalidated per-log memo.

``print(trace)`` (``interface._str_after_pass``), ``Trace._repr_html_``, and
``report.explain`` all answer "is any saved activation non-finite?", and each
answer used to cost a full ``torch.isfinite`` pass over every saved activation
in the capture -- the whole forward's payload, re-read from scratch on every
repr (67M elements for a resnet18 at batch 8), with four separate copies of the
same loop. The scan itself is unchanged here: same sequence, same order, same
skip rules, same ``detach()``/``isfinite`` kernel, same propagated exceptions.
It is only memoized per log, and the memo is revalidated against the object
identity plus autograd version counter of every tensor the recorded scan
examined, so a mutated, replaced, added, or removed activation falls back to a
real scan rather than serving a stale verdict.

Revalidation deliberately re-reads the same ``out`` attributes in the same
order as the scan it replaces, so lazy materialization, reference-mutation
tripwires, and ``ValueError`` on unsaved payloads all still fire at the same
layer they always did.

The memo is exactly as sharp as ``Tensor._version``, which is already the
mutation oracle TorchLens itself trusts (``MutatedReferenceError`` is raised off
the same counter). A write that deliberately bypasses the autograd version
counter -- ``layer.out.data[0] = float("nan")``, a write through a retained
``numpy()``/``untyped_storage()`` view -- therefore does not invalidate a memo
recorded before it, and a repeated question can still report the pre-write
verdict. This is the same host-write-through-a-detached-handle class the
runnable contract documents as out of scope (``docs/reference/
runnable_tlspec_contract.md`` section 11); an ordinary in-place op, a payload
replacement, and a first question asked after the write are all seen normally.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from .._state import pause_logging
from ..utils._torch_compat import get_fp8_dtypes
from ..utils.tensor_utils import fp8_widen_for_numeric_ops


class _ScanMemo(NamedTuple):
    """One recorded scan: what it examined, what it found, how far it got."""

    keys: tuple[tuple[weakref.ref, object], ...]
    hits: tuple[int, ...]
    complete: bool
    unchecked: tuple[int, ...] = ()
    inference: tuple[int, ...] = ()


# Version stand-in for inference-mode tensors, which torch keeps NO version
# counter for (``tensor._version`` raises RuntimeError on them). A stable
# sentinel keeps memo keys comparable across scan and revalidation.
_INFERENCE_VERSION = "inference"


def _tensor_version(out: torch.Tensor) -> object:
    """Return a tensor's mutation-oracle key, or the inference sentinel.

    ``torch.inference_mode()`` tensors track no version counter at all --
    reading ``_version`` on one raises ``RuntimeError`` ("Inference tensors do
    not track version counter."), which used to escape ``getattr``'s
    AttributeError-only default and take ``print(trace)`` / ``_repr_html_`` /
    ``report.explain`` down on any capture run under inference mode. Guard on
    ``torch.is_inference`` first and catch the RuntimeError defensively for
    exotic subclasses whose ``is_inference`` probe itself misbehaves.
    """

    try:
        if torch.is_inference(out):
            return _INFERENCE_VERSION
        return getattr(out, "_version", None)
    except RuntimeError:
        return _INFERENCE_VERSION


# Keyed by log object so a memo never keeps a Trace alive, and holding only
# weakrefs to the examined tensors so it never keeps an activation alive either.
_MEMOS: weakref.WeakKeyDictionary[Any, dict[str, _ScanMemo]] = weakref.WeakKeyDictionary()


def _trace_out(layer: Any) -> Any:
    """Read a layer's out payload directly, letting unsaved reads raise."""

    return getattr(layer, "out", None)


def _saved_out(layer: Any) -> Any:
    """Return a layer's saved output payload, or ``None`` when unavailable.

    The report anomaly scans only reason about *saved* activation payloads. On a
    selective-save trace most layers retain no payload, and reading ``.out`` on
    such an op raises ``ValueError`` (``"... was not saved; no saved payload is
    available"``) -- a per-pass ``ValueError`` that a plain ``getattr(layer,
    "out", None)`` cannot swallow, so both the JSON and prose reports previously
    crashed on the ordinary predicate-save trace shape instead of honoring their
    documented ``unknown``/scoped-clean contract. Gate on the saved-payload flag
    first, then read the property inside the known-unavailable boundary so an
    unsaved op is honestly skipped rather than aborting the whole report.

    Parameters
    ----------
    layer:
        A per-pass operation/layer entry from ``log.layer_list``.

    Returns
    -------
    Any
        The saved output tensor when a payload was retained, else ``None``.
    """

    if not bool(getattr(layer, "has_saved_activation", False)):
        return None
    if getattr(layer, "out_ref", None) is not None:
        slot = getattr(layer, "_slot", None)
        if callable(slot):
            try:
                resident = slot("out")
            except (AttributeError, KeyError, TypeError):
                resident = None
            if resident is None:
                # Repr/report surfaces are metadata queries. A disk-backed value
                # remains explicitly unexamined until the user requests it.
                return None
            return resident
    try:
        return getattr(layer, "out", None)
    except ValueError:
        return None


def _layer_list(log: Any) -> Any:
    """Return the finalized layer sequence of a log."""

    return getattr(log, "layer_list", []) or []


def _raw_layers(log: Any) -> Any:
    """Return the raw pre-postprocessing layer sequence of a partial capture."""

    return getattr(log, "raw_layers", ()) or ()


# Scan kind -> (sequence getter, out-payload gate). ``"trace"`` is the
# ``Trace.first_nonfinite`` contract (unsaved reads raise), ``"saved"`` the
# ``report.explain`` contract (unsaved layers are skipped), ``"raw"`` the
# partial-capture contract over raw layer records.
_KINDS: dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    "trace": (_layer_list, _trace_out),
    "saved": (_layer_list, _saved_out),
    "raw": (_raw_layers, _trace_out),
}


def _examined(log: Any, kind: str) -> Iterator[tuple[Any, torch.Tensor]]:
    """Yield each (layer, tensor) pair a scan of this kind would inspect."""

    sequence, gate = _KINDS[kind]
    for layer in sequence(log):
        out = gate(layer)
        if not isinstance(out, torch.Tensor) or out.numel() == 0:
            continue
        yield layer, out


def _has_nonfinite(out: torch.Tensor) -> bool | None:
    """Return whether a tensor holds any NaN or Inf, ``None`` if unrunnable.

    ``None`` means torch ships no ``isfinite`` kernel for this payload's dtype, so
    the scan has no evidence either way. It is deliberately NOT ``False``: this
    function used to swallow the ``NotImplementedError`` (a ``RuntimeError``
    subclass) and answer "finite", which made a whole-capture CLEAN verdict out of a
    payload nobody looked at. An all-NaN ``float8_e4m3fn`` activation read as clean
    that way, and ``float8_e8m0fnu`` is worse still -- torch's own ``isfinite``
    returns ``True`` for its NaN pattern, so the native kernel answers wrongly
    rather than refusing.

    fp8 is therefore widened to float32 first. The widening is exact (all 256 bit
    patterns of every variant round-trip bit-identically; see
    ``fp8_widen_for_numeric_ops``), so the verdict is the one a real fp8 kernel
    would give. Dtypes with no runnable check even after widening -- quantized and
    sparse payloads -- return ``None`` and are disclosed by
    :func:`uncheckable_payload_count`.

    Parameters
    ----------
    out:
        Saved activation payload to test.

    Returns
    -------
    bool | None
        True/False when the check ran, ``None`` when no kernel exists for it.
    """

    tensor = out.detach()
    if tensor.dtype in get_fp8_dtypes():
        # ``.to()`` is a decorated method; never let the widening log itself.
        with pause_logging():
            tensor = fp8_widen_for_numeric_ops(tensor)
    try:
        # One-pass form: ``isfinite().all()`` allocates ONE bool intermediate and
        # reduces it, where the historical ``(~isfinite()).any()`` allocated a
        # second full-size negation first. Equivalent by De Morgan
        # (``any(~x) == not all(x)``), pinned by the equivalence test in
        # tests/test_report_honesty_wave0.py.
        return not bool(torch.isfinite(tensor).all().item())
    except (RuntimeError, TypeError):
        return None


def _ref(tensor: torch.Tensor) -> weakref.ref | None:
    """Return a weak reference to a tensor, or ``None`` if it forbids one."""

    try:
        return weakref.ref(tensor)
    except TypeError:
        return None


def _nonfinite_verdict_tensor(out: torch.Tensor) -> torch.Tensor | None:
    """Return the DEVICE-SIDE 0-d finiteness verdict, ``None`` if unrunnable.

    The device-side half of :func:`_has_nonfinite` (same fp8 widening, same
    unrunnable contract) WITHOUT the host read, so a full scan can batch
    every verdict into one transfer.
    """

    tensor = out.detach()
    if tensor.dtype in get_fp8_dtypes():
        with pause_logging():
            tensor = fp8_widen_for_numeric_ops(tensor)
    try:
        return torch.isfinite(tensor).all()
    except (RuntimeError, TypeError):
        return None


def _scan(log: Any, kind: str, stop_at_first: bool) -> tuple[list[Any], _ScanMemo]:
    """Run a real scan, returning the examined layers and the memo to record.

    SYNC BATCHING (C02; sumfam item 17): the full scan computes every
    payload's 0-d finiteness verdict device-side, then reads them back in
    ONE host transfer per device -- the historical loop synced once per op
    (~151 syncs on resnet18). The stop-at-first path keeps the sequential
    early exit: there the first sync IS the point. This shares the batched
    single-transfer discipline of the stats kernel
    (``torchlens/stats/_stats_kernel.py``).
    """

    if stop_at_first:
        return _scan_stop_at_first(log, kind)
    return _scan_batched(log, kind)


def _scan_stop_at_first(log: Any, kind: str) -> tuple[list[Any], _ScanMemo]:
    """Sequential early-exit scan: one sync per op, stopping on the first hit."""

    layers: list[Any] = []
    keys: list[tuple[weakref.ref | None, object]] = []
    hits: list[int] = []
    unchecked: list[int] = []
    inference: list[int] = []
    complete = True
    for layer, out in _examined(log, kind):
        layers.append(layer)
        version = _tensor_version(out)
        keys.append((_ref(out), version))
        if version is _INFERENCE_VERSION:
            # No version counter exists to revalidate a memoized verdict
            # against, so no verdict is claimed: coverage is disclosed as
            # unknown (inference tensors) rather than risking a silently
            # stale CLEAN answer -- the disarmed-tripwire class this
            # module exists to prevent.
            inference.append(len(layers) - 1)
            continue
        verdict = _has_nonfinite(out)
        if verdict is None:
            unchecked.append(len(layers) - 1)
            continue
        if verdict:
            hits.append(len(layers) - 1)
            complete = False
            break
    return layers, _ScanMemo(
        tuple(keys),  # type: ignore[arg-type]
        tuple(hits),
        complete,
        tuple(unchecked),
        tuple(inference),
    )


def _scan_batched(log: Any, kind: str) -> tuple[list[Any], _ScanMemo]:
    """Full scan with device-side verdicts read back in one sync per device."""

    layers: list[Any] = []
    keys: list[tuple[weakref.ref | None, object]] = []
    hits: list[int] = []
    unchecked: list[int] = []
    inference: list[int] = []
    pending: dict[str, list[tuple[int, torch.Tensor]]] = {}
    for layer, out in _examined(log, kind):
        layers.append(layer)
        version = _tensor_version(out)
        keys.append((_ref(out), version))
        index = len(layers) - 1
        if version is _INFERENCE_VERSION:
            inference.append(index)
            continue
        verdict_tensor = _nonfinite_verdict_tensor(out)
        if verdict_tensor is None:
            unchecked.append(index)
            continue
        pending.setdefault(str(verdict_tensor.device), []).append((index, verdict_tensor))
    for device_pending in pending.values():
        stacked = torch.stack([verdict for _, verdict in device_pending])
        for (index, _), all_finite in zip(device_pending, stacked.tolist(), strict=True):
            if not all_finite:
                hits.append(index)
    hits.sort()
    return layers, _ScanMemo(
        tuple(keys),  # type: ignore[arg-type]
        tuple(hits),
        True,
        tuple(unchecked),
        tuple(inference),
    )


def _revalidate(log: Any, kind: str, memo: _ScanMemo) -> list[Any] | None:
    """Return the examined layers when every recorded tensor is unchanged.

    Parameters
    ----------
    log:
        Log the memo was recorded against.
    kind:
        Scan kind whose sequence and gate to replay.
    memo:
        Previously recorded scan.

    Returns
    -------
    list[Any] | None
        Currently examined layers, positionally matching ``memo.hits``, or
        ``None`` when anything the recorded scan looked at has changed.
    """

    keys = memo.keys
    layers: list[Any] = []
    for layer, out in _examined(log, kind):
        if len(layers) >= len(keys):
            # A complete scan saw every payload; a new one means new evidence.
            return None
        ref, version = keys[len(layers)]
        if ref() is not out or _tensor_version(out) != version:
            return None
        layers.append(layer)
        if not memo.complete and len(layers) == len(keys):
            # The recorded scan stopped here, so later payloads never mattered.
            return layers
    return layers if len(layers) == len(keys) else None


def _store(log: Any, kind: str, memo: _ScanMemo) -> None:
    """Record a memo for this log, skipping logs or tensors that forbid it."""

    if any(ref is None for ref, _ in memo.keys):
        return
    try:
        memos = _MEMOS.setdefault(log, {})
    except TypeError:
        return
    memos[kind] = memo


def has_scan_evidence(log: Any, kind: str = "saved") -> bool:
    """Whether a nonfinite evidence basis exists WITHOUT running a new scan.

    True when the capture-time record exists (``track_nonfinite=True`` --
    zero read cost) or a prior saved-payload scan left its memo on this
    log. Render surfaces bound by the D4 cost-attribution law (no
    implicit payload scans) branch on THIS before reading health facts;
    the explicit ``tl.report.health_facts(trace)`` door stays the one
    spelling that may pay for a first scan.
    """

    if _capture_store(log) is not None:
        return True
    try:
        memos = _MEMOS.get(log)
    except TypeError:
        memos = None
    return bool(memos and kind in memos)


def _resolve_memo(log: Any, kind: str, stop_at_first: bool) -> tuple[list[Any], _ScanMemo]:
    """Return the examined layers plus the memo, scanning only when needed."""

    try:
        memos = _MEMOS.get(log)
    except TypeError:
        memos = None
    memo = None if memos is None else memos.get(kind)
    if memo is not None and (memo.complete or stop_at_first):
        cached_layers = _revalidate(log, kind, memo)
        if cached_layers is not None:
            return cached_layers, memo
    layers, memo = _scan(log, kind, stop_at_first)
    _store(log, kind, memo)
    return layers, memo


def _resolve(log: Any, kind: str, stop_at_first: bool) -> list[Any]:
    """Return the non-finite layers of a scan, from the memo when it still holds."""

    layers, memo = _resolve_memo(log, kind, stop_at_first)
    return [layers[index] for index in memo.hits]


def first_nonfinite_layer(log: Any, *, kind: str = "trace") -> Any | None:
    """Return the first layer whose out payload holds a NaN or Inf.

    Parameters
    ----------
    log:
        Trace-like object to scan.
    kind:
        Scan contract: ``"trace"``, ``"saved"``, or ``"raw"``.

    Returns
    -------
    Any | None
        First non-finite layer record, or ``None`` when every examined payload
        is finite.
    """

    hits = _resolve(log, kind, stop_at_first=True)
    return hits[0] if hits else None


def unexamined_payload_count(log: Any, *, kind: str = "saved") -> int:
    """Return how many ops a scan of this kind cannot look at.

    A selective-save capture retains payloads for a chosen subset of ops, so a
    non-finite scan genuinely cannot speak for the rest. Callers report this count
    rather than letting a scoped clean answer read as a whole-capture one.

    Parameters
    ----------
    log:
        Trace-like object to inspect.
    kind:
        Scan contract whose sequence and gate to use.

    Returns
    -------
    int
        Number of ops in the scan sequence holding no readable out payload.

    Notes
    -----
    Counts payload availability only -- it never reads tensor values, so it adds no
    scan cost and cannot invalidate the scan memo.
    """

    sequence, gate = _KINDS[kind]
    unexamined = 0
    for layer in sequence(log):
        try:
            out = gate(layer)
        except ValueError:
            unexamined += 1
            continue
        if out is None:
            unexamined += 1
    return unexamined


def _unmaterialized_disk_payload_count(log: Any, *, kind: str) -> int:
    """Return disk-backed payloads deliberately not read by reporting.

    Parameters
    ----------
    log:
        Trace-like object to inspect.
    kind:
        Scan contract whose sequence is counted.

    Returns
    -------
    int
        Number of saved outs represented only by an unmaterialized disk ref.
    """

    sequence, _gate = _KINDS[kind]
    count = 0
    for layer in sequence(log):
        if not bool(getattr(layer, "has_saved_activation", False)):
            continue
        if getattr(layer, "out_ref", None) is None:
            continue
        slot = getattr(layer, "_slot", None)
        if not callable(slot):
            continue
        try:
            if slot("out") is None:
                count += 1
        except (AttributeError, KeyError, TypeError):
            continue
    return count


def uncheckable_payload_count(log: Any, *, kind: str = "saved") -> int:
    """Return how many examined payloads hold a dtype with no finiteness check.

    A retained payload whose dtype torch cannot run ``isfinite`` on (a quantized or
    sparse activation) yields no evidence at all. Counting it here lets a clean
    verdict say so, instead of the scan silently treating "could not look" as
    "looked and it was finite" -- the same disarmed-tripwire shape as the fp8
    ``raise_on_nan`` swallow, one layer up. fp8 payloads are NOT counted: they are
    widened exactly and really are checked (see :func:`_has_nonfinite`).

    Parameters
    ----------
    log:
        Trace-like object to inspect.
    kind:
        Scan contract whose sequence and gate to use.

    Returns
    -------
    int
        Number of examined payloads whose finiteness check could not run.

    Notes
    -----
    Reuses the scan memo, so asking after a clean
    :func:`first_nonfinite_layer` costs a revalidation, not a second pass. When the
    scan stopped early on a hit the count covers only what it examined, which is
    why callers report it on the clean path.
    """

    _, memo = _resolve_memo(log, kind, stop_at_first=False)
    return len(memo.unchecked)


def inference_payload_count(log: Any, *, kind: str = "saved") -> int:
    """Return how many examined payloads are inference-mode tensors.

    ``torch.inference_mode()`` tensors track no version counter, so a memoized
    finiteness verdict on one could go stale with no oracle to catch it. The
    scan therefore claims NO verdict for them, and every clean answer must
    disclose the gap: nonfinite coverage is unknown (inference tensors).

    Parameters
    ----------
    log:
        Trace-like object to inspect.
    kind:
        Scan contract whose sequence and gate to use.

    Returns
    -------
    int
        Number of examined payloads that are inference tensors.
    """

    _, memo = _resolve_memo(log, kind, stop_at_first=False)
    return len(memo.inference)


def coverage_gap_note(log: Any, *, kind: str = "saved") -> str:
    """Return a parenthetical naming what a clean scan could not examine.

    Both reporting surfaces that publish a clean non-finite verdict --
    ``Trace.first_nonfinite`` (and through it ``print(trace)`` / ``_repr_html_``) and
    ``report.explain``'s anomaly bullet -- must disclose the same two coverage gaps,
    so the wording lives here rather than being written twice and drifting.

    Parameters
    ----------
    log:
        Trace-like object to inspect.
    kind:
        Scan contract whose sequence and gate to use.

    Returns
    -------
    str
        Leading-space parenthetical, or ``""`` when the scan examined everything.

    Notes
    -----
    Call this only on the clean path. A scan that found a non-finite payload names
    that payload, which is a complete answer on its own.
    """

    unexamined = unexamined_payload_count(log, kind=kind)
    disk_backed = _unmaterialized_disk_payload_count(log, kind=kind)
    unsaved = max(0, unexamined - disk_backed)
    uncheckable = uncheckable_payload_count(log, kind=kind)
    inference = inference_payload_count(log, kind=kind)
    if not unexamined and not uncheckable and not inference:
        return ""
    if not uncheckable and not disk_backed and not inference:
        # Unchanged wording for the save=-scoped case, which is the common one.
        return (
            f" ({unexamined} op(s) retained no payload and could not be examined; "
            "re-run with a wider save= to cover them)"
        )
    gaps = [f"{uncheckable} op(s) hold a dtype with no runnable finiteness check"]
    if not uncheckable:
        gaps = []
    if inference:
        gaps.append(
            f"nonfinite coverage is unknown (inference tensors) for {inference} "
            "payload(s) captured under torch.inference_mode(), which track no "
            "version counter to revalidate a verdict against"
        )
    if disk_backed:
        gaps.append(f"{disk_backed} disk-backed payload(s) were not materialized by reporting")
    if unsaved:
        gaps.insert(0, f"{unsaved} op(s) retained no payload")
    return f" ({'; '.join(gaps)}, so they could not be examined)"


def nonfinite_layers(log: Any, *, kind: str = "saved") -> list[Any]:
    """Return every layer whose out payload holds a NaN or Inf.

    Parameters
    ----------
    log:
        Trace-like object to scan.
    kind:
        Scan contract: ``"trace"``, ``"saved"``, or ``"raw"``.

    Returns
    -------
    list[Any]
        Non-finite layer records in scan order.
    """

    return _resolve(log, kind, stop_at_first=False)


# ---------------------------------------------------------------------------
# Capture-time per-op recording (``CaptureOptions(track_nonfinite=True)``)
# ---------------------------------------------------------------------------

# Trace-side runtime store, set lazily on the first recorded op (same
# runtime-bookkeeping class as ``_capture_parent_edge_truth``: declared
# ``FieldPolicy.DROP``, never persisted, absent on loaded traces). Keys:
# ``events`` maps raw op label -> bool (True = the op's output held a NaN or
# Inf), ``unchecked`` lists raw labels whose dtype has no runnable finiteness
# kernel, and ``pending`` holds (raw_label, 0-dim device bool flag) pairs whose
# host read is deferred so a CUDA/MPS capture never pays a per-op device
# synchronization -- the flags are drained in ONE batch at the capture
# finalize seam, after the forward has already completed.
_CAPTURE_STORE_ATTR = "_nonfinite_capture"


def record_op_nonfinite(trace: Any, tensor: torch.Tensor, raw_label: str) -> None:
    """Record one op output's finiteness during capture.

    Called from the torch op-finalize hot path only when
    ``trace.track_nonfinite`` is enabled. The check kernel is
    ``torch.isfinite(out).all()`` (one reduction, no inverted full-size
    temporary); fp8 payloads are widened exactly first, matching the
    post-hoc scan's verdict. CPU flags are read immediately (a host read of
    a CPU scalar is free); flags on any other device are deferred as 0-dim
    bool tensors and drained at the capture finalize seam so the forward's
    stream is never synchronized per op.

    Parameters
    ----------
    trace:
        Active ``Trace`` instance.
    tensor:
        Tensor output produced by the just-logged operation.
    raw_label:
        The op's raw (pre-postprocessing) label, the store key.
    """

    store = trace.__dict__.get(_CAPTURE_STORE_ATTR)
    if store is None:
        store = {"events": {}, "unchecked": [], "pending": []}
        trace.__dict__[_CAPTURE_STORE_ATTR] = store
    try:
        # EVERY tensor read here is under pause_logging -- even ``numel()`` is a
        # wrapped call, and running it bare mid-commit on a buffer source
        # re-enters source logging and recurses without bound.
        with pause_logging():
            if tensor.numel() == 0:
                # An empty tensor holds no elements: "no NaN/Inf" is exact.
                store["events"][raw_label] = False
                return
            probe = tensor.detach()
            if probe.dtype in get_fp8_dtypes():
                probe = fp8_widen_for_numeric_ops(probe)
            flag = torch.isfinite(probe).all()
            if flag.device.type == "cpu":
                store["events"][raw_label] = not bool(flag.item())
            else:
                store["pending"].append((raw_label, flag))
    except (RuntimeError, TypeError):
        # No runnable finiteness kernel for this payload (quantized, sparse,
        # exotic layouts). Disclosed via coverage, never silently "finite".
        store["unchecked"].append(raw_label)


def drain_pending_nonfinite(trace: Any) -> None:
    """Read every deferred device flag into the capture event record.

    Runs at the capture finalize seam (the forward is complete, and the
    existing cpu_async D2H fence has already synchronized outstanding copies),
    and again defensively at query time, so a flag can never be read
    mid-forward. A flag whose host read fails is moved to the unchecked
    disclosure rather than poisoning the capture.

    Parameters
    ----------
    trace:
        Trace whose pending capture-time flags should be settled.
    """

    store = trace.__dict__.get(_CAPTURE_STORE_ATTR)
    if not store or not store["pending"]:
        return
    pending, store["pending"] = store["pending"], []
    with pause_logging():
        for raw_label, flag in pending:
            try:
                store["events"][raw_label] = not bool(flag.item())
            except (RuntimeError, TypeError):
                store["unchecked"].append(raw_label)


@dataclass(frozen=True)
class NonfiniteCoverage:
    """Disclosure of what evidence backs :attr:`Trace.nonfinite_ops`.

    A clean (empty) answer is only as strong as its coverage, so the counts
    here must accompany any programmatic read of the record -- the same
    honesty contract the prose surfaces implement via
    :func:`coverage_gap_note`.

    Attributes
    ----------
    basis:
        ``"capture"`` when the record comes from capture-time per-op checks
        (``CaptureOptions(track_nonfinite=True)``; covers every committed op,
        saved or not), or ``"saved_payloads"`` when it is derived post hoc from
        the payloads this capture retained.
    checked:
        Number of op outputs a finiteness kernel actually ran on.
    nonfinite:
        Of those, how many held at least one NaN or Inf.
    unchecked:
        Op outputs whose dtype has no runnable finiteness kernel (quantized,
        sparse); they yield no evidence either way.
    inference:
        ``"saved_payloads"`` basis only: examined payloads that are
        ``torch.inference_mode()`` tensors. They track no version counter, so
        the memoized scan claims no verdict for them -- their nonfinite
        coverage is unknown (inference tensors). Capture-time checks
        (``basis="capture"``) settle verdicts at record time and need no
        revalidation, so inference tensors ARE checked there and this count
        stays 0.
    unexamined:
        Ops the scan could not look at: on the ``"saved_payloads"`` basis,
        ops that retained no payload (or whose disk-backed payload reporting
        deliberately does not materialize); on the ``"capture"`` basis, final
        ops that no capture-time check covered (synthetic input/output mirror
        nodes).
    unmapped:
        ``"capture"`` basis only: recorded events whose op did not survive
        postprocessing (e.g. removed orphans), so they map to no final label.
    """

    basis: str
    checked: int
    nonfinite: int
    unchecked: int
    unexamined: int
    unmapped: int = 0
    inference: int = 0


def _capture_raw_to_final(log: Any) -> dict[str, str]:
    """Map each surviving op's raw label to its pass-qualified final label."""

    mapping: dict[str, str] = {}
    for op in _layer_list(log):
        raw = getattr(op, "_label_raw", None)
        label = getattr(op, "label", None)
        if raw is not None and label is not None:
            mapping[str(raw)] = str(label)
    return mapping


def _capture_store(log: Any) -> dict[str, Any] | None:
    """Return the settled capture-time store, draining any deferred flags."""

    store = getattr(log, "__dict__", {}).get(_CAPTURE_STORE_ATTR)
    if store is None:
        return None
    drain_pending_nonfinite(log)
    return store


def nonfinite_op_labels(log: Any) -> tuple[str, ...]:
    """Return the pass-qualified labels of ops whose output held NaN or Inf.

    Serves the capture-time record when this capture ran with
    ``track_nonfinite=True`` (basis ``"capture"``); otherwise derives the
    answer from the memoized saved-payload scan (basis ``"saved_payloads"``,
    zero capture-time cost). :func:`nonfinite_coverage` names the basis and
    what the answer could not examine -- read it before trusting an empty
    tuple from a capture that retained few payloads.

    Parameters
    ----------
    log:
        Finished trace-like object to query.

    Returns
    -------
    tuple[str, ...]
        Pass-qualified op labels (``Op.label``) in scan order.
    """

    store = _capture_store(log)
    if store is not None:
        raw_to_final = _capture_raw_to_final(log)
        return tuple(
            raw_to_final[raw] for raw, hit in store["events"].items() if hit and raw in raw_to_final
        )
    return tuple(str(getattr(layer, "label", layer)) for layer in _resolve(log, "saved", False))


def nonfinite_coverage(log: Any) -> NonfiniteCoverage:
    """Return the evidence basis and coverage behind :func:`nonfinite_op_labels`.

    Parameters
    ----------
    log:
        Finished trace-like object to query.

    Returns
    -------
    NonfiniteCoverage
        Frozen coverage disclosure; see the class docstring for field meaning.
    """

    store = _capture_store(log)
    if store is not None:
        raw_to_final = _capture_raw_to_final(log)
        events = store["events"]
        covered = {raw for raw in events if raw in raw_to_final}
        covered.update(raw for raw in store["unchecked"] if raw in raw_to_final)
        return NonfiniteCoverage(
            basis="capture",
            checked=len(events),
            nonfinite=sum(1 for hit in events.values() if hit),
            unchecked=len(store["unchecked"]),
            unexamined=max(0, len(raw_to_final) - len(covered)),
            unmapped=sum(1 for raw in events if raw not in raw_to_final)
            + sum(1 for raw in store["unchecked"] if raw not in raw_to_final),
        )
    _, memo = _resolve_memo(log, "saved", stop_at_first=False)
    return NonfiniteCoverage(
        basis="saved_payloads",
        checked=max(0, len(memo.keys) - len(memo.unchecked) - len(memo.inference)),
        nonfinite=len(memo.hits),
        unchecked=len(memo.unchecked),
        unexamined=unexamined_payload_count(log, kind="saved"),
        inference=len(memo.inference),
    )
