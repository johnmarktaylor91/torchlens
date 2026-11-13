"""Resident-payload probes and bounded value facts for record displays."""

from __future__ import annotations

from typing import Any

from .._comparison_tensor import _comparison_tensor
from .._io.payload_codec import get_payload_codec
from .._utils._display_card import REPR_COLS
from .._utils._display_numbers import format_count, format_exact_count, plural
from .._utils._display_text import tensor_text
from .._utils._record_group_table import record_member
from .._utils.fail_open import fail_open

NOT_SAVED = "(not saved)"
FREED = "(freed)"
METADATA_ONLY = "(metadata only)"
LAZY_DISK = "(disk-backed: metadata only until read)"
EDITED = "(edited)"


def _count_words(count: int, noun: str, *, compact: bool = False) -> str:
    """Pair a correctly inflected noun with an exact or compact display count."""

    number = format_count(count) if compact else format_exact_count(count)
    return f"{number} {plural(count, noun).partition(' ')[2]}"


def _payload_state(op: Any) -> str:
    """Return resident-now state without reading a materializing payload property."""

    slot = getattr(op, "_slot", None)
    if callable(slot):
        if slot("out") is not None:
            return "present"
        if slot("out_ref") is not None:
            return "lazy"
        return "unsaved"
    if record_member(op, "out_ref", None) is not None:
        return "lazy"
    return "present" if getattr(op, "has_saved_activation", False) else "unsaved"


def _device(record: Any) -> Any:
    """Read captured device metadata without touching a tensor payload."""

    device = getattr(record, "device", None)
    if device is None:
        device = getattr(record, "device_ref", None)
    return device


def metadata_text(record: Any, tensor: Any = None) -> str:
    """Full dtype, shape and captured device, or resident tensor metadata."""

    source = tensor if tensor is not None else record
    return tensor_text(
        getattr(source, "dtype", None), getattr(source, "shape", None), _device(source)
    )


def _statistics(tensor: Any, *, compute: bool, identity: str | None) -> Any:
    """Deferred door into the statistics layer; display failures remain cosmetic."""

    from ..stats._record_values import record_stats

    return fail_open(
        lambda: record_stats(tensor, compute=compute, identity=identity), lambda _error: None
    )


def value_words(stats: Any, *, compact: bool = False, moments: bool = True) -> str:
    """Pure record summary over existing evidence, never the glance sparkline."""

    from ..stats._record_values import _summary, render_values

    return (
        _summary(stats, separator=" ", moments=moments)
        if compact
        else render_values(stats, moments=moments)
    )


def _health(stats: Any) -> str:
    """Evidence that cannot be dropped for width."""

    from ..stats._record_values import value_health

    health = value_health(stats)
    sampled = "(~sampled)" if stats is not None and stats.policy == "sampled" else ""
    return " ".join(part for part in (health, sampled) if part)


def payload_facts(record: Any) -> tuple[str, Any, str]:
    """Return metadata, statistics and honesty for one resident output."""

    import torch

    state = _payload_state(record)
    if state == "lazy":
        return metadata_text(record), None, LAZY_DISK
    if state == "unsaved":
        note = (
            " (saved at capture; no bytes here)"
            if getattr(record, "has_saved_activation", False)
            else ""
        )
        return metadata_text(record), None, NOT_SAVED + note
    slot = getattr(record, "_slot", None)
    payload = slot("out") if callable(slot) else fail_open(lambda: record.out, lambda _error: None)
    if payload is None:
        return metadata_text(record), None, FREED
    identity = record_member(record, "site_key", None)
    if not isinstance(payload, torch.Tensor):
        host = _host_tensor(record, payload)
        stats = None if host is None else _statistics(host, compute=True, identity=identity)
        return metadata_text(record), stats, ""
    stats = _statistics(payload, compute=True, identity=identity)
    return metadata_text(record, payload), stats, ""


def _host_tensor(record: Any, payload: Any) -> Any:
    """Convert a non-torch payload to a host torch tensor through its backend codec.

    The same statistics torch shows are then computed on an exact host copy
    (NumPy conversion through the producing backend's payload codec). Returns
    ``None`` when no codec recognizes the payload.
    """

    codec = get_payload_codec(_record_backend(record) or "")
    if not fail_open(lambda: bool(codec.can_encode(payload)), lambda _error: False):
        return None
    return _comparison_tensor(payload, codec)


def _record_backend(record: Any) -> str | None:
    """Return the producing backend of a record's live trace, when it is reachable."""

    slot = getattr(record, "_slot", None)
    ref = slot("_source_trace_ref") if callable(slot) else None
    trace = ref() if callable(ref) else None
    return None if trace is None else str(trace._provenance_backend)


def edited_token(record: Any) -> str:
    """Expose user interventions without presenting them as observed values."""

    return (
        EDITED
        if getattr(record, "is_edited", False) or record_member(record, "edge_substitutions", None)
        else ""
    )


def value_line(
    head: str,
    metadata: str,
    stats: Any,
    honesty: str = "",
    *,
    compact: bool = True,
) -> str:
    """Keep identity and honesty, then degrade moments before extrema at 100 columns."""

    health = _health(stats)
    required = " ".join(part for part in (metadata, honesty, health) if part)
    base = f"{head}: {required}"
    if stats is None:
        suffix = ", values not computed" if not honesty else ""
        return base + suffix if len(base + suffix) <= REPR_COLS else base
    for summary in (
        value_words(stats, compact=compact),
        value_words(stats, compact=compact, moments=False),
        "values in print()",
    ):
        if not summary:
            return base
        line = f"{base}, {summary}"
        if len(line) <= REPR_COLS:
            return line
    return base


def payload_core(record: Any, *, identity: str | None = None, max_width: int | None = None) -> str:
    """Compatibility door for record-only payload text; no glance grammar."""

    metadata, stats, honesty = payload_facts(record)
    return " ".join(
        part for part in (metadata, honesty, _health(stats), value_words(stats)) if part
    )


def trace_of(record: Any) -> Any:
    """Owning trace when reachable, including detached module records."""

    trace = getattr(record, "_source_trace", None)
    return (
        trace
        if trace is not None
        else fail_open(
            lambda: getattr(record, "trace", None) or getattr(record, "source_trace", None),
            lambda _error: None,
        )
    )


def resolve_record(record: Any, label: str) -> Any:
    """Resolve one stored graph key without iterating or scanning a trace."""

    trace = trace_of(record)
    return fail_open(lambda: trace[label], lambda _error: None) if trace is not None else None
