"""Pure helpers behind the Trace grad-fn site and fire-timing rollups."""

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from .._quantities import Duration

if TYPE_CHECKING:
    from .trace import Trace


def _grad_fn_site_key(
    grad_fn_record: Any, layer_lookup: Mapping[str, Any]
) -> tuple[str | None, bool]:
    """Return one grad-fn's site key and whether it is op-backed."""

    if not getattr(grad_fn_record, "has_op", False) or grad_fn_record.op_label is None:
        return None, False
    op = layer_lookup.get(grad_fn_record.op_label)
    return getattr(op, "site_key", None), True


def _accumulate_grad_fn_site_record(
    entry: dict[str, Any],
    grad_fn_record: Any,
    fire_timings: Mapping[str, Duration | None] | None,
) -> None:
    """Accumulate one grad-fn record into an existing per-site summary row."""

    entry["grad_fn_labels"].append(grad_fn_record.label)
    for call_index, call in grad_fn_record.calls.items():
        entry["fire_count"] += 1
        if call.backward_pass_number is not None:
            entry["pass_coverage"].add(int(call.backward_pass_number))
        span = (
            fire_timings.get(f"{grad_fn_record.label}:{call_index}")
            if fire_timings is not None
            else None
        )
        if span is not None:
            entry["timed_fire_count"] += 1
            previous = entry["total_fire_duration"]
            entry["total_fire_duration"] = Duration(
                (0.0 if previous is None else float(previous)) + float(span)
            )


def _fire_timings_by_key(
    fired: list[Any], grad_fn_logs: Mapping[Any, Any]
) -> "OrderedDict[str, Duration | None]":
    """Fold fired events into per-fire spans keyed ``"<grad_fn_label>:<call_index>"``."""

    timings: OrderedDict[str, Duration | None] = OrderedDict()
    per_object_ordinals: dict[int, int] = {}
    # Mirrors the _fold_fired_events sort key and per-object ordinal walk
    # so keys line up 1:1 with trace.autograd.grad_fn_calls.
    for event in sorted(fired, key=lambda item: (item.pass_index, item.timestamp, item.seq)):
        record = grad_fn_logs.get(event.object_id)
        if record is None:
            continue
        ordinal = per_object_ordinals.get(event.object_id, 0) + 1
        per_object_ordinals[event.object_id] = ordinal
        started = event.fire_started_monotonic
        finished = event.fire_finished_monotonic
        span = (
            None if started is None or finished is None else Duration(max(0.0, finished - started))
        )
        timings[f"{record.label}:{ordinal}"] = span
    return timings


def _rollup_grad_fn_sites(
    grad_fn_records: Iterable[Any],
    layer_lookup: Mapping[str, Any],
    fire_timings: Mapping[str, Duration | None] | None,
) -> tuple["OrderedDict[str | None, dict[str, Any]]", bool, bool]:
    """Group grad-fn records per site key; also report whether any were op-backed or keyed."""

    summary: OrderedDict[str | None, dict[str, Any]] = OrderedDict()
    any_op_backed = False
    any_keyed = False
    for grad_fn_record in grad_fn_records:
        site_key, op_backed = _grad_fn_site_key(grad_fn_record, layer_lookup)
        any_op_backed = any_op_backed or op_backed
        any_keyed = any_keyed or site_key is not None
        entry = summary.setdefault(
            site_key,
            {
                "grad_fn_labels": [],
                "fire_count": 0,
                "pass_coverage": set(),
                "timed_fire_count": 0,
                "total_fire_duration": None,
            },
        )
        _accumulate_grad_fn_site_record(entry, grad_fn_record, fire_timings)
    return summary, any_op_backed, any_keyed


def grad_fn_site_summary(trace: "Trace") -> "OrderedDict[str | None, dict[str, Any]]":
    """Body of ``Trace.grad_fn_site_summary``; the record keeps the thin spelling and docstring."""

    trace._sync_backward_projection_if_needed()
    try:
        fire_timings: OrderedDict[str, Duration | None] | None = trace.autograd.grad_fn_fire_timings
    except InvalidArgumentError:
        # Loaded/cleaned traces carry no runtime timing evidence; the
        # count/coverage rollup still stands on the persisted records.
        fire_timings = None
    layer_lookup = {} if trace.__dict__.get("_tl_cleaned_up", False) else trace._layer_dict_all_keys
    grad_fn_records = {} if trace.__dict__.get("_tl_cleaned_up", False) else trace._grad_fn_logs
    summary, any_op_backed, any_keyed = _rollup_grad_fn_sites(
        grad_fn_records.values(), layer_lookup, fire_timings
    )
    if any_op_backed and not any_keyed:
        raise InvalidArgumentError(
            "This trace's op-backed grad-fns carry no site keys: it was "
            "captured/saved before site_key_v1 existed, so a per-site "
            "backward rollup would be silently empty.",
            code="site_key_unavailable",
            remedy="re-capture with a current TorchLens to mint site keys",
        )
    for entry in summary.values():
        entry["grad_fn_labels"] = tuple(sorted(entry["grad_fn_labels"]))
        entry["pass_coverage"] = tuple(sorted(entry["pass_coverage"]))
    return summary
