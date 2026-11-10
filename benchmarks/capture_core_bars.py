"""Bars for the capture-core counters: the bar file's writer and its checker.

``capture_core_cost.py`` measures; this module turns one baseline measurement
into ``perf_baselines/capture_core_bars.json`` and checks a later measurement
against it. A bar is stored as a fraction of the baseline value of its counter
and as the resulting number; the fraction governs.

Three kinds of rule live in the bar file:

* ``every_lane``: counters that may not rise above the baseline for any change
  (retained memory C2, bulk readers C7, held tensor bytes C9);
* ``landed``: a counter's bar once the change that lowered it has landed (the
  measured value plus 5%), kept as a regression gate;
* ``lanes``: the bar each named lane must reach, checked only when that lane
  is named.
"""

from __future__ import annotations

from typing import Any, NamedTuple

SCHEMA = 1
#: Two measurements of one commit must agree to this relative tolerance, and a
#: "may not rise" counter may exceed its baseline by no more than this.
TOLERANCE = 0.005
LANDED_HEADROOM = 0.05
FULL_MODELS = ("chain1000", "varied200", "loopcell100", "resnet18", "gpt2_tiny4")
SMALL_MODELS = ("one_op", "func_relu200", "mod_relu200", "prep_only_resnet18", "cond100")
EMIT_PHASES = (
    "wrapper.emit",
    "wrapper.emit.make_entry",
    "wrapper.emit.shared_fields",
    "wrapper.emit.commit_op",
)


class Bar(NamedTuple):
    """One lane's bar on one counter.

    ``kind`` is ``"fraction"`` (at most ``value`` times the baseline),
    ``"max"`` (at most ``value``) or ``"min"`` (at least ``value``).
    """

    lane: str
    counter: str
    kind: str
    value: float
    plan: str


def _phase(name: str, model: str = "chain1000") -> str:
    return f"c1.{model}.phase.{name}"


def _fraction(lane: str, counter: str, bar: float, today: float, plan: str) -> Bar:
    return Bar(lane, counter, "fraction", bar / today, plan)


def lane_bars() -> tuple[Bar, ...]:
    """Return every lane bar the plan states as a counter.

    A bar the plan gives as "today to bar" is stored as the fraction
    ``bar / today`` with the plan's own figures; a later bar on a counter an
    earlier lane already lowered is a fraction of the original baseline.
    """

    emit_today = 545 + 361 + 212 + 104
    bars = [
        _fraction("W1", _phase("wrapper"), 90, 361, "wrapper checks 361 to 90"),
        Bar("M", "c5.calls_per_module_call", "fraction", 0.40, "C5 to 40% of baseline"),
        Bar("P", "c6.minimal_trace_calls", "fraction", 0.50, "C6 minimal trace to 50%"),
        Bar(
            "P",
            "c6.calls_per_prepared_module",
            "fraction",
            0.20,
            "calls per prepared module on the second trace to 20%",
        ),
        Bar("D", "c3.chain1000.emit_ingest_cells_per_op", "fraction", 0.40, "C3 to 40%"),
        _fraction("D", _phase("wrapper.emit.shared_fields"), 330, 545, "emit shared fields"),
        _fraction("D", _phase("post"), 420, 563, "postprocess other 563 to 420"),
        _fraction("D", _phase("post.compaction"), 400, 559, "compaction 559 to 400"),
        _fraction("G", _phase("post.mark_layer_depths"), 5, 166, "mark depths 166 to 5"),
        _fraction("G", _phase("post.mark_tracing_finished"), 5, 77, "tracing finished 77 to 5"),
        _fraction("G-if-deferred", _phase("post.hash_graph_shape"), 2, 87, "hash graph 87 to 2"),
        _fraction("F", _phase("post.log_final_layer_info"), 20, 442, "final layer info 442 to 20"),
        _fraction("L", _phase("post.build_layer_logs"), 10, 206, "build layer records 206 to 10"),
        _fraction("L", _phase("post.build_module_logs"), 10, 36, "build module records 36 to 10"),
        _fraction(
            "L",
            _phase("post.build_module_logs", "resnet18"),
            10,
            491,
            "build module records 491 to 10 on resnet18",
        ),
        _fraction("K", _phase("post.build_lookup_keys"), 10, 342, "build lookup keys 342 to 10"),
        _fraction("I", _phase("post.compaction"), 60, 559, "compaction 400 to 60"),
        _fraction(
            "E3",
            "c1.chain1000.emit_total_per_op",
            600,
            emit_today,
            "emit total at or under 600 after E1 to E3 and F1 (REPLAN joint bar; PLAN had 500)",
        ),
        _fraction("S1", _phase("post.remove_orphan_nodes"), 60, 364, "remove orphans 364 to 60"),
        _fraction("S2", _phase("post.group_recurrent_layers"), 50, 133, "recurrence 133 to 50"),
        Bar("Q", "c7.chain1000.to_pandas_per_op", "fraction", 0.50, "to_pandas() to 50%"),
        Bar(
            "Q",
            "c7.chain1000.trace_plus_to_pandas_per_op",
            "fraction",
            1.0,
            "trace plus to_pandas() not above baseline",
        ),
        _fraction("T1", "c1.chain1000.emit_total_per_op", 80, emit_today, "emit total to 80"),
        _fraction("T2", _phase("post.map_final_labels"), 5, 75, "map final labels 75 to 5"),
        _fraction("T3", _phase("post"), 150, 563, "postprocess other 420 to 150"),
        Bar("T4a", _phase("post.find_output_ancestors"), "fraction", 0.10, "to 10% of baseline"),
        Bar(
            "T4a",
            _phase("post.fix_buffer_layers", "resnet18"),
            "fraction",
            0.10,
            "to 10% of baseline, on resnet18 (the only model of the set with buffers)",
        ),
        Bar(
            "T4b",
            _phase("post.mark_conditional_branches", "cond100"),
            "fraction",
            0.10,
            "to 10% of baseline, on cond100 (the only model of the set with a conditional)",
        ),
        _fraction("T4c", "c1.chain1000.total_per_op", 550, 4875, "total at or under 550"),
    ]
    for model in FULL_MODELS:
        total = f"c1.{model}.total_per_op"
        bars.append(Bar("X", total, "fraction", 1.01, "C1 total within 1%"))
        bars.append(Bar("E0", total, "fraction", 1.0 + TOLERANCE, "C1 unchanged"))
        bars.append(
            Bar("E2", f"c9.{model}.save_none.argument_bytes_max", "max", 0.0, "C9 arguments 0")
        )
        for kind in ("Op", "Layer"):
            bars.append(Bar("B5", f"c4.{model}.{kind}", "max", 0.0, "C4 to 0 for Op and Layer"))
        for kind in ("Module", "ModuleCall", "Param", "Buffer", "GradFn"):
            bars.append(Bar("B6", f"c4.{model}.{kind}", "max", 0.0, "C4 to 0 for these kinds"))
    for model in ("resnet18", "gpt2_tiny4"):
        bars.append(Bar("W2", f"c8.{model}.fast_lane_share", "min", 0.95, "C8 at least 95%"))
    return tuple(bars) + call_time_bars()


#: The call-time floor's models (``capture_core_floor.py``).
FLOOR_MODELS = (*FULL_MODELS, "mod_relu200")


def _floor(model: str, section: str) -> str:
    return f"floor.{model}.{section}_per_output"


def call_time_bars() -> tuple[Bar, ...]:
    """Return the stage 2 call-time lanes' bars (REPLAN section 4) on N0's floor counters.

    A fraction is of N0's baseline; a ruled absolute figure is a ``max``.
    """

    bars = [
        Bar("E1", _floor(m, "call_site"), "fraction", 0.35, "call-site calls per output to 35%")
        for m in ("chain1000", "loopcell100")
    ]
    bars += [
        Bar("E2", _floor("chain1000", "arg_scan"), "fraction", 0.40, "argument scan to 40%"),
        Bar("E2", _floor("varied200", "arg_scan"), "fraction", 0.55, "argument scan to 55%"),
        Bar("E3", _floor("chain1000", "autograd"), "fraction", 0.25, "autograd section to 25%"),
        Bar("F1", _floor("chain1000", "payload"), "max", 77.0, "payload 180 to 77 (ruled reset)"),
        Bar("A1", _floor("varied200", "autograd"), "fraction", 0.50, "autograd section to 50%"),
        Bar("W3", _phase("wrapper"), "max", 60.0, "wrapper phase 86 to 60 calls per op"),
    ]
    for model in FLOOR_MODELS:
        bars.append(
            Bar("E3", _floor(model, "output_facts"), "fraction", 0.60, "output facts to 60%")
        )
        bars.append(
            Bar(
                "W3",
                _floor(model, "label_read_stamp"),
                "fraction",
                0.50,
                "label read and stamp to 50%",
            )
        )
    for model, limit in (("varied200", 150.0), ("gpt2_tiny4", 190.0)):
        bars.append(
            Bar("MC", _phase("forward.other", model), "max", limit, "forward.other (REPLAN figure)")
        )
    return tuple(bars)


#: Bars the plan states in terms another lane's output defines; no number yet.
PENDING_BARS = (
    "E1: emit down by the call-site family's share, from lane H2's family table",
    "E2: emit down by the argument family's share, from lane H2's family table",
    "E1: code-context lookups per op to 0 for repeated sites (no counter in this tool)",
)


def flatten(measurement: dict[str, Any]) -> dict[str, float]:
    """Return the gated counters of one measurement as ``name -> number``.

    Wall time, the environment block and descriptive detail are left out:
    only deterministic counters are flattened.
    """

    flat: dict[str, float] = {}
    for model, record in measurement.get("models", {}).items():
        if "error" in record:
            continue
        rows = record["rows"]
        c1 = record["c1"]
        flat[f"c1.{model}.total_per_op"] = c1["total"] / rows
        for phase, calls in c1["phases"].items():
            flat[f"c1.{model}.phase.{phase}"] = calls / rows
        flat[f"c1.{model}.emit_total_per_op"] = (
            sum(c1["phases"].get(p, 0) for p in EMIT_PHASES) / rows
        )
        flat[f"c8.{model}.fast_lane_share"] = record["c8"]["fast_lane_share"]
        for kind, count in record.get("c4", {}).items():
            flat[f"c4.{model}.{kind}"] = count
        for mode, held in record.get("c2", {}).items():
            flat[f"c2.{model}.{mode}.blocks_per_op"] = held["blocks"] / rows
            flat[f"c2.{model}.{mode}.objects_per_op"] = held["objects"] / rows
        if "c3" in record:
            cells = record["c3"]["cells"]
            for stage, count in cells.items():
                flat[f"c3.{model}.{stage}_cells_per_op"] = count / rows
            flat[f"c3.{model}.emit_ingest_cells_per_op"] = (
                cells["capture"] + cells["ingest"]
            ) / rows
            for name, count in record["c3"]["calls"].items():
                flat[f"c3.{model}.calls.{name}_per_op"] = count / rows
        for reader, calls in record.get("c7", {}).items():
            flat[f"c7.{model}.{reader}_per_op"] = calls / rows
        if "to_pandas" in record.get("c7", {}):
            flat[f"c7.{model}.trace_plus_to_pandas_per_op"] = (
                c1["total"] + record["c7"]["to_pandas"]
            ) / rows
        for mode, held in record.get("c9", {}).items():
            for key in ("peak_bytes", "argument_bytes_max", "saved_output_bytes_max"):
                if key in held:
                    flat[f"c9.{model}.{mode}.{key}"] = held[key]
    for name, value in measurement.get("derived", {}).items():
        flat[name] = value
    return flat


def _every_lane_counters(flat: dict[str, float]) -> list[str]:
    return sorted(
        name
        for name in flat
        if name.startswith(("c2.", "c7.", "c9.")) and name.split(".")[1] in FULL_MODELS
    )


def build_bars(measurement: dict[str, Any], base_commit: str) -> dict[str, Any]:
    """Build the bar file's content from the baseline measurement."""

    flat = flatten(measurement)
    lanes: dict[str, list[dict[str, Any]]] = {}
    for bar in lane_bars():
        baseline = flat.get(bar.counter)
        entry: dict[str, Any] = {"counter": bar.counter, "kind": bar.kind, "plan": bar.plan}
        if bar.kind == "fraction":
            entry["fraction"] = round(bar.value, 6)
            entry["baseline"] = baseline
            entry["bar"] = None if baseline is None else baseline * bar.value
        else:
            entry["bar"] = bar.value
            entry["baseline"] = baseline
        lanes.setdefault(bar.lane, []).append(entry)
    return {
        "schema": SCHEMA,
        "base_commit": base_commit,
        "tolerance": TOLERANCE,
        "landed_headroom": LANDED_HEADROOM,
        "environment": measurement.get("environment", {}),
        "baseline": dict(sorted(flat.items())),
        "every_lane": _every_lane_counters(flat),
        "landed": {},
        "lanes": dict(sorted(lanes.items())),
        "pending": list(PENDING_BARS),
    }


def _over(value: float, limit: float, tolerance: float) -> bool:
    return value > limit * (1.0 + tolerance) if limit > 0 else value > limit


def check(flat: dict[str, float], bars: dict[str, Any], lane: str | None = None) -> list[str]:
    """Return the bar violations of one measurement (empty when it passes).

    Parameters
    ----------
    flat
        The measurement, flattened by :func:`flatten`.
    bars
        The bar file's content.
    lane
        A lane id: its own bars are checked as well as the standing rules.
    """

    failures: list[str] = []
    tolerance = bars.get("tolerance", TOLERANCE)
    baseline = bars["baseline"]
    for name in bars["every_lane"]:
        if name not in flat:
            failures.append(f"{name}: not measured")
        elif _over(flat[name], baseline[name], tolerance):
            failures.append(
                f"{name}: {flat[name]:.6g} rose above the baseline {baseline[name]:.6g}"
            )
    for name, limit in bars.get("landed", {}).items():
        if name not in flat:
            failures.append(f"{name}: not measured")
        elif flat[name] > limit:
            failures.append(f"{name}: {flat[name]:.6g} is above its landed bar {limit:.6g}")
    if lane is None:
        return failures
    if lane not in bars["lanes"]:
        return [*failures, f"lane {lane!r} has no bars in the bar file"]
    for entry in bars["lanes"][lane]:
        name = entry["counter"]
        limit = entry["bar"]
        if name not in flat or limit is None:
            failures.append(f"{lane} {name}: not measured")
        elif entry["kind"] == "min":
            if flat[name] < limit:
                failures.append(f"{lane} {name}: {flat[name]:.6g} is below the bar {limit:.6g}")
        elif flat[name] > limit:
            failures.append(
                f"{lane} {name}: {flat[name]:.6g} is above the bar {limit:.6g} ({entry['plan']})"
            )
    return failures


def compare(
    first: dict[str, float], second: dict[str, float], tolerance: float = TOLERANCE
) -> list[str]:
    """Return the counters on which two measurements of one commit disagree."""

    differences: list[str] = []
    for name in sorted(set(first) | set(second)):
        if name not in first or name not in second:
            differences.append(f"{name}: measured in only one run")
            continue
        a, b = first[name], second[name]
        if abs(a - b) > tolerance * max(abs(a), abs(b)):
            differences.append(f"{name}: {a:.6g} against {b:.6g}")
    return differences


def tighten(bars: dict[str, Any], lane: str, flat: dict[str, float]) -> dict[str, Any]:
    """Return the bar file with ``lane``'s counters landed at measured plus 5%."""

    landed = dict(bars.get("landed", {}))
    headroom = bars.get("landed_headroom", LANDED_HEADROOM)
    for entry in bars["lanes"][lane]:
        if entry["kind"] != "min" and entry["counter"] in flat:
            landed[entry["counter"]] = flat[entry["counter"]] * (1.0 + headroom)
    return {**bars, "landed": dict(sorted(landed.items()))}
