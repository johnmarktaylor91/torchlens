"""Cost of re-seen raw labels in ``LiveIndex``, as deterministic counters.

A multi-pass ``Recorder`` folds every pass's journal into one ``LiveIndex``;
from pass 2 on every raw label it appends has been seen before, and a failed
partial recording concatenates the failing pass's op events onto the
accumulated journal the same way. Amendment folding replaces indexed events
through ``LiveIndex.replace``. This benchmark counts what each re-seen
``append`` and each ``replace`` costs.

Counters per case: full ``rebuild_edges`` calls, re-seen appends and
replaces, ``rebuild_edges`` calls inside them, child-list operations
(append, insert, remove and the like on ``children_by_parent`` lists) inside
them against the bound of the old plus new event's distinct parent count,
and the total Python calls per forward (or per appended event for the
synthetic cases) under ``cProfile``. Counters gate. Each case also prints a
SHA-256 digest of the index state after every pass (``labels``,
``by_raw_label`` keys with the identity of each event, and
``children_by_parent`` as its key set with each list in order) and, for the
recorder cases, of the saved outputs. The order of ``children_by_parent``
gets its own digest, reported as information. The wall time of two
unprofiled runs (an A/A pair) and the Python-heap peak are context only.

Cases
-----
``recorder-12`` / ``recorder-48``
    The generation recorder on the config-built distilgpt2-shaped model of
    ``generation_record_cost.py`` (2 layers, 64 wide, eager attention, CPU,
    float32, 4-token prompt), greedy for 12 and 48 new tokens.
``synthetic-100`` / ``synthetic-1000``
    A bare ``LiveIndex`` fed N seeded events, then the same N labels again as
    fresh events with the same parents.
``synthetic-rewired-1000``
    As ``synthetic-1000``, but every tenth re-fed event draws new parents.
``failed-partial``
    A ``Recorder`` over a 60-op chain whose third pass raises halfway, with
    ``on_forward_error="return_partial"``.

Run ``python benchmarks/live_index_append_counters.py --out OUT.json``; one
JSON document is printed and written to ``--out``.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import pstats
import random
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torchlens as tl  # noqa: E402
from torchlens.fastlog import Recorder  # noqa: E402
from torchlens.fastlog._recorder import Recorder as _RecorderImpl  # noqa: E402
from torchlens.ir.live_index import LiveIndex  # noqa: E402

PROMPT = torch.tensor([[454, 137, 139, 386]])
SITE_OPS = ("transformer.h.0.mlp.dropout", "transformer.h.1.mlp.dropout", "transformer.ln_f")
CASES = (
    "recorder-12",
    "recorder-48",
    "synthetic-100",
    "synthetic-1000",
    "synthetic-rewired-1000",
    "failed-partial",
)


# ---------------------------------------------------------------- counting


class _Tally:
    """Counters shared by the instrumented ``LiveIndex`` methods."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.list_ops = 0
        self.depth = 0


_TALLY = _Tally()


class _CountingList(list):  # type: ignore[type-arg]
    """A child list that counts every mutation."""

    def _bump(self) -> None:
        _TALLY.list_ops += 1

    def append(self, item: Any) -> None:
        self._bump()
        super().append(item)

    def insert(self, index: Any, item: Any) -> None:
        self._bump()
        super().insert(index, item)

    def remove(self, item: Any) -> None:
        self._bump()
        super().remove(item)

    def pop(self, index: Any = -1) -> Any:
        self._bump()
        return super().pop(index)

    def extend(self, items: Any) -> None:
        self._bump()
        super().extend(items)

    def clear(self) -> None:
        self._bump()
        super().clear()

    def __setitem__(self, index: Any, value: Any) -> None:
        self._bump()
        super().__setitem__(index, value)

    def __delitem__(self, index: Any) -> None:
        self._bump()
        super().__delitem__(index)


def _counting_children(index: LiveIndex) -> None:
    """Make every child list of ``index`` (and future ones) a counting list."""

    children = index.children_by_parent
    if isinstance(children, defaultdict):
        children.default_factory = _CountingList
    for key, value in children.items():
        if type(value) is not _CountingList:
            children[key] = _CountingList(value)


def _distinct_parents(event: Any) -> int:
    return len({edge.parent_label_raw for edge in event.parents})


@contextmanager
def counting() -> Iterator[_Tally]:
    """Instrument ``LiveIndex.append``, ``replace`` and ``rebuild_edges``."""

    originals = (LiveIndex.append, LiveIndex.replace, LiveIndex.rebuild_edges)
    original_append, original_replace, original_rebuild = originals
    _TALLY.__init__()

    def measured(kind: str, index: LiveIndex, event: Any, call: Callable[[], None]) -> None:
        tally = _TALLY
        old = index.by_raw_label.get(event.label_raw)
        bound = (_distinct_parents(old) if old is not None else 0) + _distinct_parents(event)
        rebuilds = tally.counts["rebuild_edges"]
        ops = tally.list_ops
        tally.depth += 1
        try:
            call()
        finally:
            tally.depth -= 1
        spent_rebuilds = tally.counts["rebuild_edges"] - rebuilds
        spent_ops = tally.list_ops - ops
        tally.counts[kind] += 1
        tally.counts[f"{kind}_rebuilds"] += spent_rebuilds
        tally.counts[f"{kind}_list_ops"] += spent_ops
        tally.counts[f"{kind}_list_op_bound"] += bound
        tally.counts[f"{kind}_over_bound"] += int(spent_ops > bound)
        tally.counts[f"{kind}_max_list_ops"] = max(tally.counts[f"{kind}_max_list_ops"], spent_ops)

    def append(self: LiveIndex, event: Any) -> None:
        _counting_children(self)
        if _TALLY.depth or event.label_raw not in self.by_raw_label:
            _TALLY.counts["fresh_append"] += 1
            original_append(self, event)
            return
        measured("reseen_append", self, event, lambda: original_append(self, event))

    def replace(self: LiveIndex, event: Any) -> None:
        _counting_children(self)
        if _TALLY.depth:
            original_replace(self, event)
            return
        measured("replace", self, event, lambda: original_replace(self, event))

    def rebuild_edges(self: LiveIndex) -> None:
        _counting_children(self)
        _TALLY.counts["rebuild_edges"] += 1
        original_rebuild(self)

    LiveIndex.append = append  # type: ignore[method-assign]
    LiveIndex.replace = replace  # type: ignore[method-assign]
    LiveIndex.rebuild_edges = rebuild_edges  # type: ignore[method-assign]
    try:
        yield _TALLY
    finally:
        LiveIndex.append, LiveIndex.replace, LiveIndex.rebuild_edges = originals  # type: ignore[method-assign]


def _tally_report(tally: _Tally) -> dict[str, Any]:
    counts = dict(sorted(tally.counts.items()))
    report: dict[str, Any] = {"counts": counts}
    for kind in ("reseen_append", "replace"):
        n = counts.get(kind, 0)
        if n:
            report[f"{kind}_per_call"] = {
                "rebuild_edges": round(counts[f"{kind}_rebuilds"] / n, 4),
                "list_ops": round(counts[f"{kind}_list_ops"] / n, 4),
                "list_op_bound": round(counts[f"{kind}_list_op_bound"] / n, 4),
            }
    return report


# ---------------------------------------------------------------- digests


def index_digest(index: LiveIndex, identity: Callable[[Any], Any]) -> dict[str, Any]:
    """Digest of one index state: gated parts, and the child key order as information."""

    state = {
        "labels": list(index.labels),
        "by_raw_label": [[label, identity(event)] for label, event in index.by_raw_label.items()],
        "children_by_parent": sorted(
            [key, list(value)] for key, value in index.children_by_parent.items()
        ),
    }
    text = json.dumps(state, sort_keys=True, default=str)
    key_order = json.dumps(list(index.children_by_parent))
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "labels": len(index.labels),
        "edges": sum(len(value) for value in index.children_by_parent.values()),
        "child_order_sha256": hashlib.sha256(key_order.encode()).hexdigest(),
    }


def _journal_identity(capture_events: Any) -> Callable[[Any], Any]:
    positions = {id(event): number for number, event in enumerate(capture_events.op_events)}

    def identity(event: Any) -> Any:
        number = positions.get(id(event))
        if number is not None:
            return ["op_events", number]
        return [type(event).__name__, int(getattr(event, "seq", -1) or -1)]

    return identity


@contextmanager
def pass_digests() -> Iterator[list[dict[str, Any]]]:
    """Digest the recorder's live index after every absorbed pass and failure fold."""

    digests: list[dict[str, Any]] = []
    absorb = _RecorderImpl._absorb_pass_events
    mark_failed = _RecorderImpl._mark_recording_failed

    def record(recorder: Any, step: str) -> None:
        events = recorder._capture_events
        if events is not None:
            digest = index_digest(events.live_index, _journal_identity(events))
            digests.append({"step": step, **digest})

    def absorbed(self: Any, trace: Any) -> None:
        absorb(self, trace)
        record(self, "pass")

    def failed(self: Any, trace: Any, exc: BaseException) -> Any:
        result = mark_failed(self, trace, exc)
        record(self, "failed-fold")
        return result

    _RecorderImpl._absorb_pass_events = absorbed  # type: ignore[method-assign]
    _RecorderImpl._mark_recording_failed = failed  # type: ignore[method-assign]
    try:
        yield digests
    finally:
        _RecorderImpl._absorb_pass_events = absorb  # type: ignore[method-assign]
        _RecorderImpl._mark_recording_failed = mark_failed  # type: ignore[method-assign]


def recording_digest(recording: Any) -> dict[str, Any]:
    """Digest of every saved record: label, pass and payload bytes, in order."""

    digest = hashlib.sha256()
    for record in recording.records:
        digest.update(f"{record.ctx.label}|{record.ctx.pass_index}".encode())
        payload = record.ram_payload
        if isinstance(payload, torch.Tensor):
            value = payload.detach().cpu().contiguous()
            digest.update(f"{value.dtype}{tuple(value.shape)}".encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"sha256": digest.hexdigest(), "records": len(recording.records)}


# ---------------------------------------------------------------- cases


@dataclass(frozen=True)
class _Edge:
    parent_label_raw: str


@dataclass(frozen=True)
class _Event:
    label_raw: str
    parents: tuple[_Edge, ...]
    ordinal: int


def _draw_parents(rng: random.Random, number: int) -> tuple[_Edge, ...]:
    pool = [f"op_{j}" for j in range(number)] or ["input_0"]
    picks = [rng.choice(pool) for _ in range(rng.randint(1, 3))]
    if rng.random() < 0.1:
        picks.append("input_0")
    return tuple(_Edge(label) for label in picks)


def synthetic_stream(n: int, *, rewire_every: int = 0, seed: int = 0) -> list[list[_Event]]:
    """Two passes over N seeded labels; pass 2 re-feeds them as fresh events."""

    rng = random.Random(seed)
    first = [_Event(f"op_{i}", _draw_parents(rng, i), i) for i in range(n)]
    second = []
    for i, event in enumerate(first):
        rewired = rewire_every and i % rewire_every == 0
        parents = _draw_parents(rng, n) if rewired else event.parents
        second.append(_Event(event.label_raw, parents, n + i))
    return [first, second]


class _Chain(nn.Module):
    """A 60-op elementwise chain whose third call raises halfway."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        for step in range(20):
            if self.calls == 3 and step == 10:
                raise RuntimeError("third pass fails halfway")
            x = torch.relu(torch.mul(torch.add(x, 1.0), 0.5))
        return x


def _save_ops(ctx: Any) -> bool:
    return ctx.kind == "op"


class _SiteSave:
    def __call__(self, ctx: Any) -> bool:
        return ctx.kind == "op" and ctx.address in SITE_OPS


def _gpt() -> nn.Module:
    from tests.real_model.r0.families import build_distilgpt2

    return build_distilgpt2("eager")


def _greedy(n: int) -> dict[str, Any]:
    return {"max_new_tokens": n, "min_new_tokens": n, "do_sample": False, "use_cache": True}


def _run_recorder(model: nn.Module, n: int) -> Any:
    return tl.record(model, PROMPT, save=_SiteSave(), generate_kwargs=_greedy(n)).recording


def _run_failed_partial() -> Any:
    with Recorder(_Chain(), save=_save_ops, on_forward_error="return_partial") as recorder:
        for _ in range(3):
            recorder.log(torch.ones(4))
        return recorder.recording


def _run_synthetic(passes: list[list[_Event]], digests: list[dict[str, Any]] | None) -> None:
    index = LiveIndex()
    for events in passes:
        for event in events:
            index.append(event)
        if digests is not None:
            digests.append({"step": "pass", **index_digest(index, lambda event: event.ordinal)})


def _workload(case: str, model: nn.Module | None) -> tuple[Callable[[], Any], Callable[[Any], int]]:
    """Return the case's workload and a function from its result to its unit count."""

    if case.startswith("recorder-"):
        n = int(case.split("-")[1])
        return (lambda: _run_recorder(model, n)), (lambda recording: recording.n_passes)
    if case == "failed-partial":
        return _run_failed_partial, (lambda recording: recording.n_passes)
    n = int(case.rsplit("-", 1)[1])
    passes = synthetic_stream(n, rewire_every=10 if "rewired" in case else 0)
    return (lambda: _run_synthetic(passes, None)), (lambda _: 2 * n)


def run(case: str, *, wall: bool) -> dict[str, Any]:
    """Counters, digests and (optionally) wall time of one case."""

    torch.manual_seed(0)
    model = _gpt() if case.startswith("recorder-") else None
    workload, units = _workload(case, model)
    workload()  # warm-up: imports and caches stay out of the counted runs

    digests: list[dict[str, Any]] = []
    with counting() as tally, pass_digests() as recorder_digests:
        if case.startswith("synthetic"):
            n = int(case.rsplit("-", 1)[1])
            _run_synthetic(
                synthetic_stream(n, rewire_every=10 if "rewired" in case else 0), digests
            )
            result = None
        else:
            result = workload()
            digests = recorder_digests
        counters = _tally_report(tally)

    profiler = cProfile.Profile()
    profiler.enable()
    profiled = workload()
    profiler.disable()
    stats = pstats.Stats(profiler)
    total_calls = int(stats.total_calls)  # type: ignore[attr-defined]
    unit_count = units(profiled)
    report: dict[str, Any] = {
        "case": case,
        "units": unit_count,
        "unit": "event" if case.startswith("synthetic") else "forward",
        **counters,
        "profile": {
            "total_calls": total_calls,
            "total_calls_per_unit": round(total_calls / unit_count, 1),
            "rebuild_edges_calls": _ncalls(stats, "rebuild_edges"),
        },
        "state_digests": digests,
        "state_digest_sha256": hashlib.sha256(
            json.dumps([d["sha256"] for d in digests]).encode()
        ).hexdigest(),
    }
    if result is not None:
        report["output_digest"] = recording_digest(result)
    if wall:
        report["wall_s_aa"] = [round(_timed(workload)[0], 3), round(_timed(workload)[0], 3)]
        report["python_heap_peak_bytes"] = _peak(workload)
    return report


def _ncalls(stats: pstats.Stats, name: str) -> int:
    table = stats.stats  # type: ignore[attr-defined]
    return sum(
        row[1] for (path, _, func), row in table.items() if func == name and "live_index" in path
    )


def _timed(workload: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    result = workload()
    return time.perf_counter() - start, result


def _peak(workload: Callable[[], Any]) -> int:
    tracemalloc.start()
    workload()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-wall", action="store_true", help="skip the A/A wall-time pair")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    cases = CASES if args.case == "all" else (args.case,)
    document = {"cases": {case: run(case, wall=not args.no_wall) for case in cases}}
    text = json.dumps(document, indent=2)
    args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
