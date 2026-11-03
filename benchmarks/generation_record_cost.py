"""Cost of recording activations across a generation, as deterministic counters.

Two ways to save chosen sites at every generated token:

``baseline``
    The spelling that existed before the generation recorder: after a plain
    greedy ``generate()``, one fresh ``tl.record`` per step over that step's
    whole prefix (no KV cache), keeping the sites at every prefix position.
``recorder``
    ``tl.record(model, ids, save=..., generate_kwargs=...)``: one real cached
    ``generate()`` call, each decode forward recording only its new position.

Both runs use the same seeded, config-built distilgpt2-shaped model (2 layers,
64 wide, eager attention, CPU, float32), the same 4-token prompt and the same
three sites (both MLP outputs and the final norm). Counters per forward:
``ops`` (every capture event), ``predicate_calls`` (calls into the save
predicate, the recorder's hook-fire analogue), ``records`` (retained),
``saved_bytes`` (retained payload bytes) and ``elements`` (output elements of
every executed op, the work the forward did). Counters gate; the wall time of
two repeated runs per arm (an A/A pair) and the Python-heap peak are reported
beside them and never gate.

Run ``python benchmarks/generation_record_cost.py --lengths 12 48 --out OUT.json``;
one JSON document is printed and written to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any

import torch

import torchlens as tl

SITE_OPS = ("transformer.h.0.mlp.dropout", "transformer.h.1.mlp.dropout", "transformer.ln_f")
PROMPT = torch.tensor([[454, 137, 139, 386]])


class CountingSave:
    """A save predicate that keeps the site ops and counts its calls per pass."""

    def __init__(self) -> None:
        self.calls: Counter[int] = Counter()

    def __call__(self, ctx: Any) -> bool:
        self.calls[int(ctx.pass_index)] += 1
        return ctx.kind == "op" and ctx.address in SITE_OPS


def _per_pass(recording: Any, save: CountingSave) -> list[dict[str, int]]:
    """Return the counters of every pass of one recording."""

    ops: Counter[int] = Counter()
    elements: Counter[int] = Counter()
    for row in recording.raw_metadata():
        pass_index = int(row["pass_index"])
        ops[pass_index] += 1
        shape = row.get("shape")
        elements[pass_index] += math.prod(shape) if shape else 0
    rows = []
    for pass_index in sorted(ops):
        indexes = recording.by_pass.get(pass_index, [])
        payloads = [recording.records[index].ram_payload for index in indexes]
        rows.append(
            {
                "ops": ops[pass_index],
                "predicate_calls": save.calls[pass_index],
                "records": len(indexes),
                "saved_bytes": sum(p.numel() * p.element_size() for p in payloads),
                "elements": elements[pass_index],
            }
        )
    return rows


def _model() -> torch.nn.Module:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests.real_model.r0.families import build_distilgpt2

    return build_distilgpt2("eager")


def _greedy(n: int) -> dict[str, Any]:
    return {"max_new_tokens": n, "min_new_tokens": n, "do_sample": False, "use_cache": True}


def run_recorder(model: torch.nn.Module, n: int) -> list[dict[str, int]]:
    """Counters of one cached generation recording of ``n`` new tokens."""

    save = CountingSave()
    generation = tl.record(model, PROMPT, save=save, generate_kwargs=_greedy(n))
    return _per_pass(generation.recording, save)


def run_baseline(model: torch.nn.Module, n: int) -> list[dict[str, int]]:
    """Counters of one fresh full-prefix ``tl.record`` per generated token."""

    sequence = model.generate(PROMPT, **_greedy(n))
    rows = []
    for step in range(n):
        save = CountingSave()
        prefix = sequence[:, : PROMPT.shape[1] + step]
        recording = tl.record(model, prefix, input_kwargs={"use_cache": False}, save=save)
        rows.extend(_per_pass(recording, save))
    return rows


def _timed(fn: Any, model: torch.nn.Module, n: int) -> tuple[list[dict[str, int]], float, int]:
    tracemalloc.start()
    start = time.perf_counter()
    rows = fn(model, n)
    wall = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return rows, wall, peak


def _summary(rows: list[dict[str, int]]) -> dict[str, Any]:
    keys = rows[0].keys()
    later = rows[1:]
    return {
        "forwards": len(rows),
        "first": rows[0],
        "later_min": {k: min(r[k] for r in later) for k in keys} if later else {},
        "later_max": {k: max(r[k] for r in later) for k in keys} if later else {},
        "total": {k: sum(r[k] for r in rows) for k in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs=2, default=(12, 48))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = _model()
    report: dict[str, Any] = {"lengths": list(args.lengths), "arms": {}}
    for arm, fn in (("recorder", run_recorder), ("baseline", run_baseline)):
        for n in args.lengths:
            rows, wall_a, peak = _timed(fn, model, n)
            again, wall_b, _ = _timed(fn, model, n)
            assert again == rows, f"{arm} counters are not deterministic at {n}"
            report["arms"][f"{arm}/{n}"] = {
                **_summary(rows),
                "wall_s_aa": [round(wall_a, 3), round(wall_b, 3)],
                "python_heap_peak_bytes": peak,
            }
    short, long = args.lengths
    report["total_ratio_long_over_short"] = {
        arm: {
            key: round(
                report["arms"][f"{arm}/{long}"]["total"][key]
                / max(report["arms"][f"{arm}/{short}"]["total"][key], 1),
                3,
            )
            for key in ("ops", "predicate_calls", "records", "saved_bytes", "elements")
        }
        for arm in ("recorder", "baseline")
    }
    text = json.dumps(report, indent=2)
    args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
