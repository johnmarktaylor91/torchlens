"""Exactness fingerprints for prototype comparison against main.

For each capture configuration, writes one JSON record holding: the agent JSON dump of the trace
(graph, labels, module attribution, honesty facts; volatile timing fields dropped), a sha256 of
every saved activation by op label, the model output hash, the capture outcome, intervention fire
counts, and the results of ``tl.validate(..., scope="forward")`` and ``scope="saved"`` for the
plain model. Two commits' files are compared with ``exact.py --compare A B``.

Usage: python exact.py --model gpt2 --out out/exact.jsonl
       python exact.py --compare main.jsonl proto.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wl  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.options import CaptureOptions  # noqa: E402

VOLATILE = ("time", "duration", "elapsed", "_at", "timestamp", "memory", "pid", "address_id", "id(")


def thash(t: Any) -> str | None:
    if not isinstance(t, torch.Tensor):
        return None
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()[:16] + str(
        tuple(t.shape)
    )


def scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: scrub(v)
            for k, v in sorted(obj.items())
            if not any(tok in str(k).lower() for tok in VOLATILE)
        }
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def trace_fp(t: Any) -> dict[str, Any]:
    saved = {}
    for op in t.ops:
        try:
            out = op.out
        except Exception as exc:  # unsaved payloads refuse typed
            out = type(exc).__name__
        saved[str(op.label)] = thash(out) if isinstance(out, torch.Tensor) else str(out)[:40]
    dump = scrub(t.to_agent_json(max_ops=None))
    blob = json.dumps(dump, sort_keys=True, default=str)
    return {
        "n_ops": len(t.ops),
        "labels": [str(op.label) for op in t.ops],
        "saved": saved,
        "agent_json_sha": hashlib.sha256(blob.encode()).hexdigest()[:16],
        "agent_json_len": len(blob),
        "outcome": str(getattr(t, "outcome", None))[:200],
        "intervene_fires": getattr(t, "_tl_intervene_selector_fire_count", None),
        "save_fires": getattr(t, "_tl_save_selector_fire_count", None),
    }


def record_fp(rec: Any) -> dict[str, Any]:
    rows = []
    for r in rec.records:
        rows.append([str(r.ctx.label), str(r.ctx.layer_type), thash(r.ram_payload)])
    return {"n_ops": rec.n_ops, "records": rows, "status": rec.status}


def configs(cell: wl.Cell) -> dict[str, Callable[[], dict[str, Any]]]:
    m, ids = cell.model, cell.ids
    site = tl.module(cell.mid)
    other = tl.module(cell.blocks[1])
    logit = tl.module("logit_readout")
    steer = tl.steer(cell.direction, magnitude=wl.MAG, feature_axis=-1)
    spec = tl.when(site, steer)
    infer = CaptureOptions(inference_only=True)

    def tr(**kw: Any) -> dict[str, Any]:
        return trace_fp(tl.trace(m, ids, **kw))

    def rc(**kw: Any) -> dict[str, Any]:
        out = tl.record(m, ids, return_output=True, **kw)
        return {"output": thash(out[0]), **record_fp(out[1])}

    return {
        "trace_default": lambda: tr(),
        "trace_infer": lambda: tr(capture=infer),
        "trace_sparse_module": lambda: tr(save=site | logit),
        "trace_elicit": lambda: tr(save=site | logit, intervene=spec, capture=infer),
        "trace_steer_saveall": lambda: tr(intervene=spec),
        "trace_union_zero": lambda: tr(intervene=tl.when(site | other, tl.zero_ablate())),
        "trace_func_steer": lambda: tr(
            save=tl.func("linear") | logit, intervene=tl.when(tl.func("layer_norm"), tl.scale(0.5))
        ),
        "trace_module_and_func": lambda: tr(intervene=tl.when(site & tl.func("add"), steer)),
        "record_save": lambda: rc(save=site),
        "record_intervene": lambda: rc(save=site, intervene=spec),
        "record_intervene_nograd": lambda: _nograd(lambda: rc(save=site, intervene=spec)),
        "record_func_intervene": lambda: rc(
            save=logit, intervene=tl.when(tl.func("layer_norm"), tl.scale(0.5))
        ),
        "validate_forward": lambda: {"ok": bool(tl.validate(m, ids, scope="forward"))},
        "validate_saved": lambda: {"ok": bool(tl.validate(m, ids, scope="saved"))},
    }


def _nograd(fn: Callable[[], Any]) -> Any:
    with torch.no_grad():
        return fn()


def compare(a_path: str, b_path: str) -> int:
    def load(p: str) -> dict[tuple[str, str], Any]:
        out = {}
        for line in open(p):
            r = json.loads(line)
            out[(r["model"], r["config"])] = r["fp"]
        return out

    a, b = load(a_path), load(b_path)
    bad = 0
    for key in sorted(set(a) | set(b)):
        if a.get(key) == b.get(key):
            print("SAME", *key)
        else:
            bad += 1
            fa, fb = a.get(key) or {}, b.get(key) or {}
            diff = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
            print(
                "DIFF",
                *key,
                diff,
                {k: (str(fa.get(k))[:200], str(fb.get(k))[:200]) for k in diff[:3]},
            )
    print(f"{bad} differing of {len(set(a) | set(b))}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--out", default="out/exact.jsonl")
    ap.add_argument("--only", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    torch.set_num_threads(1)
    cell = wl.Cell(argparse.Namespace(model=args.model, seq=args.seq, prompts=1, sweep=1))
    for name, fn in configs(cell).items():
        if args.only and name not in args.only.split(","):
            continue
        try:
            fp = fn()
        except Exception as exc:
            fp = {"error": type(exc).__name__, "msg": str(exc)[:300]}
        with open(args.out, "a") as fh:
            fh.write(json.dumps({"model": args.model, "config": name, "fp": fp}) + "\n")
        print(name, "error" in fp, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
