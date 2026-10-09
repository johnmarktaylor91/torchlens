"""Run one (model, tool, workload) in this process and append JSON results.

python -m benchmarks.mi_workloads.run --model gpt2 --tool hooks --workload cache --out r.jsonl --resdir res/
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from typing import Any

import torch

from . import common as C
from .tools import Skip, make_tool

torch.set_num_threads(int(os.environ.get("TL_BENCH_THREADS", "1")))


def site(tool: Any, fam: C.Family, kind: str, i: int) -> str:
    if tool.name == "tlens":
        return f"{kind}{i}"
    return (fam.block if kind == "L" else fam.attn).format(i=i)


def unembed_fn(tool: Any, model: Any, fam: C.Family):
    """Final norm + unembed as a callable on a hidden-state tensor (for the logit lens)."""

    if tool.name == "tlens":
        ht = tool.ht
        return lambda h: ht.unembed(ht.ln_final(h))
    norm = model.get_submodule(fam.final_norm)
    head = model.get_submodule(fam.unembed)
    return lambda h: head(norm(h))


def wl_cache(tool, model, fam, a) -> dict[str, Any]:
    data = C.make_dataset(a.n_seqs, a.seq_len, model.config.vocab_size)
    L = C.n_layers(model)
    sites = [site(tool, fam, "L", i) for i in range(L)] + [
        site(tool, fam, "A", i) for i in range(L)
    ]
    store: list[torch.Tensor] = []

    def run():
        store.clear()
        for b in C.batches(data, a.batch):
            got = tool.cache(b, sites)
            store.extend(got[s] for s in sites)
        return torch.stack([t.float().mean(dim=(0, 1)) for t in store[: 2 * L]])

    times, res = C.timed(run)
    return {
        "times": times,
        "result": res,
        "bytes_stored": C.tensor_bytes(store),
        "n_tensors": len(store),
        "n_forwards": -(-a.n_seqs // a.batch),
    }


def wl_steer_sweep(tool, model, fam, a) -> dict[str, Any]:
    data = C.make_dataset(a.batch, a.seq_len, model.config.vocab_size, seed=1)
    L, d = C.n_layers(model), C.d_model(model)
    g = torch.Generator().manual_seed(2)
    direction = torch.randn(d, generator=g)
    direction = direction / direction.norm()
    layers = sorted({L // 4, L // 2, (3 * L) // 4})
    mags = [1.0, 2.0, 4.0, 8.0]

    def run():
        outs = []
        for li in layers:
            for m in mags:
                logits = tool.forward_add(data, site(tool, fam, "L", li), direction * m)
                outs.append(logits[:, -1, :].detach().clone())
        return torch.stack(outs)

    times, res = C.timed(run)
    return {"times": times, "result": res, "n_forwards": len(layers) * len(mags)}


def _metric_pair(model: Any) -> tuple[int, int]:
    return 1000, 2000  # two fixed token ids; logit difference at the last position


def wl_patch_sweep(tool, model, fam, a) -> dict[str, Any]:
    clean = C.make_dataset(1, a.patch_seq, model.config.vocab_size, seed=3)
    corrupt = clean.clone()
    corrupt[0, a.patch_seq // 2] = 7  # one corrupted token
    L = C.n_layers(model)
    sites = [site(tool, fam, "L", i) for i in range(L)]
    t1, t2 = _metric_pair(model)
    positions = list(range(0, a.patch_seq, max(1, a.patch_seq // a.patch_positions)))[
        : a.patch_positions
    ]

    def run():
        c_cache = tool.cache(clean, sites)
        k_cache = tool.cache(corrupt, sites)
        out = torch.zeros(L, len(positions))
        for li, s in enumerate(sites):
            for pj, p in enumerate(positions):
                delta = torch.zeros_like(k_cache[s])
                delta[:, p] = c_cache[s][:, p] - k_cache[s][:, p]
                logits = tool.forward_add(corrupt, s, delta)
                out[li, pj] = (logits[0, -1, t1] - logits[0, -1, t2]).item()
        return out

    times, res = C.timed(run)
    return {"times": times, "result": res, "n_forwards": 2 + L * len(positions)}


def wl_attr_patch(tool, model, fam, a) -> dict[str, Any]:
    clean = C.make_dataset(1, a.patch_seq, model.config.vocab_size, seed=3)
    corrupt = clean.clone()
    corrupt[0, a.patch_seq // 2] = 7
    L = C.n_layers(model)
    sites = [site(tool, fam, "L", i) for i in range(L)]
    t1, t2 = _metric_pair(model)
    metric = lambda logits: logits[0, -1, t1] - logits[0, -1, t2]  # noqa: E731

    def run():
        c_cache = tool.cache(clean, sites)
        ag = tool.grads(corrupt, sites, metric)
        return torch.stack([((c_cache[s] - ag[s][0]) * ag[s][1]).sum(-1)[0] for s in sites])

    times, res = C.timed(run)
    return {"times": times, "result": res, "n_forwards": 2, "n_backwards": 1}


def wl_generate(tool, model, fam, a, sample: bool = False) -> dict[str, Any]:
    prompt = C.make_dataset(1, 8, model.config.vocab_size, seed=4)
    L, d = C.n_layers(model), C.d_model(model)
    g = torch.Generator().manual_seed(2)
    direction = torch.randn(d, generator=g)
    direction = direction / direction.norm() * 4.0

    def run():
        return tool.generate(prompt, site(tool, fam, "L", L // 2), direction, a.new_tokens, sample)

    times, res = C.timed(run)
    return {"times": times, "result": res.to(torch.int64), "n_new_tokens": a.new_tokens}


def wl_sae(tool, model, fam, a) -> dict[str, Any]:
    data = C.make_dataset(a.n_seqs, a.seq_len, model.config.vocab_size)
    L, d = C.n_layers(model), C.d_model(model)
    g = torch.Generator().manual_seed(5)
    W = torch.randn(d, 8 * d, generator=g) / d**0.5
    b = torch.zeros(8 * d)
    s = site(tool, fam, "L", L // 2)

    def run():
        acc = torch.zeros(8 * d)
        n = 0
        for bt in C.batches(data, a.batch):
            h = tool.cache(bt, [s])[s].float()
            feats = torch.relu(h @ W + b)
            acc += feats.sum(dim=(0, 1))
            n += h.shape[0] * h.shape[1]
        return acc / n

    times, res = C.timed(run)
    return {"times": times, "result": res, "n_forwards": -(-a.n_seqs // a.batch)}


def wl_logit_lens(tool, model, fam, a) -> dict[str, Any]:
    data = C.make_dataset(a.n_seqs, a.seq_len, model.config.vocab_size)
    L = C.n_layers(model)
    sites = [site(tool, fam, "L", i) for i in range(L)]
    unembed = unembed_fn(tool, model, fam)

    def run():
        agree = torch.zeros(L)
        n = 0
        for bt in C.batches(data, a.batch):
            got = tool.cache(bt, sites)
            with torch.no_grad():
                final = unembed(got[sites[-1]].float()).argmax(-1)
                for li, s in enumerate(sites):
                    agree[li] += (unembed(got[s].float()).argmax(-1) == final).float().sum()
            n += bt.numel()
        return agree / n

    times, res = C.timed(run)
    return {"times": times, "result": res, "n_forwards": -(-a.n_seqs // a.batch)}


def wl_grad_capture(tool, model, fam, a) -> dict[str, Any]:
    data = C.make_dataset(4 * a.batch, a.seq_len, model.config.vocab_size, seed=6)
    L = C.n_layers(model)
    sites = [site(tool, fam, "L", i) for i in sorted({L // 4, L // 2, (3 * L) // 4})]
    t1, t2 = _metric_pair(model)
    metric = lambda logits: (logits[:, -1, t1] - logits[:, -1, t2]).sum()  # noqa: E731

    def run():
        outs = []
        for bt in C.batches(data, a.batch):
            ag = tool.grads(bt, sites, metric)
            outs.append(
                torch.stack(
                    [
                        torch.cat([ag[s][0].mean(dim=(0, 1)), ag[s][1].mean(dim=(0, 1))])
                        for s in sites
                    ]
                )
            )
        return torch.stack(outs)

    times, res = C.timed(run)
    return {"times": times, "result": res, "n_forwards": 4, "n_backwards": 4}


WORKLOADS = {
    "cache": wl_cache,
    "steer_sweep": wl_steer_sweep,
    "patch_sweep": wl_patch_sweep,
    "attr_patch": wl_attr_patch,
    "generate_greedy": lambda t, m, f, a: wl_generate(t, m, f, a, False),
    "generate_sampled": lambda t, m, f, a: wl_generate(t, m, f, a, True),
    "sae": wl_sae,
    "logit_lens": wl_logit_lens,
    "grad_capture": wl_grad_capture,
}


def _write_profile_summary(path: str) -> None:
    """Write a by-file and by-function aggregation of a pstats dump next to it."""

    import pstats
    from collections import defaultdict

    st = pstats.Stats(path)
    total = st.total_tt  # type: ignore[attr-defined]
    by_file: dict[str, float] = defaultdict(float)
    rows = []
    for (fn, line, name), (_cc, nc, tt, ct, _callers) in st.stats.items():  # type: ignore[attr-defined]
        key = fn.split("/site-packages/")[-1] if "/site-packages/" in fn else fn
        if "/torchlens/" in fn:
            key = "torchlens/" + fn.split("/torchlens/")[-1]
        by_file[key] += tt
        rows.append((tt, ct, nc, f"{key}:{line}({name})"))
    with open(path + ".txt", "w") as fh:
        fh.write(f"total tottime {total:.3f}s\n== by file (tottime)\n")
        for key, tt in sorted(by_file.items(), key=lambda kv: -kv[1])[:40]:
            fh.write(f"{tt:8.3f}s {100 * tt / total:5.1f}%  {key}\n")
        fh.write("== by function (tottime)\n")
        for tt, ct, nc, name in sorted(rows, reverse=True)[:60]:
            fh.write(f"{tt:8.3f}s tot {ct:8.3f}s cum {nc:9d} calls  {name}\n")
        fh.write("== by function (cumtime, torchlens only)\n")
        tl_rows = [r for r in rows if r[3].startswith("torchlens/")]
        for tt, ct, nc, name in sorted(tl_rows, key=lambda r: -r[1])[:60]:
            fh.write(f"{ct:8.3f}s cum {tt:8.3f}s tot {nc:9d} calls  {name}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(C.MODEL_IDS))
    p.add_argument("--tool", required=True)
    p.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    p.add_argument("--out", required=True)
    p.add_argument("--resdir", required=True)
    p.add_argument("--n-seqs", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patch-seq", type=int, default=16)
    p.add_argument("--patch-positions", type=int, default=4)
    p.add_argument("--new-tokens", type=int, default=16)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--label", default="")
    p.add_argument(
        "--profile", default="", help="cProfile one rep of the workload into this .pstats path"
    )
    a = p.parse_args()
    C.OUT_PATH = a.out
    C.REPS = a.reps
    os.makedirs(a.resdir, exist_ok=True)
    base = {
        "kind": "workload",
        "model": a.model,
        "tool": a.tool,
        "workload": a.workload,
        "label": a.label,
        "n_seqs": a.n_seqs,
        "seq_len": a.seq_len,
        "batch": a.batch,
        "patch_seq": a.patch_seq,
        "patch_positions": a.patch_positions,
        "new_tokens": a.new_tokens,
    }
    t0 = time.perf_counter()
    model, tok, fam = C.load_model(a.model)
    load_s = time.perf_counter() - t0
    rss_model = C.rss_mib()
    try:
        tool = make_tool(a.tool, model, tok, C.MODEL_IDS[a.model])
    except Skip as e:
        C.emit({**base, "skip": str(e), "rss_model_mib": rss_model})
        return
    rss_tool = C.rss_mib()
    C.emit(C.env_record(a.tool, {"model": a.model, "workload": a.workload, "load_s": load_s}))
    try:
        if a.profile:
            import cProfile

            C.REPS, C.WARMUP = 1, 1
            prof = cProfile.Profile()
            WORKLOADS[a.workload](tool, model, fam, a)  # warm (module imports, lazy wrappers)
            prof.enable()
            r = WORKLOADS[a.workload](tool, model, fam, a)
            prof.disable()
            prof.dump_stats(a.profile)
            _write_profile_summary(a.profile)
        else:
            r = WORKLOADS[a.workload](tool, model, fam, a)
    except Skip as e:
        C.emit({**base, "skip": str(e), "rss_model_mib": rss_model, "rss_tool_mib": rss_tool})
        return
    except Exception as e:  # noqa: BLE001
        import traceback

        C.emit(
            {
                **base,
                **C.err(e),
                "trace": traceback.format_exc()[-1500:],
                "rss_model_mib": rss_model,
            }
        )
        return
    res = r.pop("result")
    times = r.pop("times")
    torch.save(res, os.path.join(a.resdir, f"{a.model}.{a.workload}.{a.tool}.pt"))
    gc.collect()
    C.emit(
        {
            **base,
            **C.summarize(times),
            **r,
            "rss_model_mib": rss_model,
            "rss_tool_mib": rss_tool,
            "rss_end_mib": C.rss_mib(),
            "peak_rss_mib": C.peak_rss_mib(),
            "result_shape": list(res.shape),
        }
    )


if __name__ == "__main__":
    main()
