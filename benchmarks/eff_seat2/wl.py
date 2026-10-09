"""Mechanistic-interpretability workload harness: TorchLens paths against plain PyTorch hooks.

One process runs one (model, workload, tool) cell and prints one JSON line. Each cell runs one
warm-up and ``--reps`` timed repetitions and reports median, min and max wall time, the peak RSS
growth over the post-load baseline, the recorded op count when the tool produces a trace, and the
max abs difference of the workload's result against the plain-hook reference computed in the same
process.

Workloads (sequence length ``--seq``, batch 1, fp32, eager attention, KV cache off):

- ``cache``: cache the residual stream after every block for ``--prompts`` prompts
  (activation caching, logit lens and SAE inputs all reduce to this capture).
- ``steer``: steering sweep, ``--sweep`` magnitudes at the middle block, last-position logits.
- ``patch``: activation patching sweep, the clean middle-block output patched into the corrupted
  run at each block in turn (one forward per block).
- ``gen``: 8 greedy tokens with a steering vector at the middle block, re-running the full
  prefix per step (KV cache off).
- ``grad``: gradient of one logit with respect to every block output (attribution patching).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions

MAG = 4.0


class LastLogits(nn.Module):
    """Last-position logits with the KV cache off, plus an identity readout site."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network
        self.logit_readout = nn.Identity()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self.network(input_ids=ids, use_cache=False).logits[:, -1, :]
        return self.logit_readout(logits)


def build(name: str) -> tuple[nn.Module, list[str], int]:
    """Return the wrapped model, its block addresses and the hidden size."""
    torch.manual_seed(0)
    if name == "gpt2":
        from transformers import AutoModelForCausalLM

        net = AutoModelForCausalLM.from_pretrained(
            "gpt2", dtype=torch.float32, attn_implementation="eager"
        )
        n = net.config.n_layer
        prefix = "network.transformer.h"
        hidden = net.config.n_embd
    elif name == "q3s":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        cfg = Qwen3Config(
            vocab_size=32768,
            hidden_size=256,
            intermediate_size=768,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=4096,
            tie_word_embeddings=False,
        )
        cfg._attn_implementation = "eager"
        net = Qwen3ForCausalLM(cfg)
        n = cfg.num_hidden_layers
        prefix = "network.model.layers"
        hidden = cfg.hidden_size
    else:
        raise ValueError(name)
    model = LastLogits(net).eval()
    return model, [f"{prefix}.{i}" for i in range(n)], hidden


def first_tensor(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def rec_site_out(rec: Any, addr: str) -> torch.Tensor:
    """Last saved payload recorded at a module address in a ``Recording``."""
    idx = rec.by_address[addr][-1]
    return rec.records[idx].ram_payload


def maxdiff(a: Any, b: Any) -> float:
    if isinstance(a, (list, tuple)):
        return max(maxdiff(x, y) for x, y in zip(a, b, strict=True))
    return float((a.detach().float() - b.detach().float()).abs().max())


class Cell:
    """Holds the model, sites and inputs shared by every tool for one workload."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model, self.blocks, hidden = build(args.model)
        self.mid = self.blocks[len(self.blocks) // 2]
        gen = torch.Generator().manual_seed(1)
        self.direction = torch.randn(hidden, generator=gen)
        self.prompts = [
            torch.randint(0, 1000, (1, args.seq), generator=torch.Generator().manual_seed(100 + i))
            for i in range(args.prompts)
        ]
        self.ids = self.prompts[0]
        self.corrupt = torch.randint(0, 1000, (1, args.seq), generator=gen)
        self.mags = [MAG * (k + 1) / args.sweep for k in range(args.sweep)]
        self.last_ops: int | None = None
        self.last_engine: Any = None

    # ------------------------------------------------------------------ plain hooks
    def _hook_cache(self, ids: torch.Tensor, grad: bool = False) -> list[torch.Tensor]:
        store: list[torch.Tensor] = []

        def save(_m: nn.Module, _a: Any, out: Any) -> None:
            h = first_tensor(out)
            if grad:
                h.retain_grad()
                store.append(h)
            else:
                store.append(h.detach().clone())

        handles = [self.model.get_submodule(b).register_forward_hook(save) for b in self.blocks]
        try:
            if grad:
                logits = self.model(ids)
                logits[0, 7].backward()
                return [h.grad.detach().clone() for h in store]
            with torch.no_grad():
                self.model(ids)
            return store
        finally:
            for h in handles:
                h.remove()

    def _hook_edit(self, ids: torch.Tensor, addr: str, edit: Callable[[torch.Tensor], Any]) -> Any:
        def fn(_m: nn.Module, _a: Any, out: Any) -> Any:
            new = edit(first_tensor(out))
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

        handle = self.model.get_submodule(addr).register_forward_hook(fn)
        try:
            with torch.no_grad():
                return self.model(ids)
        finally:
            handle.remove()

    def steer_edit(self, mag: float) -> Callable[[torch.Tensor], torch.Tensor]:
        d = self.direction
        return lambda h: h + d.to(h.dtype) * mag

    # ------------------------------------------------------------------ peers
    def _peer(self, tool: str) -> Any:
        cached = getattr(self, f"_peer_{tool}", None)
        if cached is not None:
            return cached
        if tool == "tlens":
            if self.args.model != "gpt2":
                raise NotImplementedError("TransformerLens runs pretrained configs only (gpt2)")
            from transformer_lens import HookedTransformer

            obj = HookedTransformer.from_pretrained("gpt2", device="cpu")
            obj.eval()
        else:
            from nnsight import NNsight

            obj = NNsight(self.model)
        setattr(self, f"_peer_{tool}", obj)
        return obj

    def _tlens_name(self, addr: str) -> str:
        return f"blocks.{addr.rsplit('.', 1)[1]}.hook_resid_post"

    def _nn_module(self, nm: Any, addr: str) -> Any:
        obj = nm
        for part in addr.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return obj

    def peer_cache(self, tool: str, ids: torch.Tensor) -> list[torch.Tensor]:
        peer = self._peer(tool)
        if tool == "tlens":
            names = {self._tlens_name(b) for b in self.blocks}
            with torch.no_grad():
                _, cache = peer.run_with_cache(ids, names_filter=lambda n: n in names)
            return [cache[self._tlens_name(b)] for b in self.blocks]
        saved = []
        with torch.no_grad(), peer.trace(ids):
            for b in self.blocks:
                saved.append(self._nn_module(peer, b).output[0].save())
        return [s if isinstance(s, torch.Tensor) else s.value for s in saved]

    def peer_edit(
        self, tool: str, ids: torch.Tensor, addr: str, edit: Callable[[torch.Tensor], Any]
    ) -> torch.Tensor:
        peer = self._peer(tool)
        if tool == "tlens":

            def fn(h: torch.Tensor, hook: Any) -> torch.Tensor:
                return edit(h)

            with torch.no_grad():
                logits = peer.run_with_hooks(ids, fwd_hooks=[(self._tlens_name(addr), fn)])
            return logits[:, -1, :]
        with torch.no_grad(), peer.trace(ids):
            mod = self._nn_module(peer, addr)
            mod.output[0][:] = edit(mod.output[0])
            out = peer.output.save()
        return out if isinstance(out, torch.Tensor) else out.value

    # ------------------------------------------------------------------ workloads
    def run(self, workload: str, tool: str) -> Any:
        return getattr(self, f"w_{workload}")(tool)

    def w_cache(self, tool: str) -> Any:
        out = []
        for ids in self.prompts:
            if tool == "hook":
                out.append(self._hook_cache(ids))
                continue
            sites = [tl.module(b) for b in self.blocks]
            sel = sites[0]
            for s in sites[1:]:
                sel = sel | s
            if tool == "tl_trace_default":
                t = tl.trace(self.model, ids)
            elif tool == "tl_trace_infer":
                t = tl.trace(self.model, ids, capture=CaptureOptions(inference_only=True))
            elif tool == "tl_trace_sparse":
                t = tl.trace(self.model, ids, save=sel)
            elif tool == "tl_trace_sparse_infer":
                t = tl.trace(self.model, ids, save=sel, capture=CaptureOptions(inference_only=True))
            elif tool == "tl_record":
                rec = tl.record(self.model, ids, save=sel)
                self.last_ops = getattr(rec, "n_ops", None)
                out.append([rec_site_out(rec, b) for b in self.blocks])
                continue
            elif tool in ("tlens", "nnsight"):
                out.append(self.peer_cache(tool, ids))
                continue
            else:
                raise ValueError(tool)
            self.last_ops = len(t.ops)
            out.append([t.find_sites(s).first().out for s in sites])
        return out

    def w_steer(self, tool: str) -> Any:
        res = []
        site = tl.module(self.mid)
        for mag in self.mags:
            spec = tl.when(site, tl.steer(self.direction, magnitude=mag, feature_axis=-1))
            if tool == "hook":
                res.append(self._hook_edit(self.ids, self.mid, self.steer_edit(mag)))
            elif tool == "tl_bind":
                with torch.no_grad():
                    res.append(spec.bind(self.model)(self.ids))
            elif tool == "tl_record":
                out = tl.record(self.model, self.ids, save=site, intervene=spec, return_output=True)
                res.append(out[0])
            elif tool in ("tlens", "nnsight"):
                res.append(self.peer_edit(tool, self.ids, self.mid, self.steer_edit(mag)))
            elif tool == "tl_trace":
                t = tl.trace(
                    self.model,
                    self.ids,
                    save=site | tl.module("logit_readout"),
                    intervene=spec,
                    capture=CaptureOptions(inference_only=True),
                )
                self.last_ops = len(t.ops)
                res.append(t.find_sites(tl.module("logit_readout")).first().out)
            else:
                raise ValueError(tool)
        return res

    def w_patch(self, tool: str) -> Any:
        clean = self._hook_cache(self.ids)
        res = []
        for i, addr in enumerate(self.blocks):
            value = clean[i]
            if tool == "hook":
                res.append(self._hook_edit(self.corrupt, addr, lambda h, v=value: v))
                continue
            if tool in ("tlens", "nnsight"):
                res.append(self.peer_edit(tool, self.corrupt, addr, lambda h, v=value: v))
                continue
            site = tl.module(addr)
            spec = tl.when(site, tl.replace_with(value))
            if tool == "tl_bind":
                with torch.no_grad():
                    res.append(spec.bind(self.model)(self.corrupt))
            elif tool == "tl_record":
                out = tl.record(
                    self.model, self.corrupt, save=site, intervene=spec, return_output=True
                )
                res.append(out[0])
            else:
                raise ValueError(tool)
        return res

    def w_gen(self, tool: str) -> Any:
        ids = self.ids
        site = tl.module(self.mid)
        spec = tl.when(site, tl.steer(self.direction, magnitude=MAG, feature_axis=-1))
        bound = spec.bind(self.model) if tool == "tl_bind" else None
        seed: Any = None
        logit_site = tl.module("logit_readout")
        for _ in range(8):
            if tool == "tl_fastrun":
                # trace once, then the guarded fast steered rerun (wip/fast-rerun-intervene)
                if seed is None:
                    seed = tl.trace(
                        self.model,
                        ids,
                        save=site | logit_site,
                        intervene=spec,
                        capture=CaptureOptions(inference_only=True),
                    )
                else:
                    seed.run(self.model, ids)
                    self.last_engine = seed.last_run.get("engine")
                logits = seed.find_sites(logit_site).first().out
            elif tool == "hook":
                logits = self._hook_edit(ids, self.mid, self.steer_edit(MAG))
            elif tool == "tl_bind":
                with torch.no_grad():
                    logits = bound(ids)
            elif tool in ("tlens", "nnsight"):
                logits = self.peer_edit(tool, ids, self.mid, self.steer_edit(MAG))
            elif tool == "tl_record":
                logits = tl.record(self.model, ids, save=site, intervene=spec, return_output=True)[
                    0
                ]
            elif tool == "tl_trace":
                t = tl.trace(
                    self.model,
                    ids,
                    save=site | tl.module("logit_readout"),
                    intervene=spec,
                    capture=CaptureOptions(inference_only=True),
                )
                logits = t.find_sites(tl.module("logit_readout")).first().out
            else:
                raise ValueError(tool)
            ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=1)
        return ids.float()

    def w_grad(self, tool: str) -> Any:
        if tool == "hook":
            return self._hook_cache(self.ids, grad=True)
        if tool != "tl_trace_grad":
            raise ValueError(tool)
        t = tl.trace(
            self.model,
            self.ids,
            capture=CaptureOptions(backward_ready=True, save_grads=True),
            save_mode="reference",
        )
        self.last_ops = len(t.ops)
        out = t[t.output_layers[0]].out
        t.backward(out[0, 7])
        return [t.find_sites(tl.module(b)).first().grad for b in self.blocks]


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--workload", default="cache")
    ap.add_argument("--tool", default="hook")
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--sweep", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--commit", default=os.environ.get("TL_COMMIT", "?"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    cell = Cell(args)
    reference = cell.run(args.workload, "hook")
    gc.collect()
    base_rss = rss_mb()
    rec: dict[str, Any] = {
        "model": args.model,
        "workload": args.workload,
        "tool": args.tool,
        "seq": args.seq,
        "prompts": args.prompts,
        "sweep": args.sweep,
        "commit": args.commit,
        "host": platform.node(),
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "loadavg_start": os.getloadavg()[0],
    }
    try:
        import transformers

        rec["transformers"] = transformers.__version__
        result = cell.run(args.workload, args.tool)  # warm-up
        del result
        gc.collect()
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            result = cell.run(args.workload, args.tool)
            times.append(time.perf_counter() - t0)
        rec.update(
            median_s=statistics.median(times),
            min_s=min(times),
            max_s=max(times),
            reps=len(times),
            ops=cell.last_ops,
            peak_rss_growth_mb=rss_mb() - base_rss,
            maxdiff_vs_hook=maxdiff(result, reference),
            engine=cell.last_engine,
        )
    except Exception as exc:  # report the refusal or failure as data
        rec.update(error=type(exc).__name__, msg=str(exc)[:500])
    line = json.dumps(rec)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    sys.exit(main())
