"""Private CPU interpretability workload benchmark; JSON lines, one cell per process."""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torchlens as tl


class _Model(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.readout = nn.Identity()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.readout(self.net(ids, use_cache=False).logits[:, -1])


def _build(name: str) -> tuple[_Model, list[str], int]:
    import transformers

    torch.manual_seed(0)
    if name == "gpt2":
        net = transformers.AutoModelForCausalLM.from_pretrained(
            "gpt2", attn_implementation="eager", local_files_only=True
        )
        sites = [f"net.transformer.h.{i}" for i in (0, 5, 11)]
        hidden = 768
    else:
        cfg = transformers.Qwen3Config(
            vocab_size=32768,
            hidden_size=256,
            intermediate_size=768,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
        )
        cfg._attn_implementation = "eager"
        net = transformers.Qwen3ForCausalLM(cfg)
        sites = [f"net.model.layers.{i}" for i in (0, 3, 7)]
        hidden = 256
    return _Model(net).eval(), sites, hidden


def _tensor(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def _replace(out: Any, value: torch.Tensor) -> Any:
    return (value, *out[1:]) if isinstance(out, tuple) else value


class _Runner:
    def __init__(self, model: _Model, sites: list[str], hidden: int, mode: str) -> None:
        self.model, self.sites, self.mode = model, sites, mode
        self.direction = torch.randn(hidden, generator=torch.Generator().manual_seed(11)) * 0.1
        self.selector = tl.module("readout")
        for site in sites:
            self.selector = self.selector | tl.module(site)
        self.n_ops = 0
        self.bytes = 0
        self.seed_trace = None
        self.seed_traces: dict[Any, Any] = {}
        self.compiled = None
        self.bound = None
        self.latest = None
        self.peer = None

    def call(
        self,
        ids: torch.Tensor,
        action: str = "none",
        strength: float = 1.0,
        patch: torch.Tensor | None = None,
        grad: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Essential complexity: this benchmark dispatches the same workload across capture contracts.
        self.latest = None
        if self.mode in ("nnsight", "tlens"):
            from _eff_peers import _Peers

            if self.peer is None:
                self.peer = _Peers(self)
            return self.peer.call(ids, action, strength, patch, grad)
        cache: dict[str, torch.Tensor] = {}
        handles = []

        def edit(out: Any) -> Any:
            value = _tensor(out)
            changed = value + self.direction * strength if action == "steer" else patch
            return _replace(out, changed)

        # Independent hook oracle is also used to observe bound/rerun output.
        if self.mode in ("hooks", "bind", "rerun"):
            if action != "none" and self.mode == "hooks":
                handles.append(
                    self.model.get_submodule(self.sites[1]).register_forward_hook(
                        lambda m, a, o: edit(o)
                    )
                )
            for site in self.sites:

                def hook(m: nn.Module, a: Any, out: Any, key: str = site) -> None:
                    value = _tensor(out)
                    if grad:
                        value.retain_grad()
                        cache[key] = value
                    else:
                        cache[key] = value.detach().clone()

                handles.append(self.model.get_submodule(site).register_forward_hook(hook))
        spec = None
        if action == "steer":
            spec = tl.when(
                tl.module(self.sites[1]),
                tl.steer(self.direction, magnitude=strength, feature_axis=-1),
            )
        elif action == "patch":
            spec = tl.when(tl.module(self.sites[1]), tl.replace_with(patch))
        context = contextlib.nullcontext() if grad or self.mode == "default" else torch.no_grad()
        product = None
        try:
            with context:
                if self.mode in ("selected", "tape_weak", "tape_strong"):
                    from _eff_capture import _SelectedCapture, _Tape

                    if grad or action != "none":
                        raise ValueError("prototype supports forward-only selected capture")
                    if self.compiled is None:
                        self.compiled = _SelectedCapture(
                            self.model, tuple(tl.module(s) for s in self.sites)
                        )
                    tape = (
                        _Tape(strong=self.mode == "tape_strong")
                        if self.mode != "selected"
                        else None
                    )
                    with self.compiled, tape if tape is not None else contextlib.nullcontext():
                        output = self.model(ids)
                    cache = {s: value for s, count, value in self.compiled.records}
                    self.n_ops = tape.count if tape is not None else 0
                    self.latest = tape
                elif self.mode == "hooks":
                    output = self.model(ids)
                elif self.mode == "bind":
                    output = (spec.bind(self.model) if spec else self.model)(ids)
                elif self.mode == "rerun":
                    if grad:
                        raise ValueError("rerun gradient capture not benchmarked")
                    key = (action, strength)
                    self.seed_trace = self.seed_traces.get(key)
                    if self.seed_trace is None:
                        self.seed_trace = tl.trace(
                            self.model, ids, save=self.selector, intervene=spec
                        )
                        self.seed_traces[key] = self.seed_trace
                    self.seed_trace.run(self.model, ids)
                    product = self.seed_trace
                    output = product.find_sites(tl.module("readout")).first().out
                elif self.mode == "record":
                    output, product = tl.record(
                        self.model,
                        ids,
                        save=self.selector,
                        intervene=spec,
                        return_output=True,
                        backward_ready=grad,
                        save_grads=grad,
                    )
                    # Read recording payloads directly, avoiding full postprocessing.
                    for record in product.records:
                        for site in self.sites:
                            if tl.module(site)(record.ctx) and record.ram_payload is not None:
                                cache[site] = record.ram_payload
                    self.n_ops = product.n_ops_completed
                else:
                    opts: dict[str, Any] = {"intervene": spec}
                    if self.mode != "default":
                        opts["save"] = self.selector
                    if self.mode == "inference":
                        if grad:
                            raise ValueError("inference_only refuses gradients")
                        opts["capture"] = tl.options.CaptureOptions(inference_only=True)
                    if grad:
                        opts["capture"] = tl.options.CaptureOptions(
                            backward_ready=True, save_grads=True
                        )
                        opts["save_mode"] = "reference"
                    product = tl.trace(self.model, ids, **opts)
                    output = product.find_sites(tl.module("readout")).first().out
                    cache = {s: product.find_sites(tl.module(s)).first().out for s in self.sites}
                    self.n_ops = len(product.ops)
                if grad:
                    if product is not None:
                        product.log_backward(output.sum())
                    else:
                        output.sum().backward()
                    if product is not None:
                        cooked = product.to_trace() if self.mode == "record" else product
                        cache = {
                            s: cooked.find_sites(tl.module(s)).first().grad_for(bwd=1)
                            for s in self.sites
                        }
                    else:
                        cache = {
                            s: v.grad.detach().clone()
                            for s, v in cache.items()
                            if v.grad is not None
                        }
                    if len(cache) != len(self.sites):
                        raise ValueError("requested gradients missing")
                result = output.detach().clone(), {s: v.detach().clone() for s, v in cache.items()}
                self.bytes = sum(v.numel() * v.element_size() for v in result[1].values())
                if product is not None:
                    self.latest = product
                return result
        finally:
            for handle in handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)


def _workload(
    runner: _Runner, name: str, ids: torch.Tensor, hidden: int, batches: int
) -> torch.Tensor:
    outputs = []
    encoder = torch.randn(hidden, 128, generator=torch.Generator().manual_seed(7))
    if name == "patch":
        _, donor = _Runner(runner.model, runner.sites, hidden, "hooks").call((ids + 1) % 1000)
        patch = donor[runner.sites[1]]
    else:
        patch = None
    current = ids
    for step in range(batches):
        action = (
            "steer" if name in ("steer", "generate") else "patch" if name == "patch" else "none"
        )
        out, cache = runner.call(
            current,
            action,
            1.0 if name == "generate" else float(step + 1),
            patch,
            name in ("grad", "attribution"),
        )
        if name == "generate":
            token = out.argmax(-1, keepdim=True)
            current = torch.cat((current, token), dim=-1)
            outputs.append(token.flatten())
        elif name == "sae":
            outputs.extend(torch.relu(v @ encoder).flatten() for v in cache.values())
        elif name == "lens":
            norm = (
                runner.model.net.transformer.ln_f
                if hasattr(runner.model.net, "transformer")
                else runner.model.net.model.norm
            )
            with torch.no_grad():
                outputs.extend(
                    runner.model.net.lm_head(norm(v[:, -1])).flatten() for v in cache.values()
                )
        elif name == "attribution":
            oracle = _Runner(runner.model, runner.sites, hidden, "hooks")
            _, clean = oracle.call((current + 1) % 1000)
            _, corrupt = oracle.call(current)
            outputs.extend(
                ((clean[s] - corrupt[s]) * cache[s]).sum().reshape(1) for s in runner.sites
            )
        elif name in ("steer", "patch"):
            outputs.append(out.flatten())
        else:
            outputs.append(out.flatten())
            outputs.extend(v.flatten() for v in cache.values())
        if name not in ("generate", "patch"):
            current = (ids + step + 1) % 1000
    return torch.cat(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--mode", default="hooks")
    parser.add_argument("--workload", default="cache")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--profile")
    args = parser.parse_args()
    torch.set_num_threads(1)
    import transformers

    env = {
        "model": args.model,
        "mode": args.mode,
        "workload": args.workload,
        "seq": args.seq,
        "batches": args.batches,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "threads": 1,
        "host": platform.node(),
        "cores": os.cpu_count(),
        "cpu": platform.processor(),
        "load": os.getloadavg(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    try:
        model, sites, hidden = _build(args.model)
        ids = torch.randint(0, 1000, (1, args.seq), generator=torch.Generator().manual_seed(2))
        oracle = _workload(
            _Runner(model, sites, hidden, "hooks"), args.workload, ids, hidden, args.batches
        )
        runner = _Runner(model, sites, hidden, args.mode)

        def fn() -> torch.Tensor:
            return _workload(runner, args.workload, ids, hidden, args.batches)

        fn()
        times = []
        diffs = []
        for _ in range(3):
            gc.collect()
            start = time.perf_counter()
            result = fn()
            times.append(time.perf_counter() - start)
            diffs.append(
                float((result - oracle).abs().max())
                if result.shape == oracle.shape
                else "shape_mismatch"
            )
        if args.profile:
            cProfile.runctx("fn()", globals(), locals(), args.profile)
        print(
            json.dumps(
                {
                    **env,
                    "status": "ok",
                    "times_s": times,
                    "median_s": statistics.median(times),
                    "min_s": min(times),
                    "max_s": max(times),
                    "max_abs_diff": diffs,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "n_ops": runner.n_ops,
                    "selected_bytes_last_batch": runner.bytes,
                }
            ),
            flush=True,
        )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(json.dumps({**env, "status": "error", "error": str(exc)}), flush=True)
        raise


if __name__ == "__main__":
    main()
