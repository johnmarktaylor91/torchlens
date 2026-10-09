"""Tool adapters: the same four primitives spelled with each tool.

cache(ids, sites)                 -> {site: tensor}  (activations at module exits)
forward_add(ids, site, delta)     -> logits          (one additive intervention; steer and patch)
grads(ids, sites, metric)         -> {site: (act, grad)}  (activations and their gradients)
generate(ids, site, delta, n, sample) -> token ids  (greedy or sampled with an intervention)

Every adapter works on the same eager HF model instance, so hooks are the exactness reference.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from typing import Any

import torch
from torch import nn

from .common import first_tensor


class Skip(RuntimeError):
    """A tool cannot run this primitive on this model; the reason is the message."""


# ----------------------------------------------------------------------------- plain hooks
class Hooks:
    name = "hooks"

    def __init__(self, model: nn.Module, tok: Any) -> None:
        self.model = model
        self.tok = tok

    def _mods(self, sites: list[str]) -> list[nn.Module]:
        return [self.model.get_submodule(s) for s in sites]

    def cache(self, ids: torch.Tensor, sites: list[str]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        handles = []
        for s, m in zip(sites, self._mods(sites)):
            handles.append(
                m.register_forward_hook(
                    lambda _m, _i, o, s=s: out.__setitem__(s, first_tensor(o).detach())
                )
            )
        try:
            with torch.no_grad():
                self.model(input_ids=ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return out

    @staticmethod
    def _add_hook(delta: torch.Tensor) -> Callable[..., Any]:
        def hook(_m: nn.Module, _i: Any, o: Any) -> Any:
            if isinstance(o, tuple):
                return (o[0] + delta.to(o[0].dtype),) + tuple(o[1:])
            return o + delta.to(o.dtype)

        return hook

    def forward_add(self, ids: torch.Tensor, site: str, delta: torch.Tensor) -> torch.Tensor:
        h = self.model.get_submodule(site).register_forward_hook(self._add_hook(delta))
        try:
            with torch.no_grad():
                return self.model(input_ids=ids, use_cache=False).logits
        finally:
            h.remove()

    def grads(
        self, ids: torch.Tensor, sites: list[str], metric: Callable[[torch.Tensor], torch.Tensor]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        acts: dict[str, torch.Tensor] = {}
        handles = []
        for s, m in zip(sites, self._mods(sites)):

            def hook(_m: nn.Module, _i: Any, o: Any, s=s) -> None:
                t = first_tensor(o)
                t.retain_grad()
                acts[s] = t

            handles.append(m.register_forward_hook(hook))
        try:
            logits = self.model(input_ids=ids, use_cache=False).logits
            metric(logits).backward()
        finally:
            for h in handles:
                h.remove()
        res = {s: (t.detach(), t.grad.detach()) for s, t in acts.items()}
        self.model.zero_grad(set_to_none=True)
        return res

    def generate(
        self, ids: torch.Tensor, site: str, delta: torch.Tensor, n_new: int, sample: bool
    ) -> torch.Tensor:
        h = self.model.get_submodule(site).register_forward_hook(self._add_hook(delta))
        try:
            with torch.no_grad():
                torch.manual_seed(0)
                return self.model.generate(
                    ids,
                    max_new_tokens=n_new,
                    do_sample=sample,
                    top_k=0 if sample else None,
                    temperature=1.0 if sample else None,
                    use_cache=True,
                    pad_token_id=self.tok.eos_token_id or 0,
                )
        finally:
            h.remove()


# ----------------------------------------------------------------------------- torchlens
def _tl():
    import torchlens as tl

    return tl


def _union(tl: Any, sites: list[str]) -> Any:
    return reduce(lambda a, b: a | b, (tl.module(s) for s in sites))


def _site_out(trace: Any, tl: Any, site: str) -> torch.Tensor:
    return first_tensor(trace.find_sites(tl.module(site)).first().out)


class TLTrace:
    """tl.trace with a sparse save= selector and inference_only (the fast exact trace spelling)."""

    name = "tl_trace"

    def __init__(self, model: nn.Module, tok: Any, *, default_save: bool = False) -> None:
        self.model = model
        self.tok = tok
        self.default_save = default_save
        self.tl = _tl()
        from torchlens.options import CaptureOptions

        self.CaptureOptions = CaptureOptions

    def _opts(self, **kw: Any) -> Any:
        return self.CaptureOptions(inference_only=True, unwrap_when_done=False, **kw)

    def _trace(self, ids: torch.Tensor, sites: list[str], **kw: Any) -> Any:
        tl = self.tl
        if self.default_save:
            return tl.trace(
                self.model, ids, capture=self.CaptureOptions(unwrap_when_done=False), **kw
            )
        return tl.trace(self.model, ids, save=_union(tl, sites), capture=self._opts(), **kw)

    def cache(self, ids: torch.Tensor, sites: list[str]) -> dict[str, torch.Tensor]:
        t = self._trace(ids, sites)
        return {s: _site_out(t, self.tl, s) for s in sites}

    def forward_add(self, ids: torch.Tensor, site: str, delta: torch.Tensor) -> torch.Tensor:
        tl = self.tl
        spec = tl.when(tl.module(site), tl.add(delta))
        t = tl.trace(
            self.model,
            ids,
            save=tl.module("lm_head") | tl.module("embed_out"),
            intervene=spec,
            capture=self._opts(),
        )
        return _logits_from_trace(t, tl)

    def grads(self, ids, sites, metric):
        tl = self.tl
        t = tl.trace(
            self.model,
            ids,
            save=_union(tl, sites + ["lm_head", "embed_out"]),
            capture=self.CaptureOptions(backward_ready=True, unwrap_when_done=False),
            save_mode="reference",
        )
        logits = _logits_from_trace(t, tl)
        t.log_backward(metric(logits))
        res = {}
        for s in sites:
            op = t.find_sites(tl.module(s)).first()
            res[s] = (first_tensor(op.out).detach(), first_tensor(op.grad).detach())
        self.model.zero_grad(set_to_none=True)
        return res

    def generate(self, ids, site, delta, n_new, sample):
        raise Skip("tl.trace has no generation loop; see tl_record (per-step) and tl_bind")


def _logits_from_trace(t: Any, tl: Any) -> torch.Tensor:
    for head in ("lm_head", "embed_out"):
        try:
            return first_tensor(t.find_sites(tl.module(head)).first().out)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("no unembed site saved")


class TLRecord:
    """tl.record: the sparse event recorder, with return_output for logits."""

    name = "tl_record"

    def __init__(self, model: nn.Module, tok: Any) -> None:
        self.model = model
        self.tok = tok
        self.tl = _tl()

    def _rec(self, ids: torch.Tensor, sites: list[str], **kw: Any) -> Any:
        tl = self.tl
        return tl.record(self.model, ids, save=_union(tl, sites), return_output=True, **kw)

    def _get(self, rec: Any, site: str) -> torch.Tensor:
        tl = self.tl
        r = rec[0] if isinstance(rec, tuple) else rec
        for getter in (lambda: r.find_sites(tl.module(site)).first().out, lambda: r[site].out):
            try:
                return first_tensor(getter())
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError(f"record: cannot read {site}")

    def cache(self, ids, sites):
        rec = self._rec(ids, sites)
        return {s: self._get(rec, s) for s in sites}

    def forward_add(self, ids, site, delta):
        tl = self.tl
        spec = tl.when(tl.module(site), tl.add(delta))
        rec, out = tl.record(
            self.model, ids, save=tl.module(site), intervene=spec, return_output=True
        )
        return first_tensor(out.logits if hasattr(out, "logits") else out)

    def grads(self, ids, sites, metric):
        raise Skip(
            "tl.record has no backward capture in this comparison (Recording.log_backward needs a cooked trace)"
        )

    def generate(self, ids, site, delta, n_new, sample):
        tl = self.tl
        spec = tl.when(tl.module(site), tl.add(delta))
        torch.manual_seed(0)
        cur = ids
        for _ in range(n_new):
            _rec, out = tl.record(
                self.model, cur, save=tl.module(site), intervene=spec, return_output=True
            )
            logits = (out.logits if hasattr(out, "logits") else first_tensor(out))[:, -1, :]
            nxt = (
                torch.multinomial(torch.softmax(logits, -1), 1)
                if sample
                else logits.argmax(-1, keepdim=True)
            )
            cur = torch.cat([cur, nxt], dim=1)
        return cur


class TLBind:
    """spec.bind(model): the capture-free bound executor (interventions only)."""

    name = "tl_bind"

    def __init__(self, model: nn.Module, tok: Any) -> None:
        self.model = model
        self.tok = tok
        self.tl = _tl()

    def cache(self, ids, sites):
        raise Skip("bind records nothing")

    def forward_add(self, ids, site, delta):
        tl = self.tl
        b = tl.when(tl.module(site), tl.add(delta)).bind(self.model)
        with torch.no_grad():
            return b(input_ids=ids, use_cache=False).logits

    def grads(self, ids, sites, metric):
        raise Skip("bind records nothing")

    def generate(self, ids, site, delta, n_new, sample):
        tl = self.tl
        b = tl.when(tl.module(site), tl.add(delta)).bind(self.model)
        torch.manual_seed(0)
        with torch.no_grad():
            return b.generate(
                ids,
                max_new_tokens=n_new,
                do_sample=sample,
                top_k=0 if sample else None,
                temperature=1.0 if sample else None,
                use_cache=True,
                pad_token_id=self.tok.eos_token_id or 0,
            )


class TLFastRerun:
    """Trace once with the intervention staged, then trace.run(inputs=..., fast=True) per call.

    Only on branches that ship the fast steered rerun; elsewhere the first run refuses and we Skip.
    """

    name = "tl_fast_rerun"

    def __init__(self, model: nn.Module, tok: Any) -> None:
        self.model = model
        self.tok = tok
        self.tl = _tl()
        from torchlens.options import CaptureOptions

        self.CaptureOptions = CaptureOptions
        self._seed: dict[tuple[str, int], Any] = {}

    def _trace_for(self, ids: torch.Tensor, site: str, delta: torch.Tensor) -> Any:
        tl = self.tl
        key = (site, int(delta.data_ptr()))
        if key not in self._seed:
            spec = tl.when(tl.module(site), tl.add(delta))
            self._seed[key] = tl.trace(
                self.model,
                ids,
                save=tl.module("lm_head") | tl.module("embed_out"),
                intervene=spec,
                capture=self.CaptureOptions(inference_only=True, unwrap_when_done=False),
            )
        return self._seed[key]

    def cache(self, ids, sites):
        raise Skip("fast rerun is an intervention path")

    def forward_add(self, ids, site, delta):
        tl = self.tl
        t = self._trace_for(ids, site, delta)
        try:
            out = t.run(inputs=ids, fast=True)
        except Exception as e:  # noqa: BLE001
            raise Skip(f"fast rerun refused: {type(e).__name__}: {str(e)[:200]}") from e
        refused = (
            getattr(t, "last_run", {}).get("fast_refused")
            if isinstance(getattr(t, "last_run", None), dict)
            else None
        )
        if refused:
            raise Skip(f"fast rerun fell back: {refused}")
        return _logits_from_trace(out, tl) if not isinstance(out, torch.Tensor) else out

    def grads(self, ids, sites, metric):
        raise Skip("fast rerun is an intervention path")

    def generate(self, ids, site, delta, n_new, sample):
        torch.manual_seed(0)
        cur = ids
        for _ in range(n_new):
            logits = self.forward_add(cur, site, delta)[:, -1, :]
            nxt = (
                torch.multinomial(torch.softmax(logits, -1), 1)
                if sample
                else logits.argmax(-1, keepdim=True)
            )
            cur = torch.cat([cur, nxt], dim=1)
        return cur


# ----------------------------------------------------------------------------- nnsight
class NNsight:
    name = "nnsight"

    def __init__(self, model: nn.Module, tok: Any) -> None:
        try:
            from nnsight import LanguageModel
        except Exception as e:  # noqa: BLE001
            raise Skip(f"nnsight not importable: {e}") from e
        self.lm = LanguageModel(model, tokenizer=tok)
        self.model = model
        self.tok = tok

    def _env(self, site: str):
        env = self.lm
        for part in site.split("."):
            env = env[int(part)] if part.isdigit() else getattr(env, part)
        return env

    def _inp(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def cache(self, ids, sites):
        saved = {}
        with torch.no_grad(), self.lm.trace(self._inp(ids), use_cache=False):
            for s in sites:
                saved[s] = self._env(s).output.save()
        return {s: first_tensor(v) for s, v in saved.items()}

    def forward_add(self, ids, site, delta):
        with torch.no_grad(), self.lm.trace(self._inp(ids), use_cache=False):
            env = self._env(site)
            out = env.output
            if isinstance(getattr(out, "shape", None), torch.Size):
                env.output = out + delta
            else:
                env.output[0][:] = out[0] + delta
            logits = self.lm.output.logits.save()
        return logits

    def grads(self, ids, sites, metric):
        saved = {}
        with self.lm.trace(self._inp(ids), use_cache=False):
            for s in sites:
                env = self._env(s)
                o = (
                    env.output
                    if isinstance(getattr(env.output, "shape", None), torch.Size)
                    else env.output[0]
                )
                saved[s] = (o.save(), o.grad.save())
            metric(self.lm.output.logits).backward()
        res = {s: (a.detach(), g.detach()) for s, (a, g) in saved.items()}
        self.model.zero_grad(set_to_none=True)
        return res

    def generate(self, ids, site, delta, n_new, sample):
        torch.manual_seed(0)
        with (
            torch.no_grad(),
            self.lm.generate(
                self._inp(ids),
                max_new_tokens=n_new,
                do_sample=sample,
                top_k=0 if sample else None,
                temperature=1.0 if sample else None,
                pad_token_id=self.tok.eos_token_id or 0,
            ),
        ):
            env = self._env(site)
            with env.all():
                out = env.output
                if isinstance(getattr(out, "shape", None), torch.Size):
                    env.output = out + delta
                else:
                    env.output[0][:] = out[0] + delta
            tokens = self.lm.generator.output.save()
        return tokens


# ----------------------------------------------------------------------------- TransformerLens
class TLens:
    """TransformerLens reimplements the model; speed and memory only, no bit equality."""

    name = "tlens"

    def __init__(self, model: nn.Module, tok: Any, model_id: str) -> None:
        try:
            from transformer_lens import HookedTransformer
        except Exception as e:  # noqa: BLE001
            raise Skip(f"transformer_lens not importable: {e}") from e
        try:
            self.ht = HookedTransformer.from_pretrained_no_processing(
                model_id, hf_model=model, tokenizer=tok, device="cpu", dtype=torch.float32
            )
        except Exception as e:  # noqa: BLE001
            raise Skip(
                f"TransformerLens cannot load {model_id}: {type(e).__name__}: {str(e)[:200]}"
            ) from e
        self.ht.eval()
        self.tok = tok

    @staticmethod
    def _hook_name(site: str) -> str:
        # site strings for TL are "L<i>" (resid_post) or "A<i>" (attn_out)
        kind, i = site[0], int(site[1:])
        return f"blocks.{i}.hook_resid_post" if kind == "L" else f"blocks.{i}.hook_attn_out"

    def cache(self, ids, sites):
        names = {self._hook_name(s): s for s in sites}
        with torch.no_grad():
            _, cache = self.ht.run_with_cache(ids, names_filter=lambda n: n in names)
        return {names[n]: cache[n] for n in names}

    def forward_add(self, ids, site, delta):
        def hook(act, hook):
            return act + delta.to(act.dtype)

        with torch.no_grad(), self.ht.hooks(fwd_hooks=[(self._hook_name(site), hook)]):
            return self.ht(ids)

    def grads(self, ids, sites, metric):
        acts = {}

        def mk(s):
            def hook(act, hook):
                act.retain_grad()
                acts[s] = act
                return act

            return hook

        with self.ht.hooks(fwd_hooks=[(self._hook_name(s), mk(s)) for s in sites]):
            logits = self.ht(ids)
            metric(logits).backward()
        res = {s: (a.detach(), a.grad.detach()) for s, a in acts.items()}
        self.ht.zero_grad(set_to_none=True)
        return res

    def generate(self, ids, site, delta, n_new, sample):
        def hook(act, hook):
            return act + delta.to(act.dtype)

        torch.manual_seed(0)
        with torch.no_grad(), self.ht.hooks(fwd_hooks=[(self._hook_name(site), hook)]):
            return self.ht.generate(
                ids,
                max_new_tokens=n_new,
                do_sample=sample,
                temperature=1.0 if sample else 0.0,
                use_past_kv_cache=True,
                verbose=False,
                stop_at_eos=False,
            )


def make_tool(name: str, model: nn.Module, tok: Any, model_id: str) -> Any:
    if name == "hooks":
        return Hooks(model, tok)
    if name == "tl_trace":
        return TLTrace(model, tok)
    if name == "tl_trace_default":
        t = TLTrace(model, tok, default_save=True)
        t.name = "tl_trace_default"
        return t
    if name == "tl_record":
        return TLRecord(model, tok)
    if name == "tl_bind":
        return TLBind(model, tok)
    if name == "tl_fast_rerun":
        return TLFastRerun(model, tok)
    if name == "nnsight":
        return NNsight(model, tok)
    if name == "tlens":
        return TLens(model, tok, model_id)
    raise ValueError(name)
