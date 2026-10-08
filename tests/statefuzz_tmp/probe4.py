# ruff: noqa
"""Differential probe 4: last-module fan-out, user hooks under repeated calls, frozen params, set() reruns."""

from __future__ import annotations

import copy
import gc
import json
import os
import sys
import traceback
import warnings
from typing import Any, Callable

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
import models  # noqa: E402
import snap  # noqa: E402

import torchlens as tl  # noqa: E402

torch.set_num_threads(1)


def emit(scenario: str, ok: bool | None, **detail: Any) -> None:
    print(json.dumps({"scenario": scenario, "ok": ok, **detail}, default=str), flush=True)


def md(a: Any, b: Any) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max())


def out_of(t: Any) -> torch.Tensor:
    return t.output_ops[0].out


def guarded(name: str, fn: Callable[[], None]) -> None:
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn()
        kinds = sorted({type(x.message).__name__ + ":" + str(x.message)[:120] for x in w})
        if kinds:
            emit(name + ":warnings", None, warnings=kinds[:8])
    except Exception as exc:  # noqa: BLE001
        emit(
            name,
            None,
            error=f"{type(exc).__name__}: {str(exc)[:400]}",
            tb=[ln.strip()[:160] for ln in traceback.format_exc().strip().splitlines()[-7:]],
        )


def gt_forward(
    model: nn.Module, site: str, fn: Callable[[torch.Tensor], torch.Tensor], x: Any
) -> torch.Tensor:
    h = model.get_submodule(site).register_forward_hook(lambda m, a, o: fn(o))
    try:
        with torch.no_grad():
            return model(x)
    finally:
        h.remove()


# K1: tl.module on the LAST module (its output is the model output)
def k1() -> None:
    for kind, site in (("mlp", "fc3"), ("mlp", "fc2"), ("decoder", "lm_head"), ("conv", "head")):
        model, x, x2, _, _ = models.build(kind)
        gt = gt_forward(model, site, lambda o: o * 0.5, x)
        ready = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
        sites = ready.find_sites(tl.module(site))
        labels = [getattr(s, "layer_label", str(s)) for s in sites]
        fk = ready.fork()
        fk.do(tl.module(site), tl.scale(0.5))
        t = tl.trace(model, x, intervene=tl.when(tl.module(site), tl.scale(0.5)))
        with torch.no_grad():
            b = tl.when(tl.module(site), tl.scale(0.5)).bind(model)(x)
        emit(
            f"K1_{kind}_{site}",
            None,
            n_sites=len(labels),
            labels=labels,
            fork_do_ok=md(out_of(fk), gt) < 1e-5,
            fork_d=md(out_of(fk), gt),
            fork_d_vs_quarter=md(out_of(fk), gt_forward(model, site, lambda o: o * 0.25, x)),
            trace_ok=md(out_of(t), gt) < 1e-5,
            trace_d=md(out_of(t), gt),
            bind_ok=md(b, gt) < 1e-5,
        )


# K2: user forward hooks / pre-hooks registered BEFORE TorchLens calls fire once per forward
def k2() -> None:
    model, x, x2, site, _ = models.build("mlp")
    counts = {"fwd": 0, "pre": 0, "root": 0}

    def fwd(m: nn.Module, a: Any, o: torch.Tensor) -> torch.Tensor:
        counts["fwd"] += 1
        return o + 1.0

    def pre(m: nn.Module, a: Any) -> Any:
        counts["pre"] += 1
        return (a[0] * 2.0,)

    def root(m: nn.Module, a: Any, o: torch.Tensor) -> None:
        counts["root"] += 1

    model.fc2.register_forward_hook(fwd)
    model.fc1.register_forward_pre_hook(pre)
    model.register_forward_hook(root)
    with torch.no_grad():
        clean = model(x)
    base = snap.model_state(model)
    res: dict[str, Any] = {}

    def run(label: str, fn: Callable[[], Any]) -> None:
        for k in counts:
            counts[k] = 0
        out = fn()
        o = out_of(out) if hasattr(out, "output_ops") else out
        res[label] = dict(counts, d=None if not torch.is_tensor(o) else md(o, clean))

    run("eager", lambda: model(x))
    for i in range(3):
        run(f"trace{i + 1}", lambda: tl.trace(model, x))
    t = tl.trace(model, x)
    for i in range(2):
        run(f"legacy_rerun{i + 1}", lambda: (t.run(model, x), t)[1])
    run("record", lambda: tl.record(model, x, default_op=True).to_trace())
    run("bind_identity", lambda: tl.when(tl.module(site), tl.scale(1.0)).bind(model)(x))
    run(
        "trace_intervene_identity",
        lambda: tl.trace(model, x, intervene=tl.when(tl.module(site), tl.scale(1.0))),
    )
    ready = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    run(
        "fork_do_identity",
        lambda: (lambda f: (f.do(tl.module(site), tl.scale(1.0)), f)[1])(ready.fork()),
    )
    try:
        run("validate", lambda: tl.validate(model, x, scope="forward"))
    except Exception as exc:  # noqa: BLE001
        res["validate"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    run("eager_after", lambda: model(x))
    ok = all(
        isinstance(v, dict) and v.get("fwd") in (1, 0) and v.get("pre") in (1, 0)
        for v in res.values()
    )
    emit(
        "K2_user_hooks_fire_once",
        ok and res["eager_after"] == res["eager"],
        res=res,
        state_diff=snap.diff(base, snap.model_state(model)),
    )


# K3: frozen params / requires_grad / grads preserved
def k3() -> None:
    model, x, x2, site, _ = models.build("mlp")
    model.fc1.weight.requires_grad_(False)
    model.fc3.bias.requires_grad_(False)
    tl.trace(model, x)
    base = snap.model_state(model)
    calls: list[tuple[str, Callable[[], Any]]] = [
        ("trace", lambda: tl.trace(model, x)),
        (
            "trace_backward_ready",
            lambda: tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True)),
        ),
        (
            "trace_intervene",
            lambda: tl.trace(model, x, intervene=tl.when(tl.module(site), tl.scale(0.5))),
        ),
        ("record", lambda: tl.record(model, x, default_op=True)),
        ("bind", lambda: tl.when(tl.module(site), tl.scale(0.5)).bind(model)(x)),
        ("validate_forward", lambda: tl.validate(model, x, scope="forward")),
        ("validate_backward", lambda: tl.validate(model, x, scope="backward")),
    ]
    t = tl.trace(model, x)
    calls.append(("legacy_rerun", lambda: t.run(model, x2)))
    ready = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    calls.append(("fork_do", lambda: ready.fork().do(tl.module(site), tl.scale(0.5))))
    for label, fn in calls:
        err = None
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:150]}"
        d = snap.diff(base, snap.model_state(model))
        emit(f"K3_params_{label}", not d, diff=d, err=err)


# K4: Trace.set value replacement then legacy reruns (value-spec accumulation)
def k4() -> None:
    model, x, x2, site, _ = models.build("mlp")
    t = tl.trace(model, x)
    sites = t.find_sites(tl.module(site))
    label = getattr(sites[0], "layer_label", None) if len(sites) else None
    val = torch.full_like(t[label].out, 0.3)
    t.set(label, val, confirm_mutation=True)
    gt = gt_forward(model, site, lambda o: torch.full_like(o, 0.3), x2)
    outs, nspecs = [], []
    for _ in range(3):
        t.run(model, x2)
        outs.append(md(out_of(t), gt))
        sp = t._ensure_intervention_spec()
        nspecs.append((len(sp.hook_specs), len(sp.target_value_specs)))
    emit("K4_set_then_rerun_x3", all(o < 1e-5 for o in outs), d=outs, specs=nspecs, label=label)


# K5: two Trace objects from one capture call path: tl.trace(intervene=) result vs its own fork,
# then fork.do on the intervened trace's fork (inherits capture spec?)
def k5() -> None:
    model, x, x2, site, _ = models.build("mlp")
    g = torch.Generator().manual_seed(7)
    d = torch.randn(16, generator=g)
    t = tl.trace(
        model,
        x,
        intervene=tl.when(tl.module(site), tl.steer(d, magnitude=1.0, feature_axis=-1)),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    f = t.fork()
    gt1 = gt_forward(model, site, lambda o: o + d, x2)
    f.run(model, x2)
    emit(
        "K5_fork_of_intervened_trace_rerun",
        md(out_of(f), gt1) < 1e-5,
        d_vs_1x=md(out_of(f), gt1),
        d_vs_2x=md(out_of(f), gt_forward(model, site, lambda o: o + 2 * d, x2)),
        hooks=len(f._ensure_intervention_spec().hook_specs),
    )
    f2 = t.fork()
    f2.do(tl.module("fc3"), tl.scale(1.0))
    gt0 = gt_forward(model, site, lambda o: o + d, x)
    emit(
        "K5_fork_do_on_intervened_trace",
        md(out_of(f2), gt0) < 1e-5,
        d_vs_1x=md(out_of(f2), gt0),
        d_vs_2x=md(out_of(f2), gt_forward(model, site, lambda o: o + 2 * d, x)),
        d_vs_0x=md(out_of(f2), gt_forward(model, site, lambda o: o, x)),
    )


if __name__ == "__main__":
    which = sys.argv[1:] or ["k1", "k2", "k3", "k4", "k5"]
    for name in which:
        guarded(name, globals()[name])
        gc.collect()
    print("DONE", flush=True)
