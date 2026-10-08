# ruff: noqa
"""Differential probe 3: deepcopy aliasing, RNG detail, fork stacking, set/attach reruns, pass scope."""

from __future__ import annotations

import copy
import gc
import json
import os
import pickle
import sys
import traceback
import warnings
import weakref
from typing import Any, Callable

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
import models  # noqa: E402

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
        kinds = sorted({type(x.message).__name__ + ":" + str(x.message)[:100] for x in w})
        if kinds:
            emit(name + ":warnings", None, warnings=kinds[:8])
    except Exception as exc:  # noqa: BLE001
        emit(
            name,
            None,
            error=f"{type(exc).__name__}: {str(exc)[:400]}",
            tb=[ln.strip()[:160] for ln in traceback.format_exc().strip().splitlines()[-7:]],
        )


# G: deepcopy of a traced model aliases the ORIGINAL's submodule forwards
def g() -> None:
    for prep in ("trace", "trace_then_release", "record", "bind", "none"):
        torch.manual_seed(0)
        model = models.MLP().eval()
        x = torch.randn(3, 8)
        if prep in ("trace", "trace_then_release"):
            tl.trace(model, x)
            if prep == "trace_then_release":
                tl.release_model(model)
        elif prep == "record":
            tl.record(model, x, default_op=True)
        elif prep == "bind":
            tl.when(tl.module("fc2"), tl.scale(1.0)).bind(model)(x)
        other = copy.deepcopy(model)
        with torch.no_grad():
            for p in other.parameters():
                p.mul_(2.0)
        fresh = models.MLP().eval()
        fresh.load_state_dict(other.state_dict())
        with torch.no_grad():
            got = other(x)
            want = fresh(x)
            orig = model(x)
        emit(
            f"G_deepcopy_eager_after_{prep}",
            md(got, want) == 0.0,
            d_vs_true=md(got, want),
            d_vs_original=md(got, orig),
            copy_fc1_forward_in_dict="forward" in vars(other.fc1),
        )
        # state_dict round trip into the ORIGINAL changes the copy?
    # pickle round trip of a traced model (torch.save path)
    torch.manual_seed(0)
    model = models.MLP().eval()
    x = torch.randn(3, 8)
    tl.trace(model, x)
    try:
        blob = pickle.dumps(model)
        emit("G_pickle_traced_model", None, pickled=len(blob))
    except Exception as exc:  # noqa: BLE001
        emit("G_pickle_traced_model", None, error=f"{type(exc).__name__}: {str(exc)[:200]}")


# H: RNG detail
class DropNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


def h() -> None:
    torch.manual_seed(0)
    model = DropNet().train()
    det = models.MLP().eval()
    x = torch.randn(4, 8)
    tl.trace(model, x)
    tl.trace(det, x)

    def after(seed: int, fn: Callable[[], Any]) -> tuple[Any, torch.Tensor]:
        torch.manual_seed(seed)
        r = fn()
        return r, torch.rand(4)

    _, nofwd = after(42, lambda: None)
    e1, one = after(42, lambda: model(x))
    _, two = after(42, lambda: (model(x), model(x)))
    t, tla = after(42, lambda: tl.trace(model, x))
    emit(
        "H_rng_trace_vs_eager",
        torch.equal(tla, one),
        next_eq_one_eager=torch.equal(tla, one),
        next_eq_no_forward=torch.equal(tla, nofwd),
        next_eq_two_eager=torch.equal(tla, two),
        out_eq_eager=torch.equal(out_of(t), e1),
    )
    # does a TorchLens capture's output equal eager output from the same seed, under each door
    torch.manual_seed(42)
    e = model(x)
    for label, fn in (
        ("trace", lambda: out_of(tl.trace(model, x))),
        (
            "trace_intervene_identity",
            lambda: out_of(tl.trace(model, x, intervene=tl.when(tl.module("fc2"), tl.scale(1.0)))),
        ),
        ("record_to_trace", lambda: out_of(tl.record(model, x, default_op=True).to_trace())),
        ("bind_identity", lambda: tl.when(tl.module("fc2"), tl.scale(1.0)).bind(model)(x)),
    ):
        torch.manual_seed(42)
        o = fn()
        emit(f"H_rng_out_{label}", torch.equal(o, e), d=md(o, e))
    # deterministic model: does capture consume the global RNG at all?
    _, nf = after(7, lambda: None)
    _, tdet = after(7, lambda: tl.trace(det, x))
    emit("H_rng_deterministic_model_trace_consumes", torch.equal(nf, tdet))
    _, tint = after(7, lambda: tl.trace(det, x, intervene=tl.when(tl.module("fc2"), tl.scale(0.5))))
    emit("H_rng_deterministic_model_trace_intervene_consumes", torch.equal(nf, tint))
    tt = tl.trace(det, x)
    _, trr = after(7, lambda: tt.run(det, x))
    emit("H_rng_deterministic_model_legacy_rerun_consumes", torch.equal(nf, trr))
    ready = tl.trace(det, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    _, tfd = after(7, lambda: ready.fork().do(tl.module("fc2"), tl.scale(0.5)))
    emit("H_rng_deterministic_model_fork_do_consumes", torch.equal(nf, tfd))
    # noise helper: unseeded noise should differ across captures; seeded identical
    outs = [
        out_of(tl.trace(det, x, intervene=tl.when(tl.module("fc2"), tl.noise(1.0))))
        for _ in range(2)
    ]
    emit("H_noise_unseeded_differs_across_captures", not torch.equal(outs[0], outs[1]))
    outs = [
        out_of(tl.trace(det, x, intervene=tl.when(tl.module("fc2"), tl.noise(1.0, seed=3))))
        for _ in range(2)
    ]
    emit("H_noise_seeded_identical_across_captures", torch.equal(outs[0], outs[1]))


# F: fork stacking: do twice on one fork
def f() -> None:
    model, x, x2, site, _ = models.build("mlp")
    g = torch.Generator().manual_seed(7)
    d = torch.randn(16, generator=g)

    def gt(
        k: float,
        xx: Any,
        extra: Callable[[torch.Tensor], torch.Tensor] | None = None,
        extra_site: str | None = None,
    ) -> torch.Tensor:
        hs = [model.get_submodule(site).register_forward_hook(lambda m, a, o: o + k * d)]
        if extra is not None:
            hs.append(
                model.get_submodule(extra_site).register_forward_hook(lambda m, a, o: extra(o))
            )
        try:
            with torch.no_grad():
                return model(xx)
        finally:
            for hh in hs:
                hh.remove()

    ready = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    fk = ready.fork()
    fk.do(tl.module(site), tl.steer(d, magnitude=1.0, feature_axis=-1))
    o1 = out_of(fk).clone()
    fk.do(tl.module(site), tl.steer(d, magnitude=1.0, feature_axis=-1))
    o2 = out_of(fk).clone()
    emit("F_fork_do_once", md(o1, gt(1.0, x)) < 1e-5, d=md(o1, gt(1.0, x)))
    emit(
        "F_fork_do_twice_same_site",
        md(o2, gt(2.0, x)) < 1e-5,
        d_vs_2x=md(o2, gt(2.0, x)),
        d_vs_3x=md(o2, gt(3.0, x)),
        d_vs_1x=md(o2, gt(1.0, x)),
        hooks=len(fk._ensure_intervention_spec().hook_specs),
    )
    fk.run(model, x2)
    emit(
        "F_fork_two_dos_then_rerun",
        md(out_of(fk), gt(2.0, x2)) < 1e-5,
        d_vs_2x=md(out_of(fk), gt(2.0, x2)),
        d_vs_4x=md(out_of(fk), gt(4.0, x2)),
        d_vs_3x=md(out_of(fk), gt(3.0, x2)),
    )
    # different site second
    fk2 = ready.fork()
    fk2.do(tl.module(site), tl.steer(d, magnitude=1.0, feature_axis=-1))
    fk2.do(tl.module("fc3"), tl.scale(0.5))
    want = gt(1.0, x, extra=lambda o: o * 0.5, extra_site="fc3")
    emit("F_fork_do_two_sites", md(out_of(fk2), want) < 1e-5, d=md(out_of(fk2), want))
    # set() value replacement then legacy reruns
    p = tl.trace(model, x)
    lbl = p.find_sites(tl.module(site))
    labels = [s.layer_label if hasattr(s, "layer_label") else str(s) for s in lbl]
    emit("F_site_labels", None, labels=labels[:4])


# A: attach_hooks / set on a plain trace, then legacy rerun x3 (B0 family check)
def a() -> None:
    model, x, x2, site, _ = models.build("mlp")
    fires = [0]

    def cnt(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        fires[0] += 1
        return out * 0.5

    t = tl.trace(model, x)
    t.attach_hooks(tl.module(site), cnt, confirm_mutation=True)
    counts = []
    for _ in range(3):
        fires[0] = 0
        t.run(model, x)
        counts.append(fires[0])
    emit(
        "A_attach_hooks_then_rerun_x3_fires",
        counts == [1, 1, 1],
        counts=counts,
        hooks=len(t._ensure_intervention_spec().hook_specs),
    )
    t = tl.trace(model, x, intervene=tl.when(tl.module(site), cnt))
    counts = []
    for xx in (x2, x2, x):
        fires[0] = 0
        t.run(model, xx)
        counts.append(fires[0])
    emit("A_intervene_rerun_fires_B0", counts == [1, 1, 1], counts=counts, known="B0")
    # bind executor reused on a fresh copy-of-weights model: fire counts per call
    b = tl.when(tl.module(site), cnt).bind(model)
    counts = []
    for _ in range(3):
        fires[0] = 0
        with torch.no_grad():
            b(x)
        counts.append(fires[0])
    emit("A_bind_fires", counts == [1, 1, 1], counts=counts)
    # record(intervene=) repeated with one spec
    spec = tl.when(tl.module(site), cnt)
    counts = []
    for _ in range(3):
        fires[0] = 0
        tl.record(model, x, default_op=True, intervene=spec)
        counts.append(fires[0])
    emit("A_record_fires", counts == [1, 1, 1], counts=counts)
    # trace(intervene=) repeated with one spec
    counts = []
    for _ in range(3):
        fires[0] = 0
        tl.trace(model, x, intervene=spec)
        counts.append(fires[0])
    emit("A_trace_fires", counts == [1, 1, 1], counts=counts)
    # validate after intervene
    try:
        v = tl.validate(model, x, scope="forward")
        emit("A_validate_after", bool(v), v=str(v)[:80])
    except Exception as exc:  # noqa: BLE001
        emit("A_validate_after", None, error=f"{type(exc).__name__}: {str(exc)[:150]}")


# P: pass scope: module called twice; pass-qualified site only hits pass 2
class TwiceNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.out = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.fc(torch.relu(self.fc(x)))))


def p() -> None:
    torch.manual_seed(0)
    model = TwiceNet().eval()
    x, x2 = torch.randn(2, 8), torch.randn(2, 8)
    calls = [0]

    def hk(m: nn.Module, a: Any, o: torch.Tensor) -> torch.Tensor:
        calls[0] += 1
        return o * 0.0 if calls[0] == 2 else o

    def gt(xx: torch.Tensor) -> torch.Tensor:
        calls[0] = 0
        hh = model.fc.register_forward_hook(hk)
        try:
            with torch.no_grad():
                return model(xx)
        finally:
            hh.remove()

    want, want2 = gt(x), gt(x2)
    tl.trace(model, x)
    spec = tl.when(tl.module("fc:2"), tl.zero_ablate())
    t = tl.trace(model, x, intervene=spec)
    emit("P_trace_pass2", md(out_of(t), want) < 1e-6, d=md(out_of(t), want))
    with torch.no_grad():
        b = spec.bind(model)(x)
    emit("P_bind_pass2", md(b, want) < 1e-6, d=md(b, want))
    r = tl.record(model, x, default_op=True, intervene=spec).to_trace()
    emit("P_record_pass2", md(out_of(r), want) < 1e-6, d=md(out_of(r), want))
    ready = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    fk = ready.fork()
    fk.do(tl.module("fc:2"), tl.zero_ablate())
    emit("P_fork_do_pass2", md(out_of(fk), want) < 1e-6, d=md(out_of(fk), want))
    fk.run(model, x2)
    emit("P_fork_rerun_newinput_pass2", md(out_of(fk), want2) < 1e-6, d=md(out_of(fk), want2))
    t.run(model, x2)
    emit("P_legacy_rerun_newinput_pass2", md(out_of(t), want2) < 1e-6, d=md(out_of(t), want2))


# M: lifetime: traces / forks / bindings are freed and do not pin the model
def m() -> None:
    model, x, _, site, _ = models.build("mlp")
    mref = weakref.ref(model)
    t = tl.trace(model, x, intervene=tl.when(tl.module(site), tl.scale(0.5)))
    t.run(model, x)
    fk = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True)).fork()
    fk.do(tl.module(site), tl.scale(0.5))
    b = tl.when(tl.module(site), tl.scale(0.5)).bind(model)
    b(x)
    refs = {"trace": weakref.ref(t), "fork": weakref.ref(fk), "bind": weakref.ref(b)}
    del t, fk, b
    gc.collect()
    emit(
        "M_objects_freed",
        all(r() is None for r in refs.values()),
        alive=[k for k, r in refs.items() if r() is not None],
    )
    del model
    gc.collect()
    emit("M_model_freed", mref() is None)


if __name__ == "__main__":
    which = sys.argv[1:] or ["g", "h", "f", "a", "p", "m"]
    for name in which:
        guarded(name, globals()[name])
        gc.collect()
    print("DONE", flush=True)
