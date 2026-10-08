# ruff: noqa
"""Differential probe 5: output-node fan-out family, sweep, downstream user hooks under replay, rerun engine."""

from __future__ import annotations

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


class ReluTail(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc1(x))


class TupleOut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc1(x)
        return self.fc2(h), h


# Q7: which post-hoc selectors match the output node too, and does fork.do compound?
def q7() -> None:
    torch.manual_seed(0)
    x, x2 = torch.randn(3, 8), torch.randn(3, 8)
    cases = [
        ("relu_tail_module_act", ReluTail().eval(), tl.module("act"), "act"),
        ("relu_tail_func_relu", ReluTail().eval(), tl.func("relu"), "act"),
        ("relu_tail_in_module_act", ReluTail().eval(), tl.in_module("act"), "act"),
        ("tuple_out_module_fc2", TupleOut().eval(), tl.module("fc2"), "fc2"),
        ("tuple_out_module_fc1_also_returned", TupleOut().eval(), tl.module("fc1"), "fc1"),
    ]
    for name, model, sel, site in cases:
        gt = gt_forward(model, site, lambda o: o * 0.5, x)
        gt2 = gt_forward(model, site, lambda o: o * 0.5, x2)
        g = gt if torch.is_tensor(gt) else gt[0]
        ready = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
        labels = [getattr(s, "layer_label", str(s)) for s in ready.find_sites(sel)]
        fk = ready.fork()
        try:
            fk.do(sel, tl.scale(0.5))
            fo = out_of(fk)
            row = dict(fork_do_d=md(fo, g))
            fk.run(model, x2)
            row["fork_rerun_d"] = md(out_of(fk), gt2 if torch.is_tensor(gt2) else gt2[0])
        except Exception as exc:  # noqa: BLE001
            row = dict(err=f"{type(exc).__name__}: {str(exc)[:150]}")
        t = tl.trace(model, x, intervene=tl.when(sel, tl.scale(0.5)))
        row["trace_d"] = md(out_of(t), g)
        if len(t.output_ops) > 1 and not torch.is_tensor(gt):
            row["trace_second_output_d"] = md(t.output_ops[1].out, gt[1])
            row["fork_second_output_d"] = (
                md(fk.output_ops[1].out, gt[1]) if "err" not in row else None
            )
        with torch.no_grad():
            b = tl.when(sel, tl.scale(0.5)).bind(model)(x)
        row["bind_d"] = md(b if torch.is_tensor(b) else b[0], g)
        emit(
            f"Q7_{name}",
            all(v is None or v < 1e-5 for k, v in row.items() if k.endswith("_d"))
            and "err" not in row,
            sites=labels,
            **row,
        )


# Q1: sweep members equal their plain-hook ground truths; baseline clean
def q1() -> None:
    model, x, _, site, _ = models.build("mlp")
    vals = [0.0, 1.0, 2.0]
    b = (
        tl.intervention.sweep(
            model, x, tl.module(site), [torch.full((3, 16), v) for v in vals], include_baseline=True
        )
        if hasattr(tl, "intervention") and hasattr(tl.intervention, "sweep")
        else tl.sweep(
            model, x, tl.module(site), [torch.full((3, 16), v) for v in vals], include_baseline=True
        )
    )
    names = list(getattr(b, "names", None) or getattr(b, "member_names", None) or [])
    with torch.no_grad():
        clean = model(x)
    rows = {}
    for i, v in enumerate(vals):
        gt = gt_forward(model, site, lambda o, v=v: torch.full_like(o, v), x)
        member = b[names[i + 1]] if names else None
        rows[str(v)] = None if member is None else md(out_of(member), gt)
    base = b[names[0]] if names else None
    emit(
        "Q1_sweep_members",
        all(r is not None and r < 1e-5 for r in rows.values())
        and base is not None
        and md(out_of(base), clean) == 0.0,
        names=names,
        member_d=rows,
        baseline_d=None if base is None else md(out_of(base), clean),
    )


# Q2: downstream user forward hook under fork.do replay
def q2() -> None:
    model, x, x2, site, _ = models.build("mlp")
    model.fc3.register_forward_hook(lambda m, a, o: o + 1.0)

    def gt(xx: torch.Tensor) -> torch.Tensor:
        return gt_forward(model, site, lambda o: o * 0.5, xx)

    ready = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    with torch.no_grad():
        emit("Q2_capture_includes_user_hook", md(out_of(ready), model(x)) == 0.0)
    fk = ready.fork()
    fk.do(tl.module(site), tl.scale(0.5))
    emit("Q2_fork_do_downstream_user_hook", md(out_of(fk), gt(x)) < 1e-5, d=md(out_of(fk), gt(x)))
    t = tl.trace(model, x, intervene=tl.when(tl.module(site), tl.scale(0.5)))
    emit(
        "Q2_trace_intervene_downstream_user_hook",
        md(out_of(t), gt(x)) < 1e-5,
        d=md(out_of(t), gt(x)),
    )


# Q3: do(engine="rerun") repeated on one fork; layer/op do
def q3() -> None:
    model, x, x2, site, _ = models.build("mlp")
    g = torch.Generator().manual_seed(7)
    d = torch.randn(16, generator=g)
    ready = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    fk = ready.fork()
    opts = (
        tl.options.InterventionOptions(engine="rerun")
        if hasattr(tl.options, "InterventionOptions")
        else None
    )
    res = []
    for i in range(3):
        fk.do(
            tl.module(site),
            tl.steer(d, magnitude=1.0, feature_axis=-1),
            model=model,
            x=x,
            intervention=opts,
        )
        k = [
            md(out_of(fk), gt_forward(model, site, lambda o, k=k: o + k * d, x))
            for k in (1, 2, 3, 4, 7, 8)
        ]
        res.append(k)
    emit(
        "Q3_do_rerun_engine_x3",
        None,
        d_vs_k_1_2_3_4_7_8=res,
        hooks=len(fk._ensure_intervention_spec().hook_specs),
    )


if __name__ == "__main__":
    which = sys.argv[1:] or ["q7", "q1", "q2", "q3"]
    for name in which:
        guarded(name, globals()[name])
        gc.collect()
    print("DONE", flush=True)
