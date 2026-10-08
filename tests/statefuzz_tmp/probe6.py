# ruff: noqa
"""Differential probe 6: fork payload isolation with in-place ops, loaded authored spec, other-model rerun, hooks= kwarg."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import tempfile
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


class InplaceMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.act = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(torch.relu(self.fc1(x)))
        h = self.act(h)
        h.mul_(1.5)
        return self.fc3(h)


def digests(t: Any) -> dict[str, str]:
    out = {}
    for op in t.layer_list:
        v = getattr(op, "out", None)
        if torch.is_tensor(v):
            out[op.layer_label] = hashlib.sha256(
                v.detach().contiguous().numpy().tobytes()
            ).hexdigest()[:10]
    return out


def u1() -> None:
    torch.manual_seed(0)
    model = InplaceMLP().eval()
    x, x2 = torch.randn(3, 8), torch.randn(3, 8)
    parent = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    before = digests(parent)
    g = torch.Generator().manual_seed(7)
    d = torch.randn(16, generator=g)
    f = parent.fork()
    steps = []
    for label, act in (
        ("do_zero_fc2", lambda: f.do(tl.module("fc2"), tl.zero_ablate())),
        (
            "do_steer_fc1",
            lambda: f.do(tl.module("fc1"), tl.steer(d, magnitude=2.0, feature_axis=-1)),
        ),
        ("child_rerun_x2", lambda: f.run(model, x2)),
        ("second_fork_do", lambda: parent.fork().do(tl.func("relu"), tl.scale(0.0))),
    ):
        try:
            act()
            err = None
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:150]}"
        now = digests(parent)
        changed = sorted(k for k in before if now.get(k) != before[k])
        steps.append({"step": label, "parent_changed": changed, "err": err})
    emit("U1_parent_payloads_isolated", all(not s["parent_changed"] for s in steps), steps=steps)
    gt = gt_forward(model, "fc2", lambda o: torch.zeros_like(o), x)
    f2 = parent.fork()
    f2.do(tl.module("fc2"), tl.zero_ablate())
    emit("U1_inplace_model_fork_do_vs_gt", md(out_of(f2), gt) < 1e-5, d=md(out_of(f2), gt))
    t = tl.trace(model, x, intervene=tl.when(tl.module("fc2"), tl.zero_ablate()))
    emit("U1_inplace_model_trace_vs_gt", md(out_of(t), gt) < 1e-5, d=md(out_of(t), gt))


def u2() -> None:
    model, x, x2, site, _ = models.build("mlp")
    gt2 = gt_forward(model, site, lambda o: o * 0.5, x2)
    with torch.no_grad():
        clean2 = model(x2)
    for door in ("attach_hooks", "fork_do"):
        if door == "attach_hooks":
            t = tl.trace(model, x)
            t.attach_hooks(tl.module(site), tl.scale(0.5), confirm_mutation=True)
        else:
            t = tl.trace(
                model, x, capture=tl.options.CaptureOptions(intervention_ready=True)
            ).fork()
            t.do(tl.module(site), tl.scale(0.5))
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "t.tlspec")
            try:
                t.save(p)
                t2 = tl.load(p)
                n = len(t2._ensure_intervention_spec().hook_specs)
                t2.run(model, x2)
                emit(
                    f"U2_loaded_{door}_legacy_rerun",
                    md(out_of(t2), gt2) < 1e-5,
                    d_vs_gt=md(out_of(t2), gt2),
                    d_vs_clean=md(out_of(t2), clean2),
                    loaded_hooks=n,
                    live_hooks=len(t._ensure_intervention_spec().hook_specs),
                    spec_revision=[
                        getattr(t, "_spec_revision", None),
                        getattr(t2, "_spec_revision", None),
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                emit(
                    f"U2_loaded_{door}_legacy_rerun",
                    None,
                    error=f"{type(exc).__name__}: {str(exc)[:250]}",
                )


def u3() -> None:
    model, x, x2, site, _ = models.build("mlp")
    other, _, _, _, _ = models.build("mlp", seed=5)
    gt_o = gt_forward(other, site, lambda o: o * 0.5, x2)
    t = tl.trace(model, x, intervene=tl.when(tl.module(site), tl.scale(0.5)))
    try:
        t.run(other, x2)
        emit("U3_legacy_rerun_other_model", md(out_of(t), gt_o) < 1e-5, d=md(out_of(t), gt_o))
    except Exception as exc:  # noqa: BLE001
        emit("U3_legacy_rerun_other_model", None, error=f"{type(exc).__name__}: {str(exc)[:200]}")


def u5() -> None:
    model, x, x2, site, _ = models.build("mlp")
    fires = [0]

    def cnt(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        fires[0] += 1
        return out * 0.5

    gt = gt_forward(model, site, lambda o: o * 0.5, x)
    for form_name, form in (
        ("dict", lambda: {tl.module(site): cnt}),
        ("tuple", lambda: [(tl.module(site), cnt)]),
    ):
        try:
            fires[0] = 0
            t = tl.trace(model, x, hooks=form())
            first = fires[0]
            d0 = md(out_of(t), gt)
            counts = []
            for _ in range(3):
                fires[0] = 0
                t.run(model, x)
                counts.append(fires[0])
            emit(
                f"U5_hooks_kwarg_{form_name}",
                first == 1 and d0 < 1e-5 and counts == [1, 1, 1],
                first=first,
                d0=d0,
                rerun_counts=counts,
                d_last=md(out_of(t), gt),
            )
        except Exception as exc:  # noqa: BLE001
            emit(
                f"U5_hooks_kwarg_{form_name}", None, error=f"{type(exc).__name__}: {str(exc)[:200]}"
            )


if __name__ == "__main__":
    which = sys.argv[1:] or ["u1", "u2", "u3", "u5"]
    for name in which:
        guarded(name, globals()[name])
        gc.collect()
    print("DONE", flush=True)
