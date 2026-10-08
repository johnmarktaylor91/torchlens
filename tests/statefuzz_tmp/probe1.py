"""Differential probe 1: every entry point vs a fresh oracle and a plain-hook ground truth.

Usage: python probe1.py [model kinds...]
Prints one JSON line per check: {"model","helper","scenario","ok","detail"}.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import traceback
import warnings
from collections.abc import Callable
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(__file__))
import models  # noqa: E402
import snap  # noqa: E402

import torchlens as tl  # noqa: E402

warnings.simplefilter("ignore")
torch.set_num_threads(1)

RESULTS: list[dict[str, Any]] = []


def emit(model: str, helper: str, scenario: str, ok: bool | None, **detail: Any) -> None:
    row = {"model": model, "helper": helper, "scenario": scenario, "ok": ok, **detail}
    RESULTS.append(row)
    print(json.dumps(row, default=str), flush=True)


def maxdiff(a: Any, b: Any) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max())


def trace_out(t: Any) -> torch.Tensor:
    return t.output_ops[0].out


class Fires:
    def __init__(self) -> None:
        self.n = 0


def make_helpers(
    feat_dim: int, feat_axis: int
) -> dict[str, tuple[Callable[[], Any], Callable[[torch.Tensor], torch.Tensor], Fires | None]]:
    torch.manual_seed(123)
    d = torch.randn(feat_dim)
    shape = [1] * 4
    fires = Fires()

    def align(o: torch.Tensor) -> torch.Tensor:
        s = [1] * o.dim()
        s[feat_axis] = feat_dim
        return d.reshape(s)

    def steer_gt(o: torch.Tensor) -> torch.Tensor:
        return o + 3.0 * align(o)

    def counting(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        fires.n += 1
        return out + 3.0 * align(out)

    del shape
    return {
        "steer": (lambda: tl.steer(d, magnitude=3.0, feature_axis=feat_axis), steer_gt, None),
        "scale": (lambda: tl.scale(0.25), lambda o: o * 0.25, None),
        "zero": (lambda: tl.zero_ablate(), lambda o: torch.zeros_like(o), None),
        "callable": (lambda: counting, steer_gt, fires),
    }


def gt_forward(
    model: torch.nn.Module, site: str, fn: Callable[[torch.Tensor], torch.Tensor], x: Any
) -> torch.Tensor:
    mod = model.get_submodule(site)
    h = mod.register_forward_hook(lambda m, a, o: fn(o))
    try:
        with torch.no_grad():
            return model(x)
    finally:
        h.remove()


def guarded(model_name: str, helper: str, scenario: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        emit(
            model_name,
            helper,
            scenario,
            None,
            error=f"{type(exc).__name__}: {str(exc)[:300]}",
            where=traceback.format_exc().strip().splitlines()[-3][:200],
        )


def run_model(kind: str) -> None:
    model, x, x2, site, feat_dim = models.build(kind)
    feat_axis = 1 if kind == "conv" else -1
    with torch.no_grad():
        clean = model(x)
        clean2 = model(x2)
    # warm-up plain capture installs the lazy torch wrappers (by design persistent)
    tl.trace(model, x)
    base = snap.full_state(model)
    sel = tl.module(site)
    helpers = make_helpers(feat_dim, feat_axis)

    def check_state(helper: str, scenario: str) -> None:
        d = snap.diff(base, snap.full_state(model))
        emit(kind, helper, scenario + ":state", not d, diff=d)
        with torch.no_grad():
            emit(
                kind,
                helper,
                scenario + ":eager_after",
                maxdiff(model(x), clean) == 0.0,
                d=maxdiff(model(x), clean),
            )

    for hname, (mk, gt_fn, fires) in helpers.items():
        gt = gt_forward(model, site, gt_fn, x)
        gt2 = gt_forward(model, site, gt_fn, x2)
        emit(kind, hname, "gt_differs_from_clean", maxdiff(gt, clean) > 0, d=maxdiff(gt, clean))

        # S1 trace(intervene=)
        def s1() -> None:
            if fires:
                fires.n = 0
            spec = tl.when(sel, mk())
            t = tl.trace(model, x, intervene=spec)
            emit(
                kind,
                hname,
                "S1_trace_intervene",
                maxdiff(trace_out(t), gt) < 1e-5,
                d=maxdiff(trace_out(t), gt),
                fires=fires.n if fires else None,
            )
            check_state(hname, "S1")
            # S2 legacy rerun x3 (known doubling bug; recorded for completeness)
            for i in range(3):
                if fires:
                    fires.n = 0
                t.run(model, x)
                emit(
                    kind,
                    hname,
                    f"S2_legacy_rerun{i + 1}",
                    maxdiff(trace_out(t), gt) < 1e-5,
                    d=maxdiff(trace_out(t), gt),
                    fires=fires.n if fires else None,
                    known="B0",
                )
            check_state(hname, "S2")
            # S3 legacy rerun on new input
            if fires:
                fires.n = 0
            t2 = tl.trace(model, x, intervene=tl.when(sel, mk()))
            t2.run(model, x2)
            emit(
                kind,
                hname,
                "S3_legacy_rerun_newinput_first",
                maxdiff(trace_out(t2), gt2) < 1e-5,
                d=maxdiff(trace_out(t2), gt2),
                fires=fires.n if fires else None,
            )
            # S4 run(inputs=, fast=True)
            for fast in (False, True):
                try:
                    res = t2.run(inputs=x2, fast=fast)
                    out = getattr(res, "output", None)
                    if out is None and hasattr(res, "trace"):
                        out = trace_out(res.trace)
                    if out is None and hasattr(res, "output_ops"):
                        out = trace_out(res)
                    emit(
                        kind,
                        hname,
                        f"S4_run_inputs_fast{fast}",
                        None if out is None else maxdiff(out, gt2) < 1e-5,
                        d=None if out is None else maxdiff(out, gt2),
                        restype=type(res).__name__,
                        attrs=[a for a in dir(res) if not a.startswith("_")][:40]
                        if out is None
                        else None,
                    )
                except Exception as exc:  # noqa: BLE001
                    code = (
                        getattr(exc, "fields", {}).get("code") if hasattr(exc, "fields") else None
                    )
                    emit(
                        kind,
                        hname,
                        f"S4_run_inputs_fast{fast}",
                        code is not None,
                        refused=code,
                        error=f"{type(exc).__name__}: {str(exc)[:160]}",
                    )
            check_state(hname, "S4")

        guarded(kind, hname, "S1-4", s1)

        # S5 fork/do on a plain intervention-ready trace
        def s5() -> None:
            parent = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
            p_out0 = trace_out(parent).clone()
            p_spec0 = len(getattr(parent._ensure_intervention_spec(), "hook_specs", []))
            if fires:
                fires.n = 0
            f1 = parent.fork()
            f1.do(sel, mk())
            emit(
                kind,
                hname,
                "S5_fork_do",
                maxdiff(trace_out(f1), gt) < 1e-5,
                d=maxdiff(trace_out(f1), gt),
                fires=fires.n if fires else None,
            )
            emit(
                kind,
                hname,
                "S5_parent_untouched",
                maxdiff(trace_out(parent), p_out0) == 0.0,
                d=maxdiff(trace_out(parent), p_out0),
                parent_hooks=[p_spec0, len(parent._ensure_intervention_spec().hook_specs)],
            )
            f2 = parent.fork()
            if fires:
                fires.n = 0
            f2.do(sel, mk())
            emit(
                kind,
                hname,
                "S5_second_fork_do_fresh",
                maxdiff(trace_out(f2), gt) < 1e-5,
                d=maxdiff(trace_out(f2), gt),
                fires=fires.n if fires else None,
                f1_hooks=len(f1._ensure_intervention_spec().hook_specs),
                f2_hooks=len(f2._ensure_intervention_spec().hook_specs),
            )
            # fork of a fork that carries an edit, then legacy rerun of the child
            f3 = f1.fork()
            emit(
                kind,
                hname,
                "S5_fork_of_edited_fork_out",
                maxdiff(trace_out(f3), gt) < 1e-5,
                d=maxdiff(trace_out(f3), gt),
                f3_hooks=len(f3._ensure_intervention_spec().hook_specs),
            )
            # fork after do then rerun the fork on new input
            if fires:
                fires.n = 0
            f1.run(model, x2)
            emit(
                kind,
                hname,
                "S5_fork_rerun_newinput",
                maxdiff(trace_out(f1), gt2) < 1e-5,
                d=maxdiff(trace_out(f1), gt2),
                fires=fires.n if fires else None,
            )
            emit(
                kind,
                hname,
                "S5_parent_after_child_rerun",
                maxdiff(trace_out(parent), p_out0) == 0.0,
                d=maxdiff(trace_out(parent), p_out0),
            )
            check_state(hname, "S5")

        guarded(kind, hname, "S5", s5)

        # S6 tl.record(intervene=) twice with the same spec object
        def s6() -> None:
            spec = tl.when(sel, mk())
            dig0 = spec.spec_digest
            for i in range(2):
                if fires:
                    fires.n = 0
                rec = tl.record(model, x, intervene=spec)
                tr = rec.to_trace()
                emit(
                    kind,
                    hname,
                    f"S6_record{i + 1}",
                    maxdiff(trace_out(tr), gt) < 1e-5,
                    d=maxdiff(trace_out(tr), gt),
                    fires=fires.n if fires else None,
                )
            emit(kind, hname, "S6_spec_digest_stable", spec.spec_digest == dig0)
            check_state(hname, "S6")

        guarded(kind, hname, "S6", s6)

        # S7 bind x3, then unbound eager
        def s7() -> None:
            spec = tl.when(sel, mk())
            bound = spec.bind(model)
            for i in range(3):
                if fires:
                    fires.n = 0
                with torch.no_grad():
                    y = bound(x)
                emit(
                    kind,
                    hname,
                    f"S7_bind_call{i + 1}",
                    maxdiff(y, gt) < 1e-5,
                    d=maxdiff(y, gt),
                    fires=fires.n if fires else None,
                )
            with torch.no_grad():
                y2 = bound(x2)
            emit(kind, hname, "S7_bind_newinput", maxdiff(y2, gt2) < 1e-5, d=maxdiff(y2, gt2))
            check_state(hname, "S7")
            # trace through after bind with the same spec
            t = tl.trace(model, x, intervene=spec)
            emit(
                kind,
                hname,
                "S7_trace_after_bind_same_spec",
                maxdiff(trace_out(t), gt) < 1e-5,
                d=maxdiff(trace_out(t), gt),
            )
            del bound

        guarded(kind, hname, "S7", s7)

        # S8 same spec reused on a second model copy
        def s8() -> None:
            spec = tl.when(sel, mk())
            tl.trace(model, x, intervene=spec)
            other = copy.deepcopy(model)
            with torch.no_grad():
                for p in other.parameters():
                    p.mul_(1.1)
            gto = gt_forward(other, site, gt_fn, x)
            to = tl.trace(other, x, intervene=spec)
            emit(
                kind,
                hname,
                "S8_spec_reuse_other_model",
                maxdiff(trace_out(to), gto) < 1e-5,
                d=maxdiff(trace_out(to), gto),
            )
            ta = tl.trace(model, x, intervene=spec)
            emit(
                kind,
                hname,
                "S8_spec_reuse_back",
                maxdiff(trace_out(ta), gt) < 1e-5,
                d=maxdiff(trace_out(ta), gt),
            )
            check_state(hname, "S8")

        guarded(kind, hname, "S8", s8)

        # S9 save/load round trip of an intervened trace
        def s9() -> None:
            if hname == "callable":
                return
            t = tl.trace(model, x, intervene=tl.when(sel, mk()))
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "t.tlspec")
                t.save(path)
                t2 = tl.load(path)
                emit(
                    kind,
                    hname,
                    "S9_loaded_out",
                    maxdiff(trace_out(t2), gt) < 1e-5,
                    d=maxdiff(trace_out(t2), gt),
                )
                try:
                    t2.run(model, x)
                    emit(
                        kind,
                        hname,
                        "S9_loaded_legacy_rerun",
                        maxdiff(trace_out(t2), gt) < 1e-5,
                        d=maxdiff(trace_out(t2), gt),
                    )
                except Exception as exc:  # noqa: BLE001
                    emit(
                        kind,
                        hname,
                        "S9_loaded_legacy_rerun",
                        None,
                        error=f"{type(exc).__name__}: {str(exc)[:200]}",
                    )
            check_state(hname, "S9")

        guarded(kind, hname, "S9", s9)

        # S10 a plain capture after everything equals clean
        def s10() -> None:
            t = tl.trace(model, x)
            emit(
                kind,
                hname,
                "S10_plain_after",
                maxdiff(trace_out(t), clean) == 0.0,
                d=maxdiff(trace_out(t), clean),
            )
            t.run(model, x2)
            emit(
                kind,
                hname,
                "S10_plain_rerun_newinput",
                maxdiff(trace_out(t), clean2) == 0.0,
                d=maxdiff(trace_out(t), clean2),
            )

        guarded(kind, hname, "S10", s10)


if __name__ == "__main__":
    kinds = sys.argv[1:] or ["mlp", "conv", "decoder"]
    for k in kinds:
        guarded(k, "-", "model", lambda k=k: run_model(k))
    bad = [r for r in RESULTS if r["ok"] is not True]
    print(f"SUMMARY total={len(RESULTS)} not_ok={len(bad)}", flush=True)
