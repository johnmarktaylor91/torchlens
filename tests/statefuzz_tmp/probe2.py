# ruff: noqa
"""Differential probe 2: targeted follow-ups (copy, load, RNG, train mode, exceptions, episodes)."""

from __future__ import annotations

import copy
import gc
import json
import os
import sys
import tempfile
import traceback
import warnings
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
import models  # noqa: E402
import snap  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens import _state  # noqa: E402

torch.set_num_threads(1)
IGN = ("_log_registry", "TYPE_CHECKING", "annotations", "torch_rng", "py_rng", "np_rng")


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
        kinds = sorted({type(x.message).__name__ + ":" + str(x.message)[:90] for x in w})
        if kinds:
            emit(name + ":warnings", None, warnings=kinds[:8])
    except Exception as exc:  # noqa: BLE001
        emit(
            name,
            None,
            error=f"{type(exc).__name__}: {str(exc)[:400]}",
            tb=[ln.strip()[:160] for ln in traceback.format_exc().strip().splitlines()[-7:]],
        )


def steer_parts(feat: int = 16, axis: int = -1):
    g = torch.Generator().manual_seed(7)
    d = torch.randn(feat, generator=g)

    def gt(o: torch.Tensor) -> torch.Tensor:
        s = [1] * o.dim()
        s[axis] = feat
        return o + 3.0 * d.reshape(s)

    return d, gt


def gt_forward(
    model: nn.Module, site: str, fn: Callable[[torch.Tensor], torch.Tensor], x: Any
) -> torch.Tensor:
    h = model.get_submodule(site).register_forward_hook(lambda m, a, o: fn(o))
    try:
        with torch.no_grad():
            return model(x)
    finally:
        h.remove()


# R1: deepcopy of a traced model
def r1() -> None:
    for variant in ("plain", "intervene", "control_copy_before_trace"):
        model, x, _, site, _ = models.build("mlp")
        d, gt = steer_parts()
        if variant == "control_copy_before_trace":
            other = copy.deepcopy(model)
            tl.trace(model, x)
        else:
            if variant == "plain":
                tl.trace(model, x)
            else:
                tl.trace(
                    model,
                    x,
                    intervene=tl.when(tl.module(site), tl.steer(d, magnitude=3.0, feature_axis=-1)),
                )
            other = copy.deepcopy(model)
        try:
            t = tl.trace(other, x)
            with torch.no_grad():
                ref = other(x)
            emit(f"R1_deepcopy_{variant}", md(out_of(t), ref) == 0.0, d=md(out_of(t), ref))
        except Exception as exc:  # noqa: BLE001
            emit(
                f"R1_deepcopy_{variant}",
                False,
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
                tb=[ln.strip()[:160] for ln in traceback.format_exc().strip().splitlines()[-6:]],
            )
        # after the failure, is the ORIGINAL model still traceable and the process clean?
        try:
            t0 = tl.trace(model, x)
            with torch.no_grad():
                emit(f"R1_original_after_{variant}", md(out_of(t0), model(x)) == 0.0)
        except Exception as exc:  # noqa: BLE001
            emit(
                f"R1_original_after_{variant}",
                False,
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
        emit(
            f"R1_state_globals_{variant}",
            _state._active_trace is None
            and _state._active_intervention_spec is None
            and _state._active_hook_plan is None
            and not _state._logging_enabled,
            active_trace=_state._active_trace is not None,
            spec=_state._active_intervention_spec is not None,
            plan=_state._active_hook_plan is not None,
            logging=_state._logging_enabled,
        )


# R2: loaded trace that carries an intervention, legacy rerun
def r2() -> None:
    model, x, x2, site, _ = models.build("mlp")
    d, gtf = steer_parts()
    gt = gt_forward(model, site, gtf, x)
    gt2 = gt_forward(model, site, gtf, x2)
    with torch.no_grad():
        clean, clean2 = model(x), model(x2)
    t = tl.trace(
        model, x, intervene=tl.when(tl.module(site), tl.steer(d, magnitude=3.0, feature_axis=-1))
    )
    emit(
        "R2_live_hooks_after_capture",
        None,
        hook_specs=len(t._ensure_intervention_spec().hook_specs),
        targets=len(t._ensure_intervention_spec().targets),
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.tlspec")
        t.save(p)
        t2 = tl.load(p)
        spec2 = getattr(t2, "_intervention_spec", None)
        emit(
            "R2_loaded_spec",
            None,
            has_spec=spec2 is not None,
            hook_specs=None if spec2 is None else len(spec2.hook_specs),
            targets=None if spec2 is None else len(spec2.targets),
        )
        emit("R2_loaded_out_matches_gt", md(out_of(t2), gt) < 1e-5, d=md(out_of(t2), gt))
        for label, xx, g, c in (("same_input", x, gt, clean), ("new_input", x2, gt2, clean2)):
            try:
                r = t2.run(model, xx)
                o = out_of(r if hasattr(r, "output_ops") else t2)
                emit(
                    f"R2_loaded_legacy_rerun_{label}",
                    md(o, g) < 1e-5,
                    d_vs_gt=md(o, g),
                    d_vs_clean=md(o, c),
                    ret=type(r).__name__,
                )
            except Exception as exc:  # noqa: BLE001
                emit(
                    f"R2_loaded_legacy_rerun_{label}",
                    None,
                    error=f"{type(exc).__name__}: {str(exc)[:300]}",
                )
        try:
            r = t2.run(inputs=x2)
            emit(
                "R2_loaded_run_inputs",
                None,
                ret=type(r).__name__,
                out_d_vs_gt=md(r.output, gt2)
                if hasattr(r, "output") and torch.is_tensor(getattr(r, "output", None))
                else None,
                out_d_vs_clean=md(r.output, clean2)
                if hasattr(r, "output") and torch.is_tensor(getattr(r, "output", None))
                else None,
            )
        except Exception as exc:  # noqa: BLE001
            emit(
                "R2_loaded_run_inputs",
                None,
                refused=getattr(exc, "fields", {}).get("code") if hasattr(exc, "fields") else None,
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )


# R3: RNG stream equivalence vs eager (dropout model in train mode)
class DropNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


def r3() -> None:
    torch.manual_seed(0)
    model = DropNet().train()
    x = torch.randn(4, 8)
    tl.trace(model, x)  # warm
    for label, call in (
        ("trace", lambda: tl.trace(model, x)),
        (
            "trace_intervene",
            lambda: tl.trace(model, x, intervene=tl.when(tl.module("fc2"), tl.scale(1.0))),
        ),
        ("record", lambda: tl.record(model, x, default_op=True)),
        ("bind", lambda: tl.when(tl.module("fc2"), tl.scale(1.0)).bind(model)(x)),
    ):
        torch.manual_seed(42)
        eager_out = model(x)
        eager_after = torch.rand(3)
        torch.manual_seed(42)
        res = call()
        tl_after = torch.rand(3)
        o = None
        if hasattr(res, "output_ops"):
            o = out_of(res)
        elif torch.is_tensor(res):
            o = res
        emit(
            f"R3_rng_stream_{label}",
            md(eager_after, tl_after) == 0.0,
            d_next_draw=md(eager_after, tl_after),
            out_vs_eager=None if o is None else md(o, eager_out),
        )
    # legacy rerun on a dropout trace: does it consume RNG like eager?
    t = tl.trace(model, x)
    torch.manual_seed(5)
    e = model(x)
    ea = torch.rand(3)
    torch.manual_seed(5)
    t.run(model, x)
    ta = torch.rand(3)
    emit(
        "R3_rng_stream_legacy_rerun",
        md(ea, ta) == 0.0,
        d_next=md(ea, ta),
        out_vs_eager=md(out_of(t), e),
    )


# R4: train-mode BatchNorm running stats: one TorchLens call == one eager forward
def r4() -> None:
    for label in (
        "trace",
        "trace_intervene",
        "record",
        "bind",
        "legacy_rerun",
        "fork_do",
        "run_inputs",
    ):
        torch.manual_seed(0)
        model = models.ConvNet().train()
        x = torch.randn(2, 3, 6, 6)
        ref = copy.deepcopy(model)
        tl.trace(model, x)  # warm (one forward of stats on model)
        ref(x)  # keep ref in step
        before = {k: v.clone() for k, v in model.named_buffers()}
        spec = tl.when(tl.module("conv2"), tl.scale(1.0))
        t = tl.trace(model, x) if label in {"legacy_rerun", "run_inputs"} else None
        if t is not None:
            before = {k: v.clone() for k, v in model.named_buffers()}
            ref.load_state_dict(model.state_dict())
        if label == "fork_do":
            t = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
            ref.load_state_dict(model.state_dict())
            before = {k: v.clone() for k, v in model.named_buffers()}
        try:
            if label == "trace":
                tl.trace(model, x)
            elif label == "trace_intervene":
                tl.trace(model, x, intervene=spec)
            elif label == "record":
                tl.record(model, x, default_op=True)
            elif label == "bind":
                spec.bind(model)(x)
            elif label == "legacy_rerun":
                t.run(model, x)
            elif label == "run_inputs":
                t.run(inputs=x)
            elif label == "fork_do":
                f = t.fork()
                f.do(tl.module("conv2"), tl.scale(1.0))
        except Exception as exc:  # noqa: BLE001
            emit(f"R4_bn_{label}", None, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            continue
        ref(x)
        nb = dict(model.named_buffers())
        rb = dict(ref.named_buffers())
        changed = {
            k: md(nb[k], before[k])
            for k in nb
            if k.endswith(("running_mean", "num_batches_tracked"))
        }
        vs_eager = {k: md(nb[k], rb[k]) for k in nb}
        emit(
            f"R4_bn_{label}",
            max(vs_eager.values()) == 0.0,
            nbt=[
                int(before["bn.num_batches_tracked"]),
                int(nb["bn.num_batches_tracked"]),
                int(rb["bn.num_batches_tracked"]),
            ],
            vs_one_eager=vs_eager,
            changed=changed,
        )


# R5: exceptions and interrupts leave nothing behind
class Boom(Exception):
    pass


class FlakyMLP(models.MLP):
    def __init__(self) -> None:
        super().__init__()
        self.fail = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc2(torch.relu(self.fc1(x))))
        if self.fail:
            raise Boom("forward failure after the site")
        return self.fc3(h)


def r5() -> None:
    torch.manual_seed(0)
    model = FlakyMLP().eval()
    x = torch.randn(3, 8)
    with torch.no_grad():
        clean = model(x)
    tl.trace(model, x)
    base = snap.full_state(model)
    d, gtf = steer_parts()
    gt = gt_forward(model, "fc2", gtf, x)

    def kb_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        raise KeyboardInterrupt

    plain_t = tl.trace(model, x)
    ready_t = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    steer = lambda: tl.when(tl.module("fc2"), tl.steer(d, magnitude=3.0, feature_axis=-1))  # noqa: E731
    cases: list[tuple[str, Callable[[], Any], bool]] = [
        ("trace_intervene_forward_raises", lambda: tl.trace(model, x, intervene=steer()), True),
        ("trace_plain_forward_raises", lambda: tl.trace(model, x), True),
        (
            "record_intervene_forward_raises",
            lambda: tl.record(model, x, default_op=True, intervene=steer()),
            True,
        ),
        ("bind_forward_raises", lambda: steer().bind(model)(x), True),
        ("legacy_rerun_forward_raises", lambda: plain_t.run(model, x), True),
        (
            "trace_intervene_hook_kbint",
            lambda: tl.trace(model, x, intervene=tl.when(tl.module("fc2"), kb_hook)),
            False,
        ),
        ("bind_hook_kbint", lambda: tl.when(tl.module("fc2"), kb_hook).bind(model)(x), False),
        ("fork_do_hook_kbint", lambda: ready_t.fork().do(tl.module("fc2"), kb_hook), False),
        (
            "record_hook_kbint",
            lambda: tl.record(
                model, x, default_op=True, intervene=tl.when(tl.module("fc2"), kb_hook)
            ),
            False,
        ),
    ]
    for name, call, set_fail in cases:
        model.fail = set_fail
        raised = None
        try:
            call()
        except BaseException as exc:  # noqa: BLE001
            raised = type(exc).__name__
        model.fail = False
        dd = snap.diff(base, snap.full_state(model), ignore=IGN)
        dd.pop("model.<root>.__dict__keys", None)
        flags = dict(
            active_trace=_state._active_trace is not None,
            spec=_state._active_intervention_spec is not None,
            plan=_state._active_hook_plan is not None,
            logging=_state._logging_enabled,
            reserved=getattr(_state, "_capture_reserved_by", None),
        )
        try:
            t = tl.trace(model, x)
            nxt = md(out_of(t), clean)
            ti = tl.trace(model, x, intervene=steer())
            nxt_i = md(out_of(ti), gt)
            with torch.no_grad():
                eager = md(model(x), clean)
            err = None
        except BaseException as exc:  # noqa: BLE001
            nxt = nxt_i = eager = None
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
        emit(
            f"R5_{name}",
            not dd
            and nxt == 0.0
            and nxt_i is not None
            and nxt_i < 1e-5
            and eager == 0.0
            and not any(v for k, v in flags.items() if k != "reserved")
            and flags["reserved"] is None,
            raised=raised,
            diff=dd,
            flags=flags,
            next_plain=nxt,
            next_intervened=nxt_i,
            eager=eager,
            err=err,
        )


# R6: two intervened traces on one model, interleaved legacy reruns (cross-contamination)
def r6() -> None:
    model, x, x2, site, _ = models.build("mlp")
    d, gtf = steer_parts()
    gt_steer = gt_forward(model, site, gtf, x2)
    gt_scale = gt_forward(model, site, lambda o: o * 0.25, x2)
    ta = tl.trace(
        model, x, intervene=tl.when(tl.module(site), tl.steer(d, magnitude=3.0, feature_axis=-1))
    )
    tb = tl.trace(model, x, intervene=tl.when(tl.module(site), tl.scale(0.25)))
    ta.run(model, x2)
    tb.run(model, x2)
    emit("R6_interleaved_a", md(out_of(ta), gt_steer) < 1e-5, d=md(out_of(ta), gt_steer))
    emit("R6_interleaved_b", md(out_of(tb), gt_scale) < 1e-5, d=md(out_of(tb), gt_scale))
    # fork with an attached hook, parent clears hooks: fork keeps its own
    p = tl.trace(model, x)
    p.attach_hooks(tl.module(site), tl.scale(0.25), confirm_mutation=True)
    f = p.fork()
    p.clear_hooks(confirm_mutation=True)
    emit(
        "R6_fork_hooks_after_parent_clear",
        None,
        parent=len(p._ensure_intervention_spec().hook_specs),
        fork=len(f._ensure_intervention_spec().hook_specs),
    )
    f.run(model, x2)
    emit(
        "R6_fork_rerun_after_parent_clear",
        md(out_of(f), gt_scale) < 1e-5,
        d=md(out_of(f), gt_scale),
    )
    p.run(model, x2)
    with torch.no_grad():
        emit(
            "R6_parent_rerun_after_clear",
            md(out_of(p), model(x2)) == 0.0,
            d=md(out_of(p), model(x2)),
        )


# R7: multi-site selector fire counts per call across entry points
def r7() -> None:
    model, x, x2, site, _ = models.build("mlp")
    fires = [0]

    def cnt(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        fires[0] += 1
        return out * 0.5

    spec = tl.when(tl.func("relu"), cnt)
    res = {}
    fires[0] = 0
    t = tl.trace(model, x, intervene=spec)
    res["trace"] = fires[0]
    fires[0] = 0
    spec.bind(model)(x)
    res["bind1"] = fires[0]
    b = spec.bind(model)
    fires[0] = 0
    b(x)
    b(x)
    res["bind_reused_x2"] = fires[0]
    fires[0] = 0
    tl.record(model, x, default_op=True, intervene=spec)
    res["record"] = fires[0]
    ready = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    fires[0] = 0
    f = ready.fork()
    f.do(tl.func("relu"), cnt)
    res["fork_do"] = fires[0]
    fires[0] = 0
    f.do(tl.func("relu"), cnt)
    res["fork_do_again_same_fork"] = fires[0]
    emit(
        "R7_fire_counts_relu_x2_sites",
        all(
            v == 2 for k, v in res.items() if k not in {"bind_reused_x2", "fork_do_again_same_fork"}
        )
        and res["bind_reused_x2"] == 4,
        counts=res,
        fork_hooks=len(f._ensure_intervention_spec().hook_specs),
    )
    # what does a second do() on the same fork yield vs gt of applying cnt once / twice?
    with torch.no_grad():
        hs = []
        mods_gt_once = None
    del hs, mods_gt_once, t


# R8: episode capture with intervene, repeated
class Gen(nn.Module):
    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.model = models.TinyDecoder()
        self.n = n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            logits = self.model(ids)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


def r8() -> None:
    torch.manual_seed(0)
    root = Gen().eval()
    ids = torch.randint(0, 32, (1, 4), generator=torch.Generator().manual_seed(3))
    d, gtf = steer_parts()
    gt_ids = gt_forward(root, "model.blocks.0", gtf, ids)
    with torch.no_grad():
        clean_ids = root(ids)
    spec = tl.when(tl.module("model.blocks.0"), tl.steer(d, magnitude=3.0, feature_axis=-1))
    ep = tl.options.EpisodeSpec(stepped_module=root.model, n_steps=3)
    digests = []
    for i in range(2):
        t = tl.trace(root, ids, episode=ep, intervene=spec)
        led = t.annotations.get("episode")
        rows = getattr(led, "rows", None) or (led.get("rows") if isinstance(led, dict) else None)
        fc = None
        if rows is not None:
            fc = [
                getattr(r, "fire_count", None) if not isinstance(r, dict) else r.get("fire_count")
                for r in rows
            ]
        dig = (
            getattr(led, "intervention_digest", None)
            if not isinstance(led, dict)
            else led.get("header", {}).get("intervention_digest", led.get("intervention_digest"))
        )
        digests.append(dig)
        o = out_of(t)
        emit(
            f"R8_episode_intervene_{i + 1}",
            torch.equal(o, gt_ids),
            out_eq_gt=torch.equal(o, gt_ids),
            out_eq_clean=torch.equal(o, clean_ids),
            gt_eq_clean=torch.equal(gt_ids, clean_ids),
            fire_counts=fc,
            ledger_type=type(led).__name__,
        )
    emit(
        "R8_episode_digest_stable",
        digests[0] == digests[1] and digests[0] is not None,
        digests=digests,
    )
    t = tl.trace(root, ids, episode=ep)
    emit("R8_episode_plain_after", torch.equal(out_of(t), clean_ids))


if __name__ == "__main__":
    which = sys.argv[1:] or ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"]
    for name in which:
        guarded(name, globals()[name])
        gc.collect()
    print("DONE", flush=True)
