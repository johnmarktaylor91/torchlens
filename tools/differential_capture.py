"""Cross-process differential harness: mode-capture vs wrapper-capture.

Stage-1 safety-net oracle (dispatch verdict, converged staged plan): compare
what a ``TorchFunctionMode`` observes on PRISTINE torch against what the
TorchLens wrappers record for the same forward pass, op-for-op, with a 1:1
name mapping. This is the enabling oracle for every future mode-vs-wrapper
decision (the rescue net, stage 2+): any drift between the two capture
mechanisms must surface here as a NAMED, per-event rule application or a loud
mismatch — never a whole-column mask.

Why cross-process: the mode side must observe pristine torch. Inside a
process where TorchLens has wrapped torch, every namespace entry is already a
TorchLens wrapper, so a mode would see wrapper identities and miss the
protocol behavior of the real originals. Each side therefore runs in its own
subprocess (``--side mode`` never wraps; ``--side wrapper`` runs a normal
``tl.trace``), and the comparison aligns their JSON streams.

Canonicalization discipline (from the dispatch verdict): every divergence
between the streams is explained by exactly one of four NARROW named rules,
each application recorded per event index in the report:

- ``STRUCTURAL_ROW`` (wrapper side): drop ``none`` / ``identity`` rows —
  TorchLens-synthesized graph structure (input/output/buffer boundaries,
  module-boundary identity), not user torch calls.
- ``NOT_LOGGED`` (mode side): drop events whose canonical name sits in
  TorchLens's own declared non-logging inventory (``funcs_not_to_log``,
  emitted by the wrapper process — the table is TL's, never the harness's).
- ``DUNDER_RESPELL`` (pairwise): wrapper ``__X__`` matches mode ``X`` — the
  protocol receives the public method object for operator sugar, so ``y * 2``
  arrives as ``mul`` while the wrapper vocabulary says ``__mul__``.
- ``COMPOSITE_EXPANSION`` (mode side): a Python-level composite is ONE opaque
  mode event (the protocol pops the mode before the body runs) but a SPAN of
  leaf ops in the wrapper stream. The expansion is DERIVED MECHANICALLY at
  run time — the wrapper process traces a one-op probe model for each
  composite the model spec declares — never hand-pinned.

Name canonicalization on the mode side uses TorchLens's OWN wrapping
inventory (``constants.get_orig_torch_funcs``, id-keyed, first-wins — the
same dedup the wrapper installer applies), NEVER ``torch.overrides
.resolve_name`` (measured hazard: ``resolve_name(torch.mm) == 'torch.spmm'``).

CLI:
    python tools/differential_capture.py --side wrapper --model leaf_mix
    python tools/differential_capture.py --side mode --model leaf_mix
    python tools/differential_capture.py --compare --model leaf_mix
    python tools/differential_capture.py --compare-all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Model corpus
# ---------------------------------------------------------------------------
# Each spec is buildable in a bare subprocess (torch-only construction).
# ``composites`` declares the Python-level composites the model exercises so
# the wrapper process can derive their leaf expansions from probe traces.


@dataclass
class ModelSpec:
    """One differential-corpus entry."""

    name: str
    build: Callable[[], Any]
    make_input: Callable[[], Any]
    composites: dict[str, Callable[[Any], Any]] = field(default_factory=dict)


def _build_leaf_mix() -> Any:
    """Leaf functions, factories, operators, methods, in-place buffer op."""
    import torch
    from torch import nn

    class LeafMix(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.bn = nn.BatchNorm1d(4)

        def forward(self, x: Any) -> Any:
            y = torch.relu(self.lin(x))
            y = self.bn(y)
            y = y + torch.ones_like(y)
            y = y.sigmoid()
            y = y * 2
            y = torch.cat([y, y], dim=1)
            return y.tanh()

    return LeafMix()


def _build_conv_pool() -> Any:
    """Convolutional leaves and shape ops."""
    import torch
    from torch import nn

    class ConvPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3, padding=1)

        def forward(self, x: Any) -> Any:
            y = torch.nn.functional.max_pool2d(self.conv(x), 2)
            y = torch.flatten(y, 1)
            return y.mean(dim=1)

    return ConvPool()


def _build_composite_softsign() -> Any:
    """Python composite whose interior is a leaf span only wrappers see."""
    import torch
    from torch import nn

    class CompositeSoftsign(nn.Module):
        def forward(self, x: Any) -> Any:
            y = torch.nn.functional.softsign(x)
            return y.relu()

    return CompositeSoftsign()


def _input_2x4() -> Any:
    import torch

    torch.manual_seed(0)
    return torch.randn(2, 4)


def _input_img() -> Any:
    import torch

    torch.manual_seed(0)
    return torch.randn(2, 1, 8, 8)


def _softsign_probe(x: Any) -> Any:
    import torch

    return torch.nn.functional.softsign(x)


MODEL_SPECS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec("leaf_mix", _build_leaf_mix, _input_2x4),
        ModelSpec("conv_pool", _build_conv_pool, _input_img),
        ModelSpec(
            "composite_softsign",
            _build_composite_softsign,
            _input_2x4,
            composites={"softsign": _softsign_probe},
        ),
    )
}


# ---------------------------------------------------------------------------
# Mode side (pristine torch — NEVER wraps)
# ---------------------------------------------------------------------------


def _build_mode_name_table() -> dict[int, str]:
    """Id-keyed pristine-callable -> TorchLens-vocabulary name table.

    Mirrors the wrapper installer exactly: same pair inventory
    (``get_orig_torch_funcs``), same first-wins dedup for shared originals
    (``torch.cos`` vs ``torch._VF.cos``). This is the TorchLens-owned
    canonicalizer the safety-net verdict requires instead of
    ``resolve_name``.
    """
    import torch

    from torchlens._constants import get_orig_torch_funcs

    table: dict[int, str] = {}
    for namespace_name, func_name in get_orig_torch_funcs():
        namespace: Any = torch
        ok = True
        for part in namespace_name.split(".")[1:]:
            namespace = getattr(namespace, part, None)
            if namespace is None:
                ok = False
                break
        if not ok:
            continue
        try:
            func = getattr(namespace, func_name, None)
        except AttributeError:
            continue
        if func is not None and callable(func) and id(func) not in table:
            table[id(func)] = func_name
    return table


def run_mode_side(model_name: str) -> dict[str, Any]:
    """Run the forward under a recording TorchFunctionMode on pristine torch."""
    from torch.overrides import TorchFunctionMode

    from torchlens import _state

    if _state._is_decorated:
        raise RuntimeError("mode side requires pristine torch; this process has wrapped torch")

    table = _build_mode_name_table()
    spec = MODEL_SPECS[model_name]
    events: list[dict[str, Any]] = []

    class Recorder(TorchFunctionMode):
        def __torch_function__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            name = table.get(id(func))
            events.append(
                {
                    "name": name,
                    "unmapped_repr": None if name else repr(func),
                }
            )
            return func(*args, **(kwargs or {}))

    model = spec.build()
    model.eval()
    x = spec.make_input()
    # No no_grad: the wrapper side traces with autograd live, so the mode side
    # must run the identical regime or grad-only protocol traffic would skew
    # the differential.
    with Recorder():
        model(x)
    return {"side": "mode", "model": model_name, "events": events}


# ---------------------------------------------------------------------------
# Wrapper side (normal TorchLens capture)
# ---------------------------------------------------------------------------


def run_wrapper_side(model_name: str) -> dict[str, Any]:
    """Run a normal TorchLens trace and emit its op stream + derived tables."""
    import torchlens as tl
    from torchlens.backends.torch.wrappers import funcs_not_to_log

    spec = MODEL_SPECS[model_name]
    model = spec.build()
    model.eval()
    x = spec.make_input()
    trace = tl.trace(model, x)
    ops = [op.func_name for op in trace.ops]

    # Derive composite expansions mechanically: trace a one-op probe per
    # declared composite and keep its non-structural leaf sequence.
    expansions: dict[str, list[str]] = {}
    for comp_name, probe in spec.composites.items():
        from torch import nn

        class _Probe(nn.Module):
            _probe_fn = staticmethod(probe)

            def forward(self, inp: Any) -> Any:
                return self._probe_fn(inp)

        probe_trace = tl.trace(_Probe(), spec.make_input())
        expansions[comp_name] = [
            op.func_name for op in probe_trace.ops if op.func_name not in STRUCTURAL_ROWS
        ]

    return {
        "side": "wrapper",
        "model": model_name,
        "ops": ops,
        "funcs_not_to_log": sorted(funcs_not_to_log),
        "composite_expansions": expansions,
    }


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

STRUCTURAL_ROWS = frozenset({"none", "identity"})
"""TorchLens-synthesized graph rows with no user torch call behind them."""

PINNED_NOT_LOGGED = frozenset({"numpy", "__array__", "size", "dim"})
"""The HARNESS's own copy of the never-logged inventory (b9-sol R75-1).

The oracle's exception authority must not be supplied by the subject under
test: excusing mode events against the wrapper's live ``funcs_not_to_log``
means a capture regression that ADDS an op to that table is silently excused.
Excusal therefore keys on this pin, and any set drift between the pin and the
wrapper-exported table is a loud mismatch (``NOT_LOGGED_AUTHORITY_DRIFT``)
that only a reviewed harness edit can clear."""

PINNED_COMPOSITE_EXPANSIONS: dict[str, tuple[str, ...]] = {
    # softsign(x) = x / (1 + |x|): abs, add, div. Hand-reviewed against the
    # composite's arithmetic, not derived from a capture.
    "softsign": ("__abs__", "__add__", "__truediv__"),
}
"""The HARNESS's own composite leaf spans (b9-sol R75-1 round 4).

The probe-derived expansion runs through the SAME wrappers as the subject
stream, so a systematic wrapper omission (e.g. ``__add__`` never logged)
corrupts the expected span and the observed span identically and the
alignment reads matched -- a tautology. Expansion therefore keys on this
hand-reviewed pin: the live probe derivation is kept ONLY as a
cross-check, and any drift between pin and probe is a loud
``COMPOSITE_AUTHORITY_DRIFT`` mismatch that only a reviewed harness edit
(with the composite's arithmetic re-checked by hand) can clear."""


def _dunder_respell(wrapper_name: str) -> str | None:
    """Public-method spelling of an operator dunder, or None."""
    if wrapper_name.startswith("__") and wrapper_name.endswith("__"):
        inner = wrapper_name[2:-2]
        # Reflected/in-place operator forms keep their prefix; the protocol
        # hands over the plain public method either way (__radd__ -> add).
        if inner.startswith(("r", "i")) and len(inner) > 1:
            return inner[1:]
        return inner
    return None


def align_streams(mode_result: dict[str, Any], wrapper_result: dict[str, Any]) -> dict[str, Any]:
    """Align the two capture streams under the four named rules.

    Returns a report with ``matched`` (bool), the per-event alignment ledger,
    and any mismatches. Every dropped or expanded event appears in the ledger
    with the rule that consumed it — silent consumption is impossible.
    """
    live_not_logged = set(wrapper_result["funcs_not_to_log"])
    expansions = wrapper_result["composite_expansions"]

    ledger: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    # Exception authority is the harness's pin, never the subject's live
    # table (b9-sol R75-1); ANY drift between them fails the alignment.
    added = sorted(live_not_logged - PINNED_NOT_LOGGED)
    removed = sorted(PINNED_NOT_LOGGED - live_not_logged)
    if added or removed:
        mismatches.append(
            {
                "kind": "NOT_LOGGED_AUTHORITY_DRIFT",
                "added_by_subject": added,
                "missing_from_subject": removed,
            }
        )
    not_logged = PINNED_NOT_LOGGED & live_not_logged

    # Composite spans: the probe derivation shares the subject's wrappers, so
    # it can never be the expectation (b9-sol R75-1 round 4: a planted
    # systematic __add__ omission corrupted probe and stream identically and
    # aligned matched=True). The PIN is the authority; the derived span is
    # only a cross-check whose drift fails loudly.
    for comp_name in sorted(set(expansions) | set(PINNED_COMPOSITE_EXPANSIONS)):
        pinned = PINNED_COMPOSITE_EXPANSIONS.get(comp_name)
        derived = expansions.get(comp_name)
        if pinned is None:
            mismatches.append(
                {
                    "kind": "COMPOSITE_AUTHORITY_DRIFT",
                    "name": comp_name,
                    "pinned_span": None,
                    "derived_span": derived,
                    "detail": "subject declares a composite the harness pin does not know",
                }
            )
        elif derived is not None and list(pinned) != list(derived):
            mismatches.append(
                {
                    "kind": "COMPOSITE_AUTHORITY_DRIFT",
                    "name": comp_name,
                    "pinned_span": list(pinned),
                    "derived_span": derived,
                    "detail": "probe-derived span drifted from the hand-reviewed pin",
                }
            )

    wrapper_ops: list[tuple[int, str]] = []
    for i, name in enumerate(wrapper_result["ops"]):
        if name in STRUCTURAL_ROWS:
            ledger.append({"rule": "STRUCTURAL_ROW", "wrapper_index": i, "name": name})
        else:
            wrapper_ops.append((i, name))

    cursor = 0
    for j, event in enumerate(mode_result["events"]):
        name = event["name"]
        if name is None:
            mismatches.append(
                {"kind": "UNMAPPED_MODE_EVENT", "mode_index": j, "repr": event["unmapped_repr"]}
            )
            continue
        if name in not_logged:
            ledger.append({"rule": "NOT_LOGGED", "mode_index": j, "name": name})
            continue
        if name in PINNED_COMPOSITE_EXPANSIONS:
            # Expected span = the PIN, never the subject-derived probe.
            span = list(PINNED_COMPOSITE_EXPANSIONS[name])
            got = [w for _, w in wrapper_ops[cursor : cursor + len(span)]]
            if got == span:
                ledger.append(
                    {
                        "rule": "COMPOSITE_EXPANSION",
                        "mode_index": j,
                        "name": name,
                        "wrapper_span": got,
                    }
                )
                cursor += len(span)
            else:
                mismatches.append(
                    {
                        "kind": "COMPOSITE_SPAN_MISMATCH",
                        "mode_index": j,
                        "name": name,
                        "expected_span": span,
                        "got": got,
                    }
                )
            continue
        if cursor >= len(wrapper_ops):
            mismatches.append({"kind": "WRAPPER_STREAM_EXHAUSTED", "mode_index": j, "name": name})
            continue
        w_index, w_name = wrapper_ops[cursor]
        if w_name == name:
            ledger.append(
                {"rule": "EXACT", "mode_index": j, "wrapper_index": w_index, "name": name}
            )
            cursor += 1
        elif _dunder_respell(w_name) == name:
            ledger.append(
                {
                    "rule": "DUNDER_RESPELL",
                    "mode_index": j,
                    "wrapper_index": w_index,
                    "wrapper_name": w_name,
                    "mode_name": name,
                }
            )
            cursor += 1
        else:
            mismatches.append(
                {
                    "kind": "NAME_MISMATCH",
                    "mode_index": j,
                    "mode_name": name,
                    "wrapper_index": w_index,
                    "wrapper_name": w_name,
                }
            )
            cursor += 1

    for w_index, w_name in wrapper_ops[cursor:]:
        mismatches.append(
            {"kind": "UNCONSUMED_WRAPPER_OP", "wrapper_index": w_index, "name": w_name}
        )

    return {
        "model": mode_result["model"],
        "matched": not mismatches,
        "ledger": ledger,
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Cross-process orchestration
# ---------------------------------------------------------------------------


def _spawn_side(side: str, model_name: str) -> dict[str, Any]:
    """Run one side in a fresh subprocess and parse its JSON stdout."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (repo_root, env.get("PYTHONPATH")) if p)
    proc = subprocess.run(
        [sys.executable, __file__, "--side", side, "--model", model_name],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{side} side failed for {model_name!r} (exit {proc.returncode}):\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def compare_model(model_name: str) -> dict[str, Any]:
    """Spawn both sides for one model and align their streams."""
    mode_result = _spawn_side("mode", model_name)
    wrapper_result = _spawn_side("wrapper", model_name)
    return align_streams(mode_result, wrapper_result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("mode", "wrapper"))
    parser.add_argument("--model", choices=sorted(MODEL_SPECS))
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare-all", action="store_true")
    args = parser.parse_args(argv)

    if args.side:
        if not args.model:
            parser.error("--side requires --model")
        result = run_mode_side(args.model) if args.side == "mode" else run_wrapper_side(args.model)
        json.dump(result, sys.stdout)
        return 0

    names = sorted(MODEL_SPECS) if args.compare_all else [args.model]
    if names == [None]:
        parser.error("--compare requires --model (or use --compare-all)")
    failed = False
    for name in names:
        report = compare_model(name)
        print(json.dumps(report, indent=2))
        failed = failed or not report["matched"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
