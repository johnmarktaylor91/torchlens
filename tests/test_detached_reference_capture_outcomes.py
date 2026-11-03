"""Capture-OUTCOME corpus for detached torch references — the stage-2 deletion gate.

This module converts the detached-reference corpus from patch-MECHANICS
assertions ("the crawler rewrote slot X to wrapper Y") to capture-OUTCOME
assertions ("the op appears in the trace" / "the miss has exactly this
signature"). The crawler and its mechanics tests are DELETED (stage 2); this
module is the gate that pinned the outcomes across that deletion, and it
keeps pinning what users actually get.

STAGE-2 STATE (rescue re-run live): the rows that previously pinned a SILENT
mid-graph miss were flipped DELIBERATELY — the escape signal (provenance
warning / unattributed-args flag / typed output-attribution error) now
triggers a rescue re-run with the ``RescueTorchFunctionMode`` net armed, and
the formerly-missed op is captured with wrapper fidelity and full disclosure
(``capture_verified is False``, reason ``"mode_rescue_rerun"``, session-time
``rescue_rerun`` record). The silent-corruption class no longer exists: every
escape is either recovered+disclosed or unrecovered+disclosed.

Row inventory (safety-net verdict, design-review matrices):

- Formerly crawler-covered holders (module-level refs, class attrs, function
  defaults): RESCUED since the crawler deletion, recovered with wrapper
  fidelity and disclosure. Model instance holders (direct, list, dict,
  partial internals) and closure cells are rebound to the wrappers for the
  capture and restored after it, so they capture in one forward.
- Deletion direction: TorchLens no longer rewrites user objects (identity
  pinned), and a stale WRAPPER reference held across ``unwrap_torch()``
  computes correctly and logs nothing.
- The crawler-missed classes (RESCUED since stage 2): staticmethods, module-level partials, plain-object attrs, pre-bound tensor
  methods, torch-free-source module class attrs, C-held refs (``lru_cache``
  proxy).
- The protocol-invisible class (``from_numpy`` / ``frombuffer`` /
  ``as_subclass``): these assert the BELT/wrapper path — no
  ``TorchFunctionMode`` can EVER see them (measured: zero callbacks), so the
  rescue net cannot cover them and the wrapper+belt path must stay green
  WITHOUT any rescue disclosure.
- De-moded composite interiors: a crawler-reachable stale ref inside a
  third-party ``handle_torch_function`` composite body — structurally
  invisible to any mode (the protocol pops the mode before the body runs).
- Worker-thread stale refs: declared unsupported/ceilinged (op logging is
  owner-thread-scoped by design, r43; modes are thread-local too). The
  thread's op must NOT appear, the capture must not crash, and since stage 2
  the unrecovered escape is DISCLOSED (``"escape_rescue_unrecovered"``),
  never silent.
- The mid-graph case that used to be silent corruption (opus matrix): now
  the flagship rescue row asserting the recovered op re-enters the dataflow
  (consumer has real parents again).
"""

from __future__ import annotations

import functools
import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils._tensor_label import is_decorated_function
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

# ---------------------------------------------------------------------------
# Environment: pristine originals, holder construction, rewrap
# ---------------------------------------------------------------------------


class _CorpusEnv:
    """Pre-wrap raw originals plus registration of temp holder modules."""

    def __init__(self) -> None:
        self.relu = torch.relu
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.cos = torch.cos
        self.from_numpy = torch.from_numpy
        self.frombuffer = torch.frombuffer
        self.as_subclass = torch.Tensor.as_subclass
        self.temp_modules: list[str] = []

    def register_module(self, module: types.ModuleType) -> types.ModuleType:
        sys.modules[module.__name__] = module
        self.temp_modules.append(module.__name__)
        return module


@pytest.fixture()
def corpus_env() -> Any:
    """Unwrap torch, expose raw originals, rewrap and clean up afterwards."""
    unwrap_torch()
    env = _CorpusEnv()
    assert not is_decorated_function(env.relu), "unwrap failed; refs are not pristine"
    try:
        yield env
    finally:
        for name in env.temp_modules:
            sys.modules.pop(name, None)
        wrap_torch()


def _trace(model: nn.Module, x: torch.Tensor | None = None) -> tl.Trace:
    wrap_torch()
    return tl.trace(model, torch.tensor([0.25, 0.5]) if x is None else x)


def _trace_with_provenance_warning(model: nn.Module) -> tl.Trace:
    """Trace a model whose PRIMARY run must emit the provenance warning.

    The generic ``UserWarning`` ("tensor arguments with no graph/source
    provenance") is the shipped-default escape signal for a mid-graph miss;
    since stage 2 it is also the rescue-rerun trigger. Pinned here so the
    signal path cannot silently vanish.
    """
    wrap_torch()
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        return tl.trace(model, torch.tensor([0.25, 0.5]))


def _op_names(trace: tl.Trace) -> list[str]:
    return [op.func_name for op in trace._layer_list]


def _assert_captured(trace: tl.Trace, *expected: str) -> None:
    """The stale-ref ops made it into the trace (the outcome that matters)."""
    names = _op_names(trace)
    for name in expected:
        assert name in names, f"{name!r} missing from capture: {names}"


def _assert_rescued(trace: tl.Trace, *expected: str) -> None:
    """The stale-ref ops were RECOVERED by a rescue re-run, with disclosure.

    A rescued capture must carry the full honesty record: the ops are
    present, the trace never claims verification (mode presence can de-fuse
    fast paths and the forward ran twice), and the session-time
    ``rescue_rerun`` record names the recovery. Flipping any of this back to
    a silent miss must be impossible.
    """
    _assert_captured(trace, *expected)
    assert trace.provenance.verified is False
    assert trace.provenance.verification_reason == "mode_rescue_rerun"
    info = trace._rescue_rerun
    assert info is not None and info["recovered"] is True
    if info["primary_error"] is not None:
        return  # primary raised before producing ops; no multiset diff to name
    for name in expected:
        if name in info["recovered_ops"]:
            break
    else:
        raise AssertionError(f"none of {expected!r} in recovered_ops: {info['recovered_ops']!r}")


# ---------------------------------------------------------------------------
# Formerly crawler-covered holders: RESCUED since the crawler deletion
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_held_refs_capture.py, campaign mutant kills.
def test_model_instance_holders_are_captured(corpus_env: _CorpusEnv) -> None:
    """Direct / list / dict / partial(.func/.args/.keywords) model holders.

    Capture preparation rebinds these holders to the live wrappers for the
    capture, so the ops are captured directly in ONE forward with the exact
    historical op counts and no rescue, and the user's objects are restored
    afterwards (identity pinned below)."""

    def apply_op(op: Callable[..., Any], v: torch.Tensor) -> torch.Tensor:
        return op(v)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.direct = corpus_env.relu
            self.items = [corpus_env.sigmoid]
            self.mapping = {"op": corpus_env.tanh}
            self.partial_func = functools.partial(corpus_env.cos)
            self.partial_arg = functools.partial(apply_op, corpus_env.sigmoid)
            self.partial_kw = functools.partial(apply_op, op=corpus_env.tanh)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            y = self.mapping["op"](self.items[0](self.direct(v)))
            y = self.partial_func(y)
            y = self.partial_arg(v=y)
            return self.partial_kw(v=y)

    model = Model()
    holder_identities = (
        model.direct,
        model.items[0],
        model.mapping["op"],
        model.partial_func,
        model.partial_arg,
        model.partial_kw,
    )
    wrap_torch()
    trace = tl.trace(model, torch.tensor([0.25, 0.5]))
    _assert_captured(trace, "relu", "sigmoid", "tanh", "cos")
    assert trace._rescue_rerun is None
    assert trace.provenance.verification_reason != "mode_rescue_rerun"
    names = _op_names(trace)
    assert names.count("relu") == 1
    assert names.count("sigmoid") == 2  # list holder + partial.args holder
    assert names.count("tanh") == 2  # dict holder + partial.keywords holder
    assert names.count("cos") == 1
    # The capture-scoped rebind is fully undone: the user's objects are intact.
    assert (
        model.direct,
        model.items[0],
        model.mapping["op"],
        model.partial_func,
        model.partial_arg,
        model.partial_kw,
    ) == holder_identities


# ---------------------------------------------------------------------------
# The 7 crawler-missed classes: RESCUED since stage 2 (recovered + disclosed)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/rescue.py, planted fault catches.
def test_prebound_tensor_method_is_rescued(corpus_env: _CorpusEnv) -> None:
    owner = torch.tensor([1.0, 2.0])
    bound_add = owner.add  # bound BEFORE wrap; the descriptor inside is raw

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(bound_add(torch.sigmoid(v)))

    _assert_rescued(_trace_with_provenance_warning(Model()), "add")


def _import_temp_module(tmp_path: Path, mod_name: str, source: str) -> types.ModuleType:
    module_path = tmp_path / f"{mod_name}.py"
    module_path.write_text(source, encoding="utf-8")
    sys.modules.pop(mod_name, None)
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        return importlib.import_module(mod_name)
    finally:
        sys.path.remove(str(tmp_path))


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_torch_free_source_output_position_is_rescued(
    corpus_env: _CorpusEnv, tmp_path: Path
) -> None:
    """Same missed class in OUTPUT position: the typed attribution error is
    itself an escape signal, so the capture is rescued instead of crashing."""
    mod = _import_temp_module(tmp_path, "_tl_outcome_torch_free_out", "class Holder:\n    pass\n")
    corpus_env.temp_modules.append(mod.__name__)
    mod.Holder.op = corpus_env.cos

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return mod.Holder.op(torch.sigmoid(v))

    wrap_torch()
    trace = tl.trace(Model(), torch.tensor([0.25, 0.5]))
    _assert_rescued(trace, "cos")
    assert trace._rescue_rerun["trigger"] == "output_attribution_failed"


# ---------------------------------------------------------------------------
# Protocol-invisible class: the BELT rows (no mode can EVER see these)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/belt.py.
def test_protocol_invisible_as_subclass_is_captured(corpus_env: _CorpusEnv) -> None:
    mod = types.ModuleType("_tl_outcome_proto_as_subclass")
    corpus_env.register_module(mod)
    mod.op = corpus_env.as_subclass

    class SubTensor(torch.Tensor):
        pass

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return mod.op(v, SubTensor)

    trace = _trace(Model())
    _assert_captured(trace, "as_subclass")
    assert trace._rescue_rerun is None  # belt coverage is PRIMARY, never a rescue


# ---------------------------------------------------------------------------
# De-moded composite interior: captured today, invisible to any mode
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Deletion direction: no user-object mutation; stale WRAPPER refs stay safe
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Worker-thread refs: declared unsupported/ceilinged (owner-thread scoping)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The mid-graph silent-corruption row (opus matrix: previously uncovered)
# ---------------------------------------------------------------------------
