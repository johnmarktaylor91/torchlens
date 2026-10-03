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
  defaults, model instance holders incl. partial internals): RESCUED since
  the crawler deletion — recovered with wrapper fidelity and disclosure, and
  the user's objects are never mutated.
- Deletion direction: TorchLens no longer rewrites user objects (identity
  pinned), and a stale WRAPPER reference held across ``unwrap_torch()``
  computes correctly and logs nothing.
- The 7 crawler-missed classes (RESCUED since stage 2): closure cells,
  staticmethods, module-level partials, plain-object attrs, pre-bound tensor
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
import threading
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import is_decorated_function
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

pytestmark = pytest.mark.smoke

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
    return [op.func_name for op in trace.ops]


def _escape_count(trace: tl.Trace) -> int:
    return len(getattr(trace, "escape_diagnostics", []) or [])


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
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "mode_rescue_rerun"
    info = trace.rescue_rerun
    assert info is not None and info["recovered"] is True
    if info["primary_error"] is not None:
        return  # primary raised before producing ops; no multiset diff to name
    for name in expected:
        if name in info["recovered_ops"]:
            break
    else:
        raise AssertionError(f"none of {expected!r} in recovered_ops: {info['recovered_ops']!r}")


def _assert_unrecovered_disclosed(trace: tl.Trace, missing: str) -> None:
    """The escape stands (beyond any mode) but is DISCLOSED, never silent."""
    assert missing not in _op_names(trace)
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "escape_rescue_unrecovered"
    info = trace.rescue_rerun
    assert info is not None and info["recovered"] is False


# ---------------------------------------------------------------------------
# Formerly crawler-covered holders: RESCUED since the crawler deletion
# ---------------------------------------------------------------------------


def test_module_class_and_default_refs_are_captured(corpus_env: _CorpusEnv) -> None:
    """Module-level, class-attr, and function-default stale refs are traced.

    DELIBERATE FLIP (crawler deletion): these holders are no longer patched
    in place; the escape signal triggers the rescue re-run and the ops are
    recovered with disclosure instead."""
    mod = types.ModuleType("_tl_outcome_covered_holders")
    corpus_env.register_module(mod)
    exec(
        """
from typing import Any
from torch import relu, sigmoid, tanh

module_ref = relu


class Holder:
    class_ref = sigmoid


def uses_default(x: Any, op: Any = tanh) -> Any:
    return op(x)
""",
        mod.__dict__,
    )

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return mod.uses_default(mod.Holder.class_ref(mod.module_ref(v)))

    wrap_torch()
    trace = tl.trace(Model(), torch.tensor([0.25, 0.5]))
    _assert_rescued(trace, "relu", "sigmoid", "tanh")


def test_model_instance_holders_are_captured(corpus_env: _CorpusEnv) -> None:
    """Direct / list / dict / partial(.func/.args/.keywords) model holders.

    DELIBERATE FLIP (crawler deletion): recovered via rescue with the exact
    historical op counts, and — the deletion's point — the user's objects are
    never rewritten (identity pinned below)."""

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
    _assert_rescued(trace, "relu", "sigmoid", "tanh", "cos")
    names = _op_names(trace)
    assert names.count("relu") == 1
    assert names.count("sigmoid") == 2  # list holder + partial.args holder
    assert names.count("tanh") == 2  # dict holder + partial.keywords holder
    assert names.count("cos") == 1
    # The deletion's point: TorchLens never rewrites the user's objects.
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


def test_closure_cell_ref_is_rescued(corpus_env: _CorpusEnv) -> None:
    raw_cos = corpus_env.cos

    def make_closure() -> Callable[[torch.Tensor], torch.Tensor]:
        def invoke(v: torch.Tensor) -> torch.Tensor:
            return raw_cos(v)

        return invoke

    closure = make_closure()

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(closure(torch.sigmoid(v)))

    _assert_rescued(_trace_with_provenance_warning(Model()), "cos")


def test_staticmethod_ref_is_rescued(corpus_env: _CorpusEnv) -> None:
    class Holder:
        static = staticmethod(corpus_env.cos)

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(Holder.static(torch.sigmoid(v)))

    _assert_rescued(_trace_with_provenance_warning(Model()), "cos")


def test_module_level_partial_is_rescued(corpus_env: _CorpusEnv) -> None:
    mod = types.ModuleType("_tl_outcome_module_partial")
    corpus_env.register_module(mod)
    mod.partial = functools.partial(corpus_env.cos)

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(mod.partial(torch.sigmoid(v)))

    _assert_rescued(_trace_with_provenance_warning(Model()), "cos")


def test_plain_object_attr_is_rescued(corpus_env: _CorpusEnv) -> None:
    class Plain:
        pass

    holder = Plain()
    holder.op = corpus_env.cos

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(holder.op(torch.sigmoid(v)))

    _assert_rescued(_trace_with_provenance_warning(Model()), "cos")


def test_prebound_tensor_method_is_rescued(corpus_env: _CorpusEnv) -> None:
    owner = torch.tensor([1.0, 2.0])
    bound_add = owner.add  # bound BEFORE wrap; the descriptor inside is raw

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(bound_add(torch.sigmoid(v)))

    _assert_rescued(_trace_with_provenance_warning(Model()), "add")


def test_c_held_ref_is_rescued(corpus_env: _CorpusEnv) -> None:
    """``lru_cache`` C-level storage as the proxy for a compiled extension."""
    raw_cos = corpus_env.cos

    @functools.lru_cache(maxsize=1)
    def held() -> Callable[..., Any]:
        return raw_cos

    held()  # populate the C-held cache slot pre-wrap

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(held()(torch.sigmoid(v)))

    _assert_rescued(_trace_with_provenance_warning(Model()), "cos")


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


def test_torch_free_source_class_attr_is_rescued(corpus_env: _CorpusEnv, tmp_path: Path) -> None:
    """Torch-free FILE-BACKED source: the source gate skips the deep scan."""
    mod = _import_temp_module(tmp_path, "_tl_outcome_torch_free_mid", "class Holder:\n    pass\n")
    corpus_env.temp_modules.append(mod.__name__)
    mod.Holder.op = corpus_env.cos

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(mod.Holder.op(torch.sigmoid(v)))

    _assert_rescued(_trace_with_provenance_warning(Model()), "cos")


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
    assert trace.rescue_rerun["trigger"] == "output_attribution_failed"


# ---------------------------------------------------------------------------
# Protocol-invisible class: the BELT rows (no mode can EVER see these)
# ---------------------------------------------------------------------------


def test_protocol_invisible_from_numpy_is_captured(corpus_env: _CorpusEnv) -> None:
    mod = types.ModuleType("_tl_outcome_proto_from_numpy")
    corpus_env.register_module(mod)
    mod.op = corpus_env.from_numpy
    arr = np.array([0.25, 0.5], dtype=np.float32)

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return mod.op(arr) + v

    trace = _trace(Model())
    _assert_captured(trace, "from_numpy", "__add__")
    assert trace.rescue_rerun is None  # belt coverage is PRIMARY, never a rescue


def test_protocol_invisible_frombuffer_is_captured(corpus_env: _CorpusEnv) -> None:
    mod = types.ModuleType("_tl_outcome_proto_frombuffer")
    corpus_env.register_module(mod)
    mod.op = corpus_env.frombuffer
    buf = bytearray(np.array([0.25, 0.5], dtype=np.float32).tobytes())

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return mod.op(buf, dtype=torch.float32) + v

    trace = _trace(Model())
    _assert_captured(trace, "frombuffer", "__add__")
    assert trace.rescue_rerun is None  # belt coverage is PRIMARY, never a rescue


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
    assert trace.rescue_rerun is None  # belt coverage is PRIMARY, never a rescue


# ---------------------------------------------------------------------------
# De-moded composite interior: captured today, invisible to any mode
# ---------------------------------------------------------------------------


def test_demoded_composite_interior_ref_is_disclosed_unrecovered(
    corpus_env: _CorpusEnv,
) -> None:
    """A stale ref inside a third-party protocol composite body.

    The composite has torch's own ``has_torch_function`` /
    ``handle_torch_function`` shape, so under ANY TorchFunctionMode its body
    runs with the mode popped — the rescue net can never see ``interior``.

    DELIBERATE FLIP (crawler deletion, verdict disposition D): the crawler
    used to patch the module global; now the escape signal fires, the rescue
    is attempted, recovers nothing, and the standing escape is DISCLOSED
    (silence-vs-loud was the recorded design-review criterion; belt-extension only
    if real-world hits appear).
    """
    mod = types.ModuleType("_tl_outcome_composite_interior")
    corpus_env.register_module(mod)
    mod.__dict__.update(
        {
            "has_torch_function": torch.overrides.has_torch_function,
            "handle_torch_function": torch.overrides.handle_torch_function,
            "interior": corpus_env.cos,
        }
    )
    exec(
        """
def protocol_composite(x):
    if has_torch_function((x,)):
        return handle_torch_function(protocol_composite, (x,), x)
    return interior(x)
""",
        mod.__dict__,
    )

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(mod.protocol_composite(v))

    trace = _trace_with_provenance_warning(Model())
    assert "relu" in _op_names(trace)
    _assert_unrecovered_disclosed(trace, "cos")


# ---------------------------------------------------------------------------
# Deletion direction: no user-object mutation; stale WRAPPER refs stay safe
# ---------------------------------------------------------------------------


def test_stale_wrapper_ref_after_unwrap_passes_through(corpus_env: _CorpusEnv) -> None:
    """A wrapper reference held across ``unwrap_torch()`` stays computable.

    The historical reverse-direction class: user code grabs ``torch.relu``
    while TorchLens is wrapped, then torch is unwrapped. The wrapper gates on
    the logging toggle, so the stale wrapper computes normally and logs
    nothing; with the crawler deleted TorchLens also no longer plants wrapper
    references inside user objects in the first place.
    """
    wrap_torch()
    stale_wrapper = torch.relu
    assert is_decorated_function(stale_wrapper)
    unwrap_torch()
    x = torch.tensor([-1.0, 2.0])
    assert torch.equal(stale_wrapper(x), torch.tensor([0.0, 2.0]))
    wrap_torch()


# ---------------------------------------------------------------------------
# Worker-thread refs: declared unsupported/ceilinged (owner-thread scoping)
# ---------------------------------------------------------------------------


def test_worker_thread_ref_is_not_logged_and_capture_survives(corpus_env: _CorpusEnv) -> None:
    """Op logging is owner-thread-scoped by design (r43): the worker thread's
    op must NOT enter the trace (no false attribution), and the capture must
    complete. Modes are thread-local too, so stage 2 inherits this ceiling —
    the row pins the declared behavior, not a gap to fix."""
    raw_cos = corpus_env.cos

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            result: dict[str, torch.Tensor] = {}

            def work() -> None:
                result["r"] = raw_cos(v.detach())

            worker = threading.Thread(target=work)
            worker.start()
            worker.join()
            return torch.relu(v) + result["r"].sum()

    # The thread-made tensor re-enters the owner thread with no provenance,
    # so the mid-graph escape signal fires and a rescue is ATTEMPTED — but
    # modes are thread-local, so the re-run recovers nothing. Since stage 2
    # the standing escape is disclosed instead of silent.
    trace = _trace_with_provenance_warning(Model())
    names = _op_names(trace)
    assert "relu" in names and "sum" in names and "__add__" in names
    _assert_unrecovered_disclosed(trace, "cos")


# ---------------------------------------------------------------------------
# The mid-graph silent-corruption row (opus matrix: previously uncovered)
# ---------------------------------------------------------------------------


def test_midgraph_escape_is_rescued_full_signature(corpus_env: _CorpusEnv) -> None:
    """The flagship rescue row: a mid-graph escape re-enters the dataflow.

    ``sigmoid -> [escaped cos] -> relu``: before stage 2 this was SILENT
    corruption (cos absent, relu a parentless internal-source op, no claim,
    zero diagnostics). Now the escape signal triggers the rescue re-run: cos
    is captured between sigmoid and relu, relu has a real parent again, and
    the trace carries the full disclosure. Flipping this row back to a
    silent miss must be impossible."""
    raw_cos = corpus_env.cos

    def make_closure() -> Callable[[torch.Tensor], torch.Tensor]:
        def invoke(v: torch.Tensor) -> torch.Tensor:
            return raw_cos(v)

        return invoke

    closure = make_closure()

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(closure(torch.sigmoid(v)))

    trace = _trace_with_provenance_warning(Model())
    assert _op_names(trace) == ["none", "sigmoid", "cos", "relu", "none"]
    relu_op = [op for op in trace.ops if op.func_name == "relu"][0]
    assert relu_op.parents != ()
    assert not relu_op.is_internal_source
    _assert_rescued(trace, "cos")
    assert trace.rescue_rerun["trigger"] == "unattributed_tensor_args"
    assert trace.rescue_rerun["residual_signal"] is None
    assert _escape_count(trace) == 0
