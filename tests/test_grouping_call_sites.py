"""Call-site grouping (``detector="callsite_v1"``): one test per row of the rule's case table.

The rule: a layer is one operation written at one place in a module's code; running that
place again (loop, recursion, comprehension) is another pass of the same layer, and an op inside
a module runs once per module call (``m:1``, ``m:2``), so its passes match the module's calls. Two different places are two layers, even with the same
function and the same weights. Argument values, shapes, shared weights and graph shape are
never evidence. Each test traces a tiny model and compares the multiset of
``(func_name, num_passes)`` over its layers with the row's expected answer, written as
``"tanh:3 linear:1x2"`` (one tanh layer with 3 passes, two single-pass linear layers).

Rows 27, 28, 30 and 31 depend on the module-call choice
(``torchlens._utils._call_site.MODULE_CALL_IS_PASS``, decided True) and run under both values.
"""

from __future__ import annotations

import functools
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

import torchlens as tl
from torchlens.backends.torch.wrappers import wrap_torch

# ---------------------------------------------------------------- helpers


def _x(*shape: int) -> Tensor:
    """Fixed input (ones), default shape 2x4."""

    return torch.ones(*(shape or (2, 4)))


def _capture(build: Callable[[], nn.Module], x: Any) -> Any:
    """Build the model under a fixed seed, trace it, and check the call-site detector ran."""

    torch.manual_seed(0)
    model = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x)
    assert trace.options.grouping_policy["detector"] == "callsite_v1"
    return trace


def _layer_shape(trace: Any) -> Counter[tuple[str, int]]:
    """Counter of (func_name, num_passes) over layers, inputs/outputs/buffers excluded."""

    seen: dict[str, tuple[str, int]] = {}
    for op in trace._layer_list:
        if op.is_input or op.is_output or op.is_buffer:
            continue
        seen.setdefault(op.layer_label, (str(op.func_name), int(op.num_passes or 1)))
    return Counter(seen.values())


def _expect(spec: str) -> Counter[tuple[str, int]]:
    """Parse ``"func:passes"`` / ``"func:passesxlayers"`` tokens into a layer shape."""

    shape: Counter[tuple[str, int]] = Counter()
    for token in spec.split():
        func_name, _, rest = token.rpartition(":")
        passes, _, layers = rest.partition("x")
        shape[(func_name, int(passes))] += int(layers or 1)
    return shape


def _shape(build: Callable[[], nn.Module], x: Any = None) -> Counter[tuple[str, int]]:
    """Trace ``build()`` on ``x`` (default 2x4 ones) and return its layer shape."""

    return _layer_shape(_capture(build, _x() if x is None else x))


FORK = pytest.mark.parametrize("module_call_is_pass", [True, False], ids=["module", "caller"])


# ---------------------------------------------------------------- shared models


class Holder(nn.Module):
    """Registers the given children, parameters and plain attributes; subclasses add forward."""

    def __init__(self, **children: Any) -> None:
        super().__init__()
        for name, child in children.items():
            setattr(self, name, child)


class Wrap(Holder):
    """Calls its ``body`` child once."""

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


def _lin() -> nn.Linear:
    return nn.Linear(4, 4)


def _w() -> nn.Parameter:
    return nn.Parameter(torch.randn(4, 4))


def _helper(x: Tensor, w: Tensor) -> Tensor:
    return torch.relu(x @ w) * 0.5


def _sin_helper(x: Tensor) -> Tensor:
    return torch.sin(x)


def _rec(x: Tensor, n: int) -> Tensor:
    if n == 0:
        return x
    return torch.tanh(_rec(x, n - 1)) * 0.9


class RootLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.tanh(x) * 0.5
        return x


class ResidualPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a, self.b = _lin(), _lin()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.a(x)
        x = x + self.b(x)
        return x


class BlockLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = self.block(x)
        return x


class BlocksLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class HelperSites(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = _sin_helper(x)
        return _sin_helper(x)


class SameLineOps(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(torch.sin(x))


class LinearTanh(Holder):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.lin(x))


# ---------------------------------------------------------------- rows 1-8: loops and cells


class SelfChainLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.tanh(x)
        return x


class StraightChainLines(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(x) * 0.5
        x = torch.tanh(x) * 0.5
        return x


class StraightChainOneLine(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(torch.tanh(torch.tanh(x)))


class CellSteps(Holder):
    def forward(self, x: Tensor) -> Tensor:
        h = torch.zeros(x.shape[0], 4)
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h)
        return h


class LSTMLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        h, c = torch.zeros_like(x), torch.zeros_like(x)
        for _ in range(3):
            h, c = self.cell(x, (h, c))
        return h + c


class HandCell(Holder):
    def forward(self, x: Tensor) -> Tensor:
        h = torch.zeros(x.shape[0], 4)
        for t in range(x.shape[1]):
            h = torch.tanh(x[:, t] @ self.w_in + h @ self.w_h + self.b)
        return h


# ---------------------------------------------------------------- rows 9-25: places in code


class SequentialIdenticalOps(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(x)
        x = torch.relu(x)
        return x


class FunctionalReluTwoLines(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x


class HelperTwoSites(Holder):
    def forward(self, x: Tensor) -> Tensor:
        a = _helper(x, self.w1)
        return _helper(a, self.w2)


class HelperOneLoopLine(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = _helper(x, self.w)
        return x


class SinHelperLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = _sin_helper(x)
        return x


class HelperLoopAndOutside(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = _helper(x, self.w)
        for _ in range(3):
            x = _helper(x, self.w)
        return x


class HelperTwoLoops(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            x = _helper(x, self.w)
        for _ in range(3):
            x = _helper(x, self.w) + 0.1
        return x


class LambdaTwoSites(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = lambda t: torch.sinh(t) - 1.0

    def forward(self, x: Tensor) -> Tensor:
        return self.f(self.f(x))


class LambdaHelper(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        f = lambda t: torch.tanh(t) * 2  # noqa: E731
        x = f(x)
        x = f(x) + 1
        for _ in range(2):
            x = f(x)
        return x


class OneLineTwoOps(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x) + torch.tanh(x * 2)


class SameLineLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.sin(torch.sin(x))
        return x


class WeightTying(Holder):
    def forward(self, ids: Tensor) -> Tensor:
        h = torch.tanh(self.emb(ids))
        return F.linear(h, self.emb.weight)


class ManualUnrollSharedParam(Holder):
    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(x @ self.w)
        h = torch.tanh(h @ self.w)
        return h


# ---------------------------------------------------------------- rows 26-33: module identity


class ModuleThriceLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.relu(self.lin(x))
        return x


class ModuleThriceLines(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = torch.relu(x)
        x = self.lin(x)
        x = torch.sigmoid(x)
        x = self.lin(x)
        return x


class ReLUModuleReused(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        return x


class SharedChildParent(Holder):
    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.child(x))


class TwoParents(Holder):
    def forward(self, x: Tensor) -> Tensor:
        return self.p2(self.p1(x))


class SiameseEncoder(Holder):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.lin(x)) * 0.5


class SiameseTwoLines(Holder):
    def forward(self, x: Tensor) -> Tensor:
        a = self.enc(x)
        b = self.enc(x * 3)
        return a - b


class ListComp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        parts = [torch.exp(c) for c in x.chunk(3, dim=0)]
        return torch.cat(parts, dim=0)


class ModuleListComprehension(Holder):
    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([blk(x) for blk in self.blocks])


class GeneratorSumHeads(Holder):
    def forward(self, x: Tensor) -> Tensor:
        return sum(h(x) for h in self.blocks)


# ---------------------------------------------------------------- rows 34-46: control flow


class VmapLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.func.vmap(lambda z: z.sin())(x)
        return x


class GradLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.func.grad(lambda z: z.square().sum())(x)
        return x


# Kept by the shrink-r4 coverage proof: sole branch coverage of backends/torch/_transform_wrappers.py and a sole campaign mutation kill.
def test_row35_func_grad_boundary_in_loop() -> None:
    """Row 35: a `torch.func.grad` boundary in a loop gives one boundary layer 3p."""
    assert _shape(GradLoop) == _expect("grad:3")


class TinyLM(Holder):
    def forward(self, ids: Tensor) -> Tensor:
        h = torch.tanh(self.lin(self.emb(ids)))
        return F.linear(h, self.emb.weight)


class GenerateLoop(Holder):
    def forward(self, ids: Tensor) -> Tensor:
        for _ in range(3):
            logits = self.lm(ids)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


class NestedLoopsNoModule(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            for _ in range(3):
                x = torch.cos(x)
            x = x * 0.5
        return x


class NestedWithModule(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            for _ in range(3):
                x = torch.tanh(self.lin(x))
            x = x * 0.5
        return x


class EarlyExit(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(8):
            x = x * 0.5
            if x.sum() < 1:
                break
        return x


class ACTHalting(Holder):
    def forward(self, x: Tensor) -> Tensor:
        p_total = torch.zeros(x.shape[0], 1)
        out = torch.zeros_like(x)
        for _ in range(5):
            x = torch.tanh(self.lin(x))
            p = torch.sigmoid(self.halt(x)) + 0.4
            out = out + p * x
            p_total = p_total + p
            if bool((p_total > 2.5).all()):
                break
        return out


def _act_halting() -> nn.Module:
    halt = nn.Linear(4, 1)
    nn.init.zeros_(halt.weight)  # p = sigmoid(0) + 0.4 = 0.9 per step: halts after 3 steps
    nn.init.zeros_(halt.bias)
    return ACTHalting(lin=_lin(), halt=halt)


# Kept by the shrink-r4 coverage proof: sole branch coverage of _postprocess/ast_branches.py.
def test_row40_act_halting_loop() -> None:
    """Row 40: an ACT halting loop gives every body op Np (N = 3 here)."""
    expected = "zeros:1 zeros_like:1 linear:3x2 tanh:3 sigmoid:3 __add__:3x3 __mul__:3 __gt__:3"
    assert _shape(_act_halting) == _expect(f"{expected} all:3")


class CondSomeIters(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for i in range(4):
            x = torch.tanh(x)
            if i % 2 == 0:
                x = x * 2
        return x


class CondDifferentBranches(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for i in range(4):
            if i % 2 == 0:  # noqa: SIM108 -- the case is two lines, two call sites
                x = torch.relu(x) + 0.1
            else:
                x = torch.sigmoid(x)
            x = x * 0.9
        return x


class WhileLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        n = 0
        while n < 3:
            x = torch.tanh(x) * 0.5
            n += 1
        return x


class WhileData(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        while x.sum() > 1:
            x = x * 0.5
        return x


class Recursion(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return _rec(x, 3)


class RecursiveLoop(nn.Module):
    def recur(self, x: Tensor, n: int) -> Tensor:
        if n == 0:
            return x
        return self.recur(torch.sin(x), n - 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.recur(x, 3)


class RecursiveTwoSites(RecursiveLoop):
    def forward(self, x: Tensor) -> Tensor:
        x = self.recur(x, 2)
        return self.recur(x, 3)


# ---------------------------------------------------------------- rows 47-62: state and data


class InplaceBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(2, 4))

    def forward(self, x: Tensor) -> Tensor:
        self.state.zero_()
        for t in range(3):
            self.state.mul_(0.9).add_(torch.tanh(x) * (t + 1))
        return self.state * 1.0


# Kept by the shrink-r4 coverage proof: a sole campaign mutation kill.
def test_row47_inplace_buffer_update_per_iteration() -> None:
    """Row 47: an in-place buffer update per iteration gives body ops Np."""
    expected = "zero_:1 mul_:3 tanh:3 __mul__:3 add_:3 __mul__:1"
    assert _shape(InplaceBuffer) == _expect(expected)


class BufferTwoSites(InplaceBuffer):
    def forward(self, x: Tensor) -> Tensor:
        self.state.add_(x)
        self.state.add_(x)
        return self.state.clone()


class VariableScalarLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for i in range(1, 4):
            x = x * i
        return x


class ChangingShapeLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.cat((x, x), dim=0)
        return x


class ParamListLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for w in self.ws:
            x = torch.tanh(x @ w)
        return x


class ModuleListGlue(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = x + blk(x)
        return x


class CheckpointTwice(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = checkpoint(self.block, x, use_reentrant=False)
        x = checkpoint(self.block, x * 2, use_reentrant=False)
        return x


class HookedLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.tanh(self.lin(x))
        return x


class MultipleOutputs(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for _ in range(3):
            values, indexes = torch.max(x, dim=1)
            outputs.append(values + indexes)
        return torch.stack(outputs)


# Kept by the shrink-r4 coverage proof: a sole campaign mutation kill.
def test_row55_multi_output_max_in_loop() -> None:
    """Row 55: multi-output `max` in a loop gives values Np and indices Np."""
    assert _shape(MultipleOutputs) == _expect("max:3x2 __add__:3 stack:1")


class DynamicDispatch(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for fn in (torch.sin, torch.cos, torch.sin):
            x = fn(x)
        return x


_GEN_SOURCE = "def f(x):\n    return torch.sin(x)\n"


class ExecLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        scope = {"x": x, "torch": torch}
        exec("for _ in range(3):\n    x = torch.sin(x)", scope)
        return scope["x"]


class ExecBuiltLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            scope: dict[str, Any] = {"torch": torch}
            exec(compile(_GEN_SOURCE, "<tl-gen>", "exec"), scope)  # recompiled every step
            x = scope["f"](x)
        return x


class IdenticalCompiledTwoSites(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        a: dict[str, Any] = {"torch": torch}
        b: dict[str, Any] = {"torch": torch}
        exec(compile(_GEN_SOURCE, "<tl-same>", "exec"), a)
        exec(compile(_GEN_SOURCE, "<tl-same>", "exec"), b)
        x = a["f"](x)
        x = b["f"](x)
        return x


# Kept by the shrink-r4 coverage proof: sole branch coverage of _postprocess/ast_branches.py.
def test_row58_identical_compiled_functions_from_two_sites() -> None:
    """Row 58: two separately compiled identical functions from two sites stay separate."""
    assert _shape(IdenticalCompiledTwoSites) == _expect("sin:1x2")


class StepZeroSpecial(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for t in range(3):
            if t == 0:  # noqa: SIM108 -- the case is two lines, two call sites
                h = torch.tanh(x[:, t] @ self.w)
            else:
                h = torch.tanh(x[:, t] @ self.w + h)
        return h


class InnerLoopBlock(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            x = torch.tanh(self.lin(x))
        return x


class BlockTwice(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        x = self.block(x * 2)
        return x


class ResidualVQLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = torch.zeros_like(x)
        for _ in range(3):
            d = torch.cdist(residual, self.codebook)
            idx = d.argmin(-1)
            q = self.codebook[idx]
            out = out + q
            residual = residual - q
        return out


class IntegrateAndFire(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        v = torch.zeros_like(x[0])
        spikes = []
        for t in range(x.shape[0]):
            v = v + x[t]
            s = (v > 0.5).float()
            v = v - s
            spikes.append(s)
        return torch.stack(spikes)


# ---------------------------------------------------------------- rows 63-71: edges


class ScriptChild(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x) * 0.5


class ChildLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = self.child(x)
        return x


# Kept by the shrink-r4 coverage proof: sole branch coverage of _capture/prepare/door.py.
@pytest.mark.skipif(
    not hasattr(torch.jit, "trace"), reason="torch.jit.trace is retired on this torch"
)
def test_row63_torchscript_submodule_in_loop() -> None:
    """Row 63: a TorchScript submodule in a loop is refused typed, never captured.

    Row 63's "boundary op Np" presumes TorchLens captures a TorchScript submodule
    as a boundary op. It does not: logging TorchScript artifacts is out of
    contract (AGENTS.md anti-patterns) and capture refuses the submodule up
    front. Feature-detected, never skipped from an except handler (proofnet
    memo D4): the refusal is pinned, so a capture that starts accepting
    TorchScript fails here and must pin the row's pass count instead.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scripted = torch.jit.trace(ScriptChild(), _x())
    with pytest.raises(RuntimeError, match="does not support torch.jit ScriptModule"):
        _shape(lambda: ChildLoop(child=scripted))


class NoPythonFrame(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = x * 2.0
        for _ in range(2):
            x = self.act(x)
        return self.total([x, x, x])


def _no_python_frame() -> nn.Module:
    act, total = nn.Module(), nn.Module()
    act.forward = torch.relu  # type: ignore[method-assign]
    total.forward = functools.partial(functools.reduce, torch.add)  # type: ignore[method-assign]
    return NoPythonFrame(act=act, total=total)


# Kept by the shrink-r4 coverage proof: sole branch coverage of backends/torch/_session_cleanup.py, backends/torch/_session_forwards.py and more and a sole campaign mutation kill.
def test_row64_no_python_frame_inside_module_call() -> None:
    """Row 64: no Python frame inside a module call falls back to position: a builtin
    forward called twice is one layer 2p; two adds in one C-level call stay two layers."""
    wrap_torch()  # the builtin forwards must hold the wrapped torch functions
    assert _shape(_no_python_frame) == _expect("__mul__:1 relu:2 add:1x2")


class MapAndSum(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        xs = [x, x * 2.0, x * 3.0]
        ys = list(map(torch.sin, xs))
        return sum(ys)


class SinPlusSin(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x) + torch.sin(x)


class GeneratorYieldedModules(Holder):
    def _each(self) -> Iterator[nn.Module]:
        yield from self.blocks

    def forward(self, x: Tensor) -> Tensor:
        for blk in self._each():
            x = blk(x)
        return x


def _call_twice(f: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    @functools.wraps(f)
    def wrapper(x: Tensor) -> Tensor:
        y = f(x)
        return f(y)

    return wrapper


@_call_twice
def _sin_double(x: Tensor) -> Tensor:
    return torch.sin(x) * 2.0


class DecoratedTwice(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return _sin_double(x)


class MapLambda(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        xs = [x, x * 2.0, x * 3.0]
        # The row is about map itself, so the comprehension rewrite does not apply.
        ys = list(map(lambda t: torch.sin(t), xs))  # noqa: C417
        return torch.stack(ys)


class AttentionLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            x = x + self.attn(x, x, x, need_weights=False)[0]
        return x


class SeqLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = self.hooked(x)
        return x


# Kept by the shrink-r4 coverage proof: a sole campaign mutation kill.
def test_row73_layer_over_varying_weights_lists_every_weight() -> None:
    """Row 73: a layer whose passes use different weights lists all of them, and validates."""
    torch.manual_seed(0)
    model = ParamListLoop(ws=nn.ParameterList([_w() for _ in range(3)]))
    trace = tl.trace(model, _x())
    (matmul,) = [layer for layer in trace._layer_logs.values() if layer.func_name == "__matmul__"]
    assert [pl.address for pl in matmul._param_logs] == ["ws.0", "ws.1", "ws.2"]
    assert [[pl.address for pl in op._param_logs] for op in matmul.ops.values()] == [
        ["ws.0"],
        ["ws.1"],
        ["ws.2"],
    ]
    assert matmul.num_params == 48
    assert matmul.annotations["varying_across_passes"]["params"] == [["ws.0"], ["ws.1"], ["ws.2"]]
    assert tl.validate(model, _x(), scope="forward", seed=0, validate_metadata=True)


class BufferOnce(InplaceBuffer):
    def forward(self, x: Tensor) -> Tensor:
        self.state.add_(x)
        return self.state * 1.0


class BufferLoop(InplaceBuffer):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            self.state.add_(1.0)
            x = x + self.state
        return x


# Kept by the shrink-r4 coverage proof: sole branch coverage of _postprocess/_buffer_addresses.py.
def test_row74_batchnorm_running_stats_stay_separate_layers() -> None:
    """Row 74: BatchNorm writes each running stat once, so each version is its own layer."""
    trace = _capture(lambda: nn.BatchNorm1d(4).train(), _x())
    buffers = {op.layer_label: op for op in trace._layer_list if op.is_buffer}
    assert buffers, "BatchNorm in train mode logs its running stats"
    assert {int(op.num_passes or 1) for op in buffers.values()} == {1}
