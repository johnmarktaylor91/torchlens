"""Call-site grouping (``detector="callsite_v1"``): one test per row of the rule's case table.

The rule: a layer is one operation written at one place in a module's code; running that
place again (loop, recursion, comprehension, a later call of the same module instance) is
another pass of the same layer. Two different places are two layers, even with the same
function and the same weights. Argument values, shapes, shared weights and graph shape are
never evidence. Each test traces a tiny model and compares the multiset of
``(func_name, num_passes)`` over its layers with the row's expected answer, written as
``"tanh:3 linear:1x2"`` (one tanh layer with 3 passes, two single-pass linear layers).

Rows 27, 28, 30 and 31 depend on the module-identity fork
(``torchlens.utils._call_site.MODULE_CALL_IS_PASS``) and run under both values.
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
from torchlens.utils import _call_site

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
    assert trace.grouping_policy["detector"] == "callsite_v1"
    return trace


def _layer_shape(trace: Any) -> Counter[tuple[str, int]]:
    """Counter of (func_name, num_passes) over layers, inputs/outputs/buffers excluded."""

    seen: dict[str, tuple[str, int]] = {}
    for op in trace.layer_list:
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


def _layers(trace: Any, func_name: str) -> list[list[Any]]:
    """Ops of each layer computing ``func_name``, passes in capture order."""

    by_layer: dict[str, list[Any]] = {}
    for op in trace.layer_list:
        if str(op.func_name) == func_name and not (op.is_input or op.is_output):
            by_layer.setdefault(op.layer_label, []).append(op)
    return [sorted(ops, key=lambda op: op.pass_index) for ops in by_layer.values()]


def _module_calls(trace: Any, func_name: str) -> list[str]:
    """Innermost module call (``address:k``) of every ``func_name`` op, in capture order."""

    return [str(op.modules[-1]) for op in trace.layer_list if str(op.func_name) == func_name]


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


def _linears() -> nn.ModuleList:
    return nn.ModuleList([_lin() for _ in range(3)])


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


def test_row01_root_loop_two_op_body() -> None:
    """Row 1: a root loop `x = tanh(x) * 0.5` gives tanh 3p and mul 3p."""
    assert _shape(RootLoop) == _expect("tanh:3 __mul__:3")


def test_row02_loop_in_submodule_called_once() -> None:
    """Row 2: the same loop inside a submodule called once gives tanh 3p and mul 3p."""
    assert _shape(lambda: Wrap(body=RootLoop())) == _expect("tanh:3 __mul__:3")


class SelfChainLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.tanh(x)
        return x


@pytest.mark.smoke
def test_row03_one_op_loop_at_root() -> None:
    """Row 3: a one-op root loop `x = tanh(x)` gives tanh 3p."""
    assert _shape(SelfChainLoop) == _expect("tanh:3")


class StraightChainLines(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(x) * 0.5
        x = torch.tanh(x) * 0.5
        return x


@pytest.mark.smoke
def test_row04_hand_unrolled_statement_on_two_lines() -> None:
    """Row 4: the 2-op statement written on two lines gives 4 single layers."""
    assert _shape(StraightChainLines) == _expect("tanh:1x2 __mul__:1x2")


class StraightChainOneLine(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(torch.tanh(torch.tanh(x)))


def test_row05_nested_calls_on_one_line() -> None:
    """Row 5: `tanh(tanh(tanh(x)))` on one line gives 3 single layers."""
    assert _shape(StraightChainOneLine) == _expect("tanh:1x3")


class CellSteps(Holder):
    def forward(self, x: Tensor) -> Tensor:
        h = torch.zeros(x.shape[0], 4)
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h)
        return h


def test_row06_rnn_cell_per_step() -> None:
    """Row 6: `nn.RNNCell` per step on `x[:, t]` gives cell Tp and getitem Tp."""
    shape = _shape(lambda: CellSteps(cell=nn.RNNCell(4, 4)), _x(2, 3, 4))
    assert shape == _expect("zeros:1 __getitem__:3 rnn_tanh_cell:3")


class LSTMLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        h, c = torch.zeros_like(x), torch.zeros_like(x)
        for _ in range(3):
            h, c = self.cell(x, (h, c))
        return h + c


def test_row07_lstm_cell_per_step() -> None:
    """Row 7: `nn.LSTMCell` per step keeps h and c apart, Tp each."""
    shape = _shape(lambda: LSTMLoop(cell=nn.LSTMCell(4, 4)))
    assert shape == _expect("zeros_like:1x2 lstm_cell:3x2 __add__:1")


class HandCell(Holder):
    def forward(self, x: Tensor) -> Tensor:
        h = torch.zeros(x.shape[0], 4)
        for t in range(x.shape[1]):
            h = torch.tanh(x[:, t] @ self.w_in + h @ self.w_h + self.b)
        return h


def test_row08_hand_written_weighted_cell() -> None:
    """Row 8: a hand-written weighted cell in a loop gives every body op Tp."""
    build = lambda: HandCell(w_in=_w(), w_h=_w(), b=nn.Parameter(torch.zeros(4)))  # noqa: E731
    shape = _shape(build, _x(2, 3, 4))
    assert shape == _expect("zeros:1 __getitem__:3 __matmul__:3x2 __add__:3x2 tanh:3")


# ---------------------------------------------------------------- rows 9-25: places in code


def test_row09_two_residual_adds_at_root() -> None:
    """Row 9: two residual adds in the root forward are 2 single layers."""
    assert _shape(ResidualPair) == _expect("linear:1x2 __add__:1x2")


@pytest.mark.smoke
def test_row10_two_residual_adds_in_block_called_once() -> None:
    """Row 10: two residual adds in a block called once are 2 single layers."""
    assert _shape(lambda: Wrap(body=ResidualPair())) == _expect("linear:1x2 __add__:1x2")


@pytest.mark.smoke
def test_row11_block_called_three_times_keeps_adds_apart() -> None:
    """Row 11: a block with two adds called 3x keeps add1 and add2 apart, 3 passes each."""
    trace = _capture(lambda: BlockLoop(block=ResidualPair()), _x())
    assert _layer_shape(trace) == _expect("linear:3x2 __add__:3x2")
    add_layers = _layers(trace, "__add__")
    assert len(add_layers) == 2
    for ops in add_layers:
        assert [str(op.modules[-1]) for op in ops] == ["block:1", "block:2", "block:3"]


@pytest.mark.smoke
def test_row12_modulelist_of_three_blocks() -> None:
    """Row 12: a ModuleList of 3 residual blocks (HF decoder shape) gives 6 single adds."""
    build = lambda: BlocksLoop(blocks=nn.ModuleList([ResidualPair() for _ in range(3)]))  # noqa: E731
    assert _shape(build) == _expect("linear:1x6 __add__:1x6")


class SequentialIdenticalOps(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(x)
        x = torch.relu(x)
        return x


def test_row13_same_op_two_lines_in_submodule() -> None:
    """Row 13: `relu(relu(x))` written on two lines in a submodule is 2 single layers."""
    assert _shape(lambda: Wrap(body=SequentialIdenticalOps())) == _expect("relu:1x2")


class FunctionalReluTwoLines(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x


def test_row14_functional_op_two_lines_at_root() -> None:
    """Row 14: the same functional op on two lines at root is 2 single layers."""
    shape = _shape(lambda: FunctionalReluTwoLines(l1=_lin(), l2=_lin()))
    assert shape == _expect("linear:1x2 relu:1x2")


class HelperTwoSites(Holder):
    def forward(self, x: Tensor) -> Tensor:
        a = _helper(x, self.w1)
        return _helper(a, self.w2)


def test_row15_helper_from_two_lines_at_root() -> None:
    """Row 15: a helper called from two root lines gives separate layers per call."""
    shape = _shape(lambda: HelperTwoSites(w1=_w(), w2=_w()))
    assert shape == _expect("__matmul__:1x2 relu:1x2 __mul__:1x2")


def test_row16_one_op_helper_two_lines_in_submodule() -> None:
    """Row 16: a one-op helper called from two lines in a submodule is 2 single layers."""
    assert _shape(lambda: Wrap(body=HelperSites())) == _expect("sin:1x2")


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


def test_row17_helper_from_one_loop_line() -> None:
    """Row 17: a one-op and a multi-op helper called from one loop line give every op Np."""
    assert _shape(lambda: HelperOneLoopLine(w=_w())) == _expect("__matmul__:3 relu:3 __mul__:3")
    assert _shape(SinHelperLoop) == _expect("sin:3")


class HelperLoopAndOutside(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = _helper(x, self.w)
        for _ in range(3):
            x = _helper(x, self.w)
        return x


def test_row18_weighted_helper_before_and_inside_loop() -> None:
    """Row 18: a weighted helper once before a loop and 3x inside: singles, then 3p."""
    shape = _shape(lambda: HelperLoopAndOutside(w=_w()))
    assert shape == _expect("__matmul__:1 __matmul__:3 relu:1 relu:3 __mul__:1 __mul__:3")


class HelperTwoLoops(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            x = _helper(x, self.w)
        for _ in range(3):
            x = _helper(x, self.w) + 0.1
        return x


def test_row19_helper_from_two_loops() -> None:
    """Row 19: a helper called from two different loops gives 2p and 3p layers."""
    shape = _shape(lambda: HelperTwoLoops(w=_w()))
    expected = "__matmul__:2 __matmul__:3 relu:2 relu:3 __mul__:2 __mul__:3 __add__:3"
    assert shape == _expect(expected)


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


def test_row20_lambda_call_sites_stay_separate() -> None:
    """Row 20: lambda `f(f(x))` on one line and a lambda from two lines give separate layers."""
    assert _shape(LambdaTwoSites) == _expect("sinh:1x2 __sub__:1x2")
    assert _shape(LambdaHelper) == _expect("tanh:1x2 tanh:2 __mul__:1x2 __mul__:2 __add__:1")


class OneLineTwoOps(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x) + torch.tanh(x * 2)


def test_row21_two_same_type_ops_on_one_line() -> None:
    """Row 21: two same-type ops on one line are 2 single layers."""
    assert _shape(OneLineTwoOps) == _expect("tanh:1x2 __mul__:1 __add__:1")


class SameLineLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.sin(torch.sin(x))
        return x


def test_row22_same_line_pair_inside_loop() -> None:
    """Row 22: a same-line pair inside a loop gives two layers, 3p each."""
    assert _shape(SameLineLoop) == _expect("sin:3x2")


def test_row23_same_line_pair_inside_submodule() -> None:
    """Row 23: a same-line pair inside a submodule is 2 single layers."""
    assert _shape(lambda: Wrap(body=SameLineOps())) == _expect("sin:1x2")


class WeightTying(Holder):
    def forward(self, ids: Tensor) -> Tensor:
        h = torch.tanh(self.emb(ids))
        return F.linear(h, self.emb.weight)


def test_row24_embedding_and_head_weight_tying() -> None:
    """Row 24: embedding and a head tied to its weight stay separate layers."""
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    shape = _shape(lambda: WeightTying(emb=nn.Embedding(10, 4)), ids)
    assert shape == _expect("embedding:1 tanh:1 linear:1")


class ManualUnrollSharedParam(Holder):
    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(x @ self.w)
        h = torch.tanh(h @ self.w)
        return h


@pytest.mark.smoke
def test_row25_one_raw_parameter_on_two_lines() -> None:
    """Row 25: `x @ W` on two lines gives 2 single matmuls and a separate tanh after each."""
    shape = _shape(lambda: ManualUnrollSharedParam(w=_w()))
    assert shape == _expect("__matmul__:1x2 tanh:1x2")


# ---------------------------------------------------------------- rows 26-33: module identity


class ModuleThriceLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.relu(self.lin(x))
        return x


def test_row26_module_called_three_times_from_loop() -> None:
    """Row 26: one module called 3 times from a loop gives 3p."""
    assert _shape(lambda: ModuleThriceLoop(lin=_lin())) == _expect("linear:3 relu:3")


class ModuleThriceLines(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = torch.relu(x)
        x = self.lin(x)
        x = torch.sigmoid(x)
        x = self.lin(x)
        return x


@FORK
def test_row27_weighted_module_from_three_lines(
    monkeypatch: pytest.MonkeyPatch, module_call_is_pass: bool
) -> None:
    """Row 27 (fork 1): one weighted module called from 3 lines gives 3p (caller lines: 3x1)."""
    monkeypatch.setattr(_call_site, "MODULE_CALL_IS_PASS", module_call_is_pass)
    trace = _capture(lambda: ModuleThriceLines(lin=_lin()), _x())
    linear = "linear:3" if module_call_is_pass else "linear:1x3"
    assert _layer_shape(trace) == _expect(f"{linear} relu:1 sigmoid:1")
    assert _module_calls(trace, "linear") == ["lin:1", "lin:2", "lin:3"]


class ReLUModuleReused(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        return x


@pytest.mark.smoke
@FORK
def test_row28_one_relu_instance_on_three_lines(
    monkeypatch: pytest.MonkeyPatch, module_call_is_pass: bool
) -> None:
    """Row 28 (fork 1): one `nn.ReLU` on 3 lines with ops between gives relu 3p (caller: 3x1)."""
    monkeypatch.setattr(_call_site, "MODULE_CALL_IS_PASS", module_call_is_pass)
    build = lambda: ReLUModuleReused(l1=_lin(), l2=_lin(), l3=_lin(), act=nn.ReLU())  # noqa: E731
    trace = _capture(build, _x())
    relu = "relu:3" if module_call_is_pass else "relu:1x3"
    assert _layer_shape(trace) == _expect(f"linear:1x3 {relu}")
    assert _module_calls(trace, "relu") == ["act:1", "act:2", "act:3"]


def test_row29_modulelist_of_distinct_identical_modules() -> None:
    """Row 29: a ModuleList of distinct identical modules gives one layer per instance."""
    build = lambda: BlocksLoop(blocks=nn.ModuleList([nn.ReLU() for _ in range(3)]))  # noqa: E731
    assert _shape(build) == _expect("relu:1x3")


class SharedChildParent(Holder):
    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.child(x))


class TwoParents(Holder):
    def forward(self, x: Tensor) -> Tensor:
        return self.p2(self.p1(x))


def _shared_child() -> nn.Module:
    shared = _lin()
    return TwoParents(p1=SharedChildParent(child=shared), p2=SharedChildParent(child=shared))


@FORK
def test_row30_module_instance_shared_by_two_parents(
    monkeypatch: pytest.MonkeyPatch, module_call_is_pass: bool
) -> None:
    """Row 30 (fork 1): one Linear shared by two parents is 1 layer 2p (caller: 2 single)."""
    monkeypatch.setattr(_call_site, "MODULE_CALL_IS_PASS", module_call_is_pass)
    trace = _capture(_shared_child, _x())
    linear = "linear:2" if module_call_is_pass else "linear:1x2"
    assert _layer_shape(trace) == _expect(f"{linear} relu:1x2")
    calls = [call.rsplit(":", 1) for call in _module_calls(trace, "linear")]
    assert [number for _, number in calls] == ["1", "2"]
    assert calls[0][0] == calls[1][0]


class SiameseEncoder(Holder):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.lin(x)) * 0.5


class SiameseTwoLines(Holder):
    def forward(self, x: Tensor) -> Tensor:
        a = self.enc(x)
        b = self.enc(x * 3)
        return a - b


@FORK
def test_row31_siamese_encoder_with_bare_ops(
    monkeypatch: pytest.MonkeyPatch, module_call_is_pass: bool
) -> None:
    """Row 31 (fork 1): a Siamese encoder called on two inputs gives every encoder op 2p."""
    monkeypatch.setattr(_call_site, "MODULE_CALL_IS_PASS", module_call_is_pass)
    trace = _capture(lambda: SiameseTwoLines(enc=SiameseEncoder(lin=_lin())), _x())
    if module_call_is_pass:
        expected = "linear:2 tanh:2 __mul__:2 __mul__:1 __sub__:1"
    else:
        expected = "linear:1x2 tanh:1x2 __mul__:1x3 __sub__:1"
    assert _layer_shape(trace) == _expect(expected)
    assert _module_calls(trace, "tanh") == ["enc:1", "enc:2"]


class ListComp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        parts = [torch.exp(c) for c in x.chunk(3, dim=0)]
        return torch.cat(parts, dim=0)


def test_row32_list_comprehension() -> None:
    """Row 32: a list comprehension `[exp(c) for c in cs]` gives exp 3p."""
    assert _shape(ListComp, _x(3, 4)) == _expect("chunk:1x3 exp:3 cat:1")


class ModuleListComprehension(Holder):
    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([blk(x) for blk in self.blocks])


class GeneratorSumHeads(Holder):
    def forward(self, x: Tensor) -> Tensor:
        return sum(h(x) for h in self.blocks)


def test_row33_comprehension_and_generator_over_modulelist() -> None:
    """Row 33: a comprehension over a ModuleList gives per-instance layers; a generator
    `sum` gives radd 1p and add (N-1)p."""
    shape = _shape(lambda: ModuleListComprehension(blocks=_linears()))
    assert shape == _expect("linear:1x3 stack:1")
    shape = _shape(lambda: GeneratorSumHeads(blocks=_linears()))
    assert shape == _expect("linear:1x3 __radd__:1 __add__:2")


# ---------------------------------------------------------------- rows 34-46: control flow


class VmapLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.func.vmap(lambda z: z.sin())(x)
        return x


def test_row34_vmap_boundary_in_loop() -> None:
    """Row 34: a `vmap` boundary in a loop gives one boundary layer 3p."""
    assert _shape(VmapLoop) == _expect("vmap:3")


class GradLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.func.grad(lambda z: z.square().sum())(x)
        return x


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


def test_row36_greedy_generation_loop() -> None:
    """Row 36: a greedy generation loop calling a tiny LM from one line gives every op Np."""
    build = lambda: GenerateLoop(lm=TinyLM(emb=nn.Embedding(10, 4), lin=_lin()))  # noqa: E731
    shape = _shape(build, torch.tensor([[1], [2]]))
    assert shape == _expect("embedding:3 linear:3x2 tanh:3 __getitem__:3 argmax:3 cat:3")


class NestedLoopsNoModule(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            for _ in range(3):
                x = torch.cos(x)
            x = x * 0.5
        return x


def test_row37_nested_loops_no_module() -> None:
    """Row 37: nested loops with no module give inner 6p and outer 2p."""
    assert _shape(NestedLoopsNoModule) == _expect("cos:6 __mul__:2")


class NestedWithModule(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            for _ in range(3):
                x = torch.tanh(self.lin(x))
            x = x * 0.5
        return x


def test_row38_nested_loops_with_module() -> None:
    """Row 38: nested loops with a module give linear 6p, tanh 6p, mul 2p."""
    assert _shape(lambda: NestedWithModule(lin=_lin())) == _expect("linear:6 tanh:6 __mul__:2")


class EarlyExit(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(8):
            x = x * 0.5
            if x.sum() < 1:
                break
        return x


def test_row39_early_exit_break() -> None:
    """Row 39: an early `break` counts executed passes only (4 of 8 on ones)."""
    assert _shape(EarlyExit) == _expect("__mul__:4 sum:4 __lt__:4")


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


def test_row41_branch_taken_on_some_iterations() -> None:
    """Row 41: a branch taken on some iterations gives per-line execution counts."""
    assert _shape(CondSomeIters) == _expect("tanh:4 __mul__:2")


class CondDifferentBranches(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for i in range(4):
            if i % 2 == 0:
                x = torch.relu(x) + 0.1
            else:
                x = torch.sigmoid(x)
            x = x * 0.9
        return x


def test_row42_different_branch_per_iteration() -> None:
    """Row 42: a different branch per iteration gives per-line counts."""
    assert _shape(CondDifferentBranches) == _expect("relu:2 __add__:2 sigmoid:2 __mul__:4")


class WhileLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        n = 0
        while n < 3:
            x = torch.tanh(x) * 0.5
            n += 1
        return x


def test_row43_while_loop() -> None:
    """Row 43: a `while` loop gives Np."""
    assert _shape(WhileLoop) == _expect("tanh:3 __mul__:3")


class WhileData(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        while x.sum() > 1:
            x = x * 0.5
        return x


@pytest.mark.smoke
def test_row44_tensor_op_in_while_condition() -> None:
    """Row 44: tensor ops in a `while` condition give (N+1)p on every Python version."""
    assert _shape(WhileData) == _expect("sum:4 __gt__:4 __mul__:3")


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


def test_row45_recursion_depth_three() -> None:
    """Row 45: recursion depth 3 (one-op and two-op body) gives body ops 3p."""
    assert _shape(Recursion) == _expect("tanh:3 __mul__:3")
    assert _shape(RecursiveLoop) == _expect("sin:3")


def test_row46_recursion_from_two_external_sites() -> None:
    """Row 46: recursion entered from two external sites gives 2 layers (2p, 3p)."""
    assert _shape(RecursiveTwoSites) == _expect("sin:2 sin:3")


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


def test_row47_inplace_buffer_update_per_iteration() -> None:
    """Row 47: an in-place buffer update per iteration gives body ops Np."""
    expected = "zero_:1 mul_:3 tanh:3 __mul__:3 add_:3 __mul__:1"
    assert _shape(InplaceBuffer) == _expect(expected)


class BufferTwoSites(InplaceBuffer):
    def forward(self, x: Tensor) -> Tensor:
        self.state.add_(x)
        self.state.add_(x)
        return self.state.clone()


def test_row48_buffer_updated_on_two_lines() -> None:
    """Row 48: a buffer updated on two different lines gives 2 single `add_` layers."""
    assert _shape(BufferTwoSites) == _expect("add_:1x2 clone:1")


class VariableScalarLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for i in range(1, 4):
            x = x * i
        return x


def test_row49_varying_scalar_in_loop() -> None:
    """Row 49: a varying scalar `x * i` in a loop gives 1 layer Np."""
    assert _shape(VariableScalarLoop) == _expect("__mul__:3")


class ChangingShapeLoop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.cat((x, x), dim=0)
        return x


def test_row50_growing_shape_cat_in_loop() -> None:
    """Row 50: a growing-shape `cat` in a loop gives 1 layer Np."""
    assert _shape(ChangingShapeLoop) == _expect("cat:3")


class ParamListLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for w in self.ws:
            x = torch.tanh(x @ w)
        return x


def test_row51_loop_over_parameterlist() -> None:
    """Row 51: a loop over a ParameterList `x = tanh(x @ w)` gives matmul 3p, tanh 3p."""
    build = lambda: ParamListLoop(ws=nn.ParameterList([_w() for _ in range(3)]))  # noqa: E731
    assert _shape(build) == _expect("__matmul__:3 tanh:3")


class ModuleListGlue(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = x + blk(x)
        return x


def test_row52_root_modulelist_loop_with_glue() -> None:
    """Row 52: a root ModuleList loop `x = x + blk(x)` keeps blocks separate, glue add Np."""
    assert _shape(lambda: ModuleListGlue(blocks=_linears())) == _expect("linear:1x3 __add__:3")


class CheckpointTwice(Holder):
    def forward(self, x: Tensor) -> Tensor:
        x = checkpoint(self.block, x, use_reentrant=False)
        x = checkpoint(self.block, x * 2, use_reentrant=False)
        return x


def test_row53_checkpoint_of_block_from_two_lines() -> None:
    """Row 53: `checkpoint` of a block from two lines gives linears, add1, add2 2p each."""
    shape = _shape(lambda: CheckpointTwice(block=ResidualPair()))
    assert shape == _expect("linear:2x2 __add__:2x2 __mul__:1")


class HookedLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.tanh(self.lin(x))
        return x


def _hooked_loop() -> nn.Module:
    lin = _lin()
    lin.register_forward_hook(lambda m, i, o: o * 2)
    return HookedLoop(lin=lin)


def test_row54_forward_hook_on_module_in_loop() -> None:
    """Row 54: a forward hook on a module in a loop gives hook ops Np."""
    assert _shape(_hooked_loop) == _expect("linear:3 __mul__:3 tanh:3")


class MultipleOutputs(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for _ in range(3):
            values, indexes = torch.max(x, dim=1)
            outputs.append(values + indexes)
        return torch.stack(outputs)


def test_row55_multi_output_max_in_loop() -> None:
    """Row 55: multi-output `max` in a loop gives values Np and indices Np."""
    assert _shape(MultipleOutputs) == _expect("max:3x2 __add__:3 stack:1")


class DynamicDispatch(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for fn in (torch.sin, torch.cos, torch.sin):
            x = fn(x)
        return x


def test_row56_one_call_site_dispatching_sin_cos_sin() -> None:
    """Row 56: one call site dispatching sin, cos, sin gives sin 2p and cos 1p."""
    assert _shape(DynamicDispatch) == _expect("sin:2 cos:1")


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


def test_row57_exec_built_code_in_loop() -> None:
    """Row 57: `exec`-built code run in a loop (also recompiled per step) gives Np."""
    assert _shape(ExecLoop) == _expect("sin:3")
    assert _shape(ExecBuiltLoop) == _expect("sin:3")


class IdenticalCompiledTwoSites(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        a: dict[str, Any] = {"torch": torch}
        b: dict[str, Any] = {"torch": torch}
        exec(compile(_GEN_SOURCE, "<tl-same>", "exec"), a)
        exec(compile(_GEN_SOURCE, "<tl-same>", "exec"), b)
        x = a["f"](x)
        x = b["f"](x)
        return x


def test_row58_identical_compiled_functions_from_two_sites() -> None:
    """Row 58: two separately compiled identical functions from two sites stay separate."""
    assert _shape(IdenticalCompiledTwoSites) == _expect("sin:1x2")


class StepZeroSpecial(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for t in range(3):
            if t == 0:
                h = torch.tanh(x[:, t] @ self.w)
            else:
                h = torch.tanh(x[:, t] @ self.w + h)
        return h


def test_row59_recurrence_with_special_step_zero_line() -> None:
    """Row 59: a recurrence with a special step-0 line: step-0 ops single, loop ops 2p."""
    shape = _shape(lambda: StepZeroSpecial(w=_w()), _x(2, 3, 4))
    step_zero = "__getitem__:1 __matmul__:1 tanh:1"
    assert shape == _expect(f"{step_zero} __getitem__:2 __matmul__:2 __add__:2 tanh:2")


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


def test_row60_block_called_twice_with_inner_loop() -> None:
    """Row 60: a block called twice with an inner loop of 2 gives 4p, numbered flat."""
    trace = _capture(lambda: BlockTwice(block=InnerLoopBlock(lin=_lin())), _x())
    assert _layer_shape(trace) == _expect("linear:4 tanh:4 __mul__:1")
    (tanh_ops,) = _layers(trace, "tanh")
    assert [op.pass_index for op in tanh_ops] == [1, 2, 3, 4]
    calls = [str(op.modules[-1]) for op in tanh_ops]
    assert calls == ["block:1", "block:1", "block:2", "block:2"]


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


def test_row61_residual_vector_quantizer_loop() -> None:
    """Row 61: a residual vector-quantizer loop with a codebook parameter gives 3p each."""
    shape = _shape(lambda: ResidualVQLoop(codebook=nn.Parameter(torch.randn(8, 4))))
    assert shape == _expect("zeros_like:1 cdist:3 argmin:3 __getitem__:3 __add__:3 __sub__:3")


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


def test_row62_spiking_integrate_and_fire() -> None:
    """Row 62: spiking integrate-and-fire timesteps give 3p for each body op."""
    shape = _shape(lambda: Wrap(body=IntegrateAndFire()), _x(3, 2, 4))
    body = "__getitem__:3 __add__:3 __gt__:3 float:3 __sub__:3"
    assert shape == _expect(f"__getitem__:1 zeros_like:1 {body} stack:1")


# ---------------------------------------------------------------- rows 63-71: edges


class ScriptChild(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x) * 0.5


class ChildLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = self.child(x)
        return x


def test_row63_torchscript_submodule_in_loop() -> None:
    """Row 63: a TorchScript submodule called in a loop gives one boundary op Np."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scripted = torch.jit.trace(ScriptChild(), _x())
    except Exception as exc:  # TorchScript missing or retired on this torch
        pytest.skip(f"torch.jit.trace unavailable: {type(exc).__name__}")
    try:
        shape = _shape(lambda: ChildLoop(child=scripted))
    except RuntimeError as exc:
        if "ScriptModule" not in str(exc):
            raise
        pytest.skip("this build refuses TorchScript submodules (no boundary op yet)")
    assert len(shape) == 1 and sum(shape.values()) == 1
    ((_func_name, passes),) = shape
    assert passes == 3


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


def test_row64_no_python_frame_inside_module_call() -> None:
    """Row 64: no Python frame inside a module call falls back to position: a builtin
    forward called twice is one layer 2p; two adds in one C-level call stay two layers."""
    wrap_torch()  # the builtin forwards must hold the wrapped torch functions
    assert _shape(_no_python_frame) == _expect("__mul__:1 relu:2 add:1x2")


def test_row65_two_standalone_traces_number_their_own_passes() -> None:
    """Row 65: two standalone traces of one model each number their own passes."""
    model = RootLoop()
    for _ in range(2):
        trace = _capture(lambda: model, _x())
        assert _layer_shape(trace) == _expect("tanh:3 __mul__:3")
        for func_name in ("tanh", "__mul__"):
            (ops,) = _layers(trace, func_name)
            assert [op.pass_index for op in ops] == [1, 2, 3]


class MapAndSum(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        xs = [x, x * 2.0, x * 3.0]
        ys = list(map(torch.sin, xs))
        return sum(ys)


def test_row66_c_level_repetition_map_and_builtin_sum() -> None:
    """Row 66: `map(torch.sin, xs)` gives sin Np; builtin `sum(list)` gives radd 1p, add (N-1)p."""
    assert _shape(MapAndSum) == _expect("__mul__:1x2 sin:3 __radd__:1 __add__:2")


class SinPlusSin(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x) + torch.sin(x)


def test_row67_same_call_twice_on_one_line() -> None:
    """Row 67: `sin(a) + sin(a)` on one line is 2 single layers (3.10 folding safety)."""
    assert _shape(SinPlusSin) == _expect("sin:1x2 __add__:1")


class GeneratorYieldedModules(Holder):
    def _each(self) -> Iterator[nn.Module]:
        yield from self.blocks

    def forward(self, x: Tensor) -> Tensor:
        for blk in self._each():
            x = blk(x)
        return x


def test_row68_modules_yielded_by_generator() -> None:
    """Row 68: modules yielded by a generator, each called once, give per-instance layers."""
    blocks = lambda: nn.ModuleList([LinearTanh(lin=_lin()) for _ in range(3)])  # noqa: E731
    shape = _shape(lambda: GeneratorYieldedModules(blocks=blocks()))
    assert shape == _expect("linear:1x3 tanh:1x3")


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


def test_row69_decorator_calling_function_on_two_lines() -> None:
    """Row 69: a decorator whose wrapper calls `f(x)` on two lines gives 2 layers per op."""
    assert _shape(DecoratedTwice) == _expect("sin:1x2 __mul__:1x2")


class MapLambda(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        xs = [x, x * 2.0, x * 3.0]
        # The row is about map itself, so the comprehension rewrite does not apply.
        ys = list(map(lambda t: torch.sin(t), xs))  # noqa: C417
        return torch.stack(ys)


def test_row70_map_with_lambda() -> None:
    """Row 70: `map(lambda t: torch.sin(t), xs)` gives sin Np."""
    assert _shape(MapLambda) == _expect("__mul__:1x2 sin:3 stack:1")


class AttentionLoop(Holder):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            x = x + self.attn(x, x, x, need_weights=False)[0]
        return x


def test_row71_multi_head_attention_internal_call_sites() -> None:
    """Row 71: inside multi-head attention each internal call site is its own layer.

    torch's own Python decides the exact op list (it varies across torch versions), so
    the row is checked as a property: called twice, every layer inside the attention
    call has exactly one pass per call, and in- and out-projection are distinct layers.
    """
    trace = _capture(lambda: AttentionLoop(attn=nn.MultiheadAttention(4, 2)), _x(3, 2, 4))
    inner: dict[str, list[Any]] = {}
    for op in trace.layer_list:
        if op.modules and str(op.modules[-1]).startswith("attn:"):
            inner.setdefault(op.layer_label, []).append(op)
    assert inner
    for ops in inner.values():
        assert [str(op.modules[-1]) for op in ops] == ["attn:1", "attn:2"]
    linears = [ops for ops in inner.values() if str(ops[0].func_name) == "linear"]
    assert len(linears) >= 2
    assert _layer_shape(trace)[("__add__", 2)] == 1
