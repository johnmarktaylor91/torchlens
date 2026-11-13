"""Neutral ``tl.grad`` edit sites on the NumPy fake adapter.

The edited gradients equal the chain rule worked by hand with the same edit
applied at the site, edits fire in declared order on the complete gradient of
their slot, and the module refuses what it cannot do.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from tests.support import replay_fake_adapter as fake
from torchlens.backends._reverse_engine import (
    EngineNode,
    ReplayGraph,
    graph_from_recipes,
    reverse_gradients,
)
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.intervention import _grad_site_neutral as neutral, _grad_site_plan
from torchlens.intervention._grad_site_neutral import GradSiteEdit, edited_gradients
from torchlens.intervention.hooks import _target_operands

_A = np.array([0.5, -1.25, 2.0])
_B = np.array([1.5, 0.75, -0.5])


def _ops() -> list[fake.FakeOp]:
    """``loss = sum(3 * (a*b + a*a + a))`` with ``a`` used four times."""

    return [
        fake.FakeOp("a", "input", (), ("a",)),
        fake.FakeOp("b", "input", (), ("b",)),
        fake.FakeOp("c", "mul", ("a", "b"), ("c",)),
        fake.FakeOp("m", "mul", ("a", "a"), ("m",)),
        fake.FakeOp("d", "add", ("c", "m"), ("d",)),
        fake.FakeOp("e", "add", ("d", "a"), ("e",)),
        fake.FakeOp("u", "scale", ("e",), ("u",), literals=((("factor",), 3.0),)),
        fake.FakeOp("loss", "sum", ("u",), ("loss",), attrs={"axis": 0}),
    ]


def _setup(*, with_inputs: bool = True) -> tuple[fake.FakeTrace, ReplayGraph]:
    trace = fake.build_trace(_ops(), {"a": _A.copy(), "b": _B.copy()})
    ops = [op for op in trace.ops.values() if with_inputs or op.op_identity != "input"]
    return trace, graph_from_recipes(trace, fake, ops, trace.payloads.__getitem__)


def _edit(label: str, fn: Any, name: str = "edit", pass_index: int | None = None) -> GradSiteEdit:
    return GradSiteEdit(label, lambda grad, call: fn(grad), name, pass_index)


def test_no_edits_equals_the_reverse_engine() -> None:
    trace, graph = _setup()
    plain = reverse_gradients(trace, fake, graph, {"loss": 1.0})
    result = edited_gradients(trace, fake, graph, {"loss": 1.0}, ())
    assert set(result.grads) == set(plain)
    for label, grad in plain.items():
        np.testing.assert_array_equal(result.grads[label], grad)
    assert result.fires == ()


@pytest.mark.parametrize("with_inputs", [True, False])
def test_scale_at_a_site_equals_the_chain_rule_with_the_edit(with_inputs: bool) -> None:
    """Scaling the gradient of ``e`` by k scales every path through ``e``."""

    trace, graph = _setup(with_inputs=with_inputs)
    result = edited_gradients(trace, fake, graph, {"loss": 1.0}, [_edit("e", lambda g: 0.25 * g)])
    ge = np.full(3, 3.0) * 0.25
    np.testing.assert_allclose(result.grads["e"], ge)
    np.testing.assert_allclose(result.grads["a"], ge * _B + 2 * ge * _A + ge)
    np.testing.assert_allclose(result.grads["b"], ge * _A)
    np.testing.assert_allclose(result.grads["u"], np.ones(3))


def test_an_edit_on_one_branch_leaves_the_other_paths_alone() -> None:
    """Editing ``c`` (the a*b branch) changes only that contribution to ``a``."""

    trace, graph = _setup()
    result = edited_gradients(trace, fake, graph, {"loss": 1.0}, [_edit("c", np.zeros_like)])
    ge = np.full(3, 3.0)
    np.testing.assert_allclose(result.grads["c"], np.zeros(3))
    np.testing.assert_allclose(result.grads["a"], 2 * ge * _A + ge)
    np.testing.assert_allclose(result.grads["b"], np.zeros(3))


def test_edits_fire_in_declared_order_on_the_complete_gradient() -> None:
    """``a`` is reached four times; its edits fire once, after the last contribution."""

    trace, graph = _setup()
    plain = reverse_gradients(trace, fake, graph, {"loss": 1.0})["a"]
    edits = [_edit("a", lambda g: 2.0 * g, "double"), _edit("a", lambda g: g + 1.0, "plus")]
    result = edited_gradients(trace, fake, graph, {"loss": 1.0}, edits)
    np.testing.assert_allclose(result.grads["a"], 2.0 * plain + 1.0)
    assert [(f.label, f.name, f.order, f.replaced) for f in result.fires] == [
        ("a", "double", 0, True),
        ("a", "plus", 1, True),
    ]
    flipped = edited_gradients(trace, fake, graph, {"loss": 1.0}, edits[::-1])
    np.testing.assert_allclose(flipped.grads["a"], 2.0 * (plain + 1.0))


def test_leaf_sites_fire_when_inputs_are_not_nodes() -> None:
    trace, graph = _setup(with_inputs=False)
    plain = reverse_gradients(trace, fake, graph, {"loss": 1.0})["b"]
    result = edited_gradients(trace, fake, graph, {"loss": 1.0}, [_edit("b", lambda g: -g)])
    np.testing.assert_allclose(result.grads["b"], -plain)
    assert [f.label for f in result.fires] == ["b"]


def test_a_site_no_gradient_reaches_does_not_fire() -> None:
    trace, graph = _setup()
    result = edited_gradients(trace, fake, graph, {"c": np.ones(3)}, [_edit("u", lambda g: 0 * g)])
    assert result.fires == ()
    assert "u" not in result.grads


def test_the_call_carries_torch_hook_context_fields() -> None:
    trace, graph = _setup()
    seen: list[Any] = []

    def record(grad: Any, call: Any) -> Any:
        seen.append(call)
        return grad

    result = edited_gradients(
        trace, fake, graph, {"loss": 1.0}, [GradSiteEdit("e", record)], pass_index=3
    )
    assert seen == [neutral.GradSiteCall("e", 0, 0, 3)]
    assert result.fires[0].replaced is False


def test_pass_qualified_edits_fire_only_in_their_pass() -> None:
    trace, graph = _setup()
    edits = [_edit("e", lambda g: 0 * g, pass_index=2)]
    first = edited_gradients(trace, fake, graph, {"loss": 1.0}, edits, pass_index=1)
    second = edited_gradients(trace, fake, graph, {"loss": 1.0}, edits, pass_index=2)
    assert first.fires == () and len(second.fires) == 1
    np.testing.assert_allclose(second.grads["a"], np.zeros(3))


def test_the_second_output_slot_of_a_multi_output_op_is_editable() -> None:
    ops = [
        fake.FakeOp("a", "input", (), ("a",)),
        fake.FakeOp("s", "split2", ("a",), ("s", "s:1"), attrs={"index": 1}),
        fake.FakeOp("t", "sum", ("s:1",), ("t",), attrs={"axis": 0}),
        fake.FakeOp("v", "sum", ("s",), ("v",), attrs={"axis": 0}),
        fake.FakeOp("loss", "add", ("t", "v"), ("loss",)),
    ]
    trace = fake.build_trace(ops, {"a": np.array([1.0, 2.0, 3.0])})
    graph = graph_from_recipes(trace, fake, list(trace.ops.values()), trace.payloads.__getitem__)
    seen: list[int] = []

    def tripled(grad: Any, call: Any) -> Any:
        seen.append(call.tuple_index)
        return 3.0 * grad

    result = edited_gradients(trace, fake, graph, {"loss": 1.0}, [GradSiteEdit("s:1", tripled)])
    np.testing.assert_allclose(result.grads["a"], np.array([1.0, 3.0, 3.0]))
    assert seen == [1]


def test_refusals_are_typed() -> None:
    trace, graph = _setup()
    with pytest.raises(RunCapabilityUnavailableError, match="no recorded op has the output"):
        edited_gradients(trace, fake, graph, {"loss": 1.0}, [_edit("nope", lambda g: g)])
    with pytest.raises(RunCapabilityUnavailableError, match="returned None"):
        edited_gradients(trace, fake, graph, {"loss": 1.0}, [_edit("e", lambda g: None)])
    with pytest.raises(RunCapabilityUnavailableError, match="no replay adapter"):
        neutral.edited_gradients_for_backend(trace, "torch", [], lambda _l: None, {}, ())


def test_an_edit_does_not_hide_an_unreplayable_op_on_the_path() -> None:
    """Zeroing a gradient upstream still reaches the opaque op: refused, never skipped."""

    ops = [
        fake.FakeOp("a", "input", (), ("a",)),
        fake.FakeOp("o", "opaque", ("a",), ("o",)),
        fake.FakeOp("loss", "neg", ("o",), ("loss",)),
    ]
    trace = fake.build_trace(ops[:1], {"a": np.ones(2)})
    trace.ops.update({op.label: op for op in ops[1:]})
    trace.payloads.update({"o": np.ones(2), "loss": -np.ones(2)})
    graph = graph_from_recipes(trace, fake, list(trace.ops.values()), trace.payloads.__getitem__)
    with pytest.raises(RunCapabilityUnavailableError, match="cannot be replayed"):
        edited_gradients(trace, fake, graph, {"loss": np.ones(2)}, [_edit("o", np.zeros_like)])


def test_one_node_per_op_folds_repeated_multi_output_records() -> None:
    node = EngineNode(op="x", parents=("a",), outputs=("s", "s:1"))
    twin = EngineNode(op="y", parents=("a",), outputs=("s", "s:1"))
    graph = neutral._one_node_per_op(ReplayGraph(nodes=(node, twin), value=lambda _l: None))
    assert graph.nodes == (node,)


def test_grad_site_targets_reads_unions_and_pass_qualifiers() -> None:
    targets = neutral.grad_site_targets(tl.grad("relu_1") | tl.grad("relu_2"), _target_operands)
    assert [inner for inner, _pass in targets] == ["relu_1", "relu_2"]
    assert all(qualifier is None for _inner, qualifier in targets)
    ((inner, qualifier),) = neutral.grad_site_targets(
        tl.grad("relu_1") & tl.in_backward_pass(2), _target_operands
    )
    assert (inner, qualifier) == ("relu_1", 2)
    with pytest.raises(RunCapabilityUnavailableError, match="autograd-node selectors"):
        neutral.grad_site_targets(tl.grad_fn("AddmmBackward0"), _target_operands)


def test_build_arm_and_run_armed_grad_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    trace, _graph = _setup()
    edits = neutral.build_grad_site_edits(
        tl.grad("e") | tl.grad("c"),
        lambda g, call: 0 * g,
        lambda inner: (inner,),
        operands=_target_operands,
        name="zero",
    )
    assert [(e.label, e.name) for e in edits] == [("e", "zero"), ("c", "zero")]
    with pytest.raises(RunCapabilityUnavailableError, match="names no recorded op"):
        neutral.build_grad_site_edits(
            tl.grad("e"), lambda g, call: g, lambda inner: (), operands=_target_operands
        )
    neutral.arm_grad_sites(trace, edits[:1])
    neutral.arm_grad_sites(trace, edits[1:])
    assert neutral.armed_grad_sites(trace) == edits
    monkeypatch.setattr("torchlens.backends._replay_adapters.get_adapter", lambda name: fake)
    ops = [op for op in trace.ops.values() if op.op_identity != "input"]
    result = neutral.run_armed_grad_sites(
        trace, "fake", ops, trace.payloads.__getitem__, {"loss": 1.0}
    )
    np.testing.assert_allclose(result.grads["a"], np.zeros(3))
    neutral.clear_grad_sites(trace)
    assert neutral.armed_grad_sites(trace) == ()


def test_disagreeing_labels_uses_the_adapter_comparison() -> None:
    trace, graph = _setup()
    result = edited_gradients(trace, fake, graph, {"loss": 1.0}, [_edit("e", lambda g: 0.5 * g)])
    native = dict(result.grads)
    assert neutral.disagreeing_labels(fake, result, native) == ()
    native["a"] = native["a"] + 1.0
    del native["b"]
    assert neutral.disagreeing_labels(fake, result, native) == ("a", "b")


def test_do_signal_plan_routes_a_non_torch_trace_to_its_backend_door(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off torch, ``Trace.do`` with a gradient site reaches the backend door or refuses typed."""

    trace = SimpleNamespace(_provenance_backend="gsdoor")
    plan = _grad_site_plan.signal_plan(tl.grad("relu_1"), tl.scale(0.5))
    with pytest.raises(RunCapabilityUnavailableError, match="not open on the gsdoor backend"):
        _grad_site_plan.do_signal_plan(trace, plan, None)
    calls: list[Any] = []
    door = SimpleNamespace(do_signal_plan=lambda t, p, r: calls.append((t, p, r)) or t)
    monkeypatch.setitem(sys.modules, "torchlens.backends.gsdoor._grad_site_door", door)
    assert _grad_site_plan.do_signal_plan(trace, plan, "request") is trace
    assert calls == [(trace, plan, "request")]


def test_a_caller_supplied_operand_reader_drives_the_decomposition() -> None:
    """The neutral module reads composites only through the caller's ``operands``."""

    seen: list[Any] = []

    def leaf_only(site: Any) -> tuple[str | None, tuple[Any, ...]]:
        seen.append(site)
        return None, ()

    assert neutral.grad_site_targets(tl.grad("relu_1"), leaf_only) == (("relu_1", None),)
    assert seen == []
    with pytest.raises(RunCapabilityUnavailableError, match="not a tl.grad"):
        neutral.grad_site_targets(tl.grad("relu_1") | tl.grad("relu_2"), leaf_only)
    assert len(seen) == 1


def test_the_neutral_module_imports_nothing_from_the_selector_or_hook_cycle() -> None:
    """Backend runtimes import this module; it must not pull them into the largest cycle."""

    import ast
    from pathlib import Path

    tree = ast.parse(Path(neutral.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert imported <= {"backends._reverse_engine", "backends._replay_adapters", "errors"}
