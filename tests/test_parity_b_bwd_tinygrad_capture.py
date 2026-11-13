"""Native tinygrad backward: gradient UOps joined to the capture's single realize.

The tinygrad capture does not call the hooks yet, so ``_JointCapture`` stands in
for it with the documented hook contract of
``torchlens.backends._backward_native.tinygrad``: it opens the backward scope
around the forward, lists the gradient tensors in the same ``realize_once``
call, and retains the state after emission. The registry entry is patched to
admit ``backward_capture`` the way the merged spec will.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends import registry
from torchlens.backends._backward_native import native_backward_surface
from torchlens.errors import RunCapabilityUnavailableError

pytest.importorskip("tinygrad")
from tinygrad import Tensor  # noqa: E402
from tinygrad.nn.state import get_state_dict  # noqa: E402

from tests.conformance._rosters import tinygrad as roster  # noqa: E402
from torchlens.backends._backward_native import tinygrad as native  # noqa: E402
from torchlens.backends._protocols.tinygrad import TinygradProtocol  # noqa: E402
from torchlens.backends._replay_adapters import get_adapter  # noqa: E402
from torchlens.backends._reverse_engine import graph_from_recipes  # noqa: E402
from torchlens.backends.tinygrad import (  # noqa: E402
    _realize_once,
    backend as tg_backend,
)
from torchlens.backends.tinygrad._execution_guard import (  # noqa: E402
    _reject_mid_capture_execution,
)
from torchlens.validation._backward_neutral import (  # noqa: E402
    RecordedBackward,
    check_backward_records,
    recorded_backward_from_trace,
)

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]

adapter = get_adapter("tinygrad")
ROWS = roster.config_built_rows() + roster.random_init_rows()
DIAG_DIR = os.environ.get("TL_BWD_TINYGRAD_DIAG")


def _float_leaves(value: Any) -> list[Any]:
    from tinygrad import dtypes

    return [leaf for leaf in native._flat_tensors(value) if dtypes.is_float(leaf.dtype)]


def _weighted(leaf: Any) -> Any:
    """``leaf`` times a fixed non-constant pattern, summed (a plain sum of a
    normalized output has zero gradient, which would compare rounding noise)."""

    count = max(int(np.prod(leaf.shape)), 1)
    pattern = (Tensor.arange(count, dtype=leaf.dtype) * 0.37 + 0.1).sin()
    return (leaf * pattern.reshape(leaf.shape)).sum()


def scalar_loss(output: Any) -> Any:
    """Weighted sum of every float output: one scalar that reaches each output."""

    leaves = _float_leaves(output)
    total = _weighted(leaves[0])
    for leaf in leaves[1:]:
        total = total + _weighted(leaf)
    return total


class _JointCapture:
    """The tinygrad capture's hook calls, installed around one ``tl.trace``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.session: SimpleNamespace | None = None
        self.realize_calls = 0
        self._install()

    def _install(self) -> None:
        cls = tg_backend.TinygradBackend
        forward, sources = cls._run_observed_forward, cls._emit_input_sources
        realize = _realize_once.realize_recorded_values
        protocol_realize = TinygradProtocol.realize_once
        joint = self

        def observed_forward(backend: Any, model: Any, args: Any, tree: Any) -> Any:
            session = joint.session
            if session is None:
                return forward(backend, model, args, tree)
            session.param_addresses = dict(tree.param_address_by_uop_id) if tree else {}
            with native.backward_ready_scope(session):
                result = forward(backend, model, args, tree)
            session.raw_output = result[0]
            return result

        def realize_values(plan: Any, source_uops: Any, outputs: Any) -> Any:
            session = joint.session
            if session is None:
                return realize(plan, source_uops, outputs)
            with _reject_mid_capture_execution():
                extra = session.tinygrad_backward.gradient_targets(
                    plan, source_uops, session.raw_output
                )
            realized, owned = _realize_once._realization_targets(plan, source_uops)
            TinygradProtocol().realize_once([*realized.values(), *extra, *outputs])
            return realized, owned

        def input_sources(backend: Any, trace: Any, capture: Any) -> Any:
            labels = sources(backend, trace, capture)
            if joint.session is not None:
                joint.session.uop_labels = labels
            return labels

        def counting_realize(protocol: Any, tensors: Any) -> Any:
            joint.realize_calls += 1
            return protocol_realize(protocol, tensors)

        self.monkeypatch.setattr(cls, "_run_observed_forward", observed_forward)
        self.monkeypatch.setattr(cls, "_emit_input_sources", input_sources)
        self.monkeypatch.setattr(_realize_once, "realize_recorded_values", realize_values)
        self.monkeypatch.setattr(TinygradProtocol, "realize_once", counting_realize)

    def trace(self, model: Any, value: Any, loss_fn: Any = scalar_loss) -> Any:
        """Capture with a prepared backward pass for ``loss_fn``."""

        self.session = SimpleNamespace(loss_fn=loss_fn)
        try:
            trace = tl.trace(model, value, backend="tinygrad")
            native.retain_backward_state(trace, self.session)
        finally:
            self.session = None
        return trace


@pytest.fixture
def joint(monkeypatch: pytest.MonkeyPatch) -> _JointCapture:
    spec = registry.get_backend_spec("tinygrad")
    replaced = dataclasses.replace(
        spec,
        capabilities=dataclasses.replace(spec.capabilities, backward_capture=True),
        capability_implementations={
            **(spec.capability_implementations or {}),
            "backward_capture": lambda: native_backward_surface("tinygrad"),
        },
    )
    monkeypatch.setitem(registry._REGISTRY, "tinygrad", replaced)
    return _JointCapture(monkeypatch)


def _array(value: Any) -> np.ndarray:
    return np.asarray(value.numpy())


def _close(expected: Any, actual: Any) -> bool:
    return bool(adapter.close(expected, actual))


def _difference(expected: Any, actual: Any) -> tuple[bool, float, float]:
    """(bit-exact, largest absolute difference, largest absolute expected value)."""

    want, got = _array(expected), _array(actual)
    return (
        bool(np.array_equal(want, got)),
        float(np.max(np.abs(want - got), initial=0.0)),
        float(np.max(np.abs(want), initial=0.0)),
    )


def _input_label(trace: Any) -> str:
    sources = [op for op in trace._layer_list if _is_input(op)]
    assert len(sources) == 1, [op.layer_label for op in sources]
    return str(sources[0].layer_label)


def _is_input(op: Any) -> bool:
    return bool(op.is_input)


def _eager_grads(model: Any, value: Any) -> tuple[Any, dict[str, Any], Any]:
    """``Tensor.backward`` of the same loss on the same (now realized) model."""

    value.grad = None
    state = get_state_dict(model)
    for tensor in state.values():
        tensor.grad = None
    output = model(value)
    scalar_loss(output).backward()
    params = {name: tensor.grad for name, tensor in state.items() if tensor.grad is not None}
    return value.grad, params, output


def _check(trace: Any) -> Any:
    """The replay graph and the recorded pass, both keyed by pass label (``op.label``).

    Calls come from bwd-core's reader, keyed by ``GradFn.op_label`` (the final
    lookup label); each op's gradient is read per pass op, since a multi-pass
    op shares one layer label across its passes.
    """

    ops = [op for op in trace._layer_list if not _is_input(op)]
    graph = graph_from_recipes(trace, adapter, ops, lambda label: trace[label].out)
    pass_number = trace.autograd.backward_passes[-1].pass_number
    calls = recorded_backward_from_trace(trace, pass_number).calls
    lookup = trace._layer_dict_all_keys
    grads = {
        op.label: record.grad
        for op in trace._layer_list
        for record in op._slot("_grad_records") or ()
        if record.backward_pass_number == pass_number and record.grad is not None
    }
    keyed = RecordedBackward(
        calls={lookup[key].label: call for key, call in calls.items()}, grads=grads
    )
    return graph, keyed


def _node_differences(trace: Any, graph: Any, recorded: Any, result: Any) -> list[Any]:
    """For each node finding: the replayed and recorded outgoing gradients compared."""

    rows = []
    nodes = {node.label: node for node in graph.nodes}
    for finding in result.findings:
        if finding.check != "node" or finding.op_label not in nodes:
            continue
        node = nodes[finding.op_label]
        incoming, outgoing = recorded.calls[node.label]
        parents = [graph.value(parent) for parent in node.parents]
        expected = adapter.vjp(trace, node.op, parents, list(incoming))
        for parent, want, got in zip(node.parents, expected, outgoing, strict=True):
            if want is None or got is None:
                rows.append((node.label, parent, want is None, got is None))
            else:
                rows.append((node.label, parent, _difference(want, got), str(want.dtype)))
    return rows


def _diag(name: str, payload: dict[str, Any]) -> None:
    if DIAG_DIR:
        path = Path(DIAG_DIR) / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1, default=str))


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.family)
def test_recorded_pass_passes_the_backward_validator(joint: _JointCapture, row: Any) -> None:
    model, value = row.builder()
    trace = joint.trace(model, value)
    state = trace.__dict__[native.STATE_KEY]
    assert state.unavailable is None, state.unavailable
    trace.backward(scalar_loss)
    graph, recorded = _check(trace)
    input_grad, _params, _output = _eager_grads(model, value)
    label = trace[_input_label(trace)].label
    result = check_backward_records(trace, adapter, graph, recorded, {label: input_grad})
    _diag(
        f"validator-{row.family}",
        {
            "labels": [(op.label, op.layer_label, op._label_raw) for op in trace._layer_list],
            "calls": sorted(recorded.calls),
            "grads": sorted(recorded.grads),
            "graph": [(node.label, node.parents, node.blocked) for node in graph.nodes],
            "findings": [dataclasses.asdict(item) for item in result.findings],
            "node_differences": _node_differences(trace, graph, recorded, result),
            "checked": (result.nodes_checked, result.sums_checked, result.leaves_checked),
            "unavailable": state.unavailable,
        },
    )
    assert result.passed, result.findings
    assert result.nodes_checked == len(recorded.calls)


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.family)
def test_leaf_and_parameter_gradients_equal_tensor_backward(joint: _JointCapture, row: Any) -> None:
    model, value = row.builder()
    trace = joint.trace(model, value)
    trace.backward(scalar_loss)
    state = trace.__dict__[native.STATE_KEY]
    input_grad, params, _output = _eager_grads(model, value)
    recorded = recorded_backward_from_trace(trace, trace.autograd.backward_passes[-1].pass_number)
    label = _input_label(trace)
    exact = {label: _difference(input_grad, recorded.grads[label])}
    assert _close(input_grad, recorded.grads[label]), exact
    recorded_params = dict(state.param_grads)
    assert sorted(recorded_params) == sorted(params), (sorted(recorded_params), sorted(params))
    for name, grad in params.items():
        assert _close(grad, recorded_params[name]), name
        exact[name] = _difference(grad, recorded_params[name])
    _diag(f"leaves-{row.family}", {"exact": exact, "params": sorted(params)})


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.family)
def test_forward_values_and_single_realize_are_unchanged(joint: _JointCapture, row: Any) -> None:
    model, value = row.builder()
    plain = tl.trace(model, value, backend="tinygrad")
    plain_realizes = joint.realize_calls
    model, value = row.builder()
    trace = joint.trace(model, value)
    assert plain_realizes == 1
    assert joint.realize_calls - plain_realizes == 1
    assert [op.layer_label for op in trace._layer_list] == [
        op.layer_label for op in plain._layer_list
    ]
    for ours, theirs in zip(trace._layer_list, plain._layer_list, strict=True):
        assert np.array_equal(_array(ours.out), _array(theirs.out)), ours.layer_label
    for ours, theirs in zip(_float_leaves(trace.output), _float_leaves(plain.output), strict=True):
        assert np.array_equal(_array(ours), _array(theirs))


def _mlp_trace(joint: _JointCapture, loss_fn: Any = scalar_loss) -> tuple[Any, Any]:
    model, value = {row.family: row for row in ROWS}["fn_mlp_linear"].builder()
    return joint.trace(model, value, loss_fn=loss_fn), model


def test_backward_journals_one_pass_named_by_tinygrad_rules(joint: _JointCapture) -> None:
    trace, _model = _mlp_trace(joint)
    trace.backward(scalar_loss)
    assert len(trace.autograd.backward_passes) == 1
    grad_fns = list(trace._grad_fn_logs.values())
    assert grad_fns
    for record in grad_fns:
        assert record.class_qualname == native.RULE_QUALNAME_PREFIX + record.class_name
        assert record.class_name.isupper()


def test_second_backward_is_refused(joint: _JointCapture) -> None:
    trace, _model = _mlp_trace(joint)
    trace.backward(scalar_loss)
    with pytest.raises(RunCapabilityUnavailableError, match="already recorded"):
        trace.backward(scalar_loss)


def test_a_different_loss_is_refused(joint: _JointCapture) -> None:
    trace, _model = _mlp_trace(joint)
    with pytest.raises(RunCapabilityUnavailableError, match="loss_fn the capture prepared"):
        trace.backward(lambda output: output.sum() * 2)
    with pytest.raises(RunCapabilityUnavailableError, match="not connected"):
        trace.backward(Tensor(1.0))


def test_backward_kwargs_are_refused(joint: _JointCapture) -> None:
    trace, _model = _mlp_trace(joint)
    with pytest.raises(RunCapabilityUnavailableError, match="no engine options"):
        trace.backward(scalar_loss, retain_graph=True)


def test_a_capture_without_the_scope_is_refused(joint: _JointCapture) -> None:
    model, value = {row.family: row for row in ROWS}["fn_mlp_linear"].builder()
    trace = tl.trace(model, value, backend="tinygrad")
    with pytest.raises(RunCapabilityUnavailableError, match="backward_ready=True"):
        trace.backward(scalar_loss)


def test_a_non_scalar_loss_keeps_the_capture_and_refuses_backward(joint: _JointCapture) -> None:
    trace, _model = _mlp_trace(joint, loss_fn=lambda output: output)
    assert trace._layer_list
    assert joint.realize_calls == 1
    with pytest.raises(RunCapabilityUnavailableError, match="float scalar"):
        trace.backward(None)
