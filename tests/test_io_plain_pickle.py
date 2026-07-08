"""Regression tests for plain ``pickle.dump`` compatibility."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from torchlens import Trace, trace as trace_fn


class _PlainPickleModel(nn.Module):
    """Simple model used for plain pickle regression checks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the plain pickle test model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        return torch.sin(x) + torch.cos(x)


def _build_trace(seed: int = 0) -> Trace:
    """Build a deterministic ``Trace`` for plain pickle tests.

    Parameters
    ----------
    seed:
        Random seed used for model initialization and input generation.

    Returns
    -------
    Trace
        Logged forward pass with saved outs.
    """

    torch.manual_seed(seed)
    model = _PlainPickleModel()
    inputs = torch.randn(2, 4)
    return trace_fn(model, inputs, layers_to_save="all", random_seed=seed)


def _first_saved_layer(trace: Trace) -> Any:
    """Return the first saved layer from one model log.

    Parameters
    ----------
    trace:
        Model log under test.

    Returns
    -------
    Any
        First saved layer-pass entry.
    """

    return next(layer for layer in trace.layer_list if layer.has_saved_activation)


def test_plain_pickle_dump_and_load_still_work(tmp_path: Path) -> None:
    """Fresh ``Trace`` objects should still survive plain pickle round-trips."""

    trace = _build_trace()
    pickle_path = tmp_path / "trace.pkl"

    with pickle_path.open("wb") as handle:
        pickle.dump(trace, handle)
    with pickle_path.open("rb") as handle:
        restored = pickle.load(handle)

    assert isinstance(restored, Trace)
    assert restored.model_class_name == trace.model_class_name
    assert len(restored.layer_list) == len(trace.layer_list)
    assert restored[restored.output_layers[0]].layer_label == restored.output_layers[0]
    assert restored.layer_list[0].source_trace is restored
    assert isinstance(_first_saved_layer(restored).out, torch.Tensor)


def test_old_style_pickle_without_io_format_version_warns_and_remains_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forged pre-sprint pickles should warn on load and keep accessors working."""

    trace = _build_trace()
    pickle_path = tmp_path / "old_style_trace.pkl"
    original_getstate = Trace.__getstate__

    def _legacy_getstate(self: Trace) -> dict[str, Any]:
        """Return a forged pre-sprint pickle state for one ``Trace``.

        Parameters
        ----------
        self:
            Model log being pickled.

        Returns
        -------
        dict[str, Any]
            State missing the portable format version tag.
        """

        state = original_getstate(self)
        state.pop("tlspec_version", None)
        return state

    monkeypatch.setattr(Trace, "__getstate__", _legacy_getstate)
    with pickle_path.open("wb") as handle:
        pickle.dump(trace, handle)

    with pytest.warns(DeprecationWarning):
        with pickle_path.open("rb") as handle:
            restored = pickle.load(handle)

    assert isinstance(restored, Trace)
    assert restored[restored.output_layers[0]].layer_label == restored.output_layers[0]
    assert restored.layer_list[0].source_trace is restored
    assert isinstance(_first_saved_layer(restored).out, torch.Tensor)


class _TrainableModel(nn.Module):
    """Multi-layer model whose params populate ``grad_fn`` on downstream ops."""

    def __init__(self) -> None:
        """Initialize two linear layers with a nonlinearity between them."""

        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the trainable test model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        return self.fc2(torch.relu(self.fc1(x)))


class _BufferModel(nn.Module):
    """Model with a registered buffer (via BatchNorm) for buffer pickle checks."""

    def __init__(self) -> None:
        """Initialize a linear layer followed by 1D batch norm."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the buffer test model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        return self.bn(self.fc(x))


def test_plain_pickle_trainable_bare_linear_round_trips() -> None:
    """A bare ``nn.Linear`` trace must survive plain pickle.

    Regression for the ``Layer.__getstate__`` gap: ``nn.Linear``'s weight is
    trainable, so downstream ops carry a live ``AddmmBackward0`` autograd node
    on ``grad_fn_handle``. Before the fix the aggregate ``Layer`` leaked that
    node into pickle state and ``pickle.dumps`` raised
    ``TypeError: cannot pickle 'AddmmBackward0' object`` even without any
    explicit backward call.
    """

    torch.manual_seed(0)
    trace = trace_fn(nn.Linear(4, 4), torch.randn(2, 4))
    payload = pickle.dumps(trace)
    restored = pickle.loads(payload)
    # dumps(loads(dumps(...))) must also succeed (full round trip).
    assert pickle.dumps(restored)
    assert isinstance(restored, Trace)
    assert len(restored.layer_list) == len(trace.layer_list)
    assert restored.summary()


def test_plain_pickle_multilayer_trainable_round_trips() -> None:
    """A multi-layer trainable model must survive plain pickle round-trips."""

    torch.manual_seed(0)
    trace = trace_fn(_TrainableModel(), torch.randn(3, 4))
    restored = pickle.loads(pickle.dumps(trace))
    assert pickle.dumps(restored)
    assert len(restored.layer_list) == len(trace.layer_list)
    assert restored.layer_list[0].source_trace is restored


def test_plain_pickle_buffer_model_round_trips() -> None:
    """A model with registered buffers must survive plain pickle round-trips."""

    torch.manual_seed(0)
    model = _BufferModel()
    model.train()
    trace = trace_fn(model, torch.randn(8, 4))
    restored = pickle.loads(pickle.dumps(trace))
    assert pickle.dumps(restored)
    assert restored.summary()


def test_standalone_buffer_pickle_round_trips() -> None:
    """A standalone ``Buffer`` (holding a source-trace weakref) must pickle."""

    from torchlens.data_classes.buffer import Buffer

    torch.manual_seed(0)
    model = _BufferModel()
    model.train()
    trace = trace_fn(model, torch.randn(8, 4))
    buffers = list(trace.buffers)
    assert buffers, "expected at least one captured buffer"
    restored = pickle.loads(pickle.dumps(buffers[0]))
    assert isinstance(restored, Buffer)
    # The live weakref must not survive; it is rebuilt lazily via the accessor.
    assert restored.source_trace is None
    assert isinstance(restored.versions, list)


def test_trace_setstate_absent_container_fields_restore_typed() -> None:
    """Legacy/partial Trace state missing container fields must restore typed.

    Stripping ``layer_list``/``layer_logs`` used to crash inside
    ``Trace.__setstate__`` itself; stripping the other container fields used to
    silently restore ``None`` and crash on first touch.
    """

    torch.manual_seed(0)
    trace = trace_fn(_TrainableModel(), torch.randn(3, 4))
    state = trace.__getstate__()
    stripped = [
        "layer_list",
        "layer_logs",
        "op_labels",
        "layer_labels",
        "by_pass",
        "conditional_records",
        "input_layers",
        "output_layers",
        "op_equivalence_classes",
        "grad_fn_order",
        "backward_durations",
    ]
    for field_name in stripped:
        state.pop(field_name, None)

    restored = Trace.__new__(Trace)
    restored.__setstate__(state)

    assert isinstance(restored.layer_list, list)
    assert hasattr(restored.layer_logs, "values")
    assert isinstance(restored.op_labels, list)
    assert isinstance(restored.by_pass, dict)
    assert isinstance(restored.conditional_records, list)
    # The exact crash sites from __setstate__ must be iterable now.
    assert list(restored.layer_logs.values()) == []
    assert list(restored.layer_list) == []


def test_op_setstate_absent_container_fields_restore_typed() -> None:
    """Legacy/partial Op state missing container fields must restore typed."""

    from torchlens.data_classes.op import Op

    torch.manual_seed(0)
    trace = trace_fn(_TrainableModel(), torch.randn(3, 4))
    op = trace.layer_list[1]
    state = op.__getstate__()
    for field_name in (
        "input_to_module_calls",
        "output_of_modules",
        "output_of_module_calls",
        "_param_barcodes",
        "parents",
        "children",
        "equivalent_ops",
        "root_ancestors",
        "modules",
        "param_shapes",
    ):
        state.pop(field_name, None)

    restored = Op.__new__(Op)
    restored.__setstate__(state)

    assert isinstance(restored.input_to_module_calls, list)
    assert isinstance(restored.parents, list)
    assert isinstance(restored.children, list)
    assert isinstance(restored.equivalent_ops, set)
    assert isinstance(restored.root_ancestors, set)
    assert isinstance(restored._param_barcodes, list)
    assert isinstance(restored.param_shapes, list)
    assert isinstance(restored.modules, list)
    # Consumer patterns from finalization/loop_detection/invariants must work.
    assert "missing" not in restored.input_to_module_calls
    assert list(restored.input_to_module_calls) == []


def test_setstate_present_but_wrong_typed_container_is_coerced() -> None:
    """Present-but-wrong-typed container fields must be coerced on restore."""

    from torchlens.data_classes.op import Op

    torch.manual_seed(0)
    trace = trace_fn(_TrainableModel(), torch.randn(3, 4))

    # Op: `_param_barcodes` declared as list but legacy state holds a set.
    op = trace.layer_list[1]
    op_state = op.__getstate__()
    op_state["_param_barcodes"] = {"legacy_barcode_as_set"}
    restored_op = Op.__new__(Op)
    restored_op.__setstate__(op_state)
    assert isinstance(restored_op._param_barcodes, list)
    assert restored_op._param_barcodes == ["legacy_barcode_as_set"]

    # Trace: `op_labels` declared as list but legacy state holds a set.
    trace_state = trace.__getstate__()
    trace_state["op_labels"] = {"a", "b"}
    restored_trace = Trace.__new__(Trace)
    restored_trace.__setstate__(trace_state)
    assert isinstance(restored_trace.op_labels, list)


def test_all_field_order_container_defaults_are_typed() -> None:
    """Drift guard: every container FIELD_ORDER field defaults to its type.

    Constructs a live trainable-model trace and, for each ``FIELD_ORDER``
    field that is a container at runtime, strips it from state and asserts the
    restored value is a matching container (never ``None``). This catches any
    future field added without a typed default in the ``__setstate__`` fill.
    """

    from torchlens.constants import LAYER_PASS_LOG_FIELD_ORDER, MODEL_LOG_FIELD_ORDER
    from torchlens.data_classes.op import Op

    torch.manual_seed(0)
    trace = trace_fn(_TrainableModel(), torch.randn(3, 4))

    def _assert_container_defaults(instance: Any, field_order: list[str], factory: Any) -> None:
        base_state = instance.__getstate__()
        for field_name in field_order:
            if field_name not in base_state:
                continue
            live_value = base_state[field_name]
            if not isinstance(live_value, (list, dict, tuple, set)):
                continue
            partial = dict(base_state)
            partial.pop(field_name)
            restored = factory()
            restored.__setstate__(partial)
            value = getattr(restored, field_name, None)
            assert isinstance(value, (list, dict, tuple, set)), (
                f"{type(instance).__name__}.{field_name} restored as "
                f"{type(value).__name__} (expected a container) when absent"
            )

    _assert_container_defaults(trace, list(MODEL_LOG_FIELD_ORDER), lambda: Trace.__new__(Trace))
    _assert_container_defaults(
        trace.layer_list[1], list(LAYER_PASS_LOG_FIELD_ORDER), lambda: Op.__new__(Op)
    )


def test_plain_pickle_preserves_in_memory_outs(tmp_path: Path) -> None:
    """Plain pickles should keep outs resident rather than turning them into refs."""

    trace = _build_trace()
    source_layer = _first_saved_layer(trace)
    assert isinstance(source_layer.out, torch.Tensor)

    pickle_path = tmp_path / "in_memory_trace.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(trace, handle)
    with pickle_path.open("rb") as handle:
        restored = pickle.load(handle)

    restored_layer = _first_saved_layer(restored)
    assert isinstance(restored_layer.out, torch.Tensor)
    assert restored_layer.out_ref is None
    assert torch.equal(restored_layer.out, source_layer.out)
