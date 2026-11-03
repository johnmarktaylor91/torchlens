"""Regression tests for plain ``pickle.dump`` compatibility."""

from __future__ import annotations

import pickle
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
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
    return trace_fn(
        model,
        inputs,
        capture=tl.options.CaptureOptions(seed=seed),
        save="all",
    )


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

    return next(layer for layer in trace._layer_list if layer.has_saved_activation)


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

    from torchlens._data_classes.buffer import Buffer

    torch.manual_seed(0)
    model = _BufferModel()
    model.train()
    trace = trace_fn(model, torch.randn(8, 4))
    buffers = list(trace.buffers)
    assert buffers, "expected at least one captured buffer"
    restored = pickle.loads(pickle.dumps(buffers[0]))
    assert isinstance(restored, Buffer)
    # The live weakref must not survive; it is rebuilt lazily via the accessor.
    assert restored.trace is None
    assert isinstance(restored.versions, list)


def test_backend_address_none_is_preserved_and_pre_floor_refuses() -> None:
    """Explicit ``None`` survives restore; pre-floor states refuse typed.

    The v5 ``backend_address`` repair ladder is deleted: the tlspec-6 floor
    guarantees every loadable state already carries the field, so an explicit
    ``None`` must never be overwritten from ``address``, and a v4 state must
    refuse instead of triggering the migration.
    """

    from torchlens._data_classes.layer import Layer
    from torchlens._data_classes.op import Op
    from torchlens._io import TLSPEC_VERSION, ArtifactTooOldError

    trace = _build_trace()
    op_state = trace.ops[0].__getstate__()
    layer_state = trace._layer_list[0].__getstate__()
    for state, factory in (
        (op_state, lambda: Op.__new__(Op)),
        (layer_state, lambda: Layer.__new__(Layer)),
    ):
        state["address"] = "plain.attribute"
        state["backend_address"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        current = factory()
        current.__setstate__(dict(state))
        assert current.backend_address is None

        state["tlspec_version"] = 4
        legacy = factory()
        with pytest.raises(ArtifactTooOldError, match="tlspec_version=4"):
            legacy.__setstate__(dict(state))


def test_all_field_order_container_defaults_are_typed() -> None:
    """Drift guard: every container FIELD_ORDER field defaults to its type.

    Constructs a live trainable-model trace and, for each ``FIELD_ORDER``
    field that is a container at runtime, strips it from state and asserts the
    restored value is a matching container (never ``None``). This catches any
    future field added without a typed default in the ``__setstate__`` fill.
    """

    from torchlens._constants import LAYER_PASS_LOG_FIELD_ORDER, MODEL_LOG_FIELD_ORDER
    from torchlens._data_classes.op import Op

    torch.manual_seed(0)
    trace = trace_fn(_TrainableModel(), torch.randn(3, 4))

    def _assert_container_defaults(instance: Any, field_order: list[str], factory: Any) -> None:
        base_state = instance.__getstate__()
        for field_name in field_order:
            if field_name not in base_state:
                continue
            if field_name == "_capture_outcome":
                # Persisted as a string-only CODEC dict but deliberately
                # restored as the typed CaptureOutcome record (parsed against
                # closed vocabularies); an absent key restores a DERIVED
                # record, never None, so the typed-default invariant this
                # guard exists for is satisfied by a non-container type.
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
        trace._layer_list[1], list(LAYER_PASS_LOG_FIELD_ORDER), lambda: Op.__new__(Op)
    )


def test_plain_pickle_survives_a_lambda_activation_transform() -> None:
    """R10-7: a lambda transform= must not crash pickle while Trace.save works.

    Fail-before: ``pickle.dumps`` raised ``PicklingError: Can't pickle
    <lambda>`` from the op/layer records; the raw callable fields now
    serialize to the loaded-artifact form (None), like every other
    live-user-object DROP field.
    """

    log = trace_fn(
        _PlainPickleModel(),
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(activation_transform=lambda t: t * 2),
    )
    restored = pickle.loads(pickle.dumps(log))
    assert restored.options.activation_transform is None
    assert restored._layer_list[0].activation_transform is None
