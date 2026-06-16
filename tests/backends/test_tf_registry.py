"""TensorFlow backend registry and P1 mode-selection tests."""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.backends import (
    BackendMismatchError,
    BackendUnsupportedError,
    get_backend_spec,
    resolve_backend_spec,
)
from torchlens.backends.default_specs import _tf_can_handle
from torchlens.backends.tf import capabilities as tf_capabilities
from torchlens.backends.tf import TFBackend

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def test_tf_backend_registered_with_alias_capabilities_and_priority() -> None:
    """TensorFlow default spec is registered without importing TensorFlow."""

    assert "tensorflow" not in sys.modules

    spec = get_backend_spec("tf")
    assert get_backend_spec("tensorflow") is spec
    assert spec.name == "tf"
    assert spec.priority == 50
    assert spec.capture_backend is None
    assert spec.coercible is False
    assert get_backend_spec("jax").priority == 20
    assert get_backend_spec("paddle").priority == 40

    assert spec.capabilities.backward_capture == tf_capabilities.supports_backward_capture
    assert spec.capabilities.validation_replay == tf_capabilities.supports_validation_replay
    assert spec.capabilities.fastlog == tf_capabilities.supports_fastlog
    assert spec.capabilities.interventions == tf_capabilities.supports_intervention
    assert (
        spec.capabilities.intermediate_derived_grads
        == tf_capabilities.supports_intermediate_derived_grads
    )
    assert spec.capabilities.rng_replay == tf_capabilities.supports_rng_replay
    assert spec.capabilities.payload_materialization == (
        tf_capabilities.supports_payload_materialization
    )
    assert spec.capabilities.streaming == tf_capabilities.supports_streaming
    assert spec.capabilities.input_container_structure == (
        tf_capabilities.input_container_structure
    )
    assert spec.capabilities.output_container_structure == (
        tf_capabilities.output_container_structure
    )
    assert spec.capabilities.module_identity_modes == tf_capabilities.module_identity_modes
    assert spec.capabilities.trace_options == tf_capabilities.trace_options
    assert spec.serialization_policy.payload_policy == tf_capabilities.payload_policy
    assert spec.serialization_policy.runtime_name == "tf"
    assert spec.serialization_policy.manifest_schema_versions == (2,)


@pytest.mark.tf_backend
def test_tf_detector_and_resolution_accept_tf_module_and_tensor_leaf() -> None:
    """TensorFlow detector accepts tf.Module and callable TensorFlow tensor inputs."""

    tf = pytest.importorskip("tensorflow")
    spec = get_backend_spec("tf")

    class _TFModule(tf.Module):
        """Small TensorFlow module for routing tests."""

        def __call__(self, x: Any) -> Any:
            """Return the input unchanged."""

            return x

    x = tf.constant([1.0])
    assert _tf_can_handle(_TFModule(), x, None)
    assert _tf_can_handle(lambda y: y, {"nested": [x]}, None)
    assert resolve_backend_spec(None, _TFModule(), x) is spec
    assert resolve_backend_spec("tf", _TFModule(), x) is spec
    assert resolve_backend_spec("tensorflow", _TFModule(), x) is spec


@pytest.mark.tf_backend
def test_tf_detector_accepts_keras_on_tensorflow() -> None:
    """Keras 3 models route to TensorFlow when Keras is on its TensorFlow backend."""

    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    if keras.backend.backend() != "tensorflow":
        pytest.skip("Keras is not configured for the TensorFlow backend")

    model = keras.Sequential([keras.layers.Dense(1)])
    x = tf.ones((1, 2))
    assert _tf_can_handle(model, x, None)
    assert resolve_backend_spec(None, model, x).name == "tf"


@pytest.mark.tf_backend
def test_tf_detector_rejects_foreign_leaves_and_torch_modules() -> None:
    """TensorFlow detector rejects torch modules and any foreign tensor input leaves."""

    tf = pytest.importorskip("tensorflow")

    class _TFModule(tf.Module):
        """Small TensorFlow module for foreign-leaf tests."""

        def __call__(self, x: Any) -> Any:
            """Return the input unchanged."""

            return x

    class _TorchModule(torch.nn.Module):
        """Small torch module for routing rejection tests."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the input unchanged."""

            return x

    assert not _tf_can_handle(_TorchModule(), tf.constant([1.0]), None)
    assert not _tf_can_handle(_TFModule(), torch.ones(1), None)
    assert not _tf_can_handle(_TFModule(), {"tf": tf.constant([1.0]), "torch": torch.ones(1)}, None)


def test_tf_detector_reports_non_tf_keras_backend_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-TF Keras objects raise a typed mismatch naming the active backend."""

    keras_module = types.ModuleType("keras")
    keras_module.backend = types.SimpleNamespace(backend=lambda: "jax")

    tf_module = types.ModuleType("tensorflow")
    tf_module.Module = object
    tf_module.Tensor = object
    tf_module.Variable = object
    tf_module.types = types.SimpleNamespace(
        experimental=types.SimpleNamespace(ConcreteFunction=type("ConcreteFunction", (), {}))
    )

    class _KerasModel:
        """Fake Keras object without importing Keras."""

        __module__ = "keras.src.models"

    monkeypatch.setitem(sys.modules, "keras", keras_module)
    monkeypatch.setitem(sys.modules, "tensorflow", tf_module)

    with pytest.raises(BackendMismatchError, match="active keras backend is 'jax'"):
        _tf_can_handle(_KerasModel(), object(), None)


@pytest.mark.tf_backend
def test_tf_capture_trace_routes_to_p1_eager_boundary() -> None:
    """Public trace dispatch reaches TFBackend and stops at the P2 eager boundary."""

    tf = pytest.importorskip("tensorflow")

    class _TFModule(tf.Module):
        """Small TensorFlow module for dispatch tests."""

        def __call__(self, x: Any) -> Any:
            """Return a TensorFlow op result."""

            return x + 1

    with pytest.raises(BackendUnsupportedError, match="tf eager capture lands in P2"):
        tl.trace(_TFModule(), tf.constant([1.0]), backend="tf")


@pytest.mark.tf_backend
def test_tf_mode_selection_normalizes_eager_model_training_and_mask_kwargs() -> None:
    """TFBackend normalizes concrete inputs and selects eager mode for direct calls."""

    tf = pytest.importorskip("tensorflow")

    class _TFModule(tf.Module):
        """Small TensorFlow module for mode-selection tests."""

        def __call__(
            self,
            x: Any,
            *,
            training: bool = False,
            mask: Any | None = None,
        ) -> Any:
            """Return an eager TensorFlow op result."""

            del training, mask
            return x + 1

    x = tf.constant([1.0])
    mask = tf.constant([True])
    plan = TFBackend().normalize_call(
        model=_TFModule(),
        input_args=[x],
        input_kwargs={"training": True, "mask": mask},
    )

    assert plan.mode == "eager"
    assert plan.args == (x,)
    assert plan.call_kwargs == {"training": True, "mask": mask}
    assert "eager" in plan.reason


@pytest.mark.tf_backend
def test_tf_mode_selection_detects_tf_function_and_compiled_call() -> None:
    """TFBackend routes tf.function entries and compiled Model.call to graph-only mode."""

    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")

    fn_plan = TFBackend().normalize_call(model=tf.function(lambda x: x + 1), input_args=tf.ones(1))
    assert fn_plan.mode == "graph_only"
    assert "tf.function" in fn_plan.reason

    class _CompiledCallModel(keras.Model):
        """Keras model whose ``call`` is compiled with ``tf.function``."""

        @tf.function
        def call(self, x: Any) -> Any:
            """Return a graph-executed TensorFlow op result."""

            return x + 1

    model = _CompiledCallModel()
    model(tf.ones(1))
    call_plan = TFBackend().normalize_call(model=model, input_args=tf.ones(1))
    assert call_plan.mode == "graph_only"
    assert "Model.call" in call_plan.reason


@pytest.mark.tf_backend
def test_tf_mode_selection_detects_predict_and_saved_model_signatures() -> None:
    """TFBackend routes predict and SavedModel-like entries to graph-only mode."""

    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")

    model = keras.Sequential([keras.layers.Dense(1)])
    model(tf.ones((1, 2)))
    predict_plan = TFBackend().normalize_call(model=model.predict, input_args=tf.ones((1, 2)))
    assert predict_plan.mode == "graph_only"
    assert "predict" in predict_plan.reason

    class _LoadedSavedModel:
        """SavedModel-like callable with signatures."""

        signatures = {"serving_default": object()}

        def __call__(self, x: Any) -> Any:
            """Return the input unchanged."""

            return x

    saved_model_plan = TFBackend().normalize_call(
        model=_LoadedSavedModel(),
        input_args=tf.ones(1),
    )
    assert saved_model_plan.mode == "graph_only"
    assert "SavedModel" in saved_model_plan.reason
