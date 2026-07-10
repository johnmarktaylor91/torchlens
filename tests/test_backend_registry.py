"""Backend registry and public ``backend=`` routing tests."""

from __future__ import annotations

import ast
import builtins
import inspect
from pathlib import Path
import sys
import types
from typing import Any, Iterator, cast

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends import (
    BackendAmbiguityError,
    BackendCapabilities,
    BackendMismatchError,
    BackendSpec,
    BackendUnsupportedError,
    CaptureBackend,
    UnknownBackendError,
    SerializationPolicy,
    get_backend_spec,
    register_backend_spec,
    registered_backend_specs,
    resolve_backend_spec,
    unregister_backend_spec,
)
from torchlens.capture.trace import _capture_backend_from_registry
from torchlens.backends.jax import capabilities as jax_capabilities
from torchlens.backends.mlx import capabilities as mlx_capabilities
from torchlens.backends.paddle import capabilities as paddle_capabilities
from torchlens.backends.tinygrad import capabilities as tinygrad_capabilities
from torchlens.backends.default_specs import (
    _contains_other_backend_tensor,
    _jax_can_handle,
    _mlx_can_handle,
    _paddle_can_handle,
    _tf_can_handle,
    _tinygrad_can_handle,
)
from torchlens.backends.registry import _CAPTURE_BACKEND_REQUIRED_ATTRIBUTES
from torchlens.backends.tf import TFBackend
from torchlens.validation import check_metadata_invariants
from torchlens.validation.invariants import MetadataInvariantError


class _TinyModel(nn.Module):
    """Small torch model for backend routing tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple torch operation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Doubled tensor.
        """

        return x * 2


class _FakeModel:
    """Marker model accepted by fake backend specs."""


def _fake_can_handle(
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
) -> bool:
    """Return whether ``model`` is the fake marker model.

    Parameters
    ----------
    model:
        Candidate model.
    input_args:
        Positional inputs, unused.
    input_kwargs:
        Keyword inputs, unused.

    Returns
    -------
    bool
        ``True`` for fake marker models.
    """

    del input_args, input_kwargs
    return isinstance(model, _FakeModel)


def _fake_capture_trace(*args: Any, **kwargs: Any) -> tl.Trace:
    """Return a real trace relabeled as the fake backend.

    Parameters
    ----------
    *args, **kwargs:
        Public trace arguments.

    Returns
    -------
    tl.Trace
        Fake backend trace.
    """

    del args, kwargs
    trace = tl.trace(
        _TinyModel().eval(),
        torch.ones(1),
        layers_to_save="all",
        random_seed=1,
        backend="torch",
    )
    trace.backend = "fake"
    trace.module_identity_mode = "function_root"
    trace.param_source = "none"
    trace.model_class_name = "_FakeModel"
    trace.model_label = "_FakeModel"
    trace.model_class_qualname = "tests.test_backend_registry._FakeModel"
    trace.trace_label = "fake-backend-trace"
    for layer in trace.layer_list:
        layer.resolver_status = "metadata_only"
        layer.backend_address = f"fake:{layer.layer_label}"
    for layer in trace.layer_logs.values():
        layer.resolver_status = "metadata_only"
        layer.backend_address = f"fake:{layer.layer_label}"
    return trace


def _fake_validate_entry(*args: Any, **kwargs: Any) -> bool:
    """Return a visible fake validation result.

    Parameters
    ----------
    *args, **kwargs:
        Public validation arguments.

    Returns
    -------
    bool
        Always ``True``.
    """

    del args, kwargs
    return True


def _fake_validate_trace(*args: Any, **kwargs: Any) -> bool:
    """Run fake trace metadata validation.

    Parameters
    ----------
    *args, **kwargs:
        Trace validation arguments.

    Returns
    -------
    bool
        ``True`` when fake metadata invariants pass.
    """

    trace = args[0]
    validate_metadata = kwargs.get("validate_metadata", True)
    if validate_metadata:
        check_metadata_invariants(trace)
    return True


def _register_fake_backend(name: str = "fake", *, priority: int = 50) -> None:
    """Register a fake backend spec for tests.

    Parameters
    ----------
    name:
        Backend name.
    priority:
        Auto-resolution priority.

    Returns
    -------
    None
        The fake spec is registered.
    """

    register_backend_spec(
        BackendSpec(
            name=name,
            can_handle=_fake_can_handle,
            capture_trace=_fake_capture_trace,
            validate_entry=_fake_validate_entry,
            validate_trace=_fake_validate_trace,
            capabilities=BackendCapabilities(
                backward_capture=False,
                validation_replay=True,
                fastlog=False,
                interventions=False,
                rng_replay=False,
                payload_materialization=False,
                streaming=False,
                module_identity_modes=("function_root",),
                save_levels=("audit",),
            ),
            serialization_policy=SerializationPolicy(
                payload_policy="metadata_only",
                body_format="audit_only",
                manifest_schema_versions=(2,),
                runtime_name="fake",
            ),
            priority=priority,
        ),
    )


def test_explicit_torch_backend_matches_legacy_trace() -> None:
    """Explicit ``backend='torch'`` keeps torch capture reachable."""

    model = _TinyModel()
    x = torch.ones(1)
    legacy = tl.trace(model, x, layers_to_save="all", random_seed=1)
    explicit = tl.trace(model, x, layers_to_save="all", random_seed=1, backend="torch")
    assert explicit.backend == legacy.backend == "torch"
    assert explicit.layer_labels == legacy.layer_labels


@pytest.mark.parametrize(
    "kwargs",
    [
        {"module_identity_mode": "object_module"},
        {"payload_policy": "not_a_policy"},
        {"save_preview": True},
        {"jax_control_flow": "reject"},
    ],
)
def test_torch_rejects_explicit_inert_trace_option_values(kwargs: dict[str, Any]) -> None:
    """Torch should reject explicit public options it cannot honor."""

    with pytest.raises(BackendUnsupportedError):
        tl.trace(_TinyModel(), torch.ones(1), backend="torch", **kwargs)


def test_torch_accepts_default_equivalent_trace_option_values() -> None:
    """Torch accepts supported/default-equivalent public option values."""

    trace = tl.trace(
        _TinyModel(),
        torch.ones(1),
        backend="torch",
        module_identity_mode="torch_module",
        payload_policy="full",
        save_preview=False,
    )

    assert trace.backend == "torch"
    assert trace.module_identity_mode == "torch_module"


def test_capture_backend_factory_checked_at_registration() -> None:
    """Registration rejects incomplete shared capture protocol adapters."""

    def bad_capture_backend() -> CaptureBackend:
        """Return an object missing the shared capture protocol.

        Returns
        -------
        CaptureBackend
            Deliberately invalid adapter for conformance coverage.
        """

        return cast(CaptureBackend, object())

    with pytest.raises(TypeError, match="capture_backend.*missing"):
        register_backend_spec(
            BackendSpec(
                name="fake_conformance",
                can_handle=_fake_can_handle,
                capture_trace=_fake_capture_trace,
                validate_entry=_fake_validate_entry,
                validate_trace=_fake_validate_trace,
                capabilities=BackendCapabilities(
                    backward_capture=False,
                    validation_replay=False,
                    fastlog=False,
                    interventions=False,
                    rng_replay=False,
                    payload_materialization=False,
                    streaming=False,
                ),
                capture_backend=bad_capture_backend,
            )
        )


def test_capture_backend_factory_import_error_is_not_exempted() -> None:
    """Registration rejects factories that cannot construct a backend adapter."""

    def bad_capture_backend() -> CaptureBackend:
        """Raise the formerly exempt circular-import shaped error."""

        raise ImportError("cannot import name X from partially initialized module Y")

    with pytest.raises(ImportError, match="partially initialized module"):
        register_backend_spec(
            BackendSpec(
                name="fake_partial_import",
                can_handle=_fake_can_handle,
                capture_trace=_fake_capture_trace,
                validate_entry=_fake_validate_entry,
                validate_trace=_fake_validate_trace,
                capabilities=BackendCapabilities(
                    backward_capture=False,
                    validation_replay=False,
                    fastlog=False,
                    interventions=False,
                    rng_replay=False,
                    payload_materialization=False,
                    streaming=False,
                ),
                capture_backend=bad_capture_backend,
            )
        )


def test_replace_backend_spec_removes_stale_aliases() -> None:
    """Replacing a spec removes aliases owned by the old spec."""

    for name in ("fake_alias_probe", "fake_alias_probe_old"):
        unregister_backend_spec(name)
    old_spec = BackendSpec(
        name="fake_alias_probe",
        aliases=("fake_alias_probe_old",),
        can_handle=_fake_can_handle,
        capture_trace=_fake_capture_trace,
        validate_entry=_fake_validate_entry,
        validate_trace=_fake_validate_trace,
        capabilities=BackendCapabilities(
            backward_capture=False,
            validation_replay=False,
            fastlog=False,
            interventions=False,
            rng_replay=False,
            payload_materialization=False,
            streaming=False,
        ),
    )
    new_spec = BackendSpec(
        name="fake_alias_probe",
        can_handle=_fake_can_handle,
        capture_trace=_fake_capture_trace,
        validate_entry=_fake_validate_entry,
        validate_trace=_fake_validate_trace,
        capabilities=old_spec.capabilities,
    )
    try:
        register_backend_spec(old_spec)
        assert get_backend_spec("fake_alias_probe_old") is old_spec

        register_backend_spec(new_spec, replace=True)

        assert get_backend_spec("fake_alias_probe") is new_spec
        with pytest.raises(UnknownBackendError):
            get_backend_spec("fake_alias_probe_old")
    finally:
        unregister_backend_spec("fake_alias_probe")
        unregister_backend_spec("fake_alias_probe_old")


def test_capability_sources_agree_for_preview_backends() -> None:
    """Compatibility capability modules read from the registered specs."""

    jax_spec = get_backend_spec("jax")
    mlx_spec = get_backend_spec("mlx")
    paddle_spec = get_backend_spec("paddle")
    tinygrad_spec = get_backend_spec("tinygrad")

    assert jax_spec.capabilities.backward_capture == jax_capabilities.supports_backward_capture
    assert jax_spec.capabilities.validation_replay == jax_capabilities.supports_validation_replay
    assert jax_spec.capabilities.fastlog == jax_capabilities.supports_fastlog
    assert jax_spec.capabilities.interventions == jax_capabilities.supports_intervention
    assert (
        jax_spec.capabilities.intermediate_derived_grads
        == jax_capabilities.supports_intermediate_derived_grads
    )
    assert jax_spec.capabilities.rng_replay == jax_capabilities.supports_rng_replay
    assert (
        jax_spec.capabilities.payload_materialization
        == jax_capabilities.supports_payload_materialization
    )
    assert jax_spec.capabilities.module_identity_modes == jax_capabilities.module_identity_modes
    assert jax_spec.capabilities.trace_options == jax_capabilities.trace_options
    assert (
        jax_spec.capabilities.input_container_structure
        == jax_capabilities.input_container_structure
    )
    assert (
        jax_spec.capabilities.output_container_structure
        == jax_capabilities.output_container_structure
    )
    assert jax_spec.serialization_policy.payload_policy == jax_capabilities.payload_policy

    assert get_backend_spec("torch").capabilities.input_container_structure == "full_spec"
    assert get_backend_spec("torch").capabilities.output_container_structure == "full_spec"

    assert mlx_spec.capabilities.backward_capture == mlx_capabilities.supports_backward_capture
    assert mlx_spec.capabilities.validation_replay == mlx_capabilities.supports_validation_replay
    assert mlx_spec.capabilities.fastlog == mlx_capabilities.supports_fastlog
    assert mlx_spec.capabilities.interventions == mlx_capabilities.supports_intervention
    assert (
        mlx_spec.capabilities.intermediate_derived_grads
        == mlx_capabilities.supports_intermediate_derived_grads
    )
    assert mlx_spec.capabilities.rng_replay == mlx_capabilities.supports_rng_replay
    assert (
        mlx_spec.capabilities.payload_materialization
        == mlx_capabilities.supports_payload_materialization
    )
    assert mlx_spec.capabilities.module_identity_modes == mlx_capabilities.module_identity_modes
    assert mlx_spec.capabilities.trace_options == mlx_capabilities.trace_options
    assert (
        mlx_spec.capabilities.input_container_structure
        == mlx_capabilities.input_container_structure
    )
    assert (
        mlx_spec.capabilities.output_container_structure
        == mlx_capabilities.output_container_structure
    )
    assert mlx_spec.serialization_policy.payload_policy == mlx_capabilities.payload_policy

    assert (
        tinygrad_spec.capabilities.backward_capture
        == tinygrad_capabilities.supports_backward_capture
    )
    assert (
        tinygrad_spec.capabilities.validation_replay
        == tinygrad_capabilities.supports_validation_replay
    )
    assert tinygrad_spec.capabilities.fastlog == tinygrad_capabilities.supports_fastlog
    assert tinygrad_spec.capabilities.interventions == tinygrad_capabilities.supports_intervention
    assert (
        tinygrad_spec.capabilities.intermediate_derived_grads
        == tinygrad_capabilities.supports_intermediate_derived_grads
    )
    assert tinygrad_spec.capabilities.rng_replay == tinygrad_capabilities.supports_rng_replay
    assert (
        tinygrad_spec.capabilities.payload_materialization
        == tinygrad_capabilities.supports_payload_materialization
    )
    assert (
        tinygrad_spec.capabilities.module_identity_modes
        == tinygrad_capabilities.module_identity_modes
    )
    assert tinygrad_spec.capabilities.trace_options == tinygrad_capabilities.trace_options
    assert (
        tinygrad_spec.capabilities.input_container_structure
        == tinygrad_capabilities.input_container_structure
    )
    assert (
        tinygrad_spec.capabilities.output_container_structure
        == tinygrad_capabilities.output_container_structure
    )
    assert tinygrad_spec.serialization_policy.payload_policy == tinygrad_capabilities.payload_policy

    assert (
        paddle_spec.capabilities.backward_capture == paddle_capabilities.supports_backward_capture
    )
    assert (
        paddle_spec.capabilities.validation_replay == paddle_capabilities.supports_validation_replay
    )
    assert paddle_spec.capabilities.fastlog == paddle_capabilities.supports_fastlog
    assert paddle_spec.capabilities.interventions == paddle_capabilities.supports_intervention
    assert (
        paddle_spec.capabilities.intermediate_derived_grads
        == paddle_capabilities.supports_intermediate_derived_grads
    )
    assert paddle_spec.capabilities.rng_replay == paddle_capabilities.supports_rng_replay
    assert (
        paddle_spec.capabilities.payload_materialization
        == paddle_capabilities.supports_payload_materialization
    )
    assert (
        paddle_spec.capabilities.module_identity_modes == paddle_capabilities.module_identity_modes
    )
    assert paddle_spec.capabilities.trace_options == paddle_capabilities.trace_options
    assert (
        paddle_spec.capabilities.input_container_structure
        == paddle_capabilities.input_container_structure
    )
    assert (
        paddle_spec.capabilities.output_container_structure
        == paddle_capabilities.output_container_structure
    )
    assert paddle_spec.serialization_policy.payload_policy == paddle_capabilities.payload_policy


def test_paddle_backend_registered_with_alias_and_priority() -> None:
    """Paddle default spec is registered with its alias and phase priority."""

    spec = get_backend_spec("paddle")
    assert spec.name == "paddle"
    assert spec.capture_backend is None
    assert get_backend_spec("paddlepaddle") is spec
    assert spec.priority == 40
    assert get_backend_spec("torch").priority == 0
    assert get_backend_spec("mlx").priority == 10
    assert get_backend_spec("jax").priority == 20
    assert get_backend_spec("tinygrad").priority == 30


def test_paddle_detector_returns_false_without_paddle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paddle detector fails closed when Paddle cannot be imported."""

    original_import = builtins.__import__
    had_paddle = "paddle" in sys.modules

    def _raise_for_paddle(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Raise ``ImportError`` for Paddle and delegate all other imports."""

        if name == "paddle":
            raise ImportError("simulated missing paddle")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raise_for_paddle)

    assert not _paddle_can_handle(lambda x: x, object(), None)
    assert not _paddle_can_handle(_TinyModel(), torch.ones(1), None)
    if not had_paddle:
        assert "paddle" not in sys.modules


def test_paddle_detector_accepts_layer_and_nested_tensor() -> None:
    """Paddle detector accepts Paddle layers and nested Paddle tensor inputs."""

    paddle = pytest.importorskip("paddle")

    class _PaddleLayer(paddle.nn.Layer):
        """Small Paddle layer for backend routing tests."""

        def forward(self, x: Any) -> Any:
            """Return the input unchanged."""

            return x

    tensor = paddle.to_tensor([1.0])
    assert _paddle_can_handle(_PaddleLayer(), tensor, None)
    assert _paddle_can_handle(lambda x: x, {"nested": [tensor]}, None)
    assert not _paddle_can_handle(_TinyModel(), torch.ones(1), None)


def test_mlx_detector_rejects_mixed_foreign_tensor_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLX detector rejects inputs containing another backend tensor family."""

    mlx_module = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_nn = types.ModuleType("mlx.nn")
    MlxArray = type("array", (), {"__module__": "mlx.core"})
    MlxModule = type("Module", (), {"__module__": "mlx.nn", "__call__": lambda self, x: x})
    TFTensor = type("Tensor", (), {"__module__": "tensorflow.python.framework.ops"})
    mlx_core.array = MlxArray
    mlx_nn.Module = MlxModule
    mlx_module.core = mlx_core
    mlx_module.nn = mlx_nn
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx.nn", mlx_nn)

    assert _mlx_can_handle(MlxModule(), MlxArray(), None)
    assert not _mlx_can_handle(MlxModule(), (MlxArray(), TFTensor()), None)
    assert resolve_backend_spec(None, MlxModule(), (MlxArray(), TFTensor())).name == "torch"
    with pytest.raises(ValueError, match="Unsupported model type"):
        tl.trace(MlxModule(), (MlxArray(), TFTensor()))


def test_jax_detector_rejects_mixed_foreign_tensor_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JAX detector rejects inputs containing another backend tensor family."""

    jax_module = types.ModuleType("jax")
    JaxArray = type("Array", (), {"__module__": "jax"})
    PaddleTensor = type("Tensor", (), {"__module__": "paddle.base.framework"})

    def _flatten(value: object) -> tuple[list[object], None]:
        """Flatten nested fake JAX inputs for detector coverage."""

        return list(_test_leaves(value)), None

    jax_module.Array = JaxArray
    jax_module.tree = types.SimpleNamespace(flatten=_flatten)
    monkeypatch.setitem(sys.modules, "jax", jax_module)

    assert _jax_can_handle(lambda x: x, (JaxArray(),), None)
    assert not _jax_can_handle(lambda x: x, (JaxArray(),), {"other": PaddleTensor()})


def test_tinygrad_detector_rejects_mixed_foreign_tensor_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tinygrad detector rejects inputs containing another backend tensor family."""

    tinygrad_module = types.ModuleType("tinygrad")
    TinyTensor = type("Tensor", (), {"__module__": "tinygrad.tensor"})
    JaxArray = type("Array", (), {"__module__": "jaxlib.xla_extension"})
    tinygrad_module.Tensor = TinyTensor
    monkeypatch.setitem(sys.modules, "tinygrad", tinygrad_module)

    assert _tinygrad_can_handle(lambda x: x, (TinyTensor(),), None)
    assert not _tinygrad_can_handle(lambda x: x, (TinyTensor(), JaxArray()), None)


def test_paddle_detector_rejects_mixed_foreign_tensor_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paddle detector rejects inputs containing another backend tensor family."""

    paddle_module = types.ModuleType("paddle")
    PaddleTensor = type("Tensor", (), {"__module__": "paddle.base.framework"})
    PaddleLayer = type("Layer", (), {"__module__": "paddle.nn.layer", "__call__": lambda s, x: x})
    TFTensor = type("Tensor", (), {"__module__": "tensorflow.python.framework.ops"})
    paddle_module.Tensor = PaddleTensor
    paddle_module.nn = types.SimpleNamespace(Layer=PaddleLayer)
    monkeypatch.setitem(sys.modules, "paddle", paddle_module)

    assert _paddle_can_handle(PaddleLayer(), PaddleTensor(), None)
    assert not _paddle_can_handle(PaddleLayer(), (PaddleTensor(), TFTensor()), None)


def _test_leaves(value: object) -> Iterator[object]:
    """Yield leaves from simple nested test containers."""

    if isinstance(value, dict):
        for item in value.values():
            yield from _test_leaves(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _test_leaves(item)
        return
    yield value


def test_tf_foreign_leaf_guard_covers_mlx_and_tinygrad_modules() -> None:
    """TensorFlow foreign-leaf detection rejects MLX/tinygrad-shaped leaves."""

    MlxArray = type("array", (), {"__module__": "mlx.core"})
    TinygradTensor = type("Tensor", (), {"__module__": "tinygrad.tensor"})

    assert _contains_other_backend_tensor("tf", MlxArray(), None)
    assert _contains_other_backend_tensor("tf", TinygradTensor(), None)


def test_tf_detector_foreign_leaf_guard_runs_before_tensorflow_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreign leaves should make TensorFlow detection fail without importing TF."""

    original_import = builtins.__import__
    MlxArray = type("array", (), {"__module__": "mlx.core"})

    def _raise_for_tensorflow(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Raise if TensorFlow/Keras imports are attempted."""

        if name in {"tensorflow", "keras"}:
            raise AssertionError(f"unexpected import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raise_for_tensorflow)

    assert not _tf_can_handle(lambda x: x, MlxArray(), None)


def test_tf_detector_declines_unsupported_tensorflow_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TensorFlow routing requires Keras 3 on TensorFlow >= 2.16."""

    keras_module = types.ModuleType("keras")
    keras_module.__version__ = "2.13.1"
    keras_module.backend = types.SimpleNamespace(backend=lambda: "tensorflow")

    tf_module = types.ModuleType("tensorflow")
    tf_module.__version__ = "2.14.0"
    tf_module.Module = type("TFModule", (), {})
    tf_module.Tensor = type("TFTensor", (), {})
    tf_module.Variable = type("TFVariable", (), {})
    tf_module.types = types.SimpleNamespace(
        experimental=types.SimpleNamespace(ConcreteFunction=type("ConcreteFunction", (), {}))
    )

    monkeypatch.setitem(sys.modules, "keras", keras_module)
    monkeypatch.setitem(sys.modules, "tensorflow", tf_module)

    assert not _tf_can_handle(tf_module.Module(), object(), None)


def test_tf_detector_swallows_non_import_error_from_broken_tf_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken-but-importable TF/Keras install must not crash the autorouter.

    Regression test: ``_tf_can_handle`` previously wrapped ``import keras`` /
    ``import tensorflow`` in a ``try/except ImportError`` only. A TF/Keras
    install that is present but broken (e.g. a numpy/protobuf ABI mismatch)
    can raise exceptions other than ``ImportError`` -- ``TypeError``,
    ``AttributeError``, ``RuntimeError``, etc. -- from the import statements
    themselves or from any keras/tf attribute access used afterward to
    determine handleability. Those exceptions previously propagated
    uncaught out of the can-handle PROBE, up through
    ``resolve_backend_spec``'s ``[... for spec in registered_backend_specs()
    if spec.can_handle(...)]`` comprehension, crashing the entire
    autorouter -- and thus ANY capture attempt, regardless of which backend
    the caller actually wanted -- any time TF happens to be
    installed-but-broken in the environment. Confirmed by reverting the fix
    and re-running this exact scenario (see below). Now the probe treats
    any such failure as "cannot handle" and autorouting continues to work
    for other backends.
    """

    original_import = builtins.__import__

    def _raise_non_import_error(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Simulate a broken-but-technically-importable TF/Keras install."""

        if name in {"tensorflow", "keras"}:
            raise RuntimeError("simulated numpy/protobuf ABI break")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raise_non_import_error)

    # Must NOT raise -- a broken TF/Keras install is only "cannot handle",
    # never a crash.
    assert _tf_can_handle(lambda x: x, object(), None) is False

    # Autorouting must still resolve a plain torch nn.Module to the torch
    # backend even though the TF probe raised internally -- the crash must
    # not propagate up through ``resolve_backend_spec``.
    assert resolve_backend_spec(None, _TinyModel(), torch.ones(1)).name == "torch"
    trace = tl.trace(_TinyModel(), torch.ones(1))
    assert trace.backend == "torch"


def test_tf_detector_reraises_genuine_backend_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The broadened exception guard must not swallow a real backend mismatch.

    ``_tf_can_handle`` deliberately raises ``BackendMismatchError`` when a
    Keras object is configured for a non-TensorFlow Keras backend and
    ``backend='tf'`` is explicitly requested. That is actionable user-facing
    signal, not an ABI crash, so the broadened ``except Exception`` guard
    added to swallow broken-install probe failures must not also swallow it.
    """

    keras_module = types.ModuleType("keras")
    keras_module.__version__ = "3.0.0"
    keras_module.backend = types.SimpleNamespace(backend=lambda: "torch")

    tf_module = types.ModuleType("tensorflow")
    tf_module.__version__ = "2.16.0"
    tf_module.Module = type("TFModule", (), {})
    tf_module.Tensor = type("TFTensor", (), {})
    tf_module.Variable = type("TFVariable", (), {})
    tf_module.types = types.SimpleNamespace(
        experimental=types.SimpleNamespace(ConcreteFunction=type("ConcreteFunction", (), {}))
    )

    monkeypatch.setitem(sys.modules, "keras", keras_module)
    monkeypatch.setitem(sys.modules, "tensorflow", tf_module)

    keras_model = type("KerasModel", (), {"__module__": "keras.src.models.model"})()
    with pytest.raises(BackendMismatchError, match="active keras backend is 'torch'"):
        _tf_can_handle(keras_model, object(), None)


def test_tf_backend_rejects_random_seed_without_importing_tensorflow() -> None:
    """TensorFlow preview random_seed is a typed unsupported option."""

    with pytest.raises(BackendUnsupportedError, match="random_seed"):
        TFBackend().capture_trace(lambda x: x, object(), random_seed=123)


def test_paddle_shared_capture_backend_is_unsupported_typed_error() -> None:
    """Paddle shared-orchestration lookup raises the canonical unsupported error."""

    paddle = pytest.importorskip("paddle")

    class _PaddleLayer(paddle.nn.Layer):
        """Small Paddle layer for shared-capture resolution tests."""

        def forward(self, x: Any) -> Any:
            """Return the input unchanged."""

            return x

    with pytest.raises(BackendUnsupportedError, match="shared capture Protocol adapter"):
        _capture_backend_from_registry("paddle", _PaddleLayer(), paddle.to_tensor([1.0]), None)


def test_explicit_paddle_backend_resolves_to_spec() -> None:
    """Explicit Paddle backend and alias resolve to the Paddle spec."""

    paddle = pytest.importorskip("paddle")

    class _PaddleLayer(paddle.nn.Layer):
        """Small Paddle layer for explicit resolution tests."""

        def forward(self, x: Any) -> Any:
            """Return the input unchanged."""

            return x

    x = paddle.to_tensor([1.0])
    spec = get_backend_spec("paddle")
    assert resolve_backend_spec("paddle", _PaddleLayer(), x) is spec
    assert resolve_backend_spec("paddlepaddle", _PaddleLayer(), x) is spec


def test_paddle_preview_unsupported_options_raise_typed_error() -> None:
    """Paddle preview rejects unsupported options with canonical typed errors."""

    paddle = pytest.importorskip("paddle")

    class _PaddleLayer(paddle.nn.Layer):
        """Small Paddle layer for unsupported-option routing tests."""

        def forward(self, x: Any) -> Any:
            """Return the input unchanged."""

            return x

    with pytest.raises(BackendUnsupportedError):
        tl.trace(_PaddleLayer(), paddle.to_tensor([1.0]), backend="paddle", backward_ready=True)


def test_paddle_preview_applies_static_label_save_selector() -> None:
    """Paddle accepts advertised static save selectors and filters public payloads."""

    paddle = pytest.importorskip("paddle")

    class _PaddleRelu(paddle.nn.Layer):
        """Small Paddle model with a selectively saved operation."""

        def forward(self, x: Any) -> Any:
            """Apply a deterministic ReLU."""

            return paddle.nn.functional.relu(x)

    model = _PaddleRelu()
    model.eval()
    trace = tl.trace(
        model,
        paddle.to_tensor([-1.0, 2.0]),
        backend="paddle",
        save=tl.contains("relu"),
    )

    saved_ops = [op for op in trace.layer_list if op.has_saved_activation]
    assert len(saved_ops) == 1
    assert saved_ops[0].func_name == "functional.relu"


def test_tf_version_gate_reports_installed_version_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported TF/Keras versions raise the actionable explicit-backend mismatch."""

    keras_module = types.SimpleNamespace(__version__="2.13.1")
    tf_module = types.SimpleNamespace(__version__="2.14.0")
    monkeypatch.setitem(sys.modules, "keras", keras_module)
    monkeypatch.setitem(sys.modules, "tensorflow", tf_module)
    keras_model_type = type("Functional", (), {"__module__": "keras.src.models"})

    with pytest.raises(
        BackendMismatchError,
        match=r"TF backend requires Keras>=3 and TF>=2\.16; found keras 2\.13\.1 / tf 2\.14\.0",
    ):
        _tf_can_handle(keras_model_type(), (), None)


def test_registered_capture_backends_conform_to_protocol() -> None:
    """Every registered shared-capture adapter exposes the full protocol surface."""

    required_attrs = tuple(CaptureBackend.__annotations__) + tuple(
        name
        for name, value in CaptureBackend.__dict__.items()
        if not name.startswith("_") and callable(value)
    )
    dependency_modules = {
        "jax": "jax",
        "mlx": "mlx",
        "paddle": "paddle",
        "tf": "tensorflow",
        "tinygrad": "tinygrad",
    }

    for spec in registered_backend_specs():
        if spec.capture_backend is None:
            continue
        if spec.name in dependency_modules:
            pytest.importorskip(dependency_modules[str(spec.name)])
        backend = spec.capture_backend()
        missing = [attr for attr in required_attrs if not hasattr(backend, attr)]
        assert missing == [], f"{spec.name} capture backend missing attrs: {missing}"


def test_dead_correctness_protocol_methods_are_not_required() -> None:
    """Dead isolation/autocast hooks stay out of the mandatory backend contract."""

    dead_attrs = {
        "detect_in_place_isolation_required",
        "isolate_same_object_returns",
        "mark_same_object_candidates",
        "snapshot_autocast",
    }

    assert dead_attrs.isdisjoint(_CAPTURE_BACKEND_REQUIRED_ATTRIBUTES)
    assert all(not hasattr(CaptureBackend, attr) for attr in dead_attrs)


def test_public_trace_dispatches_through_backend_spec() -> None:
    """Public ``trace`` dispatch stays owned by the backend spec."""

    source = inspect.getsource(tl.trace)
    assert "capture_trace(**public_trace_kwargs)" in source
    assert "resolved_spec.name" not in source


@pytest.mark.slow
def test_public_backend_literal_branches_stay_in_registry_or_backends() -> None:
    """Public code has no new hard-coded backend literal branches."""

    project_root = Path(__file__).resolve().parents[1]
    allowed_dirs = {
        project_root / "torchlens" / "backends",
    }
    allowed_files = {
        project_root / "torchlens" / "_io" / "bundle.py",
        project_root / "torchlens" / "_io" / "tlspec.py",
        project_root / "torchlens" / "data_classes" / "trace.py",
        project_root / "torchlens" / "repgeom" / "__init__.py",
    }
    backend_literals = {"torch", "mlx", "jax", "tinygrad", "paddle", "fake"}
    offenders: list[str] = []

    for source_path in sorted((project_root / "torchlens").rglob("*.py")):
        if any(source_path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
            continue
        if source_path in allowed_files:
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        source = source_path.read_text(encoding="utf-8")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            expression = ast.get_source_segment(source, node) or ""
            if "backend" not in expression:
                continue
            compared_literals = {
                item.value
                for item in [node.left, *node.comparators]
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            if compared_literals & backend_literals:
                relpath = source_path.relative_to(project_root)
                offenders.append(f"{relpath}:{node.lineno}: {expression}")

    assert offenders == []


def test_explicit_backend_mismatch_is_deterministic() -> None:
    """Explicit torch selection rejects non-torch models before capture."""

    with pytest.raises(BackendMismatchError, match="backend='torch' cannot handle"):
        tl.trace(object(), torch.ones(1), backend="torch")


def test_fake_backend_explicit_trace_and_validate() -> None:
    """Registered fake backend drives public trace and validation entries."""

    _register_fake_backend()
    try:
        result = tl.trace(_FakeModel(), object(), backend="fake")
        assert isinstance(result, tl.Trace)
        assert result.backend == "fake"
        assert result.module_identity_mode == "function_root"
        assert result.param_source == "none"
        assert result.validate_forward_pass([]) is True
        assert tl.validate(_FakeModel(), object(), scope="forward", backend="fake")
    finally:
        unregister_backend_spec("fake")


def test_public_option_spine_rejects_unsupported_explicit_option() -> None:
    """Unsupported explicit public-spine options fail before backend capture."""

    _register_fake_backend()
    try:
        with pytest.raises(BackendUnsupportedError, match="module_identity_mode selection"):
            tl.trace(
                _FakeModel(),
                object(),
                backend="fake",
                module_identity_mode="function_root",
            )
    finally:
        unregister_backend_spec("fake")


def test_fake_backend_trace_save_load_accessors_and_invariants(tmp_path: Path) -> None:
    """Fake backend trace round-trips metadata and exposes neutral accessors."""

    _register_fake_backend()
    try:
        trace = tl.trace(_FakeModel(), object(), backend="fake")
        path = tmp_path / "fake.tlspec"

        trace.save(path, level="audit")
        loaded = tl.load(path)

        assert isinstance(loaded, tl.Trace)
        assert loaded.backend == "fake"
        assert loaded.module_identity_mode == "function_root"
        assert loaded.param_source == "none"
        assert loaded[0].resolver_status == "metadata_only"
        assert loaded[0].backend_address.startswith("fake:")
        assert check_metadata_invariants(loaded) is True
    finally:
        unregister_backend_spec("fake")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda trace: setattr(trace, "module_identity_mode", "torch_module"), "module_identity"),
        (lambda trace: setattr(trace, "has_backward_pass", True), "has_backward_pass"),
        (lambda trace: trace.grad_fn_logs.__setitem__(1, object()), "grad_fn_logs"),
        (lambda trace: setattr(trace[0], "resolver_status", "lost"), "resolver_status"),
        (lambda trace: trace.output_layers.clear(), "output layer"),
    ],
)
def test_fake_backend_invariant_corruptions_fail(
    mutate: Any,
    match: str,
) -> None:
    """Non-torch invariant gates reject representative corruptions."""

    _register_fake_backend()
    try:
        trace = tl.trace(_FakeModel(), object(), backend="fake")
        mutate(trace)

        with pytest.raises(MetadataInvariantError, match=match):
            check_metadata_invariants(trace)
    finally:
        unregister_backend_spec("fake")


def test_backend_none_ambiguity_is_deterministic() -> None:
    """Equal-priority detector collisions fail with a canonical error."""

    _register_fake_backend("fake_a", priority=99)
    _register_fake_backend("fake_b", priority=99)
    try:
        with pytest.raises(BackendAmbiguityError, match="fake_a, fake_b"):
            tl.trace(_FakeModel(), object())
    finally:
        unregister_backend_spec("fake_a")
        unregister_backend_spec("fake_b")
