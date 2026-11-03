"""Backend registry and public ``backend=`` routing tests."""

from __future__ import annotations

import ast
import builtins
import inspect
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from _source_corpus import package_ast, package_files, package_source
from torch import nn

import torchlens as tl
from torchlens._capture.prepare.dispatch import _dispatch_trace
from torchlens._capture.trace import _capture_backend_from_registry
from torchlens.backends import (
    BackendCapabilities,
    BackendSpec,
    CaptureBackend,
)
from torchlens.backends.default_specs import (
    _jax_can_handle,
    _mlx_can_handle,
    _paddle_can_handle,
    _tf_can_handle,
    _tinygrad_can_handle,
)
from torchlens.backends.registry import (
    _CAPTURE_BACKEND_REQUIRED_ATTRIBUTES,
    BackendRegistryError,
    SerializationPolicy,
)
from torchlens.backends.tf import TFBackend
from torchlens.errors import (
    AmbiguousBackendError,
    BackendMismatchError,
    BackendUnsupportedError,
    UnknownBackendError,
)
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
        backend="torch",
        capture=tl.options.CaptureOptions(seed=1),
        save="all",
    )
    trace._provenance_backend = "fake"
    trace._options_module_identity_mode = "function_root"
    trace._provenance_param_source = "none"
    trace._source_class_name = "_FakeModel"
    trace.model_name = "_FakeModel"
    trace._source_class_qualname = "tests.test_backend_registry._FakeModel"
    trace.name = "fake-backend-trace"
    for layer in trace._layer_list:
        layer.resolver_status = "metadata_only"
        layer.backend_address = f"fake:{layer.layer_label}"
    for layer in trace._layer_logs.values():
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

    tl.backends.register(
        BackendSpec(
            name=name,
            can_handle=_fake_can_handle,
            capture_trace=_fake_capture_trace,
            validate_entry=_fake_validate_entry,
            validate_trace=_fake_validate_trace,
            capabilities=BackendCapabilities(
                backward_capture=False,
                has_validation_replay=True,
                fastlog=False,
                can_intervene=False,
                rng_replay=False,
                payload_materialization=False,
                has_streaming=False,
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/prepare/dispatch.py.
@pytest.mark.parametrize(
    "kwargs",
    [
        {"module_identity_mode": "object_module"},
        {"payload_policy": "not_a_policy"},
    ],
)
def test_torch_rejects_explicit_inert_trace_option_values(kwargs: dict[str, Any]) -> None:
    """Torch should reject explicit public options it cannot honor."""

    with pytest.raises(BackendUnsupportedError):
        tl.trace(
            _TinyModel(),
            torch.ones(1),
            backend="torch",
            capture=tl.options.CaptureOptions(**kwargs),
        )


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/registry.py.
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
        tl.backends.register(
            BackendSpec(
                name="fake_conformance",
                can_handle=_fake_can_handle,
                capture_trace=_fake_capture_trace,
                validate_entry=_fake_validate_entry,
                validate_trace=_fake_validate_trace,
                capabilities=BackendCapabilities(
                    backward_capture=False,
                    has_validation_replay=False,
                    fastlog=False,
                    can_intervene=False,
                    rng_replay=False,
                    payload_materialization=False,
                    has_streaming=False,
                ),
                capture_backend=bad_capture_backend,
            )
        )


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/registry.py.
def test_capture_backend_factory_import_error_is_not_exempted() -> None:
    """Registration rejects factories that cannot construct a backend adapter."""

    def bad_capture_backend() -> CaptureBackend:
        """Raise the formerly exempt circular-import shaped error."""

        raise ImportError("cannot import name X from partially initialized module Y")

    with pytest.raises(ImportError, match="partially initialized module"):
        tl.backends.register(
            BackendSpec(
                name="fake_partial_import",
                can_handle=_fake_can_handle,
                capture_trace=_fake_capture_trace,
                validate_entry=_fake_validate_entry,
                validate_trace=_fake_validate_trace,
                capabilities=BackendCapabilities(
                    backward_capture=False,
                    has_validation_replay=False,
                    fastlog=False,
                    can_intervene=False,
                    rng_replay=False,
                    payload_materialization=False,
                    has_streaming=False,
                ),
                capture_backend=bad_capture_backend,
            )
        )


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/registry.py.
def test_replace_backend_spec_removes_stale_aliases() -> None:
    """Replacing a spec removes aliases owned by the old spec."""

    for name in ("fake_alias_probe", "fake_alias_probe_old"):
        tl.backends.unregister(name)
    old_spec = BackendSpec(
        name="fake_alias_probe",
        aliases=("fake_alias_probe_old",),
        can_handle=_fake_can_handle,
        capture_trace=_fake_capture_trace,
        validate_entry=_fake_validate_entry,
        validate_trace=_fake_validate_trace,
        capabilities=BackendCapabilities(
            backward_capture=False,
            has_validation_replay=False,
            fastlog=False,
            can_intervene=False,
            rng_replay=False,
            payload_materialization=False,
            has_streaming=False,
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
        tl.backends.register(old_spec)
        assert tl.backends.spec("fake_alias_probe_old") is old_spec

        tl.backends.register(new_spec, replace=True)

        assert tl.backends.spec("fake_alias_probe") is new_spec
        with pytest.raises(UnknownBackendError):
            tl.backends.spec("fake_alias_probe_old")
    finally:
        tl.backends.unregister("fake_alias_probe")
        tl.backends.unregister("fake_alias_probe_old")


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/default_specs.py.
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


@pytest.mark.rare
@pytest.mark.backend_paddle
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/prepare/door.py, backends/default_specs.py.
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
    assert tl.backends.resolve(None, MlxModule(), (MlxArray(), TFTensor())).name == "torch"
    with pytest.raises(ValueError, match="Unsupported model type"):
        tl.trace(MlxModule(), (MlxArray(), TFTensor()))


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/default_specs.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/default_specs.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/default_specs.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/default_specs.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/default_specs.py.
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
    assert tl.backends.resolve(None, _TinyModel(), torch.ones(1)).name == "torch"
    trace = tl.trace(_TinyModel(), torch.ones(1))
    assert trace.provenance.backend == "torch"


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/default_specs.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_options.py, backends/tf/backend.py.
def test_tf_backend_rejects_seed_without_importing_tensorflow() -> None:
    """TensorFlow preview seed= is a typed unsupported option."""

    with pytest.raises(BackendUnsupportedError, match="does not support: seed"):
        TFBackend().capture_trace(lambda x: x, object(), seed=123)


@pytest.mark.rare
@pytest.mark.backend_paddle
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


@pytest.mark.rare
@pytest.mark.backend_paddle
def test_explicit_paddle_backend_resolves_to_spec() -> None:
    """Explicit Paddle backend and alias resolve to the Paddle spec."""

    paddle = pytest.importorskip("paddle")

    class _PaddleLayer(paddle.nn.Layer):
        """Small Paddle layer for explicit resolution tests."""

        def forward(self, x: Any) -> Any:
            """Return the input unchanged."""

            return x

    x = paddle.to_tensor([1.0])
    spec = tl.backends.spec("paddle")
    assert tl.backends.resolve("paddle", _PaddleLayer(), x) is spec
    assert tl.backends.resolve("paddlepaddle", _PaddleLayer(), x) is spec


@pytest.mark.rare
@pytest.mark.backend_paddle
def test_paddle_preview_unsupported_options_raise_typed_error() -> None:
    """Paddle preview rejects unsupported options with canonical typed errors."""

    paddle = pytest.importorskip("paddle")

    class _PaddleLayer(paddle.nn.Layer):
        """Small Paddle layer for unsupported-option routing tests."""

        def forward(self, x: Any) -> Any:
            """Return the input unchanged."""

            return x

    with pytest.raises(BackendUnsupportedError):
        tl.trace(
            _PaddleLayer(),
            paddle.to_tensor([1.0]),
            backend="paddle",
            capture=tl.options.CaptureOptions(backward_ready=True),
        )


@pytest.mark.rare
@pytest.mark.backend_paddle
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
        save=tl.label_contains("relu"),
    )

    saved_ops = [op for op in trace._layer_list if op.has_saved_activation]
    assert len(saved_ops) == 1
    assert saved_ops[0].func_name == "functional.relu"


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/default_specs.py.
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

    facade_source = inspect.getsource(tl.trace)
    dispatch_source = inspect.getsource(_dispatch_trace)
    assert "_dispatch_trace(" in facade_source
    assert "capture_trace(**public_trace_kwargs)" in dispatch_source
    assert "resolved_spec.name" not in facade_source + dispatch_source


def test_public_backend_literal_branches_stay_in_registry_or_backends() -> None:
    """Public code has no new hard-coded backend literal branches."""

    project_root = Path(__file__).resolve().parents[1]
    allowed_dirs = {
        project_root / "torchlens" / "backends",
    }
    allowed_files = {
        project_root / "torchlens" / "_io" / "bundle.py",
        project_root / "torchlens" / "_io" / "tlspec.py",
        project_root / "torchlens" / "_data_classes" / "_trace_annotations.py",
        project_root / "torchlens" / "_data_classes" / "trace.py",
        project_root / "torchlens" / "repgeom" / "_trace_views.py",
    }
    backend_literals = {"torch", "mlx", "jax", "tinygrad", "paddle", "fake"}
    offenders: list[str] = []

    for source_path in package_files():
        if any(source_path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
            continue
        if source_path in allowed_files:
            continue
        tree = package_ast(source_path)
        source = package_source(source_path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            compared_literals = {
                item.value
                for item in [node.left, *node.comparators]
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            # Same conjunction as always, cheap set filter first: the
            # per-node get_source_segment scan is what pushed the whole
            # sweep to ~18s and out of the smoke-tier budget.
            if not (compared_literals & backend_literals):
                continue
            expression = ast.get_source_segment(source, node) or ""
            if "backend" not in expression:
                continue
            relpath = source_path.relative_to(project_root)
            offenders.append(f"{relpath}:{node.lineno}: {expression}")

    assert offenders == []


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/prepare/dispatch.py, backends/registry.py.
def test_explicit_backend_mismatch_is_deterministic() -> None:
    """Explicit torch selection rejects non-torch models before capture."""

    with pytest.raises(BackendMismatchError, match="backend='torch' cannot handle"):
        tl.trace(object(), torch.ones(1), backend="torch")


# Kept by shrink-r4 (per-id subset rule): sole coverage of validation/_trace_checks.py.
def test_fake_backend_explicit_trace_and_validate() -> None:
    """Registered fake backend drives public trace and validation entries."""

    _register_fake_backend()
    try:
        result = tl.trace(_FakeModel(), object(), backend="fake")
        assert isinstance(result, tl.Trace)
        assert result.provenance.backend == "fake"
        assert result.options.module_identity_mode == "function_root"
        assert result.provenance.param_source == "none"
        assert result.validate_forward_pass([]) is True
        assert tl.validate(_FakeModel(), object(), scope="forward", backend="fake")
    finally:
        tl.backends.unregister("fake")


# Kept by shrink-r4 (per-id subset rule): sole coverage of _io/bundle.py, _io/payload_codec.py, _io/rehydrate.py ....
def test_fake_backend_trace_save_load_accessors_and_invariants(tmp_path: Path) -> None:
    """Fake backend trace round-trips metadata and exposes neutral accessors."""

    _register_fake_backend()
    try:
        trace = tl.trace(_FakeModel(), object(), backend="fake")
        path = tmp_path / "fake.tlspec"

        trace.save(path, level="audit")
        loaded = tl.load(path)

        assert isinstance(loaded, tl.Trace)
        assert loaded.provenance.backend == "fake"
        assert loaded.options.module_identity_mode == "function_root"
        assert loaded.provenance.param_source == "none"
        assert loaded[0].resolver_status == "metadata_only"
        assert loaded[0].backend_address.startswith("fake:")
        assert check_metadata_invariants(loaded) is True
    finally:
        tl.backends.unregister("fake")


# Kept by shrink-r4 (per-id subset rule): sole coverage of validation/_invariants_entry.py.
@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        # R74-1 killers for the three previously-unreferenced arms of
        # _check_backend_identity_invariants: out-of-vocabulary param_source,
        # a backend-capability violation matched on the arm's own message, and
        # an empty Trace.provenance.backend.
        (
            lambda trace: setattr(trace, "_provenance_param_source", "not-a-param-source"),
            "param_source",
        ),
        (
            lambda trace: setattr(
                trace, "_options_module_identity_mode", "unsupported_identity_mode"
            ),
            "is not supported by backend",
        ),
        (lambda trace: setattr(trace, "_autograd_has_backward_pass", True), "has_backward_pass"),
        (lambda trace: trace._grad_fn_logs.__setitem__(1, object()), "_grad_fn_logs"),
        (lambda trace: setattr(trace[0], "resolver_status", "lost"), "resolver_status"),
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
        tl.backends.unregister("fake")


# Kept by shrink-r4 (per-id subset rule): sole coverage of validation/_invariants_entry.py, campaign mutant kills.
@pytest.mark.parametrize("bad_backend", [None])
def test_backend_identity_invariant_rejects_non_string_backend(bad_backend: Any) -> None:
    """The non-empty-string backend arm rejects '' and None directly (R74-1).

    Through the public ``check_metadata_invariants`` entry an empty backend
    already fails EARLIER, at contract selection (``UnknownBackendError``
    from the registry lookup), so this arm is reachable only by direct call.
    Pin the arm itself so a silent disarm (``if False:``) cannot survive.
    """

    from torchlens.validation._invariants_entry import _check_backend_identity_invariants

    class _BadBackendStub:
        _provenance_backend = bad_backend

    with pytest.raises(MetadataInvariantError, match="non-empty string"):
        _check_backend_identity_invariants(_BadBackendStub())


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/prepare/dispatch.py, backends/default_specs.py, backends/registry.py.
def test_backend_none_ambiguity_is_deterministic() -> None:
    """Equal-priority detector collisions fail with a canonical error."""

    _register_fake_backend("fake_a", priority=99)
    _register_fake_backend("fake_b", priority=99)
    try:
        with pytest.raises(AmbiguousBackendError, match="fake_a, fake_b"):
            tl.trace(_FakeModel(), object())
    finally:
        tl.backends.unregister("fake_a")
        tl.backends.unregister("fake_b")


@pytest.mark.parametrize(
    ("flag", "code"),
    [
        ("backward_capture", "backward_capture_unsupported"),
        ("fastlog", "fastlog_unsupported"),
        ("can_intervene", "interventions_unsupported"),
        ("rng_replay", "rng_replay_unsupported"),
        ("has_streaming", "streaming_unsupported"),
        ("structure_only_capture", "structure_only_capture_unsupported"),
        ("has_validation_replay", "validation_replay_unsupported"),
        ("payload_materialization", "payload_materialization_unsupported"),
        ("intermediate_derived_grads", "intermediate_derived_grads_unsupported"),
    ],
)
def test_every_refusable_capability_flag_owes_a_registered_code(flag: str, code: str) -> None:
    """Each ``BackendCapabilities`` flag that can be False maps to a registered refusal code."""

    from torchlens.backends._backend_protocol import REFUSABLE_CAPABILITY_FLAGS, REFUSAL_CODES

    assert REFUSABLE_CAPABILITY_FLAGS[flag] == code
    assert code in REFUSAL_CODES
    exc = BackendUnsupportedError(f"{flag} is not declared by this backend", code=code)
    assert exc.code == code


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_errors.py.
def test_refusal_code_passed_at_the_raise_site_is_the_instance_code() -> None:
    """``exc.code`` is the contract: a raise-site ``code=`` wins; the class code is the default."""

    exc = BackendUnsupportedError(
        "tf backend refuses backward capture", code="backward_capture_unsupported"
    )
    assert exc.code == "backward_capture_unsupported"
    assert exc.fields["code"] == "backward_capture_unsupported"
    assert BackendUnsupportedError.code == "backend_unsupported"

    default = BackendUnsupportedError("tf backend refuses something")
    assert default.code == "backend_unsupported"
    assert default.fields["code"] == "backend_unsupported"

    registry_default = BackendRegistryError("registry failed")
    assert registry_default.code == "backend_error"
    typed = UnknownBackendError("no such backend", code="runtime_missing")
    assert typed.code == "runtime_missing" and typed.fields["code"] == "runtime_missing"
    # The instance code survives the pickle recipe the class defines.
    import pickle

    restored = pickle.loads(pickle.dumps(exc))
    assert restored.code == "backward_capture_unsupported"
    assert restored.fields["code"] == "backward_capture_unsupported"


_GATED_TRACE_OPTION_FLAGS_AT_RECORDS = {
    "intervene": "can_intervene",
    "halt": "can_intervene",
    "stop_after": "can_intervene",
    "storage": "has_streaming",
    "streaming": "has_streaming",
    "save_grads": "backward_capture",
    "backward_ready": "backward_capture",
    "save_rng": "rng_replay",
    "seed": "rng_replay",
    "structure_only": "structure_only_capture",
}
_PUBLIC_OPTION_SPINE_AT_RECORDS = (
    "jax_control_flow",
    "jax_max_control_flow_unroll",
    "module_identity_mode",
    "payload_policy",
    "capture_container_structure",
    "inference_only",
    "chunk_size",
    "chunk_paths",
    "facets",
)


def test_backend_capability_records_are_consistent() -> None:
    """Every capability record is well formed and every option has one owner."""

    from dataclasses import fields

    from torchlens.backends import registry

    bool_flags = {
        field.name for field in fields(BackendCapabilities) if field.type in ("bool", bool)
    }
    seen: dict[str, str] = {}
    for record in registry._BACKEND_CAPABILITIES:
        assert isinstance(record, registry._BackendCapability)
        assert not (record.gated and record.public_spine), record.name
        if record.gated:
            assert record.name in bool_flags, record.name
        else:
            assert record.trace_options, record.name
        for option in record.trace_options:
            assert option not in seen, (option, seen.get(option), record.name)
            seen[option] = record.name
    assert len({record.name for record in registry._BACKEND_CAPABILITIES}) == len(
        registry._BACKEND_CAPABILITIES
    )


def test_capability_tables_derive_from_the_records_in_table_order() -> None:
    """The gate table, the public spine and each backend's options keep their order."""

    from torchlens.backends import registry
    from torchlens.backends._options import TRACE_OPTION_CAPABILITY_GATES

    expected_flags = frozenset(
        {
            "backward_capture",
            "fastlog",
            "can_intervene",
            "rng_replay",
            "has_streaming",
            "structure_only_capture",
        }
    )
    assert expected_flags == registry.GATED_CAPABILITY_FLAGS
    assert list(TRACE_OPTION_CAPABILITY_GATES.items()) == list(
        _GATED_TRACE_OPTION_FLAGS_AT_RECORDS.items()
    )
    assert registry.PUBLIC_OPTION_SPINE_TRACE_OPTIONS == _PUBLIC_OPTION_SPINE_AT_RECORDS
    assert registry.TORCH_TRACE_OPTIONS == _PUBLIC_OPTION_SPINE_AT_RECORDS
    assert registry.JAX_TRACE_OPTIONS == (
        "jax_static_argnums",
        "grad_options",
        "jax_control_flow",
        "jax_max_control_flow_unroll",
        "module_identity_mode",
        "facets",
    )
    for options in (
        registry.MLX_TRACE_OPTIONS,
        registry.TINYGRAD_TRACE_OPTIONS,
        registry.PADDLE_TRACE_OPTIONS,
        registry.TF_TRACE_OPTIONS,
    ):
        assert options == ("module_identity_mode", "grad_options", "facets")


def test_dispatch_filtered_names_are_the_non_gated_record_options() -> None:
    """The dispatcher's strip-or-refuse list is derived from the records, not spelled.

    ``_TRACE_OPTION_FILTERED_NAMES`` equals the options of every non-gated
    capability record in table order (the public spine plus the backend-extra
    keywords); gated options are refused through ``TRACE_OPTION_CAPABILITY_GATES``
    instead, so the two tables partition the recorded options.
    """

    from torchlens._trace_selector_helpers import _TRACE_OPTION_FILTERED_NAMES
    from torchlens.backends import registry
    from torchlens.backends._options import TRACE_OPTION_CAPABILITY_GATES

    non_gated = tuple(
        option
        for record in registry._BACKEND_CAPABILITIES
        if not record.gated
        for option in record.trace_options
    )
    assert non_gated == _TRACE_OPTION_FILTERED_NAMES
    assert (
        *_PUBLIC_OPTION_SPINE_AT_RECORDS,
        "jax_static_argnums",
        "grad_options",
    ) == _TRACE_OPTION_FILTERED_NAMES
    assert not set(_TRACE_OPTION_FILTERED_NAMES) & set(TRACE_OPTION_CAPABILITY_GATES)
    recorded = {
        option for record in registry._BACKEND_CAPABILITIES for option in record.trace_options
    }
    assert set(_TRACE_OPTION_FILTERED_NAMES) | set(TRACE_OPTION_CAPABILITY_GATES) == recorded


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/prepare/dispatch.py.
def test_torch_refuses_every_control_flow_record_option_explicitly() -> None:
    """The torch value refusal reads the control-flow record, not a spelled pair."""

    from torchlens._capture.prepare.dispatch import (
        _reject_unsupported_torch_trace_option_values,
    )
    from torchlens.backends._trace_options import _CONTROL_FLOW
    from torchlens.options import CaptureOptions

    assert _CONTROL_FLOW.trace_options == ("jax_control_flow", "jax_max_control_flow_unroll")
    for option_name, value in (("jax_control_flow", "unroll"), ("jax_max_control_flow_unroll", 2)):
        with pytest.raises(BackendUnsupportedError, match="does not support explicit"):
            _reject_unsupported_torch_trace_option_values(CaptureOptions(**{option_name: value}))
