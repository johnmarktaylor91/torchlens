"""Default public backend specs registered by TorchLens."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, cast

from packaging.version import InvalidVersion, Version
import torch
from torch import nn

from ._protocol import CaptureBackend
from .registry import (
    BackendCapabilities,
    BackendMismatchError,
    BackendSpec,
    BackendUnsupportedError,
    JAX_TRACE_OPTIONS,
    MLX_TRACE_OPTIONS,
    PADDLE_TRACE_OPTIONS,
    SerializationPolicy,
    TF_TRACE_OPTIONS,
    TINYGRAD_TRACE_OPTIONS,
    TORCH_TRACE_OPTIONS,
    register_backend_spec,
)


def _torch_can_handle(
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
) -> bool:
    """Return whether the torch spec can handle the public call.

    Parameters
    ----------
    model:
        Candidate model.
    input_args:
        Positional inputs.
    input_kwargs:
        Keyword inputs.

    Returns
    -------
    bool
        ``True`` for ``torch.nn.Module`` instances.
    """

    del input_args, input_kwargs
    return isinstance(model, nn.Module)


def _mlx_can_handle(
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
) -> bool:
    """Return whether the MLX spec can handle the public call.

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
        ``True`` when MLX is installed, ``model`` is callable, and either the
        model is an ``mlx.nn.Module`` or an input leaf is an MLX array.
    """

    if not callable(model):
        return False
    if _contains_other_backend_tensor("mlx", input_args, input_kwargs):
        return False
    try:
        import mlx.core as mx
        import mlx.nn as mlx_nn
    except ImportError:
        return False
    return isinstance(model, mlx_nn.Module) or _contains_mlx_array(input_args, input_kwargs, mx)


def _contains_mlx_array(input_args: object, input_kwargs: object, mx: object) -> bool:
    """Return whether public inputs contain at least one MLX array leaf.

    Parameters
    ----------
    input_args
        Positional public inputs.
    input_kwargs
        Keyword public inputs.
    mx
        Imported ``mlx.core`` module.

    Returns
    -------
    bool
        True when an MLX array leaf is present.
    """

    array_type = getattr(mx, "array")
    return any(_iter_mlx_input_array_flags(input_args, array_type)) or any(
        _iter_mlx_input_array_flags(input_kwargs, array_type)
    )


def _iter_mlx_input_array_flags(value: object, array_type: type[Any]) -> Iterator[bool]:
    """Yield MLX-array membership flags for nested public inputs.

    Parameters
    ----------
    value
        Candidate nested value.
    array_type
        Runtime MLX array type.

    Yields
    ------
    bool
        True for each MLX array leaf.
    """

    if isinstance(value, array_type):
        yield True
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_mlx_input_array_flags(item, array_type)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_mlx_input_array_flags(item, array_type)


def _jax_can_handle(
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
) -> bool:
    """Return whether the JAX spec can handle the public call.

    Parameters
    ----------
    model:
        Candidate callable.
    input_args:
        Positional inputs.
    input_kwargs:
        Keyword inputs.

    Returns
    -------
    bool
        ``True`` when JAX is installed, ``model`` is callable, and any input
        leaf is a JAX array.
    """

    if not callable(model):
        return False
    if _contains_other_backend_tensor("jax", input_args, input_kwargs):
        return False
    try:
        import jax
    except ImportError:
        return False
    leaves, _treedef = jax.tree.flatten(input_args)
    return any(isinstance(leaf, jax.Array) for leaf in leaves)


def _tinygrad_can_handle(
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
) -> bool:
    """Return whether the tinygrad spec can handle the public call.

    Parameters
    ----------
    model:
        Candidate callable.
    input_args:
        Positional inputs.
    input_kwargs:
        Keyword inputs.

    Returns
    -------
    bool
        ``True`` when tinygrad is installed, ``model`` is callable, and any
        input leaf is a tinygrad tensor.
    """

    if not callable(model):
        return False
    if _contains_other_backend_tensor("tinygrad", input_args, input_kwargs):
        return False
    try:
        from tinygrad import Tensor
    except ImportError:
        return False
    return any(isinstance(leaf, Tensor) for leaf in _simple_leaves(input_args))


def _paddle_can_handle(
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
) -> bool:
    """Return whether the Paddle spec can handle the public call.

    Parameters
    ----------
    model:
        Candidate callable.
    input_args:
        Positional inputs.
    input_kwargs:
        Keyword inputs.

    Returns
    -------
    bool
        ``True`` when Paddle is installed, ``model`` is callable, and either the
        model is a ``paddle.nn.Layer`` or an input leaf is a Paddle tensor.
    """

    if not callable(model) or isinstance(model, nn.Module):
        return False
    if _contains_other_backend_tensor("paddle", input_args, input_kwargs):
        return False
    try:
        import paddle
    except ImportError:
        return False
    return isinstance(model, paddle.nn.Layer) or _contains_paddle_tensor(
        input_args,
        input_kwargs,
        paddle,
    )


def _tf_can_handle(
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
) -> bool:
    """Return whether the TensorFlow spec can handle the public call.

    Parameters
    ----------
    model:
        Candidate callable.
    input_args:
        Positional inputs.
    input_kwargs:
        Keyword inputs.

    Returns
    -------
    bool
        ``True`` when TensorFlow/Keras are installed, Keras is using the
        TensorFlow backend, no foreign tensor leaves are present, and the model
        or inputs identify a TensorFlow capture entry.
    """

    if isinstance(model, nn.Module):
        return False
    if _contains_other_backend_tensor("tf", input_args, input_kwargs):
        return False
    try:
        import keras
        import tensorflow as tf

        if not _tf_runtime_supported(tf, keras):
            if _is_keras_object(model):
                keras_version = getattr(keras, "__version__", None)
                tf_version = getattr(tf, "__version__", None)
                raise BackendMismatchError(
                    "TF backend requires Keras>=3 and TF>=2.16; "
                    f"found keras {keras_version} / tf {tf_version}."
                )
            return False

        active_keras_backend = str(keras.backend.backend())
        if active_keras_backend != "tensorflow":
            if _is_keras_object(model):
                raise BackendMismatchError(
                    "backend='tf' requires keras.backend.backend() == 'tensorflow'; "
                    f"active keras backend is {active_keras_backend!r}."
                )
            return False
        if isinstance(model, tf.Module):
            return True
        if _is_tf_concrete_function(model, tf):
            return True
        if hasattr(model, "get_concrete_function"):
            return True
        if _has_saved_model_signatures(model):
            return True
        return callable(model) and _contains_tf_tensor(input_args, input_kwargs, tf)
    except BackendMismatchError:
        # A genuinely mismatched Keras backend setting (e.g. Keras configured
        # for torch/jax while ``backend='tf'`` was explicitly requested) is
        # real, actionable signal for the caller -- never swallow it.
        raise
    except ImportError:
        # TensorFlow/Keras are simply not installed.
        return False
    except Exception:
        # A broken-but-technically-importable TF/Keras install (numpy/
        # protobuf ABI mismatch, partial C-extension init, etc.) can raise
        # almost anything OTHER than ImportError from the import statements
        # above, or from any keras/tf attribute access used to determine
        # handleability. This function is only a "can this backend handle
        # the input" PROBE used by autorouting -- it must never crash the
        # whole autorouter and take down capture attempts for every OTHER
        # backend (torch, mlx, ...) just because TF happens to be
        # installed-but-broken in the environment. Treat any such failure
        # as "cannot handle" rather than letting it propagate.
        return False


def _contains_paddle_tensor(input_args: object, input_kwargs: object, paddle: object) -> bool:
    """Return whether public inputs contain at least one Paddle tensor leaf.

    Parameters
    ----------
    input_args:
        Positional public inputs.
    input_kwargs:
        Keyword public inputs.
    paddle:
        Imported ``paddle`` module.

    Returns
    -------
    bool
        True when a Paddle tensor leaf is present.
    """

    tensor_type = getattr(paddle, "Tensor")
    return any(isinstance(leaf, tensor_type) for leaf in _simple_leaves(input_args)) or any(
        isinstance(leaf, tensor_type) for leaf in _simple_leaves(input_kwargs)
    )


def _contains_tf_tensor(input_args: object, input_kwargs: object, tf: object) -> bool:
    """Return whether public inputs contain at least one TensorFlow tensor leaf.

    Parameters
    ----------
    input_args:
        Positional public inputs.
    input_kwargs:
        Keyword public inputs.
    tf:
        Imported ``tensorflow`` module.

    Returns
    -------
    bool
        True when a TensorFlow tensor or variable leaf is present.
    """

    tensor_type = getattr(tf, "Tensor")
    variable_type = getattr(tf, "Variable")
    return any(
        isinstance(leaf, tensor_type) or isinstance(leaf, variable_type)
        for leaf in (*_simple_leaves(input_args), *_simple_leaves(input_kwargs))
    )


def _contains_other_backend_tensor(
    backend_name: str,
    input_args: object,
    input_kwargs: object,
) -> bool:
    """Return whether public inputs mix in another backend's tensor family.

    Parameters
    ----------
    backend_name:
        Candidate backend family.
    input_args:
        Positional public inputs.
    input_kwargs:
        Keyword public inputs.

    Returns
    -------
    bool
        True when any tensor leaf belongs to a different backend family.
    """

    return any(
        family is not None and family != backend_name
        for family in (
            _tensor_backend_family(leaf)
            for leaf in (*_simple_leaves(input_args), *_simple_leaves(input_kwargs))
        )
    )


def _tensor_backend_family(leaf: object) -> str | None:
    """Return the backend family for a known tensor leaf.

    Parameters
    ----------
    leaf:
        Candidate public input leaf.

    Returns
    -------
    str | None
        Backend family name when the leaf is from a recognized tensor runtime,
        otherwise ``None``.
    """

    if isinstance(leaf, torch.Tensor):
        return "torch"
    module_name = type(leaf).__module__.split(".", maxsplit=1)[0]
    if module_name in {"jax", "jaxlib"}:
        return "jax"
    if module_name == "tensorflow":
        return "tf"
    if module_name in {"mlx", "paddle", "tinygrad"}:
        return module_name
    return None


def _tf_runtime_supported(tf: object, keras: object) -> bool:
    """Return whether installed TensorFlow/Keras versions are supported.

    Parameters
    ----------
    tf:
        Imported TensorFlow module.
    keras:
        Imported Keras module.

    Returns
    -------
    bool
        True for Keras 3 on TensorFlow >= 2.16.
    """

    try:
        tf_version = Version(str(getattr(tf, "__version__", "0")))
        keras_version = Version(str(getattr(keras, "__version__", "0")))
    except InvalidVersion:
        return False
    return tf_version >= Version("2.16") and keras_version >= Version("3")


def _is_keras_object(value: object) -> bool:
    """Return whether ``value`` appears to come from standalone Keras.

    Parameters
    ----------
    value:
        Candidate object.

    Returns
    -------
    bool
        True when the type module is a Keras module.
    """

    return type(value).__module__.split(".", maxsplit=1)[0] == "keras"


def _is_tf_concrete_function(value: object, tf: object) -> bool:
    """Return whether ``value`` is a TensorFlow ``ConcreteFunction``.

    Parameters
    ----------
    value:
        Candidate object.
    tf:
        Imported ``tensorflow`` module.

    Returns
    -------
    bool
        True for TensorFlow concrete functions.
    """

    tf_runtime = cast(Any, tf)
    concrete_function_type = getattr(tf_runtime.types.experimental, "ConcreteFunction", None)
    return bool(concrete_function_type is not None and isinstance(value, concrete_function_type))


def _has_saved_model_signatures(value: object) -> bool:
    """Return whether ``value`` looks like a loaded SavedModel object.

    Parameters
    ----------
    value:
        Candidate object.

    Returns
    -------
    bool
        True when a non-empty ``signatures`` mapping is present.
    """

    signatures = getattr(value, "signatures", None)
    return isinstance(signatures, Mapping) and bool(signatures)


def _simple_leaves(value: object) -> tuple[object, ...]:
    """Return leaves from simple Python containers.

    Parameters
    ----------
    value:
        Candidate tree.

    Returns
    -------
    tuple[object, ...]
        Flat leaves.
    """

    if isinstance(value, dict):
        return tuple(leaf for child in value.values() for leaf in _simple_leaves(child))
    if isinstance(value, tuple | list):
        return tuple(leaf for child in value for leaf in _simple_leaves(child))
    return (value,)


def _torch_capture_trace(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to the current torch public trace body.

    Parameters
    ----------
    *args, **kwargs:
        Public ``trace`` arguments.

    Returns
    -------
    Any
        Captured trace.
    """

    from ..user_funcs import _trace_torch_model

    return _trace_torch_model(*args, **kwargs)


def _torch_capture_backend() -> CaptureBackend:
    """Return the torch Protocol adapter registered for shared capture orchestration.

    Returns
    -------
    CaptureBackend
        Torch capture backend implementing ``CaptureBackend``.
    """

    from .torch.backend import TorchBackend

    return TorchBackend()


def _mlx_capture_trace(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to the current MLX public trace body.

    Parameters
    ----------
    *args, **kwargs:
        Public ``trace`` arguments.

    Returns
    -------
    Any
        Captured trace.
    """

    from ..user_funcs import _trace_mlx_model_from_public_kwargs

    return _trace_mlx_model_from_public_kwargs(*args, **kwargs)


def _jax_capture_trace(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to the jaxpr-first JAX backend.

    Parameters
    ----------
    *args, **kwargs:
        Public ``trace`` arguments.

    Returns
    -------
    Any
        Captured trace.
    """

    from .jax import JAXBackend

    return JAXBackend().capture_trace(*args, **kwargs)


def _tinygrad_capture_trace(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to the UOp-snapshot tinygrad backend.

    Parameters
    ----------
    *args, **kwargs:
        Public ``trace`` arguments.

    Returns
    -------
    Any
        Captured trace.
    """

    from .tinygrad import TinygradBackend

    return TinygradBackend().capture_trace(*args, **kwargs)


def _paddle_capture_trace(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to the Paddle backend preview.

    Parameters
    ----------
    *args, **kwargs:
        Public ``trace`` arguments.

    Returns
    -------
    Any
        Captured trace.
    """

    from .paddle import PaddleBackend

    return PaddleBackend().capture_trace(*args, **kwargs)


def _tf_capture_trace(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to the TensorFlow backend preview.

    Parameters
    ----------
    *args, **kwargs:
        Public ``trace`` arguments.

    Returns
    -------
    Any
        Captured trace.
    """

    from .tf import TFBackend

    return TFBackend().capture_trace(*args, **kwargs)


def _torch_validate_entry(*args: Any, **kwargs: Any) -> bool:
    """Dispatch to the current torch validation entry.

    Parameters
    ----------
    *args, **kwargs:
        Public validation arguments.

    Returns
    -------
    bool
        Validation result.
    """

    from ..user_funcs import _validate_forward_pass_torch

    return _validate_forward_pass_torch(*args, **kwargs)


def _unsupported_validate_entry(*args: Any, **kwargs: Any) -> bool:
    """Raise a canonical unsupported validation error.

    Parameters
    ----------
    *args, **kwargs:
        Public validation arguments, unused.

    Returns
    -------
    bool
        Never returns.
    """

    del args, kwargs
    raise BackendUnsupportedError("This backend does not support replay validation yet.")


def _jax_validate_entry(*args: Any, **kwargs: Any) -> bool:
    """Dispatch to JAX model/input validation.

    Parameters
    ----------
    *args, **kwargs:
        Public validation arguments.

    Returns
    -------
    bool
        Validation result.
    """

    from .jax import JAXBackend

    return JAXBackend().validate_entry(*args, **kwargs)


def _tinygrad_validate_entry(*args: Any, **kwargs: Any) -> bool:
    """Dispatch to tinygrad model/input validation.

    Parameters
    ----------
    *args, **kwargs:
        Public validation arguments.

    Returns
    -------
    bool
        Validation result.
    """

    from .tinygrad import TinygradBackend

    return TinygradBackend().validate_entry(*args, **kwargs)


def _paddle_validate_entry(*args: Any, **kwargs: Any) -> bool:
    """Dispatch to Paddle model/input validation.

    Parameters
    ----------
    *args, **kwargs:
        Public validation arguments.

    Returns
    -------
    bool
        Validation result.
    """

    from .paddle import PaddleBackend

    return PaddleBackend().validate_entry(*args, **kwargs)


def _tf_validate_entry(*args: Any, **kwargs: Any) -> bool:
    """Dispatch to TensorFlow model/input validation.

    Parameters
    ----------
    *args, **kwargs:
        Public validation arguments.

    Returns
    -------
    bool
        Validation result.
    """

    from .tf import TFBackend

    return TFBackend().validate_entry(*args, **kwargs)


def _torch_validate_trace(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to the current torch trace validation implementation.

    Parameters
    ----------
    *args, **kwargs:
        Trace validation arguments.

    Returns
    -------
    Any
        Validation result or replay status.
    """

    from ..validation.core import validate_saved_outs

    return validate_saved_outs(*args, **kwargs)


def _unsupported_validate_trace(*args: Any, **kwargs: Any) -> bool:
    """Raise a canonical unsupported trace-validation error.

    Parameters
    ----------
    *args, **kwargs:
        Trace validation arguments, unused.

    Returns
    -------
    bool
        Never returns.
    """

    del args, kwargs
    raise BackendUnsupportedError("This backend does not support trace replay validation yet.")


def _jax_validate_trace(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to JAX trace replay validation.

    Parameters
    ----------
    *args, **kwargs:
        Trace validation arguments.

    Returns
    -------
    Any
        Validation result.
    """

    from .jax import JAXBackend

    return JAXBackend().validate_trace(*args, **kwargs)


def _tinygrad_validate_trace(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to tinygrad trace replay validation.

    Parameters
    ----------
    *args, **kwargs:
        Trace validation arguments.

    Returns
    -------
    Any
        Validation result.
    """

    from .tinygrad import TinygradBackend

    return TinygradBackend().validate_trace(*args, **kwargs)


def _paddle_validate_trace(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to Paddle trace replay validation.

    Parameters
    ----------
    *args, **kwargs:
        Trace validation arguments.

    Returns
    -------
    Any
        Validation result.
    """

    from .paddle import PaddleBackend

    return PaddleBackend().validate_trace(*args, **kwargs)


def _tf_validate_trace(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to TensorFlow trace replay validation.

    Parameters
    ----------
    *args, **kwargs:
        Trace validation arguments.

    Returns
    -------
    Any
        Validation result.
    """

    from .tf import TFBackend

    return TFBackend().validate_trace(*args, **kwargs)


def register_default_backend_specs() -> None:
    """Register built-in backend specs.

    Returns
    -------
    None
        The public backend registry is populated.
    """

    register_backend_spec(
        BackendSpec(
            name="torch",
            can_handle=_torch_can_handle,
            capture_trace=_torch_capture_trace,
            validate_entry=_torch_validate_entry,
            validate_trace=_torch_validate_trace,
            capabilities=BackendCapabilities(
                backward_capture=True,
                validation_replay=True,
                fastlog=True,
                interventions=True,
                rng_replay=True,
                payload_materialization=True,
                streaming=True,
                intermediate_derived_grads=False,
                input_container_structure="full_spec",
                output_container_structure="full_spec",
                module_identity_modes=("torch_module",),
                trace_options=TORCH_TRACE_OPTIONS,
            ),
            capture_backend=_torch_capture_backend,
            serialization_policy=SerializationPolicy(
                payload_policy="full",
                body_format="safetensors",
                manifest_schema_versions=(1, 2),
                runtime_name="torch",
            ),
            priority=0,
        ),
        replace=True,
    )
    register_backend_spec(
        BackendSpec(
            name="mlx",
            can_handle=_mlx_can_handle,
            capture_trace=_mlx_capture_trace,
            validate_entry=_unsupported_validate_entry,
            validate_trace=_unsupported_validate_trace,
            capabilities=BackendCapabilities(
                backward_capture=False,
                validation_replay=False,
                fastlog=False,
                interventions=False,
                rng_replay=False,
                payload_materialization=True,
                streaming=False,
                intermediate_derived_grads=True,
                input_container_structure="none",
                output_container_structure="none",
                module_identity_modes=("function_root", "object_module"),
                trace_options=MLX_TRACE_OPTIONS,
            ),
            serialization_policy=SerializationPolicy(
                payload_policy="array_payloads",
                body_format="safetensors",
                manifest_schema_versions=(2,),
                runtime_name="mlx",
            ),
            priority=10,
        ),
        replace=True,
    )
    register_backend_spec(
        BackendSpec(
            name="jax",
            can_handle=_jax_can_handle,
            capture_trace=_jax_capture_trace,
            validate_entry=_jax_validate_entry,
            validate_trace=_jax_validate_trace,
            capabilities=BackendCapabilities(
                backward_capture=False,
                validation_replay=True,
                fastlog=False,
                interventions=False,
                rng_replay=False,
                payload_materialization=True,
                streaming=False,
                intermediate_derived_grads=True,
                input_container_structure="paths_only",
                output_container_structure="paths_only",
                module_identity_modes=("function_root", "pytree_module"),
                trace_options=JAX_TRACE_OPTIONS,
            ),
            serialization_policy=SerializationPolicy(
                payload_policy="array_payloads",
                body_format="safetensors",
                manifest_schema_versions=(2,),
                runtime_name="jax",
            ),
            priority=20,
        ),
        replace=True,
    )
    register_backend_spec(
        BackendSpec(
            name="tinygrad",
            can_handle=_tinygrad_can_handle,
            capture_trace=_tinygrad_capture_trace,
            validate_entry=_tinygrad_validate_entry,
            validate_trace=_tinygrad_validate_trace,
            capabilities=BackendCapabilities(
                backward_capture=False,
                validation_replay=True,
                fastlog=False,
                interventions=False,
                rng_replay=False,
                payload_materialization=True,
                streaming=False,
                intermediate_derived_grads=True,
                input_container_structure="paths_only",
                output_container_structure="paths_only",
                module_identity_modes=("function_root", "object_module"),
                trace_options=TINYGRAD_TRACE_OPTIONS,
            ),
            serialization_policy=SerializationPolicy(
                payload_policy="array_payloads",
                body_format="safetensors",
                manifest_schema_versions=(2,),
                runtime_name="tinygrad",
            ),
            priority=30,
        ),
        replace=True,
    )
    register_backend_spec(
        BackendSpec(
            name="paddle",
            aliases=("paddlepaddle",),
            can_handle=_paddle_can_handle,
            capture_trace=_paddle_capture_trace,
            validate_entry=_paddle_validate_entry,
            validate_trace=_paddle_validate_trace,
            capabilities=BackendCapabilities(
                backward_capture=False,
                validation_replay=True,
                fastlog=False,
                interventions=False,
                rng_replay=False,
                payload_materialization=True,
                streaming=False,
                intermediate_derived_grads=True,
                input_container_structure="paths_only",
                output_container_structure="paths_only",
                module_identity_modes=("function_root", "object_module"),
                trace_options=PADDLE_TRACE_OPTIONS,
            ),
            serialization_policy=SerializationPolicy(
                payload_policy="array_payloads",
                body_format="safetensors",
                manifest_schema_versions=(2,),
                runtime_name="paddle",
            ),
            priority=40,
            coercible=False,
        ),
        replace=True,
    )
    register_backend_spec(
        BackendSpec(
            name="tf",
            aliases=("tensorflow",),
            can_handle=_tf_can_handle,
            capture_trace=_tf_capture_trace,
            validate_entry=_tf_validate_entry,
            validate_trace=_tf_validate_trace,
            capabilities=BackendCapabilities(
                backward_capture=False,
                validation_replay=True,
                fastlog=False,
                interventions=False,
                rng_replay=False,
                payload_materialization=True,
                streaming=False,
                intermediate_derived_grads=False,
                input_container_structure="paths_only",
                output_container_structure="paths_only",
                module_identity_modes=("function_root", "object_module"),
                trace_options=TF_TRACE_OPTIONS,
            ),
            serialization_policy=SerializationPolicy(
                payload_policy="array_payloads",
                body_format="safetensors",
                manifest_schema_versions=(2,),
                runtime_name="tf",
            ),
            priority=50,
            coercible=False,
        ),
        replace=True,
    )


register_default_backend_specs()
