"""Default public backend specs registered by TorchLens."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from ._protocol import CaptureBackend
from .registry import (
    BackendCapabilities,
    BackendSpec,
    BackendUnsupportedError,
    JAX_TRACE_OPTIONS,
    MLX_TRACE_OPTIONS,
    SerializationPolicy,
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

    del input_kwargs
    if not callable(model):
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

    del input_kwargs
    if not callable(model):
        return False
    try:
        from tinygrad import Tensor
    except ImportError:
        return False
    return any(isinstance(leaf, Tensor) for leaf in _simple_leaves(input_args))


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


def _torch_validate_trace(*args: Any, **kwargs: Any) -> bool:
    """Dispatch to the current torch trace validation implementation.

    Parameters
    ----------
    *args, **kwargs:
        Trace validation arguments.

    Returns
    -------
    bool
        Validation result.
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


def register_default_backend_specs() -> None:
    """Register built-in torch and MLX backend specs.

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


register_default_backend_specs()
