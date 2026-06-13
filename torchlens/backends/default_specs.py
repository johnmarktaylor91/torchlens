"""Default public backend specs registered by TorchLens."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .registry import (
    BackendCapabilities,
    BackendSpec,
    BackendUnsupportedError,
    SerializationPolicy,
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
        Positional inputs, unused.
    input_kwargs:
        Keyword inputs, unused.

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
        ``True`` when MLX is installed and ``model`` is an ``mlx.nn.Module``.
    """

    del input_args, input_kwargs
    try:
        import mlx.nn as mlx_nn
    except ImportError:
        return False
    return isinstance(model, mlx_nn.Module)


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
                module_identity_modes=("torch_module",),
            ),
            serialization_policy=SerializationPolicy(
                payload_policy="full", body_format="safetensors"
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
                payload_materialization=False,
                streaming=False,
                module_identity_modes=("torch_module",),
            ),
            serialization_policy=SerializationPolicy(
                payload_policy="audit_only",
                body_format="audit_only",
            ),
            priority=10,
        ),
        replace=True,
    )


register_default_backend_specs()
