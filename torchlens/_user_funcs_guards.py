"""Entry-time refusals shared by the public ``trace()`` door."""

from __future__ import annotations

from typing import Any

from torch import nn

from torchlens.backends.registry import BackendUnsupportedError

from ._deprecations import MISSING
from ._errors import CaptureNotActiveError, InvalidArgumentError
from ._utils._torch_compat import is_dynamo_compiled_callable
from ._utils.env_flags import closed_bool_env

#: Filtered trace options that are a capability flag's entry (SPEC 1.3: one code per
#: cause): refusing one raises that flag's contracted code. Every other filtered option
#: refuses with ``trace_option_unsupported``.
_TRACE_OPTION_CAPABILITY_FLAGS: dict[str, str] = {
    "grad_options": "intermediate_derived_grads",
}


def _trace_option_refusal_code(option_name: str) -> str:
    """The refusal code for an explicit trace option the resolved backend lacks."""

    flag = _TRACE_OPTION_CAPABILITY_FLAGS.get(option_name)
    if flag is None:
        return "trace_option_unsupported"
    from .backends._backend_protocol import refusal_code_for_capability

    return refusal_code_for_capability(flag)


def _reject_compiled_plain_callable(model: Any) -> None:
    """Refuse a torch.compile-produced plain callable that is not an nn.Module."""
    if not isinstance(model, nn.Module) and is_dynamo_compiled_callable(model):
        raise InvalidArgumentError(
            "TorchLens cannot capture a torch.compile-produced plain callable because it is "
            "not an nn.Module",
            code="compiled_callable_unsupported",
            remedy=(
                "pass the original eager nn.Module instead of the compiled callable; to "
                "inspect what torch.compile generated, use tl.bridge.depyf.dump(fn, x, path)"
            ),
            argument="model",
        )


def _reject_unrouted_jax_options(backend: Any, jax_static_argnums: Any, grad_options: Any) -> None:
    """Refuse jax_static_argnums / grad_options when no backend was named."""
    if backend is None and (jax_static_argnums is not MISSING or grad_options is not MISSING):
        refused_option = (
            "jax_static_argnums" if jax_static_argnums is not MISSING else ("grad_options")
        )
        raise BackendUnsupportedError(
            "jax_static_argnums is only supported with backend='jax'; grad_options is "
            "only supported with backend='jax', backend='mlx', backend='tinygrad', "
            "backend='paddle', or backend='tf'.",
            code=_trace_option_refusal_code(refused_option),
            option=refused_option,
        )


def _reject_auto_environment() -> None:
    """Refuse the unsupported implicit capture mode TORCHLENS_AUTO requests."""
    if closed_bool_env("TORCHLENS_AUTO"):
        raise CaptureNotActiveError(
            "TORCHLENS_AUTO requested an unsupported implicit capture mode",
            code="auto_environment_unsupported",
            remedy="unset TORCHLENS_AUTO and call auto_capture() explicitly",
            environment_variable="TORCHLENS_AUTO",
        )


def _refuse_non_torch_episode(public_trace_kwargs: dict[str, Any], resolved_spec: Any) -> None:
    """episode= capture (capture_kind=episode) is torch-only in this release."""

    if str(resolved_spec.name) != "torch":
        episode_value = public_trace_kwargs.pop("episode", None)
        if episode_value is not None:
            raise BackendUnsupportedError(
                "episode= capture (capture_kind=episode) is torch-only in this "
                f"release; backend {str(resolved_spec.name)!r} does not support "
                "episode declarations."
            )
