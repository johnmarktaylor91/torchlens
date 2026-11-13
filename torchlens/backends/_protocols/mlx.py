"""MLX ``BackendProtocol`` shell (delegating; lazy, realize-once)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol
from .._errors import BackendUnsupportedError
from .._trace_options import MLX_TRACE_OPTIONS
from .mlx_wrapped_state import mlx_wrappers_installed

#: Every refusal code the MLX backend raises today (subset of ``REFUSAL_CODES``).
MLX_REFUSAL_CODES: tuple[str, ...] = (
    "trace_option_unsupported",
    "explicit_seed_unsupported",
    "interventions_unsupported",
    "backward_capture_unsupported",
    "runtime_missing",
    "runtime_version_unsupported",
    "foreign_tensor_input",
    "unwrapped_op_in_capture",
    "constant_unrealizable",
    "traced_transform_entry_unsupported",
    "intervention_with_derived_grads_unsupported",
    "intervention_result_invalid",
)

MLX_FRAMEWORK = FrameworkRequirement(
    distributions=("mlx",),
    minimum="0.26",
    below="0.27",
    tested_through="0.26.3",
    pinned="0.26.3",
    extra="mlx",
    excluded=("0.26.5",),
)

#: Test-only dependencies of the MLX ecosystem roster rows: installed by the
#: nightly backend leg only, never by a pyproject extra or the per-change leg.
#: The range follows the ``mlx`` pin: mlx-lm 0.26.x needs ``mlx>=0.26`` and
#: 0.27 needs ``mlx>=0.29``. The pins test holds the nightly install to it.
MLX_ECOSYSTEM_TEST_REQUIREMENTS: tuple[str, ...] = ("mlx-lm>=0.26,<0.27",)


class MLXProtocol(SpecBackedProtocol):
    """The MLX backend's protocol object."""

    def __init__(self) -> None:
        super().__init__(
            name="mlx",
            framework=MLX_FRAMEWORK,
            materialization="lazy",
            trace_options=MLX_TRACE_OPTIONS,
            refusal_codes=MLX_REFUSAL_CODES,
        )

    def capture(self, model: Any, args: Any, kwargs: dict[str, Any] | None, **options: Any) -> Any:
        """Capture through the public ``tl.trace`` door with ``backend="mlx"``.

        The spec's ``capture_trace`` callable takes the public keyword bundle
        only, so the protocol routes through ``tl.trace`` like a caller does.
        """

        from ..._user_funcs import trace  # deferred: cycle _user_funcs

        return trace(model, args, input_kwargs=kwargs, backend="mlx", **options)

    def reference_output(self, model: Any, args: Any, kwargs: dict[str, Any] | None) -> Any:
        """The model's own forward with no TorchLens wrapper installed.

        Raises
        ------
        RuntimeError
            If the process-global MLX wrappers are installed (a capture is in
            flight), since the result would not be an independent oracle arm.
        """

        if mlx_wrappers_installed():
            raise RuntimeError(
                "MLX reference_output called while capture wrappers are installed; "
                "the oracle's reference arm must run unwrapped."
            )
        return super().reference_output(model, args, kwargs)

    def host_copy(self, value: Any) -> Any:
        """A NumPy copy of a realized ``mx.array``, shape and dtype kept (0-d stays 0-d).

        Raises
        ------
        BackendUnsupportedError
            Code ``constant_unrealizable`` when the value has no readable buffer.
        """

        import numpy as np

        if not (self.is_tensor(value) or isinstance(value, (np.ndarray, np.generic))):
            raise BackendUnsupportedError(
                f"MLX host_copy needs an mx.array, got {type(value).__name__}: no readable buffer",
                code="constant_unrealizable",
            )
        try:
            return np.array(value, copy=True)
        except Exception as exc:  # noqa: BLE001 - any read failure is the typed refusal
            raise BackendUnsupportedError(
                f"MLX value of type {type(value).__name__} has no readable host buffer: {exc}",
                code="constant_unrealizable",
            ) from exc

    def runtime_available(self) -> str | None:
        """Range check plus a real import of mlx.core and mlx.nn."""

        problem = super().runtime_available()
        if problem is not None:
            return problem
        try:
            import mlx.core  # noqa: F401
            import mlx.nn  # noqa: F401
        except Exception:  # noqa: BLE001
            return "runtime_missing"
        return None

    def is_tensor(self, value: Any) -> bool:
        """Whether ``value`` is an ``mlx.core.array``."""

        try:
            import mlx.core as mx
        except Exception:  # noqa: BLE001
            return False
        return isinstance(value, mx.array)

    def realize_once(self, values: Sequence[Any]) -> None:
        """One ``mx.eval`` over every value (the realize-once rule)."""

        import mlx.core as mx

        arrays = [value for value in values if isinstance(value, mx.array)]
        if arrays:
            mx.eval(*arrays)


def protocol() -> MLXProtocol:
    """Factory bound to ``BackendSpec.protocol`` for mlx."""

    return MLXProtocol()
