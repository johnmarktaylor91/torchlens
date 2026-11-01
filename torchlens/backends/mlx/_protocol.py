"""MLX preview ``BackendProtocol`` shell (delegating; lazy, realize-once)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol

MLX_FRAMEWORK = FrameworkRequirement(
    distributions=("mlx",),
    minimum="0.26",
    below="0.27",
    tested_through="0.26.3",
    pinned="0.26.3",
    extra="mlx",
    excluded=("0.26.5",),
)


class MLXProtocol(SpecBackedProtocol):
    """The MLX preview's protocol object."""

    def __init__(self) -> None:
        from ..registry import MLX_TRACE_OPTIONS

        super().__init__(
            name="mlx",
            maturity="preview",
            framework=MLX_FRAMEWORK,
            materialization="lazy",
            trace_options=MLX_TRACE_OPTIONS,
            refusal_codes=(
                "trace_option_unsupported",
                "explicit_seed_unsupported",
                "runtime_missing",
                "runtime_version_unsupported",
                "foreign_tensor_input",
                "unwrapped_op_in_capture",
            ),
        )

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
