"""Paddle preview ``BackendProtocol`` shell (delegating; eager dygraph)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol

PADDLE_FRAMEWORK = FrameworkRequirement(
    distributions=("paddlepaddle",),
    minimum="3.3",
    below="3.4",
    tested_through="3.3",
    pinned=None,
    extra="paddle",
)


class PaddleProtocol(SpecBackedProtocol):
    """The Paddle preview's protocol object."""

    def __init__(self) -> None:
        from ..registry import PADDLE_TRACE_OPTIONS

        super().__init__(
            name="paddle",
            maturity="preview",
            framework=PADDLE_FRAMEWORK,
            materialization="eager",
            trace_options=PADDLE_TRACE_OPTIONS,
            refusal_codes=(
                "trace_option_unsupported",
                "explicit_seed_unsupported",
                "runtime_missing",
                "runtime_version_unsupported",
                "foreign_tensor_input",
            ),
        )

    def runtime_available(self) -> str | None:
        """Range check plus a real import of paddle."""

        problem = super().runtime_available()
        if problem is not None:
            return problem
        try:
            import paddle  # noqa: F401
        except Exception:  # noqa: BLE001
            return "runtime_missing"
        return None

    def is_tensor(self, value: Any) -> bool:
        """Whether ``value`` is a ``paddle.Tensor``."""

        try:
            import paddle
        except Exception:  # noqa: BLE001
            return False
        return isinstance(value, paddle.Tensor)

    def realize_once(self, values: Sequence[Any]) -> None:
        """Eager framework: no evaluation step."""

        return None


def protocol() -> PaddleProtocol:
    """Factory bound to ``BackendSpec.protocol`` for paddle."""

    return PaddleProtocol()
