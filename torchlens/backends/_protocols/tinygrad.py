"""tinygrad preview ``BackendProtocol`` shell (delegating; lazy, realize-once)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol

TINYGRAD_FRAMEWORK = FrameworkRequirement(
    distributions=("tinygrad",),
    minimum="0.13.0",
    below="0.14",
    tested_through="0.13.0",
    pinned="0.13.0",
    extra="tinygrad",
)


class TinygradProtocol(SpecBackedProtocol):
    """The tinygrad preview's protocol object."""

    def __init__(self) -> None:
        from ..registry import TINYGRAD_TRACE_OPTIONS

        super().__init__(
            name="tinygrad",
            maturity="preview",
            framework=TINYGRAD_FRAMEWORK,
            materialization="lazy",
            trace_options=TINYGRAD_TRACE_OPTIONS,
            refusal_codes=(
                "trace_option_unsupported",
                "explicit_seed_unsupported",
                "runtime_missing",
                "runtime_version_unsupported",
                "foreign_tensor_input",
                "constant_unrealizable",
                "rng_state_in_graph",
            ),
        )

    def runtime_available(self) -> str | None:
        """Range check plus a real import of tinygrad."""

        problem = super().runtime_available()
        if problem is not None:
            return problem
        try:
            import tinygrad  # noqa: F401
        except Exception:  # noqa: BLE001
            return "runtime_missing"
        return None

    def is_tensor(self, value: Any) -> bool:
        """Whether ``value`` is a ``tinygrad.Tensor``."""

        try:
            from tinygrad import Tensor
        except Exception:  # noqa: BLE001
            return False
        return isinstance(value, Tensor)

    def realize_once(self, values: Sequence[Any]) -> None:
        """One ``Tensor.realize`` over every value: each listed tensor gets its own buffer."""

        from tinygrad import Tensor

        tensors = [value for value in values if isinstance(value, Tensor)]
        if tensors:
            Tensor.realize(*tensors)


def protocol() -> TinygradProtocol:
    """Factory bound to ``BackendSpec.protocol`` for tinygrad."""

    return TinygradProtocol()
