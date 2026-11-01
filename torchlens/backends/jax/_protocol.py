"""JAX preview ``BackendProtocol`` shell (delegating; jaxpr-first, eager values)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol

JAX_FRAMEWORK = FrameworkRequirement(
    distributions=("jax",),
    minimum="0.4.30",
    below=None,
    tested_through="0.4.30",
    pinned=None,
    extra="jax",
)


class JAXProtocol(SpecBackedProtocol):
    """The JAX preview's protocol object."""

    def __init__(self) -> None:
        from ..registry import JAX_TRACE_OPTIONS

        super().__init__(
            name="jax",
            maturity="preview",
            framework=JAX_FRAMEWORK,
            materialization="eager",
            trace_options=JAX_TRACE_OPTIONS,
            refusal_codes=(
                "trace_option_unsupported",
                "explicit_seed_unsupported",
                "runtime_missing",
                "runtime_version_unsupported",
                "foreign_tensor_input",
                "constant_unrealizable",
            ),
        )

    def runtime_available(self) -> str | None:
        """Range check plus a real import of jax and jax.numpy."""

        problem = super().runtime_available()
        if problem is not None:
            return problem
        try:
            import jax  # noqa: F401
            import jax.numpy  # noqa: F401
        except Exception:  # noqa: BLE001
            return "runtime_missing"
        return None

    def is_tensor(self, value: Any) -> bool:
        """Whether ``value`` is a ``jax.Array``."""

        try:
            import jax
        except Exception:  # noqa: BLE001
            return False
        return isinstance(value, jax.Array)

    def realize_once(self, values: Sequence[Any]) -> None:
        """Values are concrete on the jaxpr path; nothing to evaluate."""

        return None


def protocol() -> JAXProtocol:
    """Factory bound to ``BackendSpec.protocol`` for jax."""

    return JAXProtocol()
