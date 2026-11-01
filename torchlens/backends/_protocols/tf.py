"""TensorFlow preview ``BackendProtocol`` shell (delegating; eager)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol

TF_FRAMEWORK = FrameworkRequirement(
    distributions=("tensorflow", "tensorflow-cpu"),
    minimum="2.16",
    below=None,
    tested_through="2.21.0",
    pinned=None,
    extra="tf",
)


class TFProtocol(SpecBackedProtocol):
    """The TensorFlow preview's protocol object."""

    def __init__(self) -> None:
        from ..registry import TF_TRACE_OPTIONS

        super().__init__(
            name="tf",
            maturity="preview",
            framework=TF_FRAMEWORK,
            materialization="eager",
            trace_options=TF_TRACE_OPTIONS,
            refusal_codes=(
                "trace_option_unsupported",
                "explicit_seed_unsupported",
                "runtime_missing",
                "runtime_version_unsupported",
                "foreign_tensor_input",
            ),
        )

    def runtime_available(self) -> str | None:
        """Range check plus a real import of TensorFlow and Keras 3."""

        problem = super().runtime_available()
        if problem is not None:
            return problem
        try:
            import keras  # noqa: F401
            import tensorflow  # noqa: F401
        except Exception:  # noqa: BLE001 - a broken install is a missing runtime
            return "runtime_missing"
        return None

    def is_tensor(self, value: Any) -> bool:
        """Whether ``value`` is a TensorFlow tensor or variable."""

        try:
            import tensorflow as tf
        except Exception:  # noqa: BLE001
            return False
        return isinstance(value, (tf.Tensor, tf.Variable))

    def realize_once(self, values: Sequence[Any]) -> None:
        """Eager framework: no evaluation step."""

        return None


def protocol() -> TFProtocol:
    """Factory bound to ``BackendSpec.protocol`` for tf."""

    return TFProtocol()
