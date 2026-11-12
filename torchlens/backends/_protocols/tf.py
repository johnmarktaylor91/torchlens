"""TensorFlow ``BackendProtocol`` (delegating; eager)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol
from .._errors import BackendUnsupportedError
from .._trace_options import TF_TRACE_OPTIONS

#: The TensorFlow requirement, stated once (SPEC 1.2; the ceiling is F-M's
#: ``tested_through`` arm). ``pinned`` is the version the per-change CI leg
#: installs through ``tensorflow-cpu``; the pins test keeps the three in step.
TF_FRAMEWORK = FrameworkRequirement(
    distributions=("tensorflow", "tensorflow-cpu"),
    minimum="2.16",
    below=None,
    tested_through="2.21.0",
    pinned="2.21.0",
    extra="tf",
)

#: Test-only dependencies of the TensorFlow ecosystem roster rows: installed by
#: the nightly backend leg only, never by a pyproject extra or the per-change
#: leg. The pins test holds the nightly install to it.
TF_ECOSYSTEM_TEST_REQUIREMENTS: tuple[str, ...] = ("keras-hub>=0.21,<1",)

#: Refusal codes the TensorFlow backend raises today (a subset of the registry).
TF_REFUSAL_CODES: tuple[str, ...] = (
    "trace_option_unsupported",
    "explicit_seed_unsupported",
    "runtime_missing",
    "runtime_version_unsupported",
    "foreign_tensor_input",
    "backward_capture_unsupported",
    "fastlog_unsupported",
    "rng_replay_unsupported",
    "streaming_unsupported",
    "structure_only_capture_unsupported",
    "constant_unrealizable",
)


class TFProtocol(SpecBackedProtocol):
    """The TensorFlow backend's protocol object."""

    def __init__(self) -> None:
        super().__init__(
            name="tf",
            framework=TF_FRAMEWORK,
            materialization="eager",
            trace_options=TF_TRACE_OPTIONS,
            refusal_codes=TF_REFUSAL_CODES,
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

    def validate(self, trace: Any, ground_truth: Any) -> Any:
        """Replay-validate a TensorFlow trace; always the status object, never a bool.

        Parameters
        ----------
        trace:
            A trace captured by this backend.
        ground_truth:
            The reference outputs (accepted for the protocol signature; the
            TensorFlow oracle replays saved payloads against each other).

        Returns
        -------
        ValidationStatus
            ``passed``, ``failed`` (naming the first offending op in
            ``message`` and every op in ``decisions``), ``unverified`` or
            ``unavailable`` (loaded traces whose runtime captures were
            stripped).
        """

        del ground_truth
        from ..tf.validation import validate_tf_trace_status

        return validate_tf_trace_status(trace)

    def is_tensor(self, value: Any) -> bool:
        """Whether ``value`` is a TensorFlow tensor or variable."""

        try:
            import tensorflow as tf
        except Exception:  # noqa: BLE001
            return False
        return isinstance(value, (tf.Tensor, tf.Variable))

    def host_copy(self, value: Any) -> Any:
        """Detached NumPy copy of a tensor or variable; shape and dtype kept (0-d stays 0-d)."""

        import numpy as np

        to_numpy = getattr(value, "numpy", None)
        try:
            array = np.array(to_numpy() if callable(to_numpy) else value, copy=True)
        except Exception as exc:  # noqa: BLE001 - a resource or variant has no host buffer
            raise BackendUnsupportedError(
                f"tf value of type {type(value).__name__} has no readable host buffer ({exc}).",
                code="constant_unrealizable",
                remedy="pass dense numeric tensors; resource and variant values have no payload",
            ) from exc
        return array

    def realize_once(self, values: Sequence[Any]) -> None:
        """Eager framework: no evaluation step."""

        return None


def protocol() -> TFProtocol:
    """Factory bound to ``BackendSpec.protocol`` for tf."""

    return TFProtocol()
