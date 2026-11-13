"""tinygrad ``BackendProtocol`` shell (delegating; lazy, realize-once).

The shell delegates capture and validation to the registered spec and adds
the framework-specific members: ``is_tensor``, ``host_copy`` (a NumPy copy
that refuses ``constant_unrealizable`` when tinygrad cannot read the buffer),
``realize_once`` (one ``Tensor.realize`` over every value) and ``validate``,
which always returns the replay status object the backend stored, never the
public entry's boolean.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol
from .._errors import BackendUnsupportedError
from .._trace_options import TINYGRAD_TRACE_OPTIONS

TINYGRAD_FRAMEWORK = FrameworkRequirement(
    distributions=("tinygrad",),
    minimum="0.13.0",
    below="0.14",
    tested_through="0.13.0",
    pinned="0.13.0",
    extra="tinygrad",
)


class TinygradProtocol(SpecBackedProtocol):
    """The tinygrad backend's protocol object."""

    def __init__(self) -> None:
        super().__init__(
            name="tinygrad",
            framework=TINYGRAD_FRAMEWORK,
            materialization="lazy",
            trace_options=TINYGRAD_TRACE_OPTIONS,
            refusal_codes=(
                "trace_option_unsupported",
                "module_identity_mode_unsupported",
                "explicit_seed_unsupported",
                "runtime_missing",
                "runtime_version_unsupported",
                "foreign_tensor_input",
                "constant_unrealizable",
                "rng_state_in_graph",
                "input_mutation_unsupported",
                "tensor_output_required",
                "tinygrad_input_required",
                "live_payloads_required",
                "mid_capture_execution_unsupported",
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

    def host_copy(self, value: Any) -> Any:
        """A NumPy copy of a tinygrad value, shape and dtype kept (0-d stays 0-d).

        Raises
        ------
        BackendUnsupportedError
            Coded ``constant_unrealizable`` when the value has no readable buffer.
        """

        import numpy as np

        if not self.is_tensor(value):
            return np.array(value, copy=True)
        try:
            return np.array(value.numpy(), copy=True)
        except NotImplementedError as error:
            from ..tinygrad._rules import CONSTANT_UNREALIZABLE_CODE, refuse_constant_unrealizable

            raise refuse_constant_unrealizable(
                value.uop, error, code=CONSTANT_UNREALIZABLE_CODE
            ) from error

    def validate(self, trace: Any, ground_truth: Any) -> Any:
        """Replay validation returning the stored status object, never a bool.

        The spec's ``validate_trace`` keeps the public boolean contract of
        ``Trace.validate_forward_pass``; the status it stores on the trace
        carries the per-op decisions (first offending op, reason) the suite reads.
        """

        result = super().validate(trace, ground_truth)
        if not isinstance(result, bool):
            return result
        # Direct read of the stamped status: the attribute is absent when the
        # backend never stored one, and absence is the typed refusal below.
        try:
            status = trace._validation_replay_status
        except AttributeError:
            status = None
        if status is None:
            raise BackendUnsupportedError(
                "tinygrad validation produced a boolean without storing its replay status",
                code="validation_replay_unsupported",
            )
        return status

    def realize_once(self, values: Sequence[Any]) -> None:
        """One ``Tensor.realize`` over every value: each listed tensor gets its own buffer."""

        from tinygrad import Tensor

        tensors = [value for value in values if isinstance(value, Tensor)]
        if tensors:
            Tensor.realize(*tensors)


def protocol() -> TinygradProtocol:
    """Factory bound to ``BackendSpec.protocol`` for tinygrad."""

    return TinygradProtocol()
