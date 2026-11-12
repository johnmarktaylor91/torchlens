"""Paddle ``BackendProtocol`` shell (delegating; eager dygraph)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol
from .._trace_options import PADDLE_TRACE_OPTIONS

# Refusal codes the Paddle backend raises, each spelled once. Every
# ``BackendUnsupportedError`` raised inside ``torchlens.backends.paddle`` passes
# one of these as ``code=``. They live in this import-light shell (not the
# Paddle package) because registration validates ``refusal_codes()`` while the
# package cannot be imported yet.

#: An option (or option combination) the Paddle backend does not implement.
PADDLE_REFUSAL_OPTION = "trace_option_unsupported"
#: ``random_seed=`` on the Paddle backend.
PADDLE_REFUSAL_SEED = "explicit_seed_unsupported"
#: A ``module_identity_mode`` outside the declared tuple, or one the root cannot honour.
PADDLE_REFUSAL_IDENTITY_MODE = "module_identity_mode_unsupported"
#: More intermediate derived-gradient records than ``max_intermediate_grads`` allows.
PADDLE_REFUSAL_INTERMEDIATE_GRADS = "intermediate_derived_grads_unsupported"
#: An intervention or halt predicate returned something other than its contract.
PADDLE_REFUSAL_PREDICATE = "predicate_return_invalid"
#: A backward-direction intervention (true backward capture is not declared).
PADDLE_REFUSAL_BACKWARD = "backward_capture_unsupported"
#: A torch tensor reached a Paddle intervention helper.
PADDLE_REFUSAL_FOREIGN = "foreign_tensor_input"
#: A tensor-derived Python scalar or control-flow escape inside the forward.
PADDLE_REFUSAL_SCALAR_ESCAPE = "value_dependent_branch_unsupported"
#: A stochastic Paddle op inside the forward (its draw cannot be replayed).
PADDLE_REFUSAL_STOCHASTIC = "rng_replay_unsupported"
#: An in-place or global-state Paddle op inside the forward.
PADDLE_REFUSAL_INPLACE = "inplace_op_unsupported"
#: Paddle runs outside dygraph mode, or in PIR mode.
PADDLE_REFUSAL_RUNTIME_MODE = "runtime_mode_unsupported"

#: Codes raised inside the Paddle package and registered in ``REFUSAL_CODES``.
PADDLE_RAISED_CODES: frozenset[str] = frozenset(
    {
        PADDLE_REFUSAL_OPTION,
        PADDLE_REFUSAL_SEED,
        PADDLE_REFUSAL_IDENTITY_MODE,
        PADDLE_REFUSAL_INTERMEDIATE_GRADS,
        PADDLE_REFUSAL_PREDICATE,
        PADDLE_REFUSAL_BACKWARD,
        PADDLE_REFUSAL_FOREIGN,
        PADDLE_REFUSAL_SCALAR_ESCAPE,
        PADDLE_REFUSAL_STOCHASTIC,
        PADDLE_REFUSAL_INPLACE,
        PADDLE_REFUSAL_RUNTIME_MODE,
    }
)


def paddle_package_refusal_codes() -> frozenset[str]:
    """Registered codes raised by Paddle-package code.

    Returns
    -------
    frozenset[str]
        :data:`PADDLE_RAISED_CODES`.
    """

    return PADDLE_RAISED_CODES


PADDLE_FRAMEWORK = FrameworkRequirement(
    distributions=("paddlepaddle",),
    minimum="3.3",
    below="3.4",
    tested_through="3.3.1",
    pinned="3.3.1",
    extra="paddle",
)


class PaddleProtocol(SpecBackedProtocol):
    """The Paddle backend's protocol object."""

    def __init__(self) -> None:
        super().__init__(
            name="paddle",
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

    def host_copy(self, value: Any) -> Any:
        """Detached NumPy copy of a Paddle tensor; 0-d stays 0-d."""

        import numpy as np

        array = np.array(value.numpy(), copy=True)
        expected = tuple(int(dim) for dim in value.shape)
        return array.reshape(expected) if array.shape != expected else array

    def validate(self, trace: Any, ground_truth: Any) -> Any:
        """Replay-validate ``trace`` and return the status object, never a bool.

        The spec entry keeps its public ``bool`` return for ``tl.validate``;
        the status it settles on the trace carries the per-op decisions and,
        when failed, names the first offending op and its reason.
        """

        self._spec().validate_trace(trace, ground_truth)
        return trace.validation_replay_status

    def refusal_codes(self) -> frozenset[str]:
        """Shell codes plus every registered code raised inside the Paddle package."""

        return super().refusal_codes() | paddle_package_refusal_codes()


def protocol() -> PaddleProtocol:
    """Factory bound to ``BackendSpec.protocol`` for paddle."""

    return PaddleProtocol()
