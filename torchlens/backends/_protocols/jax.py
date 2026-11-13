"""JAX ``BackendProtocol`` (delegating; jaxpr-first, eager values)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol
from .._errors import BackendUnsupportedError
from .._trace_options import JAX_TRACE_OPTIONS

#: The JAX requirement, stated once (SPEC 1.2; F-M ``tested_through`` arm).
#: The ``jax`` extra is unpinned, so the pin is what the CI leg resolved when
#: the JAX suite last ran green; the pins test keeps the three in step.
JAX_FRAMEWORK = FrameworkRequirement(
    distributions=("jax",),
    minimum="0.4.30",
    below=None,
    tested_through="0.10.2",
    pinned="0.10.2",
    extra="jax",
)

#: Refusal codes the JAX backend raises today (a subset of the registry).
JAX_REFUSAL_CODES: tuple[str, ...] = (
    "trace_option_unsupported",
    "explicit_seed_unsupported",
    "runtime_missing",
    "runtime_version_unsupported",
    "foreign_tensor_input",
    "constant_unrealizable",
    "backward_capture_unsupported",
    "fastlog_unsupported",
    "interventions_unsupported",
    "intervention_result_invalid",
    "intervention_with_derived_grads_unsupported",
    "predicate_return_invalid",
    "rng_replay_unsupported",
    "streaming_unsupported",
    "structure_only_capture_unsupported",
    "followed_by_unsupported",
    "value_dependent_branch_unsupported",
)


class JAXProtocol(SpecBackedProtocol):
    """The JAX backend's protocol object."""

    def __init__(self) -> None:
        super().__init__(
            name="jax",
            framework=JAX_FRAMEWORK,
            materialization="eager",
            trace_options=JAX_TRACE_OPTIONS,
            refusal_codes=JAX_REFUSAL_CODES,
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

    def validate(self, trace: Any, ground_truth: Any) -> Any:
        """Replay-validate a JAX trace; always the status object, never a bool.

        Parameters
        ----------
        trace:
            A trace captured by this backend.
        ground_truth:
            The reference outputs (accepted for the protocol signature; the
            JAX oracle replays saved payloads equation by equation).

        Returns
        -------
        ValidationStatus
            ``passed``, ``failed`` (naming the first offending op in
            ``message`` and in ``decisions``), ``unverified`` or
            ``unavailable``.
        """

        del ground_truth
        from ..jax.backend import JAXBackend

        return JAXBackend().validation_status(trace)

    def is_tensor(self, value: Any) -> bool:
        """Whether ``value`` is a ``jax.Array``."""

        try:
            import jax
        except Exception:  # noqa: BLE001
            return False
        return isinstance(value, jax.Array)

    def host_copy(self, value: Any) -> Any:
        """Detached NumPy copy of a ``jax.Array``; shape and dtype kept (0-d stays 0-d)."""

        import numpy as np

        try:
            return np.array(value, copy=True)
        except Exception as exc:  # noqa: BLE001 - a tracer or deleted buffer has no host copy
            raise BackendUnsupportedError(
                f"jax value of type {type(value).__name__} has no readable host buffer ({exc}).",
                code="constant_unrealizable",
                remedy="pass concrete arrays; tracers and deleted buffers have no payload",
            ) from exc

    def realize_once(self, values: Sequence[Any]) -> None:
        """Values are concrete on the jaxpr path; nothing to evaluate."""

        return None


def protocol() -> JAXProtocol:
    """Factory bound to ``BackendSpec.protocol`` for jax."""

    return JAXProtocol()
