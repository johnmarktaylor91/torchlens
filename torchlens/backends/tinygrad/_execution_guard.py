"""Execution guard for tinygrad capture.

Realizing a tensor, assigning into one, or running a ``TinyJit`` inside the
traced callable executes the lazy UOp graph and truncates the lineage the
capture reads afterwards. ``_reject_mid_capture_execution`` swaps tinygrad's
``run_linear`` entry points for a typed refusal while the forward runs and
restores them on exit. The backend module imports it.
"""

from __future__ import annotations

from typing import Any

from torchlens.backends.registry import BackendUnsupportedError


class _reject_mid_capture_execution:
    """Context manager rejecting tinygrad execution that truncates lazy UOp lineage."""

    def __init__(self) -> None:
        """Initialize the execution guard."""

        self.original_tensor_run_linear: Any = None
        self.original_jit_run_linear: Any = None

    def __enter__(self) -> _reject_mid_capture_execution:
        """Install guarded tinygrad realization hooks.

        Returns
        -------
        _reject_mid_capture_execution
            This context manager.
        """

        import tinygrad.engine.jit as jit_module
        import tinygrad.tensor as tensor_module

        self.original_tensor_run_linear = tensor_module.run_linear
        self.original_jit_run_linear = jit_module.run_linear

        def rejected_run_linear(*args: Any, **kwargs: Any) -> Any:
            """Reject tinygrad realization or JIT execution during capture."""

            del args, kwargs
            raise BackendUnsupportedError(
                "the tinygrad backend cannot capture Tensor.realize(), Tensor.assign(), "
                "or TinyJit execution inside the traced callable yet; return a lazy tinygrad "
                "expression instead.",
                code="mid_capture_execution_unsupported",
            )

        tensor_module.run_linear = rejected_run_linear
        jit_module.run_linear = rejected_run_linear
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Restore tinygrad realization hooks.

        Parameters
        ----------
        exc_type
            Exception type, if any.
        exc
            Exception value, if any.
        tb
            Traceback, if any.

        Returns
        -------
        None
            The monkeypatches are removed.
        """

        import tinygrad.engine.jit as jit_module
        import tinygrad.tensor as tensor_module

        tensor_module.run_linear = self.original_tensor_run_linear
        jit_module.run_linear = self.original_jit_run_linear
