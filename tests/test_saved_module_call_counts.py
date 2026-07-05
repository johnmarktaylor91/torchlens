"""Regression tests for saved ModuleCall summary counters."""

import torch
from torch import nn

import torchlens as tl


class _ModuleCallCounterModel(nn.Module):
    """Tiny model with a named child module for ModuleCall counting."""

    def __init__(self) -> None:
        """Initialize the child module."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the child module.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.linear(x)


def _assert_saved_module_count_matches_dynamic(trace: tl.Trace) -> None:
    """Assert the stored saved-module counter mirrors the dynamic accessor.

    Parameters
    ----------
    trace:
        Trace to check.
    """

    assert trace.num_saved_module_calls == len(trace.saved_module_calls)
    assert trace.num_saved_module_calls > 0


def test_num_saved_module_calls_matches_dynamic_for_all_saved_trace() -> None:
    """Stored saved ModuleCall count matches the dynamic accessor for all-saved traces."""

    trace = tl.trace(_ModuleCallCounterModel(), torch.ones(1, 3))

    _assert_saved_module_count_matches_dynamic(trace)


def test_num_saved_module_calls_matches_dynamic_for_selective_save_trace() -> None:
    """Stored saved ModuleCall count matches the dynamic accessor for selective traces."""

    trace = tl.trace(
        _ModuleCallCounterModel(),
        torch.ones(1, 3),
        save=tl.module("linear"),
    )

    _assert_saved_module_count_matches_dynamic(trace)
