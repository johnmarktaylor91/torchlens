"""Regression coverage for current TorchLens entry-point defaults."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._capture.prepare.door as door
import torchlens._capture.run_entry as run_entry
import torchlens._user_funcs as user_funcs
from torchlens import _visualization


class _DummyLog:
    """Minimal stand-in for ``Trace`` used by default-behavior tests."""

    def __init__(self) -> None:
        """Initialize a capture object with the attributes the wrappers expect."""
        self.cleanup_calls = 0
        self._layer_logs: dict[str, Any] = {}
        self._layer_list: list[Any] = []
        self._num_saved_ops = 0
        self.saved_ops: list[Any] = []
        self.render_calls: list[dict[str, Any]] = []
        self.summary_calls: list[dict[str, Any]] = []
        self.summary_result = "summary output"
        self._cost_total_activation_memory = "0 B"
        self._provenance_verified: Any = None
        self.validate_calls: list[dict[str, Any]] = []
        self._options_verbose = False
        self._options_structure_only = False
        # Save options a real ``Trace`` records from its capture call.
        self._options_output_device = "same"
        self.save_budget: Any = "auto"

    def cleanup(self) -> None:
        """Record cleanup requests from wrapper helpers."""
        self.cleanup_calls += 1

    def draw(self, **kwargs: Any) -> None:
        """Record visualization kwargs forwarded by the wrapper."""
        self.render_calls.append(kwargs)

    def summary(self, **kwargs: Any) -> str:
        """Record summary kwargs and return a fixed summary string."""
        self.summary_calls.append(kwargs)
        return self.summary_result

    def validate_forward_pass(
        self,
        ground_truth_output_tensors: list[torch.Tensor],
        verbose: bool,
        validate_metadata: bool = True,
    ) -> bool:
        """Record replay-validation inputs and return success.

        Parameters
        ----------
        ground_truth_output_tensors:
            Deduplicated model outputs collected before logging.
        verbose:
            Verbosity flag forwarded by ``validate_forward_pass``.
        validate_metadata:
            Metadata-invariant flag forwarded by ``validate_forward_pass``.

        Returns
        -------
        bool
            Always ``True`` for the stubbed validation path.
        """

        self.validate_calls.append(
            {
                "ground_truth_output_tensors": ground_truth_output_tensors,
                "verbose": verbose,
                "validate_metadata": validate_metadata,
            }
        )
        return True


class _TinyModel(nn.Module):
    """Small module used to exercise the public entry points."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a simple deterministic output tensor.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Input shifted by one.
        """

        return x + 1


def _tiny_input() -> torch.Tensor:
    """Return a stable input tensor for wrapper tests.

    Returns
    -------
    torch.Tensor
        Input tensor with a single batch element.
    """

    return torch.randn(1, 2)


@pytest.fixture
def stubbed_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], list[_DummyLog]]:
    """Patch the internal logging runner and capture forwarded kwargs.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.

    Returns
    -------
    tuple[list[dict[str, Any]], list[_DummyLog]]
        Captured runner kwargs and the dummy logs returned for each call.
    """

    captured_calls: list[dict[str, Any]] = []
    dummy_logs: list[_DummyLog] = []

    def _fake_runner(*args: Any, **kwargs: Any) -> _DummyLog:
        """Capture runner kwargs and return a fresh dummy log.

        Parameters
        ----------
        **kwargs:
            Keyword arguments forwarded by the public wrappers.

        Returns
        -------
        _DummyLog
            Stand-in log instance for the wrapper under test.
        """

        del args
        captured_calls.append(kwargs)
        dummy_log = _DummyLog()
        dummy_log._options_output_device = kwargs.get("output_device", "same")
        dummy_log.save_budget = kwargs.get("save_budget", "auto")
        dummy_logs.append(dummy_log)
        return dummy_log

    def _fake_run_capture(request: Any) -> Any:
        """Stand in for the RUN stage: record the request's run keywords."""

        plan = run_entry._with_plan_defaults(request).plan
        trace = _fake_runner(request.model, request.input_args, request.input_kwargs, **plan)
        return SimpleNamespace(trace=trace)

    monkeypatch.setattr(run_entry, "_run_model_and_save_specified_outs", _fake_runner)
    monkeypatch.setattr(user_funcs, "_run_model_and_save_specified_outs", _fake_runner)
    monkeypatch.setattr(door, "run_capture", _fake_run_capture)
    return captured_calls, dummy_logs


def test_trace_defaults_are_stable(
    stubbed_runner: tuple[list[dict[str, Any]], list[_DummyLog]],
) -> None:
    """``trace`` should preserve the audited default behavior."""

    captured_calls, dummy_logs = stubbed_runner

    result = tl.trace(_TinyModel(), _tiny_input(), save=None)

    assert result is dummy_logs[-1]
    assert captured_calls[-1]["layers_to_save"] is None
    assert captured_calls[-1]["output_device"] == "same"
    assert captured_calls[-1]["activation_transform"] is None
    assert captured_calls[-1]["mark_layer_depths"] is True
    assert captured_calls[-1]["detach_saved_activations"] is False
    assert captured_calls[-1]["save_args"] is False
    assert captured_calls[-1]["save_grads"] is False
    assert captured_calls[-1]["num_context_lines"] == 7
    assert captured_calls[-1]["save_source_context"] is False
    assert captured_calls[-1]["save_rng"] is False
    assert captured_calls[-1]["recurrence_detection"] is True
    assert captured_calls[-1]["save_outs_to"] is None
    assert captured_calls[-1]["keep_outs_in_memory"] is True
    assert captured_calls[-1]["out_sink"] is None
    assert captured_calls[-1]["verbose"] is False
    assert dummy_logs[-1].render_calls == []


def test_show_model_graph_defaults_are_stable(
    stubbed_runner: tuple[list[dict[str, Any]], list[_DummyLog]],
) -> None:
    """``show_model_graph`` should preserve its wrapper-specific defaults."""

    captured_calls, dummy_logs = stubbed_runner

    _visualization.show_model_graph(_TinyModel(), _tiny_input())

    assert captured_calls[-1]["layers_to_save"] is None
    assert captured_calls[-1]["mark_layer_depths"] is False
    assert captured_calls[-1]["detach_saved_activations"] is False
    assert captured_calls[-1]["save_grads"] is False
    assert captured_calls[-1]["recurrence_detection"] is True
    assert captured_calls[-1]["verbose"] is False
    assert dummy_logs[-1].render_calls[-1]["rolled"] is False
    assert dummy_logs[-1].cleanup_calls == 1


def test_summary_uses_metadata_only_defaults(
    stubbed_runner: tuple[list[dict[str, Any]], list[_DummyLog]],
) -> None:
    """``summary`` should still use metadata-only logging defaults."""

    captured_calls, dummy_logs = stubbed_runner

    result = tl.summary(_TinyModel(), _tiny_input(), depth=2)

    assert result == "summary output"
    assert captured_calls[-1]["layers_to_save"] is None
    assert captured_calls[-1]["recurrence_detection"] is True
    assert dummy_logs[-1].summary_calls[-1]["depth"] == 2
    assert dummy_logs[-1].cleanup_calls == 1
