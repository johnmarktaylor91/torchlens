"""Regression coverage for current TorchLens entry-point defaults."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.user_funcs as user_funcs
from torchlens.options import VisualizationOptions


class _DummyLog:
    """Minimal stand-in for ``ModelLog`` used by default-behavior tests."""

    def __init__(self) -> None:
        """Initialize a capture object with the attributes the wrappers expect."""
        self.cleanup_calls = 0
        self.layer_logs: dict[str, Any] = {}
        self.num_tensors_saved = 0
        self.render_calls: list[dict[str, Any]] = []
        self.summary_calls: list[dict[str, Any]] = []
        self.summary_result = "summary output"
        self.total_activation_memory_str = "0 B"
        self.validate_calls: list[dict[str, Any]] = []
        self.verbose = False

    def cleanup(self) -> None:
        """Record cleanup requests from wrapper helpers."""
        self.cleanup_calls += 1

    def render_graph(self, **kwargs: Any) -> None:
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

    def _fake_runner(**kwargs: Any) -> _DummyLog:
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

        captured_calls.append(kwargs)
        dummy_log = _DummyLog()
        dummy_logs.append(dummy_log)
        return dummy_log

    monkeypatch.setattr(user_funcs, "_run_model_and_save_specified_activations", _fake_runner)
    return captured_calls, dummy_logs


def test_log_forward_pass_defaults_are_stable(
    stubbed_runner: tuple[list[dict[str, Any]], list[_DummyLog]],
) -> None:
    """``log_forward_pass`` should preserve the audited default behavior."""

    captured_calls, dummy_logs = stubbed_runner

    result = tl.log_forward_pass(_TinyModel(), _tiny_input(), layers_to_save=None)

    assert result is dummy_logs[-1]
    assert captured_calls[-1]["layers_to_save"] is None
    assert captured_calls[-1]["keep_unsaved_layers"] is True
    assert captured_calls[-1]["output_device"] == "same"
    assert captured_calls[-1]["activation_postfunc"] is None
    assert captured_calls[-1]["mark_input_output_distances"] is False
    assert captured_calls[-1]["detach_saved_tensors"] is False
    assert captured_calls[-1]["save_function_args"] is False
    assert captured_calls[-1]["save_gradients"] is False
    assert captured_calls[-1]["num_context_lines"] == 7
    assert captured_calls[-1]["save_source_context"] is False
    assert captured_calls[-1]["save_rng_states"] is False
    assert captured_calls[-1]["detect_loops"] is True
    assert captured_calls[-1]["save_activations_to"] is None
    assert captured_calls[-1]["keep_activations_in_memory"] is True
    assert captured_calls[-1]["activation_sink"] is None
    assert captured_calls[-1]["verbose"] is False
    assert dummy_logs[-1].render_calls == []


def test_log_forward_pass_accepts_explicit_opt_in_overrides(
    stubbed_runner: tuple[list[dict[str, Any]], list[_DummyLog]],
) -> None:
    """Callers should still be able to opt into the non-default behaviors."""

    captured_calls, dummy_logs = stubbed_runner

    result = tl.log_forward_pass(
        _TinyModel(),
        _tiny_input(),
        layers_to_save=None,
        keep_unsaved_layers=False,
        compute_input_output_distances=True,
        save_source_context=True,
        detect_recurrent_patterns=False,
        visualization=VisualizationOptions(mode="rolled"),
    )

    assert result is dummy_logs[-1]
    assert captured_calls[-1]["keep_unsaved_layers"] is False
    assert captured_calls[-1]["mark_input_output_distances"] is True
    assert captured_calls[-1]["save_source_context"] is True
    assert captured_calls[-1]["detect_loops"] is False
    assert dummy_logs[-1].render_calls[-1]["vis_mode"] == "rolled"


def test_show_model_graph_defaults_are_stable(
    stubbed_runner: tuple[list[dict[str, Any]], list[_DummyLog]],
) -> None:
    """``show_model_graph`` should preserve its wrapper-specific defaults."""

    captured_calls, dummy_logs = stubbed_runner

    tl.show_model_graph(_TinyModel(), _tiny_input())

    assert captured_calls[-1]["layers_to_save"] is None
    assert captured_calls[-1]["mark_input_output_distances"] is False
    assert captured_calls[-1]["detach_saved_tensors"] is False
    assert captured_calls[-1]["save_gradients"] is False
    assert captured_calls[-1]["detect_loops"] is True
    assert captured_calls[-1]["verbose"] is False
    assert dummy_logs[-1].render_calls[-1]["vis_mode"] == "unrolled"
    assert dummy_logs[-1].cleanup_calls == 1


def test_show_model_graph_accepts_explicit_opt_in_overrides(
    stubbed_runner: tuple[list[dict[str, Any]], list[_DummyLog]],
) -> None:
    """``show_model_graph`` should still honor explicit non-default choices."""

    captured_calls, dummy_logs = stubbed_runner

    tl.show_model_graph(
        _TinyModel(),
        _tiny_input(),
        detect_recurrent_patterns=False,
        visualization=VisualizationOptions(mode="rolled"),
    )

    assert captured_calls[-1]["detect_loops"] is False
    assert dummy_logs[-1].render_calls[-1]["vis_mode"] == "rolled"
    assert dummy_logs[-1].cleanup_calls == 1


def test_log_model_metadata_forces_metadata_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """``log_model_metadata`` should keep its metadata-only wrapper behavior."""

    captured_kwargs: dict[str, Any] = {}
    dummy_log = _DummyLog()

    def _fake_log_forward_pass(*args: Any, **kwargs: Any) -> _DummyLog:
        """Capture wrapper kwargs and return a dummy log.

        Parameters
        ----------
        *args:
            Positional arguments passed through the wrapper.
        **kwargs:
            Keyword arguments passed through the wrapper.

        Returns
        -------
        _DummyLog
            Stand-in log instance for the wrapper under test.
        """

        del args
        captured_kwargs.update(kwargs)
        return dummy_log

    monkeypatch.setattr(user_funcs, "log_forward_pass", _fake_log_forward_pass)

    result = tl.log_model_metadata(_TinyModel(), _tiny_input())

    assert result is dummy_log
    assert captured_kwargs["layers_to_save"] is None
    assert captured_kwargs["compute_input_output_distances"] is True


def test_summary_uses_metadata_only_defaults(
    stubbed_runner: tuple[list[dict[str, Any]], list[_DummyLog]],
) -> None:
    """``summary`` should still use metadata-only logging defaults."""

    captured_calls, dummy_logs = stubbed_runner

    result = tl.summary(_TinyModel(), _tiny_input(), depth=2)

    assert result == "summary output"
    assert captured_calls[-1]["layers_to_save"] is None
    assert captured_calls[-1]["keep_unsaved_layers"] is True
    assert captured_calls[-1]["detect_loops"] is True
    assert dummy_logs[-1].summary_calls[-1] == {"depth": 2}
    assert dummy_logs[-1].cleanup_calls == 1


def test_validate_forward_pass_uses_validation_overrides(
    stubbed_runner: tuple[list[dict[str, Any]], list[_DummyLog]],
) -> None:
    """``validate_forward_pass`` should keep its replay-specific overrides."""

    captured_calls, dummy_logs = stubbed_runner

    result = tl.validate_forward_pass(
        _TinyModel(),
        _tiny_input(),
        random_seed=123,
        validate_metadata=False,
    )

    assert result is True
    assert captured_calls[-1]["layers_to_save"] == "all"
    assert captured_calls[-1]["keep_unsaved_layers"] is True
    assert captured_calls[-1]["activation_postfunc"] is None
    assert captured_calls[-1]["mark_input_output_distances"] is False
    assert captured_calls[-1]["detach_saved_tensors"] is False
    assert captured_calls[-1]["save_gradients"] is False
    assert captured_calls[-1]["save_function_args"] is True
    assert captured_calls[-1]["save_rng_states"] is True
    assert dummy_logs[-1].validate_calls[-1]["validate_metadata"] is False
    assert dummy_logs[-1].cleanup_calls == 1
