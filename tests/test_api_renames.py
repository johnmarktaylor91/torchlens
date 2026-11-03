"""Tests for additive API renames and grouped-option migrations."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _user_funcs as user_funcs, _visualization
from torchlens._capture import run_entry
from torchlens._capture.prepare import door
from torchlens.options import CaptureOptions, StorageOptions

_STREAMING_CASES = [
    ("path", "save_outs_to", Path("bundle"), "save_outs_to"),
    ("retain_in_memory", "keep_outs_in_memory", False, "keep_outs_in_memory"),
    ("on_save", "out_sink", torch.sigmoid, "out_sink"),
]


class _TinyModel(nn.Module):
    """Small model used for API-plumbing tests."""

    def __init__(self) -> None:
        """Initialize the toy model."""

        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the toy forward pass."""

        return torch.relu(self.linear(x))


class _DummyLog:
    """Test double for ``Trace`` used by API-plumbing tests."""

    def __init__(self) -> None:
        """Initialize captured state for a fake log object."""

        self._options_verbose = False
        self._layer_logs: dict[str, Any] = {}
        self._num_saved_ops = 0
        self.saved_ops: list[Any] = []
        self._cost_total_activation_memory = "0 B"
        self._provenance_verified: Any = None
        self._options_structure_only = False
        self._options_output_device = "same"
        self.render_calls: list[dict[str, Any]] = []
        self.cleaned_up = False

    def draw(self, path: str | None = None, **kwargs: Any) -> str:
        """Record render kwargs passed by the API under test."""

        self.render_calls.append({"path": path, **kwargs})
        return "graph"

    def cleanup(self) -> None:
        """Record cleanup calls from wrapper helpers."""

        self.cleaned_up = True


@pytest.fixture
def stubbed_runner(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], _DummyLog]:
    """Replace the heavy logging helper with a capturing test double."""

    captured: dict[str, Any] = {}
    dummy_log = _DummyLog()

    def fake_run_model_and_save_specified_outs(*args: Any, **kwargs: Any) -> _DummyLog:
        """Capture forwarded kwargs and return a dummy log object."""

        del args
        captured.update(kwargs)
        return dummy_log

    monkeypatch.setattr(
        user_funcs,
        "_run_model_and_save_specified_outs",
        fake_run_model_and_save_specified_outs,
    )
    monkeypatch.setattr(
        run_entry,
        "_run_model_and_save_specified_outs",
        fake_run_model_and_save_specified_outs,
    )

    def fake_run_capture(request: Any) -> Any:
        """Stand in for the RUN stage: record the request's run keywords."""

        plan = run_entry._with_plan_defaults(request).plan
        trace = fake_run_model_and_save_specified_outs(
            request.model, request.input_args, request.input_kwargs, **plan
        )
        return SimpleNamespace(trace=trace)

    monkeypatch.setattr(door, "run_capture", fake_run_capture)
    return captured, dummy_log


def _tiny_input() -> torch.Tensor:
    """Return a deterministic small input tensor."""

    return torch.randn(1, 4)


def _deprecation_messages(records: list[warnings.WarningMessage]) -> list[str]:
    """Extract deprecation-warning messages from captured warnings."""

    return [
        str(record.message) for record in records if issubclass(record.category, DeprecationWarning)
    ]


def test_get_model_activations_is_not_advertised_as_a_legacy_api() -> None:
    """The never-shipped paper-era name is absent instead of forwarding to extract."""

    with pytest.raises(AttributeError, match="get_model_activations"):
        getattr(tl, "get_model_activations")


def test_save_options_activation_transform_alias_warns() -> None:
    """CaptureOptions activation_transform should set the canonical field."""

    def transform(tensor: torch.Tensor) -> torch.Tensor:
        """Return a transformed tensor for routing assertions."""

        return tensor + 1

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        options = CaptureOptions(activation_transform=transform)

    assert options.activation_transform is transform
    assert _deprecation_messages(records) == []


def test_show_model_graph_new_recurrence_detection_has_no_warning(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
) -> None:
    """The canonical recurrent-pattern kwarg should work on ``show_model_graph``."""

    captured, _dummy_log = stubbed_runner

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        _visualization.show_model_graph(_TinyModel(), _tiny_input(), recurrence_detection=False)

    assert captured["recurrence_detection"] is False
    assert _deprecation_messages(records) == []


def test_show_model_graph_recurrence_detection(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
) -> None:
    """The recurrence-detection kwarg should pass through to capture options."""

    captured, _dummy_log = stubbed_runner

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        _visualization.show_model_graph(_TinyModel(), _tiny_input(), recurrence_detection=False)

    assert captured["recurrence_detection"] is False
    assert not _deprecation_messages(records)


@pytest.mark.parametrize(
    ("field_name", "flat_name", "value", "captured_key"),
    _STREAMING_CASES,
)
def test_streaming_options_group_supports_every_field(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
    field_name: str,
    flat_name: str,
    value: Any,
    captured_key: str,
) -> None:
    """Every streaming group field should route through the grouped object."""

    del flat_name
    captured, _dummy_log = stubbed_runner
    option_kwargs = {field_name: value}

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.trace(
            _TinyModel(),
            _tiny_input(),
            save="all",
            storage=StorageOptions(**option_kwargs),
        )

    assert captured[captured_key] == value
    assert _deprecation_messages(records) == []
