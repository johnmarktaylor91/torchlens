"""Tests for additive API renames and grouped-option migrations."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import user_funcs
from torchlens.options import CaptureOptions, SaveOptions, StreamingOptions, VisualizationOptions

_VISUALIZATION_CASES = [
    ("view", "vis_mode", "rolled", "vis_mode"),
    ("depth", "vis_call_depth", 5, "vis_call_depth"),
    ("container_path", "vis_outpath", "custom.gv", "vis_outpath"),
    ("save_only", "vis_save_only", True, "vis_save_only"),
    ("file_format", "vis_fileformat", "svg", "vis_fileformat"),
    # Canonical tri-state value on purpose: the legacy bools are themselves a
    # deprecated VALUE now (grind b4, R48-1), and these two cases assert the
    # routing of the NAME, so a bool here would smuggle a second warning in.
    ("show_buffers", "vis_buffers", "always", "show_buffer_layers"),
    ("direction", "vis_direction", "leftright", "direction"),
    ("graph_overrides", "vis_graph_overrides", {"ranksep": "2.0"}, "vis_graph_overrides"),
    ("edge_overrides", "vis_edge_overrides", {"color": "red"}, "vis_edge_overrides"),
    (
        "grad_edge_overrides",
        "vis_grad_edge_overrides",
        {"style": "dashed"},
        "vis_grad_edge_overrides",
    ),
    ("module_overrides", "vis_module_overrides", {"color": "blue"}, "vis_module_overrides"),
    ("layout", "vis_node_placement", "dot", "vis_node_placement"),
    ("renderer", "vis_renderer", "dagua", "vis_renderer"),
    ("theme", "vis_theme", "gallery", "vis_theme"),
    ("node_style", "vis_node_mode", "profiling", "node_mode"),
]
_STREAMING_CASES = [
    ("bundle_path", "save_outs_to", Path("bundle"), "save_outs_to"),
    ("retain_in_memory", "keep_outs_in_memory", False, "keep_outs_in_memory"),
    ("out_callback", "out_sink", torch.sigmoid, "out_sink"),
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

        self.verbose = False
        self.layer_logs: dict[str, Any] = {}
        self.num_saved_ops = 0
        self.total_activation_memory = "0 B"
        self.render_calls: list[dict[str, Any]] = []
        self.cleaned_up = False

    def draw(
        self,
        vis_mode: str = "unrolled",
        vis_call_depth: int = 1000,
        vis_outpath: str = "modelgraph",
        vis_graph_overrides: dict[str, Any] | None = None,
        node_mode: str = "default",
        node_spec_fn: Any = None,
        collapsed_node_spec_fn: Any = None,
        collapse_fn: Any = None,
        collapse: str = "none",
        skip_fn: Any = None,
        vis_edge_overrides: dict[str, Any] | None = None,
        vis_grad_edge_overrides: dict[str, Any] | None = None,
        vis_module_overrides: dict[str, Any] | None = None,
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        show_buffer_layers: bool = False,
        direction: str = "bottomup",
        vis_node_placement: str = "auto",
        vis_renderer: str = "graphviz",
        vis_theme: str = "torchlens",
        vis_intervention_mode: str = "node_mark",
        vis_show_cone: bool = False,
    ) -> str:
        """Record render kwargs passed by the API under test."""

        self.render_calls.append(
            {
                "vis_mode": vis_mode,
                "vis_call_depth": vis_call_depth,
                "vis_outpath": vis_outpath,
                "vis_graph_overrides": vis_graph_overrides,
                "node_mode": node_mode,
                "node_spec_fn": node_spec_fn,
                "collapsed_node_spec_fn": collapsed_node_spec_fn,
                "collapse_fn": collapse_fn,
                "skip_fn": skip_fn,
                "vis_edge_overrides": vis_edge_overrides,
                "vis_grad_edge_overrides": vis_grad_edge_overrides,
                "vis_module_overrides": vis_module_overrides,
                "vis_save_only": vis_save_only,
                "vis_fileformat": vis_fileformat,
                "show_buffer_layers": show_buffer_layers,
                "direction": direction,
                "vis_node_placement": vis_node_placement,
                "vis_renderer": vis_renderer,
                "vis_theme": vis_theme,
                "vis_intervention_mode": vis_intervention_mode,
                "vis_show_cone": vis_show_cone,
            }
        )
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


def test_log_model_metadata_new_name_has_no_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The namespaced metadata helper should not emit deprecation warnings."""

    sentinel = object()

    def fake_trace(*args: Any, **kwargs: Any) -> object:
        """Return a stable sentinel without running real logging."""

        del args, kwargs
        return sentinel

    monkeypatch.setattr(user_funcs, "trace", fake_trace)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = tl.io.log_model_metadata(_TinyModel(), _tiny_input())

    assert result is sentinel
    assert _deprecation_messages(records) == []


def test_save_options_activation_transform_alias_warns() -> None:
    """SaveOptions activation_transform should set the canonical field."""

    def transform(tensor: torch.Tensor) -> torch.Tensor:
        """Return a transformed tensor for routing assertions."""

        return tensor + 1

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        options = SaveOptions(activation_transform=transform)

    assert options.activation_transform is transform
    assert _deprecation_messages(records) == []


def test_show_model_graph_new_recurrence_detection_has_no_warning(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
) -> None:
    """The canonical recurrent-pattern kwarg should work on ``show_model_graph``."""

    captured, _dummy_log = stubbed_runner

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.visualization.show_model_graph(_TinyModel(), _tiny_input(), recurrence_detection=False)

    assert captured["recurrence_detection"] is False
    assert _deprecation_messages(records) == []


def test_show_model_graph_recurrence_detection(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
) -> None:
    """The recurrence-detection kwarg should pass through to capture options."""

    captured, _dummy_log = stubbed_runner

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.visualization.show_model_graph(_TinyModel(), _tiny_input(), recurrence_detection=False)

    assert captured["recurrence_detection"] is False
    assert not _deprecation_messages(records)


@pytest.mark.parametrize(
    ("field_name", "flat_name", "value", "render_key"),
    _VISUALIZATION_CASES,
)
def test_visualization_options_group_supports_every_field(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
    field_name: str,
    flat_name: str,
    value: Any,
    render_key: str,
) -> None:
    """Every visualization group field should route through the grouped object."""

    del flat_name
    _captured, dummy_log = stubbed_runner
    option_kwargs: dict[str, Any] = {"view": "rolled"}
    if field_name == "view":
        option_kwargs = {"view": value}
    else:
        option_kwargs[field_name] = value

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.visualization.show_model_graph(
            _TinyModel(),
            _tiny_input(),
            visualization=VisualizationOptions(**option_kwargs),
        )

    assert _deprecation_messages(records) == []
    assert dummy_log.render_calls[-1][render_key] == value


def test_visualization_defaults_preserve_per_function_behavior(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
) -> None:
    """Grouped visualization defaults must keep current per-function behavior."""

    _captured, dummy_log = stubbed_runner

    tl.trace(
        _TinyModel(),
        _tiny_input(),
        capture=CaptureOptions(layers_to_save=None),
    )
    assert dummy_log.render_calls == []

    tl.visualization.show_model_graph(_TinyModel(), _tiny_input())
    assert dummy_log.render_calls[-1]["vis_mode"] == "unrolled"


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
            capture=CaptureOptions(layers_to_save="all"),
            streaming=StreamingOptions(**option_kwargs),
        )

    assert captured[captured_key] == value
    assert _deprecation_messages(records) == []
