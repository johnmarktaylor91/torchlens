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
from torchlens._deprecations import _WARNED_DEPRECATIONS
from torchlens.options import StreamingOptions, VisualizationOptions
from torchlens.validation import core as validation_core


_VISUALIZATION_CASES = [
    ("mode", "vis_mode", "rolled", "vis_mode"),
    ("max_module_depth", "vis_nesting_depth", 5, "vis_nesting_depth"),
    ("output_path", "vis_outpath", "custom.gv", "vis_outpath"),
    ("save_only", "vis_save_only", True, "vis_save_only"),
    ("file_format", "vis_fileformat", "svg", "vis_fileformat"),
    ("show_buffers", "vis_buffer_layers", True, "show_buffer_layers"),
    ("direction", "vis_direction", "leftright", "direction"),
    ("graph_overrides", "vis_graph_overrides", {"ranksep": "2.0"}, "vis_graph_overrides"),
    ("edge_overrides", "vis_edge_overrides", {"color": "red"}, "vis_edge_overrides"),
    (
        "gradient_edge_overrides",
        "vis_gradient_edge_overrides",
        {"style": "dashed"},
        "vis_gradient_edge_overrides",
    ),
    ("module_overrides", "vis_module_overrides", {"color": "blue"}, "vis_module_overrides"),
    ("layout_engine", "vis_node_placement", "dot", "vis_node_placement"),
    ("renderer", "vis_renderer", "dagua", "vis_renderer"),
    ("theme", "vis_theme", "gallery", "vis_theme"),
]
_STREAMING_CASES = [
    ("bundle_path", "save_activations_to", Path("bundle"), "save_activations_to"),
    ("retain_in_memory", "keep_activations_in_memory", False, "keep_activations_in_memory"),
    ("activation_callback", "activation_sink", torch.sigmoid, "activation_sink"),
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
    """Test double for ``ModelLog`` used by API-plumbing tests."""

    def __init__(self) -> None:
        """Initialize captured state for a fake log object."""

        self.verbose = False
        self.layer_logs: dict[str, Any] = {}
        self.num_tensors_saved = 0
        self.total_activation_memory_str = "0 B"
        self.render_calls: list[dict[str, Any]] = []
        self.cleaned_up = False

    def render_graph(
        self,
        vis_mode: str = "unrolled",
        vis_nesting_depth: int = 1000,
        vis_outpath: str = "modelgraph",
        vis_graph_overrides: dict[str, Any] | None = None,
        node_mode: str = "default",
        node_spec_fn: Any = None,
        collapsed_node_spec_fn: Any = None,
        collapse_fn: Any = None,
        skip_fn: Any = None,
        vis_edge_overrides: dict[str, Any] | None = None,
        vis_gradient_edge_overrides: dict[str, Any] | None = None,
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
                "vis_nesting_depth": vis_nesting_depth,
                "vis_outpath": vis_outpath,
                "vis_graph_overrides": vis_graph_overrides,
                "node_mode": node_mode,
                "node_spec_fn": node_spec_fn,
                "collapsed_node_spec_fn": collapsed_node_spec_fn,
                "collapse_fn": collapse_fn,
                "skip_fn": skip_fn,
                "vis_edge_overrides": vis_edge_overrides,
                "vis_gradient_edge_overrides": vis_gradient_edge_overrides,
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


@pytest.fixture(autouse=True)
def clear_deprecation_state() -> None:
    """Reset deprecation dedup state before each test."""

    _WARNED_DEPRECATIONS.clear()


@pytest.fixture
def stubbed_runner(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], _DummyLog]:
    """Replace the heavy logging helper with a capturing test double."""

    captured: dict[str, Any] = {}
    dummy_log = _DummyLog()

    def fake_run_model_and_save_specified_activations(*args: Any, **kwargs: Any) -> _DummyLog:
        """Capture forwarded kwargs and return a dummy log object."""

        del args
        captured.update(kwargs)
        return dummy_log

    monkeypatch.setattr(
        user_funcs,
        "_run_model_and_save_specified_activations",
        fake_run_model_and_save_specified_activations,
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


def test_get_model_metadata_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deprecated top-level metadata helper should warn once per process."""

    sentinel = object()

    def fake_log_model_metadata(*args: Any, **kwargs: Any) -> object:
        """Return a stable sentinel without running real logging."""

        del args, kwargs
        return sentinel

    monkeypatch.setattr(user_funcs, "log_model_metadata", fake_log_model_metadata)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        first = tl.get_model_metadata(_TinyModel(), _tiny_input())
        second = tl.get_model_metadata(_TinyModel(), _tiny_input())

    assert first is sentinel
    assert second is sentinel
    assert _deprecation_messages(records) == [
        "torchlens.get_model_metadata is deprecated; use torchlens.io.get_model_metadata "
        "instead. Removed in v2.NN.",
        "`get_model_metadata` is deprecated; use `log_model_metadata` instead. "
        "The old name continues to work but will be removed in a future release.",
        "torchlens.get_model_metadata is deprecated; use torchlens.io.get_model_metadata "
        "instead. Removed in v2.NN.",
    ]


def test_log_model_metadata_new_name_has_no_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The namespaced metadata helper should not emit deprecation warnings."""

    sentinel = object()

    def fake_log_forward_pass(*args: Any, **kwargs: Any) -> object:
        """Return a stable sentinel without running real logging."""

        del args, kwargs
        return sentinel

    monkeypatch.setattr(user_funcs, "log_forward_pass", fake_log_forward_pass)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = tl.io.log_model_metadata(_TinyModel(), _tiny_input())

    assert result is sentinel
    assert _deprecation_messages(records) == []


def test_validate_saved_activations_alias_warns_and_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deprecated top-level validation alias should warn and forward kwargs."""

    captured: dict[str, Any] = {}

    def fake_validate_forward_pass(*args: Any, **kwargs: Any) -> bool:
        """Capture forwarded kwargs and return success."""

        del args
        captured.update(kwargs)
        return True

    monkeypatch.setattr(user_funcs, "validate_forward_pass", fake_validate_forward_pass)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = tl.validation.validate_saved_activations(
            _TinyModel(),
            _tiny_input(),
            validate_metadata=False,
        )

    assert result is True
    assert captured["validate_metadata"] is False
    assert len(_deprecation_messages(records)) == 1


def test_model_log_validate_saved_activations_warns_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deprecated ``ModelLog`` method alias should warn once and delegate."""

    model_log = tl.ModelLog("tiny")

    def fake_validate_saved_activations(*args: Any, **kwargs: Any) -> bool:
        """Return success without running the real validation path."""

        del args, kwargs
        return True

    monkeypatch.setattr(
        validation_core,
        "validate_saved_activations",
        fake_validate_saved_activations,
    )

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        first = model_log.validate_saved_activations([torch.randn(1)])
        second = model_log.validate_saved_activations([torch.randn(1)])

    assert first is True
    assert second is True
    assert len(_deprecation_messages(records)) == 1


@pytest.mark.parametrize(
    ("old_name", "new_name", "value", "captured_key"),
    [
        ("num_context_lines", "source_context_lines", 5, "num_context_lines"),
        (
            "mark_input_output_distances",
            "compute_input_output_distances",
            True,
            "mark_input_output_distances",
        ),
        ("detect_loops", "detect_recurrent_patterns", False, "detect_loops"),
    ],
)
def test_log_forward_pass_old_renamed_kwargs_warn(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
    old_name: str,
    new_name: str,
    value: Any,
    captured_key: str,
) -> None:
    """Deprecated flat rename aliases should still work on ``log_forward_pass``."""

    captured, _dummy_log = stubbed_runner
    kwargs = {old_name: value}

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.log_forward_pass(_TinyModel(), _tiny_input(), layers_to_save=None, **kwargs)

    assert captured[captured_key] == value
    assert len(_deprecation_messages(records)) == 1
    assert new_name in _deprecation_messages(records)[0]


@pytest.mark.parametrize(
    ("new_name", "value", "captured_key"),
    [
        ("source_context_lines", 6, "num_context_lines"),
        ("compute_input_output_distances", True, "mark_input_output_distances"),
        ("detect_recurrent_patterns", False, "detect_loops"),
    ],
)
def test_log_forward_pass_new_renamed_kwargs_do_not_warn(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
    new_name: str,
    value: Any,
    captured_key: str,
) -> None:
    """Canonical renamed kwargs should work on ``log_forward_pass`` without warnings."""

    captured, _dummy_log = stubbed_runner
    kwargs = {new_name: value}

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.log_forward_pass(_TinyModel(), _tiny_input(), layers_to_save=None, **kwargs)

    assert captured[captured_key] == value
    assert _deprecation_messages(records) == []


@pytest.mark.parametrize(
    ("old_name", "new_name", "old_value", "new_value"),
    [
        ("num_context_lines", "source_context_lines", 5, 6),
        ("mark_input_output_distances", "compute_input_output_distances", True, False),
        ("detect_loops", "detect_recurrent_patterns", True, False),
    ],
)
def test_log_forward_pass_mixing_old_and_new_renamed_kwargs_raises(
    old_name: str,
    new_name: str,
    old_value: Any,
    new_value: Any,
) -> None:
    """Mixing deprecated and canonical rename spellings should fail."""

    with pytest.raises(
        TypeError,
        match=f"kwarg {old_name} deprecated, use {new_name}; do not pass both",
    ):
        tl.log_forward_pass(
            _TinyModel(),
            _tiny_input(),
            layers_to_save=None,
            **{old_name: old_value, new_name: new_value},
        )


def test_old_kwarg_warning_deduplicates_per_process(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
) -> None:
    """Repeated use of the same deprecated kwarg should only warn once."""

    del stubbed_runner
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.log_forward_pass(_TinyModel(), _tiny_input(), layers_to_save=None, num_context_lines=3)
        tl.log_forward_pass(_TinyModel(), _tiny_input(), layers_to_save=None, num_context_lines=4)

    assert len(_deprecation_messages(records)) == 1


def test_show_model_graph_new_detect_recurrent_patterns_has_no_warning(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
) -> None:
    """The canonical recurrent-pattern kwarg should work on ``show_model_graph``."""

    captured, _dummy_log = stubbed_runner

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.visualization.show_model_graph(
            _TinyModel(), _tiny_input(), detect_recurrent_patterns=False
        )

    assert captured["detect_loops"] is False
    assert _deprecation_messages(records) == []


def test_show_model_graph_old_detect_loops_warns(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
) -> None:
    """The deprecated recurrent-pattern kwarg should still work on ``show_model_graph``."""

    captured, _dummy_log = stubbed_runner

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.visualization.show_model_graph(_TinyModel(), _tiny_input(), detect_loops=False)

    assert captured["detect_loops"] is False
    assert len(_deprecation_messages(records)) == 1


def test_show_model_graph_mixing_detect_loop_names_raises() -> None:
    """Mixing old and new recurrent-pattern kwarg names should fail."""

    with pytest.raises(
        TypeError,
        match="kwarg detect_loops deprecated, use detect_recurrent_patterns; do not pass both",
    ):
        tl.visualization.show_model_graph(
            _TinyModel(),
            _tiny_input(),
            detect_loops=True,
            detect_recurrent_patterns=False,
        )


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
    option_kwargs: dict[str, Any] = {"mode": "rolled"}
    if field_name == "mode":
        option_kwargs = {"mode": value}
    else:
        option_kwargs[field_name] = value

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.log_forward_pass(
            _TinyModel(),
            _tiny_input(),
            layers_to_save=None,
            visualization=VisualizationOptions(**option_kwargs),
        )

    assert _deprecation_messages(records) == []
    assert dummy_log.render_calls[-1][render_key] == value


@pytest.mark.parametrize(
    ("field_name", "flat_name", "value", "render_key"),
    _VISUALIZATION_CASES,
)
def test_visualization_flat_aliases_warn_and_route(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
    field_name: str,
    flat_name: str,
    value: Any,
    render_key: str,
) -> None:
    """Every deprecated flat visualization kwarg should still work."""

    del field_name
    _captured, dummy_log = stubbed_runner

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.visualization.show_model_graph(_TinyModel(), _tiny_input(), **{flat_name: value})

    assert dummy_log.render_calls[-1][render_key] == value
    assert len(_deprecation_messages(records)) == 1


def test_visualization_group_and_flat_same_field_raise() -> None:
    """Same-field grouped and flat visualization inputs should conflict."""

    with pytest.raises(TypeError, match="Do not pass both `vis_mode` and `visualization.mode`."):
        tl.visualization.show_model_graph(
            _TinyModel(),
            _tiny_input(),
            visualization=VisualizationOptions(mode="rolled"),
            vis_mode="unrolled",
        )


def test_visualization_group_and_flat_same_field_raise_for_explicit_default() -> None:
    """Explicit default-valued grouped visualization fields still count as supplied."""

    with pytest.raises(
        TypeError,
        match="Do not pass both `vis_nesting_depth` and `visualization.max_module_depth`.",
    ):
        tl.visualization.show_model_graph(
            _TinyModel(),
            _tiny_input(),
            visualization=VisualizationOptions(mode="rolled", max_module_depth=1000),
            vis_nesting_depth=5,
        )


def test_visualization_group_and_flat_different_fields_merge_without_mutation(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
) -> None:
    """Different visualization fields should merge and leave the caller object unchanged."""

    _captured, dummy_log = stubbed_runner
    visualization = VisualizationOptions(mode="rolled")

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.visualization.show_model_graph(
            _TinyModel(),
            _tiny_input(),
            visualization=visualization,
            vis_nesting_depth=5,
        )

    assert visualization.max_module_depth == 1000
    assert dummy_log.render_calls[-1]["vis_mode"] == "rolled"
    assert dummy_log.render_calls[-1]["vis_nesting_depth"] == 5
    assert len(_deprecation_messages(records)) == 1


def test_visualization_defaults_preserve_per_function_behavior(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
) -> None:
    """Grouped visualization defaults must keep current per-function behavior."""

    _captured, dummy_log = stubbed_runner

    tl.log_forward_pass(_TinyModel(), _tiny_input(), layers_to_save=None)
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
        tl.log_forward_pass(
            _TinyModel(),
            _tiny_input(),
            layers_to_save="all",
            streaming=StreamingOptions(**option_kwargs),
        )

    assert captured[captured_key] == value
    assert _deprecation_messages(records) == []


@pytest.mark.parametrize(
    ("field_name", "flat_name", "value", "captured_key"),
    _STREAMING_CASES,
)
def test_streaming_flat_aliases_warn_and_route(
    stubbed_runner: tuple[dict[str, Any], _DummyLog],
    field_name: str,
    flat_name: str,
    value: Any,
    captured_key: str,
) -> None:
    """Every deprecated flat streaming kwarg should still work."""

    del field_name
    captured, _dummy_log = stubbed_runner

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tl.log_forward_pass(_TinyModel(), _tiny_input(), layers_to_save="all", **{flat_name: value})

    assert captured[captured_key] == value
    assert len(_deprecation_messages(records)) == 1


def test_streaming_group_and_flat_same_field_raise() -> None:
    """Same-field grouped and flat streaming inputs should conflict."""

    with pytest.raises(
        TypeError,
        match="Do not pass both `save_activations_to` and `streaming.bundle_path`.",
    ):
        tl.log_forward_pass(
            _TinyModel(),
            _tiny_input(),
            layers_to_save="all",
            streaming=StreamingOptions(bundle_path=Path("bundle")),
            save_activations_to=Path("other"),
        )


def test_streaming_group_and_flat_same_field_raise_for_explicit_default() -> None:
    """Explicit default-valued grouped streaming fields still count as supplied."""

    with pytest.raises(
        TypeError,
        match="Do not pass both `save_activations_to` and `streaming.bundle_path`.",
    ):
        tl.log_forward_pass(
            _TinyModel(),
            _tiny_input(),
            layers_to_save="all",
            streaming=StreamingOptions(bundle_path=None),
            save_activations_to=Path("bundle"),
        )
