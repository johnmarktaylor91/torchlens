"""Tests for TorchLens 2.0 top-level deprecation shims."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens

OBJECT_ALIAS_CASES = [
    ("ActivationPostfunc", "torchlens.types", "ActivationPostfunc"),
    ("BufferLog", "torchlens.types", "BufferLog"),
    ("FuncCallLocation", "torchlens.types", "FuncCallLocation"),
    ("GradientPostfunc", "torchlens.types", "GradientPostfunc"),
    ("GradFnAccessor", "torchlens.accessors", "GradFnAccessor"),
    ("GradFnLog", "torchlens.types", "GradFnLog"),
    ("GradFnPassLog", "torchlens.types", "GradFnPassLog"),
    ("LayerAccessor", "torchlens.accessors", "LayerAccessor"),
    ("MetadataInvariantError", "torchlens.errors", "MetadataInvariantError"),
    ("ModuleAccessor", "torchlens.accessors", "ModuleAccessor"),
    ("ModuleLog", "torchlens.types", "ModuleLog"),
    ("ModulePassLog", "torchlens.types", "ModulePassLog"),
    ("NodeSpec", "torchlens.experimental.dagua", "NodeSpec"),
    ("ParamLog", "torchlens.types", "ParamLog"),
    ("RunState", "torchlens.io", "RunState"),
    ("SaveLevel", "torchlens.types", "SaveLevel"),
    ("SiteTable", "torchlens.types", "SiteTable"),
    ("SpecCompat", "torchlens.types", "SpecCompat"),
    ("StreamingOptions", "torchlens.options", "StreamingOptions"),
    ("TargetManifestDiff", "torchlens.types", "TargetManifestDiff"),
    ("TensorLog", "torchlens.types", "TensorLog"),
    ("TensorSliceSpec", "torchlens.types", "TensorSliceSpec"),
    ("TorchLensPostfuncError", "torchlens.errors", "TorchLensPostfuncError"),
    ("TrainingModeConfigError", "torchlens.errors", "TrainingModeConfigError"),
    ("VisualizationOptions", "torchlens.options", "VisualizationOptions"),
    ("build_render_audit", "torchlens.experimental.dagua", "build_render_audit"),
    ("check_metadata_invariants", "torchlens.validation", "check_metadata_invariants"),
    ("check_spec_compat", "torchlens.validation", "check_spec_compat"),
    ("cleanup_tmp", "torchlens.io", "cleanup_tmp"),
    ("get_model_metadata", "torchlens.io", "get_model_metadata"),
    ("list_logs", "torchlens.io", "list_logs"),
    ("log_model_metadata", "torchlens.io", "log_model_metadata"),
    ("model_log_to_dagua_graph", "torchlens.experimental.dagua", "model_log_to_dagua_graph"),
    ("preview_fastlog", "torchlens.fastlog", "preview"),
    ("rehydrate_nested", "torchlens.io", "rehydrate_nested"),
    ("render_lines_to_html", "torchlens.experimental.dagua", "render_lines_to_html"),
    ("render_model_log_with_dagua", "torchlens.experimental.dagua", "render_model_log_with_dagua"),
    ("reset_naming_counter", "torchlens.io", "reset_naming_counter"),
    ("resolve_sites", "torchlens.validation", "resolve_sites"),
    ("save_intervention", "torchlens.io", "save_intervention"),
    ("suppress_mutate_warnings", "torchlens.io", "suppress_mutate_warnings"),
    ("unwrap_torch", "torchlens.decoration", "unwrap_torch"),
    (
        "validate_batch_of_models_and_inputs",
        "torchlens.validation",
        "validate_batch_of_models_and_inputs",
    ),
    ("wrap_torch", "torchlens.decoration", "wrap_torch"),
    ("wrapped", "torchlens.decoration", "wrapped"),
]

WRAPPER_CASES = [
    ("validate_forward_pass", torchlens.validation.validate_forward_pass),
    ("validate_backward_pass", torchlens.validation.validate_backward_pass),
    ("validate_saved_activations", torchlens.validation.validate_saved_activations),
    ("summary", torchlens.visualization.summary),
    ("show_model_graph", torchlens.visualization.show_model_graph),
    ("show_backward_graph", torchlens.visualization.show_backward_graph),
    ("load_intervention_spec", torchlens.io.load_intervention_spec),
]


class _TinyModel(nn.Module):
    """Small deterministic model for API compatibility checks."""

    def __init__(self) -> None:
        """Initialize the toy network."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the toy network."""

        return self.net(x)


@pytest.fixture
def small_model() -> _TinyModel:
    """Return a deterministic tiny model."""

    torch.manual_seed(0)
    return _TinyModel()


@pytest.fixture
def small_input() -> torch.Tensor:
    """Return a deterministic tiny input."""

    torch.manual_seed(1)
    return torch.randn(1, 3)


@pytest.mark.parametrize(("old_name", "module_name", "new_name"), OBJECT_ALIAS_CASES)
def test_object_aliases_warn_and_return_new_object(
    old_name: str, module_name: str, new_name: str
) -> None:
    """Deprecated object aliases should warn and return the canonical object."""

    module = __import__(module_name, fromlist=[new_name])
    with pytest.warns(DeprecationWarning):
        old_obj = getattr(torchlens, old_name)
    assert old_obj is getattr(module, new_name)


@pytest.mark.parametrize(("old_name", "new_func"), WRAPPER_CASES)
def test_wrapper_signatures_preserved(old_name: str, new_func: Any) -> None:
    """``functools.wraps`` wrappers should expose canonical shipped signatures."""

    old_func = getattr(torchlens, old_name)
    assert inspect.signature(old_func) == inspect.signature(new_func)


def test_validate_forward_pass_positional_random_seed(
    small_model: _TinyModel, small_input: torch.Tensor
) -> None:
    """Legacy positional random_seed should still bind correctly."""

    with pytest.warns(DeprecationWarning):
        result = torchlens.validate_forward_pass(small_model, small_input, None, 42)
    assert isinstance(result, bool)


def test_validate_backward_pass_positional_loss_fn(
    small_model: _TinyModel, small_input: torch.Tensor
) -> None:
    """Legacy positional loss_fn should still bind as the fourth argument."""

    def loss_fn(output: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss for backward validation."""

        return output.sum()

    with pytest.warns(DeprecationWarning):
        result = torchlens.validate_backward_pass(small_model, small_input, None, loss_fn)
    assert isinstance(result, bool)


def test_validate_saved_activations_positional_random_seed(
    small_model: _TinyModel, small_input: torch.Tensor
) -> None:
    """Legacy saved-activation validator args should still bind correctly."""

    with pytest.warns(DeprecationWarning):
        result = torchlens.validate_saved_activations(small_model, small_input, None, 43)
    assert isinstance(result, bool)


def test_summary_legacy_call(small_model: _TinyModel, small_input: torch.Tensor) -> None:
    """Top-level summary should warn and return text for legacy callers."""

    with pytest.warns(DeprecationWarning):
        result = torchlens.summary(small_model, small_input, None)
    assert isinstance(result, str)


def test_show_model_graph_legacy_kwargs(
    tmp_path: Path, small_model: _TinyModel, small_input: torch.Tensor
) -> None:
    """Top-level forward graph helper should accept representative legacy kwargs."""

    outpath = tmp_path / "forward_graph"
    with pytest.warns(DeprecationWarning):
        result = torchlens.show_model_graph(
            small_model,
            small_input,
            None,
            vis_mode="unrolled",
            vis_outpath=str(outpath),
            vis_save_only=True,
            vis_fileformat="svg",
            visualization=None,
        )
    assert result is None


def test_show_backward_graph_legacy_kwargs(
    tmp_path: Path, small_model: _TinyModel, small_input: torch.Tensor
) -> None:
    """Top-level backward graph helper should accept representative legacy kwargs."""

    log = torchlens.log_forward_pass(small_model, small_input, layers_to_save="all")
    loss = log[log.output_layers[0]].activation.sum()
    log.log_backward(loss)

    def node_spec_fn(grad_fn_log: Any, default_spec: Any) -> Any:
        """Keep default grad_fn node specs."""

        del grad_fn_log
        return default_spec

    with pytest.warns(DeprecationWarning):
        result = torchlens.show_backward_graph(
            log,
            vis_outpath=str(tmp_path / "backward_graph"),
            vis_save_only=True,
            vis_fileformat="svg",
            node_spec_fn=node_spec_fn,
        )
    assert isinstance(result, str)


def test_load_intervention_spec_wrapper_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-level intervention-spec loader should warn and forward old args."""

    sentinel = object()

    def fake_load_intervention_spec(path: str) -> object:
        """Return a stable sentinel for wrapper forwarding tests."""

        assert path == "demo.tlspec"
        return sentinel

    monkeypatch.setattr(torchlens, "_moved_load_intervention_spec", fake_load_intervention_spec)
    with pytest.warns(DeprecationWarning):
        result = torchlens.load_intervention_spec("demo.tlspec")
    assert result is sentinel
