"""Phase 4c live hook execution tests."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._trace_state import TraceState
from torchlens.intervention.errors import LiveModeLabelError


class _ReluReturnModel(torch.nn.Module):
    """Model that exposes the returned ReLU out for comparison."""

    def __init__(self) -> None:
        """Initialize the model with no captured output."""

        super().__init__()
        self.latest: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a single ReLU out.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU out after live hooks.
        """

        self.latest = torch.relu(x)
        return self.latest


class _LinearModel(torch.nn.Module):
    """Single-module model for capture-time module selectors."""

    def __init__(self) -> None:
        """Initialize a linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear module.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Linear output.
        """

        return self.linear(x)


class _TwoModuleModel(torch.nn.Module):
    """Model with two module boundaries for selector specificity tests."""

    def __init__(self) -> None:
        """Initialize submodules."""

        super().__init__()
        self.a = torch.nn.ReLU()
        self.b = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two modules in sequence.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sigmoid of the relu output.
        """

        return self.b(self.a(x))


class _ChunkModel(torch.nn.Module):
    """Model with a tuple-output operation."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return two chunks.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Chunk outputs.
        """

        return torch.chunk(torch.relu(x), 2, dim=1)


def _zero_hook(out: torch.Tensor, *, hook: tl.HookContext) -> torch.Tensor:
    """Return a zeroed out.

    Parameters
    ----------
    out:
        Activation at the hook site.
    hook:
        Hook context.

    Returns
    -------
    torch.Tensor
        Zeroed out with matching metadata.
    """

    return out * 0


def _identity_hook(out: torch.Tensor, *, hook: tl.HookContext) -> torch.Tensor:
    """Return an out unchanged.

    Parameters
    ----------
    out:
        Activation at the hook site.
    hook:
        Hook context.

    Returns
    -------
    torch.Tensor
        Original out.
    """

    return out


@pytest.mark.smoke
def test_live_func_hook_replaces_returned_and_saved_out() -> None:
    """Live post-hooks run before saving so returned and saved outs match."""

    model = _ReluReturnModel()
    log = tl.trace(
        model,
        torch.randn(2, 3),
        intervention_ready=True,
        hooks={tl.func("relu"): _zero_hook},
    )

    relu_layer = next(layer for layer in log.layer_list if layer.func_name == "relu")

    assert log.state is TraceState.LIVE_CAPTURED
    assert model.latest is not None
    assert relu_layer.out is not None
    assert torch.equal(model.latest, relu_layer.out)
    assert torch.count_nonzero(relu_layer.out) == 0
    assert len(relu_layer.interventions) == 1
    assert relu_layer.interventions[0].timing == "post"
    assert relu_layer.interventions[0].direction == "forward"


@pytest.mark.smoke
def test_live_label_error_for_finalized_style_label() -> None:
    """Finalized postprocess labels fail loudly in live capture."""

    with pytest.raises(LiveModeLabelError, match="tl.where"):
        tl.trace(
            _ReluReturnModel(),
            torch.randn(2, 3),
            intervention_ready=True,
            hooks={tl.label("relu_4_27:2"): _identity_hook},
        )


@pytest.mark.smoke
def test_module_selector_matches_capture_time_module_context() -> None:
    """Module selectors can match live capture-time module context."""

    log = tl.trace(
        _LinearModel(),
        torch.randn(2, 3),
        intervention_ready=True,
        hooks={tl.module("linear"): _zero_hook},
    )

    hooked_layers = [layer for layer in log.layer_list if layer.interventions]

    assert hooked_layers
    assert hooked_layers[0].out is not None
    assert torch.count_nonzero(hooked_layers[0].out) == 0


def test_module_selector_does_not_overmatch_other_live_modules() -> None:
    """Live module selectors only match the requested module boundary."""

    log = tl.trace(
        _TwoModuleModel(),
        torch.tensor([-1.0, 2.0]),
        intervention_ready=True,
        hooks={tl.module("b"): _zero_hook},
    )

    hooked_labels = {layer.layer_label for layer in log.layer_list if layer.interventions}

    assert "sigmoid_1_2" in hooked_labels
    assert "relu_1_1" not in hooked_labels


def test_live_regex_and_output_at_selectors_execute() -> None:
    """Live regex and output-path selectors are executable hook selectors."""

    regex_log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        intervention_ready=True,
        hooks={tl.regex("relu"): _zero_hook},
    )
    chunk_log = tl.trace(
        _ChunkModel(),
        torch.randn(2, 4),
        intervention_ready=True,
        hooks={tl.output_at(1): _zero_hook},
    )

    assert any(layer.interventions for layer in regex_log.layer_list if layer.func_name == "relu")
    hooked_chunk_paths = {
        layer.container_path for layer in chunk_log.layer_list if layer.interventions
    }
    assert len(hooked_chunk_paths) == 1


@pytest.mark.smoke
def test_raw_label_where_and_in_module_selectors_work_at_capture_time() -> None:
    """Raw labels, predicates, and module containment selectors resolve live."""

    raw_log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        intervention_ready=True,
    )
    raw_label = next(
        layer._layer_label_raw for layer in raw_log.layer_list if layer.func_name == "relu"
    )

    label_log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        intervention_ready=True,
        hooks={tl.label(raw_label): _zero_hook},
    )
    where_log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        intervention_ready=True,
        hooks={tl.where(lambda p: p.func_name == "relu"): _zero_hook},
    )
    in_module_log = tl.trace(
        _LinearModel(),
        torch.randn(2, 3),
        intervention_ready=True,
        hooks={tl.in_module("linear"): _zero_hook},
    )

    assert any(layer.interventions for layer in label_log.layer_list)
    assert any(layer.interventions for layer in where_log.layer_list)
    assert any(layer.interventions for layer in in_module_log.layer_list)


@pytest.mark.smoke
def test_no_hooks_preserves_pristine_run_state() -> None:
    """Intervention-ready capture without hooks stays pristine."""

    log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        intervention_ready=True,
    )

    assert log.state is TraceState.PRISTINE


@pytest.mark.smoke
def test_live_replacement_metadata_matches_saved_out() -> None:
    """Hook replacement refreshes tensor metadata and saved-out flags."""

    log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        intervention_ready=True,
        hooks={tl.func("relu"): _zero_hook},
    )
    relu_layer = next(layer for layer in log.layer_list if layer.func_name == "relu")

    assert relu_layer.out is not None
    assert relu_layer.has_saved_activation is True
    assert relu_layer.shape == tuple(relu_layer.out.shape)
    assert relu_layer.dtype == relu_layer.out.dtype
    assert relu_layer.activation_memory == relu_layer.out.nelement() * relu_layer.out.element_size()
