"""Speed defaults for gradient capture (AGT-2132).

``tl.record`` installs gradient hooks only when a backward is requested up
front, and a capture entered with autograd off implies ``inference_only=True``.
Both defaults refuse a later backward with a message that names the cause.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import ConfigurationError, RecorderStateError
from torchlens.fastlog import Recorder


class _TinyMlp(nn.Module):
    """Two linear layers around a ReLU."""

    def __init__(self) -> None:
        """Initialize the layers."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.net(x)


def _model_and_input() -> tuple[_TinyMlp, torch.Tensor]:
    """Return a deterministic model and input.

    Returns
    -------
    tuple[_TinyMlp, torch.Tensor]
        Model and input batch.
    """

    torch.manual_seed(0)
    return _TinyMlp(), torch.randn(3, 4)


def _keep_relu(ctx: object) -> bool:
    """Keep the ReLU output.

    Parameters
    ----------
    ctx:
        Record context.

    Returns
    -------
    bool
        Whether the op is the ReLU.
    """

    return getattr(ctx, "label", None) == "relu_1"


def _has_grad_hooks(tensor: torch.Tensor) -> bool:
    """Return whether TorchLens registered a gradient hook on ``tensor``.

    Parameters
    ----------
    tensor:
        Model output tensor.

    Returns
    -------
    bool
        Whether the tensor carries a backward hook.
    """

    return bool(getattr(tensor, "_backward_hooks", None))


def test_record_installs_no_grad_hooks_by_default() -> None:
    """A plain record leaves the output without gradient hooks."""

    model, x = _model_and_input()
    out, recording = tl.record(model, x, save=_keep_relu, return_output=True)
    assert out.grad_fn is not None
    assert not _has_grad_hooks(out)
    assert recording.outcome.inference_only is False


@pytest.mark.parametrize(
    "opt_in",
    [
        {"backward_ready": True},
        {"save_grads": True},
        {"save_grads": lambda ctx: True},
        {"default_grad": True},
    ],
    ids=["backward_ready", "save_grads_true", "save_grads_predicate", "default_grad"],
)
def test_record_opt_in_restores_grad_hooks_and_backward(opt_in: dict[str, object]) -> None:
    """Each backward opt-in installs the hooks and log_backward records gradients."""

    model, x = _model_and_input()
    out, recording = tl.record(model, x, save=_keep_relu, return_output=True, **opt_in)
    assert _has_grad_hooks(out)
    recording.log_backward(out.sum(), save_grads=True)
    assert recording.grad_records


@pytest.mark.parametrize("value", [None, False])
def test_record_save_grads_off_values_do_not_opt_in(value: object) -> None:
    """``save_grads=None`` and ``save_grads=False`` request no backward."""

    model, x = _model_and_input()
    out, _ = tl.record(model, x, save=_keep_relu, return_output=True, save_grads=value)
    assert not _has_grad_hooks(out)


@pytest.mark.smoke
def test_record_log_backward_without_opt_in_refuses_naming_it() -> None:
    """A backward on a record without the opt-in refuses instead of recording nothing."""

    model, x = _model_and_input()
    out, recording = tl.record(model, x, save=_keep_relu, return_output=True)
    with pytest.raises(RecorderStateError) as exc_info:
        recording.log_backward(out.sum(), save_grads=True)
    assert exc_info.value.fields["code"] == "recording_backward_not_requested"
    message = str(exc_info.value)
    assert "backward_ready=True" in message
    assert "save_grads=" in message
    assert recording.grad_records == []


def test_recorder_log_backward_without_opt_in_refuses() -> None:
    """The with-block Recorder door refuses the same way."""

    model, x = _model_and_input()
    with Recorder(model, save=_keep_relu) as recorder:
        out = recorder.log(x)
        with pytest.raises(RecorderStateError) as exc_info:
            recorder.log_backward(out.sum())
    assert exc_info.value.fields["code"] == "recording_backward_not_requested"


@pytest.mark.parametrize("grad_off", [torch.no_grad, torch.inference_mode])
def test_record_under_no_grad_implies_inference_only(grad_off: type) -> None:
    """A record with autograd off reports inference_only and refuses backward naming why."""

    model, x = _model_and_input()
    with grad_off():
        out, recording = tl.record(model, x, save=_keep_relu, return_output=True)
    assert recording.outcome.inference_only is True
    assert recording._recording_state.runtime_trace.inference_only is True
    with pytest.raises(RecorderStateError) as exc_info:
        recording.log_backward(out.sum())
    assert exc_info.value.fields["code"] == "recording_backward_inference_only"
    message = str(exc_info.value)
    assert "torch.no_grad()" in message
    assert "inference_only=True" in message
    assert "backward_ready=True" in message


def test_record_under_no_grad_with_backward_ready_is_not_inference_only() -> None:
    """An explicit backward request keeps the no_grad record out of inference_only."""

    model, x = _model_and_input()
    with torch.no_grad():
        recording = tl.record(model, x, save=_keep_relu, backward_ready=True)
    assert recording.outcome.inference_only is False


def test_record_keeps_saved_payload_values() -> None:
    """Skipping gradient hooks changes no recorded activation value."""

    model, x = _model_and_input()
    plain = tl.record(model, x, save=_keep_relu)
    hooked = tl.record(model, x, save=_keep_relu, backward_ready=True)
    plain_outs = [record.ram_payload for record in plain if record.ram_payload is not None]
    hooked_outs = [record.ram_payload for record in hooked if record.ram_payload is not None]
    assert len(plain_outs) == len(hooked_outs) >= 1
    for plain_out, hooked_out in zip(plain_outs, hooked_outs, strict=True):
        assert torch.equal(plain_out, hooked_out.detach())


@pytest.mark.parametrize("grad_off", [torch.no_grad, torch.inference_mode])
def test_trace_under_no_grad_implies_inference_only(grad_off: type) -> None:
    """A tl.trace entered with autograd off reports inference_only=True."""

    model, x = _model_and_input()
    with grad_off():
        trace = tl.trace(model, x)
    try:
        assert trace.inference_only is True
        assert trace.outcome.inference_only is True
    finally:
        trace.cleanup()


def test_trace_under_no_grad_backward_refusal_names_the_cause() -> None:
    """The inference_only backward refusal explains the no_grad implication."""

    model, x = _model_and_input()
    with torch.no_grad():
        trace = tl.trace(model, x)
    try:
        loss = model(x).sum()
        with pytest.raises(ConfigurationError, match="autograd graph was discarded") as exc_info:
            trace.log_backward(loss)
        assert "torch.no_grad()" in str(exc_info.value)
    finally:
        trace.cleanup()


def test_trace_with_grad_enabled_is_not_inference_only() -> None:
    """The default grad-enabled trace keeps inference_only=False."""

    model, x = _model_and_input()
    trace = tl.trace(model, x)
    try:
        assert trace.inference_only is False
    finally:
        trace.cleanup()


@pytest.mark.parametrize(
    "capture",
    [
        {"inference_only": False},
        {"backward_ready": True},
        {"save_grads": True},
        {"intervention_ready": True},
    ],
    ids=["explicit_false", "backward_ready", "save_grads", "intervention_ready"],
)
def test_trace_under_no_grad_explicit_options_keep_inference_only_off(
    capture: dict[str, object],
) -> None:
    """Explicit inference_only=False or a backward option overrides the implication."""

    model, x = _model_and_input()
    with torch.no_grad():
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(**capture))
    try:
        assert trace.inference_only is False
    finally:
        trace.cleanup()


def test_trace_under_no_grad_keeps_saved_values() -> None:
    """The implied inference_only capture saves the same activations as before."""

    model, x = _model_and_input()
    with torch.no_grad():
        implied = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
        plain = tl.trace(
            model,
            x,
            capture=tl.options.CaptureOptions(layers_to_save="all", inference_only=False),
        )
    try:
        assert implied.inference_only is True
        assert plain.inference_only is False
        for label in plain.layer_labels:
            if plain[label].has_saved_activation:
                assert torch.equal(implied[label].out, plain[label].out)
    finally:
        implied.cleanup()
        plain.cleanup()


class _GradInsideForward(nn.Module):
    """A forward that turns grad back on itself (energy / implicit-layer style)."""

    def __init__(self) -> None:
        """Initialize the layers."""

        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model with grad enabled regardless of the caller.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Model output, carrying autograd history.
        """

        with torch.enable_grad():
            return torch.relu(self.lin(x))


def _grad_inside_model_and_input() -> tuple[_GradInsideForward, torch.Tensor]:
    """Return a deterministic enable_grad-inside model and input.

    Returns
    -------
    tuple[_GradInsideForward, torch.Tensor]
        Model and input batch.
    """

    torch.manual_seed(0)
    return _GradInsideForward(), torch.randn(3, 4)


def test_trace_under_no_grad_with_enable_grad_inside_keeps_backward() -> None:
    """A forward that re-enables grad keeps its graph, so the implication does not apply."""

    model, x = _grad_inside_model_and_input()
    with torch.no_grad():
        trace = tl.trace(model, x)
    try:
        assert trace.inference_only is False
        assert trace.outcome.inference_only is False
        loss = trace[trace.output_layers[0]].out.sum()
        assert loss.grad_fn is not None
        trace.log_backward(loss, save_grads=True)
        assert model.lin.weight.grad is not None
        assert any(layer.grads for layer in trace.layer_list)
    finally:
        trace.cleanup()


def test_trace_explicit_inference_only_with_enable_grad_inside_stays_on() -> None:
    """An explicit inference_only=True is never cleared, even if the forward kept a graph."""

    model, x = _grad_inside_model_and_input()
    with torch.no_grad():
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(inference_only=True))
    try:
        assert trace.inference_only is True
        assert trace.outcome.inference_only is True
        loss = model(x).sum()
        with pytest.raises(ConfigurationError, match="autograd graph was discarded"):
            trace.log_backward(loss)
    finally:
        trace.cleanup()


def test_record_under_no_grad_with_enable_grad_inside_is_not_inference_only() -> None:
    """The record door clears the implied flag too and refuses only for the missing opt-in."""

    model, x = _grad_inside_model_and_input()
    with torch.no_grad():
        out, recording = tl.record(model, x, save=_keep_relu, return_output=True)
    assert out.grad_fn is not None
    assert recording.outcome.inference_only is False
    with pytest.raises(RecorderStateError) as exc_info:
        recording.log_backward(out.sum())
    assert exc_info.value.fields["code"] == "recording_backward_not_requested"


def test_record_default_grad_false_does_not_opt_in() -> None:
    """``default_grad=False`` requests no backward: no hooks, and log_backward refuses."""

    model, x = _model_and_input()
    out, recording = tl.record(model, x, save=_keep_relu, return_output=True, default_grad=False)
    assert not _has_grad_hooks(out)
    with pytest.raises(RecorderStateError) as exc_info:
        recording.log_backward(out.sum(), save_grads=True)
    assert exc_info.value.fields["code"] == "recording_backward_not_requested"
