"""Training pattern tests for train-mode capture."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torchlens as tl
from .conftest import MultiTapModel, TeacherStudentPair, TinyResnetWithProbe, TwoLayerMlp


class SharedFrozenModule(nn.Module):
    """Model that calls one frozen module twice before a trainable head."""

    def __init__(self) -> None:
        """Initialize shared and trainable layers."""

        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.head = nn.Linear(4, 1)
        for param in self.shared.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reuse the shared module twice in one forward pass."""

        first = self.shared(x)
        second = self.shared(x + 1)
        return self.head(first + second)


class BranchMismatchModel(nn.Module):
    """Model whose graph changes based on input sign."""

    def __init__(self) -> None:
        """Initialize the branch layer."""

        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a sign-dependent graph."""

        out = self.linear(x)
        if bool(x.sum() > 0):
            out = torch.relu(out)
        return out


def _assert_params_require_grad(module: nn.Module, expected: bool) -> None:
    """Assert all parameters in a module have the expected requires_grad value."""

    assert all(param.requires_grad is expected for param in module.parameters())


def _first_payload_by_layer_type(
    recording: tl.fastlog.Recording,
    layer_type: str,
) -> torch.Tensor:
    """Return the first RAM payload matching a fastlog layer type."""

    return next(
        record.ram_payload
        for record in recording
        if record.ctx.layer_type == layer_type and record.ram_payload is not None
    )


def _assert_any_grad(module: nn.Module) -> None:
    """Assert at least one parameter in ``module`` has a grad."""

    assert any(param.grad is not None for param in module.parameters())


def _assert_no_grad(module: nn.Module) -> None:
    """Assert every parameter in ``module`` has no grad."""

    assert all(param.grad is None for param in module.parameters())


def test_train_mode_preserves_user_requires_grad(
    tiny_resnet_with_probe: TinyResnetWithProbe,
) -> None:
    """train_mode preserves frozen backbone params during and after capture."""

    observed_backbone_states: list[bool] = []

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        """Record backbone requires_grad state during the forward pass."""

        observed_backbone_states.append(
            all(
                param.requires_grad is False
                for param in tiny_resnet_with_probe.backbone.parameters()
            )
        )

    handle = tiny_resnet_with_probe.backbone.register_forward_pre_hook(hook)
    try:
        trace = tl.trace(
            tiny_resnet_with_probe,
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            random_seed=0,
        )
    finally:
        handle.remove()

    assert observed_backbone_states == [True]
    _assert_params_require_grad(tiny_resnet_with_probe.backbone, False)
    _assert_params_require_grad(tiny_resnet_with_probe.probe, True)

    saved = trace[trace.output_layers[0]].out
    tiny_resnet_with_probe.zero_grad(set_to_none=True)
    saved.sum().backward()

    assert all(param.grad is None for param in tiny_resnet_with_probe.backbone.parameters())
    assert all(param.grad is not None for param in tiny_resnet_with_probe.probe.parameters())
    trace.cleanup()


@pytest.mark.smoke
def test_aux_loss_slow(two_layer_mlp: TwoLayerMlp) -> None:
    """Pattern A: slow capture supports an auxiliary loss on an intermediate out."""

    trace = tl.trace(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    hidden = trace["relu_1_2"].out
    output = trace[trace.output_layers[0]].out

    two_layer_mlp.zero_grad(set_to_none=True)
    (output.pow(2).mean() + 0.25 * hidden.pow(2).mean()).backward()

    _assert_any_grad(two_layer_mlp.fc1)
    _assert_any_grad(two_layer_mlp.fc2)
    trace.cleanup()


@pytest.mark.smoke
def test_aux_loss_replay(two_layer_mlp: TwoLayerMlp) -> None:
    """Pattern A: replay capture supports an auxiliary loss on an intermediate out."""

    trace = tl.trace(two_layer_mlp, torch.randn(3, 4), random_seed=0)
    trace.save_new_outs(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    hidden = trace["relu_1_2"].out
    output = trace[trace.output_layers[0]].out

    two_layer_mlp.zero_grad(set_to_none=True)
    (output.pow(2).mean() + 0.25 * hidden.pow(2).mean()).backward()

    _assert_any_grad(two_layer_mlp.fc1)
    _assert_any_grad(two_layer_mlp.fc2)
    trace.cleanup()


@pytest.mark.smoke
def test_aux_loss_fastlog(two_layer_mlp: TwoLayerMlp) -> None:
    """Pattern A: fastlog train_mode supports an auxiliary out loss."""

    output, recording = tl.fastlog.record(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        return_output=True,
    )
    hidden = _first_payload_by_layer_type(recording, "relu")

    two_layer_mlp.zero_grad(set_to_none=True)
    (output.pow(2).mean() + 0.25 * hidden.pow(2).mean()).backward()

    _assert_any_grad(two_layer_mlp.fc1)
    _assert_any_grad(two_layer_mlp.fc2)


def test_probe_frozen_backbone_slow(tiny_resnet_with_probe: TinyResnetWithProbe) -> None:
    """Pattern B: slow capture trains only the probe on a frozen backbone."""

    trace = tl.trace(
        tiny_resnet_with_probe,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    saved = trace[trace.output_layers[0]].out

    tiny_resnet_with_probe.zero_grad(set_to_none=True)
    F.cross_entropy(saved, torch.tensor([0, 1, 0])).backward()

    _assert_no_grad(tiny_resnet_with_probe.backbone)
    _assert_any_grad(tiny_resnet_with_probe.probe)
    trace.cleanup()


def test_probe_frozen_backbone_replay(tiny_resnet_with_probe: TinyResnetWithProbe) -> None:
    """Pattern B: replay capture trains only the probe on a frozen backbone."""

    trace = tl.trace(tiny_resnet_with_probe, torch.randn(3, 4), random_seed=0)
    trace.save_new_outs(
        tiny_resnet_with_probe,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    saved = trace[trace.output_layers[0]].out

    tiny_resnet_with_probe.zero_grad(set_to_none=True)
    F.cross_entropy(saved, torch.tensor([0, 1, 0])).backward()

    _assert_no_grad(tiny_resnet_with_probe.backbone)
    _assert_any_grad(tiny_resnet_with_probe.probe)
    trace.cleanup()


def test_probe_frozen_backbone_fastlog(tiny_resnet_with_probe: TinyResnetWithProbe) -> None:
    """Pattern B: fastlog capture trains only the probe on a frozen backbone."""

    output, recording = tl.fastlog.record(
        tiny_resnet_with_probe,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        return_output=True,
    )
    assert any(record.ram_payload is not None for record in recording)

    tiny_resnet_with_probe.zero_grad(set_to_none=True)
    F.cross_entropy(output, torch.tensor([0, 1, 0])).backward()

    _assert_no_grad(tiny_resnet_with_probe.backbone)
    _assert_any_grad(tiny_resnet_with_probe.probe)


def test_multi_tap_loss_slow(multi_tap_model: MultiTapModel) -> None:
    """Pattern C: slow capture supports losses from multiple saved taps."""

    trace = tl.trace(
        multi_tap_model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    hidden = trace["relu_1_2"].out
    output = trace[trace.output_layers[1]].out

    multi_tap_model.zero_grad(set_to_none=True)
    (hidden.square().mean() + output.square().mean()).backward()

    _assert_any_grad(multi_tap_model.fc1)
    _assert_any_grad(multi_tap_model.fc2)
    trace.cleanup()


def test_multi_tap_loss_fastlog(multi_tap_model: MultiTapModel) -> None:
    """Pattern C: fastlog supports losses from multiple saved taps."""

    outputs, recording = tl.fastlog.record(
        multi_tap_model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        return_output=True,
    )
    hidden = _first_payload_by_layer_type(recording, "relu")

    multi_tap_model.zero_grad(set_to_none=True)
    (hidden.square().mean() + outputs[1].square().mean()).backward()

    _assert_any_grad(multi_tap_model.fc1)
    _assert_any_grad(multi_tap_model.fc2)


def test_distillation_slow(teacher_student_pair: TeacherStudentPair) -> None:
    """Pattern D: slow capture supports frozen-teacher distillation."""

    trace = tl.trace(
        teacher_student_pair,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    teacher = trace[trace.output_layers[0]].out
    student = trace[trace.output_layers[1]].out

    teacher_student_pair.zero_grad(set_to_none=True)
    F.mse_loss(student, teacher).backward()

    _assert_no_grad(teacher_student_pair.teacher)
    _assert_any_grad(teacher_student_pair.student)
    trace.cleanup()


def test_distillation_fastlog(teacher_student_pair: TeacherStudentPair) -> None:
    """Pattern D: fastlog supports frozen-teacher distillation."""

    (teacher, student), recording = tl.fastlog.record(
        teacher_student_pair,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        return_output=True,
    )
    assert any(record.ram_payload is not None for record in recording)

    teacher_student_pair.zero_grad(set_to_none=True)
    F.mse_loss(student, teacher).backward()

    _assert_no_grad(teacher_student_pair.teacher)
    _assert_any_grad(teacher_student_pair.student)


def test_multi_pass_recorder_fastlog(two_layer_mlp: TwoLayerMlp) -> None:
    """Pattern E: Recorder supports repeated train-mode ops in a training loop."""

    optimizer = torch.optim.SGD(two_layer_mlp.parameters(), lr=0.01)
    losses: list[float] = []

    with tl.fastlog.Recorder(two_layer_mlp, train_mode=True) as recorder:
        for _step in range(4):
            optimizer.zero_grad(set_to_none=True)
            output = recorder.log(torch.randn(3, 4, requires_grad=True))
            loss = output.square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    payloads = [
        record.ram_payload
        for record in recorder.recording
        if record.ctx.layer_type == "relu" and record.ram_payload is not None
    ]

    assert len(losses) == 4
    assert recorder.recording.n_ops == 4
    assert len({id(payload) for payload in payloads}) == 4
    assert all(payload.grad_fn is not None for payload in payloads)
    _assert_any_grad(two_layer_mlp)


def test_train_mode_shared_module_requires_grad() -> None:
    """train_mode preserves requires_grad on shared modules called repeatedly."""

    model = SharedFrozenModule()
    observed_shared_states: list[bool] = []

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        """Record shared-module requires_grad state during each call."""

        observed_shared_states.append(
            all(param.requires_grad is False for param in model.shared.parameters())
        )

    handle = model.shared.register_forward_pre_hook(hook)
    try:
        trace = tl.trace(
            model,
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            random_seed=0,
        )
    finally:
        handle.remove()

    assert observed_shared_states == [True, True]
    _assert_params_require_grad(model.shared, False)
    _assert_params_require_grad(model.head, True)

    saved = trace[trace.output_layers[0]].out
    model.zero_grad(set_to_none=True)
    saved.sum().backward()

    assert all(param.grad is None for param in model.shared.parameters())
    assert all(param.grad is not None for param in model.head.parameters())
    trace.cleanup()


def test_save_new_outs_train_mode_inherits() -> None:
    """save_new_outs inherits train_mode by default."""

    model = SharedFrozenModule()
    trace = tl.trace(
        model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )

    trace.save_new_outs(
        model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=None,
        random_seed=0,
    )

    saved = trace[trace.output_layers[0]].out
    assert saved.grad_fn is not None
    assert trace.train_mode is True
    trace.cleanup()


def test_save_new_outs_train_mode_overrides() -> None:
    """Explicit save_new_outs train_mode overrides detach flags temporarily."""

    model = SharedFrozenModule()
    trace = tl.trace(
        model,
        torch.randn(3, 4, requires_grad=True),
        detach_saved_tensorss=True,
        random_seed=0,
    )
    original_layer_flags = [layer.detach_saved_tensors for layer in trace]

    trace.save_new_outs(
        model,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )

    saved = trace[trace.output_layers[0]].out
    assert saved.grad_fn is not None
    assert trace.detach_saved_tensorss is True
    assert trace.train_mode is False
    assert [layer.detach_saved_tensors for layer in trace] == original_layer_flags
    trace.cleanup()


def test_fastlog_train_mode_sugar_promotes_defaults() -> None:
    """fastlog train_mode promotes omitted defaults to keep-grad capture specs."""

    recording = tl.fastlog.record(
        SharedFrozenModule(),
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
    )

    grad_records = [
        record
        for record in recording
        if record.ram_payload is not None and record.ram_payload.grad_fn is not None
    ]
    assert grad_records


def test_fastlog_train_mode_explicit_default_op_false() -> None:
    """Explicit default_op=False remains disabled under train_mode sugar."""

    recording = tl.fastlog.record(
        SharedFrozenModule(),
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        default_op=False,
    )

    assert all(record.ctx.kind != "op" for record in recording)


def test_fastlog_train_mode_default_op_true_errors() -> None:
    """default_op=True contradicts train_mode keep-grad sugar."""

    with pytest.raises(tl.TrainingModeConfigError, match="default_op=True"):
        tl.fastlog.record(
            SharedFrozenModule(),
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            default_op=True,
        )


def test_fastlog_train_mode_explicit_capspec_keepgrad_false_errors() -> None:
    """CaptureSpec keep_grad=False contradicts train_mode keep-grad sugar."""

    with pytest.raises(tl.TrainingModeConfigError, match="keep_grad=False"):
        tl.fastlog.record(
            SharedFrozenModule(),
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            default_op=tl.fastlog.CaptureSpec(keep_grad=False),
        )


def test_save_new_outs_train_mode_restored_on_graph_mismatch() -> None:
    """save_new_outs restores override flags when fast-pass graph validation fails."""

    model = BranchMismatchModel()
    trace = tl.trace(
        model,
        torch.ones(2, 4, requires_grad=True),
        detach_saved_tensorss=True,
        random_seed=0,
    )
    original_layer_flags = [layer.detach_saved_tensors for layer in trace]

    def divergent_forward(self: BranchMismatchModel, x: torch.Tensor) -> torch.Tensor:
        """Run a different operation sequence from the exhaustive pass."""

        out = self.linear(x)
        return out + out

    model.forward = types.MethodType(divergent_forward, model)

    try:
        trace.save_new_outs(
            model,
            torch.ones(2, 4, requires_grad=True),
            train_mode=True,
            random_seed=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected fast-pass graph mismatch")

    assert trace.detach_saved_tensorss is True
    assert trace.train_mode is False
    assert [layer.detach_saved_tensors for layer in trace] == original_layer_flags
    trace.cleanup()
