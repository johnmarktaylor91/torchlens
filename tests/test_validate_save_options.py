"""``tl.validate`` save options: ``output_device`` and ``save_budget``.

The validator capture saves every activation; these keywords give it the
capture API's own save options (same names, values, and defaults as
``CaptureOptions``) so a GPU model can validate with its activations held in
host memory. The CUDA tests at the bottom run only where a GPU is available.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
import torch

import torchlens as tl
import torchlens._user_public_impls as public_impls
import torchlens.user_funcs as user_funcs
from torchlens.errors import InvalidArgumentError, SaveBudgetExceededError
from torchlens.options import CaptureOptions


class _SmallNet(torch.nn.Module):
    """Small deterministic model with a conv, a nonlinearity, and a linear head."""

    def __init__(self) -> None:
        """Initialize the layers."""

        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4 * 6 * 6, 3)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network.

        Parameters
        ----------
        x:
            Input batch of shape ``(N, 2, 6, 6)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(N, 3)``.
        """

        self.forward_calls += 1
        y = torch.relu(self.conv(x))
        return self.fc(y.flatten(1)) * 2.0 + 1.0


def _net_and_input(device: str = "cpu") -> tuple[_SmallNet, torch.Tensor]:
    """Return a seeded model/input pair on ``device``.

    Parameters
    ----------
    device:
        Torch device string.

    Returns
    -------
    tuple[_SmallNet, torch.Tensor]
        Model and input.
    """

    torch.manual_seed(0)
    return _SmallNet().to(device), torch.randn(2, 2, 6, 6, device=device)


@pytest.mark.smoke_cells("test_output_device_cpu_on_cpu_model_validates[forward]")
@pytest.mark.parametrize("scope", ["forward", "saved"])
def test_output_device_cpu_on_cpu_model_validates(scope: str) -> None:
    """``output_device="cpu"`` on a CPU model is a no-op move and validates."""

    model, x = _net_and_input()
    assert tl.validate(model, x, scope=scope, output_device="cpu") is True


def test_intervention_scope_honors_save_options() -> None:
    """The intervention scope's forward validation runs under the save options."""

    model, x = _net_and_input()
    report = tl.validate(model, x, scope="intervention", output_device="cpu", save_budget=None)
    assert report.invariance is True
    with pytest.raises(SaveBudgetExceededError):
        tl.validate(model, x, scope="intervention", save_budget=64)


@pytest.mark.parametrize("scope", ["forward", "saved"])
def test_too_small_save_budget_raises_instead_of_passing(scope: str) -> None:
    """A budget the validator capture cannot fit raises the capture's own error."""

    model, x = _net_and_input()
    with pytest.raises(SaveBudgetExceededError):
        tl.validate(model, x, scope=scope, save_budget=64)


@pytest.mark.parametrize("budget", [None, 1 << 30, 1.0])
def test_unbounded_or_large_save_budget_validates(budget: Any) -> None:
    """``None`` (no budget), a large byte cap, or the full fraction all pass."""

    model, x = _net_and_input()
    assert tl.validate(model, x, scope="forward", save_budget=budget) is True


def _capture_error(**capture_kwargs: Any) -> InvalidArgumentError:
    """Return the error ``tl.trace`` raises for the given capture options.

    Parameters
    ----------
    **capture_kwargs:
        ``CaptureOptions`` fields.

    Returns
    -------
    InvalidArgumentError
        The capture-side error.
    """

    model, x = _net_and_input()
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(model, x, capture=CaptureOptions(**capture_kwargs))
    return excinfo.value


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_device": "tpu"},
        {"output_device": "cuda:0"},
        {"save_budget": "lots"},
        {"save_budget": True},
        {"save_budget": 0},
        {"save_budget": 1.5},
    ],
)
def test_invalid_values_raise_the_capture_errors_before_any_forward(
    kwargs: dict[str, Any],
) -> None:
    """Invalid values raise the same typed error as the capture, before any forward."""

    expected = _capture_error(**kwargs)
    model, x = _net_and_input()
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.validate(model, x, scope="forward", **kwargs)
    assert type(excinfo.value) is type(expected)
    assert expected.fields.get("code") is not None
    assert excinfo.value.fields.get("code") == expected.fields.get("code")
    assert model.forward_calls == 0


@pytest.mark.parametrize("scope", ["backward", "receptive_field"])
@pytest.mark.parametrize("kwargs", [{"output_device": "cpu"}, {"save_budget": 64}])
def test_scopes_without_a_saving_validator_capture_refuse(
    scope: str, kwargs: dict[str, Any]
) -> None:
    """Backward and receptive-field validation refuse non-default save options."""

    model, x = _net_and_input()
    with pytest.raises(TypeError, match="only valid for scope='forward', 'saved'"):
        tl.validate(model, x, scope=scope, **kwargs)
    assert model.forward_calls == 0


def test_default_call_forwards_no_new_keywords() -> None:
    """A default ``tl.validate`` reaches the validators with exactly the old keywords."""

    model, x = _net_and_input()
    with mock.patch.object(
        public_impls, "_validate_forward_pass_torch", return_value=True
    ) as torch_entry:
        assert tl.validate(model, x, scope="forward") is True
    assert "output_device" not in torch_entry.call_args.kwargs
    assert "save_budget" not in torch_entry.call_args.kwargs


def test_both_validator_captures_receive_the_save_options() -> None:
    """The first capture and its reproducibility re-trace run under the save options."""

    real_runner = user_funcs._run_model_and_save_specified_outs
    seen: list[tuple[Any, Any, bool]] = []

    def recording_runner(*args: Any, **kwargs: Any) -> Any:
        seen.append(
            (kwargs.get("output_device"), kwargs.get("save_budget"), kwargs["save_arg_values"])
        )
        return real_runner(*args, **kwargs)

    model, x = _net_and_input()
    with mock.patch.object(user_funcs, "_run_model_and_save_specified_outs", recording_runner):
        assert tl.validate(model, x, scope="forward", output_device="cpu", save_budget=None)
    assert seen == [("cpu", None, True), ("cpu", None, False)]

    seen.clear()
    with mock.patch.object(user_funcs, "_run_model_and_save_specified_outs", recording_runner):
        assert tl.validate(model, x, scope="forward")
    assert seen == [("same", "auto", True), ("same", "auto", False)]


def test_non_torch_backend_refuses_non_default_save_options() -> None:
    """Only the torch validator capture honors the save options."""

    from torchlens.backends import BackendUnsupportedError

    fake_spec = mock.Mock()
    fake_spec.name = "jax"
    model, x = _net_and_input()
    with mock.patch.object(public_impls, "resolve_backend_spec", return_value=fake_spec):
        with pytest.raises(BackendUnsupportedError, match="save options"):
            public_impls.validate_forward_pass(model, x, output_device="cpu")
        public_impls.validate_forward_pass(model, x)
    assert "output_device" not in fake_spec.validate_entry.call_args.kwargs
    fake_spec.validate_entry.assert_called_once()


def test_replay_device_helpers_are_identity_when_devices_match() -> None:
    """On the default path both alignment helpers return their input object."""

    from torchlens.validation._replay_device import (
        align_output_to_saved_device,
        align_parent_to_slot_device,
    )

    parent = torch.ones(3)
    args = {"args": [torch.zeros(3), [torch.zeros(1), torch.zeros(3)]], "kwargs": {}}
    assert align_parent_to_slot_device(args, "args", 0, parent) is parent
    assert align_parent_to_slot_device(args, "args", (1, 1), parent) is parent
    assert align_parent_to_slot_device(args, "args", 7, parent) is parent
    assert align_parent_to_slot_device(args, "args", 0, 3.0) == 3.0
    recomputed = torch.ones(3)
    assert align_output_to_saved_device(recomputed, torch.ones(3)) is recomputed
    assert align_output_to_saved_device(recomputed, None) is recomputed


def test_replay_device_helpers_move_across_devices() -> None:
    """A device mismatch moves the value; meta slots are left to their hooks."""

    from torchlens.validation._replay_device import (
        align_output_to_saved_device,
        align_parent_to_slot_device,
    )

    moved = align_output_to_saved_device(torch.ones(3), torch.empty(3, device="meta"))
    assert moved.device.type == "meta"
    meta_slot = {"args": [torch.empty(3, device="meta")], "kwargs": {}}
    parent = torch.ones(3)
    assert align_parent_to_slot_device(meta_slot, "args", 0, parent) is parent


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@_CUDA
@pytest.mark.parametrize("scope", ["forward", "saved", "intervention"])
def test_cuda_model_validates_with_host_saved_activations(scope: str) -> None:
    """A CUDA model validates with ``output_device="cpu"`` on every saving scope."""

    model, x = _net_and_input("cuda")
    result = tl.validate(model, x, scope=scope, output_device="cpu")
    assert bool(result) if scope != "intervention" else result.invariance


@_CUDA
def test_cuda_validator_capture_holds_activations_on_host() -> None:
    """The validator-equivalent capture keeps saved outs on the host under ``"cpu"``."""

    model, x = _net_and_input("cuda")
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save="all", save_arg_values=True, output_device="cpu"),
    )
    devices = {op.out.device.type for op in trace.layer_list if op.out is not None}
    assert devices == {"cpu"}


def _assert_corrupted_activation_fails(device: str) -> None:
    """Corrupt the validator capture's saved ``relu`` payload; validate must fail.

    Parameters
    ----------
    device:
        Device of the model and input.
    """

    real_runner = user_funcs._run_model_and_save_specified_outs

    def corrupting_runner(*args: Any, **kwargs: Any) -> Any:
        trace = real_runner(*args, **kwargs)
        if kwargs["save_arg_values"]:
            relu = next(op for op in trace.layer_list if op.func_name == "relu")
            assert relu.out.device.type == "cpu"
            relu.out = relu.out + 1.0
        return trace

    model, x = _net_and_input(device)
    with mock.patch.object(user_funcs, "_run_model_and_save_specified_outs", corrupting_runner):
        with pytest.warns(tl.errors.TorchLensWarning, match="tl.validate FAILED"):
            assert tl.validate(model, x, scope="forward", output_device="cpu") is False


def test_replay_still_detects_a_corrupted_saved_activation() -> None:
    """The replay tripwire keeps failing a corrupted payload under the save options."""

    _assert_corrupted_activation_fails("cpu")


@_CUDA
def test_cuda_replay_still_detects_a_corrupted_saved_activation() -> None:
    """With host-held activations of a CUDA model the tripwire still fires."""

    _assert_corrupted_activation_fails("cuda")


@_CUDA
@pytest.mark.parametrize("model_name", ["GetAndSetItem", "InPlaceFuncs", "BatchNormModel"])
def test_cuda_example_models_keep_their_verdict_with_host_activations(model_name: str) -> None:
    """Posthoc exemption proofs (``__setitem__`` coverage) also work across devices."""

    import example_models

    torch.manual_seed(0)
    model = getattr(example_models, model_name)().to("cuda")
    x = torch.rand(6, 3, 24, 24, device="cuda")
    assert tl.validate(model, x, scope="forward", random_seed=0) is True
    assert tl.validate(model, x, scope="forward", random_seed=0, output_device="cpu") is True


@_CUDA
def test_cuda_tensor_nanequal_compares_values_across_devices() -> None:
    """``tensor_nanequal`` compares a CUDA and a CPU tensor by value instead of raising."""

    from torchlens.utils.tensor_utils import tensor_nanequal

    t = torch.tensor([1.0, float("nan"), 3.0])
    assert tensor_nanequal(t.cuda(), t)
    assert tensor_nanequal(t, t.cuda())
    assert not tensor_nanequal(t.cuda(), t + 1)
    assert not tensor_nanequal(t.cuda(), torch.tensor([1.0, 2.0, 3.0]))
