"""Forward validation's weight copies, its restore, and its copy-failure report.

Forward validation runs the model on deep copies and restores the live model
from a snapshot. These tests pin three things:

* how many full copies of the weights are resident during validation (a
  deterministic count of distinct storages of a sentinel-shaped weight, sampled
  at every forward of the model or one of its copies);
* that the restore still puts back parameters, buffers, plain attributes and
  modes bit for bit, on the copy path and on the live-model fallback;
* that a failed deep copy is reported by name (exception type and message) in
  a coded warning and in the refusal, not only as unsupported attributes.
"""

from __future__ import annotations

import copy
import gc
import warnings
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._validation_state_snapshot import (
    restore_snapshot_if_changed,
    snapshot_state_for_restore,
)
from torchlens.validation.diagnostics import last_validation_failure

_SENTINEL_SHAPE = (89, 97)
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Raw bytes of a dense tensor, for bit-for-bit comparison (0-dim included)."""

    return tensor.reshape(-1).view(torch.uint8)


def _sentinel_storages(device_type: str) -> int:
    """Count distinct storages holding a sentinel-shaped tensor on one device type."""

    storages = set()
    for obj in gc.get_objects():
        if (
            isinstance(obj, torch.Tensor)
            and tuple(obj.shape) == _SENTINEL_SHAPE
            and obj.device.type == device_type
        ):
            storages.add(obj.untyped_storage().data_ptr())
    return len(storages)


class _SentinelModel(nn.Module):
    """Small model whose first weight has a shape nothing else in the run has."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(97, 89)
        self.bn = nn.BatchNorm1d(89)
        self.b = nn.Linear(89, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.bn(self.a(x))))


def _copies_per_forward(device: str) -> tuple[bool, list[dict[str, int]]]:
    """Validate the sentinel model; return the verdict and, per forward, copies per device.

    The forwards run in a fixed order: the ground-truth forward (on the
    ground-truth copy), the TorchLens capture (on the replay copy), then the
    pristine re-trace (on the replay copy, while the first trace is alive).
    Each count includes the model's own weight.
    """

    torch.manual_seed(0)
    model = _SentinelModel().to(device)
    samples: list[dict[str, int]] = []
    device_types = {"cpu", torch.device(device).type}

    def census(module: nn.Module, inputs: Any) -> None:
        samples.append({kind: _sentinel_storages(kind) for kind in device_types})

    model.a.register_forward_pre_hook(census)
    verdict = tl.validate(model, torch.randn(4, 97, device=device), scope="forward")
    return bool(verdict), samples


def test_forward_validation_weight_copies_on_a_cpu_model() -> None:
    """The ground-truth copy is released and the replay copy reuses the live snapshot.

    Base: 2, 4 and 5 copies beyond the model at the ground-truth, capture and
    re-trace forwards (the live snapshot; the ground-truth copy, never
    released; the replay copy and its own snapshot; then the first trace's
    saved-argument clone of the weight). Now 2, 2 and 2: on a CPU model the
    host snapshot shares the device with the one copy, and the capture keeps
    the replay copy's parameters by reference instead of cloning them.
    """

    verdict, samples = _copies_per_forward("cpu")
    assert verdict is True
    beyond_model = [sample["cpu"] - 1 for sample in samples]
    assert len(beyond_model) == 3, beyond_model
    assert beyond_model[0] <= 2 and beyond_model[1] <= 2, beyond_model
    assert beyond_model[2] <= 2, beyond_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator device")
def test_forward_validation_weight_copies_on_an_accelerator() -> None:
    """On an accelerator only the deep copy shares the card with the model.

    The restore snapshots live in host memory, and the capture keeps the
    replay copy's parameters by reference, so every forward sees exactly one
    copy beyond the model on the device.
    """

    verdict, samples = _copies_per_forward("cuda")
    assert verdict is True
    on_device = [sample["cuda"] - 1 for sample in samples]
    assert on_device == [1, 1, 1], on_device


@pytest.mark.parametrize("device", _DEVICES)
def test_snapshot_holds_accelerator_tensors_on_the_host_bit_for_bit(device: str) -> None:
    """Plain dense accelerator entries live on the host; values and source devices are exact."""

    torch.manual_seed(1)
    model = _SentinelModel().to(device).to(torch.bfloat16)
    snapshot = snapshot_state_for_restore(model)
    live = model.state_dict()
    assert list(snapshot.tensors) == list(live)
    for name, saved in snapshot.tensors.items():
        assert saved.device.type == "cpu"
        assert snapshot.source_devices[name] == live[name].device
        assert saved.dtype == live[name].dtype
        assert torch.equal(_bytes(saved), _bytes(live[name].detach().cpu()))
        assert saved.untyped_storage().data_ptr() != live[name].untyped_storage().data_ptr()


@pytest.mark.parametrize("device", _DEVICES)
def test_restore_writes_only_when_changed_and_then_bit_for_bit(device: str) -> None:
    """Unchanged state keeps its version counters; changed state comes back exactly."""

    torch.manual_seed(2)
    model = _SentinelModel().to(device)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    snapshot = snapshot_state_for_restore(model)
    versions = {name: tensor._version for name, tensor in model.state_dict(keep_vars=True).items()}
    restore_snapshot_if_changed(model, snapshot, lambda m, sd: m.load_state_dict(sd))
    after_versions = {
        name: tensor._version for name, tensor in model.state_dict(keep_vars=True).items()
    }
    assert after_versions == versions

    with torch.no_grad():
        model.a.weight.add_(0.25)
        model.bn.running_mean.fill_(3.0)
        model.bn.num_batches_tracked.add_(7)
    restore_snapshot_if_changed(model, snapshot, lambda m, sd: m.load_state_dict(sd))
    for name, tensor in model.state_dict().items():
        assert tensor.device == before[name].device
        assert torch.equal(tensor, before[name]), name
    assert model.a.weight.device.type == torch.device(device).type


class _MutatingModel(nn.Module):
    """Forward mutates a parameter, a buffer, a plain attribute and a module mode."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 4)
        self.drop = nn.Dropout(0.5)
        self.register_buffer("count", torch.zeros(()))
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.lin.bias.add_(0.5)
            self.count.add_(1.0)
        self.calls += 1
        self.drop.train(not self.drop.training)
        return self.drop(self.lin(x))


def _model_state(model: nn.Module) -> dict[str, Any]:
    """Every restorable piece of the model: tensor bytes, plain attribute, modes."""

    return {
        "tensors": {
            name: (tensor.dtype, tensor.device, tensor.detach().cpu().clone())
            for name, tensor in model.state_dict().items()
        },
        "calls": model.calls,
        "modes": [module.training for module in model.modules()],
    }


def _assert_same_state(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left["calls"] == right["calls"]
    assert left["modes"] == right["modes"]
    assert left["tensors"].keys() == right["tensors"].keys()
    for name, (dtype, device, value) in left["tensors"].items():
        other_dtype, other_device, other_value = right["tensors"][name]
        assert (dtype, device) == (other_dtype, other_device), name
        assert torch.equal(_bytes(value), _bytes(other_value)), name


@pytest.fixture
def failing_model_deepcopy(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[type]]:
    """Make ``copy.deepcopy`` raise a CUDA out-of-memory error for listed model classes."""

    failing: list[type] = []
    real_deepcopy = copy.deepcopy

    def deepcopy(obj: Any, memo: Any = None, _nil: Any = None) -> Any:
        if failing and isinstance(obj, tuple(failing)):
            raise torch.cuda.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 14.18 GiB (GPU 0; 44.53 GiB total)"
            )
        return real_deepcopy(obj, memo)

    monkeypatch.setattr(copy, "deepcopy", deepcopy)
    yield failing


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("copy_fails", [False, True], ids=["copy", "live-fallback"])
def test_validated_capture_restores_every_restorable_kind(
    device: str, copy_fails: bool, failing_model_deepcopy: list[type]
) -> None:
    """Parameters, buffers, plain attributes and modes come back bit for bit.

    The forward mutates each kind. On the copy path the live model is never
    run; on the live-model fallback (both deep copies fail) it runs three times
    and must be restored. Validation seeds the global random generator by
    design, so random state is checked as a function of the seed: two
    validations with one seed leave the same generator state.
    """

    if copy_fails:
        failing_model_deepcopy.append(_MutatingModel)
    rng_states = []
    for _ in range(2):
        torch.manual_seed(3)
        model = _MutatingModel().to(device)
        before = _model_state(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl.validate(model, torch.randn(2, 6, device=device), scope="forward", random_seed=11)
        _assert_same_state(before, _model_state(model))
        rng_states.append(torch.get_rng_state())
    assert torch.equal(rng_states[0], rng_states[1])


class _Config:
    """Stand-in for a Hugging Face ``config`` object (an unsupported plain attribute)."""

    hidden_size = 6


class _ConfiguredModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.lin = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_deepcopy_out_of_memory_is_named_in_the_warning_and_the_refusal(
    failing_model_deepcopy: list[type],
) -> None:
    """A copy that fails for lack of memory is reported as that, with a remedy that runs."""

    class OomModel(_ConfiguredModel):
        pass

    failing_model_deepcopy.append(OomModel)
    model = OomModel()
    x = torch.randn(2, 6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verdict = tl.validate(model, x, scope="forward")
    assert verdict is False

    coded = [w.message for w in caught if getattr(w.message, "fields", {}).get("code")]
    copy_warnings = [m for m in coded if m.fields["code"] == "validation_model_copy_failed"]
    assert len(copy_warnings) == 1
    warning = copy_warnings[0]
    assert isinstance(warning, RuntimeWarning)
    text = str(warning)
    assert "could not deepcopy the model (OomModel)" in text
    assert "OutOfMemoryError: CUDA out of memory. Tried to allocate 14.18 GiB" in text
    assert "warning_contract.md" in text
    assert "tl.validate(model.to('cpu'), inputs, scope='forward')" in warning.fields["remedy"]

    refusal = [
        str(w.message)
        for w in caught
        if str(w.message).startswith("TorchLens validation cannot prove model-state")
    ]
    assert len(refusal) == 1
    assert refusal[0].startswith(
        "TorchLens validation cannot prove model-state restoration after deepcopy failed "
        "with OutOfMemoryError: CUDA out of memory"
    )
    assert "tl.validate(model.to('cpu'), inputs, scope='forward')" in refusal[0]
    failure = last_validation_failure()
    assert failure is not None
    assert failure.extra.get("code") == "validation_model_copy_failed"
    assert "OutOfMemoryError" in failure.message

    # The remedy's spelling runs: with room for the copy, validation succeeds.
    failing_model_deepcopy.clear()
    assert tl.validate(model.to("cpu"), x, scope="forward") is True


def test_other_deepcopy_failures_are_named_with_their_own_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that cannot be deep-copied at all is told how to make it copyable."""

    class Unpicklable(_ConfiguredModel):
        def __deepcopy__(self, memo: Any) -> Any:
            raise TypeError("cannot pickle '_thread.lock' object")

    model = Unpicklable()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert tl.validate(model, torch.randn(2, 6), scope="forward") is False
    copy_warnings = [
        w.message
        for w in caught
        if getattr(w.message, "fields", {}).get("code") == "validation_model_copy_failed"
    ]
    assert len(copy_warnings) == 1
    assert "TypeError: cannot pickle '_thread.lock' object" in str(copy_warnings[0])
    assert "define __deepcopy__" in copy_warnings[0].fields["remedy"]
    refusal = [
        str(w.message)
        for w in caught
        if str(w.message).startswith("TorchLens validation cannot prove model-state")
    ]
    assert "deepcopy failed with TypeError: cannot pickle" in refusal[0]

    # The remedy's spelling runs once the class can be copied.
    def copyable(self: Any, memo: Any) -> Any:
        clone = Unpicklable.__new__(Unpicklable)
        clone.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return clone

    monkeypatch.setattr(Unpicklable, "__deepcopy__", copyable)
    assert tl.validate(model, torch.randn(2, 6), scope="forward") is True
