"""Safe one-call execution (A4) and lazy-module entry (A9).

Lane A07 (megasprint 2026-08-27). Spec: trilabs/summary/MEMO.md build items 5-6.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

import torchlens as tl


def _state_dict_digest(model: nn.Module) -> str:
    """Byte digest of the full state_dict (params + persistent buffers)."""

    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().cpu().contiguous().view(-1).numpy().tobytes())
    return hasher.hexdigest()


class _BNStack(nn.Module):
    """Conv + BatchNorm stack whose running stats mutate on any train forward."""

    def __init__(self) -> None:
        """Initialize conv/BN pairs."""

        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack."""

        return self.bn2(self.conv2(self.bn1(self.conv1(x))))


@pytest.mark.smoke
def test_one_call_summary_never_mutates_the_model() -> None:
    """State dict bit-identical across tl.summary on a TRAIN-mode BN model (A4)."""

    model = _BNStack()
    model.train()
    before = _state_dict_digest(model)
    tl.summary(model, torch.randn(4, 3, 8, 8))
    after = _state_dict_digest(model)
    assert before == after
    # Training flags restored: the model is still in train mode afterwards.
    assert model.training
    assert model.bn1.training


@pytest.mark.heavy
def test_one_call_summary_never_mutates_resnet18() -> None:
    """The memo pin: train-mode resnet18 state dict survives tl.summary (A4)."""

    torchvision_models = pytest.importorskip("torchvision.models")
    model = torchvision_models.resnet18()
    model.train()
    before = _state_dict_digest(model)
    tl.summary(model, torch.randn(2, 3, 32, 32))
    after = _state_dict_digest(model)
    assert before == after
    assert model.training


@pytest.mark.smoke
def test_one_call_summary_restores_rng_state() -> None:
    """RNG streams are bit-identical with and without an interleaved summary (A4)."""

    model = _BNStack()  # constructed BEFORE seeding: init draws stay out of the stream

    torch.manual_seed(1234)
    interleaved_input = torch.randn(4, 3, 8, 8)
    tl.summary(model, interleaved_input)
    resumed = torch.rand(8)

    torch.manual_seed(1234)
    _ = torch.randn(4, 3, 8, 8)
    expected = torch.rand(8)
    assert torch.equal(resumed, expected)


@pytest.mark.smoke
def test_one_call_summary_explicit_train_mode_updates_buffers() -> None:
    """execution_mode='train' is the explicit opt-in that mutates BN stats (A4)."""

    model = _BNStack()
    model.train()
    before = model.bn1.running_mean.clone()
    text = tl.summary(model, torch.randn(4, 3, 8, 8), execution_mode="train")
    assert not torch.equal(before, model.bn1.running_mean)
    assert "train mode (explicit" in text


@pytest.mark.smoke
def test_one_call_summary_discloses_execution() -> None:
    """The default summary discloses eval + no_grad + restoration (A4)."""

    text = tl.summary(_BNStack(), torch.randn(2, 3, 8, 8))
    assert "eval mode" in text
    assert "no_grad" in text
    assert "restored" in text


@pytest.mark.smoke
def test_one_call_summary_mode_refusals_teach() -> None:
    """Unknown execution/grad modes refuse typed with the valid choices (A4)."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError, match="'eval', 'train', or 'same'") as mode_exc:
        tl.summary(_BNStack(), torch.randn(2, 3, 8, 8), execution_mode="fast")
    assert mode_exc.value.fields["code"] == "summary_execution_mode_invalid"
    with pytest.raises(InvalidArgumentError, match="'off' or 'same'") as grad_exc:
        tl.summary(_BNStack(), torch.randn(2, 3, 8, 8), grad_mode="on")
    assert grad_exc.value.fields["code"] == "summary_grad_mode_invalid"


class _LazyUnused(nn.Module):
    """Model declaring a lazy submodule that never runs."""

    def __init__(self) -> None:
        """Declare one real and one never-called lazy submodule."""

        super().__init__()
        self.used = nn.Linear(8, 8, bias=False)
        self.never = nn.LazyLinear(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run only the real submodule."""

        return self.used(x)


@pytest.mark.smoke
def test_lazy_linear_traces_and_finalizes_inventory() -> None:
    """nn.LazyLinear materializes during the ONE captured forward (A9)."""

    model = nn.Sequential(nn.LazyLinear(16), nn.ReLU())
    log = tl.trace(model, torch.randn(2, 8), capture=tl.options.CaptureOptions(layers_to_save=None))
    try:
        shapes = {pl.address: tuple(pl.shape) for pl in log.param_logs}
        assert shapes == {"0.weight": (16, 8), "0.bias": (16,)}
        counts = {pl.address: pl.num_params for pl in log.param_logs}
        assert counts == {"0.weight": 128, "0.bias": 16}
        assert all(int(pl.param_memory) > 0 for pl in log.param_logs)
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_lazy_linear_summary_one_call() -> None:
    """tl.summary works on a lazy model with no priming forward (A9)."""

    text = tl.summary(nn.Sequential(nn.LazyLinear(16), nn.ReLU()), torch.randn(2, 8))
    assert isinstance(text, str)
    assert "Linear" in text


@pytest.mark.smoke
def test_lazy_linear_graph_stays_clean() -> None:
    """Lazy init machinery never pollutes the executed-op graph (A9)."""

    log = tl.trace(
        nn.Sequential(nn.LazyLinear(16), nn.ReLU()),
        torch.randn(2, 8),
        capture=tl.options.CaptureOptions(layers_to_save=None),
    )
    try:
        assert log.num_ops == 2
        labels = [lay.layer_type for lay in log.layer_list]
        assert labels == ["input", "linear", "relu", "output"]
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_never_materialized_lazy_param_tolerated() -> None:
    """A declared-but-never-run lazy param stays zero-geometry, no crash (A9)."""

    model = _LazyUnused()
    log = tl.trace(model, torch.randn(2, 8), capture=tl.options.CaptureOptions(layers_to_save=None))
    try:
        by_address = {pl.address: pl for pl in log.param_logs}
        assert by_address["used.weight"].num_params == 64
        never = by_address["never.weight"]
        assert never.num_params == 0
        assert tuple(never.shape) == ()
    finally:
        log.cleanup()
