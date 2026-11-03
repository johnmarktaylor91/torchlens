"""The one RNG restore door and the surfaces that go through it.

``RngStreamsRestore`` snapshots every global stream TorchLens seeds (torch CPU, live CUDA
devices, Python ``random``, NumPy) and puts them back exactly once, on request, at context
exit or on error. ``tl.trace`` already returns the caller's streams through it; the walk
ruling (JMT, 2026-10-31) extends the same promise to ``tl.validate``: a backward or layer
gradient validation seeds every engine with the validation seed and must leave the caller's
streams where they were. Before the door, the process stayed reseeded with the validation
seed.
"""

from __future__ import annotations

import random

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils.rng import RngStreamsRestore, restores_rng_streams

np = pytest.importorskip("numpy")


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _draw() -> tuple[list[float], float, float]:
    """One draw from each global stream; equal draws mean equal stream positions."""
    return (torch.rand(3).tolist(), random.random(), float(np.random.rand()))


@pytest.mark.smoke
def test_door_restores_every_stream_at_context_exit() -> None:
    _seed_all(7)
    expected = _draw()
    _seed_all(7)
    with RngStreamsRestore():
        torch.rand(50)
        random.random()
        np.random.rand(5)
    assert _draw() == expected


def test_door_restores_on_error() -> None:
    _seed_all(7)
    expected = _draw()
    _seed_all(7)
    with pytest.raises(RuntimeError, match="inside"), RngStreamsRestore():
        torch.rand(50)
        random.random()
        raise RuntimeError("inside")
    assert _draw() == expected


def test_door_restores_exactly_once() -> None:
    """A second ``restore`` is a no-op: it never rewinds draws made after the first."""
    _seed_all(7)
    door = RngStreamsRestore()
    torch.rand(10)
    door.restore()
    first = torch.rand(3)
    door.restore()
    second = torch.rand(3)
    assert not torch.equal(first, second)


def test_decorator_restores_around_the_call() -> None:
    @restores_rng_streams
    def consume(n: int) -> int:
        torch.rand(n)
        random.random()
        return n

    _seed_all(3)
    expected = _draw()
    _seed_all(3)
    assert consume(25) == 25
    assert _draw() == expected


def test_validate_backward_returns_the_callers_streams() -> None:
    """``tl.validate`` comes back through the door the way ``tl.trace`` does.

    The validation seed (0) differs from the caller's (11), so a leaked reseed shows up
    as different draws after the call.
    """
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    x = torch.randn(2, 4)
    _seed_all(0)
    reseeded = _draw()
    _seed_all(11)
    expected = _draw()
    assert reseeded != expected
    _seed_all(11)
    assert tl.validate(model, x, scope="backward", seed=0)
    assert _draw() == expected


def test_validate_forward_returns_the_callers_streams() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    x = torch.randn(2, 4)
    _seed_all(11)
    expected = _draw()
    _seed_all(11)
    assert tl.validate(model, x, scope="forward", seed=0)
    assert _draw() == expected


def _noisy() -> nn.Module:
    """A train-mode dropout model: its forward draws from the torch stream on every run."""
    return nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2)).train()


def test_validate_of_a_model_that_draws_returns_the_callers_streams() -> None:
    """A validation whose ground-truth and replay runs themselves draw random numbers
    (train-mode dropout) still hands the caller's streams back unchanged."""
    model = _noisy()
    x = torch.randn(2, 4)
    _seed_all(11)
    expected = _draw()
    _seed_all(11)
    assert tl.validate(model, x, scope="forward", seed=0)
    assert _draw() == expected


def test_lower_level_forward_entry_returns_the_callers_streams() -> None:
    """``torchlens.validation.validate_forward_pass`` goes through the same door."""
    from torchlens.validation import validate_forward_pass

    model = _noisy()
    x = torch.randn(2, 4)
    _seed_all(11)
    expected = _draw()
    _seed_all(11)
    assert validate_forward_pass(model, x, seed=0)
    assert _draw() == expected


def test_lower_level_backward_entry_returns_the_callers_streams() -> None:
    """``torchlens.validation.validate_backward_pass`` goes through the same door."""
    from torchlens.validation import validate_backward_pass

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    x = torch.randn(2, 4)
    _seed_all(11)
    expected = _draw()
    _seed_all(11)
    assert validate_backward_pass(model, x, seed=0)
    assert _draw() == expected


def test_trace_method_validation_returns_the_callers_streams() -> None:
    """``Trace.validate_forward_pass`` replays every op under its recorded RNG state and
    leaves the caller's streams where they were."""
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    x = torch.randn(2, 4)
    # Replay needs the saved activations and arguments; a default capture
    # keeps neither and reports an unverified status instead of a verdict.
    trace = tl.trace(
        model, x, capture=tl.options.CaptureOptions(layers_to_save="all", save_args=True)
    )
    ground_truth = model(x).detach().clone()
    _seed_all(11)
    expected = _draw()
    _seed_all(11)
    assert trace.validate_forward_pass([ground_truth]) is True
    assert _draw() == expected
