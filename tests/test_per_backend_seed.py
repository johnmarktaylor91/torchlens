"""Per-backend capture seeding keeps ``torch.manual_seed``'s exact effect (lane FX2).

Capture seeding replays ``torch.manual_seed`` backend by backend and skips only the lazy
seed of a backend this torch build does not compile in. These tests check that the
replay makes the same seed calls, in the same order, as ``torch.manual_seed`` on every
backend marked compiled (the entry points are mocked where the hardware is absent), and
that CPU randomness and a captured forward replay identically.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

import torchlens as tl
from torchlens._utils import _torch_compat, _torch_seed_backends as sb
from torchlens._utils.rng import set_random_seed

_ACCELERATORS = ("cuda", "mps", "xpu", "mtia")


@pytest.fixture
def seed_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str, int]]]:
    """Mock every accelerator seed entry point and record ``(backend, seed)`` calls."""

    calls: list[tuple[str, int]] = []
    for name in _ACCELERATORS:
        module = getattr(torch, name, None)
        if module is None:
            continue
        entry = sb._SEED_CALLS[name]

        def _record(seed: int, _name: str = name) -> None:
            calls.append((_name, int(seed)))

        monkeypatch.setattr(module, entry, _record)
        monkeypatch.setattr(module, "_is_in_bad_fork", lambda: False)

    def _record_custom(seed: int) -> None:
        calls.append(("privateuse1", int(seed)))

    monkeypatch.setattr(torch.random, "_seed_custom_device", _record_custom)
    yield calls


def _torch_calls(calls: list[tuple[str, int]], seed: int) -> list[tuple[str, int]]:
    """Return the calls ``torch.manual_seed`` makes under the current mocks."""

    calls.clear()
    torch.manual_seed(seed)
    made = list(calls)
    calls.clear()
    return made


def test_capability_flag_is_published_and_true_on_this_torch() -> None:
    """The probe reads torch's seeding code and doctor's snapshot shows the flag."""

    assert sb.HAS_PER_BACKEND_SEED is True
    assert _torch_compat.HAS_PER_BACKEND_SEED is True
    assert "HAS_PER_BACKEND_SEED" in _torch_compat._CAPABILITY_ATTRS
    assert _torch_compat.get_torch_capability_snapshot()["HAS_PER_BACKEND_SEED"] is True


def test_every_compiled_backend_gets_torch_manual_seeds_calls(
    seed_calls: list[tuple[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With every accelerator compiled in, the replay equals ``torch.manual_seed``."""

    for name in _ACCELERATORS:
        monkeypatch.setitem(sb.BACKEND_COMPILED, name, True)
    expected = _torch_calls(seed_calls, 1234)
    assert {name for name, _ in expected} >= {"cuda", "mps", "privateuse1"}
    sb.manual_seed_compiled_backends(1234)
    assert seed_calls == expected


def test_uncompiled_backends_skip_only_their_lazy_seed(
    seed_calls: list[tuple[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncompiled CUDA/XPU/MTIA backend is the only call dropped; MPS always runs."""

    for name in ("cuda", "xpu", "mtia"):
        monkeypatch.setitem(sb.BACKEND_COMPILED, name, False)
    expected = [
        call for call in _torch_calls(seed_calls, 99) if call[0] not in {"cuda", "xpu", "mtia"}
    ]
    sb.manual_seed_compiled_backends(99)
    assert seed_calls == expected


def test_bad_fork_backend_is_skipped_like_torch(
    seed_calls: list[tuple[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compiled backend in a bad fork is skipped by both torch and the replay."""

    monkeypatch.setitem(sb.BACKEND_COMPILED, "cuda", True)
    monkeypatch.setattr(torch.cuda, "_is_in_bad_fork", lambda: True)
    expected = _torch_calls(seed_calls, 5)
    assert "cuda" not in {name for name, _ in expected}
    sb.manual_seed_compiled_backends(5)
    assert seed_calls == expected


def test_real_compiled_probe_matches_torch_build() -> None:
    """The cached verdicts follow each backend's own ``_is_compiled`` probe."""

    sb.BACKEND_COMPILED.clear()
    for name in ("cuda", "xpu", "mtia"):
        module = getattr(torch, name)
        assert sb._backend_compiled(name, module) is bool(module._is_compiled())
    assert sb._backend_compiled("mps", torch.mps) is True


def test_unknown_seed_code_falls_back_to_torch_manual_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a recognised plan the call goes to ``torch.manual_seed`` itself."""

    seen: list[int] = []
    monkeypatch.setattr(sb, "_SEED_PLAN", None)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: seen.append(seed))
    sb.manual_seed_compiled_backends(17)
    assert seen == [17]


def test_cpu_generator_state_equals_torch_manual_seed() -> None:
    """The CPU default generator ends in the state ``torch.manual_seed`` leaves."""

    torch.manual_seed(2024)
    expected_state = torch.get_rng_state()
    expected = torch.randn(16)
    torch.manual_seed(1)
    sb.manual_seed_compiled_backends(2024)
    assert torch.equal(torch.get_rng_state(), expected_state)
    assert torch.equal(torch.randn(16), expected)
    set_random_seed(2024)
    assert torch.equal(torch.randn(16), expected)


class _Dropout(torch.nn.Module):
    """A forward that consumes CPU randomness at capture time."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(torch.relu(self.lin(x))) + torch.rand(8)


def test_captured_forward_replays_the_seeded_cpu_stream() -> None:
    """A capture at ``seed=s`` records what an eager forward after ``manual_seed(s)`` returns."""

    torch.manual_seed(0)
    model = _Dropout().train()
    x = torch.randn(4, 8)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(seed=31))
    try:
        captured = trace[trace.output_layers[-1]].out
        torch.manual_seed(31)
        expected = model(x)
        assert torch.equal(captured, expected)
        again = tl.trace(model, x, capture=tl.options.CaptureOptions(seed=31))
        try:
            assert torch.equal(again[again.output_layers[-1]].out, captured)
        finally:
            again.cleanup()
    finally:
        trace.cleanup()
