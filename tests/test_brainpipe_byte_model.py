"""Retained-bytes accounting oracle (F20, brainpipe memo D-17).

``Trace.saved_activation_memory`` must equal the summed REAL physical bytes
of the activations capture actually retained, in all four transform/raw
cells. The pre-fix aggregate summed the raw ``activation_memory`` field for
every saved op, over-reporting by the full reduction factor whenever a
transform dropped the raw payload (87x, 121x, and 190x observed on real
models -- inconsistent in direction with what the sweep must use).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl

pytestmark = pytest.mark.smoke  # measured <0.5s per test (W051-GATE, AUD-CODE 0.1)


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())


def _pool(t: torch.Tensor) -> torch.Tensor:
    """64x reduction: batch-mean pooling."""

    return t.mean(dim=0)


def _real_retained_bytes(log: object) -> int:
    """Sum physical storage bytes of retained payloads, each storage once."""

    seen: set[tuple[int, int]] = set()
    total = 0
    for op in log.ops:  # type: ignore[attr-defined]
        for payload in (op.out, getattr(op, "transformed_out", None)):
            if not isinstance(payload, torch.Tensor):
                continue
            storage = payload.untyped_storage()
            key = (storage.data_ptr(), int(storage.nbytes()))
            if key in seen:
                continue
            seen.add(key)
            total += int(storage.nbytes())
    return total


@pytest.mark.parametrize(
    ("transform", "keep_raw"),
    [
        (None, True),
        (_pool, True),
        (_pool, False),
    ],
    ids=["raw-only", "transform-and-raw", "transform-drop-raw"],
)
def test_saved_activation_memory_equals_real_retained_bytes(transform, keep_raw) -> None:
    """The accounting oracle: reported == summed real retained bytes."""

    save_options = tl.options.SaveOptions(
        activation_transform=transform,
        save_raw_activations=keep_raw,
    )
    log = tl.trace(_model(), torch.randn(64, 256), save=save_options)
    reported = int(log.saved_activation_memory)
    real = _real_retained_bytes(log)
    assert reported == real, (
        f"saved_activation_memory reports {reported} bytes but capture "
        f"physically retained {real} bytes (D-17 byte-model regression)"
    )
    assert real > 0


def test_reduced_configuration_reports_reduced_bytes() -> None:
    """The sweep's reduce-at-capture cell reports the REDUCED total.

    The 64x-pooling drop-raw configuration must report ~1/64th of the raw
    total, not the raw total (the exact defect shape the memo measured).
    """

    raw_log = tl.trace(_model(), torch.randn(64, 256))
    reduced_log = tl.trace(
        _model(),
        torch.randn(64, 256),
        save=tl.options.SaveOptions(activation_transform=_pool, save_raw_activations=False),
    )
    assert int(reduced_log.saved_activation_memory) * 16 < int(raw_log.saved_activation_memory)


def test_dedup_shared_payloads_charge_once() -> None:
    """Identity-deduped payloads count one physical storage, not N copies."""

    class Repeat(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = torch.relu(x)
            # Identity ops reuse the same source tensor; dedup shares payloads.
            for _ in range(3):
                y = y.reshape(y.shape)
            return y

    log = tl.trace(Repeat(), torch.randn(128, 128))
    reported = int(log.saved_activation_memory)
    real = _real_retained_bytes(log)
    assert reported == real
