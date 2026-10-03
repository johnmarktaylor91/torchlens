"""A06 capture-options truth: ``stop_after`` wired through the halt engine.

WALKTHROUGH list-A row 10 (third clause): ``stop_after=`` was a universal
silent no-op whose docstring claimed pluck support ("validated"), while the
trace path raised ``NotImplementedError`` and pluck accepted-and-ignored it.
The wired semantics are brainpipe D-16's INCLUSIVE spelling: the site
compiles into the existing save-then-halt engine, the named site IS captured,
and the never-fired policy splits by provenance (selector-shaped refuses
typed, exploratory callable warns, ambient context-manager site warns).
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.options import CaptureOptions


class ThreeStep(nn.Module):
    """fc1 -> relu -> fc2, one op per module plus the activation."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _labels(trace: object) -> list[str]:
    return [op.layer_label for op in trace]  # type: ignore[attr-defined]


def test_stop_after_module_address_halts_inclusively() -> None:
    """A module-address string halts at that module's exit, keeping its ops."""

    log = tl.trace(ThreeStep(), torch.randn(2, 4), capture=CaptureOptions(stop_after="fc1"))
    assert log.halted is True
    assert log.outcome.status.name == "HALTED"
    labels = _labels(log)
    assert any(label.startswith("linear_1") for label in labels), labels
    # Nothing past the fc1 boundary ran: no relu, no second linear.
    assert not any(label.startswith("relu") for label in labels), labels
    assert not any(label.startswith("linear_2") for label in labels), labels


def test_stop_after_selector_halts_inclusively_with_payload() -> None:
    """A live selector site is captured (payload included) before halting."""

    log = tl.trace(
        ThreeStep(), torch.randn(2, 4), capture=CaptureOptions(stop_after=tl.func("relu"))
    )
    assert log.halted is True
    relu_ops = [op for op in log if op.layer_label.startswith("relu")]
    assert len(relu_ops) == 1
    assert relu_ops[0].out is not None
    assert not any(label.startswith("linear_2") for label in _labels(log))


def test_stop_after_callable_halts() -> None:
    """A bare callable predicate halts through the same engine."""

    log = tl.trace(
        ThreeStep(),
        torch.randn(2, 4),
        capture=CaptureOptions(stop_after=lambda ctx: ctx.func_name == "relu"),
    )
    assert log.halted is True
    assert not any(label.startswith("linear_2") for label in _labels(log))


def test_stop_after_finalized_label_refuses_typed() -> None:
    """Finalized postprocess labels do not exist live and refuse typed."""

    with pytest.raises(Exception) as excinfo:
        tl.trace(ThreeStep(), torch.randn(2, 4), capture=CaptureOptions(stop_after="relu_1_2"))
    assert excinfo.value.fields["code"] == "stop_after_site_not_live"


def test_stop_after_selector_never_fired_refuses_typed() -> None:
    """A selector-shaped site that never fires is a typo-shaped wrong result."""

    with pytest.raises(Exception) as excinfo:
        tl.trace(
            ThreeStep(),
            torch.randn(2, 4),
            capture=CaptureOptions(stop_after="nonexistent.module"),
        )
    assert excinfo.value.fields["code"] == "stop_after_never_fired"


@pytest.mark.smoke
def test_stop_after_callable_never_fired_warns_with_ledger() -> None:
    """An exploratory callable that never fires warns coded and ledgers."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(
            ThreeStep(),
            torch.randn(2, 4),
            capture=CaptureOptions(stop_after=lambda ctx: False),
        )
    assert log.halted is False
    codes = [
        getattr(entry.message, "fields", {}).get("code")
        for entry in caught
        if isinstance(entry.message, TorchLensWarning)
    ]
    assert "stop_after_never_fired_callable" in codes
    slots = [row["slot"] for row in log.annotations.get("unmatched_capture_selectors", [])]
    assert "stop_after" in slots


def test_stop_after_type_invalid_refuses_typed() -> None:
    """Unsupported site types refuse typed at entry."""

    with pytest.raises(Exception) as excinfo:
        tl.trace(ThreeStep(), torch.randn(2, 4), capture=CaptureOptions(stop_after=1234))
    assert excinfo.value.fields["code"] == "stop_after_type_invalid"


def test_stop_after_conflicts_with_halt_typed() -> None:
    """stop_after= and halt= compile into one slot and refuse together."""

    with pytest.raises(Exception) as excinfo:
        tl.trace(
            ThreeStep(),
            torch.randn(2, 4),
            halt=tl.func("relu"),
            capture=CaptureOptions(stop_after="fc1"),
        )
    assert excinfo.value.fields["code"] == "stop_after_halt_conflict"


def test_stop_after_refuses_chunked_capture() -> None:
    """Chunked fan-out and a single stop frontier cannot combine."""

    from torchlens.intervention.errors import ChunkedForwardConfigError

    with pytest.raises(ChunkedForwardConfigError) as excinfo:
        tl.trace(
            ThreeStep(),
            torch.randn(8, 4),
            chunk_size=4,
            capture=CaptureOptions(stop_after="fc1"),
        )
    assert excinfo.value.fields["code"] == "stop_after_chunked_conflict"


def test_stop_after_ambient_context_manager_reaches_trace_and_pluck() -> None:
    """The experimental context manager arms torch captures in its block."""

    with tl.experimental.stop_after("fc1"):
        log = tl.trace(ThreeStep(), torch.randn(2, 4))
    assert log.halted is True
    assert not any(label.startswith("relu") for label in _labels(log))

    with tl.experimental.stop_after("fc1"):
        out = tl.pluck(ThreeStep(), torch.randn(2, 4), "linear_1_1")
    assert tuple(out.shape) == (2, 8)


def test_stop_after_ambient_never_fired_warns_never_refuses() -> None:
    """Ambient provenance is exploratory: never-fired warns, even selector-shaped."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with tl.experimental.stop_after("nonexistent.module"):
            log = tl.trace(ThreeStep(), torch.randn(2, 4))
    assert log.halted is False
    codes = [
        getattr(entry.message, "fields", {}).get("code")
        for entry in caught
        if isinstance(entry.message, TorchLensWarning)
    ]
    assert "stop_after_never_fired_callable" in codes


@pytest.mark.smoke
def test_stop_after_explicit_wins_over_ambient() -> None:
    """CaptureOptions.stop_after beats the ambient context-manager site."""

    with tl.experimental.stop_after("nonexistent.module"):
        log = tl.trace(ThreeStep(), torch.randn(2, 4), capture=CaptureOptions(stop_after="fc1"))
    assert log.halted is True


@pytest.mark.smoke
def test_stop_after_cache_key_distinct_from_halt(tmp_path) -> None:
    """A completed halt= capture must not satisfy a stop_after= request."""

    model, x = ThreeStep(), torch.randn(2, 4)
    first = tl.trace(
        model,
        x,
        capture=CaptureOptions(cache=True, cache_dir=str(tmp_path), stop_after=tl.func("relu")),
    )
    assert first.halted is True
    second = tl.trace(
        model,
        x,
        halt=tl.func("relu"),
        capture=CaptureOptions(cache=True, cache_dir=str(tmp_path)),
    )
    assert second.capture_cache_key != first.capture_cache_key


@pytest.mark.real_model
def test_stop_after_gpt2_block_frontier() -> None:
    """R0 realism row: stop after transformer block 0 on the REAL GPT-2 class."""

    pytest.importorskip("transformers")
    from tests.real_model.r0.families import build_gpt2

    model = build_gpt2("eager")
    generator = torch.Generator().manual_seed(20260826)
    input_ids = torch.randint(0, 512, (1, 8), generator=generator)
    log = tl.trace(model, input_ids, capture=CaptureOptions(stop_after="transformer.h.0"))
    assert log.halted is True
    module_addresses = {
        module_address for op in log for module_address in (getattr(op, "modules", ()) or ())
    }
    assert any(str(address).startswith("transformer.h.0") for address in module_addresses)
    assert not any(str(address).startswith("transformer.h.1") for address in module_addresses)
