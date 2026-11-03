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


def test_stop_after_finalized_label_refuses_typed() -> None:
    """Finalized postprocess labels do not exist live and refuse typed."""

    with pytest.raises(Exception) as excinfo:
        tl.trace(ThreeStep(), torch.randn(2, 4), stop_after="relu_1_2")
    assert excinfo.value.fields["code"] == "stop_after_site_not_live"


def test_stop_after_type_invalid_refuses_typed() -> None:
    """Unsupported site types refuse typed at entry."""

    with pytest.raises(Exception) as excinfo:
        tl.trace(ThreeStep(), torch.randn(2, 4), stop_after=1234)
    assert excinfo.value.fields["code"] == "stop_after_type_invalid"


def test_stop_after_refuses_chunked_capture() -> None:
    """Chunked fan-out and a single stop frontier cannot combine."""

    from torchlens.intervention.errors import ChunkedForwardConfigError

    with pytest.raises(ChunkedForwardConfigError) as excinfo:
        tl.trace(
            ThreeStep(),
            torch.randn(8, 4),
            stop_after="fc1",
            capture=CaptureOptions(chunk_size=4),
        )
    assert excinfo.value.fields["code"] == "stop_after_chunked_conflict"


@pytest.mark.smoke
def test_stop_after_ambient_never_fired_warns_never_refuses() -> None:
    """Ambient provenance is exploratory: never-fired warns, even selector-shaped."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with tl.experimental.stop_after("nonexistent.module"):
            log = tl.trace(ThreeStep(), torch.randn(2, 4))
    assert log.is_stopped is False
    codes = [
        getattr(entry.message, "fields", {}).get("code")
        for entry in caught
        if isinstance(entry.message, TorchLensWarning)
    ]
    assert "stop_after_never_fired_callable" in codes


def test_stop_after_explicit_wins_over_ambient() -> None:
    """CaptureOptions.stop_after beats the ambient context-manager site."""

    with tl.experimental.stop_after("nonexistent.module"):
        log = tl.trace(ThreeStep(), torch.randn(2, 4), stop_after="fc1")
    assert log.is_stopped is True


@pytest.mark.real_model
def test_stop_after_gpt2_block_frontier() -> None:
    """R0 realism row: stop after transformer block 0 on the REAL GPT-2 class."""

    pytest.importorskip("transformers")
    from tests.real_model.r0.families import build_gpt2

    model = build_gpt2("eager")
    generator = torch.Generator().manual_seed(20260826)
    input_ids = torch.randint(0, 512, (1, 8), generator=generator)
    log = tl.trace(model, input_ids, stop_after="transformer.h.0")
    assert log.is_stopped is True
    module_addresses = {
        module_address for op in log for module_address in (getattr(op, "modules", ()) or ())
    }
    assert any(str(address).startswith("transformer.h.0") for address in module_addresses)
    assert not any(str(address).startswith("transformer.h.1") for address in module_addresses)
