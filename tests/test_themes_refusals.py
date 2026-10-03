"""F12 pins: the lens refusal taxonomy (themes memo section 2 item 5).

Overview/blueprint never refuse; speed/memory/compute/dims refuse on
missing EVIDENCE; transformer and sequence refuse on factual absence of
SUBJECT. Every refusal names the remedy; the teaching escape is
theme='overview'.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization import lenses


@pytest.fixture(scope="module")
def plain_log() -> Any:
    """A single-pass, attention-free trace."""

    log = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(1, 4))
    yield log
    log.cleanup()


def test_transformer_refuses_on_zero_attention(plain_log: Any) -> None:
    """Never an ordinary graph under a domain name."""

    with pytest.raises(Exception) as excinfo:
        lenses.resolve_lens(plain_log, "transformer")
    assert excinfo.value.fields["code"] == "lens_attention_subject_absent"
    assert "overview" in str(excinfo.value)


def test_sequence_refuses_on_single_pass(plain_log: Any) -> None:
    """Single pass: nothing to sequence, with the teaching escape."""

    with pytest.raises(Exception) as excinfo:
        lenses.resolve_lens(plain_log, "sequence")
    assert excinfo.value.fields["code"] == "lens_single_pass_no_sequence"
    assert "overview" in str(excinfo.value)


def test_overview_and_blueprint_never_refuse(plain_log: Any) -> None:
    """The two never-refuse rows resolve on any trace."""

    assert lenses.resolve_lens(plain_log, "overview").draw_kwargs
    assert lenses.resolve_lens(plain_log, "blueprint").draw_kwargs


def test_detail_ceiling_refusal_names_both_remedies(plain_log: Any, monkeypatch: Any) -> None:
    """debug/dims above the ceiling refuse naming module= and the override."""

    from torchlens.visualization.lenses import _resolve as resolve_module

    monkeypatch.setattr(resolve_module, "LENS_DETAIL_CEILING", 2)
    for name in ("debug", "dims"):
        with pytest.raises(Exception) as excinfo:
            lenses.resolve_lens(plain_log, name)
        assert excinfo.value.fields["code"] == "lens_above_detail_budget"
        message = str(excinfo.value)
        assert "module=" in message
        assert "collapse=" in message


@pytest.mark.smoke
def test_sequence_stack_license_degrade_is_coded_and_rendered() -> None:
    """An unlicensed stack_by='auto' degrades with the coded warning AND a
    rendered notice; the lens still resolves (SECONDARY, never refusal)."""

    class TwoLoops(nn.Module):
        """Two chained tied loops: pass indexes restart (license fails)."""

        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)

        def forward(self, x: Any) -> Any:
            for _ in range(2):
                x = torch.relu(self.a(x))
            for _ in range(2):
                x = torch.sigmoid(self.b(x))
            return x

    log = tl.trace(TwoLoops(), torch.randn(1, 4))
    try:
        import warnings as warnings_module

        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            resolution = lenses.resolve_lens(log, "sequence")
        if "stack_by" not in resolution.draw_kwargs:
            codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
            assert "lens_secondary_degraded" in codes
            assert any("stack_by degraded" in line for line in resolution.disclosure)
        else:
            # The license held on this trace shape: stacking stays active.
            assert resolution.draw_kwargs["stack_by"] == "auto"
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_refusals_never_leave_partial_state(plain_log: Any) -> None:
    """A refused resolve leaves the trace drawable (no half-applied state)."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError):
        lenses.resolve_lens(plain_log, "transformer")
    resolution = lenses.resolve_lens(plain_log, "overview")
    assert resolution.draw_kwargs
