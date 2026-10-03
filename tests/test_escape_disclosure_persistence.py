"""grind-r5 P7 / R16: escape disclosure survives persistence; FakeTensor ordering.

A capture TorchLens itself refused to bless (``capture_verified=False`` with a
reason) used to save and load as a capture with NO claim -- ``False -> None``,
reason gone -- so an artifact with a provably missing op was
byte-indistinguishable from an honest one. The negative verdict now persists;
a positive one still never does (tamper must not forge ``True``).
"""

from __future__ import annotations

import copy
import warnings

import pytest
import torch
from torch import nn

import torchlens as tl


class _TinyModel(nn.Module):
    """Minimal model for persistence tests."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _traced(tmp_path):
    return tl.trace(_TinyModel(), torch.randn(1, 4))


def test_negative_verification_verdict_survives_save_load(tmp_path) -> None:
    """capture_verified=False + reason must round-trip through .tlspec."""

    trace = _traced(tmp_path)
    trace.capture_verified = False
    trace.capture_verification_reason = "escape_rescue_unrecovered"

    bundle = tmp_path / "disclosed.tlspec"
    tl.save(trace, str(bundle))
    loaded = tl.load(str(bundle))

    assert loaded.capture_verified is False, (
        "the producer's negative verification claim was erased by save/load"
    )
    assert loaded.capture_verification_reason == "escape_rescue_unrecovered"


def test_positive_verification_claim_never_persists(tmp_path) -> None:
    """A True verdict (or any non-False value) loads as None: verdicts can
    only degrade across persistence, so a tampered artifact cannot forge a
    verified capture."""

    trace = _traced(tmp_path)
    trace.capture_verified = True
    trace.capture_verification_reason = "witness_verified"

    bundle = tmp_path / "positive.tlspec"
    tl.save(trace, str(bundle))
    loaded = tl.load(str(bundle))

    assert loaded.capture_verified is None
    assert loaded.capture_verification_reason is None


def test_deepcopy_keeps_the_full_live_verdict(tmp_path) -> None:
    """A same-session deepcopy is not a persistence boundary: both verdict
    directions survive it."""

    trace = _traced(tmp_path)
    trace.capture_verified = True
    trace.capture_verification_reason = "witness_verified"
    clone = copy.deepcopy(trace)
    assert clone.capture_verified is True
    assert clone.capture_verification_reason == "witness_verified"

    trace.capture_verified = False
    trace.capture_verification_reason = "mode_rescue_rerun"
    clone = copy.deepcopy(trace)
    assert clone.capture_verified is False
    assert clone.capture_verification_reason == "mode_rescue_rerun"


def test_deferred_storage_key_never_touches_fake_tensor_storage() -> None:
    """grind-r5 b6 R16 (sol): the deferred-alias key helper must classify a
    FakeTensor ineligible WITHOUT reading its data pointer (torch emits an
    'almost definitely a bug' warning on that read, ahead of the typed
    refusal the entry guard owns)."""

    fake_tensor_mode = pytest.importorskip("torch._subclasses.fake_tensor").FakeTensorMode

    from torchlens.utils.tensor_utils import _deferred_storage_key

    with fake_tensor_mode() as mode:
        fake = mode.from_tensor(torch.randn(2, 2))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _deferred_storage_key(fake) is None
