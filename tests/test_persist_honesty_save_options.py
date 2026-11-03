"""Persistence honesty: save levels may not silently override explicit include flags.

WT1 A-IV item 19 (lane A08): ``level="executable_with_callables"`` used to
silently force ``include_saved_args=True`` / ``include_rng_states=True`` over an
EXPLICIT ``False`` -- shipping raw input tensors against an explicit opt-out.
Symmetrically, ``audit``/``runnable`` silently forced every payload flag False
over an explicit ``True``, saving less than the caller asked for. Both
directions now refuse typed (``save_payload_level_conflict``); the omitted-flag
defaults per level are unchanged.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError


@pytest.fixture(scope="module")
def trace():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    x = torch.randn(2, 4)
    log = tl.trace(model, x)
    yield log
    log.cleanup()


def _blob_count(path) -> int:
    return len(list((path / "blobs").iterdir()))


def test_executable_level_refuses_explicit_rng_states_opt_out(trace, tmp_path):
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.save(
            tmp_path / "bundle",
            level="executable_with_callables",
            include_rng_states=False,
        )
    assert excinfo.value.fields["code"] == "save_payload_level_conflict"


@pytest.mark.parametrize("level", ["audit", "runnable"])
@pytest.mark.parametrize(
    "flag",
    ["include_outs", "include_grads", "include_saved_args", "include_rng_states"],
)
def test_payload_free_levels_refuse_explicit_payload_opt_in(trace, tmp_path, level, flag):
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.save(tmp_path / "bundle", level=level, **{flag: True})
    assert excinfo.value.fields["code"] == "save_payload_level_conflict"


def test_portable_explicit_flags_keep_historical_meaning(trace, tmp_path):
    trace.save(tmp_path / "with_args", include_saved_args=True)
    trace.save(tmp_path / "no_args", include_saved_args=False)
    assert _blob_count(tmp_path / "with_args") > _blob_count(tmp_path / "no_args")
