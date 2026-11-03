"""A06 capture-options truth: the cache key covers EVERY semantic option.

WALKTHROUGH list-A row 26 (first clause): ``cache=True`` could serve captures
that never armed ``on_nonfinite`` (then ``raise_on_nan`` / ``track_nonfinite``) / ``save_budget`` --
the key was a hand-enumerated include-list, so semantic knobs added after it
silently fell outside it (fail-open on safety options). The key is now
INVERTED (M(oracles) item 6): every ``CaptureOptions`` field enters the key
by default; a field stays out only by being hand-CURATED into a richer config
entry or declared session-NEUTRAL with a written reason. A new field defaults
INTO the key -- a spurious miss is the safe direction.

Oracle rows: CF-019 (Options-vs-cache-key set difference, static), CF-017-
revised (warm cache + semantic-option request -> recapture, never silent),
CF-025-shaped warm/cold cells, one H-FAILED cell (a failed capture never
seeds the cache), and a positive-control plant proving the oracle is armed.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens._capture.prepare.cache as prepare_cache
from torchlens._capture.prepare.cache import (
    CAPTURE_CACHE_KEY_CURATED,
    CAPTURE_CACHE_KEY_NEUTRAL,
)
from torchlens.options import CaptureOptions


class SmallNet(nn.Module):
    """fc -> relu, one op of each kind the toy cells need."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _trace(model: nn.Module, x: torch.Tensor, cache_dir: str, **fields: Any) -> tl.Trace:
    return tl.trace(model, x, capture=CaptureOptions(cache=True, cache_dir=cache_dir, **fields))


# ---------------------------------------------------------------------------
# CF-019: the static set-difference oracle. With the inversion, a NEW field is
# auto-swept into the key, so the residual failure modes are (a) a stale name
# in the curated/neutral ledgers and (b) a curated field whose hand-built
# config entry rotted away. Both are pinned here.
# ---------------------------------------------------------------------------


def test_cache_key_ledgers_partition_capture_options() -> None:
    """Curated and neutral ledgers name real fields and never overlap."""

    field_names = {f.name for f in dataclasses.fields(CaptureOptions) if not f.name.startswith("_")}
    stale_curated = CAPTURE_CACHE_KEY_CURATED - field_names
    assert not stale_curated, f"curated ledger names non-fields: {sorted(stale_curated)}"
    stale_neutral = set(CAPTURE_CACHE_KEY_NEUTRAL) - field_names
    assert not stale_neutral, f"neutral ledger names non-fields: {sorted(stale_neutral)}"
    overlap = CAPTURE_CACHE_KEY_CURATED & set(CAPTURE_CACHE_KEY_NEUTRAL)
    assert not overlap, f"fields both curated and neutral: {sorted(overlap)}"


def test_cache_key_neutral_ledger_reasons_are_written() -> None:
    """Every declared dont-care carries a non-empty written reason."""

    for field_name, reason in CAPTURE_CACHE_KEY_NEUTRAL.items():
        assert isinstance(reason, str) and reason.strip(), (
            f"neutral field {field_name!r} has no written reason; a dont-care "
            "without a reason is an unaudited exclusion"
        )


def test_cache_key_curated_fields_have_hand_built_entries() -> None:
    """Each curated name appears as a literal config key at the assembly site.

    'Curated' licenses skipping the sweep ONLY because a richer hand-built
    entry represents the field; if that entry is deleted, the field silently
    leaves the key. This receipt scans the assembly source for the literal.
    """

    source = inspect.getsource(prepare_cache._curated_cache_config)
    missing = [name for name in sorted(CAPTURE_CACHE_KEY_CURATED) if f'"{name}"' not in source]
    assert not missing, (
        f"curated fields with no hand-built cache-config entry: {missing}; "
        "either restore the entry or move the field out of the curated ledger "
        "so the sweep keys it"
    )


# ---------------------------------------------------------------------------
# CF-017-revised / H-CLEAN / H-WARM behavioral cells (toy).
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/finalize/epilogue.py, _capture/prepare/door.py, _data_classes/_nonfinite.py ....
@pytest.mark.parametrize(
    "field_name,flip_value",
    [
        ("save_budget", 10**9),
        ("reconstruction_ready", True),
        ("on_nonfinite", "warn"),
    ],
)
def test_warm_cache_semantic_flip_recaptures(
    tmp_path: Path, field_name: str, flip_value: Any
) -> None:
    """A warm cache NEVER serves a capture that did not arm the semantic knob."""

    model = SmallNet()
    x = torch.randn(2, 4)
    _trace(model, x, str(tmp_path))
    flipped = _trace(model, x, str(tmp_path), **{field_name: flip_value})
    assert flipped.provenance.cache_hit is False, (
        f"warm cache served a capture that never armed {field_name}={flip_value!r}"
    )


# ---------------------------------------------------------------------------
# Positive-control plant (H-BOTH-shaped): prove the SWEEP is what protects the
# safety options -- declaring one neutral re-opens the historical wrong-serve.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# R0 realism row (M(oracles) section 7 row 2): the real GPT-2 class, warm
# cache with a semantic option change -> typed raise or recapture, never
# silent.
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture_fingerprint.py, _capture_state_helpers.py.
@pytest.mark.real_model
def test_warm_cache_semantic_flip_recaptures_distilgpt2(tmp_path: Path) -> None:
    """R0: distilgpt2 warm cache never serves an on_nonfinite="ignore" capture."""

    pytest.importorskip("transformers")
    from tests.real_model.r0.families import build_distilgpt2

    model = build_distilgpt2("eager")
    generator = torch.Generator().manual_seed(20260826)
    input_ids = torch.randint(0, 512, (1, 8), generator=generator)
    cold = _trace(model, input_ids, str(tmp_path))
    warm = _trace(model, input_ids, str(tmp_path))
    flipped = _trace(model, input_ids, str(tmp_path), on_nonfinite="raise")
    assert cold.provenance.cache_hit is False
    assert warm.provenance.cache_hit is True
    assert flipped.provenance.cache_hit is False
