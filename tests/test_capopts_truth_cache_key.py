"""A06 capture-options truth: the cache key covers EVERY semantic option.

WALKTHROUGH list-A row 26 (first clause): ``cache=True`` could serve captures
that never armed ``raise_on_nan`` / ``track_nonfinite`` / ``save_budget`` --
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
import torchlens.user_funcs as user_funcs
from torchlens.options import CaptureOptions
from torchlens.user_funcs import (
    CAPTURE_CACHE_KEY_CURATED,
    CAPTURE_CACHE_KEY_NEUTRAL,
)


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


@pytest.mark.smoke
def test_cache_key_ledgers_partition_capture_options() -> None:
    """Curated and neutral ledgers name real fields and never overlap."""

    field_names = {f.name for f in dataclasses.fields(CaptureOptions) if not f.name.startswith("_")}
    stale_curated = CAPTURE_CACHE_KEY_CURATED - field_names
    assert not stale_curated, f"curated ledger names non-fields: {sorted(stale_curated)}"
    stale_neutral = set(CAPTURE_CACHE_KEY_NEUTRAL) - field_names
    assert not stale_neutral, f"neutral ledger names non-fields: {sorted(stale_neutral)}"
    overlap = CAPTURE_CACHE_KEY_CURATED & set(CAPTURE_CACHE_KEY_NEUTRAL)
    assert not overlap, f"fields both curated and neutral: {sorted(overlap)}"


@pytest.mark.smoke
def test_cache_key_neutral_ledger_reasons_are_written() -> None:
    """Every declared dont-care carries a non-empty written reason."""

    for field_name, reason in CAPTURE_CACHE_KEY_NEUTRAL.items():
        assert isinstance(reason, str) and reason.strip(), (
            f"neutral field {field_name!r} has no written reason; a dont-care "
            "without a reason is an unaudited exclusion"
        )


@pytest.mark.smoke
def test_cache_key_curated_fields_have_hand_built_entries() -> None:
    """Each curated name appears as a literal config key at the assembly site.

    'Curated' licenses skipping the sweep ONLY because a richer hand-built
    entry represents the field; if that entry is deleted, the field silently
    leaves the key. This receipt scans the assembly source for the literal.
    """

    source = inspect.getsource(user_funcs)
    missing = [name for name in sorted(CAPTURE_CACHE_KEY_CURATED) if f'"{name}"' not in source]
    assert not missing, (
        f"curated fields with no hand-built cache-config entry: {missing}; "
        "either restore the entry or move the field out of the curated ledger "
        "so the sweep keys it"
    )


# ---------------------------------------------------------------------------
# CF-017-revised / H-CLEAN / H-WARM behavioral cells (toy).
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_warm_cache_identical_config_hits(tmp_path: Path) -> None:
    """H-WARM baseline: the second identical capture is served from cache."""

    model = SmallNet()
    x = torch.randn(2, 4)
    cold = _trace(model, x, str(tmp_path))
    warm = _trace(model, x, str(tmp_path))
    assert cold.capture_cache_hit is False
    assert warm.capture_cache_hit is True


@pytest.mark.smoke
@pytest.mark.parametrize(
    "field_name,flip_value",
    [
        ("raise_on_nan", True),
        ("track_nonfinite", True),
        ("save_budget", 10**9),
        ("measure_python_peak_memory", True),
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
    assert flipped.capture_cache_hit is False, (
        f"warm cache served a capture that never armed {field_name}={flip_value!r}"
    )


@pytest.mark.smoke
def test_warm_cache_neutral_flip_still_hits(tmp_path: Path) -> None:
    """Declared session-neutral knobs do not fragment the cache."""

    model = SmallNet()
    x = torch.randn(2, 4)
    _trace(model, x, str(tmp_path))
    warm = _trace(model, x, str(tmp_path), verbose=True)
    assert warm.capture_cache_hit is True


@pytest.mark.smoke
def test_warm_cache_restamps_batch_render(tmp_path: Path) -> None:
    """batch_render is neutral BECAUSE the hit path re-stamps the request."""

    model = SmallNet()
    x = torch.randn(2, 4)
    _trace(model, x, str(tmp_path))
    warm = _trace(model, x, str(tmp_path), batch_render="grid")
    assert warm.capture_cache_hit is True
    assert warm.batch_render == "grid"


@pytest.mark.smoke
def test_failed_capture_never_seeds_the_cache(tmp_path: Path) -> None:
    """H-FAILED: an aborted capture leaves nothing a later call can be served."""

    from torchlens.errors import CaptureError

    model = SmallNet()
    with torch.no_grad():
        model.fc.weight.fill_(float("nan"))
    x = torch.randn(2, 4)
    with pytest.raises(CaptureError):
        _trace(model, x, str(tmp_path), raise_on_nan=True)
    with pytest.raises(CaptureError):
        _trace(model, x, str(tmp_path), raise_on_nan=True)


# ---------------------------------------------------------------------------
# Positive-control plant (H-BOTH-shaped): prove the SWEEP is what protects the
# safety options -- declaring one neutral re-opens the historical wrong-serve.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_plant_neutralized_semantic_field_reopens_wrong_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing raise_on_nan from the key reproduces the defect; the sweep is load-bearing."""

    planted = dict(CAPTURE_CACHE_KEY_NEUTRAL)
    planted["raise_on_nan"] = "PLANT: pretend this safety knob is session-neutral"
    monkeypatch.setattr(user_funcs, "CAPTURE_CACHE_KEY_NEUTRAL", planted)
    model = SmallNet()
    x = torch.randn(2, 4)
    _trace(model, x, str(tmp_path))
    wrongly_served = _trace(model, x, str(tmp_path), raise_on_nan=True)
    assert wrongly_served.capture_cache_hit is True, (
        "the plant did not reproduce the historical wrong-serve; the sweep is "
        "no longer the mechanism keying semantic options -- update this control"
    )


# ---------------------------------------------------------------------------
# R0 realism row (M(oracles) section 7 row 2): the real GPT-2 class, warm
# cache with a semantic option change -> typed raise or recapture, never
# silent.
# ---------------------------------------------------------------------------


@pytest.mark.real_model
def test_warm_cache_semantic_flip_recaptures_distilgpt2(tmp_path: Path) -> None:
    """R0: distilgpt2 warm cache never serves a raise_on_nan=False capture."""

    pytest.importorskip("transformers")
    from tests.real_model.r0.families import build_distilgpt2

    model = build_distilgpt2("eager")
    generator = torch.Generator().manual_seed(20260826)
    input_ids = torch.randint(0, 512, (1, 8), generator=generator)
    cold = _trace(model, input_ids, str(tmp_path))
    warm = _trace(model, input_ids, str(tmp_path))
    flipped = _trace(model, input_ids, str(tmp_path), raise_on_nan=True)
    assert cold.capture_cache_hit is False
    assert warm.capture_cache_hit is True
    assert flipped.capture_cache_hit is False
