"""Raw scores beside ceiling-normalised ones in the Brain-Score bridge (neuro.1 row 2).

Pins: ``score_parts`` reads a ceiling-normalised Brain-Score result into
named parts without recomputing anything, a result without a ceiling stays
raw, and ``per_layer`` returns those parts for every layer. The fixture's
numbers are computed by hand: per-neuroid correlations 0.5, 0.7, 0.6 give
a raw score of 0.6; a ceiling of 0.8 gives a normalised 0.6 / 0.8 = 0.75.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from torchlens.bridge.brain_score import score_parts

_NEUROID_R = (0.5, 0.7, 0.6)
_RAW = 0.6  # mean of _NEUROID_R, by hand
_CEILING = 0.8
_NORMALISED = 0.75  # 0.6 / 0.8, by hand


class _FakeScore:
    """The two attributes of a Brain-Score ``Score`` that the bridge reads."""

    def __init__(self, value: Any, attrs: dict[str, Any] | None = None) -> None:
        self.values = np.asarray(value)
        self.attrs = attrs or {}


def _ceiled() -> _FakeScore:
    raw = _FakeScore(float(np.mean(_NEUROID_R)))
    ceiling = _FakeScore(_CEILING)
    return _FakeScore(
        float(raw.values) / float(ceiling.values),
        {"raw": raw, "ceiling": ceiling, "error": _FakeScore(0.01)},
    )


def test_ceiled_score_puts_raw_beside_normalised() -> None:
    parts = score_parts(_ceiled())
    assert parts.raw == pytest.approx(_RAW, abs=1e-15)
    assert parts.ceiling_normalised == pytest.approx(_NORMALISED, abs=1e-15)
    assert parts.ceiling == _CEILING
    assert parts.error == 0.01
    assert parts.note is None
    assert parts.to_dict() == {
        "raw": parts.raw,
        "ceiling_normalised": parts.ceiling_normalised,
        "ceiling": _CEILING,
        "error": 0.01,
        "note": None,
    }


_NO_CEILING = "the result carries no 'ceiling' attribute, so it is reported as raw"


def test_ceiled_score_with_per_item_raw_names_the_gap() -> None:
    parts = score_parts(_FakeScore(0.5, {"ceiling": 0.9, "raw": np.ones(4)}))
    assert parts.raw is None
    assert parts.ceiling_normalised == 0.5
    assert parts.note is not None and "attrs['raw']" in parts.note
    no_ceiling_value = score_parts(_FakeScore(0.5, {"ceiling": np.ones(3), "raw": 0.4}))
    assert no_ceiling_value.ceiling is None and no_ceiling_value.raw == 0.4
    assert no_ceiling_value.note is not None and "attrs['ceiling']" in no_ceiling_value.note


class _AggregatedScore:
    """A Brain-Score 1.x-style score: aggregation=[center, error], attrs beside it."""

    dims = ("aggregation",)

    def __init__(self, center: float, error: float, attrs: dict[str, Any] | None = None) -> None:
        self._parts = {"center": center, "error": error}
        self.attrs = attrs or {}
        self.values = np.array([center, error])

    def sel(self, aggregation: str) -> _FakeScore:
        return _FakeScore(self._parts[aggregation])


def test_aggregation_dimension_reads_center_and_error() -> None:
    ceiled = _AggregatedScore(
        _NORMALISED,
        0.02,
        {"raw": _AggregatedScore(_RAW, 0.03), "ceiling": _AggregatedScore(_CEILING, 0.01)},
    )
    parts = score_parts(ceiled)
    assert (parts.raw, parts.ceiling_normalised, parts.ceiling, parts.error) == (
        _RAW,
        _NORMALISED,
        _CEILING,
        0.02,
    )


def test_non_scalar_score_refuses() -> None:
    with pytest.raises(TypeError, match="single-number score"):
        score_parts(_FakeScore([0.1, 0.2]))


def test_real_brainscore_score_object() -> None:
    metrics = pytest.importorskip("brainscore_core.metrics")
    raw = metrics.Score(_RAW)
    ceiling = metrics.Score(_CEILING)
    ceiled = metrics.Score(_NORMALISED)
    ceiled.attrs["raw"] = raw
    ceiled.attrs["ceiling"] = ceiling
    ceiled.attrs["error"] = 0.01
    parts = score_parts(ceiled)
    assert (parts.raw, parts.ceiling_normalised, parts.ceiling, parts.error) == (
        _RAW,
        _NORMALISED,
        _CEILING,
        0.01,
    )
