"""U25.2: CKA refuses non-finite input typed and warns on zero variance.

``tl.stats.cka`` and the streaming ``tl.stats.CKA`` used to pass NaN/Inf
through to a NaN score and return NaN silently for a zero-variance
representation; ``tl.repgeom.rdm`` already refuses non-finite input. Both now
refuse ``cka_input_nonfinite`` and warn ``cka_zero_variance``.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

import torchlens as tl
from torchlens.errors import InvalidArgumentError, TorchLensWarning


def _with_nan() -> torch.Tensor:
    """A 6x3 representation with one NaN entry."""

    torch.manual_seed(0)
    value = torch.randn(6, 3)
    value[0, 0] = float("nan")
    return value


def test_one_shot_cka_refuses_nonfinite_and_the_remedy_runs() -> None:
    """The refusal names the representation; cleaning it (the remedy) scores."""

    a, b = _with_nan(), torch.randn(6, 2)
    with pytest.raises(InvalidArgumentError) as info:
        tl.stats.cka(a, b)
    assert info.value.fields["code"] == "cka_input_nonfinite"
    assert info.value.fields["argument"] == "a"
    with pytest.raises(InvalidArgumentError) as info_b:
        tl.stats.cka(b, a)
    assert info_b.value.fields["argument"] == "b"

    score = tl.stats.cka(torch.nan_to_num(a), b)
    assert math.isfinite(score) and 0.0 <= score <= 1.0 + 1e-9


def test_streaming_cka_refuses_nonfinite_batches() -> None:
    """The streaming accumulator refuses at update, before state is touched."""

    meter = tl.stats.CKA()
    bad = torch.randn(6, 3)
    bad[1, 2] = float("inf")
    with pytest.raises(InvalidArgumentError) as info:
        meter.update(bad, torch.randn(6, 2))
    assert info.value.fields["code"] == "cka_input_nonfinite"

    meter.update(torch.nan_to_num(bad, posinf=0.0), torch.randn(6, 2))
    assert math.isfinite(meter.result())


def test_zero_variance_cka_warns_and_returns_nan() -> None:
    """A constant representation returns NaN with a coded warning, both forms."""

    constant, other = torch.ones(6, 3), torch.randn(6, 2)
    with pytest.warns(TorchLensWarning) as caught:
        assert math.isnan(tl.stats.cka(constant, other))
    assert caught[0].message.fields["code"] == "cka_zero_variance"

    meter = tl.stats.CKA()
    meter.update(constant, other)
    with pytest.warns(TorchLensWarning) as caught_stream:
        assert math.isnan(meter.result())
    assert caught_stream[0].message.fields["code"] == "cka_zero_variance"


def test_well_conditioned_cka_stays_silent() -> None:
    """No warning on an ordinary input: the warning is not noise."""

    torch.manual_seed(1)
    a, b = torch.randn(8, 3), torch.randn(8, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        assert math.isfinite(tl.stats.cka(a, b))
