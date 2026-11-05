"""W051 regressions for the episode step-join grading (audit 3.1, 3.4, 3.5).

* 3.5 -- ``exogenous_positions`` was the length of the UNALIGNED SUFFIX, not
  a count of exogenous positions: one context token edited in an 8-token
  entry reported 8. It is now a positional diff (1).
* 3.4 -- teacher-forced episodes whose root returns the model's PREDICTIONS
  were graded as an exogenous break because settlement graded the join
  against the evidence column instead of ``forced_tokens``. Forced joins now
  grade ``forced`` (basis ``forced_tokens``) and never break the episode.
* 3.1 -- tail-aligned evidence derivation silently misattributed tokens when
  more than one position per step was appended (multi-token decoding), and
  the join re-grade then reported FALSE exogenous breaks. The derivation now
  refuses typed unless the source width is the emitted-only or the measured
  entry-plus-one-per-step shape.
"""

from __future__ import annotations

import pytest

from torchlens._capture._episode_join import _exogenous_position_count

V = 16


# ---------------------------------------------------------------------------
# 3.5 -- exogenous_positions is a positional diff.
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of planted fault catches.
@pytest.mark.parametrize(
    ("cur", "expected", "wildcard", "count"),
    [
        # One interior context edit: ONE position, not the whole entry.
        ((1, 15, 3, 9), (1, 2, 3, 9), 0, 1),
    ],
)
def test_exogenous_position_count_table(cur, expected, wildcard, count) -> None:
    assert _exogenous_position_count(cur, expected, wildcard) == count


# ---------------------------------------------------------------------------
# 3.4 -- forced joins grade against the declaration.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3.1 -- tail alignment must be licensed by the source width.
# ---------------------------------------------------------------------------
