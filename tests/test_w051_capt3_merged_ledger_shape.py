"""W051-CAPT3 (IO remainder): the per-rank ``distributed`` journal the C1 merge
JOINS is shape-validated at ``merge_ranks`` entry, live and loaded cores alike.

The trace-level shape check (W051-IO, ``_load_validators``) runs only on
``tl.load``; a LIVE rank core reaches the merge without it. Every malformed
container, row, or ledger event here must refuse TYPED through the existing
merged error codes -- never a bare ``TypeError`` from the row walk, never a
silent acceptance. The matrix below is the one the lane ran against the
integration tip: every row that previously escaped untyped is pinned.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_merged_engine import boundary, seeded_ledger, trace_for_boundaries

from torchlens._merged import MergedErrorCode, merge_ranks
from torchlens._merged._errors import MergeInputError


def _core(mutate) -> Any:
    trace = trace_for_boundaries([boundary(0, 0)], seeded_ledger())
    mutate(trace.annotations["distributed"])
    return trace


def _partner() -> Any:
    return trace_for_boundaries([boundary(1, 0)], seeded_ledger())


def _assert_schema_refusal(mutate, *, detail: str) -> None:
    with pytest.raises(MergeInputError) as excinfo:
        merge_ranks([_core(mutate), _partner()])
    assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value
    assert excinfo.value.fields["remedy"]
    assert detail in str(excinfo.value)


# --- the ``boundaries`` container (the rows that previously escaped as TypeError) ----


# Kept by shrink-r4 (per-id subset rule): sole coverage of _merged/_evidence.py, planted fault catches.
@pytest.mark.parametrize("shape", ["mapping"])
def test_mapping_or_string_boundaries_refuse_typed_naming_the_container(shape) -> None:
    """A mapping/str container refuses on the CONTAINER, never row-by-row over keys/chars."""

    def mutate(record):
        if shape == "mapping":
            record["boundaries"] = {"row": record["boundaries"][0]}
        else:
            record["boundaries"] = "junk"

    _assert_schema_refusal(mutate, detail="is not a list of boundary records")


# Kept by shrink-r4 (per-id subset rule): sole coverage of _merged/_presenter.py.
def test_tuple_boundaries_are_a_sequence_and_merge() -> None:
    """A tuple is a legitimate sequence container (loaders may rebuild one)."""

    merged = merge_ranks(
        [
            _core(lambda record: record.__setitem__("boundaries", tuple(record["boundaries"]))),
            _partner(),
        ]
    )
    assert sorted(merged.ranks) == [0, 1]


# Kept by shrink-r4 (per-id subset rule): sole coverage of _merged/_evidence.py.
@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (lambda r: r["boundaries"].append(None), "is not a mapping"),
    ],
)
def test_malformed_boundary_rows_refuse_typed(mutate, detail) -> None:
    """Every row-level shape violation refuses ``merged_schema_invalid`` with the field named."""

    _assert_schema_refusal(mutate, detail=detail)


# --- the ``group_lifecycle_ledger`` payload ------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _merged/_evidence.py, distributed/_ledger.py.
@pytest.mark.parametrize(
    "junk",
    [[5], True],
    ids=["list-of-int", "bool"],
)
def test_non_list_or_non_event_ledger_refuses_typed(junk) -> None:
    _assert_schema_refusal(
        lambda record: record.__setitem__("group_lifecycle_ledger", junk),
        detail="group_lifecycle_ledger",
    )


# Kept by shrink-r4 (per-id subset rule): sole coverage of _merged/_evidence.py.
def test_missing_or_null_install_epoch_refuses_typed() -> None:
    _assert_schema_refusal(lambda r: r.pop("install_epoch"), detail="install_epoch")
    _assert_schema_refusal(lambda r: r.__setitem__("install_epoch", None), detail="install_epoch")


# Kept by shrink-r4 (per-id subset rule): sole coverage of _merged/_evidence.py.
def test_absent_or_empty_journal_is_not_a_rank_capture() -> None:
    """Absence keeps its historical, distinct code: the input is simply not a rank core."""

    for mutate in (
        lambda r: r.__setitem__("boundaries", []),
        lambda r: r.pop("boundaries"),
    ):
        with pytest.raises(MergeInputError) as excinfo:
            merge_ranks([_core(mutate), _partner()])
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGE_INPUT_INVALID.value
        assert excinfo.value.fields["reason"] == "not_a_rank_capture"
