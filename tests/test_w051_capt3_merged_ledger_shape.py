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

import copy
from typing import Any

import pytest
from test_merged_engine import boundary, seeded_ledger, trace_for_boundaries

from torchlens.merged import MergedErrorCode, merge_ranks
from torchlens.merged._errors import MergeInputError


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


@pytest.mark.smoke_cells("test_non_sequence_boundaries_refuse_typed[True]")
@pytest.mark.parametrize("junk", [5, True, 0.5, None.__class__])
def test_non_sequence_boundaries_refuse_typed(junk) -> None:
    """An int/bool/float/type ``boundaries`` used to die as a bare TypeError."""

    _assert_schema_refusal(
        lambda record: record.__setitem__("boundaries", junk),
        detail="boundaries of live[0] is not a list of boundary records",
    )


@pytest.mark.parametrize("shape", ["mapping", "str"])
def test_mapping_or_string_boundaries_refuse_typed_naming_the_container(shape) -> None:
    """A mapping/str container refuses on the CONTAINER, never row-by-row over keys/chars."""

    def mutate(record):
        if shape == "mapping":
            record["boundaries"] = {"row": record["boundaries"][0]}
        else:
            record["boundaries"] = "junk"

    _assert_schema_refusal(mutate, detail="is not a list of boundary records")


@pytest.mark.smoke
def test_tuple_boundaries_are_a_sequence_and_merge() -> None:
    """A tuple is a legitimate sequence container (loaders may rebuild one)."""

    merged = merge_ranks(
        [
            _core(lambda record: record.__setitem__("boundaries", tuple(record["boundaries"]))),
            _partner(),
        ]
    )
    assert sorted(merged.ranks) == [0, 1]


@pytest.mark.smoke_cells(
    "test_malformed_boundary_rows_refuse_typed[<lambda>-is not a mapping]",
    "test_malformed_boundary_rows_refuse_typed[<lambda>-malformed correlation key]",
)
@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (lambda r: r["boundaries"].append(None), "is not a mapping"),
        (lambda r: r["boundaries"][0].__setitem__("correlation", [1]), "malformed correlation key"),
        (lambda r: r["boundaries"][0].__setitem__("group", "g"), "no group membership record"),
        (lambda r: r["boundaries"][0].__setitem__("events", "e"), "malformed events record"),
        (lambda r: r["boundaries"][0].__setitem__("roles", "r"), "roles"),
        (lambda r: r["boundaries"][0].__setitem__("disclosures", "d"), "disclosures"),
        (lambda r: r["boundaries"][0].__setitem__("op_labels_raw", "l"), "op_labels_raw"),
        (lambda r: r["boundaries"][0].pop("kind"), "kind"),
        (lambda r: r["boundaries"][0].pop("witness"), "witness"),
        (lambda r: r["boundaries"][0].pop("c10d_group_seq"), "c10d_group_seq"),
        (lambda r: r["boundaries"][0].pop("lifetime_evidence"), "lifetime_evidence"),
    ],
)
def test_malformed_boundary_rows_refuse_typed(mutate, detail) -> None:
    """Every row-level shape violation refuses ``merged_schema_invalid`` with the field named."""

    _assert_schema_refusal(mutate, detail=detail)


# --- the ``group_lifecycle_ledger`` payload ------------------------------------------


@pytest.mark.smoke_cells("test_non_list_or_non_event_ledger_refuses_typed[str]")
@pytest.mark.parametrize(
    "junk",
    ["junk", {"x": 1}, [["x"]], [5], 7, True],
    ids=["str", "mapping", "list-of-list", "list-of-int", "int", "bool"],
)
def test_non_list_or_non_event_ledger_refuses_typed(junk) -> None:
    _assert_schema_refusal(
        lambda record: record.__setitem__("group_lifecycle_ledger", junk),
        detail="group_lifecycle_ledger",
    )


def test_tuple_ledger_refuses_typed() -> None:
    """The ledger codec's own contract is a list; a tuple refuses typed, not silently."""

    _assert_schema_refusal(
        lambda record: record.__setitem__(
            "group_lifecycle_ledger", tuple(record["group_lifecycle_ledger"])
        ),
        detail="group_lifecycle_ledger",
    )


@pytest.mark.smoke_cells("test_malformed_ledger_events_refuse_typed[epoch-vocab]")
@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["group_lifecycle_ledger"][0].__setitem__("extra", 1),
        lambda r: r["group_lifecycle_ledger"][0].__setitem__("event_index", "0"),
        lambda r: r["group_lifecycle_ledger"][0].__setitem__("ordinal", True),
        lambda r: r["group_lifecycle_ledger"][0].__setitem__("kind", "bogus"),
        lambda r: r["group_lifecycle_ledger"][0].__setitem__("install_epoch", "forged"),
        lambda r: r["group_lifecycle_ledger"][0].pop("membership_digest"),
    ],
    ids=["extra-key", "index-str", "ordinal-bool", "kind-vocab", "epoch-vocab", "missing-digest"],
)
def test_malformed_ledger_events_refuse_typed(mutate) -> None:
    _assert_schema_refusal(mutate, detail="group_lifecycle_ledger")


@pytest.mark.smoke
def test_missing_or_null_install_epoch_refuses_typed() -> None:
    _assert_schema_refusal(lambda r: r.pop("install_epoch"), detail="install_epoch")
    _assert_schema_refusal(lambda r: r.__setitem__("install_epoch", None), detail="install_epoch")


@pytest.mark.smoke
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


def test_well_formed_cores_still_merge() -> None:
    """The shape gate admits the genuine journal unchanged (no false refusal)."""

    genuine = _core(lambda record: None)
    snapshot = copy.deepcopy(genuine.annotations["distributed"])
    merged = merge_ranks([genuine, _partner()])
    assert sorted(merged.ranks) == [0, 1]
    assert genuine.annotations["distributed"] == snapshot
