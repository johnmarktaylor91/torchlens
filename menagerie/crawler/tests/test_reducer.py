"""Canonical reducer authority and invariant tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import SingleWriterError
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.status import PartitionError, assert_partition
from menagerie.crawler.tests.conftest import make_attempt, make_gate, make_model


def _paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated canonical ledger paths.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    LedgerPaths
        Three canonical ledger paths.
    """

    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def test_reducer_is_the_single_writer(tmp_path: Path) -> None:
    """A second reducer cannot acquire canonical writer authority.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]):
        with pytest.raises(SingleWriterError):
            CanonicalReducer(paths, ["m_example"])


def test_bad_parentage_is_rejected(tmp_path: Path) -> None:
    """A superseding revision must point to the current exact parent.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    paths = _paths(tmp_path)
    first = make_model(status_code="failed:source")
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_model(first)
        second = make_model(status_code="failed:source")
        second["record_seq"] = 2
        second["parent_revision"] = "sha256:" + "f" * 64
        with pytest.raises(ReductionError, match="bad parentage"):
            reducer.append_model(second)


def test_accepted_authored_fields_require_gate(tmp_path: Path) -> None:
    """Accepted agent-authored fields cannot enter the ledger without their gate.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    paths = _paths(tmp_path)
    model = make_model(accepted=True, status_code="failed:source")
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        with pytest.raises(ReductionError, match="missing its gate"):
            reducer.append_model(model)


def test_clean_gate_and_mode_receipt_allow_run_award(tmp_path: Path) -> None:
    """The driver reducer accepts a gated model with a clean per-mode receipt.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    paths = _paths(tmp_path)
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(make_attempt())
        result = reducer.append_model(make_model(accepted=True))
        assert result.appended
        assert reducer.current_records["m_example"]["status"]["code"] == "runs"


def test_partition_duplicate_and_missing_are_rejected() -> None:
    """Current terminal rows must cover intake exactly once."""

    record = make_model(status_code="failed:source")
    duplicate = deepcopy(record)
    with pytest.raises(PartitionError):
        assert_partition(["m_example", "m_missing"], [record, duplicate])
