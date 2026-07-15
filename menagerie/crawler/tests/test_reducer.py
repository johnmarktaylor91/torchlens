"""Canonical reducer authority and invariant tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import SingleWriterError
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.status import PartitionError, assert_partition
from menagerie.crawler.metadata import authored_fact_leaves
from menagerie.crawler.tests.conftest import (
    _bind_model_identities,
    _model_facts,
    make_attempt,
    make_gate,
    make_model,
)


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


def test_cache_read_attempted_blocks_otherwise_valid_run(tmp_path: Path) -> None:
    """Parent-observed cache-backed reads poison an otherwise complete receipt."""

    paths = _paths(tmp_path)
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    attempt = make_attempt()
    attempt["policy_observation"]["cache_read_attempted"] = True
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(attempt)
        with pytest.raises(ReductionError, match="clean successful worker receipt"):
            reducer.append_model(make_model(accepted=True))


def test_partition_duplicate_and_missing_are_rejected() -> None:
    """Current terminal rows must cover intake exactly once."""

    record = make_model(status_code="failed:source")
    duplicate = deepcopy(record)
    with pytest.raises(PartitionError):
        assert_partition(["m_example", "m_missing"], [record, duplicate])


def test_stale_rung_check_and_pending_metadata_cannot_bypass_run_gate(
    tmp_path: Path,
) -> None:
    """A run always needs a current rung decision, independent of metadata state."""

    paths = _paths(tmp_path)
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(make_attempt())
        stale_rung = make_model(accepted=True)
        stale_rung["source_resolution"]["rung"] = "R2_VENDOR"
        with pytest.raises(ReductionError, match="anti-slop/rung"):
            reducer.append_model(stale_rung)

    pending_paths = _paths(tmp_path / "pending")
    with CanonicalReducer(pending_paths, stable_ids) as reducer:
        reducer.append_attempt(make_attempt())
        with pytest.raises(ReductionError, match="anti-slop/rung"):
            reducer.append_model(make_model(accepted=False))


def test_family_template_true_is_verified_against_representative(tmp_path: Path) -> None:
    """A written true cannot conceal changed family prose."""

    paths = _paths(tmp_path)
    stable_ids = ["m_rep", "m_variant", *(f"m_{index}" for index in range(8))]
    representative = make_model("m_rep", accepted=True, attempt_id="attempt-rep")
    representative_attempt = make_attempt("m_rep", attempt_id="attempt-rep")
    variant = make_model("m_variant", accepted=True, attempt_id="attempt-variant")
    variant_attempt = make_attempt("m_variant", attempt_id="attempt-variant")
    variant_attempt.pop("ledger_seq")
    variant_attempt.pop("payload_sha256")
    variant["identity"]["family_representative_id"] = "m_rep"
    variant["website"].update(
        {
            "kind": "size-variant-template",
            "template_source_model_id": "m_rep",
            "variant_parameter_input_line": "2 parameters; input [1, 3, 8, 8]",
            "template_hash": "sha256:" + "f" * 64,
            "description": "silently altered family prose",
        }
    )
    _bind_model_identities(variant)
    gate = make_gate(stable_ids)
    variant_gate_item = next(item for item in gate["items"] if item["stable_id"] == "m_variant")
    variant_gate_item["vet_identity"] = variant["accuracy_gate"]["vet_identity"]
    variant_gate_item["field_checks"] = [
        {
            "field": field,
            "verdict": "accurate",
            "evidence_ids": ["evidence-1"],
            "checked_source_ids": ["source-1"],
            "reason": "supported",
            "required_repair": None,
        }
        for field in authored_fact_leaves(_model_facts(variant))
    ]
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(gate)
        reducer.append_attempt(representative_attempt)
        reducer.append_attempt(variant_attempt)
        reducer.append_model(representative)
        with pytest.raises(ReductionError, match="family template validation failed"):
            reducer.append_model(variant)


@pytest.mark.parametrize(
    "receipt_field",
    [
        "constructor_started",
        "constructor_completed",
        "input_completed",
        "forward_started",
        "forward_completed",
    ],
)
def test_run_award_rejects_incomplete_receipt_flags(tmp_path: Path, receipt_field: str) -> None:
    """Every constructor/input/forward lifecycle predicate is mandatory for runs."""

    paths = _paths(tmp_path / receipt_field)
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    attempt = make_attempt()
    attempt["worker_receipt"][receipt_field] = False
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(attempt)
        with pytest.raises(ReductionError, match="complete zero-exit receipt"):
            reducer.append_model(make_model(accepted=True))


def test_run_award_rejects_null_input_signature(tmp_path: Path) -> None:
    """A succeeded label cannot replace a structurally complete input signature."""

    paths = _paths(tmp_path)
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    attempt = make_attempt()
    attempt["worker_receipt"]["input_signature"] = None
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(attempt)
        with pytest.raises(ReductionError, match="complete zero-exit receipt"):
            reducer.append_model(make_model(accepted=True))


def test_run_award_rejects_null_output_signature_tree(tmp_path: Path) -> None:
    """A null/empty output contract cannot satisfy reducer run-award validation."""

    paths = _paths(tmp_path)
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    attempt = make_attempt()
    attempt["worker_receipt"]["output_signature"] = {"tree": None, "leaves": []}
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(attempt)
        with pytest.raises(ReductionError, match="complete zero-exit receipt"):
            reducer.append_model(make_model(accepted=True))


def test_recursive_authored_gate_blocks_ungated_website_leaf(tmp_path: Path) -> None:
    """An accurate external-metadata subset cannot authorize the rest of proposed facts."""

    paths = _paths(tmp_path)
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    gate = make_gate(stable_ids)
    item = next(value for value in gate["items"] if value["stable_id"] == "m_example")
    item["field_checks"] = [
        check for check in item["field_checks"] if check["field"] != "website.tagline"
    ]
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(gate)
        reducer.append_attempt(make_attempt())
        with pytest.raises(ReductionError, match="ungated authored facts"):
            reducer.append_model(make_model(accepted=True))


def test_reducer_recomputes_source_identity_from_canonical_facts(tmp_path: Path) -> None:
    """Changing accepted source facts makes inherited gate/attempt identities stale."""

    paths = _paths(tmp_path)
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    model = make_model(accepted=True)
    model["source_resolution"]["sources"][0]["revision"] = "mutated-after-gate"
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(make_attempt())
        with pytest.raises(ReductionError, match="identities are stale"):
            reducer.append_model(model)


def test_reducer_rejects_gate_from_stale_checker_prompt(tmp_path: Path) -> None:
    """A syntactically valid gate from old prompt bytes is not current evidence."""

    gate = make_gate([f"m_{index}" for index in range(10)])
    gate["checker"]["prompt_sha256"] = "sha256:" + "f" * 64
    with CanonicalReducer(_paths(tmp_path), [f"m_{index}" for index in range(10)]) as reducer:
        with pytest.raises(ReductionError, match="current prompt bytes"):
            reducer.append_gate(gate)
