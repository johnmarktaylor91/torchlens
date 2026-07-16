"""Canonical reducer authority and invariant tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import MODEL_SCHEMA_VERSION, OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.family_templates import (
    instantiate_size_variant,
    specialize_size_variant_recipe,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.metadata import recompute_accepted_identities
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import JsonlLedger, SingleWriterError
from menagerie.crawler.reducer import CanonicalReducer, ReductionError, materialize_current
from menagerie.crawler.status import (
    PartitionError,
    assert_partition,
    completeness_report,
    record_is_release_eligible,
)
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


def _variant_from_representative(
    representative: dict[str, Any],
    *,
    stable_id: str = "m_variant",
    variant_token: str = "Large",
    attempt_id: str = "attempt-variant",
) -> dict[str, Any]:
    """Build an exact mechanically specialized variant fixture.

    Parameters
    ----------
    representative:
        Persisted accepted representative revision.
    stable_id, variant_token, attempt_id:
        Variant identity and execution reference.

    Returns
    -------
    dict[str, object]
        Reducer-valid family variant candidate.
    """

    variant = deepcopy(representative)
    representative_id = str(representative["stable_id"])
    variant["stable_id"] = stable_id
    variant.pop("record_seq", None)
    variant.pop("record_revision", None)
    variant["parent_revision"] = None
    identity = variant["identity"]
    assert isinstance(identity, dict)
    identity.update(
        {
            "canonical_name": f"ExampleNet {variant_token}",
            "variant": variant_token,
            "family_representative_id": representative_id,
        }
    )
    implementation, input_contract, derivation = specialize_size_variant_recipe(
        representative,
        representative_model_id=representative_id,
        variant_token=variant_token,
    )
    variant["implementation"] = implementation
    variant["input_contract"] = input_contract
    variant["family_variant_derivation"] = derivation
    variant["website"] = instantiate_size_variant(
        representative,
        representative_model_id=representative_id,
        variant_parameter_input_line="3 parameters; input [1, 3, 8, 8]",
    )
    observed = variant["observed"]
    assert isinstance(observed, dict)
    observed["parameter_count_total"] = 3
    observed["parameter_count_trainable"] = 3
    observed["measurement_attempt_ids"] = [attempt_id]
    modes = variant["modes"]
    assert isinstance(modes, dict)
    modes["per_mode_run"] = {"eval": {"attempt_id": attempt_id, "status": "succeeded"}}
    execution = variant["execution"]
    assert isinstance(execution, dict)
    execution["accepted_attempt_ids"] = [attempt_id]
    status = variant["status"]
    assert isinstance(status, dict)
    status["attempt_ids"] = [attempt_id]
    _bind_model_identities(variant)
    variant["accuracy_gate"] = deepcopy(representative["accuracy_gate"])
    return variant


def _attempt_for_model(model: dict[str, Any], attempt_id: str) -> dict[str, Any]:
    """Build an attempt whose identities and measurements bind one model fixture.

    Parameters
    ----------
    model:
        Accepted canonical model candidate.
    attempt_id:
        Exact attempt identity referenced by the candidate.

    Returns
    -------
    dict[str, object]
        Successful reducer-verifiable attempt.
    """

    stable_id = str(model["stable_id"])
    attempt = make_attempt(stable_id, attempt_id=attempt_id)
    accuracy = model["accuracy_gate"]
    assert isinstance(accuracy, dict)
    identities = recompute_accepted_identities(
        _model_facts(model),
        checker_prompt_hash=str(accuracy["prompt_sha256"]),
        checker_model="codex",
        checker_version="test",
    )
    attempt_identities = attempt["identities"]
    receipt = attempt["worker_receipt"]
    observed = model["observed"]
    assert isinstance(attempt_identities, dict)
    assert isinstance(receipt, dict)
    assert isinstance(observed, dict)
    attempt_identities.update(
        {
            "source": identities.source,
            "evidence": identities.evidence,
            "recipe": identities.recipe,
            "checker_prompt": str(accuracy["prompt_sha256"]),
        }
    )
    receipt["observed_recipe_revision"] = identities.recipe
    receipt["parameter_count_total"] = observed["parameter_count_total"]
    receipt["parameter_count_trainable"] = observed["parameter_count_trainable"]
    return attempt


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


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("parameter_count_total", 999),
        ("parameter_count_trainable", 998),
        ("native_framework", "tensorflow"),
        ("delegated_method", "__call__"),
        ("output_signature", {"tree": None, "leaves": []}),
        ("input_kind", "random-fallback"),
        ("input_asset", None),
        ("input_note", "different input"),
        ("constructor_seconds", 9.0),
        ("forward_seconds", 8.0),
        ("peak_rss_bytes", 9999),
        ("measurement_attempt_ids", []),
    ),
)
def test_reducer_rejects_observed_facts_not_earned_by_receipts(
    tmp_path: Path, field: str, replacement: object
) -> None:
    """Every published runtime observation must match its designated accepted attempt."""

    model = make_model(accepted=True)
    observed = model["observed"]
    assert isinstance(observed, dict)
    observed[field] = replacement
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_gate(make_gate(["m_example"]))
        reducer.append_attempt(make_attempt())
        with pytest.raises(ReductionError, match="observed runtime facts contradict"):
            reducer.append_model(model)


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


@pytest.mark.parametrize(
    "field",
    (
        "message",
        "mode_error",
        "observed_response",
        "receipt_error",
        "response_excerpt",
        "stderr_tail",
        "stdout_tail",
        "traceback",
    ),
)
def test_reducer_rejects_raw_c07_fields_in_attempts(tmp_path: Path, field: str) -> None:
    """No producer can append raw text through any checkpoint-protected field name."""

    attempt = make_attempt()
    attempt["c07_tripwire"] = {field: "raw externally controlled text"}
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        with pytest.raises(ReductionError, match="unredacted externally controlled text"):
            reducer.append_attempt(attempt)


@pytest.mark.parametrize(
    ("location", "field"),
    (
        ("status", "detail"),
        ("status", "traceback"),
        ("human_review", "reason"),
        ("nested", "message"),
        ("nested", "stdout_tail"),
    ),
)
def test_reducer_rejects_raw_c07_fields_in_models(
    tmp_path: Path, location: str, field: str
) -> None:
    """Terminal projections and nested model diagnostics require sidecar references."""

    model = make_model(status_code="failed:source")
    if location == "status":
        model["status"][field] = "raw externally controlled text"
    elif location == "human_review":
        model["status"]["human_review"][field] = "raw externally controlled text"
    else:
        model["c07_tripwire"] = {field: "raw externally controlled text"}
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        with pytest.raises(ReductionError, match="unredacted externally controlled text"):
            reducer.append_model(model)


def test_first_revision_rejects_public_supersession_lineage(tmp_path: Path) -> None:
    """A first revision cannot claim to supersede an unrelated public revision."""

    model = make_model(status_code="failed:source")
    model["status"]["supersedes_revision"] = "sha256:" + "f" * 64
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        with pytest.raises(ReductionError, match="first model revision cannot supersede"):
            reducer.append_model(model)


def test_later_revision_binds_public_supersession_to_parent(tmp_path: Path) -> None:
    """The public supersession field must equal the reducer-authorized parent."""

    paths = _paths(tmp_path)
    first = make_model(status_code="failed:source")
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = reducer.append_model(first).record
        second = make_model(status_code="failed:source")
        second["record_seq"] = 2
        second["parent_revision"] = persisted["record_revision"]
        second["status"]["supersedes_revision"] = "sha256:" + "f" * 64
        with pytest.raises(ReductionError, match="exactly match parent_revision"):
            reducer.append_model(second)


def test_reducer_rebuild_rejects_persisted_false_supersession(tmp_path: Path) -> None:
    """Reducer startup detects false public lineage already persisted in canonical bytes."""

    paths = _paths(tmp_path)
    first = make_model(status_code="failed:source")
    with JsonlLedger(paths.models, MODEL_SCHEMA_VERSION) as ledger:
        persisted = ledger.append(first).record
        second = make_model(status_code="failed:source")
        second["record_seq"] = 2
        second["parent_revision"] = persisted["record_revision"]
        second["status"]["supersedes_revision"] = "sha256:" + "f" * 64
        ledger.append(second)
    with pytest.raises(ReductionError, match="persisted status.supersedes_revision"):
        CanonicalReducer(paths, ["m_example"])


def test_requeue_grant_rejects_wrong_exact_parent_binding(tmp_path: Path) -> None:
    """A durable grant cannot authorize a supersession of a different parent revision."""

    paths = _paths(tmp_path)
    first = make_model(status_code="failed:source")
    first["status"]["human_review"] = {
        "required": True,
        # The test exercises requeue lineage; raw review diagnostics are rejected separately.
        "reason": None,
        "queue": "crawler-human-review",
        "requested_at": first["created_at"],
    }
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        parent = reducer.append_model(first).record["record_revision"]
        reason = "reviewed corrected source"
        grant_id = stable_hash(
            {
                "stable_id": "m_example",
                "stage": "source",
                "reason": reason,
                "grant": 1,
            }
        )
        grant = {
            "grant_id": grant_id,
            "stable_id": "m_example",
            "stage": "source",
            "reason": reason,
            "attempts": 1,
            "created_at": first["created_at"],
        }
        operational = tmp_path / "operational"
        operational.mkdir(parents=True, exist_ok=True)
        (operational / "requeue-grants.jsonl").write_bytes(canonical_json_bytes(grant) + b"\n")
        wrong_parent = "sha256:" + "f" * 64
        work_id = stable_hash(
            {
                "stable_id": "m_example",
                "grant_id": grant_id,
                "parent_revision": wrong_parent,
                "generation": 1,
            }
        )
        event = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "event_id": f"requeue-consumed-{grant_id[7:31]}",
            "created_at": first["created_at"],
            "event_kind": "requeue-grant-consumed",
            "status": "requeue-grant-consumed",
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": None,
            "run_id": "run-test",
            "machine_id": "machine-test",
            "details": {
                "grant_id": grant_id,
                "stable_id": "m_example",
                "stage": "source",
                "reason": reason,
                "attempts": 1,
                "source_record_revision": wrong_parent,
                "new_work_generation": 1,
                "new_work_id": work_id,
            },
        }
        with JsonlLedger(operational / "events.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
            ledger.append(event)
        attempt = make_attempt(attempt_id="attempt-requeue")
        attempt["work_id"] = work_id
        reducer.append_attempt(attempt)
        second = make_model(
            accepted=False,
            status_code="failed:source",
            attempt_id="attempt-requeue",
        )
        second["record_seq"] = 2
        second["parent_revision"] = parent
        second["status"]["supersedes_revision"] = parent
        second["budget"]["explicit_grants"] = [grant_id]
        with pytest.raises(ReductionError, match="wrong parent revision"):
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
    """A pending run needs exact retained proposal proof, never a blind state flip."""

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
        with pytest.raises(ReductionError, match="pending metadata run"):
            reducer.append_model(make_model(accepted=False))


def test_legacy_audit_flag_cannot_bypass_current_fidelity_gate(tmp_path: Path) -> None:
    """Reducer rejects a fresh R1 run that drops a preserved legacy audit obligation."""

    paths = _paths(tmp_path)
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    model = make_model(accepted=True)
    model["intake"]["preserved_legacy_flags"] = ["legacy-fidelity-claim"]
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(make_attempt())
        with pytest.raises(ReductionError, match="required fidelity is missing"):
            reducer.append_model(model)


def test_family_template_is_verified_even_when_candidate_declares_false(tmp_path: Path) -> None:
    """A producer false cannot disable byte-exact inherited metadata validation."""

    paths = _paths(tmp_path)
    gate_ids = ["m_rep", *(f"m_{index}" for index in range(9))]
    stable_ids = [*gate_ids, "m_variant"]
    representative = make_model("m_rep", accepted=True, attempt_id="attempt-rep")
    representative_attempt = make_attempt("m_rep", attempt_id="attempt-rep")
    with CanonicalReducer(
        paths,
        stable_ids,
        intake_variant_bindings={"m_variant": ("m_rep", "Large")},
    ) as reducer:
        reducer.append_gate(make_gate(gate_ids))
        reducer.append_attempt(representative_attempt)
        persisted_representative = reducer.append_model(representative).record
        variant = _variant_from_representative(persisted_representative)
        variant["taxonomy"]["family"] = "silently altered family"
        variant["completeness"]["family_template_valid"] = False
        variant_attempt = _attempt_for_model(variant, "attempt-variant")
        variant_attempt.pop("ledger_seq")
        variant_attempt.pop("payload_sha256")
        reducer.append_attempt(variant_attempt)
        with pytest.raises(ReductionError, match="inherited metadata field 'taxonomy'"):
            reducer.append_model(variant)


def test_family_variant_recipe_must_be_mechanical_specialization(tmp_path: Path) -> None:
    """A self-consistent unrelated recipe cannot inherit representative authority."""

    paths = _paths(tmp_path)
    gate_ids = ["m_rep", *(f"m_{index}" for index in range(9))]
    stable_ids = [*gate_ids, "m_variant"]
    representative = make_model("m_rep", accepted=True, attempt_id="attempt-rep")
    with CanonicalReducer(
        paths,
        stable_ids,
        intake_variant_bindings={"m_variant": ("m_rep", "Large")},
    ) as reducer:
        reducer.append_gate(make_gate(gate_ids))
        reducer.append_attempt(make_attempt("m_rep", attempt_id="attempt-rep"))
        persisted_representative = reducer.append_model(representative).record
        variant = _variant_from_representative(persisted_representative)
        variant["implementation"]["library_recipe"]["module"] = "unrelated.mobilenet"
        variant["implementation"]["library_recipe"]["symbol"] = "MobileNetV3"
        inherited_accuracy = deepcopy(variant["accuracy_gate"])
        _bind_model_identities(variant)
        variant["accuracy_gate"] = inherited_accuracy
        variant_attempt = _attempt_for_model(variant, "attempt-variant")
        variant_attempt.pop("ledger_seq")
        variant_attempt.pop("payload_sha256")
        reducer.append_attempt(variant_attempt)
        with pytest.raises(ReductionError, match="not the mechanical representative"):
            reducer.append_model(variant)


def test_reducer_derives_release_and_rejects_pending_true_claim(tmp_path: Path) -> None:
    """A candidate cannot publish pending metadata by flipping release true."""

    model = make_model(accepted=False, status_code="failed:source")
    model["completeness"]["release_eligible"] = True
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        with pytest.raises(ReductionError, match="completeness.release_eligible"):
            reducer.append_model(model)


def test_representative_supersession_stales_current_variant(tmp_path: Path) -> None:
    """A representative-only revision blocks completion and dependent publication."""

    paths = _paths(tmp_path)
    gate_ids = ["m_rep", *(f"m_{index}" for index in range(9))]
    stable_ids = [*gate_ids, "m_variant"]
    representative = make_model("m_rep", accepted=True, attempt_id="attempt-rep")
    with CanonicalReducer(
        paths,
        stable_ids,
        intake_variant_bindings={"m_variant": ("m_rep", "Large")},
    ) as reducer:
        reducer.append_gate(make_gate(gate_ids))
        reducer.append_attempt(make_attempt("m_rep", attempt_id="attempt-rep"))
        persisted_representative = reducer.append_model(representative).record
        variant = _variant_from_representative(persisted_representative)
        variant_attempt = _attempt_for_model(variant, "attempt-variant")
        variant_attempt.pop("ledger_seq")
        variant_attempt.pop("payload_sha256")
        reducer.append_attempt(variant_attempt)
        reducer.append_model(variant)

        superseding = deepcopy(persisted_representative)
        superseding.pop("record_seq", None)
        superseding.pop("record_revision", None)
        superseding["parent_revision"] = persisted_representative["record_revision"]
        superseding["status"]["supersedes_revision"] = persisted_representative["record_revision"]
        superseding["notes"] = "metadata-only representative re-vet"
        reducer.append_model(superseding)

        raw_current = reducer.current_records
        report = completeness_report(["m_rep", "m_variant"], raw_current)
        assert report.incomplete_by_issue["stale_family_variant"] == ("m_variant",)
        assert not report.complete
        assert not record_is_release_eligible(raw_current["m_variant"], raw_current)

    materialized = materialize_current(paths)
    assert "m_rep" in materialized
    assert "m_variant" not in materialized


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
