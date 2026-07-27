"""Canonical reducer authority and invariant tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.artifact_transactions import ArtifactInput, stage_private_artifact
from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.family_templates import (
    instantiate_size_variant,
    specialize_size_variant_recipe,
)
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.metadata import recompute_accepted_identities
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorStore
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import JsonlLedger, SingleWriterError, scan_jsonl
from menagerie.crawler.reducer import (
    CanonicalReducer as ProductionCanonicalReducer,
    ReductionError,
    materialize_current,
    project_dependency_current,
)
from menagerie.crawler.status import (
    PartitionError,
    assert_partition,
    completeness_report,
    record_is_release_eligible,
)
from menagerie.crawler.tests.conftest import (
    _bind_model_identities,
    _model_facts,
    bind_handoff_execution,
    bind_terminal_attempts,
    make_attempt,
    make_failed_attempt,
    make_gate,
    make_authority_context,
    make_model,
    make_author_proposal,
    rebind_attempt_raw_proof,
    rebind_nonaward_parent_proof,
)


class CanonicalReducer(ProductionCanonicalReducer):
    """Adapt legacy-focused reducer cases to the mandatory v3 authority context."""

    def __init__(
        self,
        paths: LedgerPaths,
        stable_ids: Any,
        *,
        intake_variant_bindings: Optional[Mapping[str, tuple[str, str]]] = None,
    ) -> None:
        """Build exact test authority, including trusted family bindings."""

        context = make_authority_context(stable_ids)
        if intake_variant_bindings:
            family_bindings = {
                stable_id: {
                    "binding_state": "variant",
                    "representative_stable_id": representative,
                    "variant_token": variant,
                    "derivation_rule_identity": stable_hash(
                        "menagerie-family-variant-derivation-v1"
                    ),
                }
                for stable_id, (representative, variant) in intake_variant_bindings.items()
            }
            context = replace(context, family_bindings=family_bindings)
        super().__init__(paths, context)

    def append_model(self, model: Mapping[str, Any]) -> Any:
        """Project reducer-owned v3 fields before exercising admission invariants."""

        return super().append_model(self.prepare_model(model))


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
            # Canonical name is author-gated and therefore inherited under the
            # representative-only template lane. The trusted variant token carries
            # the size identity unless a full independent author/checker path runs.
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
        schema_version="menagerie.crawler.model.v3",
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
    return rebind_attempt_raw_proof(attempt)


def _append_unprepared(reducer: ProductionCanonicalReducer, model: Mapping[str, Any]) -> Any:
    """Append a caller-shaped model without the legacy test adapter re-projecting it.

    Parameters
    ----------
    reducer:
        Open production reducer wrapped by the legacy fixture adapter.
    model:
        Candidate whose reducer-owned projections may have been adversarially changed.

    Returns
    -------
    Any
        Production append result when the candidate is admitted.
    """

    return ProductionCanonicalReducer.append_model(reducer, model)


def _rebind_gate_proof(gate: dict[str, Any]) -> dict[str, Any]:
    """Recompute the current-v3 gate envelope proof after a fixture mutation.

    Parameters
    ----------
    gate:
        Mutable current-v3 checker envelope.

    Returns
    -------
    dict[str, Any]
        The same gate with its exact result-envelope self-hash rebound.
    """

    gate["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in gate.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
    return gate


def _dual_mode_run_case(
    relation: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a current-v3 dual-mode run with an exact output relationship.

    Parameters
    ----------
    relation:
        ``equal`` or ``structural``.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        Model plus authenticated train and eval attempts.
    """

    model = make_model(accepted=True)
    model["modes"]["meaningful_modes"] = ["train", "eval"]
    model["external_metadata"]["modes"]["meaningful_modes"] = ["train", "eval"]
    _bind_model_identities(model)
    train = _attempt_for_model(model, "attempt-train")
    train["mode"] = "train"
    train["invocation"]["mode"] = "train"
    train["worker_receipt"]["mode"] = "train"
    rebind_attempt_raw_proof(train)
    evaluated = _attempt_for_model(model, "attempt-eval")
    evaluated["ledger_seq"] = 2
    if relation == "structural":
        evaluated["worker_receipt"]["output_signature"] = {
            "tree": {"tuple": [{"leaf": 0}, {"leaf": 1}]},
            "leaves": [
                deepcopy(model["observed"]["output_signature"]["leaves"][0]),
                deepcopy(model["observed"]["output_signature"]["leaves"][0]),
            ],
        }
    elif relation != "equal":
        raise AssertionError(f"unsupported mode relation: {relation}")
    rebind_attempt_raw_proof(evaluated)
    attempt_ids = ["attempt-train", "attempt-eval"]
    model["execution"]["accepted_attempt_ids"] = attempt_ids
    model["status"]["attempt_ids"] = attempt_ids
    model["observed"]["measurement_attempt_ids"] = attempt_ids
    return model, train, evaluated


def _terminal_case(
    status_code: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Build one failed, deferred, or skipped terminal with exact attempt evidence.

    Parameters
    ----------
    status_code:
        Closed terminal code.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]
        Model, referenced attempt, and optional metadata gate.
    """

    if status_code == "failed:forward":
        attempt = make_failed_attempt(stage="forward", reason_code="exception")
        model = make_model(status_code=status_code)
        model["status"]["reason_code"] = "exception"
        return bind_terminal_attempts(model, [attempt]), attempt, None
    if status_code == "deferred:needs-cuda":
        attempt = make_failed_attempt(stage="environment", reason_code="probe-failed")
        attempt["result"] = "observed"
        attempt["error"] = None
        attempt["defer_evidence"] = {
            "target_status": status_code,
            "source_ids": ["source-1"],
            "probe_attempt_ids": [],
            "explanation": "source requires CUDA",
        }
        model = make_model(status_code=status_code)
        return bind_terminal_attempts(model, [attempt]), attempt, None
    if status_code == "skipped:no-description":
        attempt = make_failed_attempt()
        model = make_model(accepted=True, status_code=status_code)
        model["source_resolution"]["rung"] = "R5_SKIP"
        model["source_resolution"]["attempted_rungs"][0]["rung"] = "R5_SKIP"
        bind_terminal_attempts(model, [attempt])
        _bind_model_identities(model)
        gate = make_gate(["m_example"])
        gate["items"][0]["vet_identity"] = model["accuracy_gate"]["vet_identity"]
        gate["items"][0]["rung_check"]["selected_rung"] = "R5_SKIP"
        return model, attempt, gate
    raise AssertionError(f"unsupported terminal fixture: {status_code}")


def _observed_defer_attempt(*, attempt_id: str, work_id: str = "work-m_example") -> dict[str, Any]:
    """Build one schema-valid non-awarding observation for deferral proofs.

    Parameters
    ----------
    attempt_id, work_id:
        Immutable attempt and work-generation identities.

    Returns
    -------
    dict[str, Any]
        Current-v3 parent-attested observation without award authority.
    """

    attempt = make_attempt(attempt_id=attempt_id)
    attempt["work_id"] = work_id
    attempt["result"] = "observed"
    attempt["environment"] = None
    attempt["identities"]["environment"] = None
    attempt["identities"]["execution"] = None
    attempt["worker_receipt"]["present"] = False
    attempt["raw_award_receipt"] = None
    attempt["raw_award_receipt_sha256"] = None
    attempt["supervisor_observation"]["stdout_sha256"] = None
    attempt["supervisor_observation"]["stderr_sha256"] = None
    attempt["supervisor_observation"]["stdout_completion_line"] = None
    return rebind_nonaward_parent_proof(attempt)


def _stage_deferred_admission_control(
    reducer: ProductionCanonicalReducer,
    tmp_path: Path,
    *,
    manifest_source_ids: tuple[str, ...] = ("source-1",),
    gate_source_ids: tuple[str, ...] = ("source-1",),
    defer_source_ids: tuple[str, ...] = ("source-1",),
    probe_attack: Optional[str] = None,
) -> dict[str, Any]:
    """Stage a genuine terminal transaction and return its projected model.

    Parameters
    ----------
    reducer, tmp_path:
        Open production reducer and isolated physical mirror root.
    manifest_source_ids, gate_source_ids, defer_source_ids:
        Real source inventory, accepted gate subset, and attempted defer binding.
    probe_attack:
        Optional missing/wrong/negative/unsupported probe mutation.

    Returns
    -------
    dict[str, Any]
        Reducer-projected deferred model ready for direct admission.
    """

    model = make_model(status_code="deferred:needs-cuda")
    model["evidence"]["excerpts"][0]["supports"].append("needs-cuda")
    source_template = model["source_resolution"]["sources"][0]
    contents = {
        source_id: f"terminal source bytes for {source_id}".encode()
        for source_id in manifest_source_ids
    }
    sources: list[dict[str, Any]] = []
    artifact_inputs: list[ArtifactInput] = []
    for index, source_id in enumerate(manifest_source_ids):
        source = deepcopy(source_template)
        source.update(
            {
                "source_id": source_id,
                "url": f"https://example.com/{source_id}",
                "revision": f"revision-{index}",
                "locator": f"{source_id}.py",
                "content_sha256": hash_bytes(contents[source_id]),
                "byte_count": len(contents[source_id]),
            }
        )
        sources.append(source)
        artifact_inputs.append(
            ArtifactInput(
                content=contents[source_id],
                content_sha256=hash_bytes(contents[source_id]),
                logical_role="source",
                logical_path=f"menagerie/crawler/source_cas/{source_id}.source",
                source_id=source_id,
                origin=ArtifactOrigin(url=str(source["url"]), revision=str(source["revision"])),
                fetch_recipe=str(source["fetch_recipe"]),
                evidence_ids=("evidence-1",) if source_id == "source-1" else (),
                media_type=str(source["media_type"]),
            )
        )
    model["source_resolution"]["sources"] = sources
    source_manifest: dict[str, Any] = {"sources": deepcopy(sources)}
    source_manifest["manifest_sha256"] = stable_hash(source_manifest["sources"])
    payload = {
        "arm": "DEFER_RECOMMENDATION",
        "platform": "cuda",
        "source_ids": list(manifest_source_ids),
        "evidence_ids": ["evidence-1"],
        "evidence_identity": model["evidence"]["evidence_identity"],
        "license_identity": stable_hash(model["licenses"]),
    }
    context = reducer.context
    payload["handoff_execution"] = bind_handoff_execution(
        make_author_proposal("m_example"),
        context=context,
        work_id="work-m_example",
        campaign_id="campaign-m_example",
        source_manifest_identity=source_manifest["manifest_sha256"],
    )
    payload["recommendation_sha256"] = stable_hash(payload)
    author_result = {
        "schema_version": "menagerie.crawler.author-result.v4",
        "result_id": "result-deferred-control",
        "result_sha256": stable_hash("pending"),
        "kind": "DEFER_RECOMMENDATION",
        "stable_id": "m_example",
        "work_id": "work-m_example",
        "campaign_id": "campaign-m_example",
        "created_at": "2026-07-14T12:00:00Z",
        "author_identity": context.author_model_identity,
        "prompt_identity": context.author_prompt_identity,
        "dispatcher_identity": context.author_dispatcher_identity,
        "source_manifest_identity": source_manifest["manifest_sha256"],
        "intake_snapshot_id": context.active_intake_snapshot_id,
        "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
        "intake_item_sha256": stable_hash(context.intake_by_stable_id["m_example"]),
        "payload": payload,
    }
    author_result["result_sha256"] = stable_hash(
        {key: value for key, value in author_result.items() if key != "result_sha256"}
    )
    mirrors = MirrorStore(
        tmp_path / "mirror-public",
        tmp_path / "mirror-private",
        tmp_path / "mirror-local",
    )
    stage_private_artifact(
        tuple(artifact_inputs),
        context=context,
        stable_id="m_example",
        work_id="work-m_example",
        author_result=author_result,
        proposal=None,
        source_manifest=source_manifest,
        mirrors=mirrors,
        ledger=reducer.artifact_ledger,
        created_at="2026-07-14T12:00:00Z",
    )
    gate = make_gate(["m_example"], gate_kind="fidelity")
    gate["gate_kind"] = "terminal_disposition"
    gate["items"][0]["terminal_disposition"] = {
        "author_result_id": author_result["result_id"],
        "author_result_sha256": author_result["result_sha256"],
        "kind": author_result["kind"],
        "predicate": "needs-cuda",
        "handoff_proposal_id": payload["handoff_execution"]["proposal"]["proposal_id"],
        "handoff_sha256": payload["handoff_execution"]["handoff_sha256"],
        "verdict": "accepted",
        "source_manifest_identity": source_manifest["manifest_sha256"],
        "source_ids": list(gate_source_ids),
        "evidence_identity": payload["evidence_identity"],
        "evidence_ids": ["evidence-1"],
        "license_identity": payload["license_identity"],
        "findings": [],
    }
    _rebind_gate_proof(gate)
    reducer.append_gate(gate)

    control_attempt = _observed_defer_attempt(attempt_id="attempt-deferred-control")
    control_attempt["defer_evidence"] = {
        "target_status": "deferred:needs-cuda",
        "source_ids": list(gate_source_ids),
        "probe_attempt_ids": [],
        "explanation": "source requires CUDA",
    }
    reducer.append_attempt(control_attempt)
    bind_terminal_attempts(model, [control_attempt])
    prepared = reducer.prepare_model(model)
    if defer_source_ids == gate_source_ids and probe_attack is None:
        return prepared

    probes: list[dict[str, Any]] = []
    probe_ids: list[str] = []
    if probe_attack == "missing":
        probe_ids = ["probe-missing"]
    elif probe_attack in {"positive", "wrong-work", "wrong-target", "unsupported-claim"}:
        probe = _observed_defer_attempt(
            attempt_id=f"probe-{probe_attack}",
            work_id=("work-other" if probe_attack == "wrong-work" else "work-m_example"),
        )
        probe["capability_observation"] = {
            "claim": (
                "needs-x86"
                if probe_attack in {"wrong-target", "unsupported-claim"}
                else "needs-cuda"
            ),
            "supported": True,
        }
        if probe_attack == "wrong-target":
            probe["defer_evidence"] = {
                "target_status": "deferred:needs-x86",
                "source_ids": list(gate_source_ids),
                "probe_attempt_ids": [],
                "explanation": "probe describes the wrong target",
            }
        probes = [probe]
        probe_ids = [str(probe["attempt_id"])]
    elif probe_attack == "negative":
        probe = make_failed_attempt(
            attempt_id="probe-negative", stage="environment", reason_code="probe-failed"
        )
        probe["work_id"] = "work-m_example"
        probes = [probe]
        probe_ids = ["probe-negative"]
    elif probe_attack is not None:
        raise AssertionError(f"unsupported probe attack: {probe_attack}")
    for sequence, probe in enumerate(probes, start=2):
        probe["ledger_seq"] = sequence
        reducer.append_attempt(probe)

    attempt = _observed_defer_attempt(attempt_id="attempt-deferred-adversarial")
    attempt["attempt_no"] = len(probes) + 2
    attempt["ledger_seq"] = len(probes) + 2
    attempt["defer_evidence"] = {
        "target_status": "deferred:needs-cuda",
        "source_ids": list(defer_source_ids),
        "probe_attempt_ids": probe_ids,
        "explanation": "source requires CUDA",
    }
    reducer.append_attempt(attempt)
    if probe_attack == "positive":
        bind_terminal_attempts(model, [control_attempt, *probes, attempt])
        return reducer.prepare_model(model)
    return prepared


@pytest.mark.parametrize(
    "status_code",
    ["failed:forward", "deferred:needs-cuda", "skipped:no-description"],
)
def test_terminal_case_fixture_covers_later_currency_families(status_code: str) -> None:
    """Keep the shared terminal fixture live for the Round-15 W7 matrix.

    Parameters
    ----------
    status_code:
        Representative failed, deferred, or skipped terminal family.
    """

    model, attempt, gate = _terminal_case(status_code)
    assert model["status"]["code"] == status_code
    assert attempt["attempt_id"] in model["status"]["attempt_ids"]
    assert (gate is not None) == status_code.startswith("skipped:")


def test_deferred_terminal_control_passes_production_reducer_admission(tmp_path: Path) -> None:
    """A genuine staged v3 deferral remains the positive A-05 control.

    Parameters
    ----------
    tmp_path:
        Isolated canonical ledgers and physical mirrors.
    """

    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        prepared = _stage_deferred_admission_control(reducer, tmp_path)
        assert _append_unprepared(reducer, prepared).appended


def test_deferred_terminal_positive_capability_probe_is_persisted_and_admitted(
    tmp_path: Path,
) -> None:
    """A schema-valid positive capability probe must make A-05 admission reachable.

    Parameters
    ----------
    tmp_path:
        Isolated canonical ledgers and physical mirrors.
    """

    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        prepared = _stage_deferred_admission_control(
            reducer,
            tmp_path,
            probe_attack="positive",
        )
        assert _append_unprepared(reducer, prepared).appended
        probes = [
            attempt
            for attempt in scan_jsonl(reducer.ledger_paths.attempts)
            if attempt.get("capability_observation") is not None
        ]
        assert len(probes) == 1
        assert probes[0]["capability_observation"] == {
            "claim": "needs-cuda",
            "supported": True,
        }


@pytest.mark.parametrize(
    ("manifest_source_ids", "gate_source_ids", "defer_source_ids"),
    (
        (("source-1",), ("source-1",), ("source-fabricated",)),
        (("source-1", "source-2"), ("source-1",), ("source-1", "source-2")),
        (("source-1", "source-2"), ("source-1", "source-2"), ("source-1",)),
        (("source-1", "source-2"), ("source-1",), ("source-2",)),
    ),
)
def test_deferred_terminal_rejects_nonexistent_extra_dropped_and_replaced_sources(
    tmp_path: Path,
    manifest_source_ids: tuple[str, ...],
    gate_source_ids: tuple[str, ...],
    defer_source_ids: tuple[str, ...],
) -> None:
    """A-05 defer attempts must name exactly the terminal gate's real source set.

    Parameters
    ----------
    tmp_path:
        Isolated canonical ledgers and physical mirrors.
    manifest_source_ids, gate_source_ids, defer_source_ids:
        Real manifest inventory, accepted gate set, and one adversarial attempt set.
    """

    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        prepared = _stage_deferred_admission_control(
            reducer,
            tmp_path,
            manifest_source_ids=manifest_source_ids,
            gate_source_ids=gate_source_ids,
            defer_source_ids=defer_source_ids,
        )
        with pytest.raises(
            ReductionError,
            match="deferral attempt source set is not gate-exact",
        ):
            _append_unprepared(reducer, prepared)


@pytest.mark.parametrize(
    "probe_attack",
    ["missing", "wrong-work", "wrong-target", "negative", "unsupported-claim"],
)
def test_deferred_terminal_rejects_missing_wrong_negative_and_unsupported_probes(
    tmp_path: Path, probe_attack: str
) -> None:
    """A-05 named probes require positive same-work structured capability proof.

    Parameters
    ----------
    tmp_path:
        Isolated canonical ledgers and physical mirrors.
    probe_attack:
        Missing, cross-work, wrong-target, negative, or unsupported probe mutation.
    """

    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        prepared = _stage_deferred_admission_control(
            reducer,
            tmp_path,
            probe_attack=probe_attack,
        )
        with pytest.raises(
            ReductionError,
            match="deferral probe lacks a structured positive same-work capability observation",
        ):
            _append_unprepared(reducer, prepared)


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


def test_reducer_rejects_evidence_free_terminal(tmp_path: Path) -> None:
    """A shaped failure cannot close the partition without canonical evidence."""

    model = bind_terminal_attempts(make_model(status_code="failed:source"), [])
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        with pytest.raises(ReductionError, match="does not resolve one work identity"):
            reducer.append_model(model)


def test_bad_parentage_is_rejected(tmp_path: Path) -> None:
    """A superseding revision must point to the current exact parent.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    paths = _paths(tmp_path)
    failed_attempt = make_failed_attempt()
    first = bind_terminal_attempts(make_model(status_code="failed:source"), [failed_attempt])
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
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

    failed_attempt = make_failed_attempt()
    model = bind_terminal_attempts(make_model(status_code="failed:source"), [failed_attempt])
    if location == "status":
        model["status"][field] = "raw externally controlled text"
    elif location == "human_review":
        model["status"]["human_review"][field] = "raw externally controlled text"
    else:
        model["c07_tripwire"] = {field: "raw externally controlled text"}
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
        with pytest.raises(ReductionError, match="unredacted externally controlled text"):
            reducer.append_model(model)


def test_first_revision_rejects_public_supersession_lineage(tmp_path: Path) -> None:
    """A first revision cannot claim to supersede an unrelated public revision."""

    failed_attempt = make_failed_attempt()
    model = bind_terminal_attempts(make_model(status_code="failed:source"), [failed_attempt])
    model["status"]["supersedes_revision"] = "sha256:" + "f" * 64
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
        with pytest.raises(ReductionError, match="first model revision cannot supersede"):
            reducer.append_model(model)


def test_later_revision_binds_public_supersession_to_parent(tmp_path: Path) -> None:
    """The public supersession field must equal the reducer-authorized parent."""

    paths = _paths(tmp_path)
    failed_attempt = make_failed_attempt()
    first = bind_terminal_attempts(make_model(status_code="failed:source"), [failed_attempt])
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
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
    failed_attempt = make_failed_attempt()
    first = bind_terminal_attempts(make_model(status_code="failed:source"), [failed_attempt])
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
        persisted = reducer.append_model(first).record
        second = make_model(status_code="failed:source")
        second["record_seq"] = 2
        second["parent_revision"] = persisted["record_revision"]
        second["status"]["supersedes_revision"] = "sha256:" + "f" * 64
        malformed = reducer.prepare_model(second)
    with JsonlLedger(paths.models, str(persisted["schema_version"])) as ledger:
        ledger.append(malformed)
    with pytest.raises(ReductionError, match="persisted status.supersedes_revision"):
        CanonicalReducer(paths, ["m_example"])


def test_requeue_grant_rejects_wrong_exact_parent_binding(tmp_path: Path) -> None:
    """A durable grant cannot authorize a supersession of a different parent revision."""

    paths = _paths(tmp_path)
    failed_attempt = make_failed_attempt()
    first = bind_terminal_attempts(make_model(status_code="failed:source"), [failed_attempt])
    first["status"]["human_review"] = {
        "required": True,
        # The test exercises requeue lineage; raw review diagnostics are rejected separately.
        "reason": None,
        "queue": "crawler-human-review",
        "requested_at": first["created_at"],
    }
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
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
        attempt = make_failed_attempt(attempt_id="attempt-requeue")
        attempt.pop("ledger_seq")
        attempt.pop("payload_sha256")
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
    attempt = make_failed_attempt()
    model = bind_terminal_attempts(
        make_model(accepted=True, status_code="failed:source"), [attempt]
    )
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
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


@pytest.mark.parametrize(
    ("relation", "field", "replacement"),
    (
        ("equal", "train_eval_divergence", "not-applicable"),
        ("structural", "train_eval_divergence", "none"),
        ("equal", "train_eval_divergence", "structural"),
        ("equal", "train_eval_divergence", "statistical"),
        ("equal", "divergence_evidence", "caller-invented comparison evidence"),
    ),
)
def test_append_model_rejects_caller_mode_summary_contradictions(
    tmp_path: Path,
    relation: str,
    field: str,
    replacement: str,
) -> None:
    """Both-present M-01/H2 mode lies reject after reducer projection.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    relation, field, replacement:
        Authenticated output relationship and one contradictory caller claim.
    """

    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    model, train, evaluated = _dual_mode_run_case(relation)
    gate = make_gate(stable_ids)
    gate["items"][0]["vet_identity"] = model["accuracy_gate"]["vet_identity"]
    _rebind_gate_proof(gate)
    with CanonicalReducer(_paths(tmp_path), stable_ids) as reducer:
        reducer.append_gate(gate)
        reducer.append_attempt(train)
        reducer.append_attempt(evaluated)
        prepared = reducer.prepare_model(model)
        prepared["modes"][field] = replacement
        with pytest.raises(
            ReductionError,
            match=rf"modes\.{field} contradicts reducer-derived mode authority",
        ):
            _append_unprepared(reducer, prepared)


@pytest.mark.parametrize("relation", ["equal", "structural"])
def test_append_model_accepts_reducer_derived_dual_mode_controls(
    tmp_path: Path, relation: str
) -> None:
    """Equal and different authenticated mode controls pass unchanged.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    relation:
        Equal or structurally different authenticated outputs.
    """

    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    model, train, evaluated = _dual_mode_run_case(relation)
    gate = make_gate(stable_ids)
    gate["items"][0]["vet_identity"] = model["accuracy_gate"]["vet_identity"]
    _rebind_gate_proof(gate)
    with CanonicalReducer(_paths(tmp_path), stable_ids) as reducer:
        reducer.append_gate(gate)
        reducer.append_attempt(train)
        reducer.append_attempt(evaluated)
        assert _append_unprepared(reducer, reducer.prepare_model(model)).appended


def _failed_dual_mode_case() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a failed terminal with unique current-v3 train/eval attempts.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        Model and exact train/eval terminal attempts.
    """

    train = make_failed_attempt(
        attempt_id="attempt-train", stage="forward", reason_code="exception"
    )
    train["mode"] = "train"
    train["invocation"]["mode"] = "train"
    train["worker_receipt"]["mode"] = "train"
    evaluated = make_failed_attempt(
        attempt_id="attempt-eval", stage="forward", reason_code="exception"
    )
    evaluated["attempt_no"] = 2
    evaluated["ledger_seq"] = 2
    model = make_model(status_code="failed:forward")
    model["status"]["reason_code"] = "exception"
    model["modes"]["meaningful_modes"] = ["train", "eval"]
    bind_terminal_attempts(model, [train, evaluated])
    return model, train, evaluated


@pytest.mark.parametrize(
    ("attack", "expected"),
    (
        ("same-attempt-both-modes", "per_mode_run contradicts reducer-derived"),
        ("unique-attempt-wrong-mode", "per_mode_run contradicts reducer-derived"),
        ("status-result-disagreement", "per_mode_run contradicts reducer-derived"),
    ),
)
def test_terminal_mode_map_rejects_duplicate_wrong_and_status_inconsistent_attempts(
    tmp_path: Path, attack: str, expected: str
) -> None:
    """A-06 caller maps cannot contradict the unique reducer-derived mode map.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    attack, expected:
        One adversarial relationship and its production rejection rule.
    """

    model, train, evaluated = _failed_dual_mode_case()
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(train)
        reducer.append_attempt(evaluated)
        prepared = reducer.prepare_model(model)
        if attack == "same-attempt-both-modes":
            prepared["modes"]["per_mode_run"]["eval"]["attempt_id"] = "attempt-train"
        elif attack == "unique-attempt-wrong-mode":
            prepared["modes"]["per_mode_run"] = {
                "train": {"attempt_id": "attempt-eval", "status": "failed"},
                "eval": {"attempt_id": "attempt-train", "status": "failed"},
            }
        elif attack == "status-result-disagreement":
            prepared["modes"]["per_mode_run"]["train"]["status"] = "observed"
        else:
            raise AssertionError(f"unsupported attack: {attack}")
        with pytest.raises(ReductionError, match=expected):
            _append_unprepared(reducer, prepared)


def test_cache_read_attempted_blocks_otherwise_valid_run(tmp_path: Path) -> None:
    """Parent-observed cache-backed reads poison an otherwise complete receipt."""

    paths = _paths(tmp_path)
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    attempt = make_attempt()
    attempt["policy_observation"]["cache_read_attempted"] = True
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        with pytest.raises(ReductionError, match="policy_observation.clean_flags"):
            reducer.append_attempt(attempt)


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
        with pytest.raises(ReductionError, match="inherited author-gated leaves.*taxonomy.family"):
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
        with pytest.raises(
            ReductionError,
            match="inherited author-gated leaves.*implementation.library_recipe.module",
        ):
            reducer.append_model(variant)


def test_reducer_derives_release_and_rejects_pending_true_claim(tmp_path: Path) -> None:
    """A candidate cannot publish pending metadata by flipping release true."""

    failed_attempt = make_failed_attempt()
    model = bind_terminal_attempts(
        make_model(accepted=False, status_code="failed:source"), [failed_attempt]
    )
    model["completeness"]["release_eligible"] = True
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
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

        projected_current = reducer.current_records
        assert "m_variant" not in projected_current
        report = completeness_report(["m_rep", "m_variant"], projected_current)
        assert report.partition.missing_ids == frozenset({"m_variant"})
        assert not report.complete
        assert not record_is_release_eligible(variant, projected_current)
        context = reducer.context

    materialized = materialize_current(paths, context=context)
    assert "m_rep" in materialized
    assert "m_variant" not in materialized


def test_dependency_stale_representative_cascades_to_current_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay-stale representative cannot survive through raw-current authority."""

    paths = _paths(tmp_path)
    gate_ids = ["m_rep", *(f"m_{index}" for index in range(9))]
    stable_ids = [*gate_ids, "m_variant"]
    bindings = {"m_variant": ("m_rep", "Large")}
    representative = make_model("m_rep", accepted=True, attempt_id="attempt-rep")
    with CanonicalReducer(
        paths,
        stable_ids,
        intake_variant_bindings=bindings,
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

    original = reducer_module._referenced_evidence_repasses

    def stale_representative(
        replay: CanonicalReducer,
        record: dict[str, Any],
        attempts: tuple[dict[str, Any], ...],
        gates: tuple[dict[str, Any], ...],
    ) -> None:
        """Simulate a representative-only live dependency failure."""

        if record.get("stable_id") == "m_rep":
            raise ReductionError("representative dependency changed")
        original(replay, record, attempts, gates)

    monkeypatch.setattr(reducer_module, "_referenced_evidence_repasses", stale_representative)
    projection = project_dependency_current(
        paths,
        context=reducer.context,
    )
    assert "m_rep" not in projection.current_records
    assert "m_variant" not in projection.current_records
    assert "representative dependency changed" in projection.stale_reasons["m_rep"]
    assert "representative" in projection.stale_reasons["m_variant"]


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
    rebind_attempt_raw_proof(attempt)
    with CanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        with pytest.raises(ReductionError, match=f"requires {receipt_field}=true"):
            reducer.append_attempt(attempt)


def test_run_award_rejects_null_input_signature(tmp_path: Path) -> None:
    """A succeeded label cannot replace a structurally complete input signature."""

    paths = _paths(tmp_path)
    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    attempt = make_attempt()
    attempt["worker_receipt"]["input_signature"] = None
    rebind_attempt_raw_proof(attempt)
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
    rebind_attempt_raw_proof(attempt)
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
    # Rebind the synthetic checker envelope so this semantic-admission test reaches
    # the authored-leaf rule instead of correctly tripping current-proof replay first.
    gate["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in gate.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
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
