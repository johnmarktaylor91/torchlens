"""Canonical reducer and sole model/attempt/gate ledger writer."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION,
    FAILURE_REASON_CODES,
    GATE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    TERMINAL_STATUS_CODES,
)
from menagerie.crawler.models import AppendResult, JsonObject, LedgerPaths
from menagerie.crawler.recordio import JsonlLedger, LedgerConflictError
from menagerie.crawler.state import _select_current


class ReductionError(ValueError):
    """Raised when a proposed canonical fact violates reducer invariants."""


class CanonicalReducer:
    """Exclusive canonical writer enforcing parentage, gates, runs, and statuses.

    Parameters
    ----------
    ledgers:
        Paths to the three canonical append-only ledgers.
    intake_ids:
        Stable IDs in the trusted immutable intake snapshot.
    """

    def __init__(self, ledgers: LedgerPaths, intake_ids: Iterable[str]) -> None:
        """Acquire all canonical writer locks and load current facts.

        Parameters
        ----------
        ledgers:
            Paths to canonical ledgers.
        intake_ids:
            Stable IDs in trusted intake.
        """

        self.ledger_paths = ledgers
        self.intake_ids = frozenset(intake_ids)
        opened: list[JsonlLedger] = []
        try:
            self._models = JsonlLedger(ledgers.models, MODEL_SCHEMA_VERSION)
            opened.append(self._models)
            self._attempts = JsonlLedger(ledgers.attempts, ATTEMPT_SCHEMA_VERSION)
            opened.append(self._attempts)
            self._gates = JsonlLedger(ledgers.gates, GATE_SCHEMA_VERSION)
            opened.append(self._gates)
        except Exception:
            for ledger in reversed(opened):
                ledger.close()
            raise
        self._current = _select_current(self._models.records)
        unknown = set(self._current) - self.intake_ids
        if unknown:
            self.close()
            raise ReductionError(f"model revisions exist outside intake: {sorted(unknown)}")

    def __enter__(self) -> "CanonicalReducer":
        """Return this exclusive reducer.

        Returns
        -------
        CanonicalReducer
            This reducer.
        """

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release every canonical writer lock.

        Parameters
        ----------
        exc_type, exc_value, traceback:
            Context-manager exception state.
        """

        self.close()

    def close(self) -> None:
        """Release every canonical writer lock idempotently."""

        for ledger_name in ("_gates", "_attempts", "_models"):
            ledger = getattr(self, ledger_name, None)
            if ledger is not None:
                ledger.close()

    @property
    def current_records(self) -> Mapping[str, JsonObject]:
        """Return defensive copies of materialized current revisions.

        Returns
        -------
        Mapping[str, dict[str, Any]]
            Stable-ID keyed current model revisions.
        """

        return deepcopy(self._current)

    def append_attempt(self, attempt: Mapping[str, Any]) -> AppendResult:
        """Append one immutable attempt after parent/work validation.

        Parameters
        ----------
        attempt:
            Attempt payload; ledger sequence and payload hash may be omitted.

        Returns
        -------
        AppendResult
            Idempotent append result.
        """

        stable_id = attempt.get("stable_id")
        if stable_id is not None and stable_id not in self.intake_ids:
            raise ReductionError(f"attempt stable_id is outside intake: {stable_id}")
        parent_id = attempt.get("parent_attempt_id")
        by_id = {record["attempt_id"]: record for record in self._attempts.records}
        if parent_id is not None:
            parent = by_id.get(parent_id)
            if parent is None:
                raise ReductionError(f"missing parent attempt {parent_id}")
            if parent["work_id"] != attempt.get("work_id"):
                raise ReductionError("parent attempt belongs to a different work identity")
            if int(attempt.get("attempt_no", 0)) <= int(parent["attempt_no"]):
                raise ReductionError("attempt_no must increase after its parent")
        error = attempt.get("error")
        if attempt.get("result") == "failed":
            if not isinstance(error, Mapping):
                raise ReductionError("failed attempt requires a structured error")
            stage = error.get("stage")
            if stage != attempt.get("stage"):
                raise ReductionError("attempt and error stages must match")
            if error.get("reason_code") not in FAILURE_REASON_CODES.get(str(stage), frozenset()):
                raise ReductionError("attempt error reason is not allowed for its stage")
        elif error is not None:
            raise ReductionError("non-failed attempt cannot carry an error")
        return self._attempts.append(attempt)

    def append_gate(self, gate: Mapping[str, Any]) -> AppendResult:
        """Append one immutable checker gate after intake/integrity checks.

        Parameters
        ----------
        gate:
            Gate payload; ledger sequence and payload hash may be omitted.

        Returns
        -------
        AppendResult
            Idempotent append result.
        """

        items = gate.get("items", [])
        stable_ids = [item.get("stable_id") for item in items]
        if gate.get("batch_size") != len(items):
            raise ReductionError("gate batch_size must equal the number of items")
        if len(stable_ids) != len(set(stable_ids)):
            raise ReductionError("gate items must have unique stable IDs")
        for item in items:
            if item.get("stable_id") not in self.intake_ids:
                raise ReductionError(
                    f"gate item stable_id is outside intake: {item.get('stable_id')}"
                )
            verdict = item.get("verdict")
            integrity = item.get("integrity", {}).get("verdict")
            field_verdicts = [check.get("verdict") for check in item.get("field_checks", [])]
            if verdict == "accurate" and (
                integrity != "accurate" or any(value != "accurate" for value in field_verdicts)
            ):
                raise ReductionError(
                    "an accurate gate item requires accurate integrity/field checks"
                )
        return self._gates.append(gate)

    def append_model(self, model: Mapping[str, Any]) -> AppendResult:
        """Validate and append a full canonical model revision.

        Parameters
        ----------
        model:
            Complete model revision. The reducer computes its canonical revision hash.

        Returns
        -------
        AppendResult
            Idempotent append result.

        Raises
        ------
        ReductionError
            If parentage, taxonomy, gate, attempt, mode, or partition rules fail.
        """

        candidate = deepcopy(dict(model))
        stable_id = candidate.get("stable_id")
        if stable_id not in self.intake_ids:
            raise ReductionError(f"model stable_id is outside intake: {stable_id}")
        for existing in self._models.records:
            if existing["stable_id"] != stable_id:
                continue
            logical_existing = {
                key: value
                for key, value in existing.items()
                if key not in {"record_seq", "record_revision"}
            }
            logical_candidate = {
                key: value
                for key, value in candidate.items()
                if key not in {"record_seq", "record_revision"}
            }
            if logical_existing == logical_candidate:
                return AppendResult(deepcopy(existing), appended=False)
        previous = self._current.get(str(stable_id))
        expected_parent = previous["record_revision"] if previous is not None else None
        if candidate.get("parent_revision") != expected_parent:
            raise ReductionError(
                f"bad parentage for {stable_id}: expected {expected_parent!r}, "
                f"received {candidate.get('parent_revision')!r}"
            )
        if previous is None and candidate.get("record_seq") not in {None, 1}:
            raise ReductionError("a first model revision must start at record_seq 1 or omit it")
        if previous is not None and candidate.get("record_seq") is not None:
            if int(candidate["record_seq"]) <= int(previous["record_seq"]):
                raise ReductionError("record_seq must increase over the current revision")
        self._validate_status(candidate)
        self._validate_source(candidate)
        self._validate_gates(candidate)
        self._validate_execution(candidate)
        try:
            result = self._models.append(candidate)
        except LedgerConflictError as exc:
            raise ReductionError(str(exc)) from exc
        self._current[str(stable_id)] = deepcopy(result.record)
        return result

    def _validate_status(self, model: Mapping[str, Any]) -> None:
        """Validate closed terminal code/kind/stage/reason relationships.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        status = model.get("status", {})
        code = status.get("code")
        kind = status.get("kind")
        if code not in TERMINAL_STATUS_CODES:
            raise ReductionError(f"unknown terminal status code: {code!r}")
        expected_kind = str(code).split(":", 1)[0]
        if kind != expected_kind:
            raise ReductionError(f"status kind {kind!r} does not match code {code!r}")
        stage = status.get("stage")
        reason = status.get("reason_code")
        if kind == "failed":
            expected_stage = str(code).split(":", 1)[1]
            if stage != expected_stage:
                raise ReductionError("failure stage must match the failed status code")
            if reason not in FAILURE_REASON_CODES[expected_stage]:
                raise ReductionError(
                    f"reason {reason!r} is not allowed for failure stage {expected_stage!r}"
                )
            if status.get("traceback") is None and status.get("no_traceback_reason") is None:
                raise ReductionError("failure requires a traceback or explicit no_traceback_reason")
            if code == "failed:accuracy-gate" and not status.get("human_review", {}).get(
                "required"
            ):
                raise ReductionError("failed:accuracy-gate requires human review")
        elif stage is not None or reason is not None:
            raise ReductionError("non-failed statuses cannot carry failure stage/reason")

    def _validate_source(self, model: Mapping[str, Any]) -> None:
        """Enforce the mandatory public source-link invariant.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        resolution = model.get("source_resolution", {})
        sources = resolution.get("sources", [])
        primary = resolution.get("primary_source_id")
        if not sources or not any(
            source.get("source_id") == primary
            and str(source.get("url", "")).startswith(("http://", "https://"))
            for source in sources
        ):
            raise ReductionError("missing mandatory exact public primary source link")
        if model.get("status", {}).get("code") != "failed:source" and (
            resolution.get("mandatory_link_status") != "ok"
        ):
            raise ReductionError(
                "non-source-failure terminal records require mandatory_link_status=ok"
            )

    def _gate_item(
        self, gate_id: Optional[str], stable_id: str
    ) -> Optional[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        """Find a gate and its unique item for a stable ID.

        Parameters
        ----------
        gate_id:
            Referenced gate ID.
        stable_id:
            Model stable ID.

        Returns
        -------
        tuple[Mapping[str, Any], Mapping[str, Any]] | None
            Gate envelope and matching item, if present.
        """

        if gate_id is None:
            return None
        for gate in self._gates.records:
            if gate["gate_id"] != gate_id:
                continue
            matches = [item for item in gate["items"] if item["stable_id"] == stable_id]
            if len(matches) != 1:
                return None
            return gate, matches[0]
        return None

    def _validate_gates(self, model: Mapping[str, Any]) -> None:
        """Enforce block-at-write metadata and required fidelity gates.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        stable_id = str(model["stable_id"])
        accuracy = model.get("accuracy_gate", {})
        if model.get("authored_metadata_state") == "accepted":
            found = self._gate_item(accuracy.get("gate_id"), stable_id)
            if found is None:
                raise ReductionError("accepted authored metadata is missing its gate")
            gate, item = found
            if gate["gate_kind"] != "metadata_batch":
                raise ReductionError("authored metadata must reference a metadata_batch gate")
            rung_check = item.get("rung_check")
            if (
                item["vet_identity"] != accuracy.get("vet_identity")
                or item["verdict"] != "accurate"
                or item["integrity"]["verdict"] != "accurate"
                or not isinstance(rung_check, Mapping)
                or rung_check.get("verdict") != "accurate"
                or accuracy.get("verdict") != "accurate"
                or not accuracy.get("current")
            ):
                raise ReductionError(
                    "authored metadata gate is missing, stale, inaccurate, or has a blocked rung check"
                )
        fidelity = model.get("fidelity", {})
        rung = model.get("source_resolution", {}).get("rung")
        required = bool(fidelity.get("required")) or rung in {"R3_PORT", "R4_REIMPLEMENT"}
        if required:
            found = self._gate_item(fidelity.get("gate_id"), stable_id)
            if found is None:
                raise ReductionError("required fidelity is missing its gate")
            gate, item = found
            if gate["gate_kind"] != "fidelity":
                raise ReductionError("fidelity must reference a per-model fidelity gate")
            rung_check = item.get("rung_check")
            if (
                item.get("fidelity_identity") != fidelity.get("fidelity_identity")
                or item["fidelity"]["verdict"] != fidelity.get("verdict")
                or fidelity.get("verdict") not in {"match", "minor-drift"}
                or not isinstance(rung_check, Mapping)
                or rung_check.get("verdict") != "accurate"
                or not fidelity.get("current")
            ):
                raise ReductionError(
                    "required fidelity gate is stale, unacceptable, or has a blocked rung check"
                )

    def _validate_execution(self, model: Mapping[str, Any]) -> None:
        """Enforce attempt/receipt and meaningful-mode rules for run awards.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        if model.get("status", {}).get("kind") != "runs":
            return
        execution = model.get("execution", {})
        if not execution.get("current"):
            raise ReductionError("runs requires a current execution identity")
        if execution.get("network_attempted") or execution.get("checkpoint_accessed"):
            raise ReductionError("policy-contaminated execution cannot earn runs")
        modes = model.get("modes", {})
        meaningful = set(modes.get("meaningful_modes", []))
        per_mode = modes.get("per_mode_run", {})
        if meaningful != set(per_mode):
            raise ReductionError("per_mode_run must cover exactly every meaningful mode")
        attempts_by_id = {record["attempt_id"]: record for record in self._attempts.records}
        accepted = set(execution.get("accepted_attempt_ids", []))
        stable_id = model["stable_id"]
        for mode in meaningful:
            reference = per_mode[mode]
            attempt_id = reference.get("attempt_id")
            attempt = attempts_by_id.get(attempt_id)
            if attempt_id not in accepted or attempt is None:
                raise ReductionError(f"mode {mode} references an unaccepted/missing attempt")
            receipt = attempt["worker_receipt"]
            policy = attempt["policy_observation"]
            if (
                attempt["stable_id"] != stable_id
                or attempt["stage"] != "forward"
                or attempt["mode"] != mode
                or attempt["result"] != "succeeded"
                or reference.get("status") != "succeeded"
                or attempt["identities"]["execution"] != execution.get("execution_identity")
                or not receipt["present"]
                or not receipt["forward_completed"]
                or receipt["mode"] != mode
                or any(
                    policy[key]
                    for key in (
                        "network_attempted",
                        "checkpoint_or_weight_read_attempted",
                        "write_outside_scratch_attempted",
                        "credentials_present",
                        "torchlens_import_attempted",
                    )
                )
            ):
                raise ReductionError(f"mode {mode} lacks a clean successful worker receipt")
        rung = model.get("source_resolution", {}).get("rung")
        confirmation_policy = execution.get("confirmation_policy")
        if rung in {"R3_PORT", "R4_REIMPLEMENT"}:
            if confirmation_policy != "two-cold-r3-r4" or len(accepted) < 2:
                raise ReductionError("R3/R4 runs require two cold accepted attempts")
        elif confirmation_policy == "two-cold-r3-r4":
            raise ReductionError("R1/R2 runs cannot claim the R3/R4 confirmation policy")


def materialize_current(ledgers: LedgerPaths) -> Mapping[str, JsonObject]:
    """Read and deterministically materialize current model revisions.

    Parameters
    ----------
    ledgers:
        Canonical ledger paths.

    Returns
    -------
    Mapping[str, dict[str, Any]]
        Highest valid revision per stable ID.
    """

    from menagerie.crawler.recordio import scan_jsonl

    return _select_current(scan_jsonl(ledgers.models))


def default_ledger_paths(root: Union[str, Path]) -> LedgerPaths:
    """Return conventional Slice-A ledger paths below a root directory.

    Parameters
    ----------
    root:
        Canonical records root.

    Returns
    -------
    LedgerPaths
        Models, attempts, and gates JSONL paths.
    """

    base = Path(root)
    return LedgerPaths(
        models=base / "models" / "current-shard.jsonl",
        attempts=base / "attempts" / "local.jsonl",
        gates=base / "gates" / "current-shard.jsonl",
    )
