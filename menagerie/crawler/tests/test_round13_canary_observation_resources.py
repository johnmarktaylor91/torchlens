"""Round-13 cold-forward, observation, and resource regression tripwires."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import menagerie.crawler.worker_supervisor as supervisor_module
from menagerie.crawler.driver import DriverIntegrationError, _forward_timeout_seconds
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.reducer import (
    CanonicalReducer,
    ReductionError,
    cold_forward_policy,
    default_ledger_paths,
)
from menagerie.crawler.tests.conftest import (
    HASH,
    make_attempt,
    make_author_proposal,
    make_gate,
    make_model,
)
from menagerie.crawler.worker_supervisor import _child_rss, _rusage_peak_rss_floor_bytes


_CANARY_ID = "m_canary_26"


def _canary_attempt(stable_id: str, cold_index: int) -> dict[str, Any]:
    """Build one reducer-verifiable independent canary attempt.

    Parameters
    ----------
    stable_id:
        Canary model ID.
    cold_index:
        Zero-based cold-forward index.

    Returns
    -------
    dict[str, Any]
        Complete attempt with canonical attempt and parent-process identities.
    """

    attempt_id = stable_hash(
        {
            "work_id": f"work-{stable_id}",
            "execution_identity": HASH,
            "cold_index": cold_index,
            "mode": "eval",
        }
    )
    attempt = make_attempt(stable_id, attempt_id=attempt_id)
    attempt.pop("ledger_seq", None)
    attempt.pop("payload_sha256", None)
    attempt["attempt_no"] = cold_index + 1
    attempt["retries"]["stage_attempt"] = cold_index + 1
    attempt["invocation"]["argv"] = [
        "python",
        f"/scratch/cold-{cold_index + 1}/eval/request.json",
    ]
    completion_line = (
        "MENAGERIE_WORKER_COMPLETION_V1 "
        f'{{"proof":"{HASH}","receipt_sha256":"sha256:{cold_index:064x}"}}'
    )
    completion_bytes = (completion_line + "\n").encode("utf-8")
    observation = attempt["supervisor_observation"]
    observation["stdout_completion_line"] = completion_line
    observation["stdout_sha256"] = hash_bytes(completion_bytes)
    observation["stdout_bytes"] = len(completion_bytes)
    observation["stdout_tail"] = ""
    attempt["worker_receipt"]["receipt_sha256"] = stable_hash(
        {
            "version": "menagerie.crawler.parent-success-attestation.v1",
            "completion_line": completion_line,
            "exit_code": observation["exit_code"],
            "signal": observation["signal"],
            "wall_seconds": observation["wall_seconds"],
            "cpu_seconds": observation["cpu_seconds"],
            "peak_rss_bytes": observation["peak_rss_bytes"],
            "stdout_sha256": observation["stdout_sha256"],
            "stderr_sha256": observation["stderr_sha256"],
        }
    )
    return attempt


def _canary_model(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a canary run award referencing both independent attempts.

    Parameters
    ----------
    attempts:
        Ordered cold-forward attempts.

    Returns
    -------
    dict[str, Any]
        Complete accepted model candidate.
    """

    selected = attempts[-1]
    model = make_model(_CANARY_ID, accepted=True, attempt_id=selected["attempt_id"])
    attempt_ids = [attempt["attempt_id"] for attempt in attempts]
    model["execution"]["accepted_attempt_ids"] = attempt_ids
    model["execution"]["confirmation_policy"] = "mechanical-canary"
    model["observed"]["measurement_attempt_ids"] = attempt_ids
    model["observed"]["output_signature"] = deepcopy(selected["worker_receipt"]["output_signature"])
    return model


def test_cold_forward_canary_membership_is_reproducible_and_near_two_percent() -> None:
    """Hash selection is reproducible and samples the intended population fraction."""

    first = [
        stable_id
        for index in range(10_000)
        if (stable_id := f"m_canary_{index}")
        and cold_forward_policy(stable_id, "R1_LIBRARY").confirmation_policy == "mechanical-canary"
    ]
    second = [
        stable_id
        for index in range(10_000)
        if (stable_id := f"m_canary_{index}")
        and cold_forward_policy(stable_id, "R1_LIBRARY").confirmation_policy == "mechanical-canary"
    ]

    assert first == second
    assert 150 <= len(first) <= 250
    assert cold_forward_policy(_CANARY_ID, "R1_LIBRARY").required_cold_forwards == 2
    assert cold_forward_policy("m_example", "R1_LIBRARY").required_cold_forwards == 1
    assert cold_forward_policy("anything", "R3_PORT").required_cold_forwards == 2


def test_reducer_accepts_matching_canary_and_blocks_signature_mismatch(tmp_path: Path) -> None:
    """A sampled award needs two byte-matching cold output signatures."""

    matching = [_canary_attempt(_CANARY_ID, 0), _canary_attempt(_CANARY_ID, 1)]
    with CanonicalReducer(default_ledger_paths(tmp_path / "matching"), [_CANARY_ID]) as reducer:
        reducer.append_gate(make_gate([_CANARY_ID]))
        for attempt in matching:
            reducer.append_attempt(attempt)
        reducer.append_model(_canary_model(matching))

    mismatched = [_canary_attempt(_CANARY_ID, 0), _canary_attempt(_CANARY_ID, 1)]
    mismatched[1]["worker_receipt"]["output_signature"]["leaves"][0]["shape"] = [1, 3]
    with CanonicalReducer(default_ledger_paths(tmp_path / "mismatch"), [_CANARY_ID]) as reducer:
        reducer.append_gate(make_gate([_CANARY_ID]))
        for attempt in mismatched:
            reducer.append_attempt(attempt)
        with pytest.raises(ReductionError, match="non-deterministic cold-forward"):
            reducer.append_model(_canary_model(mismatched))


def test_reducer_blocks_reused_canary_parent_witness(tmp_path: Path) -> None:
    """Copied rows cannot satisfy two-cold confirmation with one process witness."""

    attempts = [_canary_attempt(_CANARY_ID, 0), _canary_attempt(_CANARY_ID, 1)]
    first_observation = attempts[0]["supervisor_observation"]
    second_observation = attempts[1]["supervisor_observation"]
    second_observation["stdout_completion_line"] = first_observation["stdout_completion_line"]
    second_observation["stdout_sha256"] = first_observation["stdout_sha256"]
    attempts[1]["worker_receipt"]["receipt_sha256"] = attempts[0]["worker_receipt"][
        "receipt_sha256"
    ]

    with CanonicalReducer(default_ledger_paths(tmp_path), [_CANARY_ID]) as reducer:
        reducer.append_gate(make_gate([_CANARY_ID]))
        for attempt in attempts:
            reducer.append_attempt(attempt)
        with pytest.raises(ReductionError, match="reuse a parent process witness"):
            reducer.append_model(_canary_model(attempts))


def test_resource_helpers_are_platform_scoped_and_rusage_is_not_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS dispatches to its live sampler and rusage scales only the first child."""

    def macos_sample(pid: int) -> int:
        """Return a distinctive macOS sample."""

        return pid + 100

    def linux_sample(pid: int) -> int:
        """Return a distinctive Linux sample."""

        return pid + 200

    monkeypatch.setattr(supervisor_module, "_macos_rss", macos_sample)
    monkeypatch.setattr(supervisor_module, "_linux_rss", linux_sample)
    assert _child_rss(7, platform_name="darwin") == 107
    assert _child_rss(7, platform_name="linux") == 207

    first_before = SimpleNamespace(ru_maxrss=0)
    first_after = SimpleNamespace(ru_maxrss=2048)
    prior_before = SimpleNamespace(ru_maxrss=1024)
    assert _rusage_peak_rss_floor_bytes(first_before, first_after, platform_name="darwin") == 2048
    assert (
        _rusage_peak_rss_floor_bytes(first_before, first_after, platform_name="linux")
        == 2048 * 1024
    )
    assert _rusage_peak_rss_floor_bytes(prior_before, first_after, platform_name="darwin") == 0


def test_forward_timeout_override_is_bounded() -> None:
    """The lane accepts a declared slow-model timeout but rejects unbounded values."""

    proposal = make_author_proposal()
    assert _forward_timeout_seconds(proposal, 300.0) == 300.0
    proposal["proposed_facts"]["implementation"]["declared_timeout_seconds"] = 1800
    assert _forward_timeout_seconds(proposal, 300.0) == 1800.0
    proposal["proposed_facts"]["implementation"]["declared_timeout_seconds"] = 1801
    with pytest.raises(DriverIntegrationError, match=r"integer in \[1, 1800\]"):
        _forward_timeout_seconds(proposal, 300.0)
