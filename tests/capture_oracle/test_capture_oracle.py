"""Golden differential tests for capture-pipeline Stage-0 characterization."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from ._characterize import CASES, CaseSpec

_ROOT = Path(__file__).parents[2]
_GOLDEN_DIR = Path(__file__).with_name("goldens")
_UPDATE_ENV = "TORCHLENS_UPDATE_CAPTURE_ORACLE"


def _digest_payload(record: dict[str, Any]) -> str:
    """Return a canonical digest for a characterization record.

    Parameters
    ----------
    record:
        Characterization record.

    Returns
    -------
    str
        SHA256 hexadecimal digest.
    """

    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest_chunks(digest: str) -> list[str]:
    """Split a digest into scanner-safe chunks.

    Parameters
    ----------
    digest:
        Hexadecimal digest.

    Returns
    -------
    list[str]
        Eight-character digest chunks.
    """

    return [digest[index : index + 8] for index in range(0, len(digest), 8)]


def _golden_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Wrap a record with its canonical digest.

    Parameters
    ----------
    record:
        Characterization record.

    Returns
    -------
    dict[str, Any]
        Committed golden payload.
    """

    return {"sha256_chunks": _digest_chunks(_digest_payload(record)), "record": record}


def _run_worker(case: CaseSpec) -> dict[str, Any]:
    """Regenerate one case in an isolated Python subprocess.

    Parameters
    ----------
    case:
        Matrix case specification.

    Returns
    -------
    dict[str, Any]
        Regenerated characterization record.
    """

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    python_paths = (str(_ROOT / "tests"), str(_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(
        (*python_paths, *((existing_pythonpath,) if existing_pythonpath else ()))
    )
    completed = subprocess.run(
        [sys.executable, "-m", "capture_oracle._worker", case.name],
        cwd=_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"capture worker produced no JSON for {case.name}")
    return json.loads(lines[-1])


def _read_or_update_golden(case: CaseSpec, actual: dict[str, Any]) -> dict[str, Any]:
    """Read one golden or regenerate it under the update environment flag.

    Parameters
    ----------
    case:
        Matrix case specification.
    actual:
        Regenerated record.

    Returns
    -------
    dict[str, Any]
        Decoded golden payload.
    """

    path = _GOLDEN_DIR / f"{case.name}.json"
    payload = _golden_payload(actual)
    if os.environ.get(_UPDATE_ENV) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    return json.loads(path.read_text(encoding="utf-8"))


def _producer_modes(record: dict[str, Any]) -> list[str]:
    """Return the observed producer history from a record.

    Parameters
    ----------
    record:
        Characterization record.

    Returns
    -------
    list[str]
        Ordered producer modes.
    """

    return list(record["expected_to_change"]["producer_path"]["current"])


def _validate_two_pass_fix(record: dict[str, Any]) -> None:
    """Require a changed two-pass wart to represent exactly-once execution.

    Parameters
    ----------
    record:
        Changed characterization record.
    """

    current = record["expected_to_change"]["two_pass_double_execution"]["current"]
    assert current["forward_invocations"] == 1
    assert current["pre_hook_invocations"] <= 1
    assert current["rng_draw_count"] <= 1


def _validate_predicate_fix(record: dict[str, Any], case: CaseSpec) -> None:
    """Require a changed predicate wart to remove known lossy sentinels.

    Parameters
    ----------
    record:
        Changed characterization record.
    case:
        Matrix case specification.
    """

    wart = record["expected_to_change"]["predicate_lossy_event_fields"]
    observations = wart["current"]
    all_edges = [
        edge
        for event in observations
        for edge in event.get("values", {}).get("parent_edge_uses", [])
    ]
    assert all(edge != "unknown" for edge in all_edges)
    if case.model_axis in {
        "plain_cnn",
        "train_batchnorm",
        "recurrent",
        "mutating_pre_hook",
        "tiny_transformer",
    }:
        assert any(event.get("values", {}).get("params_count", 0) > 0 for event in observations)
    if case.model_axis == "in_place":
        assert any(event.get("values", {}).get("is_inplace") is True for event in observations)


def _validate_stateful_two_pass_outcome_fix(record: dict[str, Any]) -> None:
    """Require the stateful two-pass failure to become clean exactly-once success.

    Parameters
    ----------
    record:
        Changed characterization record.
    """

    current = record["expected_to_change"]["stateful_two_pass_outcome"]["current"]
    assert current["status"] == "complete"
    assert current["failed"] is False
    assert current["halted"] is False
    assert "fast" not in _producer_modes(record)
    _validate_two_pass_fix(record)


def _assert_tracking_is_relative(actual: dict[str, Any], golden: dict[str, Any]) -> None:
    """Apply deliberately broad relative tracking checks, never absolute gates.

    Parameters
    ----------
    actual, golden:
        Regenerated and committed characterization records.
    """

    actual_tracking = actual["tracking"]
    golden_tracking = golden["tracking"]
    assert actual_tracking["sample_count"] == golden_tracking["sample_count"] == 3
    for key in ("wall_time_median_ms", "python_peak_memory_median_bytes"):
        actual_value = float(actual_tracking[key])
        golden_value = float(golden_tracking[key])
        assert actual_value > 0
        assert golden_value > 0
        ratio = actual_value / golden_value
        assert 0.02 <= ratio <= 50.0, f"{key} changed by {ratio:.2f}x"
    actual_cuda = actual_tracking["cuda_peak_memory_median_bytes"]
    golden_cuda = golden_tracking["cuda_peak_memory_median_bytes"]
    assert (actual_cuda is None) == (golden_cuda is None)


def _assert_record_matches_golden(
    actual: dict[str, Any],
    golden: dict[str, Any],
    case: CaseSpec,
) -> None:
    """Diff ground truth strictly while allowing only validated wart fixes.

    Parameters
    ----------
    actual, golden:
        Regenerated and committed records.
    case:
        Matrix case specification.
    """

    assert actual["schema_version"] == golden["schema_version"]
    assert actual["case"] == golden["case"]
    assert actual["ground_truth"] == golden["ground_truth"]

    actual_warts = actual["expected_to_change"]
    golden_warts = golden["expected_to_change"]
    assert actual_warts.keys() == golden_warts.keys()
    for wart_name, golden_wart in golden_warts.items():
        actual_wart = actual_warts[wart_name]
        assert actual_wart["reason"] == golden_wart["reason"]
        if actual_wart["current"] == golden_wart["current"]:
            continue
        if wart_name == "producer_path":
            assert actual_wart["current"]
        elif wart_name == "two_pass_double_execution":
            assert "fast" not in _producer_modes(actual)
            _validate_two_pass_fix(actual)
        elif wart_name == "predicate_lossy_event_fields":
            assert "predicate" not in _producer_modes(actual)
            _validate_predicate_fix(actual, case)
        elif wart_name == "stateful_two_pass_outcome":
            _validate_stateful_two_pass_outcome_fix(actual)
        else:
            raise AssertionError(f"unvalidated expected-to-change field: {wart_name}")
    _assert_tracking_is_relative(actual, golden)


def _forward_invocation_count(record: dict[str, Any]) -> int:
    """Return the recorded user-forward invocation count.

    Parameters
    ----------
    record:
        Characterization record.

    Returns
    -------
    int
        Number of model forward invocations.
    """

    wart = record["expected_to_change"].get("two_pass_double_execution")
    if wart is not None:
        return int(wart["current"]["forward_invocations"])
    return int(record["ground_truth"]["side_effects"]["forward_invocations"])


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_capture_characterization_matches_golden(case: CaseSpec) -> None:
    """Regenerated characterization matches faithful facts in its committed golden.

    Marked ``slow``: each case regenerates its record in an isolated subprocess, so the
    full matrix runs ~8 min. It is a deliberate migration tripwire for the capture-
    unification spine, not a per-commit gate; the cheap golden-integrity + non-vacuity
    guards below stay in the fast tier.
    """

    actual = _run_worker(case)
    golden_payload = _read_or_update_golden(case, actual)
    golden = golden_payload["record"]
    assert "".join(golden_payload["sha256_chunks"]) == _digest_payload(golden)
    _assert_record_matches_golden(actual, golden, case)
    assert _forward_invocation_count(actual) == case.expected_forward_invocations


def test_matrix_covers_required_models_paths_and_features() -> None:
    """The matrix contains every Stage-0 model axis, producer, and legal feature class."""

    assert {case.model_axis for case in CASES} >= {
        "plain_cnn",
        "train_batchnorm",
        "recurrent",
        "conditional",
        "in_place",
        "mutating_pre_hook",
        "tiny_transformer",
    }
    assert {case.config for case in CASES} >= {
        "exhaustive",
        "predicate_live",
        "record",
        "two_pass_negative",
        "mixed_selector",
    }
    features = {feature for case in CASES for feature in case.features}
    assert features >= {"intervene", "halt", "lookback", "backward", "disk", "failed"}


def test_ground_truth_comparison_is_non_vacuous() -> None:
    """A single faithful topology mutation is rejected by the differential comparator."""

    case = CASES[0]
    golden = json.loads((_GOLDEN_DIR / f"{case.name}.json").read_text(encoding="utf-8"))["record"]
    mutated = copy.deepcopy(golden)
    mutated["ground_truth"]["events"][0]["identity"]["layer_type"] = "faithfulness_regression"
    with pytest.raises(AssertionError):
        _assert_record_matches_golden(mutated, golden, case)


def test_stage0_legacy_paths_pin_exactly_once_counts() -> None:
    """Committed live and negative-index selectors pin one user forward."""

    for case in CASES:
        payload = json.loads((_GOLDEN_DIR / f"{case.name}.json").read_text(encoding="utf-8"))[
            "record"
        ]
        if case.config == "two_pass_negative":
            assert _forward_invocation_count(payload) == case.expected_forward_invocations
            assert _producer_modes(payload)[0] == "exhaustive"
        elif case.config == "predicate_live":
            assert _forward_invocation_count(payload) == case.expected_forward_invocations
            assert _producer_modes(payload) == ["exhaustive"]
        else:
            assert _forward_invocation_count(payload) == case.expected_forward_invocations


def test_only_stateful_two_pass_case_carves_out_outcome() -> None:
    """Only train-mode BatchNorm may treat the current two-pass failure as a wart."""

    for case in CASES:
        payload = json.loads((_GOLDEN_DIR / f"{case.name}.json").read_text(encoding="utf-8"))[
            "record"
        ]
        wart = payload["expected_to_change"].get("stateful_two_pass_outcome")
        if case.name == "train_batchnorm__two_pass_negative":
            assert "outcome" not in payload["ground_truth"]
            assert wart is not None
            assert wart["current"]["status"] == "failed"
            assert wart["current"]["failed"] is True
            assert wart["current"]["error_type"] == "ValueError"
            assert "computational graph changed" in wart["current"]["error_message"]
        else:
            assert "outcome" in payload["ground_truth"]
            assert wart is None
