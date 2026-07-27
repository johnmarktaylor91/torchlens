"""Tests for meaningful modes and divergence classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from menagerie.crawler.constants import RunMode
from menagerie.crawler.modes import (
    classify_observed_mode_receipts,
    classify_train_eval_divergence,
    detect_meaningful_modes,
    output_signature,
    output_value_sha256,
)

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any
import pytest
from menagerie.crawler.identity import compute_recipe_revision, hash_bytes
from menagerie.crawler.policy import detect_os_sandbox
from menagerie.crawler.tests.conftest import make_worker_result_v3_mapping
from menagerie.crawler.worker_supervisor import (
    SupervisedResult,
    _parse_linux_denial_audit,
    poison_receipt_for_sandbox_denial,
    supervise_worker,
)


class _BatchNormModel(torch.nn.Module):
    """Fixture with statistical mode behavior."""

    def __init__(self) -> None:
        """Construct one BatchNorm layer."""

        super().__init__()
        self.norm = torch.nn.BatchNorm1d(3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply mode-sensitive normalization.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """

        return self.norm(value)


class _ShapeBranchModel(torch.nn.Module):
    """Fixture with structural train/eval behavior."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return mode-dependent output shapes.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Full tensor in train mode and one column in eval mode.
        """

        return value if self.training else value[:, :1]


@dataclass(frozen=True)
class _StablePayload:
    """Supported structured output used by the stable-encoding fixture."""

    label: str
    values: tuple[int, ...]


class _AddressBearingOutput:
    """Unsupported output whose default repr contains a process address."""


def test_modes_classify_none_statistical_and_structural() -> None:
    """Captured outputs distinguish equality, value drift, and shape drift."""

    value = torch.tensor([[1.0, 2.0, 4.0], [3.0, 6.0, 8.0]])
    assert classify_train_eval_divergence(value, value.clone()).classification == "none"

    statistical = _BatchNormModel()
    statistical.train()
    train_output = statistical(value)
    statistical.eval()
    eval_output = statistical(value)
    assert classify_train_eval_divergence(train_output, eval_output).classification == "statistical"
    assert detect_meaningful_modes(statistical) == (RunMode.TRAIN, RunMode.EVAL)

    structural = _ShapeBranchModel()
    structural.train()
    train_output = structural(value)
    structural.eval()
    eval_output = structural(value)
    assert classify_train_eval_divergence(train_output, eval_output).classification == "structural"
    assert detect_meaningful_modes(structural) == (RunMode.TRAIN, RunMode.EVAL)


def test_independent_mode_receipts_recover_all_divergence_classes() -> None:
    """Per-process structure and value digests mechanically recover mode divergence."""

    train = torch.tensor([[1.0, 2.0]])
    equal = train.clone()
    drifted = torch.tensor([[2.0, 3.0]])
    reshaped = torch.tensor([[1.0], [2.0]])

    def receipt(value: torch.Tensor) -> dict[str, object]:
        """Build the observation subset retained by an isolated mode receipt.

        Parameters
        ----------
        value:
            Captured mode output.

        Returns
        -------
        dict[str, object]
            Structure and value-digest observation.
        """

        return {
            "output_signature": output_signature(value),
            "output_value_sha256": output_value_sha256(value),
        }

    equal_result = classify_observed_mode_receipts(receipt(train), receipt(equal))
    drifted_result = classify_observed_mode_receipts(receipt(train), receipt(drifted))
    reshaped_result = classify_observed_mode_receipts(receipt(train), receipt(reshaped))
    assert equal_result is not None and equal_result.classification == "none"
    assert drifted_result is not None and drifted_result.classification == "statistical"
    assert reshaped_result is not None and reshaped_result.classification == "structural"


def test_output_value_digest_is_stable_for_supported_structures_and_endianness() -> None:
    """Mapping order, dataclasses, and machine byte order do not perturb the digest."""

    little = np.array([1, 2, 3], dtype="<i4")
    big = np.array([1, 2, 3], dtype=">i4")
    left = {"array": little, "payload": _StablePayload("x", (1, 2))}
    right = {"payload": _StablePayload("x", (1, 2)), "array": big}
    assert output_value_sha256(left) == output_value_sha256(right)


def test_output_value_digest_rejects_object_arrays_and_unsupported_leaves() -> None:
    """No address-bearing repr or Python object-array fallback enters a stable digest."""

    assert output_value_sha256(_AddressBearingOutput()) is None
    assert output_value_sha256(np.array([_AddressBearingOutput()], dtype=object)) is None


def _adapter_source(*, stateful: bool) -> str:
    """Return a dependency-free adapter that prints its worker process identifier.

    Parameters
    ----------
    stateful:
        Whether eval requires readiness set by an earlier train forward on the same instance.

    Returns
    -------
    str
        Complete typed-adapter source.
    """

    eval_guard = (
        "        elif not self.ready:\n"
        "            raise RuntimeError('eval requires leaked train state')\n"
        if stateful
        else ""
    )
    train_update = "            self.ready = True\n" if stateful else "            pass\n"
    return f"""from __future__ import annotations
import os
import time

PROCESS_TOKEN = time.time_ns()

class Tiny:
    def __init__(self) -> None:
        self.training = True
        self.ready = False

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def eval(self) -> None:
        self.training = False

    def forward(self, value: int) -> int:
        print(f"worker-token={{PROCESS_TOKEN}}-pid={{os.getpid()}}", flush=True)
        if self.training:
{train_update}{eval_guard}        return value

def build_model() -> object:
    return Tiny()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del device
    return ((seed,), {{}})
"""


def _run_mode(
    root: Path,
    adapter: Path,
    *,
    mode: str,
    cold_index: int,
    input_seed: int,
) -> tuple[SupervisedResult, Path]:
    """Run one explicit mode in one supervised subprocess.

    Parameters
    ----------
    root:
        Per-process request, scratch, and result root.
    adapter:
        Exact typed-adapter source.
    mode:
        Requested train or eval mode.
    cold_index:
        Driver-owned cold confirmation index used as the execution RNG seed.
    input_seed:
        Fixed accepted seed used only to construct the dummy call.

    Returns
    -------
    tuple[SupervisedResult, pathlib.Path]
        Parent result and the child's durable receipt path.
    """

    root.mkdir(parents=True)
    source_identity = "source-round6-exec"
    request_path = root / "request.json"
    receipt_path = root / "result" / "receipt.json"
    request = {
        "stable_id": "m_round6_exec",
        "recipe": {
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": hash_bytes(adapter.read_bytes()),
        },
        "modality": "unknown",
        "input_spec": {"shape": [1], "dtype": "int64"},
        "scratch_root": str(root / "supervisor"),
        "seed": cold_index,
        "input_seed": input_seed,
        "mode": mode,
        "framework": "python",
        "meaningful_modes": ["train", "eval"],
        "source_identity": source_identity,
        "recipe_revision": compute_recipe_revision(
            {"recipe_type": "typed-adapter", "path": adapter.name},
            source_identity,
            adapter_bytes=adapter.read_bytes(),
        ),
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    result = supervise_worker(
        request_path,
        receipt_path,
        root / "supervisor",
        timeout_seconds=20,
        rss_limit_bytes=1024**3,
    )
    return result, receipt_path


def _require_linux_sandbox(result: SupervisedResult) -> None:
    """Skip subprocess assertions only when Linux isolation is unavailable.

    Parameters
    ----------
    result:
        First supervised result used to confirm fail-closed behavior before skipping.
    """

    available = detect_os_sandbox("Linux") is not None and shutil.which("strace") is not None
    if sys.platform == "linux" and available:
        return
    assert result.worker_receipt is None
    pytest.skip("fresh-process regression requires the Linux worker sandbox")


def _worker_token(result: SupervisedResult) -> int:
    """Extract the adapter-reported process-start token from a supervised result.

    Parameters
    ----------
    result:
        Successful supervised worker result.

    Returns
    -------
    int
        Per-process nanosecond start token.
    """

    match = re.search(r"worker-token=(\d+)-pid=\d+", result.observation.stdout_tail)
    assert match is not None
    return int(match.group(1))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervised-process regression")
def test_each_explicit_mode_runs_in_a_fresh_process_without_train_state_leak(
    tmp_path: Path,
) -> None:
    """A cold eval cannot inherit readiness set by a separate train process."""

    adapter = tmp_path / "stateful_adapter.py"
    adapter.write_text(_adapter_source(stateful=True), encoding="utf-8")
    train, train_receipt_path = _run_mode(
        tmp_path / "train", adapter, mode="train", cold_index=0, input_seed=17
    )
    _require_linux_sandbox(train)
    eval_result, eval_receipt_path = _run_mode(
        tmp_path / "eval", adapter, mode="eval", cold_index=0, input_seed=17
    )

    assert train.worker_receipt is None
    assert train.receipt_error == "invalid-receipt:worker-result-envelope"
    train_receipt = json.loads(train_receipt_path.read_text(encoding="utf-8"))
    assert set(train_receipt["per_mode"]) == {"train"}
    assert train_receipt["per_mode"]["train"]["forward_completed"] is True
    assert eval_result.worker_receipt is None
    assert eval_result.receipt_error == "invalid-receipt:worker-result-envelope"
    eval_receipt = json.loads(eval_receipt_path.read_text(encoding="utf-8"))
    assert set(eval_receipt["per_mode"]) == {"eval"}
    assert eval_receipt["per_mode"]["eval"]["forward_completed"] is False
    assert (
        "eval requires leaked train state" in eval_receipt["per_mode"]["eval"]["error"]["message"]
    )
    assert _worker_token(train) != _worker_token(eval_result)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervised-process regression")
def test_stateless_modes_and_cold_confirmations_use_fresh_processes_and_fixed_input(
    tmp_path: Path,
) -> None:
    """Clean modes succeed independently while cold confirmations reuse one input seed."""

    adapter = tmp_path / "stateless_adapter.py"
    adapter.write_text(_adapter_source(stateful=False), encoding="utf-8")
    results: list[SupervisedResult] = []
    receipts: list[dict[str, Any]] = []
    for cold_index in range(2):
        for mode in ("train", "eval"):
            result, receipt_path = _run_mode(
                tmp_path / f"cold-{cold_index}-{mode}",
                adapter,
                mode=mode,
                cold_index=cold_index,
                input_seed=23,
            )
            if not results:
                _require_linux_sandbox(result)
            assert result.worker_receipt is None
            assert result.receipt_error == "invalid-receipt:worker-result-envelope"
            results.append(result)
            receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))

    assert len({_worker_token(result) for result in results}) == 4
    assert {receipt["input_seed"] for receipt in receipts} == {23}
    assert {receipt["seed"] for receipt in receipts} == {0, 1}
    input_signatures = [
        next(iter(receipt["per_mode"].values()))["input_signature"] for receipt in receipts
    ]
    assert input_signatures.count(input_signatures[0]) == len(input_signatures)
    assert all(len(receipt["per_mode"]) == 1 for receipt in receipts)
    assert all(
        next(iter(receipt["per_mode"].values()))["forward_completed"] is True
        for receipt in receipts
    )


def _successful_receipt(path: Path) -> dict[str, Any]:
    """Write and return a minimal self-hashed successful worker receipt.

    Parameters
    ----------
    path:
        Receipt path to create.

    Returns
    -------
    dict[str, Any]
        Complete persisted receipt.
    """

    payload = {
        "receipt_version": "menagerie.crawler.worker-receipt.v1",
        "policy_observation": {
            "network_attempted": False,
            "socket_targets": [],
            "checkpoint_or_weight_read_attempted": False,
            "checkpoint_paths": [],
            "write_outside_scratch_attempted": False,
            "write_paths": [],
        },
        "error": None,
        "per_mode": {"eval": {"forward_completed": True, "error": None}},
    }
    receipt = make_worker_result_v3_mapping(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt


def test_linux_undeclared_read_attempts_poison_even_when_the_open_fails(
    tmp_path: Path,
) -> None:
    """Linux aligns with Seatbelt: every undeclared model-data read attempt poisons."""

    failed_path = "/etc/optional-model.yaml"
    successful_path = "/etc/undeclared-model-data.yaml"
    failed_audit = tmp_path / "failed-open.log"
    failed_audit.write_text(
        '101 openat(AT_FDCWD, "/etc/optional-model.yaml", O_RDONLY|O_CLOEXEC) '
        "= -1 ENOENT (No such file or directory)\n"
        "101 +++ exited with 0 +++\n",
        encoding="utf-8",
    )
    successful_audit = tmp_path / "successful-open.log"
    successful_audit.write_text(
        '102 openat(AT_FDCWD, "/etc/undeclared-model-data.yaml", O_RDONLY|O_CLOEXEC) '
        "= 3</etc/undeclared-model-data.yaml>\n"
        "102 +++ exited with 0 +++\n",
        encoding="utf-8",
    )

    failed = _parse_linux_denial_audit(failed_audit, tmp_path, (tmp_path / "scratch",))
    successful = _parse_linux_denial_audit(successful_audit, tmp_path, (tmp_path / "scratch",))

    assert failed.poisoned is True
    assert failed.failed_read_probe_paths == (failed_path,)
    assert failed.checkpoint_paths == (failed_path,)
    failed_receipt_path = tmp_path / "receipts" / "failed-probe.json"
    _successful_receipt(failed_receipt_path)
    assert poison_receipt_for_sandbox_denial(failed_receipt_path, failed) is True
    assert (
        json.loads(failed_receipt_path.read_text(encoding="utf-8"))["diagnostic"][
            "policy_observation"
        ]["checkpoint_or_weight_read_attempted"]
        is True
    )

    assert successful.poisoned is True
    assert successful.checkpoint_paths == (successful_path,)
    successful_receipt_path = tmp_path / "receipts" / "successful-read.json"
    _successful_receipt(successful_receipt_path)
    assert poison_receipt_for_sandbox_denial(successful_receipt_path, successful) is True
    poisoned = json.loads(successful_receipt_path.read_text(encoding="utf-8"))["diagnostic"]
    assert poisoned["policy_observation"]["checkpoint_or_weight_read_attempted"] is True
    assert poisoned["error"]["reason_code"] == "checkpoint-read"
    assert poisoned["per_mode"]["eval"]["error"]["reason_code"] == "checkpoint-read"
