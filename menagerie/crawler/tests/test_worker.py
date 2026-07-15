"""Tests for honest per-mode worker receipts."""

from __future__ import annotations

from pathlib import Path

from menagerie.crawler.constants import RunMode
from menagerie.crawler.identity import compute_recipe_revision, hash_bytes
from menagerie.crawler.standard_inputs import InputSpec
from menagerie.crawler.worker import WorkerRequest, run_worker


def _adapter_revision(path: Path, source_identity: str = "unbound") -> str:
    """Return the exact typed-adapter revision expected by a worker request.

    Parameters
    ----------
    path:
        Adapter source path.
    source_identity:
        Source identity bound into the revision.

    Returns
    -------
    str
        Exact recipe revision.
    """

    return compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": path.name},
        source_identity,
        adapter_bytes=path.read_bytes(),
    )


def test_worker_builds_input_and_runs_both_modes(tmp_path: Path) -> None:
    """One tiny model produces successful train and eval observation receipts."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import torch

class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)

def build_model() -> object:
    return Tiny()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    return ((torch.zeros(1, 4, device=device),), {})
""",
        encoding="utf-8",
    )
    receipt_path = tmp_path / "result" / "receipt.json"
    request = WorkerRequest(
        stable_id="m_tiny",
        recipe={
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": hash_bytes(adapter.read_bytes()),
        },
        modality="unknown",
        input_spec=InputSpec((1, 4), "float32"),
        scratch_root=tmp_path / "scratch",
        receipt_path=receipt_path,
        meaningful_modes=(RunMode.TRAIN, RunMode.EVAL),
        recipe_revision=_adapter_revision(adapter),
    )

    receipt = run_worker(request)

    assert receipt_path.exists()
    assert receipt["awards_runs"] is False
    assert set(receipt["per_mode"]) == {"train", "eval"}
    assert all(mode_receipt["forward_completed"] for mode_receipt in receipt["per_mode"].values())
    assert all(
        mode_receipt["input_kind"] == "standard-typed-dummy-call"
        for mode_receipt in receipt["per_mode"].values()
    )


def test_worker_detects_batchnorm_and_executes_complete_dummy_call(tmp_path: Path) -> None:
    """Mechanical mode detection overrides eval-only and preserves kwargs."""

    adapter = tmp_path / "batchnorm_adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import torch

class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(4)

    def forward(self, value: torch.Tensor, *, scale: float) -> torch.Tensor:
        return self.norm(value) * scale

def build_model() -> object:
    return Tiny()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    return ((torch.ones(2, 4, device=device),), {"scale": 2.0})
""",
        encoding="utf-8",
    )
    request = WorkerRequest(
        stable_id="m_batchnorm",
        recipe={
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": hash_bytes(adapter.read_bytes()),
        },
        modality=None,
        input_spec=InputSpec((1, 1), "float32"),
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        meaningful_modes=(RunMode.EVAL,),
        recipe_revision=_adapter_revision(adapter),
    )

    receipt = run_worker(request)

    assert receipt["declared_meaningful_modes"] == ["eval"]
    assert receipt["detected_meaningful_modes"] == ["train", "eval"]
    assert receipt["meaningful_modes"] == ["train", "eval"]
    assert set(receipt["per_mode"]) == {"train", "eval"}
    assert all(value["forward_completed"] for value in receipt["per_mode"].values())
    assert all(
        any(leaf["path"].endswith(".scale") for leaf in value["input_signature"]["leaves"])
        for value in receipt["per_mode"].values()
    )


def test_worker_rejects_invalid_dummy_call_contract(tmp_path: Path) -> None:
    """A typed adapter cannot earn a forward receipt with malformed args."""

    adapter = tmp_path / "invalid_adapter.py"
    adapter.write_text(
        """from __future__ import annotations

class Tiny:
    def forward(self) -> int:
        return 1
    def train(self) -> None:
        pass
    def eval(self) -> None:
        pass

def build_model() -> object:
    return Tiny()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    return ([], {})  # type: ignore[return-value]
""",
        encoding="utf-8",
    )
    request = WorkerRequest(
        stable_id="m_invalid_dummy",
        recipe={
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": hash_bytes(adapter.read_bytes()),
        },
        modality=None,
        input_spec=InputSpec((1,), "float32"),
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        meaningful_modes=(RunMode.EVAL,),
        recipe_revision=_adapter_revision(adapter),
    )

    receipt = run_worker(request)

    assert receipt["input_completed"] is False
    assert receipt["per_mode"] == {}
    assert receipt["error"]["exception_type"] == "builtins.TypeError"


def test_r1_declarative_worker_executes_complete_args_and_kwargs_contract(
    tmp_path: Path,
) -> None:
    """R1 execution preserves every positional tensor and exact non-tensor kwarg."""

    tensor_leaf = {
        "kind": "tensor",
        "semantic_role": "attention sequence",
        "shape": [1, 2, 4],
        "dtype": "float32",
        "device_policy": "cpu",
        "distribution": "normal",
        "constraints": [],
        "source_evidence_ids": ["evidence-1"],
    }
    contract = {
        "code_path": None,
        "builder_symbol": "make_dummy_call",
        "seed": 0,
        "semantic_description": "Query, key, and value tensors with no returned weights.",
        "source_basis": ["evidence-1"],
        "smallest_valid_probe_rationale": "One batch with two tokens.",
        "args": [
            {**tensor_leaf, "path": "args[0]"},
            {**tensor_leaf, "path": "args[1]"},
            {**tensor_leaf, "path": "args[2]"},
        ],
        "kwargs": [],
        "non_tensor_values": [
            {
                "path": "kwargs.need_weights",
                "type": "bool",
                "value": False,
                "semantic_role": "output control",
                "constraints": [],
                "source_evidence_ids": ["evidence-1"],
            }
        ],
        "masks_state_and_control": ["need_weights=False"],
        "expected_output_semantics": "attention output without weights",
    }
    request = WorkerRequest(
        stable_id="m_r1_multi_call",
        recipe={
            "kind": "declarative-library",
            "recipe": {
                "distribution": "torch",
                "version": "test",
                "artifact_sha256": "sha256:" + "a" * 64,
                "module": "torch.nn",
                "symbol": "MultiheadAttention",
                "kwargs": {"embed_dim": 4, "num_heads": 1, "batch_first": True},
                "pretrained_disable_fields": [],
            },
        },
        modality=None,
        input_spec=None,
        input_contract=contract,
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        meaningful_modes=(RunMode.EVAL,),
    )

    receipt = run_worker(request)

    mode_receipt = receipt["per_mode"]["eval"]
    assert mode_receipt["forward_completed"] is True
    assert {leaf["path"] for leaf in mode_receipt["input_signature"]["leaves"]} == {
        "input.args[0]",
        "input.args[1]",
        "input.args[2]",
        "input.kwargs.need_weights",
    }
