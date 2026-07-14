"""Tests for honest per-mode worker receipts."""

from __future__ import annotations

from pathlib import Path

from menagerie.crawler.constants import RunMode
from menagerie.crawler.standard_inputs import InputSpec
from menagerie.crawler.worker import WorkerRequest, run_worker


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
        recipe={"kind": "typed-adapter", "path": str(adapter)},
        modality="unknown",
        input_spec=InputSpec((1, 4), "float32"),
        scratch_root=tmp_path / "scratch",
        receipt_path=receipt_path,
        meaningful_modes=(RunMode.TRAIN, RunMode.EVAL),
    )

    receipt = run_worker(request)

    assert receipt_path.exists()
    assert receipt["awards_runs"] is False
    assert set(receipt["per_mode"]) == {"train", "eval"}
    assert all(mode_receipt["forward_completed"] for mode_receipt in receipt["per_mode"].values())
    assert all(
        mode_receipt["input_kind"] == "random-fallback"
        for mode_receipt in receipt["per_mode"].values()
    )
