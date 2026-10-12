"""LayerProfilerCallback leftovers: skip-with-warning, run id, disclosure.

A batch the forward cannot take must not kill training (a typed warning and
no record instead); every JSONL record carries the run id (fresh per
Lightning run unless fixed) and the eval/no_grad re-run disclosure.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("lightning")

from torchlens.callbacks.lightning import PROFILE_DISCLOSURE, LayerProfilerCallback


class _TinyModel(nn.Module):
    """One linear layer taking a [N, 4] batch."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FakeTrainer:
    global_step = 3


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_unguessable_batch_skips_with_a_warning(tmp_path: Path) -> None:
    callback = LayerProfilerCallback(tmp_path / "profile.jsonl")
    model = _TinyModel().train()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        callback._maybe_profile("train", _FakeTrainer(), model, {"y": torch.ones(2, 4)}, 0)
    codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
    assert "lightning_profile_skipped" in codes
    assert callback.records == []
    assert not (tmp_path / "profile.jsonl").exists()
    assert model.training, "the skipped profile must still restore the training flags"


def test_records_carry_run_id_and_disclosure(tmp_path: Path) -> None:
    path = tmp_path / "profile.jsonl"
    callback = LayerProfilerCallback(path)
    model = _TinyModel()
    callback.setup(_FakeTrainer(), model, "fit")
    first_run = callback.run_id
    callback._maybe_profile("train", _FakeTrainer(), model, torch.ones(2, 4), 0)
    callback.setup(_FakeTrainer(), model, "validate")
    callback._maybe_profile("validation", _FakeTrainer(), model, torch.ones(2, 4), 0)
    rows = _records(path)
    assert [row["run_id"] for row in rows] == [first_run, callback.run_id]
    assert first_run != callback.run_id
    assert all(row["disclosure"] == PROFILE_DISCLOSURE for row in rows)
    assert "eval mode" in PROFILE_DISCLOSURE and "no_grad" in PROFILE_DISCLOSURE


def test_fixed_run_id_survives_setup(tmp_path: Path) -> None:
    path = tmp_path / "profile.jsonl"
    callback = LayerProfilerCallback(path, run_id="my-run")
    model = _TinyModel()
    callback.setup(_FakeTrainer(), model, "fit")
    callback._maybe_profile("train", _FakeTrainer(), model, torch.ones(2, 4), 0)
    assert _records(path)[0]["run_id"] == "my-run"


def test_skip_and_record_inside_a_real_fit(tmp_path: Path) -> None:
    """A real Trainer.fit: the unguessable batch skips, the other records, fit completes."""

    import lightning as L

    class _Module(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.net = _TinyModel()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            return self(next(iter(batch.values()))).square().mean()

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.1)

    # The second batch has no "x" entry, so the default guess hands the whole
    # mapping to the forward, which cannot take it.
    batches = [{"x": torch.ones(2, 4)}, {"inputs": torch.ones(2, 4)}]
    path = tmp_path / "fit.jsonl"
    callback = LayerProfilerCallback(path)
    trainer = L.Trainer(
        max_epochs=1,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        accelerator="cpu",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer.fit(_Module(), train_dataloaders=batches)
    codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
    assert codes.count("lightning_profile_skipped") == 1
    assert trainer.global_step == 2
    rows = _records(path)
    assert len(rows) == 1 and rows[0]["run_id"] == callback.run_id
