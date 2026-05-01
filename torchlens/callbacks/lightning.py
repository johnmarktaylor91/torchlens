"""Lightning callback integrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _lightning_callback_base() -> Any:
    """Return Lightning's callback base when the optional extra is installed.

    Returns
    -------
    Any
        ``lightning.pytorch.callbacks.Callback`` or ``object``.
    """

    try:
        from lightning.pytorch.callbacks import Callback
    except ImportError:
        return object
    return Callback


_LightningCallback: Any = _lightning_callback_base()


class LayerProfilerCallback(_LightningCallback):  # type: ignore[misc]
    """Log TorchLens layer summaries from Lightning batch hooks."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        every_n_batches: int = 1,
        input_getter: Any | None = None,
        layers_to_save: str = "all",
    ) -> None:
        """Initialize the callback.

        Parameters
        ----------
        output_path:
            JSONL path receiving one summary per profiled batch.
        every_n_batches:
            Batch interval for profiling.
        input_getter:
            Optional callable mapping a Lightning batch to model input.
        layers_to_save:
            ``log_forward_pass`` layer-save policy.
        """

        if every_n_batches < 1:
            raise ValueError("every_n_batches must be at least 1.")
        super().__init__()
        self.output_path = Path(output_path)
        self.every_n_batches = every_n_batches
        self.input_getter = input_getter
        self.layers_to_save = layers_to_save
        self.records: list[dict[str, Any]] = []

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Profile a training batch when the configured interval fires.

        Parameters
        ----------
        trainer:
            Lightning trainer.
        pl_module:
            Lightning module.
        outputs:
            Batch outputs, ignored.
        batch:
            Lightning batch.
        batch_idx:
            Batch index.
        """

        self._maybe_profile("train", trainer, pl_module, batch, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Profile a validation batch when the configured interval fires.

        Parameters
        ----------
        trainer:
            Lightning trainer.
        pl_module:
            Lightning module.
        outputs:
            Batch outputs, ignored.
        batch:
            Lightning batch.
        batch_idx:
            Batch index.
        dataloader_idx:
            Lightning dataloader index, ignored.
        """

        self._maybe_profile("validation", trainer, pl_module, batch, batch_idx)

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Profile a test batch when the configured interval fires.

        Parameters
        ----------
        trainer:
            Lightning trainer.
        pl_module:
            Lightning module.
        outputs:
            Batch outputs, ignored.
        batch:
            Lightning batch.
        batch_idx:
            Batch index.
        dataloader_idx:
            Lightning dataloader index, ignored.
        """

        self._maybe_profile("test", trainer, pl_module, batch, batch_idx)

    def on_predict_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Profile a prediction batch when the configured interval fires.

        Parameters
        ----------
        trainer:
            Lightning trainer.
        pl_module:
            Lightning module.
        outputs:
            Batch outputs, ignored.
        batch:
            Lightning batch.
        batch_idx:
            Batch index.
        dataloader_idx:
            Lightning dataloader index, ignored.
        """

        self._maybe_profile("predict", trainer, pl_module, batch, batch_idx)

    def _maybe_profile(
        self,
        stage: str,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Run TorchLens profiling for one eligible batch.

        Parameters
        ----------
        stage:
            Lightning stage label.
        trainer:
            Lightning trainer.
        pl_module:
            Lightning module.
        batch:
            Lightning batch.
        batch_idx:
            Batch index.
        """

        if batch_idx % self.every_n_batches != 0:
            return

        from torchlens import log_forward_pass

        model_input = self._model_input(batch)
        was_training = bool(getattr(pl_module, "training", False))
        pl_module.eval()
        try:
            with torch.no_grad():
                log = log_forward_pass(pl_module, model_input, layers_to_save=self.layers_to_save)
        finally:
            if was_training:
                pl_module.train()

        record = {
            "stage": stage,
            "batch_idx": batch_idx,
            "global_step": getattr(trainer, "global_step", None),
            "num_layers": len(getattr(log, "layer_list", [])),
            "layer_labels": [str(layer.layer_label) for layer in getattr(log, "layer_list", [])],
        }
        self.records.append(record)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    def _model_input(self, batch: Any) -> Any:
        """Extract the model input from a Lightning batch.

        Parameters
        ----------
        batch:
            Lightning batch.

        Returns
        -------
        Any
            Model input passed to ``log_forward_pass``.
        """

        if self.input_getter is not None:
            return self.input_getter(batch)
        if isinstance(batch, dict) and "x" in batch:
            return batch["x"]
        if isinstance(batch, (tuple, list)) and batch:
            return batch[0]
        return batch


__all__ = ["LayerProfilerCallback"]
