"""Lightning callback integrations.

``SummaryCallback`` prints ``tl.summary`` at fit start; ``LayerProfilerCallback``
logs per-batch layer records. Both resolve lazily on top of the installed
Lightning package (``lightning`` or the standalone ``pytorch_lightning``).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch


def _lightning_callback_base() -> type[Any]:
    """Return Lightning's callback base when the optional extra is installed.

    The unified ``lightning`` package wins; the standalone
    ``pytorch_lightning`` package is the fallback, so users of either get a
    working callback.

    Returns
    -------
    type[Any]
        ``lightning.pytorch.callbacks.Callback`` or
        ``pytorch_lightning.callbacks.Callback``.

    Raises
    ------
    ImportError
        If neither Lightning package is available.
    """

    try:
        from lightning.pytorch.callbacks import Callback
    except ImportError as exc:
        try:
            from pytorch_lightning.callbacks import Callback
        except ImportError:
            raise ImportError(
                "TorchLens Lightning callbacks require Lightning: install "
                "torchlens[lightning] (or the standalone pytorch_lightning package)."
            ) from exc
    return cast(type[Any], Callback)


class _LayerProfilerCallbackMixin:
    """Log TorchLens layer summaries from Lightning batch hooks."""

    def __init__(
        self,
        container_path: str | Path,
        *,
        every_n_batches: int = 1,
        input_getter: Any | None = None,
        layers_to_save: str = "all",
    ) -> None:
        """Initialize the callback.

        Parameters
        ----------
        container_path:
            JSONL path receiving one summary per profiled batch.
        every_n_batches:
            Batch interval for profiling.
        input_getter:
            Optional callable mapping a Lightning batch to model input.
        layers_to_save:
            ``trace`` layer-save policy.
        """

        if every_n_batches < 1:
            raise ValueError("every_n_batches must be at least 1.")
        super().__init__()
        self.container_path = Path(container_path)
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

        from torchlens import trace
        from torchlens.options import CaptureOptions

        model_input = self._model_input(batch)
        # Snapshot every submodule's flag: ``pl_module.train()`` afterwards
        # would flip a frozen (eval-mode) backbone back into train mode.
        training_flags = [(module, module.training) for _, module in pl_module.named_modules()]
        pl_module.eval()
        try:
            with torch.no_grad():
                log = trace(
                    pl_module,
                    model_input,
                    capture=CaptureOptions(layers_to_save=self.layers_to_save),
                )
        finally:
            for module, was_training in training_flags:
                module.training = was_training

        try:
            record = {
                "stage": stage,
                "batch_idx": batch_idx,
                "global_step": getattr(trainer, "global_step", None),
                "num_layers": len(getattr(log, "layer_list", [])),
                "layer_labels": [
                    str(layer.layer_label) for layer in getattr(log, "layer_list", [])
                ],
            }
            self.records.append(record)
            self.container_path.parent.mkdir(parents=True, exist_ok=True)
            with self.container_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")
        finally:
            log.cleanup()

    def _model_input(self, batch: Any) -> Any:
        """Extract the model input from a Lightning batch.

        Parameters
        ----------
        batch:
            Lightning batch.

        Returns
        -------
        Any
            Model input passed to ``trace``.
        """

        if self.input_getter is not None:
            return self.input_getter(batch)
        if isinstance(batch, dict) and "x" in batch:
            return batch["x"]
        if isinstance(batch, (tuple, list)) and batch:
            return batch[0]
        return batch


class _SummaryCallbackMixin:
    """Print ``tl.summary`` once at the start of each Lightning fit."""

    def __init__(
        self,
        *,
        input_getter: Callable[[Any], Any] | None = None,
        print_fn: Callable[[str], Any] = print,
        **summary_kwargs: Any,
    ) -> None:
        """Initialize the callback.

        Parameters
        ----------
        input_getter:
            Optional callable mapping the first training batch to the model's
            forward input. Without it the batch's ``"x"`` entry, else its first
            element, else the batch itself is used (a mapping as keyword
            arguments).
        print_fn:
            Receives the rendered summary text (default ``print``).
        **summary_kwargs:
            Forwarded to ``tl.summary`` (the summary grammar). Validated here,
            so a bad option fails when the callback is built, not mid-fit.
        """

        from torchlens._user_public_impls import _validate_summary_grammar

        _validate_summary_grammar(summary_kwargs)
        super().__init__()
        self.input_getter = input_getter
        self.print_fn = print_fn
        self.summary_kwargs = summary_kwargs
        self.report: str | None = None
        self._summarized = False

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Summarize from ``example_input_array`` when the module declares one.

        Parameters
        ----------
        trainer:
            Lightning trainer.
        pl_module:
            Lightning module.
        """

        self._summarized = False
        example = getattr(pl_module, "example_input_array", None)
        if example is None:
            return
        transfer = getattr(pl_module, "transfer_batch_to_device", None)
        device = getattr(pl_module, "device", None)
        if callable(transfer) and device is not None:
            example = transfer(example, device, 0)
        self._summarize(trainer, pl_module, example, "example_input_array")

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Summarize from the first training batch when nothing ran yet.

        Parameters
        ----------
        trainer:
            Lightning trainer.
        pl_module:
            Lightning module.
        batch:
            Lightning batch (already on the module's device).
        batch_idx:
            Batch index.
        """

        if self._summarized:
            return
        if self.input_getter is not None:
            model_input = self.input_getter(batch)
        elif isinstance(batch, dict) and "x" in batch:
            model_input = batch["x"]
        elif isinstance(batch, (tuple, list)) and batch:
            model_input = batch[0]
        else:
            model_input = batch
        self._summarize(trainer, pl_module, model_input, "first training batch")

    def _summarize(self, trainer: Any, pl_module: Any, model_input: Any, source: str) -> None:
        """Run the one extra summary forward and print its table.

        Parameters
        ----------
        trainer:
            Lightning trainer.
        pl_module:
            Lightning module.
        model_input:
            Forward input: a mapping becomes keyword arguments, a tuple or
            list positional arguments, anything else the single argument.
        source:
            Which sample was used, named in the printed header.
        """

        import torchlens as tl
        from torchlens.errors import TorchLensError, TorchLensWarning

        self._summarized = True
        if not getattr(trainer, "is_global_zero", True):
            return
        input_args: Any = model_input
        input_kwargs: dict[str, Any] | None = None
        if isinstance(model_input, dict):
            input_args, input_kwargs = None, dict(model_input)
        try:
            report = tl.summary(pl_module, input_args, input_kwargs, **self.summary_kwargs)
        # A fit must never die for a summary (disclosed below). These are what a
        # forward fed a guessed sample raises (wrong arity or keys, shape or dtype
        # mismatch, missing attribute) plus every TorchLens refusal.
        except (
            TorchLensError,
            RuntimeError,
            TypeError,
            ValueError,
            LookupError,
            AttributeError,
        ) as exc:
            warnings.warn(
                TorchLensWarning(
                    f"SummaryCallback skipped the fit-start summary: running "
                    f"{type(pl_module).__name__}.forward on the {source} raised "
                    f"{type(exc).__name__}: {exc}. Training continues. "
                    "Remedy: pass input_getter= mapping a batch to the forward input",
                    code="lightning_summary_skipped",
                    sample_source=source,
                    error_type=type(exc).__name__,
                ),
                stacklevel=2,
            )
            return
        self.report = report
        self.print_fn(
            f"TorchLens summary (SummaryCallback; sample: {source}; one extra "
            "forward in eval mode under no_grad, every module's mode and the RNG "
            f"state restored)\n{report}"
        )


def _build_layer_profiler_callback() -> type[_LayerProfilerCallbackMixin]:
    """Build the Lightning callback class on first public use.

    Returns
    -------
    type[_LayerProfilerCallbackMixin]
        Callback class inheriting Lightning's callback base.
    """

    callback_base = _lightning_callback_base()

    class LayerProfilerCallback(_LayerProfilerCallbackMixin, callback_base):  # type: ignore[misc, valid-type]
        """Log TorchLens layer summaries from Lightning batch hooks."""

    LayerProfilerCallback.__module__ = __name__
    return LayerProfilerCallback


def _build_summary_callback() -> type[_SummaryCallbackMixin]:
    """Build the fit-start summary callback class on first public use.

    Returns
    -------
    type[_SummaryCallbackMixin]
        Callback class inheriting Lightning's callback base.
    """

    callback_base = _lightning_callback_base()

    class SummaryCallback(_SummaryCallbackMixin, callback_base):  # type: ignore[misc, valid-type]
        """Print ``tl.summary`` at the start of each Lightning fit.

        Sample rule: the module's ``example_input_array`` when it declares one
        (summarized in ``on_fit_start``), otherwise the first training batch
        (summarized in ``on_train_batch_start`` before its training step).
        The forward input is ``input_getter(batch)`` when given, else the
        batch's ``"x"`` entry, else its first element, else the whole batch;
        a mapping is passed as keyword arguments (``forward(**batch)``).
        The summary costs ONE extra forward, run in eval mode under
        ``torch.no_grad()``; every module's training flag and the RNG state
        are restored exactly afterwards, so training is unaffected. Runs on
        global rank zero only. A sample the forward cannot take skips the
        summary with a ``lightning_summary_skipped`` warning; the fit goes on.
        The last report is kept on ``self.report``.
        """

    SummaryCallback.__module__ = __name__
    return SummaryCallback


def __getattr__(name: str) -> Any:
    """Resolve lazy Lightning callback exports.

    Parameters
    ----------
    name:
        Requested module attribute.

    Returns
    -------
    Any
        Lazily constructed callback class.

    Raises
    ------
    AttributeError
        If ``name`` is not exported by this module.
    """

    builders = {
        "LayerProfilerCallback": _build_layer_profiler_callback,
        "SummaryCallback": _build_summary_callback,
    }
    if name not in builders:
        raise AttributeError(f"module 'torchlens.callbacks.lightning' has no attribute {name!r}")
    callback_cls = builders[name]()
    globals()[name] = callback_cls
    return callback_cls


def __dir__() -> list[str]:
    """Return visible Lightning callback module members.

    Returns
    -------
    list[str]
        Sorted module globals plus lazy exports.
    """

    return sorted([*globals(), "LayerProfilerCallback", "SummaryCallback"])


__all__ = ["LayerProfilerCallback", "SummaryCallback"]  # noqa: F822
