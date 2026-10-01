"""Regression tests for r18a compat/_report.py detection-heuristic tightening.

Each detection row must anchor its structural marker on real module identity /
AST references / typed attributes, mirroring ``_is_quantized_module``. These tests
kill the false-positive and false-negative detection classes flagged in round 18
(A3-02, A3-03, A3-08, A3-09, A3-10, A3-11 + DDP/DeepSpeed siblings, A3-12, LOW-8).
"""

from __future__ import annotations

import textwrap

import pytest
import torch
from torch import nn

from torchlens.compat import report
from torchlens.compat._report import _SHOW_CELL_WIDTH_CAP

# ---------------------------------------------------------------------------
# A3-08 — HF Transformers detection must key on real transformers namespace,
# not on the mere presence of a ``.config`` attribute.
# ---------------------------------------------------------------------------


class OrdinaryConfiguredModel(nn.Module):
    """Plain module that happens to carry an application ``config``."""

    def __init__(self) -> None:
        """Attach a non-HF config object."""

        super().__init__()
        self.config = {"application": "not-huggingface"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """

        return x


class TransformersNamespaceModel(nn.Module):
    """Model advertising a real ``transformers`` module namespace."""

    __module__ = "transformers.modeling_utils"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """

        return x


def test_hf_row_ignores_plain_config_attribute() -> None:
    """A generic ``.config`` attribute must not label a module as HF Transformers."""

    row = report(OrdinaryConfiguredModel(), torch.randn(1)).row("hf_transformers")

    assert row.detected is False
    assert row.status == "pass"
    assert row.severity == "ok"


def test_hf_row_detects_real_transformers_namespace() -> None:
    """A class defined under the ``transformers`` namespace stays detected."""

    row = report(TransformersNamespaceModel(), torch.randn(1)).row("hf_transformers")

    assert row.detected is True
    assert row.status == "pass"
    assert row.severity == "info"


def test_hf_row_detects_transformers_subclass_by_mro() -> None:
    """A user subclass of a transformers-namespace base is detected via its MRO."""

    class _FakePreTrainedBase(nn.Module):
        __module__ = "transformers.modeling_utils"

    class UserModel(_FakePreTrainedBase):
        __module__ = "my_project.models"

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return input unchanged."""

            return x

    row = report(UserModel(), torch.randn(1)).row("hf_transformers")

    assert row.detected is True


# ---------------------------------------------------------------------------
# A3-11 (+ DDP/DeepSpeed siblings) — distributed-wrapper detection must key on
# real module namespaces, not on a substring of a user class name.
# ---------------------------------------------------------------------------


class _Passthrough(nn.Module):
    """Minimal passthrough module used for namespace-detection fixtures."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """

        return x


def _named(module_path: str, class_name: str) -> nn.Module:
    """Build a passthrough instance advertising a specific module/class identity.

    Parameters
    ----------
    module_path:
        Value for the synthetic type's ``__module__``.
    class_name:
        Name for the synthetic type.

    Returns
    -------
    nn.Module
        Instance of the synthesized module class.
    """

    klass = type(class_name, (_Passthrough,), {"__module__": module_path})
    return klass()


def test_fsdp_row_ignores_user_class_merely_named_fsdp() -> None:
    """A user class named ``FsdpExportHelper`` must not trip the FSDP row."""

    row = report(_named("my_project.helpers", "FsdpExportHelper"), torch.randn(1)).row("fsdp")

    assert row.detected is False
    assert row.status == "pass"


def test_fsdp_row_detects_real_fsdp_namespace() -> None:
    """A class defined under ``torch.distributed.fsdp`` stays detected."""

    model = _named("torch.distributed.fsdp.fully_sharded_data_parallel", "FullyShardedDataParallel")
    row = report(model, torch.randn(1)).row("fsdp")

    assert row.detected is True
    assert row.status == "scope"


def test_ddp_row_ignores_user_class_merely_named_distributed() -> None:
    """A user class named like DDP but in a user module must not be detected."""

    model = _named("my_project.net", "MyDistributedDataParallelHelper")
    row = report(model, torch.randn(1)).row("distributed_data_parallel")

    assert row.detected is False


def test_ddp_row_detects_real_torch_ddp_namespace() -> None:
    """A class under ``torch.nn.parallel.distributed`` stays detected."""

    model = _named("torch.nn.parallel.distributed", "DistributedDataParallel")
    row = report(model, torch.randn(1)).row("distributed_data_parallel")

    assert row.detected is True


def test_ddp_row_does_not_fire_on_data_parallel() -> None:
    """``nn.DataParallel`` (a different namespace) must not trip the DDP row."""

    row = report(nn.DataParallel(_Passthrough()), torch.randn(1)).row("distributed_data_parallel")

    assert row.detected is False


def test_deepspeed_row_ignores_user_class_merely_named_deepspeed() -> None:
    """A user class named ``DeepspeedConfigHelper`` must not trip the DeepSpeed row."""

    model = _named("my_project.cfg", "DeepspeedConfigHelper")
    row = report(model, torch.randn(1)).row("deepspeed")

    assert row.detected is False


def test_deepspeed_row_detects_real_deepspeed_namespace() -> None:
    """A class under the ``deepspeed`` namespace stays detected."""

    model = _named("deepspeed.runtime.engine", "DeepSpeedEngine")
    row = report(model, torch.randn(1)).row("deepspeed")

    assert row.detected is True
    assert row.status == "scope"


# ---------------------------------------------------------------------------
# A3-02 — Lightning training_step detection must require real Lightning identity,
# not merely a callable method named ``training_step`` on a train-mode module.
# ---------------------------------------------------------------------------


class OrdinaryTrainingUtility(nn.Module):
    """Plain module with a conventional ``training_step`` helper method."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """

        return x

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Return the batch unchanged.

        Parameters
        ----------
        batch:
            Training batch.

        Returns
        -------
        torch.Tensor
            The batch.
        """

        return batch


def test_lightning_row_ignores_plain_training_step_method() -> None:
    """A plain train-mode module with a ``training_step`` method is not Lightning."""

    model = OrdinaryTrainingUtility()
    assert model.training is True  # nn.Module defaults to train mode

    row = report(model, torch.randn(1)).row("lightning_training_step")

    assert row.detected is False
    assert row.status == "pass"
    assert row.severity == "ok"


def test_lightning_row_detects_real_lightning_module_in_train_mode() -> None:
    """A genuine LightningModule in training mode stays flagged known_broken."""

    pl = pytest.importorskip("pytorch_lightning")

    class RealLit(pl.LightningModule):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return input unchanged."""

            return x

        def training_step(self, batch: torch.Tensor) -> torch.Tensor:
            """Return batch unchanged."""

            return batch

    model = RealLit()
    model.train()
    row = report(model, torch.randn(1)).row("lightning_training_step")

    assert row.detected is True
    assert row.status == "known_broken"
    assert row.severity == "error"


def test_lightning_row_clears_when_lightning_module_in_eval_mode() -> None:
    """A LightningModule switched to eval mode is a supported plain forward."""

    pl = pytest.importorskip("pytorch_lightning")

    class RealLit(pl.LightningModule):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return input unchanged."""

            return x

        def training_step(self, batch: torch.Tensor) -> torch.Tensor:
            """Return batch unchanged."""

            return batch

    model = RealLit()
    model.eval()
    row = report(model, torch.randn(1)).row("lightning_training_step")

    assert row.detected is False
    assert row.status == "pass"


# ---------------------------------------------------------------------------
# A3-03 — functorch/vmap detection must inspect executable AST references, not
# raw source substrings that also match comments and docstrings.
# ---------------------------------------------------------------------------


class DocstringMentionsVmapModel(nn.Module):
    """Model whose docstring mentions vmap but whose code never calls it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return input directly; this model does NOT call vmap or functorch."""

        # A comment that also mentions torch.func should be ignored.
        return x


class CommentMentionsFunctorchModel(nn.Module):
    """Model whose only functorch mention is an inline comment."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """

        result = x  # note: intentionally avoids functorch.vmap for tracing
        return result


class RealVmapModel(nn.Module):
    """Model that actually calls ``torch.vmap`` in its forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply an increment through ``torch.vmap``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Incremented tensor.
        """

        return torch.vmap(lambda t: t + 1)(x)


class RealTorchFuncModel(nn.Module):
    """Model that references the ``torch.func`` submodule in its forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply an increment through ``torch.func.vmap``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Incremented tensor.
        """

        return torch.func.vmap(lambda t: t + 1)(x)


def test_functorch_row_ignores_docstring_mention() -> None:
    """A docstring mentioning vmap must not flag the model as broken."""

    row = report(DocstringMentionsVmapModel(), torch.randn(3)).row("vmap_functorch")

    assert row.detected is False
    assert row.status == "pass"
    assert row.severity == "ok"


def test_functorch_row_ignores_comment_mention() -> None:
    """An inline comment mentioning functorch.vmap must not flag the model."""

    row = report(CommentMentionsFunctorchModel(), torch.randn(3)).row("vmap_functorch")

    assert row.detected is False


def test_functorch_row_detects_real_torch_vmap_call() -> None:
    """An actual ``torch.vmap`` call in forward stays detected."""

    row = report(RealVmapModel(), torch.randn(3)).row("vmap_functorch")

    assert row.detected is True
    assert row.status == "known_broken"


def test_functorch_row_detects_torch_func_submodule_reference() -> None:
    """A ``torch.func`` submodule reference in forward stays detected."""

    row = report(RealTorchFuncModel(), torch.randn(3)).row("vmap_functorch")

    assert row.detected is True


# ---------------------------------------------------------------------------
# A3-09 / A3-10 — Accelerate offload detection must key on real offload flags,
# not on the truthiness of execution_device (which is present for plain
# single-device dispatch and falsy for device index 0).
# ---------------------------------------------------------------------------


class _StubAlignDevicesHook:
    """Minimal stand-in for accelerate's ``AlignDevicesHook`` (not installed)."""

    def __init__(
        self,
        offload: bool = False,
        offload_buffers: bool = False,
        execution_device: object = None,
    ) -> None:
        """Store the hook flags under inspection.

        Parameters
        ----------
        offload:
            Whether weights are offloaded.
        offload_buffers:
            Whether buffers are offloaded.
        execution_device:
            Device the module executes on (not an offload signal).
        """

        self.offload = offload
        self.offload_buffers = offload_buffers
        self.execution_device = execution_device


def _offload_row_for(hook: _StubAlignDevicesHook) -> object:
    """Attach a hook to a module and return its offload row.

    Parameters
    ----------
    hook:
        Stub accelerate hook to attach as ``_hf_hook``.

    Returns
    -------
    CompatRow
        The ``accelerate_cpu_disk_offload`` row.
    """

    module = nn.Linear(2, 2)
    module._hf_hook = hook  # type: ignore[assignment]
    return report(module, torch.randn(1, 2)).row("accelerate_cpu_disk_offload")


def test_offload_row_ignores_plain_dispatch_hook_with_execution_device() -> None:
    """A hook with offload=False but an execution_device must not read as offload."""

    row = _offload_row_for(
        _StubAlignDevicesHook(offload=False, execution_device=torch.device("cpu"))
    )

    assert row.detected is False
    assert row.status == "pass"


def test_offload_row_detection_is_independent_of_device_id_truthiness() -> None:
    """Device index 0 and 1 must give the same (non-offload) verdict for offload=False."""

    row_zero = _offload_row_for(_StubAlignDevicesHook(offload=False, execution_device=0))
    row_one = _offload_row_for(_StubAlignDevicesHook(offload=False, execution_device=1))

    assert row_zero.detected is False
    assert row_one.detected is False
    assert row_zero.detected == row_one.detected


def test_offload_row_detects_real_weight_offload() -> None:
    """A hook with offload=True stays detected even on device index 0.

    Lane F37: offloaded capture is SUPPORTED (hook internals under
    pause_logging, materialized weights re-attributed), so detection now
    reads pass/info rather than the historical known_broken/error.
    """

    row = _offload_row_for(_StubAlignDevicesHook(offload=True, execution_device=0))

    assert row.detected is True
    assert row.status == "pass"
    assert row.severity == "info"


def test_offload_row_detects_buffer_offload() -> None:
    """Buffer offload (offload_buffers=True) is also detected."""

    row = _offload_row_for(_StubAlignDevicesHook(offload=False, offload_buffers=True))

    assert row.detected is True


# ---------------------------------------------------------------------------
# A3-12 (+ LOW-10) — the tied-parameter fallback must enumerate without dedup so
# ties stay visible, and enumeration failure must not read as "no ties".
# ---------------------------------------------------------------------------


class LegacySignatureTiedModel(nn.Module):
    """Tied model whose ``named_parameters`` override rejects ``remove_duplicate``."""

    def __init__(self) -> None:
        """Register two names bound to one shared submodule."""

        super().__init__()
        self.left = nn.Linear(2, 2)
        self.right = self.left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared linear twice.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return self.right(self.left(x))

    def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
        """Override without a ``remove_duplicate`` keyword (legacy signature).

        Parameters
        ----------
        prefix:
            Name prefix.
        recurse:
            Whether to recurse into submodules.

        Returns
        -------
        Iterator
            Deduplicated parameter iterator from the base implementation.
        """

        return super().named_parameters(prefix=prefix, recurse=recurse)


class OrdinaryTiedModel(nn.Module):
    """Model with a shared embedding across two attribute names."""

    def __init__(self) -> None:
        """Register a shared embedding."""

        super().__init__()
        self.a = nn.Embedding(8, 4)
        self.b = self.a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Look up the shared embedding.

        Parameters
        ----------
        x:
            Token ids.

        Returns
        -------
        torch.Tensor
            Embedding output.
        """

        return self.b(x)


class UninspectableParamModel(nn.Module):
    """Model whose parameter enumeration fails outright."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """

        return x

    def named_parameters(self, *args: object, **kwargs: object):  # type: ignore[override]
        """Always fail enumeration.

        Parameters
        ----------
        args:
            Ignored positional arguments.
        kwargs:
            Ignored keyword arguments.

        Raises
        ------
        RuntimeError
            Always.
        """

        raise RuntimeError("parameter enumeration unavailable")


def test_tied_row_detects_ties_under_legacy_named_parameters_signature() -> None:
    """A legacy override that dedups must not make ties invisible (fail-open)."""

    row = report(LegacySignatureTiedModel(), torch.randn(1, 2)).row("tied_parameters")

    assert row.detected is True
    assert row.status == "pass"
    assert row.severity == "info"


def test_tied_row_detects_ordinary_shared_parameters() -> None:
    """A plain shared parameter object stays detected via the primary path."""

    row = report(OrdinaryTiedModel(), torch.tensor([1, 2, 3])).row("tied_parameters")

    assert row.detected is True


def test_tied_row_is_honest_when_enumeration_fails() -> None:
    """A failed enumeration must not be reported as a positive 'no ties' fact."""

    row = report(UninspectableParamModel(), torch.randn(1)).row("tied_parameters")

    assert row.detected is False
    assert "could not be inspected" in row.details
    assert "No tied/shared parameter objects detected" not in row.details


# ---------------------------------------------------------------------------
# LOW-8 — show() must expose the same Suggestion column as to_markdown().
# ---------------------------------------------------------------------------


def test_show_includes_suggestion_column_like_markdown() -> None:
    """``show()`` renders a Suggestion column so it agrees with ``to_markdown()``."""

    # A DataParallel model produces a row with a non-empty suggestion.
    compat_report = report(nn.DataParallel(_Passthrough()), torch.randn(1))

    text_table = compat_report.show()
    markdown_table = compat_report.to_markdown()

    header_line = text_table.splitlines()[3]
    assert "Suggestion" in header_line
    assert "Suggestion" in markdown_table

    # The concrete suggestion text present in markdown is also present in
    # show(). show() wraps long cells inside the designed width cap (bounded
    # designed reprs), so presence is asserted per wrapped fragment: every
    # fragment of the suggestion must appear on some table line.
    dp_suggestion = compat_report.row("data_parallel").suggestion
    assert dp_suggestion
    for fragment in textwrap.wrap(dp_suggestion, width=_SHOW_CELL_WIDTH_CAP):
        assert fragment in text_table
    assert dp_suggestion in markdown_table
