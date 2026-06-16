"""Phase 7 rerun engine and atomic swap tests."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.io import TraceState
from torchlens.intervention.errors import (
    ControlFlowDivergenceError,
    ControlFlowDivergenceWarning,
)
from torchlens.intervention.rerun import rerun
from torchlens.intervention.types import InterventionSpec, Relationship, TargetSpec


class ReluAdd(torch.nn.Module):
    """Small model with a hookable relu feeding downstream output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply relu and a downstream add.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Result tensor.
        """

        return torch.relu(x) + 1


class BadModel(torch.nn.Module):
    """Model that always fails during rerun."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raise a deterministic rerun failure.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            This method never returns.
        """

        raise RuntimeError("boom")


class BranchModel(torch.nn.Module):
    """Model whose control flow changes the captured graph shape."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Choose a branch based on input sign.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch output.
        """

        if bool(torch.sum(x) > 0):
            return torch.relu(x)
        return torch.sigmoid(x)


class InplaceVersionModel(torch.nn.Module):
    """Model with an in-place child that records child-version snapshots."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate an intermediate in place.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Mutated downstream output.
        """

        y = x + 1
        y.relu_()
        return y * 2


def _zero_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Return a zeroed out.

    Parameters
    ----------
    out:
        Activation passed to the hook.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Zeroed out.
    """

    del hook
    return out * 0


def _capture(model: torch.nn.Module, x: torch.Tensor) -> tl.Trace:
    """Capture an intervention-ready log for Phase 7 tests.

    Parameters
    ----------
    model:
        Model to log.
    x:
        Input tensor.

    Returns
    -------
    tl.Trace
        Captured log.
    """

    return tl.trace(model, x, intervention_ready=True)


@pytest.mark.smoke
def test_rerun_baseline_matches_original_graph_hash_and_sets_state() -> None:
    """No-op rerun re-captures the same graph and updates run state."""

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)
    original_hash = log.graph_shape_hash
    original_raw_hash = log._raw_event_shape_hash  # noqa: SLF001
    original_history_len = len(log.state_history)

    result = log.rerun(ReluAdd(), x)

    assert result is log
    assert log.state is TraceState.RERUN_PROPAGATED
    assert log.graph_shape_hash == original_hash
    assert log._raw_event_shape_hash == original_raw_hash  # noqa: SLF001
    assert log.last_run["engine"] == "rerun"
    assert log.last_run["old_raw_event_shape_hash"] == original_raw_hash
    assert log.last_run["new_raw_event_shape_hash"] == original_raw_hash
    assert len(log.state_history) == original_history_len + 1
    assert log.state_history[-1]["engine"] == "rerun"


@pytest.mark.smoke
def test_rerun_with_hook_updates_downstream_out() -> None:
    """Rerun installs the active spec so live hooks affect downstream output."""

    x = torch.tensor([[-1.0, 2.0, 3.0]])
    log = _capture(ReluAdd(), x)
    original_output = log[log.output_layers[0]].out.clone()
    log._intervention_spec = InterventionSpec(
        targets=[TargetSpec("func", "relu")],
        hook=_zero_hook,
    )

    log.rerun(ReluAdd(), x)

    relu_site = next(layer for layer in log.layer_list if layer.func_name == "relu")
    assert torch.equal(relu_site.out, torch.zeros_like(relu_site.out))
    assert torch.equal(log[log.output_layers[0]].out, torch.ones_like(original_output))
    assert not torch.equal(log[log.output_layers[0]].out, original_output)
    assert relu_site.interventions[-1].engine == "live"


@pytest.mark.smoke
def test_rerun_failure_leaves_original_log_unchanged() -> None:
    """Fresh-capture failures happen before atomic swap."""

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)
    original_hash = log.graph_shape_hash
    original_labels = tuple(log.layer_labels)
    log.rerun(ReluAdd(), x)
    history_after_success = list(log.state_history)

    with pytest.raises(RuntimeError, match="boom"):
        log.rerun(BadModel(), x)

    assert log.state is TraceState.RERUN_PROPAGATED
    assert log.graph_shape_hash == original_hash
    assert tuple(log.layer_labels) == original_labels
    assert log.state_history == history_after_success


@pytest.mark.smoke
def test_rerun_keyboard_interrupt_during_build_leaves_original_log_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interruptions before validation do not partially swap rerun state."""

    rerun_module = importlib.import_module("torchlens.intervention.rerun")

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)
    original_hash = log.graph_shape_hash
    original_labels = tuple(log.layer_labels)
    original_ledger = list(log.state_history)
    original_output = log[log.output_layers[0]].out.clone()
    original_run_state = log.state

    def interrupt_capture(*_: Any, **__: Any) -> tl.Trace:
        """Raise as if the fresh off-side capture was interrupted."""

        raise KeyboardInterrupt("simulated interrupt")

    monkeypatch.setattr(rerun_module, "_capture_with_active_spec", interrupt_capture)

    with pytest.raises(KeyboardInterrupt, match="simulated interrupt"):
        log.rerun(ReluAdd(), x)

    assert log.state is original_run_state
    assert log.graph_shape_hash == original_hash
    assert tuple(log.layer_labels) == original_labels
    assert log.state_history == original_ledger
    assert torch.equal(log[log.output_layers[0]].out, original_output)


def test_rerun_strict_divergence_raises_before_swap() -> None:
    """Strict rerun rejects graph-shape divergence and preserves old state."""

    positive = torch.ones(2, 3)
    negative = -torch.ones(2, 3)
    log = _capture(BranchModel(), positive)
    original_hash = log.graph_shape_hash
    original_raw_hash = log._raw_event_shape_hash  # noqa: SLF001

    with pytest.raises(ControlFlowDivergenceError):
        log.rerun(BranchModel(), negative, strict=True)

    assert log.graph_shape_hash == original_hash
    assert log._raw_event_shape_hash == original_raw_hash  # noqa: SLF001
    assert log.state is TraceState.PRISTINE


def test_rerun_non_strict_divergence_warns_and_swaps() -> None:
    """Non-strict rerun reports graph divergence and keeps the new run."""

    positive = torch.ones(2, 3)
    negative = -torch.ones(2, 3)
    log = _capture(BranchModel(), positive)
    original_hash = log.graph_shape_hash
    original_raw_hash = log._raw_event_shape_hash  # noqa: SLF001

    with pytest.warns(ControlFlowDivergenceWarning):
        log.rerun(BranchModel(), negative)

    assert log.graph_shape_hash != original_hash
    assert log._raw_event_shape_hash != original_raw_hash  # noqa: SLF001
    assert log.state is TraceState.RERUN_PROPAGATED
    assert log.last_run["divergence_count"] == 1


def test_rerun_honors_metadata_only_save_scope() -> None:
    """Rerun of a metadata-only trace does not save every activation."""

    x = torch.randn(2, 3)
    log = tl.trace(ReluAdd(), x, save=lambda _ctx: False)

    assert log.num_saved_ops == 0

    log.rerun(ReluAdd(), x + 1)

    assert log.num_saved_ops == 0
    assert log.last_run["engine"] == "rerun"


def test_rerun_matching_graph_refreshes_existing_ops_without_full_swap() -> None:
    """Same-shape rerun updates existing Op payload fields in place."""

    x = torch.tensor([[-1.0, 2.0, 3.0]])
    new_x = torch.tensor([[4.0, 5.0, -6.0]])
    log = tl.trace(ReluAdd(), x, intervention_ready=True, activation_transform=lambda t: t * 2)
    relu_site = next(layer for layer in log.layer_list if layer.func_name == "relu")
    original_relu_site_id = id(relu_site)
    original_out = relu_site.out.clone()

    log.rerun(ReluAdd(), new_x)
    refreshed_relu_site = next(layer for layer in log.layer_list if layer.func_name == "relu")

    assert id(refreshed_relu_site) == original_relu_site_id
    assert log.last_run["fast_refresh"] is True
    assert torch.equal(refreshed_relu_site.out, torch.tensor([[4.0, 5.0, 0.0]]))
    assert not torch.equal(refreshed_relu_site.out, original_out)
    assert refreshed_relu_site.shape == (1, 3)
    assert refreshed_relu_site.dtype == torch.float32
    assert int(refreshed_relu_site.activation_memory) == 12
    assert torch.equal(refreshed_relu_site.transformed_out, refreshed_relu_site.out * 2)
    assert refreshed_relu_site.transformed_out_shape == (1, 3)
    assert refreshed_relu_site.transformed_out_dtype == torch.float32
    assert int(refreshed_relu_site.transformed_activation_memory) == 12
    assert torch.equal(log[log.output_layers[0]].out, torch.tensor([[5.0, 6.0, 1.0]]))


def test_rerun_fast_refresh_repopulates_child_versions() -> None:
    """Same-shape rerun refreshes save_arg_values child-version snapshots."""

    x = torch.tensor([-2.0, 3.0])
    new_x = torch.tensor([-5.0, 1.0])
    log = tl.trace(InplaceVersionModel(), x, save_arg_values=True)
    add_site = log["add_1_1"]
    original_add_site_id = id(add_site)

    log.rerun(InplaceVersionModel(), new_x)

    refreshed_add_site = log["add_1_1"]
    assert id(refreshed_add_site) == original_add_site_id
    assert log.last_run["fast_refresh"] is True
    assert torch.equal(
        refreshed_add_site.out_versions_by_child["relu_1_2"],
        torch.tensor([-4.0, 2.0]),
    )


def test_replace_run_state_preserves_relationship_and_spec_fields() -> None:
    """Atomic swap keeps recipe, warning flags, history, and evidence fields."""

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)
    new_log = _capture(ReluAdd(), x + 1)
    spec = InterventionSpec(targets=[TargetSpec("func", "relu")], hook=_zero_hook)
    history: list[Any] = [{"engine": "before"}]
    log.trace_label = "kept"
    log.parent_run = "parent-sentinel"  # type: ignore[assignment]
    log._intervention_spec = spec
    log.state_history = history
    log._warned_direct_write = True
    log._warned_mutate_in_place = True
    log.model_object_id = 123
    log.model_class_qualname = "kept.Model"
    log.param_hash_quick = "weights-a"
    log.param_hash_full = "weights-full"
    log.input_object_id = 456
    log.input_signature_hash = "input-hash"
    log.is_appended = True
    log.relationship_evidence = {"model": Relationship.SAME_OBJECT}
    log._spec_revision = 7

    log.replace_state_from(new_log)

    assert log.trace_label == "kept"
    assert log.parent_run == "parent-sentinel"
    assert log._intervention_spec is spec
    assert log.state_history is history
    assert log._warned_direct_write is True
    assert log._warned_mutate_in_place is True
    assert log.model_object_id == 123
    assert log.model_class_qualname == "kept.Model"
    assert log.param_hash_quick == "weights-a"
    assert log.param_hash_full == "weights-full"
    assert log.input_object_id == 456
    assert log.input_signature_hash == "input-hash"
    assert log.is_appended is True
    assert log.relationship_evidence == {"model": Relationship.SAME_OBJECT}
    assert log._spec_revision == 7
    assert log.graph_shape_hash == new_log.graph_shape_hash
    assert log.layer_labels == new_log.layer_labels


def test_rerun_append_true_dispatches_to_append() -> None:
    """Phase 12 implements append rerun through the function API."""

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)

    rerun(log, ReluAdd(), x, append=True)

    assert log.state is TraceState.APPENDED


def test_rerun_x_none_requires_explicit_input() -> None:
    """Phase 7 does not reconstruct original inputs from ids."""

    log = _capture(ReluAdd(), torch.randn(2, 3))

    with pytest.raises(ValueError, match="Pass the forward input explicitly"):
        log.rerun(ReluAdd())


def test_rerun_rejects_torchscript_model() -> None:
    """Rerun uses the same opaque-wrapper rejection as capture."""

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)
    scripted = torch.jit.trace(ReluAdd(), x)

    with pytest.raises(RuntimeError, match="ScriptModule"):
        log.rerun(scripted, x)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_rerun_rejects_compiled_model() -> None:
    """Rerun rejects torch.compile wrappers before capture."""

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)
    compiled = torch.compile(ReluAdd(), backend="eager")

    with pytest.raises(RuntimeError, match="torch.compile"):
        log.rerun(compiled, x)


def test_rerun_rejects_fsdp_when_constructible() -> None:
    """Rerun rejects FSDP wrappers when the local environment can build one."""

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except (ImportError, RuntimeError):
        pytest.skip("FSDP unavailable")

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)
    try:
        fsdp_model = FullyShardedDataParallel(ReluAdd())
    except Exception as exc:
        pytest.skip(f"FSDP cannot be constructed in this environment: {exc}")

    with pytest.raises(RuntimeError, match="FullyShardedDataParallel"):
        log.rerun(fsdp_model, x)
