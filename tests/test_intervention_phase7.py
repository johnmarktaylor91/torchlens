"""Phase 7 rerun engine and atomic swap tests."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import pytest
import torch
from example_models import TinyReluAdd as ReluAdd

import torchlens as tl
from torchlens._capture_state_helpers import reset_compiled_model_unwrap_warning_state
from torchlens.intervention.errors import (
    ControlFlowDivergenceError,
    ControlFlowDivergenceWarning,
)
from torchlens.intervention.rerun import run
from torchlens.intervention.types import InterventionSpec, Relationship, TargetSpec
from torchlens.io import TraceState
from torchlens.options import CaptureOptions


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

    return tl.trace(model, x, capture=CaptureOptions(intervention_ready=True))


def test_rerun_baseline_matches_original_graph_hash_and_sets_state() -> None:
    """No-op rerun re-captures the same graph and updates run state."""

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)
    original_hash = log.graph_shape_hash
    original_raw_hash = log._raw_event_shape_hash  # noqa: SLF001
    original_history_len = len(log.state_history)

    result = log.run(ReluAdd(), x)

    assert result is log
    assert log.state is TraceState.RERUN_PROPAGATED
    assert log.graph_shape_hash == original_hash
    assert log._raw_event_shape_hash == original_raw_hash  # noqa: SLF001
    assert log.last_run["engine"] == "rerun"
    assert log.last_run["old_raw_event_shape_hash"] == original_raw_hash
    assert log.last_run["new_raw_event_shape_hash"] == original_raw_hash
    assert len(log.state_history) == original_history_len + 1
    assert log.state_history[-1]["engine"] == "rerun"


def test_rerun_with_hook_updates_downstream_out() -> None:
    """Rerun installs the active spec so live hooks affect downstream output."""

    x = torch.tensor([[-1.0, 2.0, 3.0]])
    log = _capture(ReluAdd(), x)
    original_output = log[log.output_layers[0]].out.clone()
    log._intervention_spec = InterventionSpec(
        targets=[TargetSpec("func", "relu")],
        hook=_zero_hook,
    )

    log.run(ReluAdd(), x)

    relu_site = next(layer for layer in log.layer_list if layer.func_name == "relu")
    assert torch.equal(relu_site.out, torch.zeros_like(relu_site.out))
    assert torch.equal(log[log.output_layers[0]].out, torch.ones_like(original_output))
    assert not torch.equal(log[log.output_layers[0]].out, original_output)
    assert relu_site.interventions[-1].engine == "live"
    assert log.last_run["hooks_fired"] >= 1
    assert log.last_run["hooks_unfired"] == 0


def test_rerun_warns_when_value_dependent_sticky_hook_stops_matching() -> None:
    """A hook resolved on the source trace reports an unfired rerun plan entry."""
    model = ReluAdd()
    x = torch.randn(2, 3)
    log = _capture(model, x)
    gate = {"active": True}
    selector = tl.func("relu") & tl.where(
        lambda _op: gate["active"],
        name_hint="runtime_gate",
    )
    log.attach_hooks(selector, tl.zero_ablate())
    gate["active"] = False

    with pytest.warns(UserWarning, match="fired at zero sites on the new inputs"):
        log.run(model, x + 1)

    assert log.last_run["hooks_fired"] == 0
    assert log.last_run["hooks_unfired"] == 1


def test_rerun_matching_sticky_hook_has_no_unfired_warning() -> None:
    """A sticky hook that fires is reconciled without a zero-site warning."""
    model = ReluAdd()
    log = _capture(model, torch.randn(2, 3))
    log.attach_hooks(tl.func("relu"), tl.zero_ablate())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log.run(model, torch.randn(2, 3))

    assert not any("fired at zero sites" in str(item.message) for item in caught)
    assert log.last_run["hooks_fired"] >= 1
    assert log.last_run["hooks_unfired"] == 0


def test_rerun_colliding_hook_identifiers_do_not_hide_misses() -> None:
    """Two plan entries sharing one fallback identifier are audited per entry.

    ``_hook_plan_identifier`` falls back through plan id -> hook id -> helper
    name -> callable qualname, so two entries with different targets but the
    same callable both keyed the audit Counter with one shared string: two
    fires of the first entry hid the second entry's total miss (``fired=2``,
    ``unfired=()``, no warning) -- incomplete f9f5b140.
    """

    from types import SimpleNamespace

    from torchlens.intervention.hooks import NormalizedHookEntry
    from torchlens.intervention.rerun import (
        _assign_unique_plan_ids,
        _hook_plan_identifier,
        _reconcile_rerun_hook_fires,
    )
    from torchlens.ir.intervention import FireResult

    def same_hook(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        """Shared callable planned under two different targets."""

        del hook
        return out

    entries = [
        NormalizedHookEntry(site_target=object(), normalized_callable=same_hook),
        NormalizedHookEntry(site_target=object(), normalized_callable=same_hook),
    ]
    raw_ids = [_hook_plan_identifier(entry) for entry in entries]
    assert raw_ids[0] == raw_ids[1], "precondition: the fallback identifiers collide"

    plan = _assign_unique_plan_ids(entries)
    plan_ids = [_hook_plan_identifier(entry) for entry in plan]
    assert len(set(plan_ids)) == 2, "colliding fallbacks must become unique accounting keys"

    def _fire(plan_id: str) -> FireResult:
        """One synthetic live fire for the given accounting id."""

        return FireResult(
            plan_id=plan_id,
            site_label="relu_1_1",
            fired_at_capture_index=0,
            pre_hook_shape=(1,),
            post_hook_shape=(1,),
            pre_hook_dtype="torch.float32",
            post_hook_dtype="torch.float32",
            replaced=True,
            fire_record=None,
        )

    # Entry 1 fires twice (two sites), entry 2 never fires: the audit must
    # report the miss instead of letting the shared string absorb it.
    stub_log = SimpleNamespace(
        layer_list=[SimpleNamespace(fire_results=[_fire(plan_ids[0]), _fire(plan_ids[0])])]
    )
    with pytest.warns(UserWarning, match="fired at zero sites on the new inputs"):
        fired, unfired = _reconcile_rerun_hook_fires(stub_log, plan)
    assert fired == 2
    assert unfired == (plan_ids[1],)


def test_rerun_failure_leaves_original_log_unchanged() -> None:
    """Fresh-capture failures happen before atomic swap."""

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)
    original_hash = log.graph_shape_hash
    original_labels = tuple(log.layer_labels)
    log.run(ReluAdd(), x)
    history_after_success = list(log.state_history)

    with pytest.raises(RuntimeError, match="boom"):
        log.run(BadModel(), x)

    assert log.state is TraceState.RERUN_PROPAGATED
    assert log.graph_shape_hash == original_hash
    assert tuple(log.layer_labels) == original_labels
    assert log.state_history == history_after_success


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
        log.run(ReluAdd(), x)

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
        log.run(BranchModel(), negative, replay=tl.options.ReplayOptions(strict=True))

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
        log.run(BranchModel(), negative)

    assert log.graph_shape_hash != original_hash
    assert log._raw_event_shape_hash != original_raw_hash  # noqa: SLF001
    assert log.state is TraceState.RERUN_PROPAGATED
    assert log.last_run["divergence_count"] == 1


def test_rerun_honors_metadata_only_save_scope() -> None:
    """Rerun of a metadata-only trace does not save every activation."""

    x = torch.randn(2, 3)
    log = tl.trace(ReluAdd(), x, save=lambda _ctx: False)

    assert log.num_saved_ops == 0

    log.run(ReluAdd(), x + 1)

    assert log.num_saved_ops == 0
    assert log.last_run["engine"] == "rerun"


@pytest.mark.smoke
def test_rerun_matching_graph_refreshes_existing_ops_without_full_swap() -> None:
    """Same-shape rerun updates existing Op payload fields in place."""

    x = torch.tensor([[-1.0, 2.0, 3.0]])
    new_x = torch.tensor([[4.0, 5.0, -6.0]])
    log = tl.trace(
        ReluAdd(),
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True),
        save=tl.options.SaveOptions(activation_transform=lambda t: t * 2),
    )
    relu_site = next(layer for layer in log.layer_list if layer.func_name == "relu")
    original_relu_site_id = id(relu_site)
    original_out = relu_site.out.clone()

    log.run(ReluAdd(), new_x)
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
    log = tl.trace(
        InplaceVersionModel(), x, capture=tl.options.CaptureOptions(save_arg_values=True)
    )
    add_site = log["add_1_1"]
    original_add_site_id = id(add_site)

    log.run(InplaceVersionModel(), new_x)

    refreshed_add_site = log["add_1_1"]
    assert id(refreshed_add_site) == original_add_site_id
    assert log.last_run["fast_refresh"] is True
    assert torch.equal(
        refreshed_add_site.out_versions_by_child["relu_1_2"],
        torch.tensor([-4.0, 2.0]),
    )


class MultiPassAddBlock(torch.nn.Module):
    """Block whose two chained adds group into one two-pass layer."""

    def __init__(self) -> None:
        """Initialize three summed projections plus an output projection."""

        super().__init__()
        self.first = torch.nn.Linear(4, 4)
        self.second = torch.nn.Linear(4, 4)
        self.third = torch.nn.Linear(4, 4)
        self.out_proj = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum three projections through two structurally corresponding adds.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Projected sum.
        """

        return self.out_proj(self.first(x) + self.second(x) + self.third(x))


class MultiPassAdd(torch.nn.Module):
    """Wrapper giving the two-pass block a stable module address."""

    def __init__(self) -> None:
        """Initialize the wrapped block."""

        super().__init__()
        self.block = MultiPassAddBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the wrapped block.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Block output.
        """

        return self.block(x)


def test_rerun_fast_refresh_keeps_every_pass_of_a_multi_pass_layer_distinct() -> None:
    """Fast refresh must pair passes positionally, not through a label map.

    Neither the raw nor the final layer label is pass-qualified, so a label-keyed
    refresh map collapses an N-pass layer to its last pass and overwrites every
    earlier pass' activation with it -- silent corruption that only replay
    validation catches.
    """

    torch.manual_seed(0)
    model = MultiPassAdd()
    x = torch.randn(2, 4)
    new_x = torch.randn(2, 4)
    log = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    assert log.layer_logs["add_1_3"].num_passes == 2

    log.run(model, new_x)

    assert log.last_run["fast_refresh"] is True
    first_pass, second_pass = (log["add_1_3:1"], log["add_1_3:2"])
    assert first_pass.label == "add_1_3:1"
    assert second_pass.label == "add_1_3:2"
    first, second, third = (
        log["linear_1_1"].out,
        log["linear_2_2"].out,
        log["linear_3_4"].out,
    )
    assert torch.equal(first_pass.out, first + second)
    assert torch.equal(second_pass.out, first + second + third)
    assert not torch.equal(first_pass.out, second_pass.out)
    assert log.validate_forward_pass([log[log.output_layers[0]].out])


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

    run(log, ReluAdd(), x, replay=tl.options.ReplayOptions(append=True))

    assert log.state is TraceState.APPENDED


def test_rerun_x_none_requires_explicit_input() -> None:
    """Phase 7 does not reconstruct original inputs from ids."""

    log = _capture(ReluAdd(), torch.randn(2, 3))

    with pytest.raises(ValueError, match="forward input explicitly"):
        log.run(ReluAdd())


def test_rerun_rejects_torchscript_model() -> None:
    """Rerun uses the same opaque-wrapper rejection as capture."""

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)
    scripted = torch.jit.trace(ReluAdd(), x)

    with pytest.raises(RuntimeError, match="ScriptModule"):
        log.run(scripted, x)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_rerun_unwraps_compiled_model() -> None:
    """Rerun traces a torch.compile wrapper through its eager source module."""

    x = torch.randn(2, 3)
    log = _capture(ReluAdd(), x)
    compiled = torch.compile(ReluAdd(), backend="eager")

    reset_compiled_model_unwrap_warning_state()
    with pytest.warns(UserWarning, match="compiled model detected"):
        result = log.run(compiled, x)
    assert result is log


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
        log.run(fsdp_model, x)


def test_colliding_plan_identifiers_get_per_entry_occurrence_ids() -> None:
    """One entry's fires can never mask a colliding entry's total miss.

    grind-p5 3.3 rollup pin (b3p3-sol R15-1 / opus complementary): the
    identifier fallback ladder (helper name / callable qualname) collides
    across entries with different targets, and the fire audit compares
    Counter values keyed by that string -- two fires of entry ``a`` used to
    satisfy the whole bucket while entry ``b`` never fired
    (``fired=2, unfired=(), warnings=0``). ``_assign_unique_plan_ids`` stamps
    per-entry occurrence ids and ``_reconcile_rerun_hook_fires`` accounts
    per entry, with the FireRecord-only fallback pool capped at the
    shortfall.
    """

    from types import SimpleNamespace

    from torchlens.intervention.hooks import NormalizedHookEntry
    from torchlens.intervention.rerun import (
        _assign_unique_plan_ids,
        _reconcile_rerun_hook_fires,
    )

    def shared_callable(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        """One callable shared by two differently-targeted entries."""

        return out

    plan = _assign_unique_plan_ids(
        [
            NormalizedHookEntry(
                site_target="a",
                normalized_callable=shared_callable,
                helper_spec=None,
                metadata={},
            ),
            NormalizedHookEntry(
                site_target="b",
                normalized_callable=shared_callable,
                helper_spec=None,
                metadata={},
            ),
        ]
    )
    ids = [entry.metadata.get("plan_id") for entry in plan]
    assert len(set(ids)) == 2 and all(ids), ids

    fired_twice = SimpleNamespace(
        fire_results=(SimpleNamespace(plan_id=ids[0]), SimpleNamespace(plan_id=ids[0])),
        interventions=(),
    )
    log = SimpleNamespace(layer_list=[fired_twice])
    with pytest.warns(UserWarning, match="fired at zero sites"):
        total, unfired = _reconcile_rerun_hook_fires(log, plan)
    assert total == 2
    assert unfired == (ids[1],), "entry a's fires masked entry b's total miss"
