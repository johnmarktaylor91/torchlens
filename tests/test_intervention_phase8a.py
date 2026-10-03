"""Phase 8a mutator-method tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from example_models import TinyReluAdd as ReluAdd

import torchlens as tl
from torchlens.data_classes.cleanup import _scrub_intervention_fields_after_removal
from torchlens.intervention.errors import SpecMutationError
from torchlens.intervention.handles import HookHandle
from torchlens.intervention.types import FireRecord, TargetSpec
from torchlens.io import TraceState
from torchlens.options import CaptureOptions


def _identity_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Return the out unchanged.

    Parameters
    ----------
    out:
        Activation passed to the hook.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Original out.
    """

    del hook
    return out


def _capture() -> Any:
    """Capture an intervention-ready log for Phase 8a tests.

    Returns
    -------
    Any
        Intervention-ready model log.
    """

    return tl.trace(
        ReluAdd(),
        torch.randn(2, 3),
        capture=CaptureOptions(intervention_ready=True),
    )


def _first_internal_label(log: Any) -> str:
    """Return the first non-input/non-output/non-buffer layer label.

    Parameters
    ----------
    log:
        Trace-like object with ``layer_list`` entries.

    Returns
    -------
    str
        First internal layer label.
    """

    for layer in log.layer_list:
        if not (layer.is_input or layer.is_output or layer.is_buffer):
            return layer.layer_label
    raise AssertionError("expected at least one internal layer")


def test_set_tensor_marks_spec_stale_and_returns_self() -> None:
    """``set(site, tensor)`` updates recipe state without propagation."""

    log = _capture()
    initial_revision = log._spec_revision
    replacement = torch.zeros(2, 3)

    result = log.set(tl.func("relu"), replacement)

    assert result is log
    assert log._spec_revision == initial_revision + 1
    assert log.state is TraceState.SPEC_STALE
    assert log._out_recipe_revision == initial_revision
    assert len(log._intervention_spec.target_value_specs) == 1
    value_spec = log._intervention_spec.target_value_specs[0]
    assert value_spec.value is replacement
    assert value_spec.metadata == {}
    assert not log._recipe_is_clean()


def test_set_callable_tags_one_shot_metadata() -> None:
    """``set(site, fn)`` tags the spec entry as a one-shot callable set."""

    log = _capture()

    def replacement_fn(out: torch.Tensor) -> torch.Tensor:
        """Return a zero out matching the original.

        Parameters
        ----------
        out:
            Matched out.

        Returns
        -------
        torch.Tensor
            Zero replacement.
        """

        return out * 0

    log.set(tl.func("relu"), replacement_fn)

    value_spec = log._intervention_spec.target_value_specs[-1]
    assert value_spec.value is replacement_fn
    assert value_spec.metadata["created_by"] == "set_callable_one_shot"


def test_attach_clear_and_detach_hooks_are_sticky_mutators() -> None:
    """Sticky hook mutators return handles, increment revisions, and mark stale."""

    log = _capture()
    initial_revision = log._spec_revision

    handle = log.attach_hooks({tl.func("relu"): _identity_hook})
    assert isinstance(handle, HookHandle)
    assert log._spec_revision == initial_revision + 1
    assert log.state is TraceState.SPEC_STALE
    assert len(log._intervention_spec.hook_specs) == 1

    assert log.clear_hooks() is log
    assert log._spec_revision == initial_revision + 2
    assert log._intervention_spec.hook_specs == []

    log.attach_hooks({tl.func("relu"): _identity_hook})
    assert log._spec_revision == initial_revision + 3
    assert len(log._intervention_spec.hook_specs) == 1

    assert log.detach_hooks(tl.func("relu")) is log
    assert log._spec_revision == initial_revision + 4
    assert log._intervention_spec.hook_specs == []


def test_attach_hooks_handle_removes_only_its_own_specs() -> None:
    """Each attach handle removes only the hook specs it created."""

    log = _capture()

    first = log.attach_hooks(tl.func("relu"), _identity_hook, confirm_mutation=True)
    second = log.attach_hooks(tl.func("__add__"), _identity_hook, confirm_mutation=True)

    assert isinstance(first, HookHandle)
    assert isinstance(second, HookHandle)
    assert first is not second
    assert len(log._intervention_spec.hook_specs) == 2

    first.remove()

    assert len(log._intervention_spec.hook_specs) == 1
    remaining = log._intervention_spec.hook_specs[0]
    assert remaining.handle in second.handle_ids


def test_intervention_spec_cached_property_invalidates_after_mutators() -> None:
    """Frozen intervention-spec snapshots refresh after each public mutator."""

    log = _capture()
    first_snapshot = log.intervention_spec
    assert log.intervention_spec is first_snapshot

    replacement = torch.zeros(2, 3)
    log.set(tl.func("relu"), replacement, confirm_mutation=True)
    after_set = log.intervention_spec
    assert after_set is not first_snapshot
    assert len(after_set.target_value_specs) == 1

    log.attach_hooks(tl.func("relu"), _identity_hook, confirm_mutation=True)
    after_attach = log.intervention_spec
    assert after_attach is not after_set
    assert len(after_attach.hook_specs) == 1

    log.detach_hooks(tl.func("relu"), confirm_mutation=True)
    after_detach = log.intervention_spec
    assert after_detach is not after_attach
    assert after_detach.hook_specs == ()

    log.attach_hooks(tl.func("relu"), _identity_hook, confirm_mutation=True)
    after_second_attach = log.intervention_spec
    log.clear_hooks(confirm_mutation=True)
    after_clear = log.intervention_spec
    assert after_clear is not after_second_attach
    assert after_clear.hook_specs == ()

    log.do(
        tl.func("relu"),
        torch.ones(2, 3),
        intervention=tl.options.InterventionOptions(engine="set_only", confirm_mutation=True),
    )
    after_do = log.intervention_spec
    assert after_do is not after_clear
    assert len(after_do.target_value_specs) == 2


def test_cleanup_scrubs_all_label_bearing_intervention_spec_fields() -> None:
    """Cleanup removes deleted-label entries from every list-backed spec field."""

    log = _capture()
    target = _first_internal_label(log)
    log.set(target, torch.zeros_like(log[target].out), confirm_mutation=True)
    log.attach_hooks(tl.label(target), _identity_hook, confirm_mutation=True)
    log._intervention_spec.targets.append(TargetSpec("label", target))
    log._intervention_spec.records.append(
        FireRecord(target_label=target, site_label=target, call_label=log[target].label)
    )
    log.state_history.append({"op": "manual", "site": target})

    _scrub_intervention_fields_after_removal(
        log,
        {target},
        [layer for layer in log.layer_list if layer.layer_label != target],
    )

    assert log._intervention_spec.targets == []
    assert log._intervention_spec.target_value_specs == []
    assert log._intervention_spec.hook_specs == []
    assert log._intervention_spec.records == []
    assert all(
        record.get("site") != target for record in log.state_history if isinstance(record, dict)
    )


def test_cleanup_invalidates_cached_intervention_spec_snapshot() -> None:
    """Cleanup refreshes the cached frozen intervention-spec view after spec mutation."""

    log = _capture()
    target = _first_internal_label(log)
    log._intervention_spec.records.append(
        FireRecord(target_label=target, site_label=target, call_label=log[target].label)
    )

    frozen_before = log.intervention_spec
    assert len(frozen_before.records) == 1

    _scrub_intervention_fields_after_removal(
        log,
        {target},
        [layer for layer in log.layer_list if layer.layer_label != target],
    )

    frozen_after = log.intervention_spec
    assert frozen_after is not frozen_before
    assert len(log._intervention_spec.records) == 0
    assert frozen_after.records == ()


def test_detach_hooks_no_site_is_noop_unless_strict() -> None:
    """``detach_hooks()`` is a non-mutating no-op unless strict mode is requested."""

    log = _capture()
    initial_revision = log._spec_revision

    assert log.detach_hooks() is log
    assert log._spec_revision == initial_revision

    with pytest.raises(SpecMutationError, match="requires a site or handle"):
        log.detach_hooks(strict=True)


def test_rerun_advances_out_recipe_revision_after_set() -> None:
    """Successful rerun advances the out recipe revision."""

    x = torch.randn(2, 3)
    log = tl.trace(ReluAdd(), x, capture=CaptureOptions(intervention_ready=True))

    log.set(tl.func("relu"), torch.zeros(2, 3), confirm_mutation=True)
    assert log._out_recipe_revision == 0

    result = log.run(ReluAdd(), x)

    assert result is log
    assert log.state is TraceState.RERUN_PROPAGATED
    assert log._out_recipe_revision == log._spec_revision
    assert log._recipe_is_clean()


def test_fork_and_auto_do_are_implemented_by_phase8b() -> None:
    """Phase 8b implements fork and auto dispatch paths."""

    log = _capture()

    fork = log.fork("candidate")
    assert fork is not log
    assert fork.trace_label == "candidate"
    assert fork.parent_run() is log

    result = log.do(
        {tl.func("relu"): _identity_hook},
        intervention=tl.options.InterventionOptions(confirm_mutation=True),
    )
    assert result is log
    assert log.state is TraceState.REPLAY_PROPAGATED
