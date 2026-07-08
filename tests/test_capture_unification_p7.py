"""Phase 7 public captured-run type regression tests."""

from __future__ import annotations

import time

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation.invariants import (
    _check_special_layer_lists,
    _check_trace_self_consistency,
    check_metadata_invariants,
)


class ConvReluAdd(nn.Module):
    """Small convolutional model with saved and unsaved operation types."""

    def __init__(self) -> None:
        """Initialize deterministic module layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a conv, relu, and add operation."""

        return torch.relu(self.conv(x)) + 1


class FlatFunctional(nn.Module):
    """Submodule-free model (pure functional forward, no ``nn.Module`` children)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run only free functions so the module tree is just the root ``self``."""

        return torch.relu(x) * 2 + 1


class NestedBlocks(nn.Module):
    """Two-plus-level module tree (``block.0`` / ``block.1`` are depth 2)."""

    def __init__(self) -> None:
        """Initialize a top-level linear plus a nested Sequential block."""

        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route through the top-level linear then the nested block."""

        return self.block(self.l1(x))


def _module_tree(trace: tl.Trace) -> dict[str, tuple[str | None, tuple[str, ...]]]:
    """Return each module's (address_parent, address_children) address tree."""

    return {
        module.address: (module.address_parent, tuple(module.address_children))
        for module in trace.modules
    }


def _module_call_stacks(trace: tl.Trace) -> dict[str, list[str]]:
    """Return each module call's reconstructed ``module_call_stack``."""

    return {
        module_call.call_label: list(module_call.module_call_stack)
        for module_call in trace.module_calls
    }


def _structure(trace: tl.Trace) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    """Return a payload-independent structural summary for a Trace."""

    return [
        (
            op.layer_type,
            op.layer_label,
            tuple(op.parents),
            tuple(op.children),
        )
        for op in trace.layer_list
    ]


def test_recording_and_trace_are_captured_run_activation_lookup_siblings() -> None:
    """Recording and Trace subclass CapturedRun and satisfy ActivationLookup."""

    model = ConvReluAdd()
    x = torch.randn(1, 1, 4, 4)

    recording = tl.record(model, x, save=tl.func("conv2d"), random_seed=17)
    trace = tl.trace(model, x, save=tl.func("conv2d"), random_seed=17)

    assert issubclass(type(recording), tl.CapturedRun)
    assert issubclass(type(trace), tl.CapturedRun)
    assert not issubclass(type(trace), type(recording))
    assert isinstance(recording, tl.ActivationLookup)
    assert isinstance(trace, tl.ActivationLookup)


def test_recording_to_trace_matches_trace_structure_and_unsaved_out_fails() -> None:
    """record(save=...).to_trace() cooks structure and rejects unsaved payload reads."""

    model = ConvReluAdd()
    x = torch.randn(1, 1, 4, 4)

    recording = tl.record(model, x, save=tl.func("conv2d"), random_seed=23)
    cooked = recording.to_trace()
    full = tl.trace(model, x, random_seed=23)

    assert _structure(cooked) == _structure(full)
    saved = [op for op in cooked.layer_list if op.has_saved_activation]
    assert saved
    assert {op.layer_type for op in saved} == {"conv2d"}

    unsaved = next(op for op in cooked.layer_list if op.layer_type == "relu")
    with pytest.raises(ValueError, match="no saved payload.*save="):
        _ = unsaved.out


def test_recording_to_trace_backfills_timing_and_input_layers() -> None:
    """record().to_trace() seeds live-dispatch bookkeeping the replay path skips.

    Regression test for the ``Recording.to_trace()`` bridge: unlike every real
    capture backend, replaying captured events into a fresh ``Trace`` used to
    skip the live-dispatch setup that stamps ``capture_start_time`` and
    back-fills ``input_layers`` (only ``output_layers`` was handled). That left
    ``cleanup_duration`` computing as ``time.time() - 0 - 0 - 0`` (a
    multi-decade-off ``Duration``) and made ``input_layers`` empty even though
    individual ``Op`` entries still carried ``is_input=True`` -- an
    unconditional failure of the special_layer_lists metadata invariant for
    every ``record().to_trace()`` output.

    Checks the two specific invariant groups this fix originally targeted
    (``special_layer_lists`` for input_layers<->is_input consistency,
    ``trace_self_consistency`` for the timing/output_layers checks). The
    once-separate ``graph_ordering`` (raw-index) and ``module_hierarchy``
    (module-address-tree) gaps that used to keep the FULL
    ``check_metadata_invariants()`` suite red for ``record().to_trace()`` are
    now closed too; the full-chain guarantee is exercised by
    ``test_recording_to_trace_passes_full_metadata_invariants`` below.
    """

    model = ConvReluAdd()
    x = torch.randn(1, 1, 4, 4)

    before = time.time()
    recording = tl.record(model, x, save=tl.func("conv2d"), random_seed=31)
    cooked = recording.to_trace()
    after = time.time()

    # capture_start_time must be a real, in-window wall-clock timestamp, not
    # the dataclass default of 0 (epoch).
    assert before <= cooked.capture_start_time <= after

    # cleanup_duration must be a small, real duration -- not the
    # `time.time() - 0 - 0 - 0` garbage value (effectively "now", i.e. ~56
    # years for a 2026 run) the missing back-fill used to produce.
    assert 0 <= cooked.cleanup_duration < 60.0

    # input_layers must be symmetrically back-filled just like output_layers
    # already was, and must match the set of Ops actually flagged is_input.
    assert cooked.input_layers
    expected_inputs = {op.layer_label for op in cooked.layer_list if op.is_input}
    assert set(cooked.input_layers) == expected_inputs

    # The two invariant groups this fix targets must pass cleanly.
    _check_trace_self_consistency(cooked)
    _check_special_layer_lists(cooked)


@pytest.mark.parametrize(
    "model_factory",
    [FlatFunctional, NestedBlocks],
    ids=["flat_submodule_free", "nested_two_levels"],
)
def test_recording_to_trace_passes_full_metadata_invariants(model_factory) -> None:
    """record().to_trace() passes the FULL invariant chain for both shapes.

    Round-6 certification found two independent producer gaps that each left a
    ``record(...).to_trace()`` Trace failing ``check_metadata_invariants()`` --
    but the round-6 regression test only exercised two named invariant groups,
    so both slipped through:

    * ``graph_ordering`` -- ``Recording.to_trace()`` never seeded
      ``trace._layer_counter`` from the replayed event stream, so postprocess's
      synthetic output node was stamped ``raw_index=1``, colliding with
      ``input_1`` (fails even for a flat, submodule-free model).
    * ``module_hierarchy`` -- the fastlog recorder dropped the real
      ``ModulePrepEvent``s (root ``address_children`` came back empty) and emitted
      no module-call-stack metadata (every ``ModuleCall.module_call_stack`` was
      ``[]``), failing for any model with a submodule.

    This asserts the WHOLE ``check_metadata_invariants()`` chain -- not a hand-
    picked subset -- passes for both a flat model and a >=2-level nested model,
    the exact mistake (checking only named invariants) that let round 6 ship
    these gaps.
    """

    model = model_factory().eval()
    x = torch.randn(2, 4)

    cooked = tl.record(model, x, save=tl.func("relu"), random_seed=41).to_trace()

    # FULL chain -- must not raise MetadataInvariantError for any contract.
    check_metadata_invariants(cooked)

    # graph_ordering specifics: raw_index unique AND monotonically increasing.
    raw_indices = [op.raw_index for op in cooked.layer_list]
    assert len(raw_indices) == len(set(raw_indices)), "raw_index values must be unique"
    assert raw_indices == sorted(raw_indices), "raw_index must be monotonically increasing"


@pytest.mark.parametrize(
    "model_factory",
    [FlatFunctional, NestedBlocks],
    ids=["flat_submodule_free", "nested_two_levels"],
)
def test_recording_to_trace_module_tree_matches_exhaustive(model_factory) -> None:
    """to_trace() rebuilds the SAME module address tree and call stacks as tl.trace().

    The module-hierarchy fix must reconstruct the module tree identically to a
    live exhaustive capture, not merely "pass the invariant". Compares the full
    ``(address_parent, address_children)`` address tree and every
    ``ModuleCall.module_call_stack`` against the exhaustive ``tl.trace`` of the
    same model.
    """

    x = torch.randn(2, 4)

    cooked = tl.record(model_factory().eval(), x, save=tl.func("relu"), random_seed=41).to_trace()
    exhaustive = tl.trace(model_factory().eval(), x, random_seed=41)

    assert _module_tree(cooked) == _module_tree(exhaustive)
    assert _module_call_stacks(cooked) == _module_call_stacks(exhaustive)


def test_record_save_matches_deprecated_keep_op_alias() -> None:
    """record(save=...) and deprecated record(keep_op=...) retain the same ops."""

    model = ConvReluAdd()
    x = torch.randn(1, 1, 4, 4)

    save_recording = tl.record(model, x, save=tl.func("conv2d"), random_seed=29)
    with pytest.warns(DeprecationWarning, match="keep_op"):
        alias_recording = tl.record(model, x, keep_op=tl.func("conv2d"), random_seed=29)

    assert [record.ctx.raw_label for record in save_recording] == [
        record.ctx.raw_label for record in alias_recording
    ]
