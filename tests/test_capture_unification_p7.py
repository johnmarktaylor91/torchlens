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


class DuplicateTensorOutput(nn.Module):
    """Model whose ``forward()`` returns the SAME tensor object more than once."""

    def __init__(self) -> None:
        """Initialize a small linear-relu stack."""

        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the same computed tensor twice (dual-head / siamese-style)."""

        y = self.relu(self.l1(x))
        return (y, y)


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


# ---------------------------------------------------------------------------
# Round-8 (cert8-g1): Recording.to_trace() finalization COMPLETENESS.
#
# Rounds 6-7 found SIX separate producer gaps in Recording.to_trace() because
# it never mirrored the full set of finalization steps tl.trace() performs.
# The two round-7 gaps closed here:
#   1. buffer_layers + internal_source_ops (2 of the 5 special-layer lists the
#      special_layer_lists invariant checks) were never backfilled -- empty
#      buffer_layers silently disabled postprocess Step 6 _fix_buffer_layers
#      (no buffer_pass, empty Module.buffer_layers) for every buffer-bearing or
#      internal-source model captured via record(...).to_trace().
#   2. halted recordings yielded .halted == False and crashed with "No output
#      layers found" -- the halt-frontier finalization (_finalize_halted_trace)
#      was never mirrored.
# The all-category acceptance model below exercises EVERY category in one shape.
# ---------------------------------------------------------------------------


class _Inner(nn.Module):
    """Depth-2 leaf submodule (linear + relu)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class _Mid(nn.Module):
    """Depth-1 submodule holding a nested submodule plus a buffer-bearing BN."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _Inner()
        self.bn = nn.BatchNorm1d(4)  # running_mean/running_var buffers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.inner(x))


class AllCategory(nn.Module):
    """One model exercising every special-layer-list category at once.

    * BatchNorm buffers (running stats)   -> buffer_layers / is_buffer
    * a registered constant buffer         -> buffer_layers / is_buffer
    * an internal-source op (torch.zeros)  -> internal_source_ops
    * nested submodules (>= 2 levels)      -> module_hierarchy
    * a recurrent/repeated-call submodule  -> multi-pass labeling
    """

    def __init__(self) -> None:
        super().__init__()
        self.mid = _Mid()  # 2+ levels of nesting (mid.inner.fc)
        self.shared = nn.Linear(4, 4)  # invoked repeatedly (recurrent)
        self.register_buffer("bias_const", torch.ones(4))  # registered constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mid(x)
        z = torch.zeros(x.shape)  # internal-source op (no tensor ancestor)
        h = self.shared(h)  # recurrent call 1
        h = self.shared(h)  # recurrent call 2
        return h + z + self.bias_const


class _HaltNested(nn.Module):
    """Nested model whose relu is a natural mid-forward halt frontier."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.l2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.block(self.l1(x)))


def _buffer_read_addresses(trace: tl.Trace) -> list[str]:
    """Return sorted addresses of buffer *read* (source) layers only.

    Predicate/fastlog capture deliberately omits ``install_buffer_write_tracker``
    (``backends/torch/model_prep.py`` installs it for ``capture_mode ==
    'exhaustive'`` only), so buffer WRITE-back sinks/duplicates that exhaustive
    ``tl.trace()`` records are legitimately absent from a ``record().to_trace()``
    reconstruction. Buffer READS are captured by both paths and must match.
    """

    return sorted(
        trace[label].address for label in trace.buffer_layers if not trace[label].is_internal_sink
    )


def test_recording_to_trace_backfills_all_special_layer_lists() -> None:
    """to_trace() backfills EVERY special-layer list, not just input/output.

    Round-7 BLOCKER: ``Recording.to_trace()`` backfilled ``input_layers`` and
    ``output_layers`` but not ``buffer_layers`` or ``internal_source_ops`` (2 of
    the 5 ``_SPECIAL_LIST_FLAG_PAIRS`` the ``special_layer_lists`` invariant
    checks). Empty ``buffer_layers`` silently disabled postprocess Step 6
    ``_fix_buffer_layers`` -- ``buffer_pass`` stayed ``None``, ``buffer_source``
    links never formed, and every ``Module.buffer_layers`` came back empty for
    buffer-bearing modules -- silent corruption for any BatchNorm/registered-
    constant model, not merely a raised invariant.

    Asserts the FULL invariant chain plus the downstream postprocess effects the
    backfill re-enables, and matches the reconstruction against an exhaustive
    ``tl.trace()`` for module tree, input/output layers, and buffer reads.
    """

    model = AllCategory().eval()
    x = torch.randn(2, 4)

    cooked = tl.record(model, x, save=tl.func("relu"), random_seed=11).to_trace()
    exhaustive = tl.trace(AllCategory().eval(), x, random_seed=11)

    # FULL invariant chain -- both must be clean.
    check_metadata_invariants(cooked)
    check_metadata_invariants(exhaustive)

    # All FIVE special lists must be bidirectionally consistent with per-op
    # flags (the whole point of _check_special_layer_lists) -- exercised via the
    # full chain above, asserted explicitly here for a second, focused datapoint.
    _check_special_layer_lists(cooked)

    # buffer_layers must be populated (not silently empty) and Step 6
    # _fix_buffer_layers must have RUN -- proven by a real buffer_pass number.
    assert cooked.buffer_layers, "buffer_layers must be backfilled, not empty"
    for label in cooked.buffer_layers:
        assert cooked[label].is_buffer
        assert cooked[label].buffer_pass is not None, "Step 6 must assign buffer_pass"

    # internal_source_ops must be populated (the torch.zeros op has no ancestor).
    assert cooked.internal_source_ops, "internal_source_ops must be backfilled"
    zeros_sources = [op for op in cooked.internal_source_ops if op.startswith("zeros")]
    assert zeros_sources, "the torch.zeros internal-source op must be listed"

    # Module.buffer_layers must come back NON-empty for the buffer-bearing
    # module (mid.bn) -- the public field that silently emptied before the fix.
    bn_modules = [m for m in cooked.modules if m.address == "mid.bn"]
    assert bn_modules and bn_modules[0].buffer_layers, "Module.buffer_layers must be non-empty"

    # Match exhaustive tl.trace() EXACTLY where predicate capture is complete:
    # module address tree, input/output layers, and buffer READ addresses.
    assert _module_tree(cooked) == _module_tree(exhaustive)
    assert list(cooked.input_layers) == list(exhaustive.input_layers)
    assert list(cooked.output_layers) == list(exhaustive.output_layers)
    assert _buffer_read_addresses(cooked) == _buffer_read_addresses(exhaustive)


def test_recording_to_trace_halted_produces_valid_halted_trace() -> None:
    """A halted record().to_trace() yields a VALID halted Trace matching tl.trace().

    Round-7 MAJOR: ``record(model, x, halt=...).to_trace()`` returned a Trace
    with ``.halted == False`` (silently lying about the capture) and then crashed
    in ``check_metadata_invariants`` with "No output layers found", because the
    fastlog bridge never mirrored ``_finalize_halted_trace``'s frontier-output
    recovery. The bridge now recovers the halt frontier and binds it as the
    output parent, so the reconstruction matches an exhaustive halted capture.
    """

    x = torch.randn(2, 4)
    halt = lambda ctx: getattr(ctx, "func_name", "") == "relu"  # noqa: E731

    exhaustive = tl.trace(_HaltNested().eval(), x, halt=halt)
    cooked = tl.record(
        _HaltNested().eval(), x, save=tl.func("relu"), halt=halt, random_seed=9
    ).to_trace()

    # .halted must be TRUE (not silently False) and match exhaustive exactly.
    assert cooked.halted is True
    assert cooked.halted == exhaustive.halted
    assert cooked.halt_reason == exhaustive.halt_reason
    assert cooked.halt_frontier == exhaustive.halt_frontier
    assert list(cooked.output_layers) == list(exhaustive.output_layers)

    # And it must pass the FULL invariant chain rather than crashing.
    check_metadata_invariants(cooked)


def test_recording_to_trace_halted_without_payload_rejected() -> None:
    """A halted recording with no retained payload is rejected with a clear error.

    Full halt support requires a frontier tensor. Mirroring the exhaustive path's
    "could not identify a tensor frontier" contract, a halted recording that
    saved no raw activation (nothing to bind as the output node) must raise a
    clear, actionable ``RuntimeError`` -- not silently mislabel or crash opaquely.
    """

    x = torch.randn(2, 4)
    recording = tl.record(
        _HaltNested().eval(),
        x,
        save=tl.func("this_function_name_never_matches"),
        halt=lambda ctx: getattr(ctx, "func_name", "") == "relu",
        random_seed=9,
    )
    assert recording.halted is True
    assert len(recording.records) == 0
    with pytest.raises(RuntimeError, match="halted Recording that retained no raw activation"):
        recording.to_trace()


class _BNBeforeHalt(nn.Module):
    """Linear -> BatchNorm (buffer op) -> Linear -> Linear, halt on 2nd linear."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.l2 = nn.Linear(4, 4)
        self.l3 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l3(self.l2(self.bn(self.l1(x))))


def _second_linear_halt(ctx: object) -> bool:
    """Halt on the second ``linear`` call (a buffer op precedes it)."""

    return getattr(ctx, "func_name", "") == "linear" and getattr(ctx, "label", "").startswith(
        "linear_2"
    )


def test_recording_to_trace_reuse_does_not_corrupt_frozen_recording() -> None:
    """cert8 BLOCKER-1: to_trace() must not drain the frozen Recording's own events.

    ``Recording`` is ``frozen=True`` and its ``_capture_events`` back the public
    read-only ``records`` / ``recording_trace`` accessors. ``to_trace()`` used to
    ALIAS ``self._capture_events`` into the new Trace, whose ``_postprocess()``
    destructively drains it. Consequences: a second ``to_trace()`` crashed, and
    ``recording_trace`` / ``records`` first read AFTER ``to_trace()`` memoized an
    empty/wrong answer. ``to_trace()`` now hands postprocess a structural copy.
    """

    x = torch.randn(2, 4)

    # (a) Two consecutive to_trace() calls both succeed and are equivalent.
    rec = tl.record(NestedBlocks().eval(), x, save=tl.func("relu"))
    trace1 = rec.to_trace()
    trace2 = rec.to_trace()
    assert [op.layer_label for op in trace1.layer_list] == [
        op.layer_label for op in trace2.layer_list
    ]
    assert trace1.num_ops == trace2.num_ops
    check_metadata_invariants(trace1)
    check_metadata_invariants(trace2)

    # (b) recording_trace read AFTER to_trace() still returns the full stream,
    #     matching a control recording read BEFORE to_trace().
    rec_after = tl.record(NestedBlocks().eval(), x, save=tl.func("relu"))
    _ = rec_after.to_trace()
    contexts_after = len(rec_after.recording_trace.contexts)

    rec_before = tl.record(NestedBlocks().eval(), x, save=tl.func("relu"))
    contexts_before = len(rec_before.recording_trace.contexts)
    _ = rec_before.to_trace()

    assert contexts_after == contexts_before
    assert contexts_after > 0

    # (c) records accessor is likewise intact after to_trace().
    rec_records = tl.record(NestedBlocks().eval(), x, save=tl.func("relu"))
    _ = rec_records.to_trace()
    assert rec_records.n_records > 0


def test_recording_to_trace_reuse_survives_output_tensor_label_undecoration() -> None:
    """cert9 BLOCKER-1: repeat to_trace() must survive Step 12 undecorating shared outputs.

    ``copy_for_replay()`` (cert8 BLOCKER-1 fix) protects the frozen Recording's
    ``CaptureEvents``/``LiveIndex`` containers from ``_postprocess()``'s
    destructive draining, but structurally cannot protect a sibling side
    channel: the model's real output tensor OBJECTS, which ``to_trace()``
    retains and hands to ``_postprocess()`` BY REFERENCE across every call
    (never copied, for cost reasons). Postprocess Step 12
    (``_undecorate_all_saved_tensors``) strips the private ``._tl`` metadata --
    including ``label_raw`` -- off those same tensor objects. Whenever the
    output-tensor count doesn't equal the output-label count (e.g. a model
    returning the same tensor object twice, ``return (y, y)``),
    ``graph_traversal.py``'s ``_resolve_output_parent_labels`` falls off its
    fast path and re-reads that now-cleared ``label_raw`` on the SECOND
    ``to_trace()`` call, raising "could not attribute a model output tensor to
    any traced op". ``to_trace()`` must snapshot and restore each retained
    output tensor's label across ``_postprocess()`` so repeat calls stay safe
    regardless of output-tensor/output-label count parity.
    """

    x = torch.randn(2, 4)
    rec = tl.record(DuplicateTensorOutput().eval(), x, save=tl.func("relu"))

    trace1 = rec.to_trace()
    trace2 = rec.to_trace()

    assert trace1.output_layers == trace2.output_layers
    assert len(trace1.output_layers) == 2
    assert [op.layer_label for op in trace1.layer_list] == [
        op.layer_label for op in trace2.layer_list
    ]
    check_metadata_invariants(trace1)
    check_metadata_invariants(trace2)

    # A third call keeps working too -- the restore isn't a one-shot patch.
    trace3 = rec.to_trace()
    assert trace3.output_layers == trace1.output_layers
    check_metadata_invariants(trace3)


def test_recording_to_trace_halt_labels_match_exhaustive_across_buffer() -> None:
    """cert8 MAJOR-1: halt provenance matches exhaustive when a buffer op precedes halt.

    ``halt_reason`` / ``halt_frontier`` used to store raw capture-time labels,
    which diverge between capture modes because exhaustive counts buffer ops that
    predicate mode skips. Both paths now remap them to FINAL labels in postprocess
    Step 10. For an eval-mode BatchNorm model (buffer READS only, matched node
    counts) the halt labels match exhaustive ``tl.trace(halt=...)`` exactly.
    """

    x = torch.randn(3, 4)

    exhaustive = tl.trace(_BNBeforeHalt().eval(), x, halt=_second_linear_halt)
    cooked = tl.record(
        _BNBeforeHalt().eval(),
        x,
        save=tl.func("linear"),
        halt=_second_linear_halt,
        random_seed=11,
    ).to_trace()

    assert cooked.halted is True
    # Final labels (not raw) -- and they match exhaustive exactly.
    assert cooked.halt_reason == exhaustive.halt_reason
    assert cooked.halt_frontier == exhaustive.halt_frontier
    assert not cooked.halt_reason.endswith("_raw")
    check_metadata_invariants(cooked)
    check_metadata_invariants(exhaustive)


def test_recording_to_trace_buffer_writes_are_honestly_unavailable() -> None:
    """cert8 MAJOR (buffers): cooked buffer-write fields are honest + documented.

    Predicate capture does not track buffer WRITES, so a train-mode BatchNorm
    model's mutated buffers must not masquerade as tracked writes. The cooked
    Trace reports no buffer-write ops (``buffer_write_kind`` stays ``None``) and
    still passes the full invariant chain; the limitation is documented on
    ``Recording.to_trace`` and ``Trace.buffer_write_ops`` so ``None`` is not
    silently misread as "read-only".
    """

    x = torch.randn(4, 4)
    recording = tl.record(_BNBeforeHalt().train(), x, save=tl.func("linear"))
    cooked = recording.to_trace()

    # Honest: buffer reads are present, but no buffer WRITE is claimed.
    assert cooked.buffer_write_ops == []
    for label in cooked.buffer_layers:
        assert cooked[label].buffer_write_kind is None

    # Not corruption: the full invariant chain still passes.
    check_metadata_invariants(cooked)

    # Documented: the limitation is spelled out where a user would look.
    assert "buffer-write" in type(recording).to_trace.__doc__
    assert "not tracked" in type(cooked).buffer_write_ops.fget.__doc__


def test_recording_to_trace_backfills_random_seed() -> None:
    """cert8 MINOR-1: to_trace() backfills the real capture seed onto the Trace.

    ``random_seed`` is a serialized (KEEP) field; it used to stay ``None`` after
    ``to_trace()``. The seed genuinely used by the predicate-mode primary pass is
    now copied from the runtime trace, so seed provenance survives cooking.
    """

    x = torch.randn(2, 4)
    cooked = tl.record(NestedBlocks().eval(), x, save=tl.func("relu"), random_seed=4242).to_trace()
    assert cooked.random_seed == 4242
