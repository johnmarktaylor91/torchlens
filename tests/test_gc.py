"""GC and memory leak tests for TorchLens.

Verifies that Trace, Op, Param, and model parameters
are garbage-collectible after use / cleanup.

These are marked ``smoke``: a lifetime regression is invisible to call-count,
``tracemalloc``, and wall-clock gates, so this file is the only per-step gate
that can see one. It is fast (~5 s) and it has caught the same module-global
cache root twice now.
"""

import gc
import tracemalloc
import weakref

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import trace as trace_fn
from torchlens._io import FieldPolicy
from torchlens.data_classes._trace_accessors import _TRACE_MODULE_CALL_ACCESSOR_ATTR
from torchlens.data_classes.trace import Trace

# Marks are PER-TEST: four session-state-scaling gc tests are re-tiered
# `heavy` below (charged 9.4-19.3s min(wall, cpu) in the merged full
# not-slow session -- their explicit gc.collect() pays for the whole
# session's accumulated cycles -- while running 0.6-1.5s isolated). Marks
# are additive, so a file-level smoke pytestmark would drag them back
# into the smoke tier; the remaining leak gates stay smoke-marked.


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc(x)


class _TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTraceGC:
    @pytest.mark.smoke
    def test_trace_gc_without_cleanup(self):
        """del trace; gc.collect() should release the Trace."""
        model = _SimpleLinear()
        trace = tl.trace(model, torch.randn(1, 5))
        ref = weakref.ref(trace)
        del trace
        gc.collect()
        assert ref() is None

    @pytest.mark.smoke
    def test_trace_gc_with_cleanup(self):
        """cleanup() + del + gc.collect() should release the Trace."""
        model = _SimpleLinear()
        trace = tl.trace(model, torch.randn(1, 5))
        ref = weakref.ref(trace)
        trace.cleanup()
        del trace
        gc.collect()
        assert ref() is None

    @pytest.mark.smoke
    def test_model_params_not_pinned_after_cleanup(self):
        """After cleanup + del trace, model params should be GC-able."""
        model = _SimpleLinear()
        param_ref = weakref.ref(list(model.parameters())[0])
        trace = tl.trace(model, torch.randn(1, 5))
        trace.cleanup()
        del trace
        del model
        gc.collect()
        assert param_ref() is None

    @pytest.mark.smoke
    def test_fast_live_hooks_finalize_when_trace_is_deleted(self):
        """Deleting a fast Trace removes its hooks and leaves parameters collectible."""

        model = _TwoLayerNet().eval()
        trace = tl.trace(model, torch.randn(1, 5), save=tl.module("fc1"))
        trace.run(inputs=torch.randn(1, 5), fast=True)
        trace_ref = weakref.ref(trace)
        param_ref = weakref.ref(next(model.parameters()))
        assert model.fc1._forward_hooks

        del trace
        gc.collect()

        assert trace_ref() is None
        assert not model.fc1._forward_hooks
        del model
        gc.collect()
        assert param_ref() is None

    @pytest.mark.smoke
    def test_model_gc_after_release_param_refs(self):
        """release_param_refs() then del model -> model GC'd while log alive."""
        model = _TwoLayerNet()
        model_ref = weakref.ref(model)
        trace = tl.trace(model, torch.randn(1, 5))
        trace.release_param_refs()
        del model
        gc.collect()
        assert model_ref() is None
        # trace is still usable
        assert len(trace) > 0
        trace.cleanup()

    @pytest.mark.heavy
    def test_no_memory_growth_across_sessions(self):
        """5x trace + del should not leak memory."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        # Warm up
        ml = trace_fn(model, x)
        ml.cleanup()
        del ml
        gc.collect()

        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()

        for _ in range(5):
            ml = trace_fn(model, x)
            ml.cleanup()
            del ml
            gc.collect()

        after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare: filter to torchlens allocations
        stats = after.compare_to(baseline, "lineno")
        tl_growth = sum(s.size_diff for s in stats if "torchlens" in str(s.traceback))
        # Allow up to 256KB of noise (caches, interned strings, etc.)
        assert tl_growth < 256 * 1024, f"Memory grew by {tl_growth} bytes across 5 sessions"

    @pytest.mark.heavy
    def test_save_new_outs_no_leak(self):
        """5x save_new_outs should not leak memory."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        # Two-pass path: exhaustive first, then fast via save_new_outs
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=None))

        # Warm up
        trace.save_new_outs(model, torch.randn(1, 5), layers_to_save="all")
        gc.collect()

        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()

        for _ in range(5):
            trace.save_new_outs(model, torch.randn(1, 5), layers_to_save="all")
            gc.collect()

        after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = after.compare_to(baseline, "lineno")
        tl_growth = sum(s.size_diff for s in stats if "torchlens" in str(s.traceback))
        assert tl_growth < 256 * 1024, f"Memory grew by {tl_growth} bytes across 5 save_new_outs"
        trace.cleanup()

    @pytest.mark.smoke
    def test_cleanup_breaks_param_ref(self):
        """After cleanup, all Param._param_ref should be None."""
        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        param_logs = list(trace.param_logs)
        trace.cleanup()
        for pl in param_logs:
            assert pl._param_ref is None

    @pytest.mark.smoke
    def test_release_param_refs_preserves_grad_metadata(self):
        """backward(), release_param_refs(), verify grad info is cached."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
        # Run backward to populate grads
        out = model(x)
        out.sum().backward()
        # Access grad metadata to cache it
        for pl in trace.param_logs:
            pl._check_param_grad()
        # Now release
        trace.release_param_refs()
        # Grad metadata should still be accessible
        has_any_grad = False
        for pl in trace.param_logs:
            assert pl._param_ref is None
            if pl._has_grad:
                has_any_grad = True
                assert pl._grad_shape is not None
                assert pl._grad_dtype is not None
                assert pl._grad_memory > 0
        assert has_any_grad, "Expected at least one param to have grad metadata cached"
        trace.cleanup()

    @pytest.mark.smoke
    def test_transient_data_cleared(self):
        """Verify module build scratch is removed after postprocess."""
        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        assert "_build_state" not in trace.__dict__
        assert "_raw_graph_ws" not in trace.__dict__
        assert "_module_capture_ws" not in trace.__dict__
        assert "_wrapper_runtime_ws" not in trace.__dict__
        assert not hasattr(trace, "_module_build_data")
        assert not hasattr(trace, "_module_metadata")
        assert not hasattr(trace, "_module_forward_args")
        trace.cleanup()

    @pytest.mark.smoke
    def test_raw_layer_dict_cleared_after_cleanup(self):
        """Verify raw layer scratch is absent after postprocess and cleanup."""
        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        assert "_build_state" not in trace.__dict__
        assert "_raw_graph_ws" not in trace.__dict__
        assert "_module_capture_ws" not in trace.__dict__
        assert "_wrapper_runtime_ws" not in trace.__dict__
        assert not hasattr(trace, "_raw_layer_dict")
        trace.cleanup()
        assert not hasattr(trace, "_raw_layer_dict")

    @pytest.mark.smoke
    def test_module_calls_accessor_is_cached_on_the_instance(self):
        """The flattened ModuleCall accessor memo lives on the Trace, not a global.

        A module-global cache keyed by the Trace cannot hold this value: the
        accessor holds ModuleCalls and ``ModuleCall._source_trace`` keeps a
        strong reference back to the Trace, so even a ``WeakKeyDictionary``
        value would reach its own key and pin every Trace forever.
        """

        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        accessor = trace.module_calls

        assert trace.__dict__[_TRACE_MODULE_CALL_ACCESSOR_ATTR] is accessor
        assert trace.module_calls is accessor
        assert Trace.PORTABLE_STATE_SPEC[_TRACE_MODULE_CALL_ACCESSOR_ATTR] is FieldPolicy.DROP

    @pytest.mark.heavy
    def test_populated_module_call_accessor_does_not_pin_trace(self):
        """Reading ``module_calls`` must not make the Trace immortal.

        Every capture populates this accessor internally through the
        saved-summary refresh, so a root here leaks on the default
        ``tl.trace()`` path with no user API call at all.
        """

        model = _TwoLayerNet()
        refs = []
        for _ in range(3):
            trace = tl.trace(model, torch.randn(1, 5))
            assert len(trace.module_calls) > 0
            refs.append(weakref.ref(trace))
            del trace
            gc.collect()
        assert [ref() for ref in refs] == [None, None, None]

    @pytest.mark.smoke
    def test_held_module_call_still_keeps_its_trace_alive(self):
        """The intentional ModuleCall -> Trace ownership edge survives the fix."""

        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        module_call = trace.module_calls[0]
        ref = weakref.ref(trace)

        del trace
        gc.collect()
        assert ref() is not None, "holding a ModuleCall must keep its Trace alive"
        assert module_call.trace is ref()

        del module_call
        gc.collect()
        assert ref() is None

    @pytest.mark.smoke
    def test_trace_reclamation_is_cyclic_gc_not_prompt_refcount(self):
        """Traces are reclaimed by the CYCLIC collector, and that is the contract.

        Deliberately pinned, because the docs claimed the opposite. Every capture
        populates ``module_calls`` through the saved-summary refresh, and
        ``ModuleCall._source_trace`` is a STRONG owner edge (kept on purpose --
        see ``test_held_module_call_still_keeps_its_trace_alive``, and the module
        accessor cache above reasons from it). That closes
        Trace -> accessor -> ModuleCall -> Trace, so a dropped Trace and its saved
        activations are freed at the next ``gc.collect()``, NOT at zero refcount.

        Every other assertion in this file calls ``gc.collect()`` first, so
        nothing here observed the difference; this arm runs with the collector
        DISABLED so the real behavior is stated and cannot drift silently.
        """

        model = _TwoLayerNet()
        gc.collect()
        gc.disable()
        try:
            trace = tl.trace(
                model, torch.randn(1, 5), capture=tl.options.CaptureOptions(layers_to_save="all")
            )
            activation = trace["relu_1_2"].out
            trace_ref = weakref.ref(trace)
            activation_ref = weakref.ref(activation)

            del trace, activation
            assert trace_ref() is not None, (
                "Trace became refcount-reclaimable: the ModuleCall owner edge or "
                "the saved-summary refresh changed, so the documented lifetime "
                "contract needs updating (and this test with it)"
            )
            assert activation_ref() is not None
        finally:
            gc.enable()

        gc.collect()
        assert trace_ref() is None, "Trace survived a cyclic collection"
        assert activation_ref() is None, "saved activation survived a cyclic collection"

    @pytest.mark.smoke
    def test_last_captures_model_class_is_not_pinned_after_the_epilogue(self):
        """A dynamically created module class dies with its last capture.

        ``_module_class_metadata_cache`` is a plain dict keyed by the module
        CLASS (and stores the class again in its value), and it was only ever
        cleared at the START of the next capture. A process whose final capture
        used a generated / function-local class therefore kept that class, its
        code objects and its closure alive for the whole process lifetime.
        """

        def build_class():
            """Return a fresh module class defined in this call's scope."""

            class _Generated(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(4, 3)

                def forward(self, x):
                    return torch.relu(self.fc(x))

            return _Generated

        generated = build_class()
        class_ref = weakref.ref(generated)
        model = generated()
        trace = tl.trace(model, torch.randn(2, 4))

        del trace, model, generated
        gc.collect()

        assert class_ref() is None, (
            "the last capture's module class is still pinned after the capture "
            "epilogue (class-metadata cache not released)"
        )

    @pytest.mark.smoke
    def test_failed_capture_registry_does_not_pin_a_dead_exception(self):
        """The partial-recovery fallback table holds its exception weakly.

        The table is keyed by ``id(exception)`` and was capped at 128 ENTRIES
        with no byte bound and no time eviction — but each retained exception
        keeps its ``__traceback__``, and that pins every frame local (the model,
        the inputs, the partial outputs). Recovery only ever reaches an entry
        through ``from_failed_capture(exc)``, which requires the caller to hold
        that exception, so once it dies the entry is unreachable garbage.
        """

        from torchlens import partial as partial_module

        class _FrameLocalMarker:
            """Stands in for the model/inputs a traceback frame keeps alive."""

        class _RejectsAttachment(Exception):
            """The realistic shape that reaches this fallback at all.

            Builtin exceptions and ``__slots__`` subclasses both ACCEPT
            ``exc.partial_log = ...`` (BaseException always carries a dict), so
            the registry is only ever reached by types that refuse attribute
            assignment outright — which are user-defined and weak-referenceable.
            """

            def __setattr__(self, name, value):
                raise AttributeError("read-only exception")

        def raise_holding_a_local():
            """Raise an exception whose traceback frame holds a marker object."""

            marker = _FrameLocalMarker()
            marker_ref = weakref.ref(marker)
            try:
                raise _RejectsAttachment("registry retention probe")
            except _RejectsAttachment as error:
                return error, marker_ref

        trace = tl.trace(_SimpleLinear(), torch.randn(1, 5))
        exception, marker_ref = raise_holding_a_local()
        partial_log = partial_module.PartialTrace(trace=trace, original_exception=exception)
        key = id(exception)
        partial_module._register_failed_capture(exception, partial_log)

        # Recovery works for as long as the caller holds the exception, and
        # repeated lookups hand back the same wrapper.
        recovered = partial_module.from_failed_capture(exception)
        assert recovered.original_exception is exception
        assert recovered.trace is trace
        assert partial_module.from_failed_capture(exception) is recovered
        assert key in partial_module._FAILED_CAPTURE_REGISTRY

        del exception, partial_log, recovered
        gc.collect()

        assert key not in partial_module._FAILED_CAPTURE_REGISTRY, (
            "the registry kept an entry for a collected exception"
        )
        assert marker_ref() is None, (
            "the registry pinned the dead exception's traceback frame locals"
        )

    @pytest.mark.smoke
    def test_type_keyed_caches_do_not_pin_model_classes(self):
        """Per-type caches keyed on a model class must not outlive that class.

        ``_state._dir_cache`` and the validation deepcopy warn-once set are both
        keyed by ``type``. Strong keys made every captured model class immortal
        for the process, which matters exactly for the generated / notebook /
        function-local classes users actually feed a tracer.
        """

        from torchlens import _capture_state_helpers, _state

        def build_class():
            """Return a fresh model class defined in this call's scope."""

            class _TypeKeyed(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(4, 3)

                def forward(self, x):
                    return self.fc(x)

            return _TypeKeyed

        generated = build_class()
        class_ref = weakref.ref(generated)
        _capture_state_helpers._VALIDATION_DEEPCOPY_WARNING_TYPES.add(generated)
        _state._dir_cache[generated] = ["fc"]
        trace = tl.trace(generated(), torch.randn(2, 4))

        del trace, generated
        gc.collect()

        assert class_ref() is None, (
            "a type-keyed cache still pins the model class after it was dropped"
        )

    @pytest.mark.smoke
    def test_backward_trigger_registry_evicts_with_its_trace(self):
        """Dropping a backward-armed trace clears its grad-fn registry keys.

        The table maps ``id(grad_fn) -> weakref(trace)``. Entries for a trace
        dropped WITHOUT ``cleanup()`` used to linger until some later, unrelated
        backward happened to walk past that grad-fn id, so a process that
        discarded traces accreted dead keys indefinitely.
        """

        from torchlens.backends.torch import backward as backward_module

        registry = backward_module._BACKWARD_GRAD_FN_REGISTRY
        model = _TwoLayerNet()
        x = torch.randn(1, 5, requires_grad=True)

        trace = tl.trace(
            model,
            x,
            capture=tl.options.CaptureOptions(backward_ready=True),
            save_mode="reference",
        )
        armed_keys = {key for key, ref in registry.items() if ref() is trace}
        assert armed_keys, "capture registered no backward triggers to observe"

        del trace
        gc.collect()

        leaked = armed_keys & set(registry)
        assert not leaked, (
            f"{len(leaked)} backward-registry keys survived their trace "
            "(eviction still waits for an unrelated later backward)"
        )

    @pytest.mark.smoke
    def test_failed_capture_registry_falls_back_for_unweakrefable_exceptions(self):
        """A non-weak-referenceable exception still recovers, under the entry cap.

        Builtin exception instances cannot be weak-referenced. They never reach
        this table (they accept ``partial_log`` attachment), but a C-extension
        exception type could, so the fallback must keep working rather than
        losing recovery.
        """

        from torchlens import partial as partial_module

        trace = tl.trace(_SimpleLinear(), torch.randn(1, 5))
        exception = RuntimeError("unweakrefable probe")
        with pytest.raises(TypeError):
            weakref.ref(exception)  # the precondition this arm exists for
        partial_log = partial_module.PartialTrace(trace=trace, original_exception=exception)
        partial_module._register_failed_capture(exception, partial_log)
        try:
            assert partial_module.from_failed_capture(exception) is partial_log
            assert len(partial_module._FAILED_CAPTURE_REGISTRY) <= (
                partial_module._FAILED_CAPTURE_REGISTRY_LIMIT
            )
        finally:
            partial_module._FAILED_CAPTURE_REGISTRY.pop(id(exception), None)

    @pytest.mark.smoke
    def test_nonweakrefable_entry_never_pins_the_exception_graph(self):
        """A dropped locked+non-weakrefable exception must not pin its capture (R37).

        An ``Exception`` subclass declaring ``__slots__`` is non-weakrefable,
        and one that also refuses ``__setattr__`` rejects the ``partial_log``
        attachment, so it reaches the registry fallback. The stub design
        (fixwave-5 integration; supersedes the strong-retention fallback and
        its refcount sweep) NEVER retains the exception: its graph --
        traceback, frame locals, model, inputs -- frees the moment the
        caller drops it, with NO later registry interaction needed. The
        price, disclosed on the registry docstring, is that the partial
        TRACE stays pinned by the entry until the registry cap; this test
        pins that the registry entry is the ONLY thing retaining it.
        """

        from torchlens import partial as partial_module

        class _LockedSlots(Exception):
            __slots__ = ()

            def __setattr__(self, name, value):
                raise AttributeError("locked")

        with pytest.raises(TypeError):
            weakref.ref(_LockedSlots("x"))  # the precondition this arm exists for

        class _FrameLocalMarker:
            pass

        def raise_locked():
            marker = _FrameLocalMarker()
            marker_ref = weakref.ref(marker)
            try:
                raise _LockedSlots("stub retention probe")
            except _LockedSlots as error:
                return error, marker_ref

        trace = tl.trace(_SimpleLinear(), torch.randn(1, 5))
        exception, marker_ref = raise_locked()
        partial_log = partial_module.PartialTrace(trace=trace, original_exception=exception)
        key = id(exception)
        partial_module._register_failed_capture(exception, partial_log)
        del partial_log

        # Recovery works while the caller holds the exception, and the sweep
        # (which runs inside every lookup) never evicts a stub entry.
        recovered = partial_module.from_failed_capture(exception)
        assert recovered.trace is trace
        del recovered
        assert key in partial_module._FAILED_CAPTURE_REGISTRY

        # The exception graph frees IMMEDIATELY on drop -- no sweep, no
        # later registry interaction (stronger than the superseded strong
        # fallback, which kept it until the next registration or lookup).
        del exception
        gc.collect()
        assert marker_ref() is None, (
            "the stub entry pinned the dead exception's traceback frame locals"
        )

        # The trace pin is the registry entry ALONE (cap-bounded): popping
        # the entry must be the last strong reference standing.
        trace_ref = weakref.ref(trace)
        del trace
        gc.collect()
        assert trace_ref() is not None, "the cap-bounded registry entry should hold the trace"
        partial_module._FAILED_CAPTURE_REGISTRY.pop(key, None)
        gc.collect()
        assert trace_ref() is None, (
            "something besides the registry entry retained the partial trace"
        )

    @pytest.mark.smoke
    def test_live_run_after_model_collection_refuses_typed_and_names_the_weak_ref(self):
        """A collected source model refuses run() typed AND discloses why (R37).

        The refusal itself is correct by design (the trace holds its model
        weakly), but it used to be undocumented and its message named neither
        the weak reference nor a remedy -- the plainest documented idiom
        ``tl.trace(Model(), x)`` then failed gc-timing-dependently with no
        actionable explanation.
        """

        from torchlens.errors import RunCapabilityUnavailableError

        trace = tl.trace(_SimpleLinear(), torch.randn(1, 5))
        gc.collect()
        assert trace._source_model_ref() is None, "inline model should be collected"
        with pytest.raises(RunCapabilityUnavailableError) as excinfo:
            trace.run(inputs=torch.randn(1, 5))
        message = str(excinfo.value)
        assert "weakly" in message and "strong reference" in message, (
            "the collected-model refusal must disclose the weak-reference "
            f"dependency and its remedy; got: {message}"
        )
        with pytest.raises(RunCapabilityUnavailableError) as fast_excinfo:
            trace.run(inputs=torch.randn(1, 5), fast=True)
        assert "weakly" in str(fast_excinfo.value)

    @pytest.mark.smoke
    def test_cleanup_drops_the_receptive_field_solution_cache(self):
        """cleanup() must evict the rf solution cache, its only eviction path (R33).

        ``_receptive_field_solution`` (~54 MB on a resnet18 trace) lives
        outside MODEL_LOG_FIELD_ORDER, so the husking loop skipped it and the
        cache survived cleanup() with no eviction path at all.
        """

        trace = tl.trace(_SimpleLinear(), torch.randn(1, 5))
        trace.__dict__["_receptive_field_solution"] = object()
        trace.cleanup()
        assert "_receptive_field_solution" not in trace.__dict__, (
            "cleanup() left the receptive-field solution cache pinned"
        )

    @pytest.mark.smoke
    def test_transient_write_after_finish_does_not_recreate_build_state(self) -> None:
        """Finished traces reject writes after the build-state owner is dropped."""

        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))

        with pytest.raises(AttributeError):
            trace._module_capture_ws.mod_call_index = {"x": 1}

        assert "_build_state" not in trace.__dict__
        assert "_raw_graph_ws" not in trace.__dict__
        assert "_module_capture_ws" not in trace.__dict__
        assert "_wrapper_runtime_ws" not in trace.__dict__


class TestLifetimeCoverageGaps:
    """Lifetime arms for products the file did not observe (B20).

    The pre-existing arms were strict about the plain-capture path and blind
    everywhere the sprint added mass: pickle round-trips, forks, failed and
    partial captures, repeated failure/success cycles, and loaded archived
    activations all had no lifetime assertion at all.
    """

    @pytest.mark.smoke
    def test_pickled_round_trip_trace_is_collectible(self):
        """A trace rebuilt by pickle owns no extra roots."""

        import pickle

        trace = tl.trace(
            _TwoLayerNet(),
            torch.randn(1, 5),
            capture=tl.options.CaptureOptions(layers_to_save="all"),
        )
        restored = pickle.loads(pickle.dumps(trace))
        assert len(restored) > 0
        restored_ref = weakref.ref(restored)

        del restored
        gc.collect()
        assert restored_ref() is None
        trace.cleanup()

    @pytest.mark.smoke
    def test_forked_trace_and_its_parent_are_both_collectible(self):
        """A fork does not keep its parent alive, nor the parent the fork."""

        parent = tl.trace(
            _TwoLayerNet(),
            torch.randn(1, 5),
            capture=tl.options.CaptureOptions(layers_to_save="all"),
        )
        fork = parent.fork()
        fork_ref = weakref.ref(fork)
        parent_ref = weakref.ref(parent)

        del fork
        gc.collect()
        assert fork_ref() is None, "the fork was pinned by its parent"
        assert parent_ref() is not None

        del parent
        gc.collect()
        assert parent_ref() is None, "the parent was pinned after its fork died"

    @pytest.mark.smoke
    def test_run_result_fork_is_collectible(self):
        """The fork returned by a non-fast live ``trace.run()`` is reclaimable.

        R37: fork record shells used to translate the parent's
        ``_source_trace_strong`` extra into a STRONG self-edge to the fork, so
        the module-global weak-keyed op-accessor cache entry populated during
        ``run()`` strongly reached its own weak key and pinned the entire
        result fork (activation payloads included) for the process lifetime.
        """

        model = _TwoLayerNet().eval()
        source = tl.trace(
            model, torch.randn(1, 5), capture=tl.options.CaptureOptions(layers_to_save="all")
        )
        result = source.run(inputs=torch.randn(1, 5))
        fork_ref = weakref.ref(result.trace)

        del result
        gc.collect()
        assert fork_ref() is None, "trace.run() leaked its result fork"

        source_ref = weakref.ref(source)
        del source
        gc.collect()
        assert source_ref() is None

    @pytest.mark.smoke
    def test_failed_capture_partial_is_collectible_with_its_exception(self):
        """A failed capture's partial trace dies with the exception holding it.

        The capture and the recovery MUST happen inside a helper whose frame is
        gone before the assertion: an ``except`` block in the test body leaves the
        exception reachable from the test frame (which pytest keeps alive), and
        the partial trace hangs off ``exc.partial_log``. That is a measurement
        artifact, not a leak — this arm is structured so it cannot report one.
        """

        from torchlens import partial as partial_module

        class _Boom(nn.Module):
            def forward(self, x):
                _ = torch.relu(x)
                raise ValueError("failed-capture lifetime probe")

        def capture_and_recover():
            """Fail a capture, recover its partial, return only a weak handle."""

            try:
                tl.trace(_Boom(), torch.ones(2))
            except ValueError as error:
                return weakref.ref(partial_module.from_failed_capture(error).trace)
            return None  # pragma: no cover - the model always raises

        partial_ref = capture_and_recover()
        assert partial_ref is not None, "the failing capture did not raise"
        gc.collect()

        assert partial_ref() is None, (
            "the partial trace outlived both the exception and the wrapper"
        )

    @pytest.mark.heavy
    def test_repeated_failure_and_success_cycles_do_not_accumulate(self):
        """Alternating failed and successful captures leave nothing behind."""

        class _Boom(nn.Module):
            def forward(self, x):
                _ = torch.relu(x)
                raise ValueError("cycle probe")

        model = _TwoLayerNet()
        refs = []
        for _ in range(3):
            with pytest.raises(ValueError, match="cycle probe"):
                tl.trace(_Boom(), torch.ones(2))
            trace = tl.trace(model, torch.randn(1, 5))
            refs.append(weakref.ref(trace))
            del trace
            gc.collect()

        assert [ref() for ref in refs] == [None, None, None]

    @pytest.mark.smoke
    def test_loaded_archived_activations_die_with_their_trace(self, tmp_path):
        """A loaded runnable trace's archived activations are not process state."""

        model = _TwoLayerNet().eval()
        x = torch.randn(1, 5)
        trace = tl.trace(
            model,
            x,
            capture=tl.options.CaptureOptions(
                intervention_ready=True, cache=False, layers_to_save="all"
            ),
        )
        path = tmp_path / "gc_archived.tlspec"
        trace.save(path, level="runnable", include_weights=True, include_activations=True)
        del trace
        gc.collect()

        loaded = tl.load(path)
        assert loaded.archived_activations, "no archived activations were loaded"
        loaded_ref = weakref.ref(loaded)

        del loaded
        gc.collect()
        assert loaded_ref() is None, (
            "a loaded trace carrying archived activations was pinned by process state"
        )


@pytest.mark.smoke
def test_delattr_capture_events_releases_the_working_projection():
    """``del trace._capture_events`` clears a held stream's working lanes.

    Regression: ``Trace.__delattr__`` popped the attribute FIRST and then
    called ``forget_event_stream``, which looks up the attribute it just
    removed -- so ``release_working_projection()`` never ran and an outside
    holder kept every op event alive.
    """

    trace = tl.trace(_TwoLayerNet(), torch.randn(2, 5))
    stream = trace._capture_events
    assert stream.op_events

    del trace._capture_events

    assert trace.__dict__.get("_capture_events") is None
    assert not stream.op_events
    assert not stream.module_prep_events


@pytest.mark.smoke
def test_cleaned_trace_refuses_typed_and_settles_unknown():
    """A husked Trace refuses public reads TYPED and settles outcome UNKNOWN.

    b6-opus R25: seven public reads raised raw AttributeError naming
    whichever private field they touched first (``_tracing_finished``,
    ``_layers_logged``, ``layer_list``), ``Trace.outcome`` returned ``None``
    (outside the frozen vocabulary), and a second ``cleanup()`` crashed on
    the first cleanup's own output.
    """

    from torchlens._errors import TraceCleanedUpError
    from torchlens.capture.outcome import CaptureStatus

    trace = tl.trace(_TwoLayerNet(), torch.randn(2, 5))
    assert trace.outcome is not None and trace.outcome.status is CaptureStatus.COMPLETE
    trace.cleanup()

    # One typed code for every public reader.
    readers = {
        "summary": lambda: trace.summary(),
        "iteration": lambda: list(trace),
        "getitem": lambda: trace["relu_1_2"],
        "draw": lambda: trace.draw(vis_save_only=True),
        "receptive_fields": lambda: trace.receptive_fields(),
    }
    for name, reader in readers.items():
        with pytest.raises(TraceCleanedUpError) as exc_info:
            reader()
        assert exc_info.value.fields["code"] == "trace_cleaned_up", name
        assert exc_info.value.fields["remedy"], name

    # AttributeError lineage keeps hasattr/getattr-default degrade paths.
    assert getattr(trace, "layer_list", None) is None
    assert not hasattr(trace, "_tracing_finished")

    # The settled outcome is UNKNOWN (most restrictive), never None.
    assert trace.outcome is not None
    assert trace.outcome.status is CaptureStatus.UNKNOWN

    # Idempotent teardown and surviving diagnostics.
    trace.cleanup()
    assert repr(trace)
    assert tl.report.explain(trace)


@pytest.mark.smoke
def test_failed_capture_registry_never_pins_a_nonweakrefable_exception_graph():
    """The registry fallback stores an identity stub, never the exception.

    Regression (R37, REOPENED b2:C5): a non-weakrefable exception that also
    rejected ``partial_log`` attachment was retained STRONGLY, pinning its
    traceback's frame locals (model, inputs) until 128 unrelated failures
    evicted it.
    """

    import warnings

    class _LockedError(Exception):
        __slots__ = ()

        def __setattr__(self, name: str, value: object) -> None:
            if name == "partial_log":
                raise AttributeError("locked")
            super().__setattr__(name, value)

    class _FailingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.linear(x)
            raise _LockedError("boom")

    model = _FailingModel()
    model_ref = weakref.ref(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tl.trace(model, torch.ones(1, 3))
        except _LockedError as exc:
            partial = tl.partial.from_failed_capture(exc)
            assert partial is not None
            del exc, partial

    del model
    gc.collect()
    gc.collect()
    assert model_ref() is None, "the failed-capture registry pinned the exception graph"


@pytest.mark.smoke
def test_static_attr_memo_releases_a_self_referencing_class():
    """A class-valued cached answer must not pin its own weak memo key (R37).

    Regression (b2-sol): ``cls.self_ref = cls`` made the memo's strong VALUE
    reach its weak KEY, so eviction could never start and the class leaked for
    the process lifetime.
    """

    from torchlens._io.state_keys import _STATIC_ATTR_MEMO, static_class_attr

    ephemeral = type("_EphemeralSelfRef", (), {})
    ephemeral.self_ref = ephemeral
    assert static_class_attr(ephemeral, "self_ref") is ephemeral
    cls_ref = weakref.ref(ephemeral)
    del ephemeral
    gc.collect()
    gc.collect()
    assert cls_ref() is None, "self-referencing class pinned by _STATIC_ATTR_MEMO"
    assert all(key is not None for key in _STATIC_ATTR_MEMO)


@pytest.mark.smoke
def test_merged_trace_release_drops_member_traces():
    """MergedTrace.release() unpins the rank traces it held strongly (b2:B20).

    Regression guard for the presenter's lifetime contract: without release(),
    a presenter over live traces transitively pinned every member (and its
    activations) with no counterpart to Trace.cleanup().
    """

    from torchlens.merged._presenter import MergedTrace, _RankHandle

    trace_a = tl.trace(_SimpleLinear(), torch.randn(2, 5))
    trace_b = tl.trace(_SimpleLinear(), torch.randn(2, 5))
    refs = [weakref.ref(trace_a), weakref.ref(trace_b)]
    merged = MergedTrace(
        derivation=None,  # placeholder: release() must not need the derivation
        handles={0: _RankHandle(0, trace=trace_a), 1: _RankHandle(1, trace=trace_b)},
    )
    del trace_a, trace_b
    gc.collect()
    assert all(ref() is not None for ref in refs), "presenter should pin members"

    merged.release()
    gc.collect()
    gc.collect()
    assert all(ref() is None for ref in refs), "release() left a member pinned"
    merged.release()  # idempotent
