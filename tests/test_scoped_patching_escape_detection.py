"""Adversarial certification for scoped patching and shadow escape detection."""

from __future__ import annotations

import functools
import sys
import threading
import types
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import example_models
import numpy as np
import pytest
import torch
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._errors import TorchLensCaptureGapWarning
from torchlens.backends.torch.escape_detection import (
    AUDITED_ESCAPE_EXEMPTIONS,
    MAX_AUDITED_EXEMPTIONS,
    _find_monitoring_tool_id,
    reset_detector_tables,
)
from torchlens.backends.torch.wrappers import (
    torch_func_decorator,
    unwrap_torch,
    wrap_torch,
)
from torchlens.options import CaptureOptions


@pytest.fixture(autouse=True)
def _isolated_wrapper_epoch(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give every certification test a clean wrapper policy epoch.

    The stage-2 rescue re-run is bypassed here: this module certifies the
    escape DETECTOR (the sensor) in isolation — with rescue live, a detected
    escape would be recovered and the shadow report would move into the
    ``rescue_rerun`` disclosure. The integrated sensor->rescue path is
    covered by ``test_rescue_rerun.py`` and the outcome corpus.
    """

    from torchlens.backends.torch import rescue as rescue_module

    monkeypatch.setattr(
        rescue_module, "capture_with_rescue", lambda run_capture, **_kw: run_capture()
    )
    # Restore the PRE-TEST diagnostic modes on teardown instead of hardcoding
    # them off: a fixed escape_detector="off" re-wrap disarmed diagnostics a
    # surrounding session had deliberately armed (R77 fixture-health finding 3).
    saved_escape_detector = _state._escape_detector_mode
    saved_completeness_witness = _state._completeness_witness_mode
    unwrap_torch()
    yield
    unwrap_torch()
    wrap_torch(
        escape_detector=saved_escape_detector,
        completeness_witness=saved_completeness_witness,
    )


def _gap_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    """Return only callable/thread capture-gap warnings from a warning list."""

    return [item for item in caught if isinstance(item.message, TorchLensCaptureGapWarning)]


def _run_shadow_capture(model: nn.Module) -> tuple[BaseException | None, list[str]]:
    """Run one scoped shadow capture and return its exception and gap warnings."""

    wrap_torch(escape_detector="shadow")
    error: BaseException | None = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            tl.trace(model, torch.randn(3))
        except BaseException as exc:  # noqa: BLE001 - test captures the public failure path
            error = exc
    return error, [str(item.message) for item in _gap_warnings(caught)]


def _module_with_hidden_class_ref(name: str, raw: Callable[..., Any]) -> types.ModuleType:
    """Create a module whose only raw reference is a class attribute."""

    module = types.ModuleType(name)
    exec("class Holder:\n    pass\n", module.__dict__)
    module.Holder.op = raw
    sys.modules[name] = module
    return module


@pytest.mark.parametrize("holder_kind", ["closure", "dict", "list", "instance"])
def test_builtin_holder_attacks_are_rebound_or_shadow_reported(holder_kind: str) -> None:
    """Builtin holders are rebound for the capture; a custom object is reported.

    Capture preparation rebinds pristine refs held in closure cells and exact
    ``dict``/``list`` containers, so those calls are no escape at all. A plain
    custom object is never rebound, and the shadow detector reports its call.
    """

    raw = torch.relu

    class Holder:
        """Plain non-model callable holder."""

    holder: Any
    if holder_kind == "closure":
        holder = raw
    elif holder_kind == "dict":
        holder = {"op": raw}
    elif holder_kind == "list":
        holder = [raw]
    else:
        holder = Holder()
        holder.op = raw

    def invoke(value: torch.Tensor) -> torch.Tensor:
        """Invoke the callable through the selected holder shape."""

        if holder_kind == "closure":
            return holder(value)
        if holder_kind == "dict":
            return holder["op"](value)
        if holder_kind == "list":
            return holder[0](value)
        return holder.op(value)

    class Model(nn.Module):
        """Invoke the selected hidden holder."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Call the escaped raw builtin."""

            return invoke(x)

    error, reports = _run_shadow_capture(Model())
    if holder_kind != "instance":
        assert error is None
        assert reports == []
        return
    assert isinstance(error, RuntimeError)
    assert len(reports) == 1
    assert "relu" in reports[0]
    assert hasattr(error, "partial_log")
    assert error.partial_log.trace.escape_diagnostics


def test_raw_python_functional_and_partial_emit_shadow_reports() -> None:
    """Raw Python code identity is visible directly and through partial.

    Each call sits in a custom callable object, which capture preparation
    never rebinds, so the raw code still runs and the detector must see it.
    """

    raw = torch.nn.functional.softsign
    invocations = (OpaqueCallable(raw), OpaqueCallable(functools.partial(raw)))
    for invoke in invocations:

        class Model(nn.Module):
            """Invoke a raw Python functional holder."""

            def forward(self, x: torch.Tensor, invoke: Callable[..., Any] = invoke) -> torch.Tensor:
                """Call the hidden raw Python functional (bound per iteration)."""

                return invoke(x)

        error, reports = _run_shadow_capture(Model())
        assert error is None or isinstance(error, RuntimeError)
        assert len(reports) == 1
        assert "softsign" in reports[0]
        unwrap_torch()


def test_hidden_class_ref_is_shadow_reported_and_default_ref_rebound() -> None:
    """Unrelated local class storage is reported; a default-arg ref is rebound.

    Capture preparation rebinds a helper's default argument for the capture,
    so that call is no escape; a class attribute is never rebound, and the
    shadow detector reports it.
    """

    raw = torch.relu

    class Holder:
        """Class-attribute holder outside model provenance."""

        op = raw

    def uses_default(x: torch.Tensor, op: Callable[..., Any] = raw) -> torch.Tensor:
        """Invoke a hidden default-argument callable."""

        return op(x)

    for invoke, reported in ((lambda value: Holder.op(value), True), (uses_default, False)):

        class Model(nn.Module):
            """Invoke one unrelated hidden helper."""

            def forward(self, x: torch.Tensor, invoke: Callable[..., Any] = invoke) -> torch.Tensor:
                """Delegate to the helper (bound per iteration)."""

                return invoke(x)

        error, reports = _run_shadow_capture(Model())
        if reported:
            assert isinstance(error, RuntimeError)
            assert len(reports) == 1
        else:
            assert error is None
            assert reports == []
        unwrap_torch()


def test_tensor_descriptor_uses_identity_compatible_bound_builtin_rule() -> None:
    """A transient bound builtin identifies a hidden Tensor descriptor escape."""

    raw_descriptor = torch.Tensor.add

    class Model(nn.Module):
        """Invoke a saved unbound Tensor method descriptor."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Call the descriptor with ``x`` as its receiver."""

            return raw_descriptor(x, 1)

    error, reports = _run_shadow_capture(Model())
    assert isinstance(error, RuntimeError)
    assert len(reports) == 1
    assert "TensorBase.add" in reports[0]


def test_saved_bound_builtin_is_convicted_outside_token_edge() -> None:
    """A saved Tensor-bound builtin is reportable without stable target identity."""

    owner = torch.zeros(3)
    raw_bound_add = owner.add

    class Model(nn.Module):
        """Invoke the saved bound builtin outside a registered wrapper edge."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Use the model input as the saved method's argument."""

            return raw_bound_add(x)

    error, reports = _run_shadow_capture(Model())
    assert isinstance(error, RuntimeError)
    assert len(reports) == 1
    assert "TensorBase.add" in reports[0]


def test_descriptor_escape_inside_wrapper_token_is_not_window_exempted() -> None:
    """A callback escape inside a composite wrapper remains reportable."""

    raw_descriptor = torch.Tensor.add
    wrap_torch(escape_detector="shadow")

    def composite(
        x: torch.Tensor, callback: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Delegate to a user callback inside the original's dynamic extent."""

        return callback(x)

    decorated = torch_func_decorator(composite, "test_callback_composite")
    _state._orig_to_decorated[id(composite)] = decorated
    _state._decorated_to_orig[id(decorated)] = composite
    _state._decorated_func_mapper[composite] = decorated
    _state._decorated_func_mapper[decorated] = composite
    reset_detector_tables()

    class Model(nn.Module):
        """Invoke a raw descriptor from inside the composite's callback."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the adversarial callback under an outer wrapper token."""

            def callback(value: torch.Tensor) -> torch.Tensor:
                """Invoke the hidden descriptor."""

                return raw_descriptor(value, 1)

            return decorated(x, callback)

    try:
        with pytest.warns(TorchLensCaptureGapWarning, match="TensorBase.add"):
            trace = tl.trace(Model(), torch.randn(3))
        assert len(trace.escape_diagnostics) == 1
        assert trace.escape_diagnostics[0]["guard_pass_index"] >= 1
    finally:
        _state._orig_to_decorated.pop(id(composite), None)
        _state._decorated_to_orig.pop(id(decorated), None)
        _state._decorated_func_mapper.pop(composite, None)
        _state._decorated_func_mapper.pop(decorated, None)
        reset_detector_tables()


def test_c_partial_blind_spot_is_machine_readably_unverified() -> None:
    """Profile-blind C partial remains an honest pre-witness boundary."""

    raw_partial = functools.partial(torch.relu)

    class Model(nn.Module):
        """Feed a C-partial result into a live wrapped operation."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the known profile-blind holder channel."""

            return torch.sigmoid(raw_partial(x))

    wrap_torch(escape_detector="shadow")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(Model(), torch.randn(3))
    assert _gap_warnings(caught) == []
    assert trace.escape_diagnostics == []
    # The detector cannot see the C-partial channel; shadow mode's blanket
    # no-claim ceiling is what keeps the blind spot machine-readably honest.
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "shadow_diagnostic_mode"


def test_clean_composites_and_descriptor_wrappers_do_not_convict() -> None:
    """Ordinary Torch composites and legitimate Tensor methods do not convict."""

    class Model(nn.Module):
        """Exercise Python composite, getset, method, and print wrapper paths."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run representative legitimate wrapper-to-original edges."""

            value = torch.nn.functional.softsign(x).add(1)
            _ = repr(value)
            return value.real

    wrap_torch(escape_detector="shadow")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(Model(), torch.randn(3))
    assert trace.escape_diagnostics == []
    assert _gap_warnings(caught) == []
    assert len(AUDITED_ESCAPE_EXEMPTIONS) == 16
    assert all(
        row.caller_filename_suffix == "torch/_jit_internal.py" for row in AUDITED_ESCAPE_EXEMPTIONS
    )
    assert all(row.caller_name == "fn" for row in AUDITED_ESCAPE_EXEMPTIONS)
    assert all(row.reason for row in AUDITED_ESCAPE_EXEMPTIONS)
    assert len(AUDITED_ESCAPE_EXEMPTIONS) <= MAX_AUDITED_EXEMPTIONS


@pytest.mark.parametrize(
    ("function_name", "input_shape", "kwargs"),
    [
        ("max_pool1d", (1, 1, 8), {"kernel_size": 2}),
        ("max_pool2d", (1, 1, 8, 8), {"kernel_size": 2}),
        ("max_pool3d", (1, 1, 6, 6, 6), {"kernel_size": 2}),
        ("adaptive_max_pool1d", (1, 1, 8), {"output_size": 2}),
        ("adaptive_max_pool2d", (1, 1, 8, 8), {"output_size": 2}),
        ("adaptive_max_pool3d", (1, 1, 6, 6, 6), {"output_size": 2}),
        (
            "fractional_max_pool2d",
            (1, 1, 8, 8),
            {"kernel_size": 2, "output_size": 3},
        ),
        (
            "fractional_max_pool3d",
            (1, 1, 6, 6, 6),
            {"kernel_size": 2, "output_size": 2},
        ),
    ],
)
def test_boolean_dispatch_pooling_family_does_not_convict(
    function_name: str,
    input_shape: tuple[int, ...],
    kwargs: dict[str, Any],
) -> None:
    """Pre-wrap boolean-dispatch closure branches are audited clean composites.

    One capture drives BOTH branches of the dispatcher (``return_indices``
    False and True): the per-cell cost is the wrapper re-arm around each
    capture, so a cell per branch doubled the family's cost for no extra
    coverage.
    """

    class Model(nn.Module):
        """Invoke one live functional pooling dispatcher through both branches."""

        def forward(self, x: torch.Tensor) -> Any:
            """Run the values branch, then the indices branch."""

            pool = getattr(torch.nn.functional, function_name)
            values = pool(x, return_indices=False, **kwargs)
            pooled, indices = pool(x, return_indices=True, **kwargs)
            return values, pooled, indices

    wrap_torch(escape_detector="shadow")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(Model(), torch.randn(input_shape))
    assert trace.escape_diagnostics == []
    assert _gap_warnings(caught) == []
    compute_ops = [op for op in trace.layer_list if op.layer_type not in {"input", "output"}]
    assert len(compute_ops) >= 2, "both dispatcher branches must be captured"


def test_pause_logging_excludes_raw_internal_work() -> None:
    """A raw call while logging is paused is outside detector scope."""

    raw = torch.relu

    class Model(nn.Module):
        """Discard one deliberately paused raw result."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Pause around raw work, then return represented work."""

            with _state.pause_logging():
                raw(x)
            return torch.sigmoid(x)

    wrap_torch(escape_detector="shadow")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(Model(), torch.randn(3))
    assert trace.escape_diagnostics == []
    assert _gap_warnings(caught) == []


def test_synchronous_dataloader_callback_is_in_owner_thread_domain() -> None:
    """A num_workers=0 callback escape is reported inside forward.

    The raw call sits in a custom callable object, which capture preparation
    never rebinds.
    """

    raw = OpaqueCallable(torch.relu)

    def collate(value: torch.Tensor) -> torch.Tensor:
        """Invoke the hidden raw callable synchronously."""

        return raw(value)

    class Model(nn.Module):
        """Iterate a synchronous DataLoader inside forward."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Feed the callback result into a represented operation."""

            loader = torch.utils.data.DataLoader(
                [x], batch_size=None, num_workers=0, collate_fn=collate
            )
            return torch.sigmoid(next(iter(loader)))

    wrap_torch(escape_detector="shadow")
    with pytest.warns(TorchLensCaptureGapWarning, match="relu"):
        trace = tl.trace(Model(), torch.randn(3))
    assert len(trace.escape_diagnostics) == 1


def test_monitoring_tool_exhaustion_fails_loudly() -> None:
    """A Python 3.12 monitoring conflict cannot silently disable diagnostics."""

    class FullMonitoring:
        """Minimal monitoring facade with every tool id occupied."""

        def get_tool(self, tool_id: int) -> str:
            """Return a non-None owner for every tool id."""

            return f"owner-{tool_id}"

        def use_tool_id(self, tool_id: int, name: str) -> None:
            """Reject an attempted reservation; unreachable for this facade."""

            raise AssertionError((tool_id, name))

    with pytest.raises(RuntimeError, match="No free sys.monitoring tool id"):
        _find_monitoring_tool_id(FullMonitoring())


def test_external_profile_hook_is_chained_and_restored_on_success_and_error() -> None:
    """Shadow profiling composes with and restores the user's exact hook."""

    calls = 0

    def prior(frame: types.FrameType, event: str, arg: Any) -> None:
        """Count chained profile events without changing dispatch."""

        del frame, event, arg
        nonlocal calls
        calls += 1

    class Clean(nn.Module):
        """Simple successful model."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one represented operation."""

            return torch.relu(x)

    class Failing(nn.Module):
        """Model that raises after represented work."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Raise a deterministic user exception."""

            torch.relu(x)
            raise ValueError("profile restoration probe")

    wrap_torch(escape_detector="shadow")
    sys.setprofile(prior)
    try:
        tl.trace(Clean(), torch.randn(3))
        assert sys.getprofile() is prior
        with pytest.raises(ValueError, match="profile restoration probe"):
            tl.trace(Failing(), torch.randn(3))
        assert sys.getprofile() is prior
        assert calls > 0
    finally:
        sys.setprofile(None)


def test_rng_and_escape_profile_detectors_coarm_without_lost_detection() -> None:
    """The nested RNG profile hook chains the escape detector and restores both.

    The capture must be runnable-capable. ``_runnable_host_rng_channels`` only exists
    when the host-RNG monitor is ARMED, and a plain ``tl.trace`` deliberately does not
    arm it -- that fail-closed contract is pinned by
    ``test_plain_trace_does_not_arm_monitor_and_stamps_fail_closed`` in
    ``tests/test_rng_witness_gating.py``, which asserts the field is ABSENT after a
    plain trace. This test previously used a plain trace, so only the escape
    detector's hook was ever installed and the co-arming it names could not be
    observed at all.
    """

    if hasattr(sys, "monitoring"):
        pytest.skip("escape detection uses sys.monitoring instead of setprofile on Python 3.12+")
    generator = np.random.default_rng(123)
    # A custom callable object: capture preparation never rebinds it.
    raw_relu = OpaqueCallable(torch.relu)

    class DualDetectionModel(nn.Module):
        """Exercise one NumPy RNG draw and one raw torch callable escape."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Trigger both profile-hook receivers in one owner-thread forward."""

            generator.random()
            return torch.sigmoid(raw_relu(x))

    wrap_torch(escape_detector="shadow")
    with pytest.warns(TorchLensCaptureGapWarning, match="relu"):
        trace = tl.trace(
            DualDetectionModel(),
            torch.randn(3),
            capture=CaptureOptions(
                intervention_ready=True,
                capture_container_structure=True,
                cache=False,
                random_seed=7,
            ),
        )

    # Assert the property under test -- BOTH receivers fired -- by naming the escape,
    # not by counting diagnostics. An armed capture also self-reports one escape for
    # TorchLens's own `completeness_witness._raw_storage_ptr_no_observe` calling
    # `TensorBase.untyped_storage`, which is absent from a plain capture. That
    # self-trip is tracked separately; a total-count assertion here would silently
    # couple this test to it.
    escaped = {
        candidate
        for diagnostic in trace.escape_diagnostics
        for candidate in diagnostic["callable_candidates"]
    }
    assert any("relu" in candidate for candidate in escaped), escaped
    assert trace._runnable.rng_monitor_uncertain is False
    assert "c_rng_instance_draw" in trace._runnable.host_rng_channels
    assert sys.getprofile() is None


def test_record_fastlog_uses_same_guard_and_backward_boundary_is_explicit() -> None:
    """Public record is armed; deferred backward is explicitly not armed."""

    class Model(nn.Module):
        """Small differentiable model for record/backward coverage."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return one differentiable represented operation."""

            return torch.relu(x)

    wrap_torch(escape_detector="shadow")
    recording = tl.record(Model(), torch.randn(3), save=tl.func("relu"))
    assert recording.escape_detector_mode == "shadow"
    assert recording.capture_owner_thread_qualified is True
    assert recording.escape_diagnostics == []
    trace = tl.trace(Model(), torch.randn(3, requires_grad=True))
    assert trace.escape_detector_backward_coverage == "not_armed"
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    assert trace.has_backward_pass is True


def test_thread_count_tripwire_marks_trace_unverified() -> None:
    """A thread-count delta is warned and exposed machine-readably."""

    release = threading.Event()

    class Model(nn.Module):
        """Start a live worker during forward without doing tensor work there."""

        def __init__(self) -> None:
            """Initialize worker storage."""

            super().__init__()
            self.worker: threading.Thread | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Leave one daemon worker alive until the caller releases it."""

            self.worker = threading.Thread(target=release.wait, daemon=True)
            self.worker.start()
            return torch.relu(x)

    model = Model()
    wrap_torch(escape_detector="shadow")
    try:
        with pytest.warns(TorchLensCaptureGapWarning, match="thread-count change"):
            trace = tl.trace(model, torch.randn(3))
        assert trace.capture_owner_thread_qualified is True
        assert trace.capture_thread_activity_detected is True
        assert trace.capture_verified is False
        assert trace.capture_verification_reason == "owner_thread_tripwire_changed"
    finally:
        release.set()
        if model.worker is not None:
            model.worker.join(timeout=2)


def test_default_detector_is_off_and_default_trace_makes_no_claim() -> None:
    """The callable detector remains opt-in; a default trace claims nothing.

    (The historical scoped-policy honesty ceiling — every scoped trace marked
    unverified — died with the crawler's policy machinery.)
    """

    class Model(nn.Module):
        """Simple represented model."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one live torch lookup."""

            return torch.relu(x)

    wrap_torch()
    trace = tl.trace(Model(), torch.randn(3))
    assert trace.escape_detector_mode == "off"
    assert trace.capture_verified is None
    assert trace.capture_verification_reason is None


def test_scoped_and_legacy_match_on_in_scope_standard_model() -> None:
    """Scoped and release-default discovery produce the same in-scope graph."""

    class Model(nn.Module):
        """Small standard model with only live/in-scope references."""

        def __init__(self) -> None:
            """Create deterministic submodules."""

            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run linear and live activation operations."""

            return torch.sigmoid(self.linear(x)).add(1)

    torch.manual_seed(7)
    model = Model()
    inputs = torch.randn(2, 4)
    wrap_torch()
    legacy = tl.trace(model, inputs)
    unwrap_torch()
    wrap_torch()
    scoped = tl.trace(model, inputs)
    assert scoped.graph_shape_hash == legacy.graph_shape_hash
    assert [op.func_name for op in scoped.ops] == [op.func_name for op in legacy.ops]
    assert scoped.layer_labels == legacy.layer_labels


@pytest.mark.parametrize(
    "model_type",
    [
        example_models.SimpleFF,
        example_models.GeluModel,
        example_models.SimpleInternallyGenerated,
        example_models.SimpleBranching,
        example_models.SimpleLoopNoParam,
        example_models.NestedModules,
    ],
)
def test_scoped_and_legacy_match_standard_test_model_zoo(
    model_type: type[nn.Module],
) -> None:
    """Scoped matches legacy graphs across representative standard zoo axes."""

    inputs = torch.full((5,), 2.0)
    wrap_torch()
    legacy = tl.trace(model_type(), inputs)
    unwrap_torch()
    wrap_torch()
    scoped = tl.trace(model_type(), inputs)
    assert scoped.graph_shape_hash == legacy.graph_shape_hash
    assert [op.func_name for op in scoped.ops] == [op.func_name for op in legacy.ops]
    assert scoped.layer_labels == legacy.layer_labels


def test_guard_pass_metadata_is_machine_readable() -> None:
    """Each active-logging pass carries an owner-thread-qualified record."""

    class Model(nn.Module):
        """Simple model used with the legacy two-pass save spelling."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one represented operation."""

            return torch.relu(x)

    wrap_torch(escape_detector="shadow")
    trace = tl.trace(
        Model(), torch.randn(3), capture=tl.options.CaptureOptions(layers_to_save=["relu"])
    )
    assert trace.capture_guard_passes
    assert all(
        item["owner_thread_id"] == trace.capture_owner_thread_id
        for item in trace.capture_guard_passes
    )


def test_witness_internal_storage_read_does_not_self_trip_detector() -> None:
    """The armed completeness witness never reports its OWN raw storage reads.

    Regression for the reds-lane finding: with scoped wrapping and the shadow
    escape detector, an armed (runnable-capable) capture reported
    ``TensorBase.untyped_storage`` from TorchLens's own
    ``_raw_storage_ptr_no_observe`` frame, degrading otherwise-verified armed
    captures with user-directed remediation no user action could clear. The
    witness now authorizes its raw-original reads through the detector's
    FRAME-BOUND ``expected_original_call`` accounting, so the exemption covers
    exactly that call site: raw callable reaches anywhere else still trip the
    detector (see the coarm test above for the positive detection control).
    """

    from torchlens.options import CaptureOptions

    class Model(nn.Module):
        """Two represented ops; nothing escapes."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run a clean forward."""

            return torch.sigmoid(torch.relu(x))

    wrap_torch(escape_detector="shadow")
    trace = tl.trace(
        Model(),
        torch.randn(3),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
            random_seed=7,
        ),
    )
    self_trips = [
        diagnostic
        for diagnostic in trace.escape_diagnostics
        if any(
            "untyped_storage" in str(candidate)
            for candidate in diagnostic.get("callable_candidates", ())
        )
    ]
    assert not self_trips, f"witness self-trip diagnostics: {self_trips}"


def test_user_call_into_witness_storage_helper_degrades_verification() -> None:
    """User model code calling the witness's raw-storage helper is an escape.

    Inverse control for the self-trip regression above: the helper's
    authorization is bound to TorchLens's OWN calling frames, not to the
    helper's identity. User model code that imports and calls
    ``_raw_storage_ptr_no_observe`` executes pointer-dependent control flow
    outside every wrapper, so an armed capture must NOT report
    ``capture_verified`` with empty escape diagnostics.
    """

    from torchlens.backends.torch.completeness_witness import _raw_storage_ptr_no_observe
    from torchlens.options import CaptureOptions

    class CallsAuthorizedWitnessFrame(nn.Module):
        """Branches on a raw storage pointer read through the helper."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Choose an op from an unobserved raw pointer."""

            ptr = _raw_storage_ptr_no_observe(x)
            if ptr is not None and ((ptr >> 8) & 1):
                return torch.relu(x)
            return torch.sigmoid(x)

    wrap_torch(escape_detector="shadow", completeness_witness=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            CallsAuthorizedWitnessFrame(),
            torch.randn(3),
            capture=CaptureOptions(
                intervention_ready=True,
                capture_container_structure=True,
                cache=False,
            ),
        )
    assert not (trace.capture_verified and not trace.escape_diagnostics), (
        trace.capture_verification_reason,
        trace.escape_diagnostics,
    )
    storage_trips = [
        diagnostic
        for diagnostic in trace.escape_diagnostics
        if any(
            "untyped_storage" in str(candidate) or "data_ptr" in str(candidate)
            for candidate in diagnostic.get("callable_candidates", ())
        )
    ]
    assert storage_trips


def test_forged_frame_metadata_cannot_impersonate_witness_authorization() -> None:
    """FAIL-AFTER-WHERE-PASSED-BEFORE: frame-metadata forgery gains no authorization.

    Review be2-closure probe regression: the internal-caller check used to trust
    the caller frame's ``f_globals['__name__']`` and ``co_filename``, both of
    which user code controls -- a ``forward`` compiled with a torchlens-ish
    module name and a fabricated filename under the package directory was
    granted the witness's detector authorization and produced a verified
    capture with empty escape diagnostics around pointer-dependent control
    flow. Authorization is now code-object IDENTITY against the import-time
    roster, which ``exec``/``compile`` forgery cannot reproduce: the forged
    frame's raw reads run bare and the shadow detector convicts them exactly
    like the undisguised user call in the test above.
    """

    from torchlens.backends.torch import completeness_witness as witness_module
    from torchlens.backends.torch.completeness_witness import _raw_storage_ptr_no_observe
    from torchlens.options import CaptureOptions

    forged_globals = {
        "__name__": "torchlens.user_supplied_model",
        "__builtins__": __builtins__,
        "torch": torch,
        "_raw_storage_ptr_no_observe": _raw_storage_ptr_no_observe,
    }
    forged_filename = str(Path(witness_module.__file__).resolve().parent / "user_supplied_model.py")
    code = compile(
        "def forward(self, x):\n"
        "    ptr = _raw_storage_ptr_no_observe(x)\n"
        "    if ptr is not None and ((ptr >> 8) & 1):\n"
        "        return torch.relu(x)\n"
        "    return torch.sigmoid(x)\n",
        forged_filename,
        "exec",
    )
    exec(code, forged_globals)
    ForgedFrameModel = type(
        "ForgedFrameModel", (nn.Module,), {"forward": forged_globals["forward"]}
    )

    wrap_torch(escape_detector="shadow", completeness_witness=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            ForgedFrameModel(),
            torch.randn(3),
            capture=CaptureOptions(
                intervention_ready=True,
                capture_container_structure=True,
                cache=False,
            ),
        )
    assert not (trace.capture_verified and not trace.escape_diagnostics), (
        trace.capture_verification_reason,
        trace.escape_diagnostics,
    )
    storage_trips = [
        diagnostic
        for diagnostic in trace.escape_diagnostics
        if any(
            "untyped_storage" in str(candidate) or "data_ptr" in str(candidate)
            for candidate in diagnostic.get("callable_candidates", ())
        )
    ]
    assert storage_trips


def test_detector_teardown_failure_demotes_and_discloses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed detector uninstall must demote the verdict, not be swallowed."""

    from torchlens.backends.torch import escape_detection as escape_detection_module

    real_uninstall = escape_detection_module._uninstall_detector
    fail_once = {"armed": True}

    def failing_uninstall(guard: Any) -> None:
        real_uninstall(guard)
        if fail_once["armed"]:
            fail_once["armed"] = False
            raise RuntimeError("injected teardown failure")

    monkeypatch.setattr(escape_detection_module, "_uninstall_detector", failing_uninstall)

    class Clean(nn.Module):
        """Escape-free control model."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

    wrap_torch(escape_detector="shadow")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(Clean(), torch.randn(3))
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "escape_detector_teardown_failed"
    assert trace.escape_detector_verified is False
    teardown_reports = [
        diagnostic
        for diagnostic in trace.escape_diagnostics
        if diagnostic.get("kind") == "detector_teardown_failed"
    ]
    assert teardown_reports and "injected teardown failure" in teardown_reports[0]["error"]
    assert any(
        "failed to uninstall its escape detector" in str(item.message)
        for item in _gap_warnings(caught)
    )

    # The failure must not poison the process: the next capture runs and settles
    # its ordinary shadow verdict with no teardown diagnostic.
    second = tl.trace(Clean(), torch.randn(3))
    assert second.capture_verification_reason == "shadow_diagnostic_mode"
    assert not any(
        diagnostic.get("kind") == "detector_teardown_failed"
        for diagnostic in second.escape_diagnostics
    )


def test_owner_thread_scalar_only_stale_escape_is_shadow_reported() -> None:
    """The owner-thread scalar-only stale crossing is observable via shadow mode.

    On DEFAULT captures this class (``stale_norm(x).item() > t`` -- no
    intermediate tensor op consumes the stale output) is a DECLARED silent
    residual (docs/migration/scoped_detached_patching.md, Honest boundaries):
    the scalar-protocol read of the untracked intermediate emits no record,
    and unlabeled receivers cannot be flagged without false-positives on
    parameter/attribute scalar reads. This pin proves the opt-in shadow
    detector reports the stale CALL itself, so the documented remediation
    path is real (b3-fable R02-1). The stale ref sits in a custom callable
    object, which capture preparation never rebinds.
    """

    unwrap_torch()
    stale_norm = OpaqueCallable(torch.linalg.norm)
    wrap_torch(escape_detector="shadow")

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if stale_norm(x).item() > 0.001:
                return torch.relu(self.lin(x))
            return torch.sigmoid(self.lin(x))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(Model(), torch.randn(3, 4))

    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "callable_escape_shadow_report"
    assert trace.escape_diagnostics
    assert _gap_warnings(caught)
