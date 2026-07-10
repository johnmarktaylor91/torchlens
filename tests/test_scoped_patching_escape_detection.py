"""Adversarial certification for scoped patching and shadow escape detection."""

from __future__ import annotations

import functools
import gc
import sys
import threading
import types
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._errors import TorchLensCaptureGapWarning
import example_models
from torchlens.backends.torch.escape_detection import (
    AUDITED_ESCAPE_EXEMPTIONS,
    MAX_AUDITED_EXEMPTIONS,
    _find_monitoring_tool_id,
    reset_detector_tables,
)
from torchlens.backends.torch.wrappers import (
    _remember_crawled_module_identity,
    _remember_positive_module,
    _scoped_hot_module_ids,
    clear_patch_detached_references_cache,
    patch_detached_references,
    torch_func_decorator,
    unwrap_torch,
    wrap_torch,
)


@pytest.fixture(autouse=True)
def _isolated_wrapper_epoch() -> Iterator[None]:
    """Give every certification test a clean wrapper policy epoch."""

    unwrap_torch()
    clear_patch_detached_references_cache()
    yield
    unwrap_torch()
    clear_patch_detached_references_cache()
    wrap_torch(patch_policy="legacy", escape_detector="off")


def _gap_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    """Return only callable/thread capture-gap warnings from a warning list."""

    return [item for item in caught if isinstance(item.message, TorchLensCaptureGapWarning)]


def _run_shadow_capture(model: nn.Module) -> tuple[BaseException | None, list[str]]:
    """Run one scoped shadow capture and return its exception and gap warnings."""

    wrap_torch(patch_policy="scoped", escape_detector="shadow")
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


def test_scoped_policy_is_real_not_legacy_alias() -> None:
    """Scoped must leave an unrelated hidden class ref that legacy deep-scans."""

    raw = torch.relu
    name = "_tl_scoped_deleted_alias_contract"
    module = _module_with_hidden_class_ref(name, raw)
    try:
        wrap_torch(patch_policy="scoped")
        report = patch_detached_references(policy="scoped")
        assert report.policy == "scoped"
        assert vars(module.Holder)["op"] is raw

        unwrap_torch()
        wrap_torch(patch_policy="legacy")
        assert vars(module.Holder)["op"] is _state._orig_to_decorated[id(raw)]
    finally:
        sys.modules.pop(name, None)


def test_scoped_model_provenance_patches_class_and_default_refs() -> None:
    """Model-defining modules receive bounded class/default deep scanning."""

    raw_relu = torch.relu
    raw_tanh = torch.tanh
    name = "_tl_scoped_model_provenance"
    module = types.ModuleType(name)
    module.__dict__.update({"nn": nn, "torch": torch, "relu": raw_relu, "tanh": raw_tanh})
    sys.modules[name] = module
    try:
        exec(
            "def helper(x, op=tanh):\n"
            "    return op(x)\n"
            "class Model(nn.Module):\n"
            "    activation = relu\n"
            "    def forward(self, x):\n"
            "        return type(self).activation(helper(x))\n"
            "del relu\n"
            "del tanh\n",
            module.__dict__,
        )
        wrap_torch(patch_policy="scoped", escape_detector="shadow")
        trace = tl.trace(module.Model(), torch.randn(3))
        assert {"relu", "tanh"} <= {op.func_name for op in trace.ops}
        assert trace.escape_diagnostics == []
    finally:
        sys.modules.pop(name, None)


def test_scoped_allowlist_patches_unrelated_helper() -> None:
    """An additive module allowlist deep-scans a hidden helper class ref."""

    raw = torch.relu
    name = "_tl_scoped_allowlisted_helper"
    module = _module_with_hidden_class_ref(name, raw)
    try:
        wrap_torch(patch_policy="scoped", patch_modules=(name,))
        assert vars(module.Holder)["op"] is _state._orig_to_decorated[id(raw)]
    finally:
        sys.modules.pop(name, None)


def test_scoped_never_reads_module_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scoped discovery performs exact identity work without opening source files."""

    wrap_torch(patch_policy="scoped")
    clear_patch_detached_references_cache()
    opened: list[Any] = []
    original_open = open

    def recording_open(*args: Any, **kwargs: Any) -> Any:
        """Record source reads while preserving ordinary open behavior."""

        opened.append(args[0] if args else None)
        return original_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", recording_open)
    report = patch_detached_references(policy="scoped")
    assert report.source_files_opened == 0
    assert opened == []


def test_legacy_reports_source_files_opened(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy reports successful source reads instead of a hardcoded zero."""

    wrap_torch(patch_policy="legacy")
    source_path = tmp_path / "torchlens_source_count_probe.py"
    source_path.write_text("import torch\n", encoding="utf-8")
    module_name = "_tl_legacy_source_count_probe"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    opened: list[Any] = []
    original_open = open

    def recording_open(*args: Any, **kwargs: Any) -> Any:
        """Record successful source-open attempts and preserve normal behavior."""

        file_obj = original_open(*args, **kwargs)
        opened.append(args[0] if args else None)
        return file_obj

    monkeypatch.setattr("builtins.open", recording_open)
    try:
        report = patch_detached_references(policy="legacy")
    finally:
        sys.modules.pop(module_name, None)
    assert report.source_files_opened == len(opened)
    assert str(source_path) in opened


def test_module_identity_replacement_under_same_key_is_scanned() -> None:
    """Replacing a sys.modules object under an old key creates a new candidate."""

    raw = torch.relu
    name = "_tl_scoped_module_identity_replacement"
    first = types.ModuleType(name)
    sys.modules[name] = first
    try:
        wrap_torch(patch_policy="scoped")
        second = types.ModuleType(name)
        second.op = raw
        sys.modules[name] = second
        report = patch_detached_references(policy="scoped")
        assert report.module_identities_scanned >= 1
        assert second.op is _state._orig_to_decorated[id(raw)]
    finally:
        sys.modules.pop(name, None)


def test_nonweakrefable_module_identity_uses_conservative_fallback() -> None:
    """CFFI-style module identities need not support weak references."""

    class NonWeakOwner:
        """Minimal object with no weak-reference slot."""

        __slots__ = ()

    owner = NonWeakOwner()
    _remember_crawled_module_identity(owner)  # type: ignore[arg-type]
    assert _state._crawled_module_identities[id(owner)]() is owner


def test_dead_positive_module_identity_is_pruned() -> None:
    """Weak positive candidates cannot leave an id-reuse authorization behind."""

    name = "_tl_scoped_dead_positive"
    module = types.ModuleType(name)
    module_id = id(module)
    _remember_positive_module(module)
    del module
    gc.collect()
    hot_ids = _scoped_hot_module_ids(None, ())
    assert module_id not in hot_ids
    assert module_id not in _state._detached_positive_module_ids


def test_ledger_restores_owned_slot_and_preserves_user_reassignment() -> None:
    """Unwrap uses an identity three-way merge rather than rediscovery."""

    raw = torch.relu
    restore_name = "_tl_scoped_ledger_restore"
    preserve_name = "_tl_scoped_ledger_preserve"
    restore_module = types.ModuleType(restore_name)
    preserve_module = types.ModuleType(preserve_name)
    restore_module.op = raw
    preserve_module.op = raw
    sys.modules[restore_name] = restore_module
    sys.modules[preserve_name] = preserve_module

    def replacement(value: Any) -> Any:
        """Return a user-owned replacement value unchanged."""

        return value

    try:
        wrap_torch(patch_policy="scoped")
        assert restore_module.op is _state._orig_to_decorated[id(raw)]
        preserve_module.op = replacement
        unwrap_torch()
        assert restore_module.op is raw
        assert preserve_module.op is replacement
    finally:
        sys.modules.pop(restore_name, None)
        sys.modules.pop(preserve_name, None)


def test_model_callable_reassignment_is_patched_between_captures() -> None:
    """The bounded model scan runs on every capture, not only preparation."""

    raw = torch.relu

    class Model(nn.Module):
        """Apply a callable stored directly on the instance."""

        def __init__(self) -> None:
            """Store the pre-wrap callable."""

            super().__init__()
            self.op = raw

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the stored callable."""

            return self.op(x)

    model = Model()
    wrap_torch(patch_policy="scoped", escape_detector="shadow")
    first = tl.trace(model, torch.randn(3))
    model.op = raw
    second = tl.trace(model, torch.randn(3))
    assert first.escape_diagnostics == second.escape_diagnostics == []
    assert model.op is _state._orig_to_decorated[id(raw)]


@pytest.mark.parametrize("holder_kind", ["closure", "dict", "list", "instance"])
def test_builtin_holder_attacks_emit_shadow_report(holder_kind: str) -> None:
    """Ordinary-call builtin holders missed by crawling are reported at runtime."""

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
    assert isinstance(error, RuntimeError)
    assert len(reports) == 1
    assert "relu" in reports[0]
    assert hasattr(error, "partial_log")
    assert error.partial_log.trace.escape_diagnostics


def test_raw_python_functional_and_partial_emit_shadow_reports() -> None:
    """Raw Python code identity is visible directly and through partial."""

    raw = torch.nn.functional.softsign
    invocations = (raw, functools.partial(raw))
    for invoke in invocations:

        class Model(nn.Module):
            """Invoke a raw Python functional holder."""

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Call the hidden raw Python functional."""

                return invoke(x)

        error, reports = _run_shadow_capture(Model())
        assert error is None or isinstance(error, RuntimeError)
        assert len(reports) == 1
        assert "softsign" in reports[0]
        unwrap_torch()


def test_hidden_class_and_default_refs_emit_shadow_reports() -> None:
    """Unrelated local class/default storage is not silently accepted."""

    raw = torch.relu

    class Holder:
        """Class-attribute holder outside model provenance."""

        op = raw

    def uses_default(x: torch.Tensor, op: Callable[..., Any] = raw) -> torch.Tensor:
        """Invoke a hidden default-argument callable."""

        return op(x)

    for invoke in (lambda value: Holder.op(value), uses_default):

        class Model(nn.Module):
            """Invoke one unrelated hidden helper."""

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Delegate to the helper."""

                return invoke(x)

        error, reports = _run_shadow_capture(Model())
        assert isinstance(error, RuntimeError)
        assert len(reports) == 1
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
    wrap_torch(patch_policy="scoped", escape_detector="shadow")

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

    wrap_torch(patch_policy="scoped", escape_detector="shadow")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(Model(), torch.randn(3))
    assert _gap_warnings(caught) == []
    assert trace.escape_diagnostics == []
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "scoped_dispatch_witness_not_enabled"


def test_full_policy_does_not_claim_closure_completeness() -> None:
    """Even full crawling reports an executed closure escape in shadow mode."""

    raw = torch.relu

    class Model(nn.Module):
        """Invoke a closure-held raw callable."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Call the closure."""

            return raw(x)

    wrap_torch(patch_policy="full", escape_detector="shadow")
    with pytest.warns(TorchLensCaptureGapWarning, match="relu"):
        with pytest.raises(RuntimeError):
            tl.trace(Model(), torch.randn(3))


def test_clean_composites_and_descriptor_wrappers_do_not_convict() -> None:
    """Ordinary Torch composites and legitimate Tensor methods do not convict."""

    class Model(nn.Module):
        """Exercise Python composite, getset, method, and print wrapper paths."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run representative legitimate wrapper-to-original edges."""

            value = torch.nn.functional.softsign(x).add(1)
            _ = repr(value)
            return value.real

    wrap_torch(patch_policy="scoped", escape_detector="shadow")
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
@pytest.mark.parametrize("return_indices", [False, True], ids=["values", "indices"])
def test_boolean_dispatch_pooling_family_does_not_convict(
    function_name: str,
    input_shape: tuple[int, ...],
    kwargs: dict[str, Any],
    return_indices: bool,
) -> None:
    """Pre-wrap boolean-dispatch closure branches are audited clean composites."""

    class Model(nn.Module):
        """Invoke one live functional pooling dispatcher."""

        def forward(self, x: torch.Tensor) -> Any:
            """Run the selected boolean-dispatch branch."""

            pool = getattr(torch.nn.functional, function_name)
            return pool(x, return_indices=return_indices, **kwargs)

    wrap_torch(patch_policy="scoped", escape_detector="shadow")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(Model(), torch.randn(input_shape))
    assert trace.escape_diagnostics == []
    assert _gap_warnings(caught) == []


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

    wrap_torch(patch_policy="scoped", escape_detector="shadow")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(Model(), torch.randn(3))
    assert trace.escape_diagnostics == []
    assert _gap_warnings(caught) == []


def test_synchronous_dataloader_callback_is_in_owner_thread_domain() -> None:
    """A num_workers=0 callback escape is reported inside forward."""

    raw = torch.relu

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

    wrap_torch(patch_policy="scoped", escape_detector="shadow")
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

    wrap_torch(patch_policy="scoped", escape_detector="shadow")
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


def test_record_fastlog_uses_same_guard_and_backward_boundary_is_explicit() -> None:
    """Public record is armed; deferred backward is explicitly not armed."""

    class Model(nn.Module):
        """Small differentiable model for record/backward coverage."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return one differentiable represented operation."""

            return torch.relu(x)

    wrap_torch(patch_policy="scoped", escape_detector="shadow")
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
    wrap_torch(patch_policy="scoped", escape_detector="shadow")
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


def test_default_detector_is_off_and_scoped_trace_is_unverified() -> None:
    """The callable detector remains opt-in throughout the shadow rollout."""

    class Model(nn.Module):
        """Simple represented model."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one live torch lookup."""

            return torch.relu(x)

    wrap_torch(patch_policy="scoped")
    trace = tl.trace(Model(), torch.randn(3))
    assert trace.escape_detector_mode == "off"
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "scoped_dispatch_witness_not_enabled"


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
    wrap_torch(patch_policy="legacy")
    legacy = tl.trace(model, inputs)
    unwrap_torch()
    wrap_torch(patch_policy="scoped")
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
    wrap_torch(patch_policy="legacy")
    legacy = tl.trace(model_type(), inputs)
    unwrap_torch()
    wrap_torch(patch_policy="scoped")
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

    wrap_torch(patch_policy="scoped", escape_detector="shadow")
    trace = tl.trace(Model(), torch.randn(3), layers_to_save=["relu"])
    assert trace.capture_guard_passes
    assert all(
        item["owner_thread_id"] == trace.capture_owner_thread_id
        for item in trace.capture_guard_passes
    )
