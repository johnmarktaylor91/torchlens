"""Robustness sprint PR 4 — opaque-wrapper guards + limitations doc.

Covers:
    - ``torch.compile`` / ``torch.jit.script`` / ``torch.jit.trace`` /
      ``torch.export.ExportedProgram`` models raise a clear error up front,
      rather than running an empty or misleading forward pass.
    - ``docs/LIMITATIONS.md`` exists, is referenced from ``README.md``, and
      is discoverable from the repo root.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_state_helpers import (
    reset_compiled_model_unwrap_warning_state,
    unwrap_compiled_submodules,
)
from torchlens.user_funcs import _reject_opaque_wrappers
from torchlens.utils._torch_compat import get_dynamo_optimized_module_type


class _Tiny(nn.Module):
    """Two-layer model small enough to script / compile / export cheaply."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


class _ParentWithChild(nn.Module):
    """Parent module that delegates part of forward to a child module."""

    def __init__(self, child: nn.Module) -> None:
        super().__init__()
        self.child = child
        self.out = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.child(x)))


class _ParentRaisesAfterChild(nn.Module):
    """Parent that raises after invoking its compiled child."""

    def __init__(self, child: nn.Module) -> None:
        super().__init__()
        self.child = child

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.child(x)
        raise RuntimeError("parent forward failure")


class _ChildRaises(nn.Module):
    """Child module that always raises during forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        raise RuntimeError("child forward failure")


class _WrapperWithOriginal(nn.Module):
    """Minimal compiled-wrapper stand-in used to test traversal restoration."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self._orig_mod = original


class _WrapperWithBrokenOriginalProbe(nn.Module):
    """Wrapper stand-in that fails while TorchLens probes ``_orig_mod``."""

    def __getattr__(self, name: str) -> object:
        if name == "_orig_mod":
            raise RuntimeError("broken _orig_mod probe")
        return super().__getattr__(name)


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------


def _torch_compile_available() -> bool:
    """Return whether this torch runtime can create Dynamo OptimizedModule wrappers."""

    return get_dynamo_optimized_module_type() is not None and hasattr(torch, "compile")


@pytest.fixture(autouse=True)
def _reset_dynamo_after_test() -> Generator[None, None, None]:
    """Reset Dynamo and TorchLens unwrap warning state around each test."""

    reset_compiled_model_unwrap_warning_state()
    yield
    reset_compiled_model_unwrap_warning_state()
    dynamo = getattr(torch, "_dynamo", None)
    reset = getattr(dynamo, "reset", None)
    if callable(reset):
        reset()


@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile not available")
def test_torch_compile_top_level_unwrap_matches_eager_trace() -> None:
    """A top-level ``torch.compile`` wrapper should trace like its eager source."""
    model = _Tiny()
    eager_twin = copy.deepcopy(model)
    compiled = torch.compile(model, backend="eager")
    input_tensor = torch.randn(2, 4)

    reset_compiled_model_unwrap_warning_state()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        compiled_trace = tl.trace(
            compiled, input_tensor, capture=tl.options.CaptureOptions(layers_to_save="none")
        )
    eager_trace = tl.trace(
        eager_twin, input_tensor, capture=tl.options.CaptureOptions(layers_to_save="none")
    )

    assert [op.layer_label for op in compiled_trace.layer_list] == [
        op.layer_label for op in eager_trace.layer_list
    ]
    assert len(compiled_trace.modules) == len(eager_trace.modules)


@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile not available")
def test_double_torch_compile_callable_names_eager_module_remedy() -> None:
    """A double-compiled plain callable gets a specific eager-module rejection."""
    model = _Tiny()
    compiled_once = torch.compile(model, backend="eager")
    compiled_twice = torch.compile(compiled_once, backend="eager")
    if isinstance(compiled_twice, nn.Module):
        pytest.skip("this torch runtime keeps double compile as an nn.Module")

    with pytest.raises(ValueError, match=r"torch\.compile.*original eager nn\.Module"):
        tl.trace(compiled_twice, torch.randn(2, 4))  # type: ignore[arg-type]


@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile not available")
def test_torch_compile_unwrap_note_emits_once_across_two_traces() -> None:
    """Compiled-model eager unwrapping should emit one process-local note."""
    first = torch.compile(_Tiny(), backend="eager")
    second = torch.compile(_Tiny(), backend="eager")
    input_tensor = torch.randn(2, 4)

    reset_compiled_model_unwrap_warning_state()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(first, input_tensor, capture=tl.options.CaptureOptions(layers_to_save="none"))
        tl.trace(second, input_tensor, capture=tl.options.CaptureOptions(layers_to_save="none"))

    unwrap_warnings = [
        warning
        for warning in caught
        if "compiled model detected; tracing the eager source module" in str(warning.message)
    ]
    assert len(unwrap_warnings) == 1


@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile not available")
def test_torch_compile_top_level_wrapper_remains_callable_after_trace() -> None:
    """Tracing a compiled root should leave the user's wrapper object in place."""
    compiled = torch.compile(_Tiny(), backend="eager")
    optimized_module_type = get_dynamo_optimized_module_type()
    input_tensor = torch.randn(2, 4)

    reset_compiled_model_unwrap_warning_state()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tl.trace(compiled, input_tensor, capture=tl.options.CaptureOptions(layers_to_save="none"))

    assert optimized_module_type is not None
    assert isinstance(compiled, optimized_module_type)
    assert compiled(input_tensor).shape == (2, 4)


@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile not available")
def test_torch_compile_nested_submodule_traces_and_restores_parent() -> None:
    """A compiled child should trace through its eager source and be restored afterward."""
    child = torch.compile(nn.Linear(4, 4), backend="eager")
    parent = _ParentWithChild(child)
    eager_parent = _ParentWithChild(copy.deepcopy(child._orig_mod))
    eager_parent.out.load_state_dict(parent.out.state_dict())
    optimized_module_type = get_dynamo_optimized_module_type()
    input_tensor = torch.randn(2, 4)

    reset_compiled_model_unwrap_warning_state()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        nested_trace = tl.trace(
            parent, input_tensor, capture=tl.options.CaptureOptions(layers_to_save="none")
        )
    eager_trace = tl.trace(
        eager_parent, input_tensor, capture=tl.options.CaptureOptions(layers_to_save="none")
    )

    assert optimized_module_type is not None
    assert parent.child is child
    assert isinstance(parent.child, optimized_module_type)
    assert parent(input_tensor).shape == (2, 4)
    assert [op.layer_label for op in nested_trace.layer_list] == [
        op.layer_label for op in eager_trace.layer_list
    ]
    assert len(nested_trace.modules) == len(eager_trace.modules)


@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile not available")
def test_compiled_models_unwrap_at_all_public_entry_points(tmp_path: Path) -> None:
    """Public eager-capture entry points unwrap compiled root models consistently."""
    input_tensor = torch.randn(2, 4)

    def assert_single_note(caught: list[warnings.WarningMessage]) -> None:
        """Assert exactly one eager-source compiled-model note was emitted."""
        notes = [
            warning
            for warning in caught
            if "compiled model detected; tracing the eager source module" in str(warning.message)
        ]
        assert len(notes) == 1

    reset_compiled_model_unwrap_warning_state()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metadata = tl.io.log_model_metadata(torch.compile(_Tiny(), backend="eager"), input_tensor)
    assert "_orig_mod" not in {module.address for module in metadata.modules.values()}
    assert_single_note(caught)

    reset_compiled_model_unwrap_warning_state()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered_summary = tl.visualization.summary(
            torch.compile(_Tiny(), backend="eager"), input_tensor
        )
    assert "_orig_mod" not in rendered_summary
    assert_single_note(caught)

    reset_compiled_model_unwrap_warning_state()
    graph_path = tmp_path / "compiled_graph"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.visualization.show_model_graph(
            torch.compile(_Tiny(), backend="eager"),
            input_tensor,
            visualization=tl.options.VisualizationOptions(
                container_path=str(graph_path), save_only=True, file_format="svg"
            ),
        )
    assert "_orig_mod" not in graph_path.with_suffix(".svg").read_text()
    assert_single_note(caught)

    reset_compiled_model_unwrap_warning_state()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert tl.validation.validate_forward_pass(
            torch.compile(_Tiny(), backend="eager"), input_tensor
        )
    assert_single_note(caught)

    log = tl.trace(_Tiny(), input_tensor, capture=tl.options.CaptureOptions(layers_to_save="none"))
    reset_compiled_model_unwrap_warning_state()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log.run(torch.compile(_Tiny(), backend="eager"), input_tensor)
    assert "_orig_mod" not in {module.address for module in log.modules.values()}
    assert_single_note(caught)


def test_compiled_submodule_traversal_failure_restores_earlier_swaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed compiled-child probe restores all prior traversal swaps."""
    import torchlens._capture_state_helpers as helpers

    good_wrapper = _WrapperWithOriginal(nn.Linear(4, 4))
    parent = nn.Module()
    parent.good = good_wrapper
    parent.bad = _WrapperWithBrokenOriginalProbe()
    monkeypatch.setattr(helpers, "get_dynamo_optimized_module_type", lambda: nn.Module)

    with (
        pytest.raises(RuntimeError, match="broken _orig_mod probe"),
        unwrap_compiled_submodules(parent),
    ):
        pass

    assert parent.good is good_wrapper


class _HostileModules(dict):
    """``_modules`` stand-in that raises on one keyed restore once armed."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.armed_key: str | None = None

    def __setitem__(self, key: str, value: object) -> None:
        if self.armed_key == key:
            raise RuntimeError("hostile _modules restore")
        super().__setitem__(key, value)


def test_compiled_submodule_unwind_completes_past_raising_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """b3-sol sibling of the fixed 6896e8a9 unwind cluster: one raising restore
    in the reversed compiled-submodule unwind used to skip every REMAINING
    swap, stranding those children in their eager form for the life of the
    process. The unwind must complete and re-raise the first failure."""
    import torchlens._capture_state_helpers as helpers

    wrapper_a = _WrapperWithOriginal(nn.Linear(4, 4))
    wrapper_b = _WrapperWithOriginal(nn.Linear(4, 4))
    parent = nn.Module()
    parent.child_a = wrapper_a
    parent.child_b = wrapper_b
    hostile = _HostileModules(parent._modules)
    object.__setattr__(parent, "_modules", hostile)
    monkeypatch.setattr(helpers, "get_dynamo_optimized_module_type", lambda: _WrapperWithOriginal)

    with (
        pytest.raises(RuntimeError, match="hostile _modules restore"),
        unwrap_compiled_submodules(parent),
    ):
        assert parent._modules["child_a"] is wrapper_a._orig_mod
        assert parent._modules["child_b"] is wrapper_b._orig_mod
        hostile.armed_key = "child_b"

    # child_b's restore raised (it honestly stays eager); child_a's restore
    # runs anyway instead of being skipped by the propagating failure.
    assert parent._modules["child_a"] is wrapper_a


@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile not available")
@pytest.mark.parametrize("parent_raises", [False, True])
def test_compiled_submodule_restored_after_forward_exception(parent_raises: bool) -> None:
    """Compiled child identity survives child- and parent-originated forward failures."""
    child_source: nn.Module = nn.Linear(4, 4) if parent_raises else _ChildRaises()
    compiled_child = torch.compile(child_source, backend="eager")
    model: nn.Module
    if parent_raises:
        model = _ParentRaisesAfterChild(compiled_child)
    else:
        model = _ParentWithChild(compiled_child)

    with pytest.raises(RuntimeError, match="forward failure"):
        tl.trace(model, torch.randn(2, 4), capture=tl.options.CaptureOptions(layers_to_save="none"))

    assert model.child is compiled_child


# ---------------------------------------------------------------------------
# torch.jit.script / torch.jit.trace
# ---------------------------------------------------------------------------


def test_torch_jit_script_raises_at_entry() -> None:
    """A ``torch.jit.script``'d model must raise up front."""
    model = _Tiny()
    scripted = torch.jit.script(model)
    assert isinstance(scripted, torch.jit.ScriptModule)

    with pytest.raises(RuntimeError, match="ScriptModule"):
        tl.trace(
            scripted, torch.randn(2, 4), capture=tl.options.CaptureOptions(layers_to_save="none")
        )


def test_torch_jit_trace_raises_at_entry() -> None:
    """A ``torch.jit.trace``'d model is also a ScriptModule and must raise."""
    model = _Tiny()
    traced = torch.jit.trace(model, torch.randn(2, 4))
    assert isinstance(traced, torch.jit.ScriptModule)

    with pytest.raises(RuntimeError, match="ScriptModule"):
        tl.trace(
            traced, torch.randn(2, 4), capture=tl.options.CaptureOptions(layers_to_save="none")
        )


def test_torch_jit_unwrap_suggestion_matches_reality() -> None:
    """Logging the un-scripted Python module still works after scripting."""
    model = _Tiny()
    _ = torch.jit.script(model)  # must not poison the original
    log = tl.trace(
        model, torch.randn(2, 4), capture=tl.options.CaptureOptions(layers_to_save="none")
    )
    assert len(log.layer_logs) > 0


# ---------------------------------------------------------------------------
# torch.export.ExportedProgram
# ---------------------------------------------------------------------------


def _torch_export_available() -> bool:
    try:
        from torch.export import ExportedProgram, export  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _torch_export_available(), reason="torch.export not available")
def test_torch_export_exported_program_raises_at_entry() -> None:
    """A ``torch.export``'d model is not a callable ``nn.Module`` — must raise."""
    from torch.export import export

    model = _Tiny()
    example = (torch.randn(2, 4),)
    exported = export(model, example)

    # The entry guard rejects any non-``nn.Module`` input with the documented
    # typed refusal (an ``ExportedProgram`` is not an ``nn.Module``); the old
    # accidental AttributeError leak this test used to match was itself a bug.
    with pytest.raises(ValueError, match="Unsupported model type"):
        tl.trace(
            exported, torch.randn(2, 4), capture=tl.options.CaptureOptions(layers_to_save="none")
        )


# ---------------------------------------------------------------------------
# Sanity: helper-level
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_reject_opaque_wrappers_clean_model_is_noop() -> None:
    """A bare nn.Module must pass through ``_reject_opaque_wrappers`` silently."""
    model = _Tiny()
    _reject_opaque_wrappers(model)  # should not raise


def test_reject_opaque_wrappers_script_module_raises_directly() -> None:
    """The helper raises for ScriptModule without needing the full entry point."""
    scripted = torch.jit.script(_Tiny())
    with pytest.raises(RuntimeError, match="ScriptModule"):
        _reject_opaque_wrappers(scripted)


# ---------------------------------------------------------------------------
# Limitations documentation discoverability
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_limitations_doc_exists() -> None:
    """The limitations pages ship with the repo.

    The canonical catalog moved to docs/reference/limitations.md (grind/f1-docs
    reconcile); docs/LIMITATIONS.md remains as a compatibility redirect stub so
    existing links do not break. Both must exist, and the canonical page must
    carry the real content.
    """
    stub = _repo_root() / "docs" / "LIMITATIONS.md"
    assert stub.is_file(), f"Expected docs/LIMITATIONS.md redirect stub at {stub}"
    assert "reference/limitations.md" in stub.read_text(), (
        "docs/LIMITATIONS.md must point readers at the canonical page"
    )
    canonical = _repo_root() / "docs" / "reference" / "limitations.md"
    assert canonical.is_file(), f"Expected canonical doc at {canonical}"
    content = canonical.read_text()
    assert len(content) > 500, "docs/reference/limitations.md looks suspiciously short"


def test_readme_links_to_limitations_doc() -> None:
    """README must link to the limitations doc so users can find it."""
    readme = (_repo_root() / "README.md").read_text()
    assert "docs/LIMITATIONS.md" in readme, (
        "README.md should link to docs/LIMITATIONS.md so users can discover "
        "supported / unsupported contexts."
    )


def test_limitations_doc_covers_key_contexts() -> None:
    """Every context with a runtime guard must be explained in the doc.

    This is the doc-accuracy regression: if we add a new guard we must
    remember to document it. Conversely, if we remove a guard without
    updating this list, the test catches the stale doc.
    """
    content = (_repo_root() / "docs" / "reference" / "limitations.md").read_text().lower()
    must_mention = [
        "torch.compile",
        "torch.jit",
        "torch.export",
        "fullyshardeddataparallel",
        "meta tensor",
        "sparse tensor",
        "symbolic",
        "quantized",
        "vmap",
    ]
    for phrase in must_mention:
        assert phrase in content, (
            f"docs/reference/limitations.md should mention '{phrase}'. "
            f"Missing phrase suggests a stale or incomplete limitations doc."
        )
