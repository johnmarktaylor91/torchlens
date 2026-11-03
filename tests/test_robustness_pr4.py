"""Robustness sprint PR 4 — opaque-wrapper guards + limitations doc.

Covers:
    - ``torch.compile`` / ``torch.jit.script`` / ``torch.jit.trace`` /
      ``torch.export.ExportedProgram`` models raise a clear error up front,
      rather than running an empty or misleading forward pass.
    - ``docs/LIMITATIONS.md`` exists, is referenced from ``README.md``, and
      is discoverable from the repo root.
"""

from __future__ import annotations

import warnings
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _visualization
from torchlens._capture_state_helpers import (
    reset_compiled_model_unwrap_warning_state,
    unwrap_compiled_submodules,
)
from torchlens._utils._torch_compat import get_dynamo_optimized_module_type


class _Tiny(nn.Module):
    """Two-layer model small enough to script / compile / export cheaply."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _user_funcs.py, _user_funcs_guards.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of report/_summary_ladder.py.
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
        metadata = tl.trace(
            torch.compile(_Tiny(), backend="eager"),
            input_tensor,
            save="none",
        )
    assert "_orig_mod" not in {module.address for module in metadata.modules.values()}
    assert_single_note(caught)

    reset_compiled_model_unwrap_warning_state()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered_summary = tl.summary(torch.compile(_Tiny(), backend="eager"), input_tensor)
    assert "_orig_mod" not in rendered_summary
    assert_single_note(caught)

    reset_compiled_model_unwrap_warning_state()
    graph_path = tmp_path / "compiled_graph"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _visualization.show_model_graph(
            torch.compile(_Tiny(), backend="eager"),
            input_tensor,
            visualization=tl.options.VisualizationOptions(
                container_path=str(graph_path), file_format="svg"
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

    log = tl.trace(_Tiny(), input_tensor, save="none")
    reset_compiled_model_unwrap_warning_state()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log.run(torch.compile(_Tiny(), backend="eager"), input_tensor)
    assert "_orig_mod" not in {module.address for module in log.modules.values()}
    assert_single_note(caught)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture_state_helpers.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture_state_helpers.py.
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


# ---------------------------------------------------------------------------
# torch.jit.script / torch.jit.trace
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_torch_jit_unwrap_suggestion_matches_reality() -> None:
    """Logging the un-scripted Python module still works after scripting."""
    model = _Tiny()
    _ = torch.jit.script(model)  # must not poison the original
    log = tl.trace(model, torch.randn(2, 4), save="none")
    assert len(log._layer_logs) > 0


# ---------------------------------------------------------------------------
# torch.export.ExportedProgram
# ---------------------------------------------------------------------------


def _torch_export_available() -> bool:
    try:
        from torch.export import ExportedProgram, export  # noqa: F401
    except ImportError:
        return False
    return True


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/prepare/door.py.
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
        tl.trace(exported, torch.randn(2, 4), save="none")


# ---------------------------------------------------------------------------
# Sanity: helper-level
# ---------------------------------------------------------------------------


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
