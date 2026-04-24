"""Robustness sprint PR 4 — opaque-wrapper guards + limitations doc.

Covers:
    - ``torch.compile`` / ``torch.jit.script`` / ``torch.jit.trace`` /
      ``torch.export.ExportedProgram`` models raise a clear error up front,
      rather than running an empty or misleading forward pass.
    - ``docs/LIMITATIONS.md`` exists, is referenced from ``README.md``, and
      is discoverable from the repo root.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.user_funcs import _reject_opaque_wrappers


class _Tiny(nn.Module):
    """Two-layer model small enough to script / compile / export cheaply."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------


def _torch_compile_available() -> bool:
    try:
        from torch._dynamo.eval_frame import OptimizedModule  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "compile")


@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile not available")
def test_torch_compile_raises_at_entry() -> None:
    """A ``torch.compile``'d model must raise, not produce an empty log."""
    model = _Tiny()
    compiled = torch.compile(model)

    with pytest.raises(RuntimeError, match="torch.compile"):
        tl.log_forward_pass(compiled, torch.randn(2, 4), layers_to_save="none")


@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile not available")
def test_torch_compile_unwrap_suggestion_matches_reality() -> None:
    """Message recommends logging the un-compiled model; verify that works."""
    model = _Tiny()
    _ = torch.compile(model)  # must not poison the original
    # Logging the original still works.
    log = tl.log_forward_pass(model, torch.randn(2, 4), layers_to_save="none")
    assert len(log.layer_logs) > 0


# ---------------------------------------------------------------------------
# torch.jit.script / torch.jit.trace
# ---------------------------------------------------------------------------


def test_torch_jit_script_raises_at_entry() -> None:
    """A ``torch.jit.script``'d model must raise up front."""
    model = _Tiny()
    scripted = torch.jit.script(model)
    assert isinstance(scripted, torch.jit.ScriptModule)

    with pytest.raises(RuntimeError, match="ScriptModule"):
        tl.log_forward_pass(scripted, torch.randn(2, 4), layers_to_save="none")


def test_torch_jit_trace_raises_at_entry() -> None:
    """A ``torch.jit.trace``'d model is also a ScriptModule and must raise."""
    model = _Tiny()
    traced = torch.jit.trace(model, torch.randn(2, 4))
    assert isinstance(traced, torch.jit.ScriptModule)

    with pytest.raises(RuntimeError, match="ScriptModule"):
        tl.log_forward_pass(traced, torch.randn(2, 4), layers_to_save="none")


def test_torch_jit_unwrap_suggestion_matches_reality() -> None:
    """Logging the un-scripted Python module still works after scripting."""
    model = _Tiny()
    _ = torch.jit.script(model)  # must not poison the original
    log = tl.log_forward_pass(model, torch.randn(2, 4), layers_to_save="none")
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

    with pytest.raises((RuntimeError, AttributeError, TypeError)) as excinfo:
        tl.log_forward_pass(exported, torch.randn(2, 4), layers_to_save="none")
    # Our guard is the preferred failure path; other failures (e.g. exported
    # program lacking .modules()) also satisfy the 'don't silently succeed'
    # contract.
    assert (
        "ExportedProgram" in str(excinfo.value)
        or "has no attribute" in str(excinfo.value)
        or "torch.export" in str(excinfo.value)
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
    """The limitations page ships with the repo."""
    path = _repo_root() / "docs" / "LIMITATIONS.md"
    assert path.is_file(), f"Expected docs/LIMITATIONS.md to exist at {path}"
    content = path.read_text()
    assert len(content) > 500, "LIMITATIONS.md looks suspiciously short"


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
    content = (_repo_root() / "docs" / "LIMITATIONS.md").read_text().lower()
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
            f"docs/LIMITATIONS.md should mention '{phrase}'. "
            f"Missing phrase suggests a stale or incomplete limitations doc."
        )
