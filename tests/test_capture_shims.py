"""Absence tests for the deleted capture/IR compatibility shims.

The kernel/ledger/projector layer and the ``ModuleEvent``/``BufferEvent``
event kinds were deleted in the backend migration; the compatibility shims
that kept their imports resolving were removed in the species-1 cleanup.
Contract now: the historical spellings fail loudly and typed
(``ModuleNotFoundError`` / ``AttributeError``), and the live replacements
stay importable.
"""

import importlib

import pytest


def test_kernel_shim_module_is_gone() -> None:
    """``torchlens._capture.kernel`` no longer exists as an import path."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("torchlens._capture.kernel")


def test_trace_projector_shim_is_gone() -> None:
    """``TraceProjector`` no longer resolves on ``torchlens._capture.projectors``."""
    from torchlens._capture import projectors

    with pytest.raises(AttributeError):
        projectors.TraceProjector  # noqa: B018
    assert not hasattr(projectors, "_DeprecatedTraceProjector")


def test_ir_event_shims_are_gone() -> None:
    """``ModuleEvent``/``BufferEvent`` no longer resolve on ``torchlens._ir``.

    The live replacements (ModuleEnterEvent/ModuleExitEvent for module
    containment, BufferWriteEvent for buffer capture) must stay importable.
    """
    import torchlens._ir as tl_ir

    for name in ("ModuleEvent", "BufferEvent"):
        with pytest.raises(AttributeError):
            getattr(tl_ir, name)
        assert name not in tl_ir.__all__
    from torchlens._ir.events import (  # noqa: F401
        BufferWriteEvent,
        ModuleEnterEvent,
        ModuleExitEvent,
    )

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("torchlens._ir._deprecated")
