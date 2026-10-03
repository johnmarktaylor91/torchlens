"""grind-r5 corroborated LOWs: belt sweep watermark + rescue warning routing.

* The belt's sweep watermark was a ``len(sys.modules)`` EQUALITY test, so a
  same-length module-table mutation (del one key + insert another) skipped
  the sweep and left a protocol-invisible stale
  ``from_numpy``/``frombuffer``/``as_subclass`` reference unpatched -- the
  zero-signal class the belt exists to close (b3 opus+sol, 2 rounds).
* The fixwave-4 rescue-ineligible escape disclosure landed as a bare
  ``UserWarning``, invisible to users routing torchlens advisories via
  ``filterwarnings(category=TorchLensWarning)`` (b8 sol).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._errors import TorchLensWarning
from torchlens.backends.torch.belt import sweep_stale_belt_references
from torchlens.backends.torch.wrappers import wrap_torch


def test_same_length_sys_modules_mutation_is_still_swept() -> None:
    """A del+insert mutation keeping len(sys.modules) constant must not skip
    the belt sweep."""

    wrap_torch()
    tl.trace(nn.Linear(2, 2), torch.randn(1, 2))  # belt swept at least once
    original_from_numpy = _state._decorated_to_orig.get(id(torch.from_numpy))
    if original_from_numpy is None:
        pytest.skip("torch.from_numpy is not belt/wrapper-covered on this build")

    placeholder_name = "tl_test_belt_placeholder_module"
    victim_name = "tl_test_belt_victim_module"
    placeholder = types.ModuleType(placeholder_name)
    victim = types.ModuleType(victim_name)
    victim.from_numpy = original_from_numpy  # pristine protocol-invisible ref
    sys.modules[placeholder_name] = placeholder
    try:
        sweep_stale_belt_references()  # baseline sweep including placeholder
        del sys.modules[placeholder_name]  # same-length mutation:
        sys.modules[victim_name] = victim  # -1 key, +1 key
        sweep_stale_belt_references()
        assert victim.from_numpy is not original_from_numpy, (
            "the same-length sys.modules mutation skipped the belt sweep; "
            "a protocol-invisible stale reference stayed unpatched"
        )
    finally:
        sys.modules.pop(placeholder_name, None)
        sys.modules.pop(victim_name, None)


class _StubEscapeTrace:
    """Minimal trace stand-in carrying a live escape signal."""

    def __init__(self) -> None:
        self.capture_verified: Any = None
        self.capture_verification_reason: Any = None
        self.rescue_rerun: Any = None
        self.escape_diagnostics = [{"kind": "provenance_warning", "op": "relu"}]
        self.layer_labels = ["relu_1_1"]


def test_rescue_ineligible_disclosure_uses_the_torchlens_category() -> None:
    """The action-demanding ineligibility advisory must be routable."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    def run_capture() -> Any:
        return _StubEscapeTrace()

    with pytest.warns(TorchLensWarning, match="skipped the rescue"):
        capture_with_rescue(run_capture, eligible=False)
