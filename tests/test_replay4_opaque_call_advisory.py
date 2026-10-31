"""The opaque-call advisory: a witness gap whose call-site line calls a TorchScript
function (or hands a tensor out through DLPack) is named with its line and the spelling
that captures it, and no verdict field moves.

A scripted call is normally one ``jitscript`` boundary op with no gap
(``tests/test_torchscript_boundary.py``); the scripted helpers here write into a tensor
argument in place, which the boundary leaves out, so they keep the gap the advisory names.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import _opaque_call_advisory as advisory
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.errors import TorchLensWarning

_VERDICT_FIELDS = (
    "capture_verified",
    "capture_verification_reason",
    "completeness_witness_verified",
    "completeness_witness_event_count",
    "completeness_witness_accounted_count",
    "completeness_witness_expected_opaque_count",
    "completeness_witness_unaccounted_count",
)


@torch.jit.script
def scripted_gate(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Scripted helper that writes into its argument, so no boundary op covers it."""

    x.mul_((x >= threshold).to(x.dtype))
    return x.sin()


class _ScriptedCaller(nn.Module):
    """Eager forward that calls a module-level TorchScript function."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x).clone()
        return scripted_gate(h, 0.1) + h


class _EagerTwin(nn.Module):
    """The same computation written eagerly."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return (h >= 0.1).to(h.dtype) * h.sin() + h


class _StaticScriptedCaller(nn.Module):
    """Calls a scripted static method through ``self``, as spiking-neuron resets do."""

    @staticmethod
    @torch.jit.script
    def jit_reset(v: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        v.mul_(1.0 - spike)
        return v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spike = torch.sigmoid(x)
        return self.jit_reset(x.clone(), spike) + 1


@pytest.fixture(autouse=True)
def _witness_armed() -> Iterator[None]:
    """Arm the completeness witness; the conftest autouse reset clears the latch."""

    saved_escape = _state._escape_detector_mode
    saved_witness = _state._completeness_witness_mode
    unwrap_torch()
    wrap_torch(completeness_witness=True)
    yield
    unwrap_torch()
    wrap_torch(escape_detector=saved_escape, completeness_witness=saved_witness)


def _capture(model: nn.Module) -> tuple[Any, list[warnings.WarningMessage]]:
    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, torch.randn(2, 4))
    return trace, list(caught)


def _advisories(caught: list[warnings.WarningMessage]) -> list[TorchLensWarning]:
    found = []
    for item in caught:
        message = item.message
        if isinstance(message, TorchLensWarning) and message.fields.get("code") in {
            advisory.SCRIPTED_CODE,
            advisory.DLPACK_CODE,
        }:
            found.append(message)
    return found


def _line_of(fragment: str) -> int:
    lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    return next(number for number, text in enumerate(lines, start=1) if fragment in text)


def test_scripted_call_is_named_with_its_line_and_the_jit_hint() -> None:
    _trace, caught = _capture(_ScriptedCaller())

    (warning,) = _advisories(caught)
    call_line = _line_of("return scripted_gate(h, 0.1) + h")
    assert warning.fields["code"] == "scripted_function_not_captured"
    assert warning.fields["scripted_function"] == "scripted_gate"
    assert warning.file_path == __file__ and warning.line_no == call_line
    assert warning.fields["dispatch_count"] >= 1
    assert str(warning.fields["defined_at"]).startswith(f"{__file__}:")
    text = str(warning)
    assert f"{__file__}:{call_line}" in text and "`scripted_gate`" in text
    assert "PYTORCH_JIT=0" in warning.fields["remedy"]


def test_eager_twin_emits_nothing() -> None:
    trace, caught = _capture(_EagerTwin())

    assert _advisories(caught) == []
    assert trace.provenance.verified is True


def test_scripted_static_method_through_self_is_resolved() -> None:
    _trace, caught = _capture(_StaticScriptedCaller())

    (warning,) = _advisories(caught)
    assert warning.fields["scripted_function"] == "jit_reset"
    assert warning.line_no == _line_of("return self.jit_reset(x.clone(), spike) + 1")


@pytest.mark.parametrize("model_class", [_ScriptedCaller, _EagerTwin])
def test_verdict_fields_equal_with_and_without_the_advisory(
    model_class: type[nn.Module], monkeypatch: pytest.MonkeyPatch
) -> None:
    # TorchScript's profiling executor runs a scripted function's profiling graph
    # on its first call and its optimized graph afterwards; the two dispatch
    # different aten ops (7 against 10 witness events for scripted_gate). Settle
    # the executor with plain forwards first, so both captures see the same graph
    # whichever test reached the scripted function first in this process.
    for _ in range(2):
        model_class()(torch.randn(2, 4))
    with_advisory, _ = _capture(model_class())
    monkeypatch.setattr(advisory, "advise_opaque_calls", lambda diagnostics: None)
    without_advisory, _ = _capture(model_class())

    for field in _VERDICT_FIELDS:
        assert getattr(with_advisory, field) == getattr(without_advisory, field), field
    assert len(with_advisory.completeness_diagnostics) == len(
        without_advisory.completeness_diagnostics
    )


def test_a_capture_without_a_gap_never_reaches_the_advisory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    real = advisory.advise_opaque_calls
    monkeypatch.setattr(
        advisory,
        "advise_opaque_calls",
        lambda diagnostics: calls.append(1) or real(diagnostics),
    )

    _capture(_EagerTwin())
    assert calls == []
    _capture(_ScriptedCaller())
    assert calls  # once per forward pass that left a gap


def test_advisory_fires_once_per_session() -> None:
    _, first = _capture(_ScriptedCaller())
    _, second = _capture(_ScriptedCaller())

    assert len(_advisories(first)) == 1
    assert _advisories(second) == []
    assert advisory.SCRIPTED_CODE in advisory._WARNED_CODES


def test_advisory_warns_again_after_the_autouse_reset(request: pytest.FixtureRequest) -> None:
    specs = {(module, name) for module, name, _ in request.config._tl_warn_once_sentinel_specs}
    assert ("torchlens.backends.torch._opaque_call_advisory", "_WARNED_CODES") in specs
    assert set() == advisory._WARNED_CODES  # the previous test set it; the reset cleared it

    _, caught = _capture(_ScriptedCaller())
    assert len(_advisories(caught)) == 1


def test_global_warning_filter_silences_it() -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=TorchLensWarning)
        tl.trace(_ScriptedCaller(), torch.randn(2, 4))

    assert _advisories(list(caught)) == []


def _dlpack_line_fixture(x: torch.Tensor) -> object:
    capsule = torch.utils.dlpack.to_dlpack(x)  # dlpack call site
    return capsule


def test_dlpack_call_site_gets_the_advisory_without_the_jit_hint() -> None:
    line = _line_of("# dlpack call site")
    diagnostics = [
        {"reason": "owner_not_captured", "file": __file__, "line": line, "function": "f"},
        {"reason": "owner_not_captured", "file": __file__, "line": line, "function": "f"},
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        advisory.advise_opaque_calls(diagnostics)

    (warning,) = _advisories(list(caught))
    assert warning.fields["code"] == "dlpack_call_not_captured"
    assert warning.line_no == line and warning.fields["dispatch_count"] == 2
    assert "PYTORCH_JIT" not in str(warning)


def test_lines_with_no_opaque_call_get_no_advisory() -> None:
    line = _line_of("h = self.lin(x)")
    diagnostics = [{"reason": "unowned_dispatch", "file": __file__, "line": line, "function": "f"}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        advisory.advise_opaque_calls(diagnostics)

    assert _advisories(list(caught)) == []


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import json, torch
    from torch import nn
    import torchlens as tl
    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    @torch.jit.script
    def scripted_gate(x: torch.Tensor, threshold: float) -> torch.Tensor:
        return (x >= threshold).to(x.dtype) * x.sin()

    @torch.jit.script
    def scripted_write(x: torch.Tensor) -> torch.Tensor:
        x.mul_(2.0)
        return x.sin()

    class M(nn.Module):
        def forward(self, x):
            return scripted_gate(x, 0.1) + scripted_write(x.clone())

    unwrap_torch()
    wrap_torch(completeness_witness=True)
    trace = tl.trace(M(), torch.randn(2, 4))
    print(json.dumps({"verified": trace.provenance.verified,
                      "funcs": [layer.func_name for layer in trace.layers]}))
    """
)


@pytest.mark.heavy
@pytest.mark.parametrize("jit_env", ["0", None])
def test_pytorch_jit_0_spelling_captures_the_scripted_body(
    jit_env: str | None, tmp_path: Path
) -> None:
    script = tmp_path / "scripted_model.py"
    script.write_text(_SUBPROCESS_SCRIPT, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if key != "PYTORCH_JIT"}
    if jit_env is not None:
        env["PYTORCH_JIT"] = jit_env
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    if jit_env == "0":
        assert report["verified"] is True
        assert "sin" in report["funcs"] and "__ge__" in report["funcs"]
        assert "jitscript" not in report["funcs"]
    else:
        # The functional helper is one boundary op; the in-place one stays the gap.
        assert report["verified"] is False
        assert report["funcs"].count("jitscript") == 1
        assert "sin" not in report["funcs"]
        assert "PYTORCH_JIT=0" in result.stderr
