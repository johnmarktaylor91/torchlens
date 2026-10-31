"""Wrap-state-independent ground truth (R75-1, b9-fable round-2 probe).

Once torch is wrapped, the phase-0 ground-truth forward, the per-op replay,
and backward validation's stock-autograd pass all observed through the SAME
installed wrapper closures, so a TorchLens-induced numeric distortion (a
planted 0.1% ``tanh`` distortion in the wrapper layer) validated CLEAN in a
wrapped process and was caught only in a fresh one. The ground-truth
oracles now run with the wrappers removed (restored from the pre-decoration
originals ledger) and reinstalled afterwards.

The probe needs a process whose FIRST wrap installs the distorted wrapper,
so the red-capable cases run in a subprocess with a fresh interpreter.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.validation._pristine import pristine_torch_oracle

_PROBE_SCRIPT = textwrap.dedent(
    """
    import torch
    import torch.nn as nn
    import torchlens as tl
    import torchlens.backends.torch.wrappers as wrappers

    # Plant the b9-fable distortion BEFORE the first wrap: the wrapper built
    # for tanh silently multiplies its result by 1.001 using pristine
    # pre-wrap references. This models a wrapper-layer capture bug.
    pristine_mul = torch.mul
    real_decorator = wrappers.torch_func_decorator

    def planted_decorator(func, func_name, property_accessor=None):
        if func_name == "tanh":
            orig = func

            def distorted(*args, **kwargs):
                return pristine_mul(orig(*args, **kwargs), 1.001)

            distorted.__name__ = getattr(orig, "__name__", "tanh")
            distorted.__qualname__ = getattr(orig, "__qualname__", "tanh")
            return real_decorator(distorted, func_name, property_accessor)
        return real_decorator(func, func_name, property_accessor)

    wrappers.torch_func_decorator = planted_decorator

    class TanhModel(nn.Module):
        def forward(self, x):
            return torch.tanh(x) + 1.0

    model = TanhModel().eval()
    x = torch.randn(4, 8)

    # Step 1: run a capture first, so the process is in the WRAPPED state
    # (the probe's capture-then-validate arm -- the one that validated
    # clean pre-fix).
    trace = tl.trace(model, x.clone())

    # Step 2: validate the same model in the wrapped process.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verdict = tl.validate(model, x.clone(), scope="forward", validate_metadata=False)
    print(f"VERDICT={verdict}")
    """
)

_CLEAN_SCRIPT = _PROBE_SCRIPT.replace(
    "wrappers.torch_func_decorator = planted_decorator",
    "# no plant: positive control",
)


def _run_probe(script: str) -> str:
    """Run a probe script in a fresh interpreter and return its stdout."""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


@pytest.mark.heavy
def test_planted_wrapper_distortion_fails_validation_in_a_wrapped_process() -> None:
    """The b9-fable probe, now caught: wrapped-process validation FAILS.

    Red-capable: pre-fix this exact script printed VERDICT=True (the
    distorted tanh corrupted ground truth, capture, and replay
    identically), and only a fresh-process validation caught the bug.
    """

    stdout = _run_probe(_PROBE_SCRIPT)
    assert "VERDICT=False" in stdout, stdout


@pytest.mark.heavy
def test_clean_wrapped_process_validation_still_passes() -> None:
    """Positive control: no plant, same wrapped-process flow, verdict True."""

    stdout = _run_probe(_CLEAN_SCRIPT)
    assert "VERDICT=True" in stdout, stdout


def test_pristine_oracle_unwraps_and_restores_wrap_state() -> None:
    """Inside the context torch is pristine; after it, wrappers return."""

    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh()).eval()
    tl.trace(model, torch.randn(2, 4))
    assert _state._is_decorated
    wrapped_tanh = torch.tanh
    with pristine_torch_oracle() as pristine:
        assert pristine
        assert not _state._is_decorated
        assert torch.tanh is not wrapped_tanh
    assert _state._is_decorated
    # The reinstalled wrapper intercepts again (identity may differ; the
    # ledger keyed restore is what matters).
    assert id(torch.tanh) in _state._decorated_to_orig


def test_pristine_oracle_is_noop_when_never_wrapped() -> None:
    """A never-wrapped process yields pristine=True without touching state."""

    script = textwrap.dedent(
        """
        import torch
        from torchlens import _state
        from torchlens.validation._pristine import pristine_torch_oracle

        assert not _state._is_decorated
        before = torch.tanh
        with pristine_torch_oracle() as pristine:
            assert pristine
            assert torch.tanh is before
        assert not _state._is_decorated
        print("NOOP-OK")
        """
    )
    assert "NOOP-OK" in _run_probe(script)


def test_backward_validation_runs_pristine_stock_pass() -> None:
    """Backward validation's stock pass also settles under the pristine root.

    Functional pin: in a wrapped process, validate_backward_pass still
    passes on an honest model (the unwrap/rewrap cycle is transparent), and
    the wrappers are back afterwards.
    """

    from torchlens.validation.backward import validate_backward_pass

    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh()).eval()
    tl.trace(model, torch.randn(2, 4))
    assert _state._is_decorated
    assert validate_backward_pass(
        model,
        torch.randn(2, 4),
        seed=3,
        validate_metadata=False,
    )
    assert _state._is_decorated


class TestUnwrapLedgerIntegrity:
    """r4 census MED-1: ``_decorated_to_orig`` is the single shared root for
    capture's orig grab, the pristine restoration, AND sparse-runnable
    callable resolution. A ledger entry whose "orig" is itself a TorchLens
    wrapper (the orphaned prior-generation-snapshot failure mode) would run
    the "pristine" forward through a wrapper and bless coherently-corrupt
    captures -- the oracle now refuses on a wrapper-marked orig value.

    Discipline note: these tests only ADD one synthetic entry under a fake
    key and remove exactly that key afterwards; the ledger itself is
    append-only and must never be cleared (2026-08-14 lesson).
    """

    @staticmethod
    def _poison() -> int:
        def fake_prior_generation_wrapper(*args, **kwargs):  # pragma: no cover
            raise AssertionError("poisoned orig must never be executed")

        fake_prior_generation_wrapper.__tl_wrapper_name__ = "torch_func:poisoned"  # type: ignore[attr-defined]
        key = id(fake_prior_generation_wrapper)
        _state._decorated_to_orig[key] = fake_prior_generation_wrapper
        return key

    def test_poisoned_ledger_refuses_the_pristine_oracle(self) -> None:
        from torchlens._errors import CaptureNotActiveError

        model = nn.Sequential(nn.Linear(4, 4), nn.Tanh()).eval()
        tl.trace(model, torch.randn(2, 4))
        assert _state._is_decorated
        key = self._poison()
        try:
            with (
                pytest.raises(CaptureNotActiveError) as excinfo,
                pristine_torch_oracle(),
            ):
                raise AssertionError("oracle must refuse before yielding")
            assert excinfo.value.fields["code"] == "pristine_ledger_poisoned"
        finally:
            _state._decorated_to_orig.pop(key, None)
        # The refusal must not have unwrapped torch and left it that way.
        assert _state._is_decorated

    def test_poisoned_ledger_fails_validation_closed(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Tanh()).eval()
        x = torch.randn(2, 4)
        assert tl.validate(model, x, scope="forward", validate_metadata=False)
        key = self._poison()
        try:
            with pytest.warns(RuntimeWarning, match="pristine"):
                verdict = tl.validate(model, x, scope="forward", validate_metadata=False)
            assert verdict is False
        finally:
            _state._decorated_to_orig.pop(key, None)

    def test_honest_ledger_carries_no_wrapper_marked_origs(self) -> None:
        """Positive control: after real captures the append-only ledger holds
        only unwrapped originals -- the integrity scan has no false positive
        surface on an honest process."""

        model = nn.Sequential(nn.Linear(4, 4), nn.Tanh()).eval()
        tl.trace(model, torch.randn(2, 4))
        offenders = [
            getattr(orig, "__tl_wrapper_name__")
            for orig in _state._decorated_to_orig.values()
            if hasattr(orig, "__tl_wrapper_name__")
        ]
        assert not offenders, offenders
        with pristine_torch_oracle() as pristine:
            assert pristine


_MASKED_PROBE_SCRIPT = _PROBE_SCRIPT.replace(
    "return torch.tanh(x) + 1.0",
    "return (torch.tanh(x) > -5.0).float()",
)


@pytest.mark.heavy
def test_masked_interior_distortion_boundary_is_pinned() -> None:
    """r4 census MED-2: pin the DISCLOSED shared-root residual's boundary.

    The same planted 0.1% tanh wrapper distortion, masked by a threshold
    head, matches the pristine final output AND its own corrupt replay
    (``layer.func`` identity-shared), so in-process validation reports
    True. This is the documented scope limit of the pristine phase-0
    oracle -- the cross-process differential harness is the independent
    root for the masked-interior class. If this pin starts FAILING, the
    in-process oracle grew interior teeth: celebrate, then update the
    disclosure in ``validation/_pristine.py``.
    """

    stdout = _run_probe(_MASKED_PROBE_SCRIPT)
    assert "VERDICT=True" in stdout, stdout
