"""The ``"mi"`` appliance (forks.8): kit calls bound to one trace.

Each bound call must return the same values as the module-level function it
wraps, both on the class and through the ``trace.using("mi")`` door.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp._appliance import MechInterpAppliance

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_model" / "r0"))


@pytest.fixture(scope="module")
def gpt2_trace():
    from families import build_gpt2

    model = build_gpt2("eager")
    torch.manual_seed(6)
    first = torch.randint(0, 500, (1, 5))
    x = torch.cat([first, first], dim=1)  # repeated tokens give induction heads signal
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_appliance_head_scores_equals_direct_call(gpt2_trace):
    appliance = MechInterpAppliance(gpt2_trace)
    assert appliance.trace is gpt2_trace
    for kind in ("previous_token", "duplicate_token", "induction"):
        bound = appliance.head_scores(kind=kind)
        direct = mi.head_scores(gpt2_trace, kind=kind)
        for field in ("kind", "measure", "module_addresses", "scores", "eligible_rows"):
            assert getattr(bound, field) == getattr(direct, field), field
        assert any(score is not None for row in bound.scores for score in row)


def test_appliance_circuits_equal_direct_calls(gpt2_trace):
    appliance = MechInterpAppliance(gpt2_trace)
    for address in ("transformer.h.0.attn", "transformer.h.1.attn"):
        assert torch.equal(appliance.qk_circuit(address).AB, mi.qk_circuit(gpt2_trace, address).AB)
        assert torch.equal(appliance.ov_circuit(address).AB, mi.ov_circuit(gpt2_trace, address).AB)


def test_using_mi_returns_the_bound_appliance(gpt2_trace):
    appliance = gpt2_trace.using("mi")
    assert isinstance(appliance, MechInterpAppliance)
    assert appliance.trace is gpt2_trace
    assert tl.appliances.names()["mi"] == ("mechinterp kit: head_scores, qk_circuit, ov_circuit")
    # The registration is a lazy string: a fresh ``import torchlens`` lists "mi"
    # without importing the kit, and the first ``trace.using("mi")`` imports it.
    script = textwrap.dedent(
        """
        import sys

        import torch

        import torchlens as tl

        assert "torchlens.mechinterp" not in sys.modules, "kit imported eagerly"
        assert "mi" in tl.appliances.names()
        trace = tl.trace(torch.nn.Linear(3, 2), torch.ones(1, 3))
        appliance = trace.using("mi")
        assert type(appliance).__name__ == "MechInterpAppliance"
        assert appliance.trace is trace
        assert "torchlens.mechinterp._appliance" in sys.modules
        print("ok")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("ok")


def test_using_mi_head_scores_equals_direct_call(gpt2_trace):
    for kind in ("previous_token", "duplicate_token", "induction"):
        bound = gpt2_trace.using("mi").head_scores(kind=kind)
        direct = mi.head_scores(gpt2_trace, kind=kind)
        for field in ("kind", "measure", "module_addresses", "scores", "eligible_rows"):
            assert getattr(bound, field) == getattr(direct, field), field
        assert any(score is not None for row in bound.scores for score in row)


def test_using_mi_circuits_equal_direct_calls(gpt2_trace):
    appliance = gpt2_trace.using("mi")
    for address in ("transformer.h.0.attn", "transformer.h.1.attn"):
        assert torch.equal(appliance.qk_circuit(address).AB, mi.qk_circuit(gpt2_trace, address).AB)
        assert torch.equal(appliance.ov_circuit(address).AB, mi.ov_circuit(gpt2_trace, address).AB)
