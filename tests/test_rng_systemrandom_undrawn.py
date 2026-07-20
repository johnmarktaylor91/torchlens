"""r55 corr_1: an UNDRAWN ``random.SystemRandom`` must not ceiling a capture.

``random.SystemRandom`` is a ``random.Random`` subclass whose ``getstate()``
raises ``NotImplementedError`` BY DESIGN (stateless OS-entropy engine). The r53
inventory digested every ``random.Random`` and routed that exception through the
generic ``inventory_state_read_failed`` path -- so a deterministic model that
merely STORES an undrawn ``SystemRandom`` was over-triggered to UNVERIFIABLE /
NOT_APPLICABLE. The r55 close classifies ``getstate() -> NotImplementedError``
as monitored-not-digestible (``_NotADigestableRng``): possession alone never
ceilings, while actual draws stay witnessed by the class-method patches on
``random.SystemRandom.{random,getrandbits,randbytes}``.

Tripwire strengthening pinned here: the carve-out is NARROW -- any OTHER
``getstate()`` exception (a genuinely broken state read) still fails closed to
``inventory_state_read_failed``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness
from torchlens.utils.rng import (
    _NotADigestableRng,
    _rng_exempt_instances,
    host_nondeterminism_monitor,
)

_CAPTURE = CaptureOptions(
    intervention_ready=True, capture_container_structure=True, cache=False, random_seed=1
)


class _SystemRandomHolder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.sys_rng = random.SystemRandom()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).relu()


class _SystemRandomDrawer(_SystemRandomHolder):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.lin(x)
        return hidden * 2.0 if self.sys_rng.random() < 0.5 else hidden * 3.0


def _run_report(model: nn.Module, tmp_path: Path) -> Any:
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    trace = tl.trace(model.eval(), x.clone(), capture=_CAPTURE)
    path = tmp_path / "sysrandom.tlspec"
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    return tl.load(path).run(inputs=x.clone()).report


def test_digest_classifies_systemrandom_not_digestable() -> None:
    """``getstate() -> NotImplementedError`` is monitored-not-digestible, never a
    generic inventory failure."""

    with pytest.raises(_NotADigestableRng):
        host_nondeterminism_monitor._digest_rng_instance(random.SystemRandom())


def test_other_getstate_failures_still_fail_closed() -> None:
    """NARROW carve-out (tripwire intact): a ``random.Random`` subclass whose
    ``getstate()`` raises anything OTHER than ``NotImplementedError`` still
    flags ``inventory_state_read_failed``."""

    class _BrokenRandom(random.Random):
        def getstate(self) -> Any:
            raise RuntimeError("corrupted state read")

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.rng = _BrokenRandom()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    monitor = host_nondeterminism_monitor(_Model())
    monitor._exempt_ids = frozenset(id(inst) for inst in _rng_exempt_instances())
    monitor._sweep_model_generators()
    assert monitor.result.uncertain is True
    assert "inventory_state_read_failed" in monitor.result.uncertain_detail


def test_undrawn_systemrandom_stays_verified_and_attested(tmp_path: Path) -> None:
    """The corr_1 over-trigger closed: mere possession of an undrawn
    ``SystemRandom`` on a deterministic model keeps VERIFIED / ATTESTED."""

    report = _run_report(_SystemRandomHolder(), tmp_path)
    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert report.nondeterministic_sources == ()


def test_drawn_systemrandom_still_ceilings(tmp_path: Path) -> None:
    """No under-catch traded for the over-trigger fix: an actual ``SystemRandom``
    draw during forward (class-method patch witness) still ceilings the run."""

    report = _run_report(_SystemRandomDrawer(), tmp_path)
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
