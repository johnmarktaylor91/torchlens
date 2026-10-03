"""W051 (AUD-CODE 2.17): the Python / NumPy GLOBAL-engine state surface is witnessed.

The seed-replayable global engines were witnessed only by a before/after state snapshot
at the capture boundary, so an owner-thread ``getstate -> draw -> setstate`` (or a
``seed``/``set_state`` from a pre-window value) left the compare EQUAL while the forward
consumed a host scalar: ``host_rng_consumed=False``, the replay did not require the
capture seed, and a run under different ambient host state read ``verified`` with an
output that a fresh oracle run does not reproduce.

``random.getstate/setstate/seed`` and ``numpy.random.get_state/set_state/seed`` now carry
``replayable_read`` rows (the torch ``initial_seed`` analog): consumption is recorded, so
a seedless or off-seed replay ceilings while a replay AT the capture seed reproduces the
draw and stays verified -- these rows never ceiling permanently, matching the contract's
asymmetry note (a python/numpy in-forward reseed is self-reproducing on-seed).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.utils import _rng_channels
from torchlens.utils.rng import HOST_NONDETERMINISM_REGISTRY, host_nondeterminism_monitor


class _LinearNoBias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _PyGetDrawSet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = random.getstate()
        value = random.random()
        random.setstate(state)
        return self.lin(x) * (1.0 + value)


class _NpGetDrawSet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = np.random.get_state()
        value = float(np.random.random())
        np.random.set_state(state)
        return self.lin(x) * (1.0 + value)


# ---- monitor level ------------------------------------------------------------------------------


def test_python_get_draw_set_records_replayable_reads() -> None:
    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        state = random.getstate()
        random.random()
        random.setstate(state)
    assert {"random.getstate", "random.setstate"} <= result.replayable_reads
    assert not result.channels, "global-engine state rows are consumed-only, never a ceiling"
    assert not result.uncertain


def test_numpy_legacy_get_draw_set_records_replayable_reads() -> None:
    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        state = np.random.get_state()
        np.random.random()
        np.random.set_state(state)
    assert {"numpy.random.get_state", "numpy.random.set_state"} <= result.replayable_reads
    assert not result.channels


def test_in_forward_reseeds_record_consumption_without_ceiling() -> None:
    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        random.seed(0)
        np.random.seed(0)
    assert {"random.seed", "numpy.random.seed"} <= result.replayable_reads
    assert not result.channels


def test_mtrand_module_spelling_marks_too() -> None:
    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        np.random.mtrand.get_state()
    assert "numpy.random.get_state" in result.replayable_reads


def test_held_reference_python_getstate_is_classified_by_c_call() -> None:
    """``from random import getstate`` calls the singleton's bound Python method, bypassing
    the module-attr patch; its body enters the C base ``_random.Random.getstate`` on the
    exempt singleton receiver, which the c_call layer classifies by name."""

    held_getstate = random.getstate
    held_setstate = random.setstate
    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        state = held_getstate()
        random.random()
        held_setstate(state)
    assert {"random.getstate", "random.setstate"} <= result.replayable_reads


def test_private_instance_state_methods_do_not_mark_global_rows() -> None:
    """A user's own ``random.Random`` instance is not the global engine: its state methods
    carry no global-engine row (its DRAWS keep their instance-draw channel)."""

    private = random.Random(3)
    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        private.setstate(private.getstate())
        private.seed(4)
    assert not {"random.getstate", "random.setstate", "random.seed"} & result.replayable_reads


def test_torchlens_own_state_bookkeeping_never_self_marks() -> None:
    """A deterministic capture records NO global-engine state row even though TorchLens's
    per-op state logging reads ``random.getstate()``/``np.random.get_state()`` in-window."""

    seam = tl.trace(
        _LinearNoBias().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )._runnable
    assert not seam.host_rng_consumed
    assert seam.host_rng_replayable_reads == ()
    assert not seam.rng_monitor_uncertain


def test_state_patches_restore_exactly() -> None:
    originals = {
        (random, "getstate"): random.getstate,
        (random, "setstate"): random.setstate,
        (random, "seed"): random.seed,
        (np.random, "get_state"): np.random.get_state,
        (np.random, "set_state"): np.random.set_state,
        (np.random, "seed"): np.random.seed,
        (np.random.mtrand, "seed"): np.random.mtrand.seed,
    }
    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        for (holder, name), original in originals.items():
            assert getattr(holder, name) is not original, (holder, name)
    for (holder, name), original in originals.items():
        assert getattr(holder, name) is original, (holder, name)
    assert not result.uncertain


# ---- end to end through the runnable seam --------------------------------------------------


@pytest.mark.parametrize("model_class", [_PyGetDrawSet, _NpGetDrawSet])
def test_get_draw_set_capture_requires_the_capture_seed(
    tmp_path: Path, model_class: type[nn.Module]
) -> None:
    """The audited false-verified: consumption is now recorded, a seedless replay under
    different ambient host state ceilings, and a replay AT the capture seed reproduces the
    fresh oracle exactly (the disposition is replayable, not a permanent ceiling)."""

    x = torch.randn(2, 4)
    model = model_class().eval()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    seam = trace._runnable
    assert seam.host_rng_consumed
    assert not seam.host_rng_unreplayable
    assert not seam.rng_monitor_uncertain
    capture_seed = trace.random_seed
    assert isinstance(capture_seed, int)

    bundle = tmp_path / f"{model_class.__name__}.tlspec"
    tl.save(trace, str(bundle), level="runnable", include_weights=True)
    loaded = tl.load(str(bundle))

    random.seed(999)
    np.random.seed(999)
    seedless = loaded.run(inputs=x.clone())
    assert seedless.report.path_faithfulness.value != "verified"

    random.seed(999)
    np.random.seed(999)
    seeded = loaded.run(inputs=x.clone(), seed=capture_seed)
    assert seeded.report.path_faithfulness.value == "verified"
    tl.utils.rng.set_random_seed(capture_seed)
    fresh = model(x)
    assert torch.allclose(seeded.output, fresh)


# ---- registry tripwires ------------------------------------------------------------------------


def test_registry_carries_global_engine_state_rows() -> None:
    rows = [row for row in HOST_NONDETERMINISM_REGISTRY if row.family == "rng_global_state"]
    targets = {row.target for row in rows}
    assert targets == {channel for _m, _n, channel in _rng_channels.GLOBAL_ENGINE_STATE_ROWS}
    assert all(row.classification == "replayable_read" for row in rows)
    strategies = {(row.target, row.strategy) for row in rows}
    assert ("random.getstate", "c_call_identity") in strategies
    assert ("numpy.random.set_state", "module_patch") in strategies
    assert ("numpy.random.set_state", "c_call_identity") not in strategies
