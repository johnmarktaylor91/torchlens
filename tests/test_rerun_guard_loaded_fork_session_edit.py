"""The legacy rerun guard judges the ARTIFACT's values, never this session's edits.

``trace.save()`` drops the intervention recipe, so ``run(model, x)`` on a loaded
trace whose recorded values were intervened refuses
(``run_intervention_spec_not_persisted``). The guard used to read the live
trace for that evidence; a ``set_only`` edit staged in this session on a fork
of a PRISTINE loaded trace marks ops edited too, so the fork was refused with
the wrong cause. The loader now snapshots the artifact's state before any
session edit and the guard reads the snapshot; a genuinely intervened
artifact still refuses, forked or not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import EngineDispatchError
from torchlens.options import CaptureOptions, InterventionOptions

_NOT_PERSISTED = "run_intervention_spec_not_persisted"


def _model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()


def _input() -> torch.Tensor:
    return torch.randn(2, 8, generator=torch.Generator().manual_seed(0))


def _mask(x: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., :3] = True
    return mask


def _saved(model: nn.Module, x: torch.Tensor, path: Path, *, intervened: bool) -> Path:
    trace = tl.trace(model, x, save="all", capture=CaptureOptions(replayable=True))
    if intervened:
        trace.do(
            tl.units(trace.input_ops[0].label, _mask(x)),
            tl.zero_ablate(),
            intervention=InterventionOptions(confirm_mutation=True),
        )
    trace.save(path, level="portable", overwrite=True)
    return path


def _set_only_fork(loaded: tl.Trace, x: torch.Tensor) -> tl.Trace:
    fork = loaded.fork()
    fork.do(
        tl.units(loaded.input_ops[0].label, _mask(x)),
        tl.zero_ablate(),
        intervention=InterventionOptions(engine="set_only"),
    )
    return fork


def test_loaded_pristine_fork_with_a_session_set_only_edit_reruns(tmp_path: Path) -> None:
    model, x = _model(), _input()
    loaded = tl.load(_saved(model, x, tmp_path / "pristine.tlspec", intervened=False))
    assert loaded._runnable.loaded_values_intervened is False

    fork = _set_only_fork(loaded, x)
    assert fork._runnable.loaded_values_intervened is False, "forks copy the snapshot"

    masked = x.masked_fill(_mask(x), 0)
    fork.run(model, masked)

    expected = model(masked)
    assert torch.allclose(fork[fork.output_layers[0]].out, expected)


def test_loaded_pristine_trace_reruns(tmp_path: Path) -> None:
    model, x = _model(), _input()
    loaded = tl.load(_saved(model, x, tmp_path / "pristine.tlspec", intervened=False))
    loaded.run(model, x)
    assert torch.allclose(loaded[loaded.output_layers[0]].out, model(x))


def test_loaded_intervened_trace_still_refuses(tmp_path: Path) -> None:
    model, x = _model(), _input()
    loaded = tl.load(_saved(model, x, tmp_path / "intervened.tlspec", intervened=True))
    assert loaded._runnable.loaded_values_intervened is True

    with pytest.raises(EngineDispatchError) as excinfo:
        loaded.run(model, x)
    assert excinfo.value.fields["code"] == _NOT_PERSISTED


def test_loaded_intervened_fork_with_a_session_edit_still_refuses(tmp_path: Path) -> None:
    """A session edit never launders an intervened artifact into a pristine one."""

    model, x = _model(), _input()
    loaded = tl.load(_saved(model, x, tmp_path / "intervened.tlspec", intervened=True))
    fork = _set_only_fork(loaded, x)

    with pytest.raises(EngineDispatchError) as excinfo:
        fork.run(model, x.masked_fill(_mask(x), 0))
    assert excinfo.value.fields["code"] == _NOT_PERSISTED


def test_live_trace_has_no_snapshot() -> None:
    trace = tl.trace(_model(), _input(), capture=CaptureOptions(replayable=True))
    assert trace._runnable.loaded_values_intervened is None
