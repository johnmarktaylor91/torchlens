"""A rerun re-saves the trace's requested sites, not the old raw op positions.

Regression pin: the capture-engine rerun re-saved by the ORIGINAL capture's
raw op indices. A staged edit inserts an ``intervention_replacement`` op, so
every later op shifts by one: after ``attach_hooks`` on a plain capture saved
with ``tl.module("blocks.1") | tl.module("head")``, the rerun saved the op
before the head (an ``add``) instead of the head, and left the steered site
output unsaved. Every rerun door here must save exactly what a fresh steered
capture with the same ``save=`` saves (``set`` on a module site included),
compared site by site through the save selectors (never by positional label).
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import ControlFlowDivergenceWarning

_VOCAB = 32
_DIM = 8
_SITE = "blocks.1"
_HEAD = "head"
_REPEATS = 2


class _Block(nn.Module):
    """Residual MLP block."""

    def __init__(self) -> None:
        """Build the two projections."""

        super().__init__()
        self.fc1 = nn.Linear(_DIM, 2 * _DIM)
        self.fc2 = nn.Linear(2 * _DIM, _DIM)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the residual MLP."""

        return h + self.fc2(torch.relu(self.fc1(h)))


class _Decoder(nn.Module):
    """Embedding, three residual blocks, and a vocabulary head."""

    def __init__(self) -> None:
        """Build the embedding, blocks, and head."""

        super().__init__()
        self.embed = nn.Embedding(_VOCAB, _DIM)
        self.blocks = nn.ModuleList(_Block() for _ in range(3))
        self.head = nn.Linear(_DIM, _VOCAB)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return the logits."""

        h = self.embed(ids)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def _setup() -> tuple[nn.Module, torch.Tensor]:
    """Return a seeded model and token ids."""

    torch.manual_seed(0)
    return _Decoder().eval(), torch.randint(0, _VOCAB, (2, 5))


def _steer() -> Any:
    """Return a deterministic steer at the site."""

    return tl.steer(torch.linspace(-1.0, 1.0, _DIM), magnitude=3.0, feature_axis=-1)


def _save() -> Any:
    """Return the save selector: the steered site and the head."""

    return tl.module(_SITE) | tl.module(_HEAD)


def _site_outs(trace: Any) -> dict[str, torch.Tensor]:
    """Return each requested site's saved output, keyed by its selector."""

    return {
        address: trace.find_sites(tl.module(address)).first().out.detach().clone()
        for address in (_SITE, _HEAD)
    }


def _saved_labels(trace: Any) -> list[str]:
    """Return the labels of every op holding a saved activation."""

    return [op.label for op in trace.layer_list if getattr(op, "has_saved_activation", False)]


def _rerun(trace: Any, model: nn.Module, x: torch.Tensor, **kwargs: Any) -> None:
    """Rerun, filtering only the known plain-capture divergence warning.

    Known false positive: the rerun divergence check compares the rerun's
    raw-event hash with the trace's last capture, not with that capture plus
    its staged edits, so the first rerun of a trace captured without the edit
    warns even when it is right. Only that message is filtered; the saved-op
    assertions below compare the graph to a fresh steered capture's.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="rerun raw-event shape hash diverged",
            category=ControlFlowDivergenceWarning,
        )
        trace.run(model, x, **kwargs)


@pytest.fixture
def fresh() -> dict[str, Any]:
    """A fresh steered capture with the same ``save=`` (the reference)."""

    model, x = _setup()
    plain = tl.trace(model, x, save=_save())
    trace = tl.trace(model, x, save=_save(), intervene=tl.when(tl.module(_SITE), _steer()))
    reference = {
        "model": model,
        "x": x,
        "outs": _site_outs(trace),
        "saved": _saved_labels(trace),
    }
    # The steer must move both sites, or the comparison proves nothing.
    plain_outs = _site_outs(plain)
    assert not torch.equal(plain_outs[_SITE], reference["outs"][_SITE])
    assert not torch.equal(plain_outs[_HEAD], reference["outs"][_HEAD])
    return reference


def _assert_matches_fresh(trace: Any, fresh: dict[str, Any]) -> None:
    """Assert the trace saved exactly what the fresh steered capture saved."""

    assert _saved_labels(trace) == fresh["saved"]
    outs = _site_outs(trace)
    for address, expected in fresh["outs"].items():
        assert torch.equal(outs[address], expected), address


@pytest.mark.smoke
def test_rerun_after_attach_hooks_resaves_the_requested_sites(fresh: dict[str, Any]) -> None:
    """``attach_hooks`` on a plain capture, then reruns, save the steered sites."""

    model, x = fresh["model"], fresh["x"]
    trace = tl.trace(model, x, save=_save())
    trace.attach_hooks(tl.module(_SITE), _steer(), confirm_mutation=True)
    for _ in range(_REPEATS):
        _rerun(trace, model, x)
        _assert_matches_fresh(trace, fresh)


def test_rerun_after_clear_hooks_resaves_the_plain_sites() -> None:
    """Removing the staged edit shifts ops back; the rerun saves the plain sites."""

    model, x = _setup()
    plain = tl.trace(model, x, save=_save())
    trace = tl.trace(model, x, save=_save(), intervene=tl.when(tl.module(_SITE), _steer()))
    trace.clear_hooks()
    _rerun(trace, model, x)
    assert _saved_labels(trace) == _saved_labels(plain)
    for address, expected in _site_outs(plain).items():
        assert torch.equal(_site_outs(trace)[address], expected), address


def test_fork_do_rerun_engine_resaves_the_requested_sites(fresh: dict[str, Any]) -> None:
    """``fork().do(..., engine="rerun")`` saves the steered sites."""

    model, x = fresh["model"], fresh["x"]
    root = tl.trace(model, x, save=_save())
    fork = root.fork()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="rerun raw-event shape hash diverged",
            category=ControlFlowDivergenceWarning,
        )
        fork.do(
            tl.module(_SITE),
            _steer(),
            model=model,
            x=x,
            intervention=tl.options.InterventionOptions(engine="rerun"),
        )
    _assert_matches_fresh(fork, fresh)
    assert _saved_labels(root) == _saved_labels(tl.trace(model, x, save=_save()))


def test_chunked_rerun_resaves_the_requested_sites(fresh: dict[str, Any]) -> None:
    """A chunked rerun after ``attach_hooks`` saves the steered sites."""

    model, x = fresh["model"], fresh["x"]
    trace = tl.trace(model, x, save=_save())
    trace.attach_hooks(tl.module(_SITE), _steer(), confirm_mutation=True)
    _rerun(trace, model, x, replay=tl.options.ReplayOptions(chunk_size=1))
    _assert_matches_fresh(trace, fresh)


def test_rerun_after_set_resaves_the_requested_sites(fresh: dict[str, Any]) -> None:
    """``set(module_site, value)``, then a rerun, saves the set site and the head."""

    model, x = fresh["model"], fresh["x"]
    trace = tl.trace(model, x, save=_save())
    value = torch.full_like(fresh["outs"][_SITE], 0.5)
    trace.set(tl.module(_SITE), value, confirm_mutation=True)
    _rerun(trace, model, x)
    # Same graph shape as the steered capture: one replacement op at the site.
    assert _saved_labels(trace) == fresh["saved"]
    outs = _site_outs(trace)
    assert torch.equal(outs[_SITE], value)
    rest = value
    for block in model.blocks[2:]:
        rest = block(rest)
    assert torch.equal(outs[_HEAD], model.head(rest).detach())
