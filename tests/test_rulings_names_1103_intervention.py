"""Walk 2026-11-03-decisions, intervention rulings names.13, names.14 and names.9.

names.13: ``Trace.do`` takes its site positionally; the parameter is ``site``
and a do refusal names ``argument="site"``.
names.14: ``InterventionOptions.confirm_mutation`` is gone; a root trace edit
warns ``TraceMutationWarning`` once, a fork and the session-level
``suppress_mutate_warnings()`` context stay quiet.
names.9: ``Trace.run`` takes ``model`` and ``x`` positionally only; the
transactional ``inputs=`` form is unchanged.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from example_models import TinyReluAdd as ReluAdd

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._runnable import RunResult
from torchlens.intervention.errors import TraceMutationWarning
from torchlens.io import suppress_mutate_warnings
from torchlens.options import CaptureOptions, InterventionOptions


def _capture() -> Any:
    """Capture a replayable log of the tiny relu-add model."""

    return tl.trace(
        ReluAdd(),
        torch.randn(2, 3),
        capture=CaptureOptions(replayable=True),
    )


def _first_internal_label(log: Any) -> str:
    """Return the first layer label that is neither input, output nor buffer."""

    for layer in log._layer_list:
        if not (layer.is_input or layer.is_output or layer.is_buffer):
            return layer.layer_label
    raise AssertionError("expected at least one internal layer")


def _set_only() -> InterventionOptions:
    return InterventionOptions(engine="set_only")


# names.13 ------------------------------------------------------------------


def test_do_site_is_positional_only() -> None:
    """``do(site=...)`` is refused by the signature; positional works."""

    log = _capture().fork("edit")
    with pytest.raises(TypeError):
        log.do(site=tl.func("relu"), value_or_hook=torch.zeros(2, 3), intervention=_set_only())
    assert log.do(tl.func("relu"), torch.zeros(2, 3), intervention=_set_only()) is log


def test_do_refusal_names_site_argument() -> None:
    """A mixed selection batch refuses with ``argument="site"``."""

    log = _capture()
    selection = log.layers[_first_internal_label(log)].__selection__()
    with pytest.raises(InvalidArgumentError) as excinfo:
        log.fork("batch").do([(selection, tl.scale(2.0)), ("relu_1_1", tl.zero_ablate())])
    assert excinfo.value.fields["code"] == "selection_batch_pair_invalid"
    assert excinfo.value.fields["argument"] == "site"


# names.14 ------------------------------------------------------------------


def test_intervention_options_has_no_confirm_mutation() -> None:
    """The per-call mutation opt-out keyword is gone from the options."""

    with pytest.raises(TypeError):
        InterventionOptions(confirm_mutation=True)  # type: ignore[call-arg]
    assert not hasattr(InterventionOptions(), "confirm_mutation")


def test_root_do_warns_once_fork_and_context_stay_quiet() -> None:
    """A root edit warns; a fork and ``suppress_mutate_warnings()`` do not."""

    root = _capture()
    with pytest.warns(TraceMutationWarning):
        root.do(tl.func("relu"), torch.zeros(2, 3), intervention=_set_only())

    forked = _capture().fork("quiet")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        forked.do(tl.func("relu"), torch.zeros(2, 3), intervention=_set_only())
    assert record == []

    session = _capture()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with suppress_mutate_warnings():
            session.do(tl.func("relu"), torch.zeros(2, 3), intervention=_set_only())
    assert record == []


# names.9 -------------------------------------------------------------------


def test_run_model_and_x_are_positional_only() -> None:
    """``run(model=...)`` and ``run(x=...)`` are refused; positional reruns."""

    x = torch.randn(2, 3)
    log = tl.trace(ReluAdd(), x, capture=CaptureOptions(replayable=True)).fork("rerun")
    log.do(tl.func("relu"), torch.zeros(2, 3), intervention=_set_only())
    with pytest.raises(TypeError):
        log.run(model=ReluAdd(), x=x)
    with pytest.raises(TypeError):
        log.run(ReluAdd(), x=x)
    assert log.run(ReluAdd(), x) is log


def test_run_inputs_transaction_returns_run_result() -> None:
    """The transactional ``inputs=`` door still returns a ``RunResult``."""

    x = torch.randn(2, 3)
    model = ReluAdd()  # the trace holds the model weakly; keep it alive for a live run
    log = tl.trace(model, x)
    result = log.run(inputs=x)
    assert isinstance(result, RunResult)
    assert result.trace is not log
