"""U34.2: explicit watch callbacks read configuration from the environment.

An explicitly constructed callback may take signals / select / every /
hist_every from ``TORCHLENS_WATCH_*`` variables and prints where each setting
came from when it attaches: argument beats environment beats the framework's
derived cadence beats the package default. The environment never activates
instrumentation (no callback, no watch), and a malformed variable refuses
typed, naming the variable.
"""

from __future__ import annotations

import pytest
import torch

import torchlens.trackers as trk
from torchlens.trackers import WatchConfigError
from torchlens.trackers._callback_config import UNSET, resolve_callback_settings


class _Trainer:
    """Minimal Lightning trainer stand-in."""

    log_every_n_steps = 5
    max_steps = 100
    global_step = 0

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizers = [optimizer]


def _lightning_attach(callback: trk.LightningWatchCallback) -> None:
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = _Trainer(optimizer)
    callback.on_train_start(trainer, model)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    callback.on_train_end(trainer, model)


def test_env_configured_lightning_callback_prints_each_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance U34.2: env values are used and every setting names its source."""

    pytest.importorskip("lightning")
    monkeypatch.setenv("TORCHLENS_WATCH_SIGNALS", "gradients,parameters")
    monkeypatch.setenv("TORCHLENS_WATCH_HIST_EVERY", "none")
    printed: list[str] = []
    sink = trk.MemorySink()
    callback = trk.LightningWatchCallback(to=sink, select=("weight",), print_fn=printed.append)
    _lightning_attach(callback)
    assert len(printed) == 1
    text = printed[0]
    assert "signals=('gradients', 'parameters')  [env TORCHLENS_WATCH_SIGNALS]" in text
    assert "select=('weight',)  [argument]" in text
    assert "every=5  [derived: logging_steps=5]" in text
    assert "hist_every=None  [env TORCHLENS_WATCH_HIST_EVERY]" in text
    assert callback.settings.signals == ("gradients", "parameters")
    assert any(p.tag.startswith("parameters/") for p in sink.scalars)


def test_argument_beats_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHLENS_WATCH_EVERY", "9")
    monkeypatch.setenv("TORCHLENS_WATCH_SIGNALS", "parameters")
    settings = resolve_callback_settings(
        signals=("gradients",), select=None, every=3, hist_every=UNSET
    )
    assert settings.signals == ("gradients",) and settings.sources["signals"] == "argument"
    assert settings.every == 3 and settings.sources["every"] == "argument"
    assert settings.sources["select"] == "default"
    assert settings.sources["hist_every"] == "default"


def test_environment_beats_the_derived_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("transformers")
    monkeypatch.setenv("TORCHLENS_WATCH_EVERY", "4")
    printed: list[str] = []
    callback = trk.HFTrainerWatchCallback(to=trk.MemorySink(), print_fn=printed.append)
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    class _Args:
        logging_steps = 2
        max_steps = 10

    class _State:
        global_step = 0

    callback.on_train_begin(_Args(), _State(), None, model=model, optimizer=optimizer)
    assert callback.session is not None and callback.session.every == 4
    assert "every=4  [env TORCHLENS_WATCH_EVERY]" in printed[0]
    assert "signals=('gradients',)  [default]" in printed[0]
    callback.session.close(unwinding=True)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TORCHLENS_WATCH_EVERY", "0"),
        ("TORCHLENS_WATCH_EVERY", "often"),
        ("TORCHLENS_WATCH_HIST_EVERY", "-2"),
        ("TORCHLENS_WATCH_SIGNALS", "grads"),
        ("TORCHLENS_WATCH_SELECT", " , "),
    ],
)
def test_malformed_variable_refuses_naming_it(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(WatchConfigError) as info:
        resolve_callback_settings(signals=None, select=None, every=None, hist_every=UNSET)
    assert info.value.fields["code"] == "watch_env_config_invalid"
    assert info.value.fields["variable"] == name


def test_environment_never_activates_instrumentation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config variables alone install nothing: no callback, no hook."""

    monkeypatch.setenv("TORCHLENS_WATCH_SIGNALS", "gradients")
    monkeypatch.setenv("TORCHLENS_WATCH_EVERY", "1")
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    assert not optimizer._optimizer_step_pre_hooks
    assert not model._forward_hooks and not model._forward_pre_hooks
