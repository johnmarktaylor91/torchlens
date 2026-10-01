"""Regression tests: ``force_shape_change`` flows from helper kwargs to execution.

Round 26 W3 audit MED-1: ``normalize_hook_plan`` stamped the per-entry
``force_shape_change`` metadata from its own default-False parameter instead of
the helper's kwargs. No production caller passes the parameter, so the
documented escape hatch was dead through every public path (``intervene=``,
``hooks=``, ``Trace.push(replay=ReplayOptions(hooks=...))``): a requested shape/dtype-changing
intervention raised ``HookValueError``. These tests drive the flag through the
PUBLIC chain (the prior test called ``_execute_hook`` directly, which bypassed
the exact hop that dropped the flag) and pin the default-False guard.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.errors import HookValueError
from torchlens.intervention.hooks import normalize_hook_plan
from torchlens.intervention.types import HelperSpec


class _ReluNet(nn.Module):
    """Tiny net whose relu out is shape (1, 4)."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.l1(x)).sum()


class _ToDouble(nn.Module):
    """Spliced module that changes dtype float32 -> float64."""

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        return out.double()


def _model_and_input() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _ReluNet().eval(), torch.randn(1, 4)


def _output_out(log: tl.Trace) -> torch.Tensor:
    key = [layer.layer_label for layer in log.layers if "output" in layer.layer_label][0]
    return log[key].out


def _relu_layer(log: tl.Trace):
    key = [layer.layer_label for layer in log.layers if "relu" in layer.layer_label][0]
    return log[key]


_FLAG_HELPERS: dict[str, Callable[[], HelperSpec]] = {
    "zero_ablate": lambda: tl.zero_ablate(force_shape_change=True),
    "replace_with": lambda: tl.replace_with(torch.ones(1, 8), force_shape_change=True),
    "scale": lambda: tl.scale(0.5, force_shape_change=True),
    "noise": lambda: tl.noise(0.1, seed=0, force_shape_change=True),
    "clamp": lambda: tl.clamp(min=0.0, force_shape_change=True),
    "mean_ablate": lambda: tl.mean_ablate(force_shape_change=True),
    "resample_ablate": lambda: tl.resample_ablate(seed=0, force_shape_change=True),
    "add": lambda: tl.add(1.0, force_shape_change=True),
    "splice_module": lambda: tl.splice_module(_ToDouble(), force_shape_change=True),
    "grad_zero": lambda: tl.grad_zero(force_shape_change=True),
    "grad_scale": lambda: tl.grad_scale(2.0, force_shape_change=True),
}


@pytest.mark.smoke
@pytest.mark.parametrize("helper_name", sorted(_FLAG_HELPERS))
def test_helper_flag_reaches_entry_metadata(helper_name: str) -> None:
    """helper(force_shape_change=True) must stamp True on every plan entry."""

    helper = _FLAG_HELPERS[helper_name]()
    entries = normalize_hook_plan(helper, default_site_target=tl.func("relu"))
    assert entries, f"{helper_name} produced no hook-plan entries"
    for entry in entries:
        assert entry.metadata.get("force_shape_change") is True, (
            f"{helper_name}: helper kwargs request force_shape_change=True but the "
            f"normalized entry metadata says {entry.metadata.get('force_shape_change')!r}"
        )


@pytest.mark.smoke
def test_helper_flag_default_false_metadata() -> None:
    """Default helpers stamp False; the safety guard stays armed by default."""

    entries = normalize_hook_plan(tl.zero_ablate(), default_site_target=tl.func("relu"))
    assert entries
    for entry in entries:
        assert entry.metadata.get("force_shape_change") is False


@pytest.mark.smoke
def test_intervene_shape_change_applies() -> None:
    """intervene= with force_shape_change=True applies a (1,4)->(1,8) replacement."""

    model, x = _model_and_input()
    log = tl.trace(
        model,
        x,
        save=lambda ctx: True,
        intervene=tl.when(
            tl.func("relu"), tl.replace_with(torch.ones(1, 8), force_shape_change=True)
        ),
    )
    assert float(_output_out(log)) == pytest.approx(8.0)
    relu = _relu_layer(log)
    assert tuple(relu.out.shape) == (1, 8)
    assert relu.intervention_replaced is True


@pytest.mark.smoke
def test_hooks_kwarg_shape_change_applies() -> None:
    """hooks= with force_shape_change=True applies the shape-changed replacement."""

    model, x = _model_and_input()
    log = tl.trace(
        model,
        x,
        save=lambda ctx: True,
        capture=tl.options.CaptureOptions(
            hooks={tl.func("relu"): tl.replace_with(torch.ones(1, 8), force_shape_change=True)}
        ),
    )
    assert float(_output_out(log)) == pytest.approx(8.0)
    assert tuple(_relu_layer(log).out.shape) == (1, 8)


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_replay_hooks_shape_change_applies() -> None:
    """Trace.push(replay=ReplayOptions(hooks=...)) with force_shape_change=True applies the replacement."""

    model, x = _model_and_input()
    log = tl.trace(
        model,
        x,
        save=lambda ctx: True,
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    replayed = log.push(
        replay=tl.options.ReplayOptions(
            hooks={tl.func("relu"): tl.replace_with(torch.ones(1, 8), force_shape_change=True)}
        )
    )
    assert float(_output_out(replayed)) == pytest.approx(8.0)


@pytest.mark.smoke
def test_push_replay_options_shape_change_applies() -> None:
    """The canonical push(replay=ReplayOptions(hooks=...)) spelling honors the flag."""

    from torchlens.options import ReplayOptions

    model, x = _model_and_input()
    log = tl.trace(
        model,
        x,
        save=lambda ctx: True,
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    pushed = log.push(
        replay=ReplayOptions(
            hooks={tl.func("relu"): tl.replace_with(torch.ones(1, 8), force_shape_change=True)}
        )
    )
    assert float(_output_out(pushed)) == pytest.approx(8.0)


@pytest.mark.smoke
def test_splice_module_dtype_change_applies() -> None:
    """splice_module(force_shape_change=True) may change dtype through intervene=."""

    model, x = _model_and_input()
    log = tl.trace(
        model,
        x,
        save=lambda ctx: True,
        intervene=tl.when(tl.func("relu"), tl.splice_module(_ToDouble(), force_shape_change=True)),
    )
    assert _output_out(log).dtype == torch.float64


@pytest.mark.smoke
def test_default_false_still_raises_on_shape_change_intervene() -> None:
    """Without the flag, an unexpected shape change still raises HookValueError."""

    model, x = _model_and_input()
    with pytest.raises(HookValueError):
        tl.trace(
            model,
            x,
            save=lambda ctx: True,
            intervene=tl.when(tl.func("relu"), tl.replace_with(torch.ones(1, 8))),
        )


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_default_false_still_raises_on_shape_change_replay() -> None:
    """Replay without the flag still rejects an unexpected shape change."""

    model, x = _model_and_input()
    log = tl.trace(
        model,
        x,
        save=lambda ctx: True,
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    with pytest.raises(HookValueError):
        log.push(
            replay=tl.options.ReplayOptions(
                hooks={tl.func("relu"): tl.replace_with(torch.ones(1, 8))}
            )
        )
