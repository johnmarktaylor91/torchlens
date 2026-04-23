"""Completeness lint for portable scrub policy coverage."""

from __future__ import annotations

import torch
from torch import nn

from torchlens import log_forward_pass
from torchlens.data_classes.buffer_log import BufferLog
from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.data_classes.layer_log import LayerLog
from torchlens.data_classes.layer_pass_log import LayerPassLog
from torchlens.data_classes.model_log import ModelLog
from torchlens.data_classes.module_log import ModuleLog, ModulePassLog
from torchlens.data_classes.param_log import ParamLog


class _TinyIOModel(nn.Module):
    """Small model covering every target log class."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny test model."""
        return torch.relu(self.linear(self.bn(x)))


def _build_live_log() -> ModelLog:
    """Create one canonical live ``ModelLog`` for completeness checks."""

    torch.manual_seed(0)
    model = _TinyIOModel()
    x = torch.randn(2, 4)
    return log_forward_pass(
        model,
        x,
        layers_to_save="all",
        save_function_args=True,
        save_rng_states=True,
        save_source_context=True,
        random_seed=0,
    )


def test_portable_state_specs_cover_every_live_attribute() -> None:
    """Each target class must map every live attribute to a scrub policy."""

    live_log = _build_live_log()
    instances = {
        ModelLog: live_log,
        LayerPassLog: next(layer for layer in live_log.layer_list if type(layer) is LayerPassLog),
        LayerLog: next(iter(live_log.layer_logs.values())),
        ModuleLog: next(iter(live_log.modules)),
        ModulePassLog: next(iter(live_log.modules._pass_dict.values())),
        ParamLog: next(iter(live_log.param_logs)),
        BufferLog: next(layer for layer in live_log.layer_list if isinstance(layer, BufferLog)),
        FuncCallLocation: next(
            frame
            for layer in live_log.layer_list
            for frame in layer.func_call_stack
            if layer.func_call_stack
        ),
    }

    missing_by_class = {}
    for cls, instance in instances.items():
        missing = sorted(set(vars(instance)) - set(cls.PORTABLE_STATE_SPEC))
        if missing:
            missing_by_class[cls.__name__] = missing

    assert missing_by_class == {}
