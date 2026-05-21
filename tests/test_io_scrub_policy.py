"""Completeness lint for portable scrub policy coverage."""

from __future__ import annotations

import torch
from torch import nn

from torchlens import trace as trace_fn
from torchlens.data_classes.buffer_log import Buffer
from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.data_classes.layer_log import Layer
from torchlens.data_classes.op_log import Op
from torchlens.data_classes.model_log import Trace
from torchlens.data_classes.module_log import Module, ModuleCall
from torchlens.data_classes.param_log import Param


class _TinyIOModel(nn.Module):
    """Small model covering every target log class."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny test model."""
        return torch.relu(self.linear(self.bn(x)))


def _build_live_log() -> Trace:
    """Create one canonical live ``Trace`` for completeness checks."""

    torch.manual_seed(0)
    model = _TinyIOModel()
    x = torch.randn(2, 4)
    return trace_fn(
        model,
        x,
        layers_to_save="all",
        save_arg_values=True,
        save_rng_states=True,
        save_code_context=True,
        random_seed=0,
    )


def test_portable_state_specs_cover_every_live_attribute() -> None:
    """Each target class must map every live attribute to a scrub policy."""

    live_log = _build_live_log()
    instances = {
        Trace: live_log,
        Op: next(layer for layer in live_log.layer_list if type(layer) is Op),
        Layer: next(iter(live_log.layer_logs.values())),
        Module: next(iter(live_log.modules)),
        ModuleCall: next(iter(live_log.modules._pass_dict.values())),
        Param: next(iter(live_log.param_logs)),
        Buffer: next(layer for layer in live_log.layer_list if isinstance(layer, Buffer)),
        FuncCallLocation: next(
            frame
            for layer in live_log.layer_list
            for frame in layer.code_context
            if layer.code_context
        ),
    }

    missing_by_class = {}
    for cls, instance in instances.items():
        missing = sorted(set(vars(instance)) - set(cls.PORTABLE_STATE_SPEC))
        if missing:
            missing_by_class[cls.__name__] = missing

    assert missing_by_class == {}
