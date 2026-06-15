"""Completeness lint for portable scrub policy coverage."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import trace as trace_fn
from torchlens._io.scrub import _RAW_IMAGE_SENTINEL, _RAW_INPUT_IMAGE_BYTES_LIMIT
from torchlens.data_classes.buffer import Buffer
from torchlens.data_classes._state_adapter import state_items
from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.op import Op
from torchlens.data_classes.trace import Trace
from torchlens.data_classes.module import Module, ModuleCall
from torchlens.data_classes.param import Param


class _TinyIOModel(nn.Module):
    """Small model covering every target log class."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny test model."""
        return torch.relu(self.linear(self.bn(x)))


class _PixelOutput:
    """Small image-processor return object."""

    def __init__(self, pixel_values: torch.Tensor) -> None:
        """Store processed image pixels.

        Parameters
        ----------
        pixel_values:
            Processed image tensor.
        """

        self.pixel_values = pixel_values


class _TinyImageInputModel(nn.Module):
    """Small model that accepts auto-coerced PIL image input."""

    def __init__(self) -> None:
        """Initialize the convolution."""

        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def image_processor(self, image: Any, *, return_tensors: str) -> _PixelOutput:
        """Convert a PIL image into a tensor batch.

        Parameters
        ----------
        image:
            Raw image input.
        return_tensors:
            Requested tensor backend.

        Returns
        -------
        _PixelOutput
            Processed pixel tensor.
        """

        del image, return_tensors
        return _PixelOutput(torch.ones(1, 3, 8, 8))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the tiny image model."""

        return self.conv(pixel_values).mean()


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
        Buffer: next(iter(live_log.buffers)),
        FuncCallLocation: next(
            frame
            for layer in live_log.layer_list
            for frame in layer.code_context
            if layer.code_context
        ),
    }

    missing_by_class = {}
    for cls, instance in instances.items():
        missing = sorted(
            {field_name for field_name, _ in state_items(instance)} - set(cls.PORTABLE_STATE_SPEC)
        )
        if missing:
            missing_by_class[cls.__name__] = missing

    assert missing_by_class == {}


def test_small_raw_input_pil_round_trips_bounded_image(tmp_path: Path) -> None:
    """PIL raw input should survive ``save_raw_input='small'`` as a bounded image."""

    pil_image = pytest.importorskip("PIL.Image")
    image = pil_image.new("RGB", (512, 300), color=(10, 120, 200))
    trace = trace_fn(_TinyImageInputModel(), image, layers_to_save="none")
    path = tmp_path / "pil_raw_input.tlspec"

    trace.save(path)
    with (path / "metadata.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    loaded = tl.load(path)

    raw_image_record = metadata["raw_input"]
    assert raw_image_record[_RAW_IMAGE_SENTINEL] is True
    assert isinstance(raw_image_record["data"], bytes)
    assert len(raw_image_record["data"]) <= _RAW_INPUT_IMAGE_BYTES_LIMIT
    assert loaded.raw_input is not None
    assert loaded.raw_input.size[0] <= 256
    assert loaded.raw_input.size[1] <= 256
