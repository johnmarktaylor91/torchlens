"""Tests for IO-S2 pandas and file-export surfaces."""

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
import torch.nn as nn

from torchlens import log_forward_pass
from torchlens.constants import (
    BUFFER_LOG_FIELD_ORDER,
    MODULE_PASS_LOG_FIELD_ORDER,
    PARAM_LOG_FIELD_ORDER,
)
from torchlens.data_classes._summary import format_call_arg
from torchlens.data_classes.buffer_log import BufferAccessor
from torchlens.data_classes.module_log import ModulePassLog
from torchlens.data_classes.param_log import ParamAccessor


class _SummaryBlock(nn.Module):
    """Submodule used to exercise module-pass argument summaries."""

    def forward(
        self,
        x: torch.Tensor,
        payload: list[Any],
        config: dict[str, Any],
        *,
        flag: bool,
        note: str,
        optional: dict[str, Any],
    ) -> torch.Tensor:
        """Apply a small transform using mixed-typed call arguments.

        Parameters
        ----------
        x:
            Input tensor.
        payload:
            Mixed positional payload.
        config:
            Mixed positional configuration.
        flag:
            Boolean keyword switch.
        note:
            String keyword metadata.
        optional:
            Nested keyword payload.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """
        scale = config["scale"]
        bias = config["bias"]
        if payload[0]:
            x = x + bias
        if flag:
            x = x * scale
        return x


class _IoPandasModel(nn.Module):
    """Small model with params, buffers, and a submodule call carrying mixed args."""

    def __init__(self) -> None:
        """Initialize the test module graph."""
        super().__init__()
        self.block = _SummaryBlock()
        self.proj = nn.Linear(2, 2)
        self.register_buffer("offset", torch.tensor([0.5, -0.5], dtype=torch.float32))

    def build_block_inputs(
        self, x: torch.Tensor
    ) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
        """Construct the exact args passed into ``self.block``.

        Parameters
        ----------
        x:
            Input tensor for the current forward pass.

        Returns
        -------
        tuple[list[Any], dict[str, Any], dict[str, Any]]
            Positional payload, positional config, and keyword args.
        """
        payload = [True, 7, 2.5, "tag", self.offset]
        config = {
            "scale": 2.0,
            "bias": self.offset,
            "nested": [None, {"inner": x}],
        }
        kwargs = {
            "flag": False,
            "note": "hello",
            "optional": {"path": [1, self.offset]},
        }
        return payload, config, kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the summary block followed by a linear projection.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """
        payload, config, kwargs = self.build_block_inputs(x)
        x = self.block(x, payload, config, **kwargs)
        return self.proj(x)


@pytest.fixture
def io_pandas_log() -> tuple[Any, _IoPandasModel, torch.Tensor]:
    """Build a real ModelLog covering params, buffers, and module-pass summaries.

    Returns
    -------
    tuple[Any, _IoPandasModel, torch.Tensor]
        Logged model output context.
    """
    model = _IoPandasModel()
    x = torch.randn(2, 2)
    log = log_forward_pass(model, x)
    return log, model, x


@pytest.mark.parametrize(
    ("accessor", "field_order"),
    [
        (ParamAccessor({}), PARAM_LOG_FIELD_ORDER),
        (BufferAccessor({}), BUFFER_LOG_FIELD_ORDER),
    ],
)
def test_empty_accessors_to_pandas_schema(accessor: Any, field_order: list[str]) -> None:
    """Empty accessors should still export the correct schema.

    Parameters
    ----------
    accessor:
        Empty accessor under test.
    field_order:
        Expected column order.
    """
    df = accessor.to_pandas()
    assert df.empty
    assert list(df.columns) == field_order


def test_param_accessor_to_pandas_schema(
    io_pandas_log: tuple[Any, _IoPandasModel, torch.Tensor],
) -> None:
    """ParamAccessor should export one row per parameter with canonical columns.

    Parameters
    ----------
    io_pandas_log:
        Logged model fixture.
    """
    log, _, _ = io_pandas_log
    df = log.params.to_pandas()
    assert list(df.columns) == PARAM_LOG_FIELD_ORDER
    assert len(df) == len(log.params)


def test_buffer_accessor_to_pandas_schema(
    io_pandas_log: tuple[Any, _IoPandasModel, torch.Tensor],
) -> None:
    """BufferAccessor should export one row per buffer with canonical columns.

    Parameters
    ----------
    io_pandas_log:
        Logged model fixture.
    """
    log, _, _ = io_pandas_log
    df = log.buffers.to_pandas()
    assert list(df.columns) == BUFFER_LOG_FIELD_ORDER
    assert len(df) == len(log.buffers)


def test_module_pass_log_to_pandas_schema(
    io_pandas_log: tuple[Any, _IoPandasModel, torch.Tensor],
) -> None:
    """ModulePassLog should export a single canonical row including summaries.

    Parameters
    ----------
    io_pandas_log:
        Logged model fixture.
    """
    log, _, _ = io_pandas_log
    pass_log = log.modules["block:1"]
    df = pass_log.to_pandas()
    assert isinstance(pass_log, ModulePassLog)
    assert list(df.columns) == MODULE_PASS_LOG_FIELD_ORDER
    assert len(df) == 1
    assert df.loc[0, "forward_args_summary"]
    assert df.loc[0, "forward_kwargs_summary"]


def test_module_pass_summaries_are_formatted_before_gc(
    io_pandas_log: tuple[Any, _IoPandasModel, torch.Tensor],
) -> None:
    """Module-pass summary strings should survive the forward-arg GC step.

    Parameters
    ----------
    io_pandas_log:
        Logged model fixture.
    """
    log, model, x = io_pandas_log
    payload, config, kwargs = model.build_block_inputs(x)
    pass_log = log.modules["block:1"]

    expected_args_summary = format_call_arg((x, payload, config))
    expected_kwargs_summary = format_call_arg(kwargs)

    assert pass_log.forward_args is None
    assert pass_log.forward_kwargs is None
    assert pass_log.forward_args_summary == expected_args_summary
    assert pass_log.forward_kwargs_summary == expected_kwargs_summary


@pytest.mark.parametrize("surface_name", ["params", "buffers", "module_pass"])
def test_tabular_exports_round_trip(
    io_pandas_log: tuple[Any, _IoPandasModel, torch.Tensor],
    surface_name: str,
    tmp_path: Path,
) -> None:
    """CSV/JSON exports should round-trip and Parquet should succeed or hint clearly.

    Parameters
    ----------
    io_pandas_log:
        Logged model fixture.
    surface_name:
        Export surface selector.
    tmp_path:
        Temporary output directory.
    """
    log, _, _ = io_pandas_log
    if surface_name == "params":
        surface = log.params
    elif surface_name == "buffers":
        surface = log.buffers
    else:
        surface = log.modules["block:1"]

    expected_df = surface.to_pandas()

    csv_path = tmp_path / f"{surface_name}.csv"
    surface.to_csv(csv_path)
    csv_df = pd.read_csv(csv_path)
    assert list(csv_df.columns) == list(expected_df.columns)
    assert len(csv_df) == len(expected_df)

    json_path = tmp_path / f"{surface_name}.json"
    surface.to_json(json_path)
    json_df = pd.read_json(json_path, orient="records")
    assert list(json_df.columns) == list(expected_df.columns)
    assert len(json_df) == len(expected_df)

    parquet_path = tmp_path / f"{surface_name}.parquet"
    if importlib.util.find_spec("pyarrow") is None:
        with pytest.raises(ImportError, match=r"torchlens\[io\]"):
            surface.to_parquet(parquet_path)
    else:
        surface.to_parquet(parquet_path)
        parquet_df = pd.read_parquet(parquet_path)
        assert list(parquet_df.columns) == list(expected_df.columns)
        assert len(parquet_df) == len(expected_df)
