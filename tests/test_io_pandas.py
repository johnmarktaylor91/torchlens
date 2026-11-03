"""Tests for IO-S2 pandas and file-export surfaces."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

pd = pytest.importorskip("pandas")

import torchlens as tl  # noqa: E402
from torchlens import trace as trace_fn  # noqa: E402
from torchlens._constants import (  # noqa: E402
    BUFFER_LOG_FIELD_ORDER,
    PARAM_LOG_FIELD_ORDER,
)
from torchlens._data_classes.buffer import (  # noqa: E402
    _TO_PANDAS_EXCLUDED_BUFFER_FIELDS,
    BufferAccessor,
)
from torchlens._data_classes.param import ParamAccessor  # noqa: E402


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
    """Build a real Trace covering params, buffers, and module-pass summaries.

    Returns
    -------
    tuple[Any, _IoPandasModel, torch.Tensor]
        Logged model output context.
    """
    model = _IoPandasModel()
    x = torch.randn(2, 2)
    log = trace_fn(model, x)
    return log, model, x


@pytest.mark.parametrize(
    ("accessor", "field_order"),
    [
        (ParamAccessor({}), PARAM_LOG_FIELD_ORDER),
        (
            BufferAccessor({}),
            [f for f in BUFFER_LOG_FIELD_ORDER if f not in _TO_PANDAS_EXCLUDED_BUFFER_FIELDS],
        ),
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
    tl.export.csv(surface, csv_path)
    # The capture-honesty preamble rides '#' comment lines (WT1 A-V row 24).
    csv_df = pd.read_csv(csv_path, comment="#")
    assert list(csv_df.columns) == list(expected_df.columns)
    assert len(csv_df) == len(expected_df)

    json_path = tmp_path / f"{surface_name}.json"
    tl.export.json(surface, json_path)
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "capture_honesty" in json_payload
    json_df = pd.DataFrame(json_payload["rows"])
    assert list(json_df.columns) == list(expected_df.columns)
    assert len(json_df) == len(expected_df)

    parquet_path = tmp_path / f"{surface_name}.parquet"
    if importlib.util.find_spec("pyarrow") is None:
        with pytest.raises(ImportError, match=r"torchlens\[tabular\]"):
            tl.export.parquet(surface, parquet_path)
    else:
        tl.export.parquet(surface, parquet_path)
        parquet_df = pd.read_parquet(parquet_path)
        assert list(parquet_df.columns) == list(expected_df.columns)
        assert len(parquet_df) == len(expected_df)
