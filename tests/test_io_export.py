"""Tests for IO-S3 export wrappers across all DataFrame-producing surfaces."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
import torch.nn as nn

from torchlens import log_forward_pass

PARQUET_IMPORT_ERROR = "to_parquet requires pyarrow. Install with: pip install torchlens[io]"

SURFACE_COLUMNS: dict[str, list[str]] = {
    "model_log": [
        "layer_label",
        "layer_type",
        "pass_num",
        "func_name",
        "is_input_layer",
        "is_output_layer",
    ],
    "layers": [
        "layer_label",
        "layer_type",
        "func_name",
        "num_passes",
        "is_input_layer",
        "is_output_layer",
    ],
    "modules": [
        "address",
        "module_class_name",
        "nesting_depth",
        "num_params",
        "num_layers",
        "num_passes",
    ],
    "module_log": [
        "layer_label",
        "layer_type",
        "pass_num",
        "func_name",
    ],
    "module_pass": [
        "module_address",
        "pass_num",
        "pass_label",
        "num_layers",
        "forward_args_summary",
        "forward_kwargs_summary",
    ],
    "params": [
        "address",
        "name",
        "num_params",
        "trainable",
        "module_address",
    ],
    "buffers": [
        "buffer_address",
        "name",
        "pass_num",
        "has_saved_activations",
    ],
}


class _ExportBlock(nn.Module):
    """Submodule used to exercise module-pass export summaries."""

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
        del note, optional
        if payload[0]:
            x = x + config["bias"]
        if flag:
            x = x * config["scale"]
        return x


class _IoExportModel(nn.Module):
    """Small model covering all seven tabular export surfaces."""

    def __init__(self) -> None:
        """Initialize the module graph used by the export tests."""
        super().__init__()
        self.block = _ExportBlock()
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
def io_export_log() -> Any:
    """Build a real ModelLog covering every DataFrame export surface.

    Returns
    -------
    Any
        Logged forward-pass result.
    """
    torch.manual_seed(0)
    model = _IoExportModel()
    x = torch.randn(2, 2)
    return log_forward_pass(model, x)


def _get_surface(log: Any, surface_name: str) -> Any:
    """Resolve one named export surface from the logged model.

    Parameters
    ----------
    log:
        Logged model output under test.
    surface_name:
        Export surface selector.

    Returns
    -------
    Any
        Surface exposing ``to_pandas()``, ``to_csv()``, ``to_json()``, and ``to_parquet()``.
    """
    if surface_name == "model_log":
        return log
    if surface_name == "layers":
        return log.layers
    if surface_name == "modules":
        return log.modules
    if surface_name == "module_log":
        return log.modules["block"]
    if surface_name == "module_pass":
        return log.modules["block:1"]
    if surface_name == "params":
        return log.params
    return log.buffers


def _assert_round_trip_matches(
    expected_df: pd.DataFrame, actual_df: pd.DataFrame, columns: list[str]
) -> None:
    """Assert that a selected set of export columns survives a file round-trip.

    Parameters
    ----------
    expected_df:
        Original DataFrame produced by the export surface.
    actual_df:
        DataFrame loaded back from disk.
    columns:
        Stable columns that should match after serialization.
    """
    assert list(actual_df.columns) == list(expected_df.columns)
    expected_subset = expected_df.loc[:, columns].astype(object)
    actual_subset = actual_df.loc[:, columns].astype(object)
    expected_subset = expected_subset.where(pd.notna(expected_subset), "<NA>").astype(str)
    actual_subset = actual_subset.where(pd.notna(actual_subset), "<NA>").astype(str)
    pd.testing.assert_frame_equal(
        expected_subset.reset_index(drop=True),
        actual_subset.reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize("surface_name", list(SURFACE_COLUMNS))
def test_tabular_exports_round_trip_csv_and_json(
    io_export_log: Any, surface_name: str, tmp_path: Path
) -> None:
    """CSV and JSON exports should preserve schema and stable values.

    Parameters
    ----------
    io_export_log:
        Logged model fixture.
    surface_name:
        Export surface selector.
    tmp_path:
        Temporary output directory.
    """
    surface = _get_surface(io_export_log, surface_name)
    expected_df = surface.to_pandas()
    stable_columns = SURFACE_COLUMNS[surface_name]

    csv_path = tmp_path / f"{surface_name}.csv"
    surface.to_csv(csv_path)
    csv_df = pd.read_csv(csv_path)
    _assert_round_trip_matches(expected_df, csv_df, stable_columns)

    json_path = tmp_path / f"{surface_name}.json"
    surface.to_json(json_path, orient="records")
    json_df = pd.read_json(json_path, orient="records")
    _assert_round_trip_matches(expected_df, json_df, stable_columns)


@pytest.mark.parametrize("surface_name", list(SURFACE_COLUMNS))
def test_tabular_exports_round_trip_parquet(
    io_export_log: Any, surface_name: str, tmp_path: Path
) -> None:
    """Parquet exports should preserve schema and stable values when pyarrow exists.

    Parameters
    ----------
    io_export_log:
        Logged model fixture.
    surface_name:
        Export surface selector.
    tmp_path:
        Temporary output directory.
    """
    pytest.importorskip("pyarrow")

    surface = _get_surface(io_export_log, surface_name)
    expected_df = surface.to_pandas()
    stable_columns = SURFACE_COLUMNS[surface_name]

    parquet_path = tmp_path / f"{surface_name}.parquet"
    surface.to_parquet(parquet_path)
    parquet_df = pd.read_parquet(parquet_path)
    _assert_round_trip_matches(expected_df, parquet_df, stable_columns)


@pytest.mark.parametrize("surface_name", list(SURFACE_COLUMNS))
def test_to_parquet_requires_pyarrow_when_missing(
    io_export_log: Any, surface_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All parquet wrappers should raise the shared install-hint message without pyarrow.

    Parameters
    ----------
    io_export_log:
        Logged model fixture.
    surface_name:
        Export surface selector.
    tmp_path:
        Temporary output directory.
    monkeypatch:
        Pytest monkeypatch fixture.
    """
    surface = _get_surface(io_export_log, surface_name)
    monkeypatch.setitem(sys.modules, "pyarrow", None)

    with pytest.raises(ImportError, match=re.escape(PARQUET_IMPORT_ERROR)):
        surface.to_parquet(tmp_path / f"{surface_name}.parquet")
