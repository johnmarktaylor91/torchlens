"""Tests for access-time theoretical roofline properties."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import constants
from torchlens.quantities import Bytes


class _ViewModel(nn.Module):
    """Model composed of logical view operations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape and transpose without logical tensor movement.

        Parameters
        ----------
        x:
            Contiguous input tensor.

        Returns
        -------
        torch.Tensor
            Transposed view.
        """

        return x.reshape(2, 6).transpose(0, 1)


def _op(trace: tl.Trace, func_name: str) -> Any:
    """Return the first operation with a requested function name.

    Parameters
    ----------
    trace:
        Captured trace.
    func_name:
        Operation function name.

    Returns
    -------
    Any
        Matching operation.
    """

    return next(op for op in trace.layer_list if op.func_name == func_name)


def _bundle_bytes(path: Path) -> dict[str, bytes]:
    """Return relative file bytes for one tlspec bundle.

    Parameters
    ----------
    path:
        Bundle directory.

    Returns
    -------
    dict[str, bytes]
        Relative-path to exact file contents.
    """

    return {
        str(item.relative_to(path)): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_linear_theoretical_bytes_and_intensity() -> None:
    """Linear traffic is input plus weights/bias read and output written."""

    trace = tl.trace(nn.Linear(4, 3), torch.ones(2, 4, dtype=torch.float32))
    op = _op(trace, "linear")
    # Read: input 2*4*4 = 32; params (3*4 + 3)*4 = 60; total 92 bytes.
    # Write: output 2*3*4 = 24 bytes.
    assert op.bytes_read == Bytes(92)
    assert op.bytes_written == Bytes(24)
    assert op.arithmetic_intensity == pytest.approx(float(op.flops_forward) / 116)


def test_conv2d_theoretical_bytes_and_intensity() -> None:
    """Conv2d traffic includes the input, kernel, bias, and output once."""

    trace = tl.trace(
        nn.Conv2d(2, 3, kernel_size=3, bias=True),
        torch.ones(1, 2, 5, 5, dtype=torch.float32),
    )
    op = _op(trace, "conv2d")
    # Read: input 1*2*5*5*4 = 200; params (3*2*3*3 + 3)*4 = 228; total 428.
    # Write: output 1*3*3*3*4 = 108 bytes; total traffic 536 bytes.
    assert op.bytes_read == Bytes(428)
    assert op.bytes_written == Bytes(108)
    assert op.arithmetic_intensity == pytest.approx(float(op.flops_forward) / 536)


def test_view_ops_report_zero_logical_traffic() -> None:
    """Logical views do not overcount their tensor shape as moved bytes."""

    trace = tl.trace(_ViewModel(), torch.ones(3, 4))

    for func_name in ("reshape", "transpose"):
        op = _op(trace, func_name)
        assert op.bytes_read == Bytes(0)
        assert op.bytes_written == Bytes(0)
        assert op.arithmetic_intensity is None


def test_roofline_properties_are_absent_from_all_field_orders() -> None:
    """Computed roofline names never enter a stored schema field order."""

    property_names = {"bytes_read", "bytes_written", "arithmetic_intensity"}
    for constant_name, value in vars(constants).items():
        if constant_name.endswith("_FIELD_ORDER"):
            assert property_names.isdisjoint(set(value)), constant_name

    trace = tl.trace(nn.Linear(2, 2), torch.ones(1, 2))
    state = _op(trace, "linear").__getstate__()
    assert property_names.isdisjoint(state)


def _bundle_bytes_sans_save_time(path: Path) -> dict[str, bytes]:
    """Return bundle file bytes with the manifest's ``created_at`` normalized out.

    Parameters
    ----------
    path:
        Bundle directory.

    Returns
    -------
    dict[str, bytes]
        Relative-path to file contents; ``manifest.json`` is re-serialized
        canonically without its save-time ``created_at`` stamp.
    """

    files = _bundle_bytes(path)
    manifest = json.loads(files["manifest.json"])
    manifest.pop("created_at", None)
    files["manifest.json"] = json.dumps(manifest, sort_keys=True).encode("utf-8")
    return files


def test_roofline_access_does_not_change_pickle_or_tlspec_bytes(tmp_path: Path) -> None:
    """Property access adds no cached state to pickle or portable bundles.

    The bundle comparison normalizes ``manifest.json``'s ``created_at`` out
    because the claim is that roofline access changes no persisted bytes IT
    owns, and the per-second save timestamp is save-time provenance that two
    consecutive saves legitimately disagree on across a second boundary
    (GATE-FIX row 7: 3/16 bare-tip flakes were exactly that stamp).
    """

    trace = tl.trace(nn.Linear(2, 2), torch.ones(1, 2))
    op = _op(trace, "linear")
    before_pickle = pickle.dumps(trace)
    before_bundle = tmp_path / "before.tlspec"
    after_bundle = tmp_path / "after.tlspec"
    tl.save(trace, before_bundle)

    _ = op.bytes_read, op.bytes_written, op.arithmetic_intensity

    after_pickle = pickle.dumps(trace)
    tl.save(trace, after_bundle)
    assert before_pickle == after_pickle
    assert _bundle_bytes_sans_save_time(before_bundle) == _bundle_bytes_sans_save_time(after_bundle)

    loaded = tl.load(after_bundle)
    loaded_op = _op(loaded, "linear")
    assert {
        "bytes_read",
        "bytes_written",
        "arithmetic_intensity",
    }.isdisjoint(loaded_op.__getstate__())
