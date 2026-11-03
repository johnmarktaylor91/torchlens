"""Regression tests for RF view/table surface honesty.

Covers r19f findings W2A3-08 (projective ``.at('center')``), W2A3-09
(projective table ``input_op=`` filter), W2A3-10 (table accepts trace input
accessor handles), W2A3-14/16/18 (docstring + diagnostic honesty), and
W2A3-15b (``.at()`` negative-index reject contract).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field._errors import (
    ReceptiveFieldError,
    ReceptiveFieldUnavailableError,
)
from torchlens.receptive_field._view import ReceptiveFieldView, _optional_callable

# These tests read the process-global RF rule registry (builtins installed at
# ``import torchlens``); they register nothing, so no registry isolation is
# needed. Clearing the registry would strip the builtin geometric rules and
# make every conv report "geometry unavailable".


class _TwoConv(nn.Module):
    """Two-convolution chain with a single spatial model input."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 4, 3, padding=1)
        self.c2 = nn.Conv2d(4, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c2(torch.relu(self.c1(x)))


def _trace() -> object:
    """Capture the two-convolution chain on an 8x8 image."""

    return tl.trace(_TwoConv().eval(), torch.randn(1, 3, 8, 8))


# --- W2A3-08: projective .at("center") -------------------------------------


def test_projective_at_center_matches_explicit_midpoint() -> None:
    log = _trace()
    view = log["conv2d_1_1:1"].projective_field
    box = view.at("center")
    # 8x8 source grid -> windowed-axis midpoints (4, 4).
    assert box.unit == (4, 4)
    assert view.at((4, 4)).unit == box.unit


def test_projective_at_center_not_cast_as_string() -> None:
    # Regression: "center" must not become a 6-character coordinate sequence.
    log = _trace()
    box = log["conv2d_1_1:1"].projective_field.at("center")
    assert len(box.unit) == 2


def test_receptive_at_center_unchanged() -> None:
    log = _trace()
    assert log["conv2d_2_3:1"].receptive_field.at("center").unit == (4, 4)


def test_receptive_source_center_unchanged() -> None:
    log = _trace()
    box = log["conv2d_2_3:1"].receptive_field.at("center", source=log["conv2d_1_1:1"])
    assert box.unit == (4, 4)


# --- W2A3-15b: .at() negative-index reject contract ------------------------


def test_at_rejects_negative_index() -> None:
    log = _trace()
    with pytest.raises(ReceptiveFieldError, match="out of bounds"):
        log["conv2d_1_1:1"].receptive_field.at((-1, -1))


# --- W2A3-10: table accepts trace input accessor handles -------------------


def test_table_accepts_input_ops_accessor_handle() -> None:
    pytest.importorskip("pandas")
    log = _trace()
    handle = log.input_ops[0]
    canonical = next(op for op in log._layer_list if getattr(op, "is_input", False))
    via_handle = log.receptive_fields(level="op", input_op=handle)
    via_canonical = log.receptive_fields(level="op", input_op=canonical)
    assert len(via_handle.frame) > 0
    assert len(via_handle.frame) == len(via_canonical.frame)


def test_table_receptive_input_filter_works() -> None:
    pytest.importorskip("pandas")
    log = _trace()
    unfiltered = log.receptive_fields(level="op")
    filtered = log.receptive_fields(level="op", input_op=log.input_ops[0])
    assert 0 < len(filtered.frame) <= len(unfiltered.frame)


def test_table_rejects_foreign_input_handle() -> None:
    pytest.importorskip("pandas")
    log_a = _trace()
    log_b = _trace()
    with pytest.raises(ValueError):
        log_a.receptive_fields(level="op", input_op=log_b.input_ops[0])


def test_table_rejects_non_input_op() -> None:
    pytest.importorskip("pandas")
    log = _trace()
    with pytest.raises(ValueError):
        log.receptive_fields(level="op", input_op=log["conv2d_1_1:1"])


def test_table_rejects_string_input_with_typeerror() -> None:
    pytest.importorskip("pandas")
    log = _trace()
    with pytest.raises(TypeError):
        log.receptive_fields(level="op", input_op="input_1:1")


# --- W2A3-09: projective table input_op= filter -------------------------------


def test_projective_input_filter_raises_not_silent_empty() -> None:
    pytest.importorskip("pandas")
    log = _trace()
    assert len(log.projective_fields(level="op").frame) > 0
    with pytest.raises(ValueError, match="direction='receptive'"):
        log.projective_fields(level="op", input_op=log.input_ops[0])


def test_projective_unfiltered_still_returns_rows() -> None:
    pytest.importorskip("pandas")
    log = _trace()
    assert len(log.projective_fields(level="op").frame) > 0


# --- W2A3-16: unavailable-capability message has no internal task ID -------


def test_optional_callable_message_has_no_task_id() -> None:
    with pytest.raises(ReceptiveFieldUnavailableError) as excinfo:
        _optional_callable("._nonexistent_r19f_module", "nope", "projective gradient support")
    message = str(excinfo.value)
    assert "Task T" not in message
    assert "projective gradient support" in message


# --- W2A3-14 / W2A3-18: docstring honesty ----------------------------------


def test_at_docstring_documents_source_space_and_optional_params() -> None:
    doc = ReceptiveFieldView.at.__doc__ or ""
    assert "SOURCE operation's output-grid" in doc  # W2A3-14
    assert "direction:" in doc and "target:" in doc  # W2A3-18


def test_build_rf_profile_docstring_documents_direction() -> None:
    from torchlens.receptive_field._table import build_rf_profile

    assert "direction:" in (build_rf_profile.__doc__ or "")
