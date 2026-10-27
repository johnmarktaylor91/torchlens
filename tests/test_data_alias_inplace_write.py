"""An in-place write through ``t.data`` is connected to ``t`` and validates.

``attn.data.masked_fill_(mask, -inf)`` (OpenNMT GlobalAttention; Menagerie
rung 4: f-coarse2fine-text-to-sql, f-op-gan) captures the ``.data`` getter as
``detach`` and storage-alias propagation advances ``attn`` to the
``masked_fill_`` op, so the softmax that reads ``attn`` binds to it. The
capture still marked the alias receiver as an unattributed argument, on the
premise that the write never reaches the base, and validation failed
``source_provenance``. The marker now stays only where that premise holds:
captures that keep the historical topology (intervention-ready, backward-ready,
saved grads) or a receiver with no labeled base.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import last_validation_failure


class _DataMaskedFill(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.lin(x)
        attn.data.masked_fill_(x > 1.0, -float("inf"))
        return torch.softmax(attn, -1)


@pytest.mark.smoke
def test_data_alias_masked_fill_validates() -> None:
    torch.manual_seed(0)
    model = _DataMaskedFill().eval()
    x = torch.randn(3, 4) * 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="forward") is True, last_validation_failure()


@pytest.mark.smoke
def test_data_alias_write_binds_the_base_consumer() -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_DataMaskedFill().eval(), torch.randn(3, 4) * 2)
    fill = next(op for op in trace.ops if op.func_name == "masked_fill_")
    softmax = next(op for op in trace.ops if op.func_name == "softmax")
    assert fill.unattributed_tensor_args == ()
    assert list(softmax.parents) == [fill.layer_label]


@pytest.mark.smoke
def test_data_alias_marker_stays_without_base_propagation() -> None:
    """A saved-grads capture keeps the historical topology, so the marker stands."""

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            _DataMaskedFill().train(),
            torch.randn(3, 4) * 2,
            capture=tl.options.CaptureOptions(save_grads=True),
        )
    fill = next(op for op in trace.ops if op.func_name == "masked_fill_")
    assert fill.unattributed_tensor_args == ("arg0",)
