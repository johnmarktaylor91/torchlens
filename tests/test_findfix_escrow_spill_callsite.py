"""The ``escrow_spill_detached`` disclosure points at the caller's own line.

A deferred op-label save escrows every op's output and spills the oldest to
temporary disk past the escrow RAM budget; a spilled payload comes back
detached, and the capture discloses it with the coded ``escrow_spill_detached``
``TorchLensWarning``. Like the other coded TorchLens disclosures it is
attributed to the user's call, so warning filters keyed on the ``torchlens``
module (``error::UserWarning:torchlens``) treat it as the user's warning rather
than as an internal one. The budget is shrunk to one byte so a four-op model
spills.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture.plan import CapturePlan
from torchlens.errors import TorchLensWarning
from torchlens.options import CaptureOptions


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(x)))


@pytest.fixture
def one_byte_escrow(monkeypatch: pytest.MonkeyPatch) -> None:
    real_compile = CapturePlan.compile.__func__  # type: ignore[attr-defined]

    def tiny_budget_compile(cls: Any, **kwargs: Any) -> Any:
        plan = real_compile(cls, **kwargs)
        return dataclasses.replace(
            plan,
            retention_profile=dataclasses.replace(
                plan.retention_profile, activation_ram_budget_bytes=1
            ),
        )

    monkeypatch.setattr(CapturePlan, "compile", classmethod(tiny_budget_compile))


def _model_input_and_first_label() -> tuple[nn.Module, torch.Tensor, str]:
    torch.manual_seed(0)
    model = _Mlp()
    x = torch.randn(3, 4)
    first_linear = next(
        op.layer_label for op in tl.trace(model, x).layer_list if op.func_name == "linear"
    )
    return model, x, first_linear


def _spill_records(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [
        record
        for record in caught
        if isinstance(record.message, TorchLensWarning)
        and record.message.fields.get("code") == "escrow_spill_detached"
    ]


@pytest.mark.usefixtures("one_byte_escrow")
def test_escrow_spill_warning_names_the_callers_file() -> None:
    model, x, first_linear = _model_input_and_first_label()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model, x, capture=CaptureOptions(layers_to_save=[first_linear]))

    spills = _spill_records(caught)
    assert len(spills) == 1, [str(record.message) for record in caught]
    assert spills[0].filename == __file__
    assert spills[0].message.fields["detached_spill_count"] == 1


@pytest.mark.usefixtures("one_byte_escrow")
def test_escrow_spill_warning_is_not_promoted_by_the_torchlens_module_filter() -> None:
    """The suite's ``error::UserWarning:torchlens`` filter leaves the user's warning alone."""

    model, x, first_linear = _model_input_and_first_label()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("error", category=UserWarning, module=r"torchlens(\.|$)")
        trace = tl.trace(model, x, capture=CaptureOptions(layers_to_save=[first_linear]))

    assert len(_spill_records(caught)) == 1
    saved = trace[first_linear]
    assert saved.has_saved_activation
    assert torch.equal(saved.out, model.first(x).detach())
