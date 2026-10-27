"""Structural argument slots whose value is pinned (``validation/_structural_slot_checks.py``).

Each carve-out excuses one op family's one argument slot from perturbation, because
every other value is invalid for the call. Each has a positive test (the slot from
the real Menagerie row is excused) and an armed negative: with the condition not met,
the edge's replay frozen to fake a dead edge still FAILS ``perturbation_insensitive``.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation import validate_forward_pass
from torchlens.validation.core import (
    _check_whether_func_on_saved_parents_yields_saved_tensor,
)


def _capture(model: nn.Module, *inputs: torch.Tensor) -> Any:
    """Capture a full-save trace with saved argument values."""

    torch.manual_seed(0)
    return tl.trace(
        model.eval(),
        list(inputs) if len(inputs) > 1 else inputs[0],
        capture=CaptureOptions(layers_to_save="all", save_arg_values=True, random_seed=0),
    )


def _validates(model: nn.Module, x: torch.Tensor) -> bool:
    """Public-path validation with provenance warnings silenced."""

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate_forward_pass(model.eval(), x))


def _op(trace: Any, func_name: str) -> Any:
    """Return the first op with ``func_name``."""

    return next(op for op in trace.layer_list if op.func_name == func_name)


def _decision(trace: Any, op: Any, parent: str) -> tuple[str, str]:
    """Run the perturbation check of ``parent`` -> ``op`` and return (decision, reason)."""

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[parent]
    )
    return result.decision, result.reason


def _assert_frozen_edge_fails(trace: Any, op: Any, parent: str) -> None:
    """Freeze ``op``'s replay to its saved output and assert ``parent`` fails."""

    saved = op.out.detach().clone()
    object.__setattr__(op, "func", lambda *args, **kwargs: saved.clone())
    assert _decision(trace, op, parent) == ("failed", "perturbation_insensitive")


_EXEMPT = ("exempted", "pre_perturbation_exemption")

# ------------------------------------------------ packed gru / rnn batch_sizes


class _PackedRecurrence(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.rnn = nn.GRU(3, 4) if kind == "gru" else nn.RNN(3, 4, nonlinearity=kind)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # A packed sequence, as m20812's GRU encoder runs it.
        _, hidden = self.rnn(pack_padded_sequence(x, lengths))
        return hidden


def _packed_trace(kind: str) -> tuple[Any, Any]:
    trace = _capture(_PackedRecurrence(kind), torch.randn(3, 2, 3), torch.tensor([3, 2]))
    func_name = "gru" if kind == "gru" else f"rnn_{kind}"
    return trace, _op(trace, func_name)


def test_packed_recurrence_batch_sizes_slot_is_exempt() -> None:
    for kind in ("gru", "tanh", "relu"):
        trace, op = _packed_trace(kind)
        assert _decision(trace, op, op.parent_arg_positions["args"][1]) == _EXEMPT


def test_packed_recurrence_data_slot_still_fails_when_frozen() -> None:
    trace, op = _packed_trace("gru")
    _assert_frozen_edge_fails(trace, op, op.parent_arg_positions["args"][0])
