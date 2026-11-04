"""An if-chain whose only tensor-tested arm is an ``elif`` keeps its conditional records consistent.

transformers 5.18.0's flash-attention wrapper has this shape: ``if attention_mask is not
None`` (pure Python, never witnessed), ``elif ... or is_packed`` (a tensor bool), ``else``
(the attention call). The elif arm must name the bool op that evaluated its test, or the
conditional has no terminal bool anywhere and every body op fails invariant 15.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import torchlens as tl
from torchlens.validation import last_validation_failure


class ElifOnlyWitnessModel(nn.Module):
    """``if`` on a Python value, ``elif`` on a tensor bool (False), ``else`` body runs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = None
        is_packed = (x - x).abs().sum().bool()
        if mask is not None:
            y = x * 2
        elif is_packed:
            y = x * 3
        else:
            y = x * 4
        return y


def test_elif_only_witnessed_if_chain_validates_and_names_its_terminal_bool() -> None:
    model = ElifOnlyWitnessModel()
    x = torch.arange(6.0).reshape(2, 3)
    assert tl.validate(model, (x,), scope="forward") is True
    assert last_validation_failure() is None
    trace = tl.trace(model, (x,))
    try:
        assert len(trace.graph.conditionals) == 1
        conditional = trace.graph.conditionals[0]
        assert [arm.kind for arm in conditional.arms] == ["then", "elif", "else"]
        then_arm, elif_arm, else_arm = conditional.arms
        assert then_arm.terminal_bool_op_label is None
        assert then_arm.condition_evaluated is False
        assert elif_arm.condition_evaluated is True
        assert elif_arm.terminal_bool_op_label is not None
        assert elif_arm.terminal_bool_op_label.startswith("bool")
        assert elif_arm.bool_value_at_run is False
        assert elif_arm.fired is False
        assert else_arm.fired is True
        assert conditional.fired_arm_kind == "else"
        body = [op.label for op in trace._layer_list if op.conditional_branch_stack]
        assert body, "the else body op must carry the arm membership"
    finally:
        trace.cleanup()
