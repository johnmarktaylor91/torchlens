"""A historical label spelling never pulls an op into another op's cone.

Pass grouping renumbers ops: the raw label an op carried while the forward
ran (kept in ``lookup_keys``) can equal the FINAL label of a different op.
Child edges are spelled in final labels, so the replay cone must resolve a
child spelling to the op whose final label it is, never also to the op that
once carried it; otherwise an unrelated branch enters the cone and, under a
sticky recipe, its sites are re-edited (Decagon's relation loop).
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.intervention._replay_cone import _label_key_map, cone_of_effect


class _TwoBranches(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(x)
        b = torch.tanh(x)
        return a.sum() + b.sum()


def test_final_label_spelling_resolves_to_its_own_op_only() -> None:
    trace = tl.trace(_TwoBranches(), torch.randn(2, 3))
    relu, tanh = trace["relu_1_1"], trace["tanh_1_2"]
    # Give the relu op a historical spelling equal to the tanh op's final label.
    relu.lookup_keys = [*list(relu.lookup_keys or ()), tanh.layer_label]

    label_keys = _label_key_map(trace)
    assert label_keys[tanh.layer_label] == (tanh.label,)
    assert label_keys[relu.label] == (relu.label,)
    cone = [op.label for op in cone_of_effect(trace, [tanh])]
    assert relu.label not in cone
    assert tanh.label in cone
