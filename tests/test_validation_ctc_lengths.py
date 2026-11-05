"""``ctc_loss`` length edges are perturbation-proved with lengths rotated across the batch.

The generic integer perturbation of ``target_lengths`` changed their sum, so with
concatenated targets the perturbed replay raised ("Expected tensor to have size ...
for argument #2 'targets'") and a real length edge stayed unverified. The lengths
are now rolled by one batch position, which keeps the sum and the values, while the
perturbation check still requires the output to move.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlens.validation import validate_forward_pass
from torchlens.validation._ctc_lengths_perturbation import ctc_length_rotation_values


class _CtcHead(nn.Module):
    """A CTC head over concatenated targets, as an ASR model computes its loss."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 5)

    def forward(
        self,
        feats: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = self.proj(feats).log_softmax(-1).transpose(0, 1)
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction="sum")


def _layer() -> SimpleNamespace:
    """Return a minimal ``ctc_loss`` op record with the lengths at ``args[2]`` and ``args[3]``."""

    return SimpleNamespace(
        func_name="ctc_loss",
        parent_arg_positions={
            "args": {0: "log_probs", 1: "targets", 2: "input_lengths", 3: "target_lengths"},
            "kwargs": {},
        },
    )


def test_ctc_length_edges_validate() -> None:
    """Concatenated targets with unequal lengths validate instead of staying unverified."""

    feats = torch.randn(2, 6, 4)
    targets = torch.tensor([1, 2, 3, 4, 1])
    inputs = (feats, targets, torch.tensor([6, 5]), torch.tensor([3, 2]))
    assert validate_forward_pass(_CtcHead(), inputs, seed=0) is True


def test_rotated_lengths_keep_sum_and_move() -> None:
    """Both length slots are rolled; the sum and the multiset of values are unchanged."""

    lengths = torch.tensor([3, 2, 4])
    for slot in ("input_lengths", "target_lengths"):
        rolled = ctc_length_rotation_values(_layer(), slot, lengths)
        assert rolled is not None
        assert not torch.equal(rolled, lengths)
        assert int(rolled.sum()) == int(lengths.sum())
        assert sorted(rolled.tolist()) == sorted(lengths.tolist())


def test_unmovable_lengths_and_other_operands_fall_through() -> None:
    """One sample, equal lengths and the non-length operands keep the generic perturbation."""

    assert ctc_length_rotation_values(_layer(), "target_lengths", torch.tensor([4])) is None
    assert ctc_length_rotation_values(_layer(), "target_lengths", torch.tensor([2, 2])) is None
    assert ctc_length_rotation_values(_layer(), "targets", torch.tensor([1, 2])) is None
    assert ctc_length_rotation_values(_layer(), "input_lengths", torch.tensor([1.0, 2.0])) is None
