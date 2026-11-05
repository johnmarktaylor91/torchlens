"""``masked_scatter`` mask edges are perturbation-proved with a count-preserving mask.

The generic boolean perturbation changed the number of true mask entries, so the
perturbed replay raised ("Number of elements of source < number of ones in mask")
and a real mask edge stayed unverified, leaving the whole trace unverified. The
mask is now perturbed by moving its true entries, which keeps the call valid while
the perturbation check still requires the output to move.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from torchlens.validation import validate_forward_pass
from torchlens.validation._masked_scatter_perturbation import masked_scatter_mask_values


class _TrilScatter(nn.Module):
    """Write a lower-triangular factor the way a full-covariance mixture does."""

    def __init__(self, dim: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.diag = nn.Linear(4, dim)
        self.off = nn.Linear(4, dim * (dim - 1) // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(self.dim, self.dim, dtype=torch.bool).tril(-1)
        factor = torch.diag_embed(self.diag(x).exp())
        return factor.masked_scatter(mask, self.off(x))


class _SparseMaskScatter(nn.Module):
    """``masked_scatter`` under a computed one-hot row mask with a source of exactly one per row.

    Any draw with more than one true entry needs more source elements than exist, so the
    generic boolean perturbation raises here on every seed.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(16) == 3
        return torch.zeros(x.shape[0], 16).masked_scatter(mask, x[:, :1] * 2.0)


def _layer(mask_label: str = "mask", source: torch.Tensor | None = None) -> SimpleNamespace:
    """Return a minimal op record with the mask at ``args[1]``."""

    target = torch.zeros(2, 3)
    return SimpleNamespace(
        func_name="masked_scatter",
        parent_arg_positions={"args": {0: "input", 1: mask_label, 2: "source"}, "kwargs": {}},
        saved_args=(target, None, torch.ones(6) if source is None else source),
        saved_kwargs={},
    )


def test_broadcast_tril_mask_edge_validates() -> None:
    """A computed tril mask broadcast over a batch validates instead of staying unverified."""

    assert validate_forward_pass(_TrilScatter(), torch.randn(5, 2, 4), seed=0) is True


def test_sparse_mask_edge_validates() -> None:
    """A mask whose true count equals the source size validates instead of staying unverified."""

    assert validate_forward_pass(_SparseMaskScatter(), torch.randn(3, 4), seed=0) is True


def test_all_true_mask_clears_one_entry() -> None:
    """An all-true mask (a roll is the identity) gets one entry cleared, which stays valid."""

    mask = torch.ones(2, 3, dtype=torch.bool)
    perturbed = masked_scatter_mask_values(_layer(), "mask", mask)
    assert perturbed is not None and int(perturbed.sum()) == mask.numel() - 1


def test_rolled_mask_keeps_true_count_and_moves() -> None:
    """The perturbed mask differs from the saved one and has the same number of true entries."""

    mask = torch.tensor([[True, False, False], [True, True, False]])
    perturbed = masked_scatter_mask_values(_layer(), "mask", mask)

    assert perturbed is not None
    assert perturbed.shape == mask.shape
    assert not torch.equal(perturbed, mask)
    assert int(perturbed.sum()) == int(mask.sum())


def test_all_false_mask_needs_a_long_enough_source() -> None:
    """Setting an entry of an all-false mask is offered only when the source can fill it."""

    mask = torch.zeros(2, 3, dtype=torch.bool)

    grown = masked_scatter_mask_values(_layer(), "mask", mask)
    assert grown is not None and int(grown.sum()) == 1
    assert masked_scatter_mask_values(_layer(source=torch.ones(0)), "mask", mask) is None


def test_other_operands_keep_the_generic_perturbation() -> None:
    """Only the mask slot is handled here; the input and source slots fall through."""

    values = torch.tensor([True, False])
    assert masked_scatter_mask_values(_layer(), "input", values) is None
    assert masked_scatter_mask_values(_layer(), "mask", torch.tensor([1.0, 0.0])) is None
