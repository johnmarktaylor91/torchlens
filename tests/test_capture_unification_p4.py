"""Phase 4 capture-unification retroactive-save regression tests."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl


class PartialConvRelu(nn.Module):
    """Conv model where only one conv output feeds a relu."""

    def __init__(self) -> None:
        """Initialize the two conv branches."""

        super().__init__()
        self.conv_relu = nn.Conv2d(1, 1, 1)
        self.conv_plain = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one relu-fed conv and one plain conv."""

        relu_parent = self.conv_relu(x)
        plain = self.conv_plain(x)
        return torch.relu(relu_parent) + plain


class ReusedConvRelu(nn.Module):
    """Conv output consumed by multiple relu successors."""

    def __init__(self) -> None:
        """Initialize the shared conv."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed one conv output to two relus."""

        conv = self.conv(x)
        return torch.relu(conv) + torch.relu(conv)


class ConvWithoutRelu(nn.Module):
    """Conv model with no matching successor."""

    def __init__(self) -> None:
        """Initialize the conv."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a conv without a relu successor."""

        return self.conv(x) + 1


class EvictedParentRelu(nn.Module):
    """Conv parent is evicted before its relu successor appears."""

    def __init__(self) -> None:
        """Initialize the conv."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Insert an unrelated op between the conv and relu."""

        conv = self.conv(x)
        _ = conv + 1
        return torch.relu(conv)


class BranchyAddParents(nn.Module):
    """Two conv parents feed one branchy successor."""

    def __init__(self) -> None:
        """Initialize both conv branches."""

        super().__init__()
        self.left = nn.Conv2d(1, 1, 1)
        self.right = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed two matching conv parents into one add successor."""

        return self.left(x) + self.right(x)


class CandidateEvictionModel(nn.Module):
    """Model that creates more candidates than the lookback payload window."""

    def __init__(self) -> None:
        """Initialize two conv candidates."""

        super().__init__()
        self.first = nn.Conv2d(1, 1, 1)
        self.second = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two conv candidates without a matching successor."""

        return self.first(x) + self.second(x)


def _conv_ops(log: tl.Trace) -> list[tl.Op]:
    """Return conv ops in execution order."""

    return [op for op in log.layer_list if op.layer_type == "conv2d"]


def _saved_conv_ops(log: tl.Trace) -> list[tl.Op]:
    """Return saved conv ops in execution order."""

    return [op for op in _conv_ops(log) if op.has_saved_activation]


def _retro_trace(model: nn.Module, x: torch.Tensor, successor: object) -> tl.Trace:
    """Run a retroactive conv save trace."""

    return tl.trace(
        model,
        x,
        save=tl.func("conv2d") & tl.followed_by(successor),  # type: ignore[arg-type]
        lookback=4,
        lookback_payload_policy="detached_raw",
        random_seed=123,
    )


def test_followed_by_saves_only_conv_feeding_relu_with_matching_payload() -> None:
    """followed_by saves exactly relu-feeding conv payloads."""

    model = PartialConvRelu()
    x = torch.randn(1, 1, 4, 4)
    full = tl.trace(model, x.clone(), layers_to_save="all", random_seed=123)
    log = _retro_trace(model, x.clone(), tl.func("relu"))
    saved = _saved_conv_ops(log)
    assert [op.label for op in saved] == [_conv_ops(log)[0].label]
    assert torch.equal(saved[0].out, full[saved[0].label].out)
    assert not _conv_ops(log)[1].has_saved_activation
    with pytest.raises(ValueError, match="not saved"):
        _ = _conv_ops(log)[1].out


def test_followed_by_multiple_successors_are_idempotent() -> None:
    """Multiple matching successors mark the same retained parent once."""

    log = _retro_trace(ReusedConvRelu(), torch.randn(1, 1, 4, 4), tl.func("relu"))
    saved = _saved_conv_ops(log)
    assert len(saved) == 1
    assert saved[0].has_saved_activation


def test_followed_by_never_appears_does_not_save_candidate() -> None:
    """Candidates are dropped unsaved when no matching successor appears."""

    log = _retro_trace(ConvWithoutRelu(), torch.randn(1, 1, 4, 4), tl.func("relu"))
    assert _saved_conv_ops(log) == []


def test_followed_by_out_of_window_parent_warns() -> None:
    """A parent evicted before its successor emits an explicit warning."""

    with pytest.warns(RuntimeWarning, match="outside the lookback window"):
        log = tl.trace(
            EvictedParentRelu(),
            torch.randn(1, 1, 4, 4),
            save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
            lookback=1,
            lookback_payload_policy="detached_raw",
            random_seed=123,
        )
    assert _saved_conv_ops(log) == []


def test_followed_by_branchy_successor_marks_all_matching_parents() -> None:
    """A multi-parent successor marks all matching parents in the window."""

    log = _retro_trace(BranchyAddParents(), torch.randn(1, 1, 4, 4), tl.func("add"))
    saved = _saved_conv_ops(log)
    assert [op.label for op in saved] == [op.label for op in _conv_ops(log)]


def test_lookback_payload_window_is_bounded_and_evicts_candidates() -> None:
    """Payload retention holds no more than K candidates and evicts older ones."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log = tl.trace(
            CandidateEvictionModel(),
            torch.randn(1, 1, 4, 4),
            save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
            lookback=1,
            lookback_payload_policy="detached_raw",
            random_seed=123,
        )
    candidates = getattr(log, "_predicate_lookback_candidates")
    assert len(candidates) <= 1
    assert tuple(candidate.raw_label for candidate in candidates) == (_conv_ops(log)[1]._label_raw,)
    assert _saved_conv_ops(log) == []
