"""Parameter truth (A2/A3/A12): object identity, executed split, tri-state.

Lane A07 (megasprint 2026-08-27). Spec: trilabs/summary/MEMO.md 3.3 + build item 7.
Canonical rule: total_params == sum(p.numel() for p in model.parameters()) by
construction; ties counted once and NAMED; storage aliasing disclosed, never merged.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class TiedLM(nn.Module):
    """Tied-embedding toy: emb.weight IS head.weight (one Parameter object)."""

    def __init__(self) -> None:
        """Tie the head to the embedding."""

        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.head = nn.Linear(4, 10, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed then project back to vocab."""

        return self.head(self.emb(x))


class SharedViews(nn.Module):
    """Fused-QKV-style toy: DISTINCT Parameters that are views of ONE storage.

    torch's identity rule counts 24 (12 + 12); storage-pointer dedup would
    collapse the overlap to 20. The count must be 24.
    """

    def __init__(self) -> None:
        """Create two overlapping parameter views over one 20-element buffer."""

        super().__init__()
        flat = torch.randn(20)
        self.a = nn.Parameter(flat[:12].view(3, 4))
        self.b = nn.Parameter(flat[8:20].view(3, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume both views."""

        return x * self.a.sum() + self.b.sum()


class DeadSubmodule(nn.Module):
    """Declared-but-never-called submodule: params exist, never execute."""

    def __init__(self) -> None:
        """Declare a used and an unused branch."""

        super().__init__()
        self.used = nn.Linear(6, 6, bias=False)
        self.dead = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run only the used branch."""

        return self.used(x)


class PartialFreeze(nn.Module):
    """One trainable and one frozen tensor inside the same module."""

    def __init__(self) -> None:
        """Freeze the bias only."""

        super().__init__()
        self.fc = nn.Linear(4, 4, bias=True)
        self.fc.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear."""

        return self.fc(x)


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Metadata-only capture."""

    return tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=None))


@pytest.mark.smoke
def test_tied_params_counted_once_by_object_identity() -> None:
    """A2: tied params are ONE identity; total matches torch's own sum."""

    model = TiedLM()
    log = _capture(model, torch.tensor([[1, 2]]))
    try:
        assert log.num_params == sum(p.numel() for p in model.parameters())
        assert log.num_params == 40
        assert log.num_param_tensors == 1
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_tie_is_named_and_per_path_total_printed() -> None:
    """A2: the footer prints BOTH totals when they differ and names the tie."""

    model = TiedLM()
    log = _capture(model, torch.tensor([[1, 2]]))
    try:
        assert log.tied_param_groups == (("emb.weight", "head.weight"),)
        assert log.num_params_by_path == 80
        text = log.summary()
        assert "Params: 40 unique (parameter identity)" in text
        assert "emb.weight = head.weight" in text
        assert "per-module-path total: 80" in text
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_shared_storage_views_stay_distinct_parameters() -> None:
    """A2: storage-pointer overlap never merges distinct Parameters (24 not 20)."""

    model = SharedViews()
    log = _capture(model, torch.randn(2, 3, 4))
    try:
        assert log.num_params == 24
        assert log.num_params == sum(p.numel() for p in model.parameters())
        # Two distinct parameter identities, no tie group (different objects).
        assert log.num_param_tensors == 2
        assert log.tied_param_groups == ()
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_declared_executed_unexecuted_split_named() -> None:
    """A3: declared-but-unexecuted params split out and NAMED, never dropped."""

    model = DeadSubmodule()
    log = _capture(model, torch.randn(2, 6))
    try:
        assert log.num_params == 36 + 21  # declared: used 6*6 + dead 6*3+3
        assert log.num_params_executed == 36
        assert log.num_params_unexecuted == 21
        assert set(log.unexecuted_param_names) == {"dead.weight", "dead.bias"}
        text = log.summary()
        assert "Never executed: 21 params" in text
        assert "dead.weight" in text
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_trainability_is_a_tri_state() -> None:
    """A12: a mixed module reads 'partial', never a boolean OR 'yes'."""

    model = PartialFreeze()
    log = _capture(model, torch.randn(2, 4))
    try:
        text = log.summary()
        assert "| partial" in text
        assert log.num_params_trainable == 16
        assert log.num_params_frozen == 4
        assert "trainable: 16 (80.0%); frozen: 4" in text
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_headline_matches_torch_on_plain_models() -> None:
    """The identity rule reproduces torch's count on an untied model too."""

    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Linear(6, 6))
    log = _capture(
        nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.Flatten(), nn.Linear(4 * 8 * 8, 5)),
        torch.randn(1, 3, 8, 8),
    )
    try:
        assert log.num_params == 3 * 4 * 9 + 4 + (4 * 8 * 8) * 5 + 5
    finally:
        log.cleanup()
    del model
