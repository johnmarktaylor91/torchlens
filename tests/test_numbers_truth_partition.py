"""The identity-partition invariant suite (A1, accuracy pyramid top).

Lane A07 (megasprint 2026-08-27). Spec: trilabs/summary/MEMO.md 3.1: every
parameter identity, executed op event, and tracked tensor identity is owned by
exactly ONE accounting row. Alias (input/output) rows display but own nothing
-- input rows own only the external inputs' bytes.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class BranchCNN(nn.Module):
    """Small branching CNN with BN buffers and a functional add."""

    def __init__(self) -> None:
        """Initialize two branches and a head."""

        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.left = nn.Conv2d(4, 4, 1)
        self.right = nn.Conv2d(4, 4, 1)
        self.head = nn.Linear(4 * 4 * 4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both branches and merge with a bare functional add."""

        x = self.norm(self.stem(x))
        x = self.left(x) + self.right(x)
        return self.head(x.flatten(1))


class TiedRecurrent(nn.Module):
    """Tied embedding + a 3-pass reused linear."""

    def __init__(self) -> None:
        """Initialize tied and reused layers."""

        super().__init__()
        self.emb = nn.Embedding(12, 6)
        self.mix = nn.Linear(6, 6, bias=False)
        self.head = nn.Linear(6, 12, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed, mix three times, project back."""

        h = self.emb(x)
        for _ in range(3):
            h = self.mix(h)
        return self.head(h)


def _fixture_traces() -> list[tuple[str, tl.Trace]]:
    """Capture the partition fixtures (metadata-only)."""

    fixtures: list[tuple[str, tl.Trace]] = []
    cnn = BranchCNN()
    cnn.eval()
    fixtures.append(
        (
            "branch_cnn",
            tl.trace(
                cnn, torch.randn(2, 3, 4, 4), capture=tl.options.CaptureOptions(layers_to_save=None)
            ),
        )
    )
    tied = TiedRecurrent()
    fixtures.append(
        (
            "tied_recurrent",
            tl.trace(
                tied,
                torch.tensor([[1, 2, 3]]),
                capture=tl.options.CaptureOptions(layers_to_save=None),
            ),
        )
    )
    return fixtures


@pytest.mark.smoke
def test_compute_partition_totals_are_owned_exactly_once() -> None:
    """total_flops_forward == sum over REAL op rows; alias rows own None."""

    for name, log in _fixture_traces():
        try:
            real_ops = [op for op in log.layer_list if not (op.is_input or op.is_output)]
            partition_sum = sum(
                int(op.flops_forward) for op in real_ops if op.flops_forward is not None
            )
            assert int(log.total_flops_forward) == partition_sum, name
            for op in log.layer_list:
                if op.is_output:
                    assert op.flops_forward is None, (name, op.label)
                if op.is_input:
                    assert int(op.flops_forward or 0) == 0, (name, op.label)
        finally:
            log.cleanup()


@pytest.mark.smoke
def test_tracked_bytes_partition_input_inclusive_alias_exclusive() -> None:
    """Tracked bytes = external input bytes + real op bytes; output rows own 0."""

    for name, log in _fixture_traces():
        try:
            owned = sum(int(op.activation_memory or 0) for op in log.layer_list if not op.is_output)
            assert int(log.total_activation_memory) == owned, name
        finally:
            log.cleanup()


@pytest.mark.smoke
def test_param_partition_each_identity_owned_once() -> None:
    """Every parameter identity appears exactly once in param_logs; totals match torch."""

    for name, log in _fixture_traces():
        try:
            barcodes = [pl.barcode for pl in log.param_logs]
            assert len(barcodes) == len(set(barcodes)), name
            addresses = [pl.address for pl in log.param_logs]
            assert len(addresses) == len(set(addresses)), name
            assert log.num_params == sum(int(pl.num_params) for pl in log.param_logs), name
        finally:
            log.cleanup()


@pytest.mark.smoke
def test_module_param_rollups_match_torch_subtree_counts() -> None:
    """Each module row's param count equals torch's own subtree unique count."""

    model = BranchCNN()
    model.eval()
    log = tl.trace(
        model, torch.randn(2, 3, 4, 4), capture=tl.options.CaptureOptions(layers_to_save=None)
    )
    try:
        for address in ("stem", "norm", "left", "right", "head"):
            submodule = model.get_submodule(address)
            torch_count = sum(p.numel() for p in submodule.parameters())
            assert log.modules[address].num_params == torch_count, address
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_multipass_layer_owns_all_its_pass_events() -> None:
    """A 3-pass layer's aggregate owns exactly its three pass events' compute."""

    model = TiedRecurrent()
    log = tl.trace(
        model,
        torch.tensor([[1, 2, 3]]),
        capture=tl.options.CaptureOptions(layers_to_save=None),
    )
    try:
        layer = log["linear_1_2"]  # the reused mix layer (3 passes)
        per_pass = [int(op.flops_forward or 0) for op in layer.ops.values()]
        assert len(per_pass) == 3
        assert int(layer.total_flops_forward) == sum(per_pass)
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_boundary_rows_render_dash_in_every_additive_cell() -> None:
    """The costreport D7 CI plant: a numeric additive cell on a boundary row fails."""

    model = BranchCNN()
    model.eval()
    log = tl.trace(
        model, torch.randn(2, 3, 4, 4), capture=tl.options.CaptureOptions(layers_to_save=None)
    )
    try:
        frame = log.profile().to_pandas()
        boundary = frame[frame["kind"] == "boundary"]
        assert not boundary.empty
        assert boundary["flops"].isna().all()
        assert boundary["time"].isna().all()
        assert boundary["param_count"].isna().all()
        outputs = boundary[boundary["name"].str.startswith("output")]
        assert outputs["activation_memory"].isna().all()
    finally:
        log.cleanup()
