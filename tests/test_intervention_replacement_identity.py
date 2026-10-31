"""Replacement-object identity honesty (round-3 T-RUNNABLE fix lane).

One live tensor object injected at two or more matched sites lets the LAST
fire steal the site label: every downstream consumer records the last site as
parent, the earlier site's children vanish (byte-identical payloads make the
misattributed graph validate clean), and chained edits at the orphaned site
become silent no-ops -- the exact failure class the zero-match warnings exist
to prevent, through a different door.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _TwoBranchModel(nn.Module):
    """Two parallel relu branches with one consumer each."""

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both branch products."""

        first = torch.relu(value)
        second = torch.relu(-value)
        return first * 2, second * 3


def _branch_children(log: tl.Trace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Children of the two relu sites, in branch order."""

    return tuple(log["relu_1_1"].children), tuple(log["relu_2_3"].children)


def test_replace_with_multiple_sites_keeps_parent_attribution() -> None:
    """``replace_with`` mints a distinct per-fire object when ``.to()`` no-ops.

    ``replacement.to(device, dtype)`` returns the SAME object when both
    already match, so every matched site injected one shared live tensor and
    the recorded graph hung both downstream muls off the second relu while the
    first relu sat childless.
    """

    replacement = torch.ones(3)
    log = tl.trace(
        _TwoBranchModel().eval(),
        torch.randn(3),
        intervene=tl.when(tl.func("relu"), tl.replace_with(replacement)),
    )
    first_children, second_children = _branch_children(log)
    assert first_children == ("mul_1_4",)
    assert second_children == ("mul_2_5",)


def test_user_hook_shared_object_cannot_steal_labels() -> None:
    """A custom hook returning ONE reused object is copied, never relabeled in place.

    The commit path overwrites the result's live label metadata in place, so a
    reused object carried whichever site fired last; the guard copies a result
    that already carries live capture metadata (under ``pause_logging``, so
    the internal mint never appears as a spurious clone op).
    """

    shared = torch.ones(3)

    def reuse_hook(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        """Return the same live object at every matched site."""

        del out, hook
        return shared

    log = tl.trace(
        _TwoBranchModel().eval(),
        torch.randn(3),
        intervene=tl.when(tl.func("relu"), reuse_hook),
    )
    first_children, second_children = _branch_children(log)
    assert first_children == ("mul_1_4",)
    assert second_children == ("mul_2_5",)
    # The internal mint must not enter the captured graph.
    assert not any("clone" in op.label for op in log.layer_list)


def test_inplace_mutating_hook_records_replaced_true() -> None:
    """A hook that mutates ``out`` in place and returns it is a REPLACEMENT.

    ``replaced`` was derived purely from object identity (``result is not
    current_out``), so ``out.mul_(0); return out`` changed execution and the
    saved payload while minting ``replaced=False`` and no replacement
    evidence -- downstream validation then failed forward replay in a
    capture-bug shape on a genuine intervention, and an unvalidated trace
    carried the false no-replacement claim.
    """

    def inplace_hook(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        """Zero the output in place and hand back the same object."""

        del hook
        out.mul_(0)
        return out

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
    log = tl.trace(
        model,
        torch.randn(2, 3),
        intervene=tl.when(tl.func("relu"), inplace_hook),
    )
    op = log["relu_1_2"]
    assert op.intervention_replaced is True
    fire_records = list(op.interventions)
    assert fire_records and fire_records[-1].replaced is True
    assert bool((op.out == 0).all())


def test_data_alias_mutating_hook_records_replaced_true() -> None:
    """A hook mutating through the ``.data`` alias is still a REPLACEMENT.

    grind-p5 rollup (incomplete 2289e56c): ``.data`` returns a storage-sharing
    alias with a FRESH version counter, so ``out.data.mul_(0); return out``
    changed execution and the saved payload while the ``_version`` witness read
    "no mutation" and minted ``replaced=False`` -- the exact false
    no-replacement claim the version-counter fix closed for direct in-place
    mutation.
    """

    def data_alias_hook(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        """Zero the output through the autograd-invisible ``.data`` channel."""

        del hook
        out.data.mul_(0)
        return out

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
    log = tl.trace(
        model,
        torch.randn(2, 3),
        intervene=tl.when(tl.func("relu"), data_alias_hook),
    )
    op = log["relu_1_2"]
    assert op.intervention_replaced is True
    fire_records = list(op.interventions)
    assert fire_records and fire_records[-1].replaced is True
    assert bool((op.out == 0).all())
