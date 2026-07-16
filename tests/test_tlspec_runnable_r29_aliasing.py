"""H3 input-aliasing honesty, both directions (r29-C3).

Torch capture de-aliases model inputs, so an aliased-input runtime call that mutates a model
input is unreproducible against the recorded de-aliased path. Two r29 gaps:

* F2-hon (GAP): a VIEW-mediated input mutation (``a[0].add_(100)`` with ``a is b``) targets the
  getitem view slot, whose ``version_of`` never links back to the input, so the fail-closed
  gate missed it -> false VERIFIED. The gate now follows view-op lineage from the input slots.
* codex-F4 (OVER-TRIGGER): two DISJOINT views of one base (``base[:2]`` / ``base[2:]``) share a
  base storage pointer but touch non-overlapping bytes, so a mutation of one cannot reach the
  other; keying aliasing on the base pointer falsely diverged them. Aliasing is now the OVERLAP
  of storage byte spans.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import DivergencePolicy, NumericAttestationStatus, PathFaithfulness


def _save(model: nn.Module, args: object, path: Path) -> Path:
    """Capture and save a runnable artifact."""

    trace = tl.trace(
        model,
        args,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    trace.save(path, level="runnable", include_activations=True)
    return path


class ViewMutateModel(nn.Module):
    """Mutate a VIEW of one input in place, then read another input site."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """In-place add on the ``a[0]`` view; a fresh aliased model sees it through ``b``."""

        a[0].add_(100.0)
        return a + b


class DirectMutateModel(nn.Module):
    """Mutate one whole input in place, then read another input site."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """In-place add on ``a``."""

        a.add_(1.0)
        return a + b


@pytest.mark.smoke
def test_r29_view_mediated_input_mutation_aliased_fails_closed(tmp_path: Path) -> None:
    """``a[0].add_()`` with ``a is b`` at runtime must fail closed (F2-hon gap)."""

    ca, cb = torch.randn(4), torch.randn(4)
    path = _save(ViewMutateModel(), (ca, cb), tmp_path / "vm.tlspec")

    # Distinct inputs match the de-aliased capture -> VERIFIED (no over-trigger).
    assert tl.load(path).run(inputs=(ca.clone(), cb.clone())).report.path_faithfulness is (
        PathFaithfulness.VERIFIED
    )

    shared = torch.randn(4)
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=(shared, shared))
    diverged = tl.load(path).run(
        inputs=(shared, shared), on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.smoke
def test_r29_disjoint_views_of_one_base_do_not_over_trigger(tmp_path: Path) -> None:
    """Two disjoint views of the same base storage must NOT falsely diverge (codex-F4)."""

    base = torch.randn(8)
    path = _save(DirectMutateModel(), (base[:4].clone(), base[4:].clone()), tmp_path / "dm.tlspec")

    big = torch.randn(8)
    va, vb = big[:4], big[4:]  # share base storage, disjoint byte spans
    assert va.untyped_storage().data_ptr() == vb.untyped_storage().data_ptr()
    result = tl.load(path).run(inputs=(va, vb))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_r29_overlapping_views_still_fail_closed(tmp_path: Path) -> None:
    """Two OVERLAPPING views of one base with an in-place input mutation must fail closed."""

    base = torch.randn(8)
    path = _save(DirectMutateModel(), (base[:4].clone(), base[2:6].clone()), tmp_path / "ov.tlspec")

    big = torch.randn(8)
    va, vb = big[:4], big[2:6]  # overlap on bytes [2, 4)
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=(va, vb))
