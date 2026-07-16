"""Structural non-tensor input-tree honesty: empty containers + key types (r29-C2).

The H1 non-tensor leaf-path SET witness records only LEAF paths, so it was blind to two
structural changes that steer unobserved Python control flow:

* An EXTRA/removed EMPTY container (codex-F1): ``{'flag': {}}`` adds no leaf path, so
  ``if 'flag' in d`` / ``if not lst`` branches would replay the captured arm for a runtime
  input whose empty container was added or removed -- a false VERIFIED+ATTESTED. Empty
  containers now contribute a synthetic KIND marker leaf so the change diverges.
* A bool-vs-int dict KEY twin (F6): ``{True: v}`` vs ``{1: v}`` collide in the raw leaf-path
  set (``hash(True) == hash(1)``); the bool key is now type-tagged so the twin diverges.

The over-trigger guard: the SAME structure (same empty containers, same key types) on the
same input must stay VERIFIED.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness


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


class FlagPresenceBranch(nn.Module):
    """Branch on the presence of a dict key whose value is an EMPTY container."""

    def forward(self, x: torch.Tensor, d: dict) -> torch.Tensor:
        """Add or subtract by whether ``'flag'`` is present."""

        return x + 100 if "flag" in d else x - 100


class EmptyListBranch(nn.Module):
    """Branch on whether a list input is empty."""

    def forward(self, x: torch.Tensor, lst: list) -> torch.Tensor:
        """Add or subtract by list emptiness."""

        return x + 1 if len(lst) == 0 else x - 1


class BoolKeyBranch(nn.Module):
    """Read a value under a BOOL dict key -- distinct from the equal-valued int key."""

    def forward(self, x: torch.Tensor, d: dict) -> torch.Tensor:
        """Use a bool key present only when the runtime dict keeps the bool type."""

        has_bool = any(isinstance(k, bool) for k in d)
        return x + 2 if has_bool else x - 2


@pytest.mark.smoke
def test_r29_extra_empty_container_diverges(tmp_path: Path) -> None:
    """Removing an empty-valued dict key must diverge, never false VERIFIED."""

    x = torch.randn(3)
    path = _save(FlagPresenceBranch(), (x, {"flag": {}}), tmp_path / "flag.tlspec")

    same = tl.load(path).run(inputs=(x.clone(), {"flag": {}}))
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=(x.clone(), {}))


@pytest.mark.smoke
def test_r29_empty_list_to_nonempty_diverges(tmp_path: Path) -> None:
    """A captured empty list becoming non-empty at runtime must diverge."""

    x = torch.randn(3)
    path = _save(EmptyListBranch(), (x, []), tmp_path / "el.tlspec")
    assert tl.load(path).run(inputs=(x.clone(), [])).report.path_faithfulness is (
        PathFaithfulness.VERIFIED
    )

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=(x.clone(), [5.0]))


@pytest.mark.smoke
def test_r29_empty_container_kind_change_diverges(tmp_path: Path) -> None:
    """An empty dict replaced by an empty list at the same path must diverge (kind change)."""

    x = torch.randn(3)
    path = _save(FlagPresenceBranch(), (x, {"flag": {}}), tmp_path / "kind.tlspec")

    # 'flag' still present but now an empty LIST -- structurally different container.
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=(x.clone(), {"flag": []}))


@pytest.mark.smoke
def test_r29_bool_int_key_twin_diverges(tmp_path: Path) -> None:
    """A bool dict key twinned to the equal-valued int key must diverge (F6)."""

    x = torch.randn(3)
    path = _save(BoolKeyBranch(), (x, {True: 2.0}), tmp_path / "kt.tlspec")
    assert tl.load(path).run(inputs=(x.clone(), {True: 2.0})).report.path_faithfulness is (
        PathFaithfulness.VERIFIED
    )

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=(x.clone(), {1: 2.0}))


@pytest.mark.smoke
def test_r29_same_empty_structure_verifies(tmp_path: Path) -> None:
    """Identical empty-container structure on the same input must stay VERIFIED (no over-trigger)."""

    x = torch.randn(3)
    path = _save(FlagPresenceBranch(), (x, {"flag": {}, "extra": []}), tmp_path / "ok.tlspec")

    result = tl.load(path).run(inputs=(x.clone(), {"flag": {}, "extra": []}))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    torch.testing.assert_close(result.output, x + 100)
