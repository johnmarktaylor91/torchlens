"""Intervention honesty: a failed ``do()`` detaches the hooks it attached
(WT A-II item 11 -- the contamination defect).

A hook that raises at fire time used to stay in the sticky spec after the
failed ``do()``, so every LATER push (including unrelated edits) re-fired it.
``do()`` is transactional about its own attachments: engine failure detaches
exactly the hooks that call attached; a refused site mid-selection-plan rolls
back the earlier sites' attachments. Hooks attached through the explicit
``attach_hooks`` door stay sticky by contract (the caller holds the handle).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class _ConvRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c1(x))


def _log() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        _ConvRelu().eval(),
        torch.randn(2, 1, 4, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )


def _bad_hook(out: torch.Tensor, *, hook) -> torch.Tensor:
    raise RuntimeError("boom")


def _hook_count(trace: tl.Trace) -> int:
    spec = getattr(trace, "_intervention_spec", None)
    return len(spec.hook_specs) if spec is not None else 0


def test_failed_do_detaches_its_hook_and_later_edits_run_clean() -> None:
    log = _log()
    fork = log.fork()
    with pytest.raises(RuntimeError, match="boom"):
        fork.do("relu_1_2", _bad_hook)
    assert _hook_count(fork) == 0

    # The measured contamination: this used to re-fire the dead hook.
    fork.do("conv2d_1_1", tl.scale(1.0))
    assert torch.allclose(fork["relu_1_2"].out, log["relu_1_2"].out)


def test_failed_selection_do_detaches_its_hooks() -> None:
    log = _log()
    fork = log.fork()
    selection = fork["relu_1_2"].__selection__()
    with pytest.raises(RuntimeError, match="boom"):
        fork.do(selection, _bad_hook)
    assert _hook_count(fork) == 0
    fork.do("relu_1_2", tl.scale(1.0))
    assert torch.allclose(fork["relu_1_2"].out, log["relu_1_2"].out)


@pytest.mark.smoke
def test_failed_do_keeps_prior_healthy_hooks() -> None:
    """Cleanup removes exactly the failing call's hooks, nothing else."""

    log = _log()
    fork = log.fork()
    fork.do("relu_1_2", tl.scale(2.0))
    assert _hook_count(fork) == 1
    with pytest.raises(RuntimeError, match="boom"):
        fork.do("conv2d_1_1", _bad_hook)
    assert _hook_count(fork) == 1
    assert torch.allclose(fork["relu_1_2"].out, 2.0 * log["relu_1_2"].out)


@pytest.mark.smoke
def test_explicit_attach_hooks_stays_sticky_by_contract() -> None:
    """The manual door keeps its semantics: the caller owns the handle."""

    log = _log()
    fork = log.fork()
    handle = fork.attach_hooks(tl.func("relu"), _bad_hook, confirm_mutation=True)
    with pytest.raises(RuntimeError, match="boom"):
        fork.push()
    assert _hook_count(fork) == 1
    handle.remove()
    assert _hook_count(fork) == 0
