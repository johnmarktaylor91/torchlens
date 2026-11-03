"""Fixwave-2 FW2-POLISH pins for R19 visualization-honesty fixes.

R19-1 (b6, HIGH): the run-fold "+N more <Class>" ellipsis claims the hidden
members are interchangeable, but the uniformity fingerprint was 4 integers
(layer/param counts) — a kwargs-different conv (dilation 2) and a
tanh-for-relu block both folded under the homogeneity claim, and two
DIFFERENT models rendered byte-identical DOT. The fingerprint now folds in
the per-member op-type sequence and func_config digest.

T9 (grind-p3, HIGH): the fingerprint was compared only across the HIDDEN
members (``addresses[1:]``), so a plateau uniformly different from its own
visible representative still folded — an all-Tanh plateau behind a ReLU
representative rendered byte-identical DOT to the all-ReLU model. The
uniformity check now spans EVERY member including the representative; a
run whose representative differs splits, so the plateau re-folds from its
own structurally-matching representative instead of hiding the change.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization.auto_collapse import _run_fold_members_uniform


def _trace(model: nn.Module) -> tl.Trace:
    return tl.trace(model, torch.randn(1, 2, 12, 12))


class _ResidualBlock(nn.Module):
    """Linear+activation residual block for run-fold ellipsis renders."""

    def __init__(self, width: int = 8, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.lin(x))


class _ResidualStack(nn.Module):
    """Repeated residual blocks that auto-collapse folds into one ellipsis."""

    def __init__(self, activations: list[type[nn.Module]]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ResidualBlock(activation=act) for act in activations])
        self.out = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.out(x)


@pytest.mark.smoke
def test_fold_honesty_uniform_plateau_dot_differs(tmp_path) -> None:
    """A uniformly-changed hidden plateau never renders byte-identical DOT.

    T9 (grind-p3, HIGH) red pin: with the hidden-only comparison, a model
    whose blocks 1..7 all swap ReLU for Tanh folded to "+7 more" behind the
    unchanged ReLU representative and rendered the SAME bytes as the
    all-ReLU model. Post-fix the changed model splits (representative box
    plus a plateau fold of its own), so the sources differ.
    """

    torch.manual_seed(0)
    sources: list[str] = []
    for variant in ("plain", "tanh_plateau"):
        activations: list[type[nn.Module]] = [nn.ReLU] * 8
        if variant == "tanh_plateau":
            activations = [nn.ReLU] + [nn.Tanh] * 7
        trace = tl.trace(_ResidualStack(activations), torch.randn(2, 8))
        outpath = tmp_path / f"plateau_{variant}"
        trace.draw(
            collapse="auto",
            fold_repeats=True,
            file_format="dot",
            path=str(outpath),
        )
        sources.append((tmp_path / f"plateau_{variant}.dot").read_text())
    assert sources[0] != sources[1], (
        "a uniformly-changed hidden plateau rendered byte-identical DOT; the "
        "fold is claiming sameness with a representative it never checked"
    )


class _WiringBlock(nn.Module):
    """Same class / params / ordered op types; wiring differs by flag.

    ``x + y`` (residual skip) and ``y + y`` (self-add) share the op-type
    sequence ``linear -> relu -> add`` and every parameter count — only the
    DAG edges differ (r3 b6-opus R19-1).
    """

    def __init__(self, self_add: bool) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.self_add = self_add

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.lin(x))
        return (y + y) if self.self_add else (x + y)


class _WiringStack(nn.Module):
    """Repeated wiring blocks that auto-collapse folds into one ellipsis."""

    def __init__(self, self_add_flags: list[bool]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_WiringBlock(flag) for flag in self_add_flags])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _BindBlock(nn.Module):
    """Same class / op / wiring shape; exterior-operand BINDING differs.

    ``a - b`` and ``b - a`` both canonicalize their two exterior parents to
    per-member first-seen numbers, so the historical wiring digest read both
    as ``(("x", 0), ("x", 1))`` — the cross-member source correspondence was
    lost (r4 b6-sol R19-1).
    """

    def __init__(self, swap: bool) -> None:
        super().__init__()
        self.swap = swap

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (b - a) if self.swap else (a - b)


class _BindStack(nn.Module):
    """Sibling bind blocks all fed the same two exterior tensors."""

    def __init__(self, swap_flags: list[bool]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_BindBlock(flag) for flag in swap_flags])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(x)
        b = torch.tanh(x)
        outs = [block(a, b) for block in self.blocks]
        return torch.stack(outs).sum(dim=0)


def test_fold_honesty_exterior_binding_dot_differs(tmp_path) -> None:
    """Two models differing only in hidden exterior-operand binding never
    render byte-identical DOT.

    The r4 b6-sol probe's shape: 24 sibling blocks all receive the same two
    exterior tensors; model A subtracts ``a - b`` everywhere, model B swaps
    to ``b - a`` in blocks 1..23. Materially different outputs; before the
    binding-consistency check both folded into one multiplicity-24 fold.
    """

    for mode in ("auto", "max"):
        sources: list[str] = []
        for variant in ("plain", "swapped"):
            flags = [False] * 24
            if variant == "swapped":
                flags = [False] + [True] * 23
            torch.manual_seed(0)
            trace = tl.trace(_BindStack(flags), torch.randn(2, 8))
            outpath = tmp_path / f"bind_{mode}_{variant}"
            trace.draw(
                collapse=mode,
                fold_repeats=True,
                file_format="dot",
                path=str(outpath),
            )
            sources.append((tmp_path / f"bind_{mode}_{variant}.dot").read_text())
        assert sources[0] != sources[1], (
            f"collapse={mode!r}: two models with different hidden exterior "
            "bindings rendered byte-identical DOT; the wiring digest erases "
            "cross-member source correspondence"
        )


def test_run_fold_never_folds_unresolvable_wiring(monkeypatch) -> None:
    """A member whose wiring cannot be resolved must NEVER fold.

    r4 b6-fable R19 (fresh LOW): the degrade arm used to collapse every
    unresolvable member to the same exception TYPE NAME, so two members with
    genuinely different (but both unresolvable) wiring compared EQUAL and the
    fold fell back to the op-signature-only comparison the r3 HIGH proved
    insufficient. Degradation must be a unique per-member sentinel.
    """

    # F11 (collapse memo item 6): the wiring walk moved to
    # _collapse_signatures with the B1 fingerprint memo; the patch target
    # follows the callee's defining module so the degrade arm still fires.
    from torchlens._visualization import _collapse_signatures, auto_collapse

    trace = tl.trace(_WiringStack([False, False, False]), torch.randn(2, 8))
    assert _run_fold_members_uniform(trace, ("blocks.0", "blocks.1", "blocks.2"))

    def _boom(module):  # noqa: ANN001, ANN202
        raise KeyError("orphan relation label")

    monkeypatch.setattr(_collapse_signatures, "_module_wiring_walk", _boom)
    assert not _run_fold_members_uniform(trace, ("blocks.0", "blocks.1", "blocks.2"))
    sig_a = auto_collapse._module_structural_signature(cast_module(trace, "blocks.0"))
    sig_b = auto_collapse._module_structural_signature(cast_module(trace, "blocks.1"))
    assert sig_a != sig_b, "degraded members must never compare equal"


def cast_module(trace: tl.Trace, address: str):  # noqa: ANN201
    """Resolve one Module record by address."""

    return trace.modules[address]
