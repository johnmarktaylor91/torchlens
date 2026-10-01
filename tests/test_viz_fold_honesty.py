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
from torchlens.visualization.auto_collapse import _run_fold_members_uniform

pytestmark = pytest.mark.smoke


class _ConvBlock(nn.Module):
    """Conv+activation block whose output shape is dilation-invariant."""

    def __init__(self, dilation: int = 1, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=dilation, dilation=dilation)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


def _trace(model: nn.Module) -> tl.Trace:
    return tl.trace(model, torch.randn(1, 2, 12, 12))


def test_run_fold_rejects_kwargs_different_hidden_members() -> None:
    """A dilation-2 conv cannot hide inside a "+N more" of dilation-1 blocks.

    Parameter and layer COUNTS are identical across the run, so the old
    4-int fingerprint read the members as uniform; only the func_config
    digest tells them apart.
    """

    trace = _trace(nn.Sequential(_ConvBlock(1), _ConvBlock(1), _ConvBlock(2)))
    assert not _run_fold_members_uniform(trace, ("0", "1", "2"))


def test_run_fold_rejects_optype_different_hidden_members() -> None:
    """A Tanh block cannot hide inside a "+N more" of ReLU blocks.

    Activations carry zero parameters, so every count in the old
    fingerprint matched; only the op-type sequence tells them apart.
    """

    trace = _trace(
        nn.Sequential(
            _ConvBlock(1),
            _ConvBlock(1),
            _ConvBlock(1, activation=nn.Tanh),
        )
    )
    assert not _run_fold_members_uniform(trace, ("0", "1", "2"))


def test_run_fold_still_accepts_genuinely_uniform_members() -> None:
    """Identical members keep folding (no over-rejection)."""

    trace = _trace(nn.Sequential(_ConvBlock(1), _ConvBlock(1), _ConvBlock(1)))
    assert _run_fold_members_uniform(trace, ("0", "1", "2"))


def test_run_fold_rejects_representative_unlike_hidden_members() -> None:
    """A run whose representative differs from its plateau must not fold whole.

    REVIEWED rebaseline (T9, grind-p3): this exact shape was previously
    pinned as foldable on the theory that the representative's own stats
    stay visible. The theory was wrong in the other direction — the HIDDEN
    plateau's structure appears nowhere, so a plateau uniformly different
    from the representative folded invisibly (see
    ``test_fold_honesty_uniform_plateau_dot_differs``). The run must split
    so the plateau folds from its own structurally-matching representative.
    """

    representative_differs = _trace(nn.Sequential(_ConvBlock(2), _ConvBlock(1), _ConvBlock(1)))
    assert not _run_fold_members_uniform(representative_differs, ("0", "1", "2"))


def test_run_fold_rejects_plateau_unlike_representative() -> None:
    """A uniform Tanh plateau cannot fold behind a ReLU representative.

    T9 (grind-p3, HIGH) red pin: the hidden members agree with each other,
    so the old hidden-only comparison accepted the fold; only comparing the
    representative too tells the run apart.
    """

    trace = _trace(
        nn.Sequential(
            _ConvBlock(1),
            _ConvBlock(1, activation=nn.Tanh),
            _ConvBlock(1, activation=nn.Tanh),
        )
    )
    assert not _run_fold_members_uniform(trace, ("0", "1", "2"))


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
            vis_save_only=True,
            vis_fileformat="dot",
            vis_outpath=str(outpath),
        )
        sources.append((tmp_path / f"plateau_{variant}.dot").read_text())
    assert sources[0] != sources[1], (
        "a uniformly-changed hidden plateau rendered byte-identical DOT; the "
        "fold is claiming sameness with a representative it never checked"
    )


def test_fold_honesty_dot_sources_differ(tmp_path) -> None:
    """Two structurally different models never render byte-identical DOT.

    The b6 probe's failure shape: with the count-only fingerprint, the
    dilated-conv model and the plain model folded to the SAME rendered
    graph. Renders are save-only DOT so the pin is a pure byte comparison.
    """

    sources: list[str] = []
    for variant in ("plain", "dilated"):
        blocks = [_ConvBlock(1), _ConvBlock(1), _ConvBlock(2 if variant == "dilated" else 1)]
        trace = _trace(nn.Sequential(*blocks))
        outpath = tmp_path / f"fold_{variant}"
        trace.draw(
            collapse="max",
            fold_repeats=True,
            vis_save_only=True,
            vis_fileformat="dot",
            vis_outpath=str(outpath),
        )
        sources.append((tmp_path / f"fold_{variant}.dot").read_text())
    assert sources[0] != sources[1], (
        "structurally different models rendered byte-identical DOT; the fold "
        "is hiding a non-uniform member"
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


def test_run_fold_rejects_wiring_different_members() -> None:
    """A self-add block cannot hide inside a "+N more" of residual blocks.

    r3 b6-opus R19-1 red pin: op types, kwargs, and every parameter count
    match across the run — only the intra-module dataflow (skip edge vs
    self-edge) tells the members apart. RED before the wiring component.
    """

    trace = tl.trace(_WiringStack([False, False, True]), torch.randn(2, 8))
    assert not _run_fold_members_uniform(trace, ("blocks.0", "blocks.1", "blocks.2"))


def test_run_fold_accepts_wiring_uniform_members() -> None:
    """Identically-wired members keep folding (no over-rejection).

    Exterior sources are numbered per member, so consecutive residual
    blocks fed by DIFFERENT upstream blocks still compare equal.
    """

    trace = tl.trace(_WiringStack([False, False, False]), torch.randn(2, 8))
    assert _run_fold_members_uniform(trace, ("blocks.0", "blocks.1", "blocks.2"))


def test_fold_honesty_topology_dot_differs(tmp_path) -> None:
    """Two models differing only in hidden-member WIRING never render
    byte-identical DOT.

    The r3 b6-opus probe verbatim: 24 blocks, model A all residual
    ``x + y``, model B blocks 1..23 self-add ``y + y``. Same class, same
    params, same ordered op types; the underlying edge sets differ. Before
    the wiring component both folded behind ``+23 more`` and the DOT was
    byte-identical at auto and max.
    """

    for mode in ("auto", "max"):
        sources: list[str] = []
        for variant in ("residual", "self_add"):
            flags = [False] * 24
            if variant == "self_add":
                flags = [False] + [True] * 23
            torch.manual_seed(0)
            trace = tl.trace(_WiringStack(flags), torch.randn(2, 8))
            outpath = tmp_path / f"topo_{mode}_{variant}"
            trace.draw(
                collapse=mode,
                fold_repeats=True,
                vis_save_only=True,
                vis_fileformat="dot",
                vis_outpath=str(outpath),
            )
            sources.append((tmp_path / f"topo_{mode}_{variant}.dot").read_text())
        assert sources[0] != sources[1], (
            f"collapse={mode!r}: two models with different hidden wiring "
            "rendered byte-identical DOT; the fold fingerprint is "
            "topology-blind"
        )


class _ScalarBlock(nn.Module):
    """Same class / params / ordered op types; only a scalar operand differs.

    ``linear -> relu -> * k`` blocks share every count, op type, kwargs dict
    (``func_config`` is empty for the dunder mul), and interior wiring — only
    the non-tensor operand VALUE ``k`` tells the members' computations apart
    (r4 b6-opus R19-1).
    """

    def __init__(self, k: float) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x)) * self.k


class _ScalarStack(nn.Module):
    """Repeated scalar blocks that auto-collapse folds into one ellipsis."""

    def __init__(self, scales: list[float]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ScalarBlock(k) for k in scales])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def test_run_fold_rejects_scalar_operand_different_members() -> None:
    """A ``* 3.0`` block cannot hide inside a "+N more" of ``* 1.0`` blocks.

    r4 b6-opus R19-1 red pin: op types, kwargs, params, and wiring all match
    across the run — only the scalar operand value tells the members apart.
    RED before the non-tensor operand component.
    """

    trace = tl.trace(_ScalarStack([1.0, 1.0, 3.0]), torch.randn(2, 8))
    assert not _run_fold_members_uniform(trace, ("blocks.0", "blocks.1", "blocks.2"))


def test_run_fold_accepts_scalar_operand_uniform_members() -> None:
    """Identically-scaled members keep folding (no over-rejection)."""

    trace = tl.trace(_ScalarStack([2.0, 2.0, 2.0]), torch.randn(2, 8))
    assert _run_fold_members_uniform(trace, ("blocks.0", "blocks.1", "blocks.2"))


def test_fold_honesty_scalar_operand_dot_differs(tmp_path) -> None:
    """Two models differing only in a hidden scalar operand never render
    byte-identical DOT.

    The r4 b6-opus probe verbatim: 24 blocks, model A all ``* 1.0``, model B
    blocks 1..23 ``* 3.0``. The models compute numerically different
    functions; before the operand-value component both folded behind
    ``+23 more`` and the DOT was byte-identical at auto and max.
    """

    for mode in ("auto", "max"):
        sources: list[str] = []
        for variant in ("ones", "threes"):
            scales = [1.0] * 24
            if variant == "threes":
                scales = [1.0] + [3.0] * 23
            torch.manual_seed(0)
            trace = tl.trace(_ScalarStack(scales), torch.randn(2, 8))
            outpath = tmp_path / f"scalar_{mode}_{variant}"
            trace.draw(
                collapse=mode,
                fold_repeats=True,
                vis_save_only=True,
                vis_fileformat="dot",
                vis_outpath=str(outpath),
            )
            sources.append((tmp_path / f"scalar_{mode}_{variant}.dot").read_text())
        assert sources[0] != sources[1], (
            f"collapse={mode!r}: two models with different hidden scalar "
            "operands rendered byte-identical DOT; the fold fingerprint is "
            "operand-value-blind"
        )


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


def test_run_fold_rejects_swapped_exterior_binding_members() -> None:
    """A ``b - a`` member cannot hide inside a "+N more" of ``a - b`` members.

    r4 b6-sol R19-1 red pin: every member's per-member wiring digest is
    identical — only the correspondence of the exterior sources ACROSS
    members (both fed the same two tensors, bound to swapped operand slots)
    tells them apart. RED before the cross-member binding-consistency check.
    """

    trace = tl.trace(_BindStack([False, False, True]), torch.randn(2, 8))
    assert not _run_fold_members_uniform(trace, ("blocks.0", "blocks.1", "blocks.2"))


def test_run_fold_accepts_shared_exterior_binding_uniform_members() -> None:
    """Members binding shared exteriors to the SAME slots keep folding."""

    trace = tl.trace(_BindStack([False, False, False]), torch.randn(2, 8))
    assert _run_fold_members_uniform(trace, ("blocks.0", "blocks.1", "blocks.2"))


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
                vis_save_only=True,
                vis_fileformat="dot",
                vis_outpath=str(outpath),
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
    from torchlens.visualization import _collapse_signatures, auto_collapse

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
