"""Blind acceptance tests for multi-pass (recurrent) replay fidelity of ``do()``.

Written as an independent oracle from the specification alone, on top of main,
without reading the fix branch. The specification under test:

- Internals operate on PASS-QUALIFIED op labels (``layer_label:pass``); an edit
  addressed to one pass affects exactly that pass, every pass's stored
  activation is committed correctly, and the cone of effect includes all
  genuinely downstream layers (including later passes of the same layers).
- A BARE layer label naming a MULTI-PASS layer is ambiguous and must refuse
  typed, naming the layer, its pass count, and the available pass-qualified
  spellings.
- A bare label naming a SINGLE-PASS layer keeps working exactly as before.
- No spurious ``ControlFlowDivergenceWarning`` on ordinary multi-pass replay.
- ``strict=True`` multi-pass replay works instead of raising
  ``ControlFlowDivergenceError``.

Test-design rules baked in (do not weaken):

- Every fidelity assertion is PER PASS, never aggregate: an aggregate check
  passes even when only the last pass was touched, which is exactly how the
  bug shipped.
- All looped models use ``nn.Linear(..., bias=True)`` (or ``nn.RNNCell`` with
  biases) so that zero is NOT a fixed point of the cell: with ``bias=False``,
  ``relu(cell(0)) == 0`` and a wrong replay coincidentally equals ground
  truth. Each test asserts this confound guard explicitly.
- Ground truth is computed BY HAND from the model's own weights (a manual
  forward pass with the intervention applied), never from the engine itself.
"""

from __future__ import annotations

import re
import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import TorchLensError
from torchlens.intervention.errors import ControlFlowDivergenceWarning

ATOL = 1e-6
RTOL = 1e-5


class LoopedCell(nn.Module):
    """Hand-rolled recurrence: one shared Linear+ReLU cell applied N times."""

    def __init__(self, n_passes: int) -> None:
        """Build the shared cell.

        Parameters
        ----------
        n_passes:
            Number of times the cell is applied in ``forward``.
        """
        super().__init__()
        # bias=True is LOAD-BEARING: it makes relu(cell(0)) != 0, so a replay
        # that fails to propagate a zeroed pass cannot coincidentally match
        # the hand-computed ground truth.
        self.cell = nn.Linear(4, 4, bias=True)
        self.n_passes = n_passes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared cell ``n_passes`` times."""
        for _ in range(self.n_passes):
            x = torch.relu(self.cell(x))
        return x


class SinglePassMLP(nn.Module):
    """Plain two-layer MLP with no module reuse (single-pass regression)."""

    def __init__(self) -> None:
        """Build two distinct Linear layers around one ReLU."""
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=True)
        self.fc2 = nn.Linear(4, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fc1 -> relu -> fc2."""
        return self.fc2(torch.relu(self.fc1(x)))


class LoopedRNNCell(nn.Module):
    """Real torch recurrent module (``nn.RNNCell``) driven by a python loop.

    ``nn.RNN`` itself is not usable here: it executes as one fused kernel and
    therefore never produces a multi-pass layer, so the cell-in-a-loop form is
    the faithful multi-pass spelling of torch recurrence.
    """

    def __init__(self, n_steps: int) -> None:
        """Build the shared RNN cell.

        Parameters
        ----------
        n_steps:
            Number of timesteps consumed from the input sequence.
        """
        super().__init__()
        self.cell = nn.RNNCell(3, 5)
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume ``x[:, t]`` for each timestep, threading the hidden state."""
        h = torch.zeros(x.shape[0], self.cell.hidden_size)
        for t in range(self.n_steps):
            h = self.cell(x[:, t], h)
        return h


def _capture_looped(n_passes: int) -> tuple[LoopedCell, torch.Tensor, tl.Trace]:
    """Build, seed, and capture an intervention-ready looped-cell trace."""
    torch.manual_seed(0)
    model = LoopedCell(n_passes).eval()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    assert "linear_1_1" in log.layer_labels, (
        f"precondition: expected shared-cell layer label 'linear_1_1'; got {log.layer_labels}"
    )
    assert "relu_1_2" in log.layer_labels, (
        f"precondition: expected shared-cell layer label 'relu_1_2'; got {log.layer_labels}"
    )
    assert log["relu_1_2"].num_passes == n_passes
    return model, x, log


def _hand_truth(
    model: LoopedCell, x: torch.Tensor, zero_pass: int
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Manual forward pass with the relu output of ``zero_pass`` zeroed.

    Uses only the model's own weights; never consults the capture engine.

    Returns
    -------
    dict
        ``pass_number -> (linear_out, relu_out_after_edit)`` where
        ``relu_out_after_edit`` is the value downstream passes consume (zeros
        on the edited pass, the genuine relu output elsewhere).
    """
    acts: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        h = x
        for p in range(1, model.n_passes + 1):
            lin = model.cell(h)
            h = torch.relu(lin)
            if p == zero_pass:
                h = torch.zeros_like(h)
            acts[p] = (lin, h)
    return acts


def _clone_originals(log: tl.Trace, labels: list[str]) -> dict[str, torch.Tensor]:
    """Snapshot stored activations before any intervention, defensively cloned."""
    return {lbl: log[lbl].out.detach().clone() for lbl in labels}


def _pass_labels(n_passes: int) -> list[str]:
    """All pass-qualified labels of the looped-cell model, plus the output."""
    labels = []
    for p in range(1, n_passes + 1):
        labels.append(f"linear_1_1:{p}")
        labels.append(f"relu_1_2:{p}")
    labels.append("output_1")
    return labels


def _assert_confound_guards(
    orig: dict[str, torch.Tensor],
    hand: dict[int, tuple[torch.Tensor, torch.Tensor]],
    n_passes: int,
    zero_pass: int,
) -> None:
    """Prove the test can actually distinguish a wrong replay from a right one.

    Guards (all must hold by construction, none is the behavior under test):

    - Zero is not a fixed point: the first post-edit relu output is nonzero.
    - Every genuinely downstream stored activation differs from its
      unintervened value, so an uncommitted (stale) pass cannot pass.
    - The edited pass's original value was nonzero, so the edit is observable.
    """
    if zero_pass < n_passes:
        first_downstream = hand[zero_pass + 1][1]
        assert first_downstream.abs().max() > 1e-3, (
            "confound guard: relu(cell(0)) must be nonzero (bias=True); "
            "otherwise a wrong replay coincidentally equals ground truth"
        )
    assert orig[f"relu_1_2:{zero_pass}"].abs().max() > 1e-3, (
        "confound guard: the edited pass's original activation must be "
        "nonzero for the edit to be observable"
    )
    for p in range(zero_pass + 1, n_passes + 1):
        for fam, idx in (("linear_1_1", 0), ("relu_1_2", 1)):
            assert not torch.allclose(orig[f"{fam}:{p}"], hand[p][idx], rtol=RTOL, atol=1e-3), (
                f"confound guard: hand truth for {fam}:{p} must differ from "
                "the unintervened activation, else a stale stored value "
                "would pass"
            )


def _assert_per_pass_fidelity(
    fork: tl.Trace,
    orig: dict[str, torch.Tensor],
    hand: dict[int, tuple[torch.Tensor, torch.Tensor]],
    n_passes: int,
    zero_pass: int,
) -> None:
    """Assert every pass's stored activation individually, never in aggregate."""
    # Upstream of the edit (strictly earlier passes, plus the edited pass's
    # own linear input): must be byte-stable at the original values.
    for p in range(1, zero_pass + 1):
        torch.testing.assert_close(
            fork[f"linear_1_1:{p}"].out,
            orig[f"linear_1_1:{p}"],
            rtol=RTOL,
            atol=ATOL,
            msg=f"linear_1_1:{p} is upstream of the pass-{zero_pass} edit and "
            "must keep its original stored activation",
        )
    for p in range(1, zero_pass):
        torch.testing.assert_close(
            fork[f"relu_1_2:{p}"].out,
            orig[f"relu_1_2:{p}"],
            rtol=RTOL,
            atol=ATOL,
            msg=f"relu_1_2:{p} is upstream of the pass-{zero_pass} edit and "
            "must keep its original stored activation",
        )

    # The edited pass itself: stored activation must be the edited value.
    torch.testing.assert_close(
        fork[f"relu_1_2:{zero_pass}"].out,
        torch.zeros_like(orig[f"relu_1_2:{zero_pass}"]),
        rtol=0.0,
        atol=0.0,
        msg=f"relu_1_2:{zero_pass} was zero-ablated; its stored activation must be exactly zero",
    )

    # Downstream cone: every later pass recomputed from the edited value.
    for p in range(zero_pass + 1, n_passes + 1):
        torch.testing.assert_close(
            fork[f"linear_1_1:{p}"].out,
            hand[p][0],
            rtol=RTOL,
            atol=ATOL,
            msg=f"linear_1_1:{p} must equal the hand-computed forward from "
            f"the zeroed pass-{zero_pass} activation",
        )
        torch.testing.assert_close(
            fork[f"relu_1_2:{p}"].out,
            hand[p][1],
            rtol=RTOL,
            atol=ATOL,
            msg=f"relu_1_2:{p} must equal the hand-computed forward from "
            f"the zeroed pass-{zero_pass} activation",
        )
    torch.testing.assert_close(
        fork["output_1"].out,
        hand[n_passes][1],
        rtol=RTOL,
        atol=ATOL,
        msg="output_1 must equal the hand-computed final activation",
    )


def _do_quietly(fork: tl.Trace, site: object, hook: object, **kwargs: object) -> None:
    """Run ``fork.do`` with warning promotion disabled.

    The repo-level filterwarnings config promotes TorchLens warnings to
    errors; fidelity tests assert on VALUES, and the dedicated warning test
    asserts on warnings, so value tests must not couple to warning policy.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        fork.do(site, hook, **kwargs)


# ---------------------------------------------------------------------------
# Single-pass regression: bare labels on single-pass layers keep working.
# ---------------------------------------------------------------------------


def test_single_pass_bare_label_do_keeps_working() -> None:
    """Bare-label ``do()`` on a single-pass layer stays exactly as before.

    GREEN on main by design (regression pin); must stay green after the fix.
    """
    torch.manual_seed(0)
    model = SinglePassMLP().eval()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    assert log["relu_1_2"].num_passes == 1
    with torch.no_grad():
        hand_out = model.fc2(torch.zeros(2, 4))
    orig_fc1 = log["linear_1_1"].out.detach().clone()

    fork = log.fork()
    _do_quietly(fork, "relu_1_2", tl.zero_ablate())

    torch.testing.assert_close(
        fork["relu_1_2"].out,
        torch.zeros(2, 4),
        rtol=0.0,
        atol=0.0,
        msg="single-pass relu must be exactly zeroed",
    )
    torch.testing.assert_close(
        fork["linear_1_1"].out,
        orig_fc1,
        rtol=RTOL,
        atol=ATOL,
        msg="upstream fc1 must be untouched",
    )
    torch.testing.assert_close(
        fork["linear_2_3"].out,
        hand_out,
        rtol=RTOL,
        atol=ATOL,
        msg="downstream fc2 must equal fc2(0) computed by hand",
    )
    torch.testing.assert_close(fork["output_1"].out, hand_out, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Multi-pass per-pass fidelity: first-pass and middle-pass edits.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_passes", "zero_pass"),
    [
        pytest.param(2, 1, id="2pass-edit-first"),
        pytest.param(3, 1, id="3pass-edit-first"),
        pytest.param(3, 2, id="3pass-edit-middle"),
    ],
)
def test_pass_qualified_edit_per_pass_fidelity(n_passes: int, zero_pass: int) -> None:
    """Zero-ablating one pass touches exactly that pass, per hand ground truth.

    The middle-pass case is the sharpest: a last-pass-wins bug and a
    first-pass-only bug both survive a last-pass-edit test, so the edit is
    never addressed to the final pass here.
    """
    model, x, log = _capture_looped(n_passes)
    labels = _pass_labels(n_passes)
    orig = _clone_originals(log, labels)
    hand = _hand_truth(model, x, zero_pass)
    _assert_confound_guards(orig, hand, n_passes, zero_pass)

    fork = log.fork()
    _do_quietly(fork, f"relu_1_2:{zero_pass}", tl.zero_ablate())

    _assert_per_pass_fidelity(fork, orig, hand, n_passes, zero_pass)

    # The source trace must be untouched by the fork's edit (per pass).
    for lbl in labels:
        torch.testing.assert_close(
            log[lbl].out,
            orig[lbl],
            rtol=0.0,
            atol=0.0,
            msg=f"source trace activation {lbl} mutated by a fork edit",
        )


# ---------------------------------------------------------------------------
# Ambiguity refusal: bare label naming a multi-pass layer refuses typed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_passes", [2, 3], ids=["2pass", "3pass"])
def test_bare_label_on_multipass_layer_refuses_typed(n_passes: int) -> None:
    """A bare label naming a multi-pass layer is ambiguous: typed refusal.

    The message must name the layer, its pass count, and every available
    pass-qualified spelling. The refused call must not half-apply the edit.
    """
    _model, _x, log = _capture_looped(n_passes)
    labels = _pass_labels(n_passes)
    orig = _clone_originals(log, labels)

    fork = log.fork()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.raises(TorchLensError) as excinfo:
            fork.do("relu_1_2", tl.zero_ablate())

    msg = str(excinfo.value)
    assert "relu_1_2" in msg, "refusal must name the ambiguous layer"
    for p in range(1, n_passes + 1):
        assert f"relu_1_2:{p}" in msg, (
            f"refusal must list the available pass-qualified spelling 'relu_1_2:{p}'; got: {msg}"
        )
    # The pass count must be stated independently of the qualified spellings
    # (strip those first so ':3' etc. cannot satisfy the check vacuously).
    stripped = re.sub(r"relu_1_2:\d+", "", msg)
    assert re.search(rf"\b{n_passes}\b", stripped), (
        f"refusal must state the pass count ({n_passes}); got: {msg}"
    )

    # Refusal means REFUSAL: no stored activation on any pass may have moved.
    for lbl in labels:
        torch.testing.assert_close(
            fork[lbl].out,
            orig[lbl],
            rtol=0.0,
            atol=0.0,
            msg=f"refused bare-label do() must not touch {lbl}",
        )


# ---------------------------------------------------------------------------
# No spurious ControlFlowDivergenceWarning on ordinary multi-pass replay.
# ---------------------------------------------------------------------------


def test_no_spurious_control_flow_divergence_warning() -> None:
    """Ordinary multi-pass replay emits no ``ControlFlowDivergenceWarning``."""
    _model, _x, log = _capture_looped(3)
    fork = log.fork()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fork.do("relu_1_2:2", tl.zero_ablate())
    divergence = [w for w in caught if issubclass(w.category, ControlFlowDivergenceWarning)]
    assert not divergence, (
        "ordinary multi-pass replay must not emit ControlFlowDivergenceWarning; "
        f"got: {[str(w.message) for w in divergence]}"
    )


# ---------------------------------------------------------------------------
# Strict mode: multi-pass replay works instead of raising.
# ---------------------------------------------------------------------------


def test_strict_multipass_replay_works() -> None:
    """``strict=True`` multi-pass replay succeeds with full per-pass fidelity.

    On broken main this raised ``ControlFlowDivergenceError`` ("... not in the
    saved parent edge set"); any raise here fails the test. Uses the typed
    ``tl.label`` selector because bare strings are refused in strict mode by
    an independent, pre-existing portability rule that is not under test.
    """
    n_passes, zero_pass = 3, 2
    model, x, log = _capture_looped(n_passes)
    labels = _pass_labels(n_passes)
    orig = _clone_originals(log, labels)
    hand = _hand_truth(model, x, zero_pass)
    _assert_confound_guards(orig, hand, n_passes, zero_pass)

    fork = log.fork()
    _do_quietly(
        fork,
        tl.label(f"relu_1_2:{zero_pass}"),
        tl.zero_ablate(),
        intervention=tl.options.InterventionOptions(strict=True),
    )

    _assert_per_pass_fidelity(fork, orig, hand, n_passes, zero_pass)


# ---------------------------------------------------------------------------
# Real torch recurrent module: nn.RNNCell reused across timesteps.
# ---------------------------------------------------------------------------


def test_rnncell_loop_middle_pass_edit_per_pass_fidelity() -> None:
    """Middle-pass edit on an ``nn.RNNCell`` loop, hand truth per timestep.

    RNNCell biases default on, so zero is not a fixed point of the cell
    (tanh(W x + b_ih + b_hh) != tanh(W x + U h + ...) in general); guarded
    below.
    """
    torch.manual_seed(1)
    model = LoopedRNNCell(3).eval()
    x = torch.randn(2, 3, 3)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    cell_label = "rnntanhcell_1_3"
    assert cell_label in log.layer_labels, (
        f"precondition: expected RNN cell layer '{cell_label}'; got {log.layer_labels}"
    )
    assert log[cell_label].num_passes == 3

    orig = {f"{cell_label}:{p}": log[f"{cell_label}:{p}"].out.detach().clone() for p in (1, 2, 3)}
    with torch.no_grad():
        h1 = model.cell(x[:, 0], torch.zeros(2, 5))
        h3_from_zero = model.cell(x[:, 2], torch.zeros(2, 5))

    # Confound guards: the edit must be observable and non-coincidental.
    assert orig[f"{cell_label}:2"].abs().max() > 1e-3
    assert not torch.allclose(h3_from_zero, orig[f"{cell_label}:3"], rtol=RTOL, atol=1e-3), (
        "confound guard: hand truth for timestep 3 must differ from the unintervened activation"
    )

    fork = log.fork()
    _do_quietly(fork, f"{cell_label}:2", tl.zero_ablate())

    torch.testing.assert_close(
        fork[f"{cell_label}:1"].out,
        h1,
        rtol=RTOL,
        atol=ATOL,
        msg="timestep 1 is upstream of the edit and must keep the genuine hand-computed h1",
    )
    torch.testing.assert_close(
        fork[f"{cell_label}:2"].out,
        torch.zeros_like(orig[f"{cell_label}:2"]),
        rtol=0.0,
        atol=0.0,
        msg="timestep 2 was zero-ablated; its stored activation must be zero",
    )
    torch.testing.assert_close(
        fork[f"{cell_label}:3"].out,
        h3_from_zero,
        rtol=RTOL,
        atol=ATOL,
        msg="timestep 3 must equal cell(x3, 0) computed by hand",
    )
    torch.testing.assert_close(
        fork["output_1"].out,
        h3_from_zero,
        rtol=RTOL,
        atol=ATOL,
        msg="final output must equal the hand-computed timestep-3 activation",
    )
