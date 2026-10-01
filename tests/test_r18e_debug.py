"""Regression tests for the r18e debug-module hardening fixes.

Covers:
- H5: ``tl.debug.lineage`` no longer crashes on recurrent models (getattr-multipass class).
- H2/M8: ``audit_trace`` honesty -- surfaces real bisect_nan/dtype_range findings and
  counts empty/insufficient checks as SKIPPED, not RUN.
- H6/H7: ``graph_breaks`` runs the Dynamo probe on clean torch and does not mutate the
  caller's model.
- M4/M5: ``infer_input_shape`` restores global RNG and labels failed results honestly.
- M6: live ``find_nan`` does not leak the internal ``_raw`` label namespace.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.utils import _torch_compat


class _Recurrent(nn.Module):
    """Weight-shared recurrent block that rolls into aggregate multi-pass Layers."""

    def __init__(self) -> None:
        """Build a single shared linear applied three times."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``relu(lin(x))`` three times.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Final activation.
        """

        for _ in range(3):
            x = torch.relu(self.lin(x))
        return x


# ---------------------------------------------------------------------------
# H5 -- lineage on recurrent models (getattr-multipass class)
# ---------------------------------------------------------------------------


def test_lineage_recurrent_model_never_raises() -> None:
    """lineage honours its non-raising contract on aggregate multi-pass Layers."""

    trace = tl.trace(_Recurrent(), torch.randn(2, 4), save=tl.where(lambda record: True))

    # A single-pass boundary start walks the pass-qualified graph without crashing.
    for label, direction in (
        ("output_1", "ancestors"),
        ("input_1", "descendants"),
    ):
        result = tl.debug.lineage(trace, label, direction=direction)
        assert result.nodes, (label, direction)
        # No internal raw-namespace markers leak into node labels.
        assert all(not node[0].endswith("_raw") for node in result.nodes)

    # A bare recurrent label is per-pass ambiguous -> honest message, no crash.
    ambiguous = tl.debug.lineage(trace, "linear_1_1", direction="descendants")
    assert ambiguous.nodes == []
    assert "recurrent" in ambiguous.message
    assert "select a pass" in ambiguous.message

    # A pass-qualified start resolves cleanly.
    qualified = tl.debug.lineage(trace, "linear_1_1:2", direction="descendants")
    assert qualified.start_label == "linear_1_1:2"
    assert qualified.nodes


# ---------------------------------------------------------------------------
# H2 / M8 -- audit_trace stops manufacturing reassuring coverage
# ---------------------------------------------------------------------------


class _DeadModel(nn.Module):
    """Model whose ReLU output is structurally all-zero (fully dead)."""

    def __init__(self) -> None:
        """Build a linear whose large negative bias zeroes every ReLU output."""

        super().__init__()
        self.lin = nn.Linear(4, 4)
        with torch.no_grad():
            self.lin.bias.fill_(-1e6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``relu(lin(x))`` (all zeros).

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            An all-zero activation.
        """

        return torch.relu(self.lin(x))


def test_audit_dead_model_does_not_manufacture_coverage() -> None:
    """A fully dead model no longer reports a validated 'no issues; N checks run' lie.

    dead_neurons is an insufficient-sample signal on one trace, so it is listed
    as SKIPPED with a reason instead of being counted as a health check that ran.
    """

    audit = tl.trace(_DeadModel(), torch.randn(2, 4), save=tl.where(lambda record: True)).audit()

    skipped = dict(audit.skipped)
    assert "dead_neurons" not in audit.checks_run
    assert "dead_neurons" in skipped
    assert "insufficient-sample" in skipped["dead_neurons"]
    # hot_path / recompute_candidates are perf rankings, not health checks.
    assert "hot_path" not in audit.checks_run
    assert "recompute_candidates" not in audit.checks_run
    assert "hot_path" in skipped
    # bisect_nan is a real health check that runs on a fully saved trace.
    assert "bisect_nan" in audit.checks_run


def test_audit_multi_backward_gradient_is_skipped_not_run() -> None:
    """A multi-backward trace refuses gradient_flow_audit; audit must SKIP, not RUN it.

    MUTATION PROOF (false-VERIFIED tripwire): if the multi-backward guard in
    ``audit_trace`` is reverted so the empty refusal frame is counted as a check
    that ran, this test's ``not in checks_run`` / ``in skipped`` assertions fail.
    """

    trace = tl.trace(
        nn.Linear(3, 3), torch.randn(2, 3), capture=tl.options.CaptureOptions(save_grads=True)
    )
    output = trace[trace.output_layers[0]].out
    trace.log_backward(output.sum(), retain_graph=True)
    trace.log_backward((output**2).sum())

    audit = trace.audit()
    skipped = dict(audit.skipped)

    assert "gradient_flow_audit" not in audit.checks_run
    assert "gradient_flow_audit" in skipped
    assert "backward passes captured" in skipped["gradient_flow_audit"]


def test_audit_wires_dtype_range_and_surfaces_finding() -> None:
    """audit runs dtype_range_audit (M8) and surfaces its findings.

    MUTATION PROOF (docstring-completeness / report-truth): removing the
    dtype_range_audit wiring drops ``dtype_range_audit`` from checks_run and the
    dtype_near_max finding, failing both assertions.
    """

    dtype_max = torch.finfo(torch.float16).max
    audit = tl.trace(nn.ReLU(), torch.full((2, 3), dtype_max * 0.95, dtype=torch.float16)).audit()

    assert "dtype_range_audit" in audit.checks_run
    assert any(finding.check == "dtype_near_max" for finding in audit.findings)


# ---------------------------------------------------------------------------
# H6 / H7 -- graph_breaks probes clean torch and preserves model state
# ---------------------------------------------------------------------------


class _BreakFree(nn.Module):
    """Fully traceable model with no genuine graph break."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply traceable tensor operations only.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Rectified affine tensor.
        """

        return torch.relu(x + 1)


class _StatefulBranch(nn.Module):
    """Stateful branching model whose buffer increments each forward."""

    def __init__(self) -> None:
        """Register the call-count buffer and a linear."""

        super().__init__()
        self.register_buffer("n", torch.zeros(()))
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on call parity and mutate the buffer.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Linear output.
        """

        self.n += 1
        if self.n.item() % 2 == 1:
            x = x + 1.0
            torch._dynamo.graph_break()
            x = x * 2.0
        else:
            x = torch.relu(x)
            x = x - 1.0
        return self.lin(x)


@pytest.mark.skipif(
    not _torch_compat.HAS_DYNAMO_EXPLAIN,
    reason="torch._dynamo.explain unavailable",
)
def test_graph_breaks_after_prior_capture_is_break_free() -> None:
    """A break-free model is empty even when TorchLens wrappers are installed.

    MUTATION PROOF (H6): the prior ``tl.trace`` installs persistent wrappers.
    Without the ``_clean_torch`` probe, Dynamo reports those wrappers as
    'Attempted to inline function marked as skipped' breaks and this assertion
    fails (non-zero breaks).
    """

    tl.trace(_BreakFree(), torch.ones(2, 3))  # install persistent wrappers
    report = tl.debug.graph_breaks(_BreakFree(), torch.ones(2, 3))
    assert report.breaks == ()


@pytest.mark.skipif(
    not _torch_compat.HAS_DYNAMO_EXPLAIN,
    reason="torch._dynamo.explain unavailable",
)
def test_graph_breaks_does_not_mutate_stateful_model() -> None:
    """graph_breaks restores the caller's model state (H7)."""

    model = _StatefulBranch()
    before = model.n.item()
    report = tl.debug.graph_breaks(model, torch.randn(2, 4))
    assert model.n.item() == before  # neither probe nor eager trace leaks mutation
    # The correlated ops must come from the branch actually taken by both runs
    # (the odd/IF branch: add then mul), not the else branch (relu/sub).
    matched = {label for graph_break in report.breaks for label in graph_break.matched_op_labels}
    assert not any(label.startswith(("relu", "sub")) for label in matched)


# ---------------------------------------------------------------------------
# M4 / M5 -- infer_input_shape is RNG-neutral and labels failures honestly
# ---------------------------------------------------------------------------


def test_infer_input_shape_preserves_global_rng() -> None:
    """infer_input_shape does not advance the caller's global RNG (M4).

    The model is constructed BEFORE seeding so only infer_input_shape's own RNG
    use is measured (a synthesized probe tensor + torch.manual_seed(seed)).

    MUTATION PROOF: dropping the wrapper's torch.set_rng_state restore leaves the
    RNG advanced by the probe and this equality fails.
    """

    model = nn.Linear(4, 4)  # constructed first: its weight init is not under test

    torch.manual_seed(0)
    before = torch.randn(3).tolist()

    torch.manual_seed(0)
    result = tl.debug.infer_input_shape(model, seed=1234)
    after = torch.randn(3).tolist()

    assert result.found  # inference still works
    assert before == after  # global RNG stream untouched by the diagnostic


def test_infer_input_shape_failed_result_strategy_is_not_probe_success() -> None:
    """A failed inference must not claim ``strategy='probe_success'`` (M5)."""

    class _DictModel(nn.Module):
        def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
            """Return a value from a dict input the inferrer cannot synthesize.

            Parameters
            ----------
            x:
                Mapping input.

            Returns
            -------
            torch.Tensor
                The ``tokens`` entry.
            """

            return x["tokens"]

    result = tl.debug.infer_input_shape(_DictModel())

    assert result.found is False
    assert result.strategy != "probe_success"
    assert result.strategy == "none"


# ---------------------------------------------------------------------------
# M6 -- live find_nan exposes public labels, not the internal _raw namespace
# ---------------------------------------------------------------------------


def test_find_nan_live_label_has_no_raw_namespace_leak() -> None:
    """Live find_nan does not leak the internal ``_raw`` label suffix (M6).

    MUTATION PROOF: dropping the ``removesuffix(RAW_LABEL_SUFFIX)`` strip makes
    the label end with ``_raw`` and this assertion fails.
    """

    class _NanMid(nn.Module):
        def __init__(self) -> None:
            """Build two linears around a NaN-producing log."""

            super().__init__()
            self.a = nn.Linear(2, 2)
            self.b = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Produce a NaN in a hidden op via ``log(h - h)``.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Output built on a non-finite hidden activation.
            """

            h = self.a(x)
            h = torch.log(h - h)
            return self.b(h)

    result = tl.debug.find_nan(_NanMid(), torch.ones(1, 2))

    assert result.found
    assert result.label is not None
    assert not result.label.endswith("_raw")  # no internal namespace leak
    # The public-namespace label resolves against a completed trace.
    completed = tl.trace(_NanMid(), torch.ones(1, 2), save=tl.where(lambda record: True))
    assert completed[result.label] is not None
