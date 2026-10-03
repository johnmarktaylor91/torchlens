"""Observe-kit item 1: NaN label unification identity fixtures + the raw-label lint.

One op, ONE public name: every NaN surface (find_nan, bisect_nan, the
CaptureError message and structured fields, CaptureOutcome.boundary_label,
PartialTrace summaries) reads its label out of the step-8 raw-to-final
identity map produced by the failure-safe prefix finalization -- never by
stripping the raw suffix or by ordinal arithmetic, both of which fabricate
names (the raw-to-final offset is not a constant, and finalization really
deletes committed dead-chain records).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture._nonfinite_prefix import resolve_raw_label
from torchlens.errors import CaptureError

TORCHLENS_DIR = Path(tl.__file__).resolve().parent


class _FaultModel(nn.Module):
    """Linear -> tanh -> 0/0 fault; one committed non-finite site."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce a NaN at the truediv op."""

        y = torch.tanh(self.linear(x))
        return y / torch.zeros_like(y)


class _DeadChainFaultModel(nn.Module):
    """Input-disconnected dead chain PLUS a live fault site."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Build a dead internally-initialized chain, then fault on the live path."""

        dead = torch.ones(3) * 2.0
        dead = dead + 1.0  # dead-chain bystander, never used again
        y = torch.tanh(x)
        return y / torch.zeros_like(y)


class _DeadBranchOffenderModel(nn.Module):
    """The FAULT itself sits on an input-disconnected chain."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fault on an internally-initialized chain; the live path is clean."""

        dead = torch.ones(3) * 2.0
        _bad = dead / torch.zeros(3)  # inf on the dead chain
        return torch.tanh(x)


class _ReusedBlockModel(nn.Module):
    """One Linear called twice; pass 2 overflows to inf INSIDE the reused layer."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.inner.weight.copy_(torch.ones(3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass 1 is finite; pass 2 receives 1e38-scale inputs and overflows."""

        h = self.inner(x)  # pass 1: finite
        return self.inner(h * 1e38)  # pass 2: 3e38 > fp32 max -> inf


def _raise_on_nan_error(model: nn.Module, x: torch.Tensor) -> CaptureError:
    """Run one raise_on_nan capture and return the CaptureError it raises."""

    with pytest.raises(CaptureError) as exc_info:
        tl.trace(model, x, capture=tl.options.CaptureOptions(raise_on_nan=True))
    return exc_info.value


def _assert_no_raw_spelling(text: str) -> None:
    """Assert no user-visible label in ``text`` carries the raw suffix."""

    assert not re.search(r"\w+_\d+_raw\b", text), text


def test_one_label_across_every_nan_surface() -> None:
    """find_nan, bisect_nan, error fields, message, and outcome agree on ONE name."""

    torch.manual_seed(0)
    x = torch.ones(1, 3)

    exc = _raise_on_nan_error(_FaultModel(), x)
    error_label = exc.fields["layer"]
    assert isinstance(error_label, str) and error_label
    assert exc.fields["layer_status"] == "final"
    assert repr(error_label) in str(exc)
    assert exc.affected_sites == [error_label]

    outcome = exc.partial_log.trace.outcome
    assert outcome.boundary_label == error_label

    live = tl.debug.find_nan(_FaultModel(), x)
    assert live.found and live.label == error_label
    assert live.label_status == "final"

    completed = tl.trace(_FaultModel(), x)
    try:
        bisected = tl.debug.bisect_nan(completed)
        assert bisected.found and bisected.label == error_label
        in_trace = completed.find_nan()
        assert in_trace.found and in_trace.label == error_label
    finally:
        completed.cleanup()

    for text in (str(exc), str(live.message), exc.partial_log.first_nonfinite()):
        _assert_no_raw_spelling(text)
    for parent in exc.fields["parents"]:
        assert not parent.endswith("_raw")


def test_dead_chain_bystanders_prune_to_none_and_offender_survives() -> None:
    """Pruned bystanders resolve (None, pruned); the offender is always labeled."""

    exc = _raise_on_nan_error(_DeadChainFaultModel(), torch.ones(3))
    trace = exc.partial_log.trace
    assert exc.fields["layer_status"] == "final"
    assert isinstance(exc.fields["layer"], str)

    orphan_labels = trace.__dict__.get("_orphan_labels") or []
    assert orphan_labels, "the dead chain should have been pruned"
    for raw_label in orphan_labels:
        label, status = resolve_raw_label(trace, raw_label)
        assert label is None
        assert status == "pruned"


def test_offender_on_dead_branch_is_never_elided() -> None:
    """Frontier seeding guarantees the OFFENDER survives the orphan flood."""

    exc = _raise_on_nan_error(_DeadBranchOffenderModel(), torch.ones(3))
    assert exc.fields["layer_status"] == "final"
    label = exc.fields["layer"]
    assert isinstance(label, str) and label.startswith("truediv")
    # The finalized prefix serves the offender through ordinary lookup.
    assert exc.partial_log.trace[label] is not None


def test_multipass_offender_gets_pass_qualified_canonical_label() -> None:
    """A pass-2 fault in a reused layer names PASS 2, never both passes."""

    exc = _raise_on_nan_error(_ReusedBlockModel(), torch.ones(1, 3))
    label = exc.fields["layer"]
    assert exc.fields["layer_status"] == "final"
    assert isinstance(label, str)
    assert label.endswith(":2"), (
        f"expected a pass-qualified canonical label naming pass 2, got {label!r}: "
        "an unqualified spelling would equally name pass 1"
    )


@pytest.mark.smoke
def test_failed_finalization_yields_none_plus_status_never_raw(monkeypatch) -> None:
    """When the prefix postprocess cannot run, labels are None + status."""

    from torchlens.data_classes.trace import Trace

    def _boom(self: object, *args: object, **kwargs: object) -> None:
        raise RuntimeError("forced finalization failure (fixture)")

    monkeypatch.setattr(Trace, "_postprocess", _boom)
    exc = _raise_on_nan_error(_FaultModel(), torch.ones(1, 3))
    assert exc.fields["layer"] is None
    assert exc.fields["layer_status"] == "unavailable"
    assert isinstance(exc.fields["layer_raw"], str)
    _assert_no_raw_spelling(str(exc))
    outcome = tl.partial.from_failed_capture(exc).trace.outcome
    assert str(outcome.status) == "CaptureStatus.ABORTED_NONFINITE"
    assert outcome.boundary_label is None


def test_resolver_statuses_are_closed_and_total() -> None:
    """The resolver answers every input with a label or an explicit status."""

    exc = _raise_on_nan_error(_FaultModel(), torch.ones(1, 3))
    trace = exc.partial_log.trace
    label, status = resolve_raw_label(trace, exc.fields["layer_raw"])
    assert (label, status) == (exc.fields["layer"], "final")
    assert resolve_raw_label(trace, "never_recorded_9_raw") == (None, "unavailable")
    assert resolve_raw_label(None, "anything_raw") == (None, "unavailable")
    assert resolve_raw_label(trace, None) == (None, "unavailable")


def test_no_raw_label_strip_sites_outside_the_resolver() -> None:
    """LINT: stripping the raw suffix fabricates public names; new sites refuse.

    The raw-to-final ordinal offset is not a constant, so ``.removesuffix``/
    slice-stripping ``RAW_LABEL_SUFFIX`` into a user-facing spelling is always
    wrong. The ONLY sanctioned sites are the constant's definition, the
    explicitly raw-named ``Op.raw_label`` identity property, and the resolver
    door module itself (``capture/_nonfinite_prefix.py``, home of both
    ``resolve_raw_label`` and the constant-suffix ``strip_raw_label_suffix``
    the preview/neutral finishers use before any ``Trace`` identity map
    exists); every other consumer routes through one of those two functions.
    """

    allowlist = {
        TORCHLENS_DIR / "constants.py",
        TORCHLENS_DIR / "data_classes" / "op.py",
        TORCHLENS_DIR / "capture" / "_nonfinite_prefix.py",
    }
    offenders: list[str] = []
    for path in sorted(TORCHLENS_DIR.rglob("*.py")):
        if path in allowlist:
            continue
        text = path.read_text(encoding="utf-8")
        if "RAW_LABEL_SUFFIX" in text:
            offenders.append(str(path.relative_to(TORCHLENS_DIR)))
    assert not offenders, (
        "raw-label strip/consume sites outside the resolver door "
        f"(route through capture._nonfinite_prefix.resolve_raw_label): {offenders}"
    )
