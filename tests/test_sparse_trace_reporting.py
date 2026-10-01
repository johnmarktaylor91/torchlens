"""Sparse captures must stay printable, and their reports must stay honest.

``save=<predicate>`` is the mode the performance guide recommends for large models,
and it was the one mode in which ``print(trace)`` raised: the non-finite scan behind
the summary read ``.out`` on ops that retained no payload, and ``Op.__getattribute__``
raises ``ValueError`` for those (a plain ``getattr(..., None)`` cannot swallow it).

The fix skips unreadable payloads, which is only honest because the clean answer now
says how many ops it could not examine -- a scoped "no NaNs found" must never read as
a whole-capture guarantee.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, NoReturn

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.lazy import LazyActivationRef
from torchlens.data_classes._nonfinite import (
    first_nonfinite_layer,
    unexamined_payload_count,
)
from torchlens.options import CaptureOptions


class _NonFiniteModel(nn.Module):
    """Model whose middle activation is non-finite."""

    def __init__(self) -> None:
        """Build a linear layer feeding a divide-by-zero."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce a non-finite intermediate activation.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Output derived from a non-finite intermediate.
        """

        hidden = self.fc(x)
        return (hidden / torch.zeros_like(hidden)) * 2.0


def _plain_model() -> nn.Module:
    """Return a small finite two-layer model.

    Returns
    -------
    nn.Module
        Linear + ReLU + Linear stack.
    """

    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


# ---------------------------------------------------------------------------
# Printing a sparse capture
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_str_works_on_a_predicate_sparse_capture() -> None:
    """``print(trace)`` used to raise ValueError on any selective-save capture."""

    trace = tl.trace(_plain_model(), torch.randn(2, 4), save=tl.func("relu"))
    text = str(trace)
    assert "Log of Sequential forward pass" in text
    assert "with saved outs" in text


def test_repr_html_works_on_a_predicate_sparse_capture() -> None:
    """The notebook representation goes through the same scan."""

    trace = tl.trace(_plain_model(), torch.randn(2, 4), save=tl.func("relu"))
    assert isinstance(trace._repr_html_(), str)


def test_explain_works_on_a_predicate_sparse_capture() -> None:
    """``report.explain`` asks the trace the same question."""

    trace = tl.trace(_plain_model(), torch.randn(2, 4), save=tl.func("relu"))
    assert isinstance(tl.report.explain(trace), str)


def test_disk_backed_reports_do_not_materialize_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text, HTML, and JSON reporting must remain metadata-only for disk payloads."""

    trace = tl.trace(
        _plain_model(),
        torch.randn(2, 4),
        save=tl.func("relu"),
        storage=tl.to_disk(str(tmp_path / "run.tlspec")),
    )
    assert any(getattr(op, "out_ref", None) is not None for op in trace.layer_list)

    def fail_materialization(
        self: LazyActivationRef,
        *,
        map_location: Any = "cpu",
        payload_hints: Any | None = None,
        resolved_blobs_dir: Path | None = None,
    ) -> NoReturn:
        """Fail if a reporting surface attempts a lazy value read.

        Parameters
        ----------
        self:
            Lazy payload reference.
        map_location:
            Requested materialization device.
        payload_hints:
            Optional codec hints.
        resolved_blobs_dir:
            Optional resolved blob directory.

        Raises
        ------
        AssertionError
            Always; reporting must never reach this method.
        """

        del self, map_location, payload_hints, resolved_blobs_dir
        raise AssertionError("reporting materialized a disk-backed activation")

    monkeypatch.setattr(LazyActivationRef, "materialize", fail_materialization)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))

    text = str(trace)
    html = trace._repr_html_()
    report = tl.report.explain(trace, format="json")
    assert isinstance(text, str)
    assert "disk-backed" in html
    assert isinstance(report, dict)
    assert "disk-backed" in report["first_nonfinite"]


def test_json_and_html_clean_answers_disclose_sparse_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Machine and notebook reports preserve the same coverage hedge as text."""

    trace = tl.trace(_plain_model(), torch.randn(2, 4), save=tl.func("relu"))
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))

    report = tl.report.explain(trace, format="json")
    assert isinstance(report, dict)
    assert "could not be examined" in report["first_nonfinite"]
    assert "could not be examined" in trace._repr_html_()


# ---------------------------------------------------------------------------
# The scoped answer discloses its own scope
# ---------------------------------------------------------------------------


def test_sparse_clean_answer_names_the_ops_it_could_not_examine() -> None:
    """A scoped clean verdict must not read as a whole-capture verdict."""

    trace = tl.trace(_plain_model(), torch.randn(2, 4), save=tl.func("relu"))
    answer = trace.first_nonfinite(link_format="text")
    assert answer.startswith("No non-finite")
    assert "could not be examined" in answer
    assert str(unexamined_payload_count(trace, kind="saved")) in answer


def test_full_capture_answer_claims_no_gap() -> None:
    """A full capture examined everything, so it must not hedge."""

    trace = tl.trace(_plain_model(), torch.randn(2, 4))
    assert unexamined_payload_count(trace, kind="saved") == 0
    assert trace.first_nonfinite(link_format="text") == (
        "No non-finite tensor values found in saved outs."
    )


def test_answer_hedges_when_nothing_is_saved() -> None:
    """``layers_to_save="none"`` retains no payloads, and says so."""

    trace = tl.trace(
        _plain_model(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="none"),
    )
    assert unexamined_payload_count(trace, kind="saved") > 0
    assert "could not be examined" in trace.first_nonfinite(link_format="text")


# ---------------------------------------------------------------------------
# Skipping unreadable payloads must not blind the scan on a full capture
# ---------------------------------------------------------------------------


def test_full_capture_still_finds_a_non_finite_activation() -> None:
    """The tripwire is unchanged where payloads exist -- the load-bearing check.

    The scan gate changed from "raise on unsaved" to "skip unsaved". On a full
    capture every op is readable, so the examined set and the verdict must be
    exactly what the raising gate produced.
    """

    trace = tl.trace(_NonFiniteModel(), torch.randn(2, 4))
    raising_gate = first_nonfinite_layer(trace, kind="trace")
    skipping_gate = first_nonfinite_layer(trace, kind="saved")
    assert raising_gate is not None
    assert skipping_gate is raising_gate
    assert "non-finite saved out" in trace.first_nonfinite(link_format="text")


def test_sparse_capture_reports_a_non_finite_saved_activation() -> None:
    """A retained non-finite payload is still found and named."""

    trace = tl.trace(_NonFiniteModel(), torch.randn(2, 4), save=tl.func("__truediv__"))
    answer = trace.first_nonfinite(link_format="text")
    assert "First non-finite saved out" in answer
    assert "truediv" in answer


# ---------------------------------------------------------------------------
# The trace can self-report its footprint
# ---------------------------------------------------------------------------


def test_activation_footprint_fields_are_populated() -> None:
    """Both footprint fields carry real byte counts, sparse or not."""

    full = tl.trace(_plain_model(), torch.randn(2, 4))
    assert int(full.total_activation_memory) > 0
    assert int(full.saved_activation_memory) == int(full.total_activation_memory)

    sparse = tl.trace(_plain_model(), torch.randn(2, 4), save=tl.func("relu"))
    assert int(sparse.total_activation_memory) > 0
    assert 0 < int(sparse.saved_activation_memory) < int(sparse.total_activation_memory)


@pytest.mark.parametrize("guessed", ["activation_memory", "memory", "total_memory", "footprint"])
def test_footprint_guesses_route_to_the_real_fields(guessed: str) -> None:
    """A bare AttributeError read as "TorchLens cannot tell you".

    Parameters
    ----------
    guessed:
        Attribute name a user at scale plausibly reaches for.
    """

    trace = tl.trace(_plain_model(), torch.randn(2, 4))
    with pytest.raises(AttributeError) as excinfo:
        getattr(trace, guessed)
    message = str(excinfo.value)
    assert "total_activation_memory" in message
    assert "forward_peak_memory" in message


def test_unknown_attributes_still_raise_a_plain_error() -> None:
    """Only the curated guesses get guidance; everything else is unchanged."""

    trace = tl.trace(_plain_model(), torch.randn(2, 4))
    with pytest.raises(AttributeError) as excinfo:
        trace.definitely_not_a_field  # noqa: B018
    assert "total_activation_memory" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# recurrence_detection=False: the documented speed-knob tradeoff
# ---------------------------------------------------------------------------


class _LoopModel(nn.Module):
    """Model with a hand-rolled top-level loop over one layer."""

    def __init__(self) -> None:
        """Build the single reused layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer six times.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Output after six iterations.
        """

        for _ in range(6):
            x = torch.relu(self.fc(x))
        return x


def test_recurrence_detection_off_matches_its_documented_tradeoff() -> None:
    """The speed-knobs row must describe what the knob actually does.

    Documented: repeated ops stay separate layers instead of rolling into one
    multi-pass layer, ``is_recurrent`` and ``max_layer_op_count`` are still
    reported, and retained bytes are unchanged.
    """

    x = torch.randn(2, 4)
    rolled = tl.trace(_LoopModel(), x, capture=CaptureOptions(recurrence_detection=True))
    unrolled = tl.trace(_LoopModel(), x, capture=CaptureOptions(recurrence_detection=False))

    assert rolled["relu_1_2"].num_passes == 6
    assert len(unrolled.layer_labels) > len(rolled.layer_labels)
    assert unrolled["relu_1_2"].num_passes == 1
    assert "relu_6_7" in unrolled.layer_labels

    assert unrolled.is_recurrent is True
    assert unrolled.max_layer_op_count == rolled.max_layer_op_count
    assert int(unrolled.total_activation_memory) == int(rolled.total_activation_memory)
