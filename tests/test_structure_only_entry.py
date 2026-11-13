"""Entry contract for structure-only capture (L7a wave 0, D8-default branch).

Covers the memo's sec 1.2 options threading, sec 1.3/2.2 Layer-0 entry
conflicts, the sec 1.5 entry-matrix cells reachable without D8 (E-1
degenerate form (b), E-6 flag-off zero-diff), and the S3-registrar mode-marker
prep. Every structure-only spelling here is DOCUMENTED-UNSTABLE pending
naming-session/S2 ratification.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import (
    ArgumentTypeError,
    InvalidArgumentError,
    StructureOnlyOptionConflictError,
)
from torchlens.backends.registry import get_backend_spec
from torchlens.options import CaptureOptions


class TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


def _structure_capture(**trace_kwargs):
    return tl.trace(
        TwoLayer(),
        torch.randn(2, 4),
        capture=CaptureOptions(structure_only=True),
        **trace_kwargs,
    )


# ---------------------------------------------------------------------------
# Options threading (memo sec 1.2)
# ---------------------------------------------------------------------------


def test_structure_only_defaults_false_and_validates_bool() -> None:
    assert CaptureOptions().structure_only is False
    assert CaptureOptions(structure_only=True).structure_only is True
    with pytest.raises(ArgumentTypeError) as excinfo:
        CaptureOptions(structure_only="yes")  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "structure_only_type_invalid"


def test_non_torch_backend_refuses_explicit_structure_only_typed() -> None:
    """The capability-gate row: a backend without structure_only_capture
    refuses the explicit option instead of silently ignoring it."""

    from torchlens._capture.prepare.dispatch import _enforce_capability_option_gates
    from torchlens.backends.registry import BackendUnsupportedError

    tinygrad_spec = get_backend_spec("tinygrad")
    assert tinygrad_spec.capabilities.structure_only_capture is False
    with pytest.raises(BackendUnsupportedError, match="structure_only"):
        _enforce_capability_option_gates({"structure_only": True}, tinygrad_spec)


def test_torch_declares_and_binds_the_structure_only_capability() -> None:
    spec = get_backend_spec("torch")
    assert spec.capabilities.structure_only_capture is True
    implementations = spec.capability_implementations or {}
    surface = implementations["structure_only_capture"]()
    assert surface is not None


# ---------------------------------------------------------------------------
# Entry matrix: E-1 (all-real degenerate form (b)) and E-6 (flag off)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Layer-0 entry conflicts (one code, structure_only_option_conflict)
# ---------------------------------------------------------------------------


def test_raise_on_nan_conflicts_at_entry() -> None:
    with pytest.raises(StructureOnlyOptionConflictError) as excinfo:
        tl.trace(
            TwoLayer(),
            torch.randn(2, 4),
            capture=CaptureOptions(structure_only=True, on_nonfinite="raise"),
        )
    assert excinfo.value.fields["code"] == "structure_only_option_conflict"
    assert "on_nonfinite" in excinfo.value.fields["arguments"]


def test_intervention_ready_conflicts_at_entry() -> None:
    """The opus B3a row: runnable eligibility would disable the plain escape
    belt, so the combination must be unreachable, not quietly belt-less."""

    with pytest.raises(StructureOnlyOptionConflictError) as excinfo:
        tl.trace(
            TwoLayer(),
            torch.randn(2, 4),
            capture=CaptureOptions(structure_only=True, replayable=True),
        )
    assert excinfo.value.fields["code"] == "structure_only_option_conflict"
    assert "replayable" in excinfo.value.fields["arguments"]


def test_value_touching_halt_predicates_conflict_at_entry() -> None:
    # A bare callable's value use is unprovable: fail closed.
    with pytest.raises(StructureOnlyOptionConflictError) as excinfo:
        _structure_capture(stop_after=lambda ctx: False)
    assert excinfo.value.fields["code"] == "structure_only_option_conflict"
    # tl.where escapes the static-kind allowlist: fail closed too.
    with pytest.raises(StructureOnlyOptionConflictError):
        _structure_capture(stop_after=tl.where(lambda ctx: False))


def test_value_free_structured_halt_stays_legal() -> None:
    """Memo 4.2: truncation composes when the predicate is provably
    value-free (structured selectors)."""

    log = _structure_capture(stop_after=tl.func("relu"))
    assert log.options.structure_only is True


# ---------------------------------------------------------------------------
# Explicit value-payload requests refuse (structure_only_values_unsupported)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "trace_kwargs", "capture_kwargs"),
    [
        ("save predicate", {"save": tl.func("relu")}, {}),
        ("explicit save all", {"save": "all"}, {}),
        ("save_grads", {}, {"save_grads": True}),
        ("save_args", {}, {"save_args": True}),
        ("save_raw_input", {}, {"save_raw_input": True}),
        (
            "save_transformed_output",
            {},
            {"save_transformed_output": True, "output_transform": lambda out: out},
        ),
        ("lookback payloads", {}, {"lookback": 4, "lookback_payload_policy": "detached_raw"}),
    ],
)
def test_explicit_value_payload_requests_refuse_typed(
    description: str, trace_kwargs: dict, capture_kwargs: dict
) -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(
            TwoLayer(),
            torch.randn(2, 4),
            capture=CaptureOptions(structure_only=True, **capture_kwargs),
            **trace_kwargs,
        )
    assert excinfo.value.fields["code"] == "structure_only_values_unsupported", description


def test_streaming_sink_refuses_typed(tmp_path) -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        _structure_capture(storage=tl.to_disk(str(tmp_path / "run.tlspec")))
    assert excinfo.value.fields["code"] == "structure_only_values_unsupported"


# ---------------------------------------------------------------------------
# Mode marker: ACTIVE since the coordinated tlspec v8 bump
# ---------------------------------------------------------------------------
