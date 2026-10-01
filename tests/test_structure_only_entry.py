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
from torchlens._io import FieldPolicy
from torchlens._io.prerelease import registered_prerelease_fields
from torchlens.backends.registry import get_backend_spec
from torchlens.data_classes.trace import Trace
from torchlens.options import CaptureOptions

smoke = pytest.mark.smoke


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


@smoke
def test_structure_only_defaults_false_and_validates_bool() -> None:
    assert CaptureOptions().structure_only is False
    assert CaptureOptions(structure_only=True).structure_only is True
    with pytest.raises(ArgumentTypeError) as excinfo:
        CaptureOptions(structure_only="yes")  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "structure_only_type_invalid"


@smoke
def test_grouped_structure_only_reaches_the_field() -> None:
    from torchlens.options import CaptureOptions

    merged = CaptureOptions(structure_only=True)
    assert merged.structure_only is True
    assert merged.is_field_explicit("structure_only")


@smoke
def test_non_torch_backend_refuses_explicit_structure_only_typed() -> None:
    """The capability-gate row: a backend without structure_only_capture
    refuses the explicit option instead of silently ignoring it."""

    from torchlens.backends.registry import BackendUnsupportedError
    from torchlens.user_funcs import _enforce_capability_option_gates

    mlx_spec = get_backend_spec("mlx")
    assert mlx_spec.capabilities.structure_only_capture is False
    with pytest.raises(BackendUnsupportedError, match="structure_only"):
        _enforce_capability_option_gates({"structure_only": True}, mlx_spec)


@smoke
def test_torch_declares_and_binds_the_structure_only_capability() -> None:
    spec = get_backend_spec("torch")
    assert spec.capabilities.structure_only_capture is True
    implementations = spec.capability_implementations or {}
    surface = implementations["structure_only_capture"]()
    assert surface is not None


# ---------------------------------------------------------------------------
# Entry matrix: E-1 (all-real degenerate form (b)) and E-6 (flag off)
# ---------------------------------------------------------------------------


@smoke
def test_e1_all_real_flag_on_capture_is_marked_gated_and_payload_free() -> None:
    log = _structure_capture()
    assert log.structure_only is True
    # No value payloads are retained anywhere: the contract records structure
    # and shape/dtype hypotheses only.
    for layer in log.layer_list:
        assert getattr(layer, "out", None) is None
    # The structural record itself is intact.
    assert len(log.layer_list) >= 3  # input + linear + relu
    assert tuple(log["relu_1_2"].shape) == (2, 4)


@smoke
def test_e6_flag_off_default_capture_is_zero_diff() -> None:
    log = tl.trace(TwoLayer(), torch.randn(2, 4))
    assert log.structure_only is False
    assert log["relu_1_2"].out is not None


# ---------------------------------------------------------------------------
# Layer-0 entry conflicts (one code, structure_only_option_conflict)
# ---------------------------------------------------------------------------


@smoke
def test_raise_on_nan_conflicts_at_entry() -> None:
    with pytest.raises(StructureOnlyOptionConflictError) as excinfo:
        tl.trace(
            TwoLayer(),
            torch.randn(2, 4),
            capture=CaptureOptions(structure_only=True, raise_on_nan=True),
        )
    assert excinfo.value.fields["code"] == "structure_only_option_conflict"
    assert "raise_on_nan" in excinfo.value.fields["arguments"]


@smoke
def test_intervention_ready_conflicts_at_entry() -> None:
    """The opus B3a row: runnable eligibility would disable the plain escape
    belt, so the combination must be unreachable, not quietly belt-less."""

    with pytest.raises(StructureOnlyOptionConflictError) as excinfo:
        tl.trace(
            TwoLayer(),
            torch.randn(2, 4),
            capture=CaptureOptions(structure_only=True, intervention_ready=True),
        )
    assert excinfo.value.fields["code"] == "structure_only_option_conflict"
    assert "intervention_ready" in excinfo.value.fields["arguments"]


@smoke
def test_value_touching_halt_predicates_conflict_at_entry() -> None:
    # A bare callable's value use is unprovable: fail closed.
    with pytest.raises(StructureOnlyOptionConflictError) as excinfo:
        _structure_capture(halt=lambda ctx: False)
    assert excinfo.value.fields["code"] == "structure_only_option_conflict"
    # tl.where escapes the static-kind allowlist: fail closed too.
    with pytest.raises(StructureOnlyOptionConflictError):
        _structure_capture(halt=tl.where(lambda ctx: False))


@smoke
def test_value_free_structured_halt_stays_legal() -> None:
    """Memo 4.2: truncation composes when the predicate is provably
    value-free (structured selectors)."""

    log = _structure_capture(halt=tl.func("relu"))
    assert log.structure_only is True


# ---------------------------------------------------------------------------
# Explicit value-payload requests refuse (structure_only_values_unsupported)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "trace_kwargs", "capture_kwargs"),
    [
        ("save predicate", {"save": tl.func("relu")}, {}),
        ("explicit layers_to_save", {}, {"layers_to_save": "all"}),
        ("save_grads", {}, {"save_grads": True}),
        ("save_arg_values", {}, {"save_arg_values": True}),
        ("save_raw_input", {}, {"save_raw_input": True}),
        ("save_raw_output", {}, {"save_raw_output": True}),
        ("output_style", {}, {"output_style": "classification"}),
        ("lookback payloads", {"lookback": 4, "lookback_payload_policy": "detached_raw"}, {}),
    ],
)
@smoke
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


@smoke
def test_streaming_sink_refuses_typed(tmp_path) -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        _structure_capture(storage=tl.to_disk(str(tmp_path / "run.tlspec")))
    assert excinfo.value.fields["code"] == "structure_only_values_unsupported"


@smoke
def test_explicit_metadata_only_save_stays_legal() -> None:
    log = tl.trace(
        TwoLayer(),
        torch.randn(2, 4),
        capture=CaptureOptions(structure_only=True, layers_to_save="none"),
    )
    assert log.structure_only is True


# ---------------------------------------------------------------------------
# Mode marker: ACTIVE since the coordinated tlspec v8 bump
# ---------------------------------------------------------------------------


@smoke
def test_mode_marker_persists_and_left_the_prerelease_registrar() -> None:
    """The marker is a live persisted field now, not a DROP-gated one.

    Pre-bump this test asserted the inverse (DROP + prerelease-registered), which
    was the correct invariant while the S3 registrar held the field. The
    coordinated v8 bump flipped every gated family to its intended policy and
    retired the registrar to empty, so the honest invariant is the opposite one:
    the marker persists, and nothing still gates it.
    """

    assert Trace.PORTABLE_STATE_SPEC["structure_only"] is FieldPolicy.KEEP
    inventory = registered_prerelease_fields()
    assert "structure_only" not in inventory.get("Trace", ())
