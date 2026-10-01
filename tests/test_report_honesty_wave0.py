"""A09 wave-0 report honesty: escape-free text, one-pass scan, log_value,
agent-guide triage, presenter refusals, poisoned/episode banners.

Covers listA rows 23 and 25a plus sumfam wave-0 items 1, 3, 4, and 6:
- ``first_nonfinite`` defaults to the plain-text register; report text never
  carries ESC bytes or resolved-absolute-path URIs (LAUNCH BLOCKER).
- The health-scan predicate is the one-pass ``isfinite().all()`` form,
  equivalence-pinned against the two-pass reference.
- ``log_value``'s canonical home is ``torchlens.observers`` with the
  ``tl.report`` compat alias, and its values read back and render.
- ``to_agent_json``'s guide menu is executable-by-the-reader and conditional.
- explain/to_agent_json refuse known presenters typed; poisoned traces stop
  rendering clean on explain/agent_json/draw; TraceSlice.summary carries the
  parent's honesty banner.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_honesty import capture_honesty_facts, honesty_banner_lines
from torchlens.data_classes._nonfinite import _has_nonfinite
from torchlens.runnable import PathFaithfulness

ESC = "\x1b"


class _NaNNet(nn.Module):
    """Two-op model whose first op emits NaN/Inf."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x / 0.0
        return y * 2


def _nan_trace() -> tl.Trace:
    return tl.trace(_NaNNet(), torch.ones(2, 3))


def _clean_trace() -> tl.Trace:
    return tl.trace(nn.Sequential(nn.Linear(3, 4), nn.ReLU()), torch.randn(2, 3))


# --- SF1: escape-free publishable text (LAUNCH BLOCKER) ---------------------


@pytest.mark.smoke
def test_first_nonfinite_default_is_plain_text() -> None:
    """The default register carries no ESC byte and no file:// / vscode:// URI."""

    trace = _nan_trace()
    answer = trace.first_nonfinite()
    assert ESC not in answer
    assert "file://" not in answer and "vscode://" not in answer
    assert answer == trace.first_nonfinite(link_format="text")


@pytest.mark.smoke
def test_styled_link_registers_stay_opt_in() -> None:
    """terminal/html remain available but only by explicit request."""

    trace = _nan_trace()
    assert ESC in trace.first_nonfinite(link_format="terminal")
    assert "vscode://file/" in trace.first_nonfinite(link_format="html")


@pytest.mark.smoke
def test_explain_text_and_json_are_escape_free() -> None:
    """explain() output is publishable data: no ESC bytes on any fixture."""

    trace = _nan_trace()
    # D14 (F09): explain serves the basis in hand and never scans; arm the
    # evidence through the explicit spelling first.
    assert trace.nonfinite_ops
    report = tl.report.explain(trace)
    assert ESC not in report
    payload = tl.report.explain(trace, format="json")
    assert ESC not in str(payload)
    # The non-finite evidence still names the site in plain text.
    assert "truediv" in payload["first_nonfinite"]


@pytest.mark.smoke
def test_str_trace_is_escape_free() -> None:
    """print(trace) inherits the plain-text default (OSC 8 leak regression)."""

    assert ESC not in str(_nan_trace())


# --- SF6: one-pass health-scan predicate ------------------------------------


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([1.0, float("nan")]),
        torch.tensor([float("inf"), 1.0]),
        torch.tensor([-float("inf")]),
        torch.zeros(3, 4),
        torch.tensor([1, 2, 3]),
        torch.tensor([True, False]),
        torch.tensor([1.0 + 2.0j, complex(float("nan"), 0.0)]),
        torch.tensor([], dtype=torch.float32),
    ],
    ids=["clean", "nan", "inf", "neginf", "zeros", "int", "bool", "complex", "empty"],
)
@pytest.mark.smoke
def test_one_pass_predicate_matches_two_pass_reference(tensor: torch.Tensor) -> None:
    """not all(isfinite) is pinned equivalent to the historical any(~isfinite)."""

    reference = bool((~torch.isfinite(tensor.detach())).any().item())
    assert _has_nonfinite(tensor) is reference


@pytest.mark.smoke
def test_predicate_unrunnable_dtype_returns_none() -> None:
    """Dtypes with no isfinite kernel stay None (no evidence), never False."""

    quantized = torch.quantize_per_tensor(torch.ones(3), scale=1.0, zero_point=0, dtype=torch.qint8)
    assert _has_nonfinite(quantized) is None


# --- SF4: log_value canonical home + read-back ------------------------------


@pytest.mark.smoke
def test_log_value_canonical_home_and_alias() -> None:
    """observers owns log_value; tl.report keeps the identical compat alias."""

    from torchlens.observers import log_value as canonical
    from torchlens.report import log_value as alias

    assert canonical is alias


@pytest.mark.smoke
def test_log_value_read_back_and_renders() -> None:
    """Logged values read back via Trace.logged_values and render in reports."""

    class _Logging(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tl.observers.log_value("gate_mean", 0.25)
            return torch.relu(x)

    trace = tl.trace(_Logging(), torch.randn(2, 3))
    assert trace.logged_values == {"gate_mean": 0.25}
    # Read-back is a copy, not a mutation door.
    view = trace.logged_values
    view["gate_mean"] = -1.0
    assert trace.logged_values == {"gate_mean": 0.25}
    report = tl.report.explain(trace)
    assert "Logged values" in report and "gate_mean = 0.25" in report
    dump = trace.to_agent_json(max_ops=1)
    assert dump["logged_values"] == {"gate_mean": 0.25}
    assert "logged_values" in dump["guide"]["next_steps"]


@pytest.mark.smoke
def test_logged_values_render_is_bounded() -> None:
    """Arbitrary logged objects render bounded: caps disclosed, never a spew."""

    class _Logging(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for index in range(15):
                tl.observers.log_value(f"value_{index}", "x" * 500)
            return torch.relu(x)

    trace = tl.trace(_Logging(), torch.randn(2, 3))
    report = tl.report.explain(trace)
    section = report[report.index("Logged values") :]
    for line in section.splitlines():
        assert len(line) < 120
    assert "and 5 more" in section


# --- SF3: agent-guide triage -------------------------------------------------


@pytest.mark.smoke
def test_guide_next_steps_are_reader_executable() -> None:
    """No guide entry requires live objects the dump's reader does not hold."""

    steps = _clean_trace().to_agent_json(max_ops=1)["guide"]["next_steps"]
    assert "tl.compat.report(model, x).to_markdown()" not in steps.values()
    for spelling in steps.values():
        assert "model" not in spelling.replace("trace", "") or spelling.startswith("trace")
    # The omitted members are now taught.
    assert steps["health_audit"] == "trace.audit()"
    assert steps["inventory"] == "trace.bill_of_materials()"
    assert steps["resource_profile"] == "trace.profile(level='module')"
    assert steps["nonfinite_evidence"] == "trace.first_nonfinite()"
    assert steps["environment_diagnosis"] == "tl.utils.doctor()"


@pytest.mark.smoke
def test_guide_next_steps_are_conditional_on_state() -> None:
    """State-dependent entries appear only when the trace can honor them."""

    saved = _clean_trace().to_agent_json(max_ops=1)["guide"]["next_steps"]
    assert saved["one_activation"] == "trace[<layer_label>].out"
    with pytest.warns(UserWarning, match="matched zero sites"):
        unsaved_trace = tl.trace(
            nn.Sequential(nn.Linear(3, 4), nn.ReLU()),
            torch.randn(2, 3),
            save=tl.func("no_such_op"),
        )
    unsaved = unsaved_trace.to_agent_json(max_ops=1)["guide"]["next_steps"]
    assert "one_activation" not in unsaved
    assert "output_table" not in saved  # no decoded output on this capture


# --- WT23: presenter refusals + poisoned/episode banners ---------------------


@pytest.mark.smoke
def test_explain_and_agent_json_refuse_trace_slice_typed() -> None:
    """A TraceSlice subject refuses typed instead of a hollow wrong report."""

    trace = _clean_trace()
    trace_slice = trace.between(trace.layer_labels[0], trace.layer_labels[-1])
    with pytest.raises(tl.errors.InvalidArgumentError) as exc_info:
        tl.report.explain(trace_slice)
    assert exc_info.value.fields["code"] == "report_subject_unsupported"
    with pytest.raises(tl.errors.InvalidArgumentError):
        tl.Trace.to_agent_json(trace_slice)


@pytest.mark.smoke
def test_explain_refuses_recording_typed() -> None:
    """A sparse Recording subject is refused with the to_trace() remedy."""

    recording = tl.record(
        nn.Sequential(nn.Linear(3, 4), nn.ReLU()), torch.randn(2, 3), save=tl.func("relu")
    )
    with pytest.raises(tl.errors.InvalidArgumentError) as exc_info:
        tl.report.explain(recording)
    assert "to_trace" in str(exc_info.value)


@pytest.mark.smoke
def test_poisoned_trace_stops_rendering_clean() -> None:
    """Poison facts surface on explain, agent_json, and the honesty banner."""

    trace = _clean_trace()
    trace._runnable.poisoned = True
    trace._runnable.path_faithfulness = PathFaithfulness.DIVERGED

    report = tl.report.explain(trace)
    assert "POISONED" in report and "NOT" in report
    dump = trace.to_agent_json(max_ops=1)
    assert dump["capture"]["poisoned"] is True
    assert dump["capture"]["path_faithfulness"] == "diverged"
    banner = honesty_banner_lines(trace)
    assert any("POISONED" in line for line in banner)
    facts = capture_honesty_facts(trace)
    assert facts["poisoned"] is True


@pytest.mark.smoke
def test_poisoned_trace_draw_caption_carries_banner() -> None:
    """draw() DOT source of a poisoned trace names the poison."""

    trace = _clean_trace()
    trace._runnable.poisoned = True
    trace._runnable.path_faithfulness = PathFaithfulness.DIVERGED
    dot_source = trace.draw(vis_save_only=True)
    assert "POISONED" in dot_source


@pytest.mark.smoke
def test_trace_slice_summary_carries_parent_honesty_banner() -> None:
    """A slice of a poisoned capture summarizes with the banner, not clean."""

    trace = _clean_trace()
    trace_slice = trace.between(trace.layer_labels[0], trace.layer_labels[-1])
    assert "POISONED" not in trace_slice.summary()
    trace._runnable.poisoned = True
    trace._runnable.path_faithfulness = PathFaithfulness.DIVERGED
    assert "POISONED" in trace_slice.summary()


# --- SF2: bounded reprs ------------------------------------------------------


@pytest.mark.smoke
def test_compat_report_repr_is_designed_and_bounded() -> None:
    """repr/str are the findings-first summary, never a dataclass wall."""

    report = tl.compat.report(nn.Linear(3, 4), torch.randn(2, 3))
    rendered = repr(report)
    assert rendered == str(report)
    assert rendered.startswith("TorchLens compatibility report for")
    assert "finding(s)" in rendered
    assert ESC not in rendered
    for line in rendered.splitlines():
        assert len(line) <= 160
    # The fixed-width table wraps long cells inside capped columns.
    for line in report.show().splitlines():
        assert len(line) <= 400
    assert max(len(line) for line in report.show().splitlines()) < 500


@pytest.mark.smoke
def test_capability_dump_moved_to_detail_accessor() -> None:
    """The row cell carries the grouped summary; the accessor the full dump."""

    report = tl.compat.report(nn.Linear(3, 4), torch.randn(2, 3))
    row = report.row("torch_capabilities")
    assert "capabilities present" in row.details
    assert len(row.details) < 600
    snapshot = report.capability_snapshot()
    assert isinstance(snapshot, dict) and len(snapshot) > 10
    assert all(isinstance(value, bool) for value in snapshot.values())
    # Absent flags stay NAMED in the cell (absences-first doctrine).
    for name, available in snapshot.items():
        if not available:
            assert name in row.details


@pytest.mark.heavy  # doctor probe walk crossed the 7s smoke budget at T55 (7.2s cpu)
def test_doctor_report_repr_is_bounded() -> None:
    """DoctorReport repr is the designed multi-line render with capped lines."""

    doctor_report = tl.utils.doctor()
    rendered = repr(doctor_report)
    assert rendered.startswith("TorchLens doctor report:")
    for line in rendered.splitlines():
        assert len(line) <= 200
    assert isinstance(doctor_report.capability_snapshot(), dict)


@pytest.mark.smoke
def test_episode_facts_surface_on_report_surfaces() -> None:
    """An episode ledger (incl. forced-tokens basis) is visible in reports."""

    trace = _clean_trace()
    trace.annotations["episode"] = {
        "header": {
            "episode_id": "ep-1",
            "capture_kind": "episode",
            "stepped_module": "self",
            "n_steps_declared": 3,
            "entry_seed": 0,
            "token_feed": "forced",
            "provenance_tier": "exact",
            "structure_only": False,
            "escalated_from": None,
            "reason": None,
            "fidelity_basis": "forced",
        },
        "rows": [],
    }
    report = tl.report.explain(trace)
    assert "Episode capture" in report
    assert "NON-VERIFYING" in report
    dump = trace.to_agent_json(max_ops=1)
    assert dump["capture"]["episode"]["fidelity_basis"] == "forced"
    banner = honesty_banner_lines(trace)
    assert any("episode capture" in line for line in banner)
