"""C02 safety tranche: bounded reprs, honest units, coherent index bases.

Lane C02 (megasprint 2026-08-27), M(lovely) item 1 -- the measured bug sites:
the Recording auto-repr that OOMed an 8 GiB box at four saved sites (bug 1),
the doubled duration unit (bug 2), auto-reprs on run products (bug 9),
Bundle's quoted-'None' baseline (bug 17), the OpAccessor basis incoherence
(bug 27, BREAKING fix), unexported option classes + non-runnable remedies
(bugs 20/25), and the capture-oracle sentinel blind spot (bug 24).
"""

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


class _TinyMLP(nn.Module):
    """Four-activation toy: the measured Recording-OOM shape (4 saved sites)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.relu(self.fc4(x))


class _LoopModel(nn.Module):
    """Three-pass recurrent layer for the multi-pass basis pins."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.linear(x)
        return x


@pytest.fixture(scope="module")
def recording() -> "tl.Recording":
    """One predicate recording over the OOM-shaped toy (4 saved relu sites)."""

    return tl.record(_TinyMLP().eval(), torch.randn(2, 8), save=tl.func("relu"))


@pytest.fixture(scope="module")
def loop_trace():
    """One finished multi-pass trace, cleaned up at module teardown."""

    trace = tl.trace(_LoopModel().eval(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_recording_repr_is_bounded_identity_card(recording) -> None:
    """repr(Recording) is an O(1) card, never a payload/context dump (bug 1)."""

    text = repr(recording)
    assert text.startswith("Recording(status=")
    assert len(text) < 400
    assert "tensor(" not in text
    assert "RecordContext(" not in text


def test_recording_str_is_bounded_per_pass_table(recording) -> None:
    """str(Recording) stays within the collection-view line budget (bug 1)."""

    text = str(recording)
    assert len(text.splitlines()) <= 24
    assert "tensor(" not in text


def test_activation_record_repr_is_metadata_descriptor(recording) -> None:
    """One record reprs as one line of metadata, never contents (bug 1)."""

    record = recording.records[0]
    text = repr(record)
    assert text.startswith("ActivationRecord(")
    assert len(text) < 300
    assert "\n" not in text
    assert "tensor(" not in text


def test_record_context_repr_counts_lookback(recording) -> None:
    """RecordContext repr renders lookback tuples as counts (bug 1)."""

    ctx = recording.records[-1].ctx
    text = repr(ctx)
    assert text.startswith("RecordContext(")
    assert "\n" not in text
    assert len(text) < 400
    assert "recent_events=" in text and "recent_ops=" in text
    # The recursive form would repeat the class name per lookback entry.
    assert text.count("RecordContext(") == 1


def test_op_repr_never_doubles_duration_units(loop_trace) -> None:
    """Duration renders through its own unit grammar -- no 'mss' (bug 2)."""

    for op in loop_trace.ops:
        text = str(op)
        assert "mss" not in text
        assert "uss" not in text


def test_bundle_repr_baseline_none_is_unquoted() -> None:
    """A missing baseline is None, not the member name 'None' (bug 17)."""

    model = _TinyMLP().eval()
    x = torch.randn(2, 8)
    bundle = tl.bundle({"clean": tl.trace(model, x), "other": tl.trace(model, x)})
    text = repr(bundle)
    assert "baseline=None" in text
    assert "baseline='None'" not in text


def test_op_accessor_iteration_yields_ops(loop_trace) -> None:
    """BREAKING basis fix (bug 27): iteration yields Op records, not ints."""

    layer = next(layer for layer in loop_trace.layers if layer.num_passes == 3)
    members = list(layer.ops)
    assert len(members) == 3
    assert all(type(member).__name__ == "Op" for member in members)
    assert [member.pass_index for member in members] == [1, 2, 3]


def test_op_accessor_get_matches_getitem_basis(loop_trace) -> None:
    """get/[] share the 0-based positional basis (bug 27)."""

    layer = next(layer for layer in loop_trace.layers if layer.num_passes == 3)
    assert layer.ops.get(0) is layer.ops[0]
    assert layer.ops.get(2) is layer.ops[2]
    assert layer.ops.get(99) is None
    label = layer.ops[1].label
    assert layer.ops.get(label) is layer.ops[1]


def test_op_accessor_repr_teaches_real_basis(loop_trace) -> None:
    """The repr names the 0-based/pass-qualified basis, bounded (bug 27)."""

    layer = next(layer for layer in loop_trace.layers if layer.num_passes == 3)
    text = repr(layer.ops)
    assert "0-based positions" in text
    assert layer.ops[0].label[:6] in text
    assert len(text) < 400


def test_four_option_classes_ride_the_options_namespace() -> None:
    """CaptureOptions/SaveOptions/ReplayOptions/InterventionOptions (bug 25).

    Bug 25 wanted the four option dataclasses importable from a documented
    public spelling. That spelling is the ``tl.options`` namespace -- the
    root surface is a FROZEN BUDGET under the arch-spine lockstep
    (tests/test_arch_spine_surface.py; megaplan conflict-ledger rows 3/9/17:
    C02 rides submodules, zero new top-level names), so the classes must NOT
    appear as bare ``tl.NAME`` root attributes.
    """

    from torchlens import options

    for name in ("CaptureOptions", "SaveOptions", "ReplayOptions", "InterventionOptions"):
        assert hasattr(options, name), name
        assert name not in tl.__all__, f"{name} widened the frozen root budget"
        assert name not in tl._LAZY_ATTRS, f"{name} widened the frozen root budget"


def test_edge_refusal_remedy_is_runnable_spelling(loop_trace) -> None:
    """The edges refusal teaches an importable spelling (bugs 20/25)."""

    with pytest.raises(Exception, match=r"tl\.options\.CaptureOptions") as excinfo:
        _ = loop_trace.edges
    assert getattr(excinfo.value, "fields", {}).get("code") == "edge_provenance_unavailable"


def test_run_products_have_bounded_reprs() -> None:
    """RunResult/RunReport reprs are verdict-first descriptors (bug 9)."""

    model = _TinyMLP().eval()
    x = torch.randn(2, 8)
    trace = tl.trace(model, x)
    result = trace.run(inputs=torch.randn(2, 8))
    result_text = repr(result)
    assert result_text.startswith("RunResult(output=")
    assert "tensor(" not in result_text
    assert len(result_text) < 400
    report_text = repr(result.report)
    assert report_text.startswith("RunReport(")
    assert "\n" not in report_text
    assert len(report_text) < 500
    readiness_text = repr(result.report.readiness)
    assert readiness_text.startswith("ReadinessReport(status=")
    assert len(readiness_text) < 500


def test_population_state_sentinel_aware() -> None:
    """String sentinels stop certifying as populated (bug 24)."""

    from capture_oracle._characterize import _population_state

    assert _population_state("unknown") == "sentinel_unknown"
    assert _population_state("unavailable") == "sentinel_unknown"
    assert _population_state("n/a") == "sentinel_unknown"
    assert _population_state("copy") == "populated"
    assert _population_state(None) == "unknown"
    assert _population_state(0) == "defaulted_zero"


def test_tree_surfaces_are_ascii(loop_trace) -> None:
    """Module/profile trees emit ASCII rails only (bug 10)."""

    from io import StringIO

    stream = StringIO()
    loop_trace.modules["self"].show_call_tree(file=stream)
    tree_text = stream.getvalue()
    profile_tree = loop_trace.profile().tree()
    for text in (tree_text, profile_tree):
        assert text == text.encode("ascii", errors="replace").decode("ascii"), (
            "non-ASCII byte in a tree surface"
        )
