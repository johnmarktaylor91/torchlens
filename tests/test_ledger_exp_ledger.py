"""F03 ledger memo items 9-11: the experiment ledger + read-only serving.

Pins the failure rows the memo funds: arm-time fail-closed preflight,
per-event durability with the kill -9 invariant (exactly k finalized events
recovered, torn tail discarded AND disclosed), interior tamper refusal,
quarantine-on-sink-death returning the material result
(material_action_completed), on_record_error='raise' opt-in, single-writer
lock, ContextVar isolation, verdict basis honesty ("asserted; no recorded
basis"), declared_at_seq, close-without-verdict refusal, and the three
read-only serving tools over the fresh artifact.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from torchlens.errors import TorchLensWarning
from torchlens.errors.episode import BundleExperimentError
from torchlens.experiment import (
    active_ledger,
    ledger,
    ledger_entry,
    ledger_evidence,
    ledger_overview,
)
from torchlens.experiment._ledger import EvidenceRef, read_ledger_artifact

pytestmark = pytest.mark.smoke


def _armed(tmp_path: Path, **kwargs):
    return ledger(tmp_path / "run.tlledger", question="does head 5 matter?", **kwargs)


def test_arm_opens_entry_and_contextvar_scopes(tmp_path: Path) -> None:
    assert active_ledger() is None
    led = _armed(tmp_path, hypothesis="head 5 carries the effect", metric="logit diff")
    try:
        assert active_ledger() is led
        entries = led.entries()
        assert list(entries) == ["e1"]
        entry = entries["e1"]
        assert not entry.draft
        assert entry.hypothesis_declared_at_seq is not None
        assert entry.metric_declared_at_seq is not None
        assert entry.metric == "logit diff"
    finally:
        led.close()
    assert active_ledger() is None


def test_draft_without_hypothesis_and_verdict_basis_honesty(tmp_path: Path) -> None:
    led = ledger(tmp_path / "run.tlledger")
    try:
        assert led.entries()["e1"].draft
        assert " DRAFT" in led.entries()["e1"].overview_line()
        # A verdict with no basis is legal and VISIBLY weaker.
        led.verdict("inconclusive", text="eyeballed")
        line = led.entries()["e1"].overview_line()
        assert "asserted; no recorded basis" in line
        # The lint disclosure fired (verdict over zero observations).
        assert any(event.kind == "lint" for event in led.events)
        # An amended verdict APPENDS and records what it amends.
        led.observe("baseline logit -94.9", value=-94.9)
        led.verdict("supports", text="now with data", basis=["step-1"])
        entry = led.entries()["e1"]
        assert entry.verdict is not None
        assert entry.verdict["token"] == "supports"
        assert entry.verdict["amends"] == "inconclusive"
        # The old verdict event is preserved in the stream.
        verdict_events = [event for event in led.events if event.kind == "verdict_set"]
        assert len(verdict_events) == 2
    finally:
        led.close()


def test_close_without_verdict_refuses_typed(tmp_path: Path) -> None:
    led = ledger(tmp_path / "run.tlledger")
    try:
        with pytest.raises(BundleExperimentError) as excinfo:
            led.close_entry()
        assert excinfo.value.fields["code"] == "ledger_close_without_verdict"
        led.close_entry(exploratory=True)
        assert led.entries()["e1"].status == "closed"
    finally:
        led.close()


def test_verdict_vocabulary_closed(tmp_path: Path) -> None:
    led = ledger(tmp_path / "run.tlledger")
    try:
        with pytest.raises(BundleExperimentError) as excinfo:
            led.verdict("proven")
        assert excinfo.value.fields["code"] == "ledger_verdict_invalid"
    finally:
        led.close()


def test_kill_minus_nine_invariant_and_torn_tail_disclosure(tmp_path: Path) -> None:
    path = tmp_path / "run.tlledger"
    led = ledger(path, hypothesis="h", metric="m")
    step = led.material_step("site_sweep")
    led.finalize_step(step, outcome="completed")
    n_events = len(led.events)
    led.close()

    # Simulate a SIGKILL mid-write: append half an event line.
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 99, "event_id": "torn')

    with pytest.warns(TorchLensWarning, match="torn") as caught:
        recovered = ledger(path)
    assert caught[0].message.fields["code"] == "ledger_torn_tail_discarded"
    try:
        # Exactly the k finalized events recovered (+ the new entry's rows).
        recovered_kinds = [event.kind for event in recovered.events]
        assert recovered_kinds[:n_events] == [
            "entry_opened",
            "hypothesis_set",
            "metric_chosen",
            "material_step_started",
            "material_step_finalized",
        ]
    finally:
        recovered.close()


def test_interior_tamper_refuses_at_the_break(tmp_path: Path) -> None:
    path = tmp_path / "run.tlledger"
    led = ledger(path, hypothesis="h")
    led.observe("one")
    led.observe("two")
    led.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    # Tamper an INTERIOR event (flip its payload without re-deriving digests).
    record = json.loads(lines[2])
    record["payload"] = {"hypothesis": "forged"}
    lines[2] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(BundleExperimentError) as excinfo:
        read_ledger_artifact(path)
    assert excinfo.value.fields["code"] == "ledger_artifact_invalid"


def test_quarantine_on_sink_death_returns_material_result(tmp_path: Path) -> None:
    import contextlib

    led = ledger(tmp_path / "run.tlledger", hypothesis="h")
    try:
        step = led.material_step("vary")
        # Kill the sink AFTER material work.
        led._handle.close()
        material = {"the": "bundle"}
        with pytest.warns(TorchLensWarning, match="QUARANTINED"):
            result = led.finalize_step(step, outcome="completed", material_result=material)
        assert result is material  # the material result is returned unharmed
        with pytest.raises(BundleExperimentError) as excinfo:
            led.observe("entry is disarmed")
        assert excinfo.value.fields["code"] == "ledger_entry_quarantined"
        # Explicit save() RAISES on a dead sink (no material result at risk).
        with pytest.raises(BundleExperimentError):
            led.save()
    finally:
        with contextlib.suppress(BundleExperimentError):
            led.close()


def test_on_record_error_raise_opt_in(tmp_path: Path) -> None:
    import contextlib

    led = ledger(tmp_path / "run.tlledger", hypothesis="h", on_record_error="raise")
    try:
        step = led.material_step("vary")
        led._handle.close()
        material = {"the": "bundle"}
        with pytest.raises(BundleExperimentError) as excinfo:
            led.finalize_step(step, outcome="completed", material_result=material)
        fields = excinfo.value.fields
        assert fields["code"] == "ledger_write_failed"
        assert fields["material_action_completed"] is True
        assert fields["result"] is material  # nothing lost: the exception carries it
    finally:
        with contextlib.suppress(BundleExperimentError):
            led.close()


def test_single_writer_lock_refuses_second_writer(tmp_path: Path) -> None:
    path = tmp_path / "run.tlledger"
    led = ledger(path, hypothesis="h")
    try:
        failure: list[BaseException] = []

        def second_writer() -> None:
            try:
                ledger(path)
            except BaseException as exc:  # noqa: BLE001
                failure.append(exc)

        # A different context (thread) does NOT inherit the arm; the lock
        # refuses the competing writer.
        worker = threading.Thread(target=second_writer)
        worker.start()
        worker.join()
        assert failure and isinstance(failure[0], BundleExperimentError)
        assert failure[0].fields["code"] == "ledger_locked"
    finally:
        led.close()


def test_nested_arming_refuses_same_scope(tmp_path: Path) -> None:
    led = ledger(tmp_path / "one.tlledger", hypothesis="h")
    try:
        with pytest.raises(BundleExperimentError) as excinfo:
            ledger(tmp_path / "two.tlledger")
        assert excinfo.value.fields["code"] == "ledger_already_armed"
    finally:
        led.close()


def test_ledger_json_is_reference_only(tmp_path: Path) -> None:
    path = tmp_path / "run.tlledger"
    led = ledger(path, hypothesis="ablating head 5 drops the Paris logit", metric="logit")
    step = led.material_step(
        "site_sweep", refs=[EvidenceRef(kind="bundle", uri="sweep_bundle", digest="a" * 64)]
    )
    led.finalize_step(step, outcome="completed")
    led.observe("head 5 moved the logit", value=11.09)
    led.verdict("supports", basis=[step])
    led.close()
    text = path.read_text(encoding="utf-8")
    # References only: no tensors, no prompts beyond the user's own strings.
    assert "torch.Tensor" not in text
    assert '"digest":"' + "a" * 64 in text


# ---------------------------------------------------------------------------
# item 11: the three read-only serving tools (fresh mid-experiment artifact)
# ---------------------------------------------------------------------------


def test_serving_tools_read_live_artifact(tmp_path: Path) -> None:
    path = tmp_path / "run.tlledger"
    led = ledger(path, hypothesis="h", metric="m", question="q?")
    step = led.material_step(
        "site_sweep", refs=[EvidenceRef(kind="bundle", uri="missing_bundle", digest="b" * 64)]
    )
    led.finalize_step(step, outcome="completed")
    led.observe("reading", value=1.0)
    led.verdict("supports", basis=[step])

    # The artifact is FRESH while the ledger is still armed (per-event fsync).
    overview = ledger_overview(str(path))
    assert overview["n_events"] >= 6
    assert len(overview["entries"]) == 1
    assert "verdict=supports (basis=1)" in overview["entries"][0]

    served = ledger_entry(str(path), "e1")
    assert served["hypothesis"] == "h"
    assert served["metric_declared_at_seq"] is not None
    assert served["n_events"] >= 6
    assert served["evidence_refs"][0]["availability"] == "missing"

    evidence = ledger_evidence(str(path), "e1", 0)
    assert evidence["served"] is None
    assert "missing" in (evidence["disclosure"] or "")
    led.close()

    with pytest.raises(BundleExperimentError) as excinfo:
        ledger_entry(str(path), "e9")
    assert excinfo.value.fields["code"] == "ledger_entry_unknown"


def test_unarmed_context_writes_zero_rows(tmp_path: Path) -> None:
    # No armed ledger: material verbs emit nothing, behavior byte-identical.
    assert active_ledger() is None
    path = tmp_path / "never.tlledger"
    assert not path.exists()


def test_emission_one_step_per_operation(tmp_path: Path) -> None:
    """9b: an armed sweep/vary is ONE step each; unarmed emits nothing."""

    import torch
    from torch import nn

    import torchlens as tl
    from torchlens.experiment import site_sweep

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))

    def metric(member: tl.Trace) -> float:
        return float(member[member.output_layers[0]].out.sum().item())

    led = ledger(tmp_path / "run.tlledger", hypothesis="h", metric="m")
    try:
        bundle = site_sweep(
            baseline,
            candidates=["relu_1_2"],
            edit=tl.zero_ablate(),
            metric=metric,
            retain="all",
            model=model,
            x=x,
        )
        bundle.vary({"baseline": None, "c0": None})
        entry = led.entries()["e1"]
        # ONE step per top-level operation: one sweep + one vary.
        assert len(entry.steps) == 2
        operations = sorted(step["operation"] for step in entry.steps.values())
        assert operations == ["site_sweep", "vary"]
        assert all(step["finalized"] for step in entry.steps.values())
        # The step references the material record, never copies it.
        refs = [ref for step in entry.steps.values() for ref in step["refs"]]
        assert all(ref["uri"].startswith("live://") for ref in refs)
    finally:
        led.close()


def test_mcp_server_serves_the_three_ledger_tools(tmp_path: Path) -> None:
    """Item 11: the stdio server's pure tool layer wraps the same functions."""

    from torchlens.bridge.mcp import TOOL_SPECS, call_tool

    names = {spec["name"] for spec in TOOL_SPECS}
    assert {
        "torchlens_ledger_overview",
        "torchlens_ledger_entry",
        "torchlens_ledger_evidence",
    } <= names

    path = tmp_path / "run.tlledger"
    led = ledger(path, hypothesis="h", metric="m")
    step = led.material_step(
        "site_sweep", refs=[EvidenceRef(kind="bundle", uri="gone", digest="c" * 64)]
    )
    led.finalize_step(step, outcome="completed")
    led.verdict("not_assessed")
    led.close()

    overview = call_tool("torchlens_ledger_overview", {"path": str(path)})
    assert overview["entries"]
    served = call_tool("torchlens_ledger_entry", {"path": str(path), "entry_id": "e1"})
    assert served["hypothesis"] == "h"
    evidence = call_tool(
        "torchlens_ledger_evidence", {"path": str(path), "entry_id": "e1", "ref_index": 0}
    )
    assert evidence["served"] is None and "missing" in evidence["disclosure"]


def test_arm_option_vocabulary_refuses_typed(tmp_path: Path) -> None:
    """on_record_error is a closed two-token vocabulary (D4g opt-in)."""

    with pytest.raises(BundleExperimentError) as excinfo:
        ledger(tmp_path / "run.tlledger", on_record_error="ignore")  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "ledger_option_invalid"


def test_arm_preflight_fails_closed_on_dead_sink(tmp_path: Path) -> None:
    """A path that cannot host the artifact refuses BEFORE material work."""

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("occupied", encoding="utf-8")
    with pytest.raises(BundleExperimentError) as excinfo:
        ledger(blocker / "run.tlledger")
    assert excinfo.value.fields["code"] == "ledger_arm_failed"
    assert active_ledger() is None  # the failed arm never latched the context


def test_semantic_write_without_entry_refuses_typed(tmp_path: Path) -> None:
    """The no-open-entry belt refuses typed (fail-closed guard).

    Arming always opens entry 1, so the guarded state is constructed
    directly: the belt must hold even if a future lifecycle change (an
    explicit entry-close that clears the pointer) exposes it publicly.
    """

    led = _armed(tmp_path)
    try:
        led._current_entry_id = None
        with pytest.raises(BundleExperimentError) as excinfo:
            led.observe("orphan row")
        assert excinfo.value.fields["code"] == "ledger_no_entry"
    finally:
        led._current_entry_id = "e1"
        led.close()


def test_evidence_ref_index_out_of_range_refuses_typed(tmp_path: Path) -> None:
    """ledger_evidence on an entry with zero refs refuses typed."""

    path = tmp_path / "run.tlledger"
    led = ledger(path, hypothesis="h", metric="m")
    led.verdict("not_assessed")
    led.close()
    with pytest.raises(BundleExperimentError) as excinfo:
        ledger_evidence(str(path), "e1", 0)
    assert excinfo.value.fields["code"] == "ledger_evidence_unavailable"
