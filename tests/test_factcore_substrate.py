"""C02 FactCore substrate: identity partition, grains, joins, health, scopes.

sumfam item 7 (FactCore + IdentityIndex + grain menu + capture fingerprint),
item 8 (serialized HealthFacts + three-state verdict + alias exclusion),
item 9 (at_capture/retained_now), and the costreport item-2 aggregation face
with its D7 boundary-row CI plant.
"""

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.report._compute_truth import compute_aggregation
from torchlens.report._factcore import GRAINS, factcore
from torchlens.report._health import health_facts


class _Loop(nn.Module):
    """Three-pass recurrent layer (multi-pass join rows)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.relu(self.linear(x))
        return x


@pytest.fixture(scope="module")
def loop_trace():
    """One finished multi-pass trace, cleaned up at module teardown."""

    trace = tl.trace(_Loop().eval(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


# Kept by the shrink-r4 coverage proof: sole branch coverage of report/_factcore.py.
def test_counts_record_names_its_grains(loop_trace) -> None:
    """D3: one counts record; the bare word 'operations' has no home."""

    core = factcore(loop_trace)
    counts = core.counts
    alias_rows = sum(
        1
        for op in loop_trace._layer_list
        if getattr(op, "is_input", False)
        or getattr(op, "is_output", False)
        or getattr(op, "is_buffer", False)
    )
    assert counts.tracked_tensor_rows == len(loop_trace._layer_list)
    assert counts.alias_rows == alias_rows
    assert counts.compute_ops == counts.tracked_tensor_rows - alias_rows
    assert counts.for_grain("op") == counts.compute_ops
    assert counts.for_grain("module") == counts.modules
    for grain in GRAINS:
        assert counts.for_grain(grain) >= 0
    with pytest.raises(Exception, match="grain") as excinfo:
        counts.for_grain("operations")
    assert excinfo.value.fields["code"] == "factcore_grain_invalid"


# Kept by the shrink-r4 coverage proof: sole branch coverage of report/_factcore.py.
def test_identity_index_joins_refuse_ambiguity(loop_trace) -> None:
    """D2: shared identities; joins are total or refuse typed."""

    core = factcore(loop_trace)
    identity = core.identity
    multi_pass_layer = next(
        layer.layer_label for layer in loop_trace.layers if layer.num_passes == 3
    )
    member_ops = identity.ops_of_layer(multi_pass_layer)
    assert len(member_ops) == 3
    for op_label in member_ops:
        assert identity.layer_of_op(op_label) == multi_pass_layer
    with pytest.raises(Exception) as excinfo:
        identity.ops_of_layer("not_a_layer")
    assert excinfo.value.fields["code"] == "factcore_identity_unknown"
    with pytest.raises(Exception) as excinfo:
        identity.layer_of_op("not_an_op:9")
    assert excinfo.value.fields["code"] == "factcore_identity_unknown"


# Kept by the shrink-r4 coverage proof: sole branch coverage of report/_compute_truth.py.
def test_compute_face_partition_and_coverage_totality(loop_trace) -> None:
    """The partition total equals the op-row sum; coverage is total."""

    aggregation = compute_aggregation(loop_trace)
    row_sum = sum(row.flops_fma2 or 0 for row in aggregation.rows if row.kind == "op")
    assert int(aggregation.partition_total) == row_sum
    assert aggregation.coverage.total_rows == len(aggregation.rows)
    by_dtype_sum = sum(value for _, value in aggregation.by_dtype)
    assert by_dtype_sum == row_sum
    assert dict(aggregation.execution_modes).get("forward", 0) == row_sum
    assert int(loop_trace.cost.total_flops_forward) == row_sum


# Kept by the shrink-r4 coverage proof: sole branch coverage of report/_compute_truth.py.
def test_ratio_facts_carry_denominator_ids(loop_trace) -> None:
    """D6: a percent names its numerator and denominator identities."""

    aggregation = compute_aggregation(loop_trace)
    op_row = next(row for row in aggregation.rows if row.kind == "op" and row.flops_fma2)
    ratio = aggregation.ratio(op_row.row_id)
    assert ratio.numerator_id == op_row.row_id
    assert ratio.denominator_id == "partition_total"
    assert ratio.scope == "whole_trace"
    assert 0 < ratio.value <= 1
    missing = aggregation.ratio("op:not_a_row")
    assert missing.value is None


# Kept by the shrink-r4 coverage proof: sole branch coverage of report/_health.py.
def test_health_facts_invalid_persisted_payload_never_clean() -> None:
    """A forged/garbled persisted record degrades to NOT-CHECKED, never clean."""

    model = _Loop().eval()
    trace = tl.trace(model, torch.randn(2, 4))
    trace.annotations["health_facts"] = {"schema_version": "forged"}
    facts = health_facts(trace)
    # The invalid payload is discarded; a LIVE trace still re-derives its
    # own basis, so the verdict recovers from evidence, not the forgery.
    assert facts.basis in ("capture", "saved_payloads")


# Kept by the shrink-r4 coverage proof: sole branch coverage of _data_classes/trace.py, report/_trace_facts.py.
def test_health_alias_rows_excluded_and_disclosed(loop_trace) -> None:
    """D19: alias rows never inflate the identity-partition health counts."""

    model = _Loop().eval()
    x = torch.randn(2, 4)
    x[0, 0] = float("nan")
    trace = tl.trace(model, x)
    facts = trace.health_facts
    alias_labels = {
        str(op.label)
        for op in trace._layer_list
        if getattr(op, "is_input", False)
        or getattr(op, "is_output", False)
        or getattr(op, "is_buffer", False)
    }
    for label in facts.nonfinite_labels:
        assert label not in alias_labels
    for label in facts.alias_nonfinite_labels:
        assert label in alias_labels


# Kept: its run phase reached no library code in the per-id coverage run, so it has no measurement (shrink rule 3).
def test_factcore_grain_vocabulary_is_closed() -> None:
    """The grain menu is the D3 vocabulary, verbatim."""

    assert GRAINS == ("op", "layer", "site", "module", "call")


# Kept by the shrink-r4 coverage proof: sole branch coverage of _data_classes/_trace_groups.py, _data_classes/_trace_inventory.py and more and a sole campaign mutation kill.
def test_scope_label_on_payload_stripped_artifact(tmp_path) -> None:
    """The D8 scope-label test: no surface claims bytes THIS object lacks.

    Before-pin (sumfam exhibit): three surfaces claimed the capture-time
    bytes on an artifact holding zero, with the honest number in the same
    object; and the health channel answered clean with no evidence.
    """

    model = _Loop().eval()
    trace = tl.trace(model, torch.randn(2, 4))
    at_capture = factcore(trace).memory.captured_bytes
    assert at_capture > 0
    trace.save(tmp_path / "stripped.tlspec", include_outs=False)
    loaded = tl.load(tmp_path / "stripped.tlspec")

    memory = factcore(loaded).memory
    assert memory.retained_bytes == 0
    assert memory.captured_bytes == at_capture

    bom = loaded.bill_of_materials()
    assert int(bom["activations"]["retained_now_memory"]) == 0
    assert int(bom["activations"]["at_capture_memory"]) == at_capture

    assert loaded.nonfinite_verdict == "not_checked"  # never clean without evidence

    # Rebuilt default footer: the Memory subject carries BOTH payload scopes.
    footer = [line for line in loaded.summary().splitlines() if line.startswith("Memory")]
    assert footer and "retained now" in footer[0] and "at capture" in footer[0]
    health = [line for line in loaded.summary().splitlines() if "health" in line.lower()]
    assert health and "NOT-CHECKED" in health[0]
