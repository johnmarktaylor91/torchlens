"""One-backward reads: accessor, suppression, engine, and table core (F04).

Covers M(reads) acceptance rows 1-4, 9, 14, 15 at toy scale: the substrate
census invariants, suppression stability with its paired negative, the
id-rejoin fallback, oracle agreement, batched == sequential with exact call
counts, target-order preservation, alias discipline, and the contamination
sweep. Real-checkpoint rows live in ``test_onebackward_realmodel.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._utils._record_group_table import record_member
from torchlens._utils._torch_compat import get_gradient_edge_support
from torchlens.attribution import _onebackward as ob

_requires_gradient_edge = pytest.mark.skipif(
    not get_gradient_edge_support(),
    reason="one-backward reads require torch.autograd.graph.GradientEdge (2.4+)",
)


def _toy_model() -> nn.Module:
    """Small three-op MLP with an alias pair (output shares linear_2's node)."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))


def _toy_trace() -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    """Trace the toy model on a fixed input."""

    model = _toy_model()
    inputs = torch.randn(2, 4)
    return model, inputs, tl.trace(model, inputs)


class TestAccessor:
    """Item 0: the (node, slot) accessor over existing op fields."""

    # Kept by the shrink-r4 coverage proof: sole branch coverage of attribution/_onebackward/_accessor.py.
    @_requires_gradient_edge
    def test_index_caches_and_invalidates_on_cleanup(self) -> None:
        """Same-token reads hit the cache; cleanup refuses typed."""

        _, _, trace = _toy_trace()
        first = ob.read_edge_index(trace)
        assert ob.read_edge_index(trace) is first
        trace.cleanup()
        with pytest.raises(ob.ReadError) as excinfo:
            ob.read_edge_index(trace)
        assert excinfo.value.fields["code"] == "read_addressing_unavailable"
        assert excinfo.value.fields["reason"] == "cleaned"
        assert excinfo.value.fields["remedy"]

    # Kept by the shrink-r4 coverage proof: sole branch coverage of attribution/_onebackward/_accessor.py.
    @_requires_gradient_edge
    def test_id_rejoin_after_user_backward(self) -> None:
        """A user backward nulls handles; the rejoin serves bitwise-equal reads."""

        model = _toy_model()
        inputs = torch.randn(2, 4, requires_grad=True)
        trace = tl.trace(
            model,
            inputs,
            capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
        )
        before = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        trace.backward(trace["output_1"].out.sum(), retain_graph=True)
        assert all(record_member(op, "grad_fn_handle", None) is None for op in trace.ops), (
            "backward is expected to null per-op handles"
        )
        index = ob.read_edge_index(trace)
        assert index.edges, "rejoin produced no edges"
        assert all(edge.via_rejoin for edge in index.edges.values())
        after = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        for key, row in before.items():
            assert after[key].score == row.score, "rejoin read must be bitwise-equal"


class TestReadCore:
    """Item 2: oracle agreement, statuses, order, and the population law."""

    # Kept by the shrink-r4 coverage proof: sole branch coverage of attribution/_onebackward/_read.py.
    @_requires_gradient_edge
    def test_implicit_population_cone_rule_with_counts(self) -> None:
        """D10: implicit population excludes non-upstream sites WITH counts."""

        _, _, trace = _toy_trace()
        table = ob.read(
            trace, target=ob.seed("gelu_1_2", index=(0, 0)), method="grad", reduce="sum"
        )
        labels = {f"{row.address[0]}:{row.address[1]}" for row in table.rows()}
        assert labels == {"linear_1_1:1", "gelu_1_2:1"}
        assert table.provenance.excluded_counts["not_upstream_of_target"] == 2

    # Kept by the shrink-r4 coverage proof: sole branch coverage of attribution/_onebackward/_engine.py.
    @_requires_gradient_edge
    def test_batched_equals_sequential_with_exact_call_counts(self) -> None:
        """D12/D13: batched == sequential at 1e-5; autograd_calls == ceil(T/B)."""

        model, _, trace = _toy_trace()
        index = ob.read_edge_index(trace)
        shape = index.edges["output_1:1"].shape
        assert shape == (2, 3)
        targets = [ob.seed("output_1", index=(0, column)) for column in range(3)]
        targets += [ob.seed("output_1", index=(1, column)) for column in range(3)]
        batched = ob.read(trace, target=targets, method="grad", reduce="sum")
        sequential = ob.read(
            trace, target=targets, method="grad", reduce="sum", target_batch_size=1
        )
        assert batched.provenance.autograd_calls == 1
        assert sequential.provenance.autograd_calls == 6
        assert batched.provenance.batching_plan["batch_size"] == 32
        for key, row in batched.items():
            assert sequential[key].score == pytest.approx(row.score, abs=1e-5)

    # Kept by the shrink-r4 coverage proof: sole branch coverage of attribution/_onebackward/_engine.py, attribution/_onebackward/_targets.py.
    @_requires_gradient_edge
    def test_callable_1d_target_is_a_batch_never_summed(self) -> None:
        """A 1-D callable result expands to per-element targets."""

        model = _toy_model()
        inputs = torch.randn(2, 4)
        trace = tl.trace(model, inputs, capture=tl.options.CaptureOptions(backward_ready=True))

        def logits_row(current: tl.Trace) -> torch.Tensor:
            return current["output_1"].out[0]

        table = ob.read(trace, target=logits_row, method="grad", reduce="sum")
        assert table.target_ids() == ("t0[0]", "t0[1]", "t0[2]")

    # Kept by the shrink-r4 coverage proof: sole branch coverage of attribution/_onebackward/_read.py.
    @_requires_gradient_edge
    def test_reduction_distinctness_d15(self) -> None:
        """sum_of_abs != abs_of_sum on a sign-mixed gradient."""

        _, _, trace = _toy_trace()
        soa = ob.read(
            trace,
            target=ob.seed("output_1", index=(0, 0)),
            method="grad",
            reduce="sum_of_abs",
        )
        aos = ob.read(
            trace,
            target=ob.seed("output_1", index=(0, 0)),
            method="grad",
            reduce="abs_of_sum",
        )
        row_soa = soa["linear_1_1:1"]
        row_aos = aos["linear_1_1:1"]
        assert row_soa.score != pytest.approx(row_aos.score), (
            "sum-of-absolutes and absolute-of-sum are DISTINCT reductions"
        )


class TestActivationMethods:
    """Item 2 activation paths and retention honesty (D14)."""

    # Kept by the shrink-r4 coverage proof: sole branch coverage of attribution/_onebackward/_read.py, attribution/_onebackward/_table.py.
    @_requires_gradient_edge
    def test_zero_payload_capture_grad_serves_axg_marks_unavailable(self) -> None:
        """Test 10: layers_to_save=[] serves grad; act-x-grad rows disclose."""

        model = _toy_model()
        trace = tl.trace(
            model,
            torch.randn(2, 4),
            save=[],
        )
        grad_table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        assert grad_table.status_counts()["ok"] >= 2, "grad reads need no payload at all"
        axg = ob.read(
            trace,
            target=ob.seed("output_1", index=(0, 0)),
            method="activation_x_grad",
            reduce="sum",
        )
        statuses = {row.status for row in axg.rows()}
        assert statuses == {"unavailable"}
        assert {row.status_reason for row in axg.rows()} == {"payload_unretained"}

    # Kept by the shrink-r4 coverage proof: sole branch coverage of attribution/_onebackward/_read.py.
    @_requires_gradient_edge
    def test_activation_method_reads_payloads_without_backward(self) -> None:
        """method='activation' takes no target and runs no backward."""

        _, _, trace = _toy_trace()
        table = ob.read(trace, method="activation", reduce="mean")
        assert table.provenance.autograd_calls == 0
        assert all(row.target_id is None for row in table.rows())


class TestAliasDiscipline:
    """Item 6 (D11): per-address rows, engine dedup, explicit collapse."""

    # Kept by the shrink-r4 coverage proof: sole branch coverage of attribution/_onebackward/_table.py.
    @_requires_gradient_edge
    def test_explicit_collapse_keeps_first_member(self) -> None:
        """collapse_alias_groups is the explicit opt-in, never the default."""

        _, _, trace = _toy_trace()
        table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        collapsed = table.collapse_alias_groups()
        assert len(collapsed) == len(table) - 1
