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
from torchlens.attribution import onebackward as ob
from torchlens.utils._torch_compat import get_gradient_edge_support

pytestmark = pytest.mark.smoke

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

    @_requires_gradient_edge
    def test_substrate_census_no_walkable_holes(self) -> None:
        """Every differentiable op resolves; only the graph input is refused."""

        _, _, trace = _toy_trace()
        index = ob.read_edge_index(trace)
        assert set(index.unaddressable) == {"input_1:1"}
        assert index.unaddressable["input_1:1"] == "no_grad_fn"
        assert len(index.edges) == 4
        for edge in index.edges.values():
            assert edge.node is not None
            assert not edge.via_rejoin
            assert edge.slot == 0

    @_requires_gradient_edge
    def test_alias_identity_is_node_and_slot(self) -> None:
        """The output op and its producing linear share one (node, slot)."""

        _, _, trace = _toy_trace()
        index = ob.read_edge_index(trace)
        assert len(index.alias_groups) == 1
        (members,) = index.alias_groups.values()
        assert set(members) == {"linear_2_3:1", "output_1:1"}

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

    @_requires_gradient_edge
    def test_id_rejoin_after_user_backward(self) -> None:
        """A user log_backward nulls handles; the rejoin serves bitwise-equal reads."""

        model = _toy_model()
        inputs = torch.randn(2, 4, requires_grad=True)
        trace = tl.trace(
            model,
            inputs,
            capture=tl.options.CaptureOptions(backward_ready=True),
            save_mode="reference",
        )
        before = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        trace.log_backward(trace["output_1"].out.sum(), retain_graph=True)
        assert all(getattr(op, "grad_fn_handle", None) is None for op in trace.ops), (
            "log_backward is expected to null per-op handles"
        )
        index = ob.read_edge_index(trace)
        assert index.edges, "rejoin produced no edges"
        assert all(edge.via_rejoin for edge in index.edges.values())
        after = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        for key, row in before.items():
            assert after[key].score == row.score, "rejoin read must be bitwise-equal"


class TestSuppression:
    """Item 0b: suppression stability and its paired negative."""

    @_requires_gradient_edge
    def test_suppressed_reads_leave_capture_state_unchanged(self) -> None:
        """Handles, counters, and pinned refs are unchanged after N reads."""

        _, _, trace = _toy_trace()
        refs_before = len(trace.__dict__["_backward_gradfn_refs"])
        passes_before = trace.num_backward_passes
        for _ in range(4):
            ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum")
        assert trace.num_backward_passes == passes_before
        assert len(trace.__dict__["_backward_gradfn_refs"]) == refs_before
        index = ob.read_edge_index(trace)
        assert all(not edge.via_rejoin for edge in index.edges.values()), (
            "suppressed reads must not consume live handles"
        )

    def test_unsuppressed_backward_is_the_negative(self) -> None:
        """The paired negative: an ordinary backward does advance capture state."""

        model = _toy_model()
        inputs = torch.randn(2, 4, requires_grad=True)
        trace = tl.trace(
            model,
            inputs,
            capture=tl.options.CaptureOptions(backward_ready=True),
            save_mode="reference",
        )
        assert trace.num_backward_passes == 0
        trace.log_backward(trace["output_1"].out.sum(), retain_graph=True)
        assert trace.num_backward_passes == 1

    @_requires_gradient_edge
    def test_user_installed_hooks_still_run(self) -> None:
        """Documented boundary: user PyTorch hooks fire inside suppression."""

        _, _, trace = _toy_trace()
        index = ob.read_edge_index(trace)
        fired: list[str] = []
        handle = index.edges["gelu_1_2:1"].node.register_prehook(
            lambda grads: fired.append("user") or None
        )
        try:
            ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum")
        finally:
            handle.remove()
        assert fired, "user-installed prehooks must still run inside suppression"


class TestReadCore:
    """Item 2: oracle agreement, statuses, order, and the population law."""

    @_requires_gradient_edge
    def test_hand_rolled_oracle_agreement(self) -> None:
        """The read's gradient equals a hand-rolled plain-autograd reference."""

        model, inputs, trace = _toy_trace()
        table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        h1 = model[0](inputs)
        h2 = model[1](h1)
        h3 = model[2](h2)
        h1.retain_grad(), h2.retain_grad()
        manual_h1, manual_h2 = torch.autograd.grad(h3[0, 0], [h1, h2])
        assert table["linear_1_1:1"].score == pytest.approx(float(manual_h1.sum()), abs=1e-6)
        assert table["gelu_1_2:1"].score == pytest.approx(float(manual_h2.sum()), abs=1e-6)

    @_requires_gradient_edge
    def test_numeric_zero_stays_ok_none_becomes_unreachable(self) -> None:
        """D3: autograd None is 'unreachable'; a real zero is 'ok'."""

        _, _, trace = _toy_trace()
        table = ob.read(
            trace,
            target=ob.seed("gelu_1_2", index=(0, 0)),
            method="grad",
            reduce="sum",
            within=tl.units("linear_2_3", [(0, 0)]) | tl.units("linear_1_1", [(0, 0)]),
        )
        by_label = {f"{row.address[0]}:{row.address[1]}": row for row in table.rows()}
        unreachable = by_label["linear_2_3:1"]
        assert unreachable.status == "unreachable"
        assert unreachable.status_reason == "not_upstream_of_target"
        assert unreachable.score is None, "an unreachable row must never fabricate a zero"
        assert by_label["linear_1_1:1"].status == "ok"

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

    @_requires_gradient_edge
    def test_target_order_preserved(self) -> None:
        """Target ids are t0, t1, ... in request order."""

        _, _, trace = _toy_trace()
        table = ob.read(
            trace,
            target=[
                ob.seed("output_1", index=(0, 1)),
                ob.seed("output_1", index=(0, 0)),
            ],
            method="grad",
            reduce="sum",
        )
        assert table.target_ids() == ("t0", "t1")

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

    @_requires_gradient_edge
    def test_element_grain_values_on_carrier(self) -> None:
        """reduce=None emits detached dense values on the carrier."""

        _, _, trace = _toy_trace()
        table = ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce=None)
        row = table["linear_1_1:1"]
        assert row.grain == "element"
        assert isinstance(row.value, torch.Tensor)
        assert row.value.shape == (2, 8)
        assert row.value.grad_fn is None, "carried values must be detached"

    @_requires_gradient_edge
    def test_repeat_read_and_later_user_backward_work(self) -> None:
        """retain_graph contract: reads repeat and a later backward works."""

        model = _toy_model()
        inputs = torch.randn(2, 4, requires_grad=True)
        trace = tl.trace(
            model,
            inputs,
            capture=tl.options.CaptureOptions(backward_ready=True),
            save_mode="reference",
        )
        first = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        second = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        for key, row in first.items():
            assert second[key].score == row.score
        trace.log_backward(trace["output_1"].out.sum(), retain_graph=True)
        assert trace.num_backward_passes == 1

    @_requires_gradient_edge
    def test_contamination_sweep(self) -> None:
        """Test 15: params, RNG, module modes, and grads unchanged by a read."""

        model, inputs, trace = _toy_trace()
        params_before = [param.detach().clone() for param in model.parameters()]
        grads_before = [param.grad for param in model.parameters()]
        rng_before = torch.get_rng_state()
        modes_before = [module.training for module in model.modules()]
        ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum")
        for param, before in zip(model.parameters(), params_before, strict=True):
            assert torch.equal(param.detach(), before)
        for param, before in zip(model.parameters(), grads_before, strict=True):
            assert param.grad is before or torch.equal(param.grad, before)
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert [module.training for module in model.modules()] == modes_before


class TestActivationMethods:
    """Item 2 activation paths and retention honesty (D14)."""

    @_requires_gradient_edge
    def test_activation_x_grad_equals_manual_product(self) -> None:
        """act x grad rows equal payload * gradient elementwise."""

        _, _, trace = _toy_trace()
        grad_table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce=None
        )
        axg_table = ob.read(
            trace,
            target=ob.seed("output_1", index=(0, 0)),
            method="activation_x_grad",
            reduce=None,
        )
        payload = trace["gelu_1_2"].out
        manual = payload * grad_table["gelu_1_2:1"].value
        assert torch.allclose(axg_table["gelu_1_2:1"].value, manual, atol=0, rtol=0)

    @_requires_gradient_edge
    def test_zero_payload_capture_grad_serves_axg_marks_unavailable(self) -> None:
        """Test 10: layers_to_save=[] serves grad; act-x-grad rows disclose."""

        model = _toy_model()
        trace = tl.trace(
            model,
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(layers_to_save=[]),
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

    @_requires_gradient_edge
    def test_activation_method_reads_payloads_without_backward(self) -> None:
        """method='activation' takes no target and runs no backward."""

        _, _, trace = _toy_trace()
        table = ob.read(trace, method="activation", reduce="mean")
        assert table.provenance.autograd_calls == 0
        assert all(row.target_id is None for row in table.rows())


class TestAliasDiscipline:
    """Item 6 (D11): per-address rows, engine dedup, explicit collapse."""

    @_requires_gradient_edge
    def test_alias_rows_share_score_and_disclose_group(self) -> None:
        """Alias members appear as separate rows with one shared gradient."""

        _, _, trace = _toy_trace()
        table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        linear_row = table["linear_2_3:1"]
        output_row = table["output_1:1"]
        assert linear_row.alias_group is not None
        assert linear_row.alias_group == output_row.alias_group
        assert linear_row.score == output_row.score
        assert table.provenance.alias_group_count == 1
        assert table.provenance.aliased_site_count == 2

    @_requires_gradient_edge
    def test_explicit_collapse_keeps_first_member(self) -> None:
        """collapse_alias_groups is the explicit opt-in, never the default."""

        _, _, trace = _toy_trace()
        table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        collapsed = table.collapse_alias_groups()
        assert len(collapsed) == len(table) - 1
