"""Lazy completion unit acceptance (F17 B12): SOL's four clauses + the flip signal.

Quickstart memo 4.4: the completion unit ships as ONE unit, and the flip
signal is mechanical -- ``check_metadata_invariants(trace)`` passes on the
real lazy-head fixture (the shipped ``trainable + frozen == total`` invariant
that used to fail silently, with zeros). The four-clause acceptance condition
(memo 4.4 item 4) runs as an executable test regardless of the prediction
that some clauses pass vacuously.

The fixture is the way people actually hit lazy modules: a probe head on a
real backbone (toy-scale here; the real-resnet18 variant lives in
``test_quickstart_realmodel.py``).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

import torchlens as tl
from torchlens.validation import check_metadata_invariants

pytestmark = pytest.mark.smoke


def _lazy_head_model() -> nn.Sequential:
    """Backbone + two-layer lazy probe head (the memo's fixture shape)."""

    return nn.Sequential(
        nn.Conv2d(3, 8, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(2),
        nn.Flatten(),
        nn.LazyLinear(6),
        nn.ReLU(),
        nn.LazyLinear(4),
    )


class TestFourClauseAcceptance:
    """SOL's acceptance condition as an executable test."""

    def test_clause_1_saved_activations_byte_correct(self) -> None:
        """COW clause: captured payloads equal a plain forward, byte-exact."""

        torch.manual_seed(0)
        model = _lazy_head_model().eval()
        x = torch.randn(2, 3, 16, 16)
        log = tl.trace(model, x)
        with torch.no_grad():
            expected = model(x)
        assert torch.equal(log.output_ops[0].out, expected)

    def test_clause_2_identity_stable_across_class_swap(self) -> None:
        """Alias clause: object id and optimizer membership survive the swap."""

        model = _lazy_head_model()
        head = model[4]
        pending = head.weight
        assert isinstance(pending, UninitializedParameter)
        pending_id = id(pending)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        log = tl.trace(model, torch.randn(1, 3, 16, 16))
        assert log is not None
        assert not isinstance(head.weight, UninitializedParameter)
        assert id(head.weight) == pending_id
        optimizer_params = {
            id(param) for group in optimizer.param_groups for param in group["params"]
        }
        assert pending_id in optimizer_params

    def test_clause_3_param_records_resolve_post_materialization(self) -> None:
        """Storage clause: the inventory reports the real geometry."""

        model = _lazy_head_model()
        log = tl.trace(model, torch.randn(1, 3, 16, 16))
        head_rows = [
            pl for pl in log.param_logs if getattr(pl, "shape", None) not in (None, (0,), ())
        ]
        ground_truth = {tuple(p.shape) for p in model.parameters()}
        recorded = {tuple(pl.shape) for pl in head_rows}
        assert ground_truth == recorded
        assert all(int(pl.num_params) > 0 for pl in head_rows)

    def test_clause_4_no_op_holds_a_live_pending_payload(self) -> None:
        """Payload clause: no logged record retains an uninitialized tensor."""

        model = _lazy_head_model()
        log = tl.trace(model, torch.randn(1, 3, 16, 16))
        for op in log:
            out = getattr(op, "out", None)
            assert not isinstance(out, UninitializedParameter)
            for arg in getattr(op, "saved_args", None) or ():
                assert not isinstance(arg, UninitializedParameter)


class TestFlipSignal:
    """The mechanical flip signal: metadata invariants on the lazy fixture."""

    def test_invariants_pass_and_totals_match_ground_truth(self) -> None:
        """trainable + frozen == total, and total == the live model's count."""

        model = _lazy_head_model()
        log = tl.trace(model, torch.randn(2, 3, 16, 16))
        check_metadata_invariants(log)
        ground_truth = sum(p.numel() for p in model.parameters())
        assert log.num_params == ground_truth
        assert log.num_params_trainable + log.num_params_frozen == log.num_params
        assert int(log.total_param_memory) > 0

    def test_summary_totals_are_truthful(self) -> None:
        """The one-call summary's totals include the materialized head."""

        model = _lazy_head_model()
        text = str(tl.summary(model, torch.randn(1, 3, 16, 16)))
        ground_truth = sum(p.numel() for p in model.parameters())
        assert f"{ground_truth:,}" in text or str(ground_truth) in text


class TestLazyBoundaries:
    """Refusals that stay in place around the completion unit."""

    def test_zero_arg_rung_refuses_before_probing(self) -> None:
        """Memo 4.2: inference on pending lazy state refuses structurally --
        a lazy module accepts any width, so a probe has no error signal and
        would permanently install the guess in the caller's model."""

        model = _lazy_head_model()
        with pytest.raises(Exception) as excinfo:
            tl.trace(model)
        assert "lazy" in str(excinfo.value).lower()
        assert isinstance(model[4].weight, UninitializedParameter), (
            "the refusal must fire BEFORE any probe touches the model"
        )

    def test_armed_lane_refuses_typed_before_mutation(self) -> None:
        """Memo 4.5: intervention-ready capture on pending state refuses
        state_baseline_unavailable (a pending parameter has no bytes to
        witness), before any mutation."""

        model = _lazy_head_model()
        with pytest.raises(Exception) as excinfo:
            tl.trace(
                model,
                torch.randn(1, 3, 16, 16),
                capture=tl.options.CaptureOptions(intervention_ready=True),
            )
        assert excinfo.value.fields["code"] == "state_baseline_unavailable"
        assert isinstance(model[4].weight, UninitializedParameter)

    def test_taught_self_prime_remedy_executes(self) -> None:
        """The teach's two-line remedy is executed here so it cannot rot."""

        model = _lazy_head_model()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            model(x)  # the taught self-prime
        log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
        assert log.num_params == sum(p.numel() for p in model.parameters())

    def test_lazy_buffer_entry_captures_without_prematerialization(self) -> None:
        """Pending lazy BUFFERS capture directly (F20 buffer-side completion).

        The buffer-side lazy completion flipped the entry refusal off for
        lazy-BUFFER models (LazyBatchNorm*): capture runs the real forward
        and the buffers materialize in place. The lazy_uninitialized teach
        survives only for pending lazy PARAMETERS (the tests above); the
        buffer capture contract is pinned in
        tests/test_entry_facade_lazy_entry.py::test_trace_tolerates_lazy_buffer_model.
        """

        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.LazyBatchNorm2d()).eval()
        log = tl.trace(model, torch.randn(1, 3, 8, 8))
        assert log.outcome.status.name == "COMPLETE"
        assert not isinstance(model[1].running_mean, UninitializedBuffer)
