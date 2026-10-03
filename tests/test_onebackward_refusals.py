"""One-backward reads: the typed refusal surface (F04, teaching refusals).

Every refusal branches on ``exc.fields["code"]`` (never message text) and
carries a non-empty remedy. Covers the liveness gate, target normalizer
teaching (memo test 10's TypeError kill), option vocabularies, the explicit
population contract (D9), the byte budget, save-mode honesty (D14), and the
table's own refusals.
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


def _trace(**capture_kwargs) -> tl.Trace:
    """Trace a small MLP with optional capture options."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    options = capture_kwargs.pop("capture", None)
    if options is not None:
        return tl.trace(model, torch.randn(2, 4), capture=options, **capture_kwargs)
    return tl.trace(model, torch.randn(2, 4), **capture_kwargs)


def _read_error(callable_, code: str, **fields):
    """Assert a ReadError with the given code and field subset; return it."""

    with pytest.raises(ob.ReadError) as excinfo:
        callable_()
    error = excinfo.value
    assert error.fields["code"] == code, error.fields
    assert error.fields.get("remedy"), f"refusal {code} has no remedy"
    for key, value in fields.items():
        assert error.fields[key] == value, (key, error.fields)
    return error


def _grad_table(trace: tl.Trace) -> ob.ReadTable:
    """One site-grain gradient table over the implicit population."""

    return ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum")


class TestLivenessGate:
    """The closed trace-level addressing gate (D1/D14)."""

    @_requires_gradient_edge
    def test_inference_only_refuses(self) -> None:
        trace = _trace(capture=tl.options.CaptureOptions(inference_only=True))
        _read_error(
            lambda: ob.read_edge_index(trace),
            "read_addressing_unavailable",
            reason="inference_only",
        )

    @_requires_gradient_edge
    def test_structure_only_refuses(self) -> None:
        trace = _trace(capture=tl.options.CaptureOptions(structure_only=True))
        _read_error(
            lambda: ob.read_edge_index(trace),
            "read_addressing_unavailable",
            reason="structure_only",
        )

    @_requires_gradient_edge
    def test_loaded_trace_refuses_not_live(self, tmp_path) -> None:
        trace = _trace()
        path = tmp_path / "artifact.tlspec"
        tl.save(trace, str(path))
        loaded = tl.load(str(path))
        _read_error(
            lambda: ob.read_edge_index(loaded),
            "read_addressing_unavailable",
            reason="not_live",
        )

    @_requires_gradient_edge
    def test_graph_freed_refuses_typed(self) -> None:
        trace = _trace(
            capture=tl.options.CaptureOptions(backward_ready=True),
            save_mode="reference",
        )
        # Free the graph with a user backward WITHOUT retain_graph.
        trace["output_1"].out.sum().backward()
        _read_error(
            lambda: ob.read(
                trace,
                target=ob.seed("output_1", index=(0, 0)),
                method="grad",
                reduce="sum",
            ),
            "read_addressing_unavailable",
            reason="graph_freed",
        )


class TestTargetNormalizer:
    """Memo test 10: teaching refusals, never a bare TypeError."""

    @_requires_gradient_edge
    def test_detached_tensor_target_teaches_edge_seeding(self) -> None:
        """A graph-disconnected tensor target refuses naming seed(...).

        The realistic Captum-migrant mistake: a target computed from detached
        values (a detached payload slice, a fresh tensor). The refusal is
        typed and names the edge-seeded spelling, never a bare TypeError.
        """

        trace = _trace()
        detached_target = trace["output_1"].out.detach()[0, 0]
        error = _read_error(
            lambda: ob.read(trace, target=detached_target, method="grad", reduce="sum"),
            "read_target_invalid",
        )
        assert "seed(" in str(error), "the refusal must name the edge-seeded spelling"

    @_requires_gradient_edge
    def test_unsaved_payload_spelling_never_reaches_a_bare_typeerror(self) -> None:
        """On a selectively-retained trace the payload route dies before the
        read; the read-level refusal for a non-tensor target is typed."""

        trace = _trace(capture=tl.options.CaptureOptions(layers_to_save=[]))
        unsaved = trace["output_1"].out  # None on a zero-retention capture
        assert unsaved is None
        error = _read_error(
            lambda: ob.read(trace, target=unsaved, method="grad", reduce="sum"),
            "read_target_invalid",
        )
        assert "seed(" in str(error)

    @_requires_gradient_edge
    def test_wrong_shape_cotangent_names_recorded_shape(self) -> None:
        trace = _trace(capture=tl.options.CaptureOptions(layers_to_save=[]))
        error = _read_error(
            lambda: ob.read(
                trace,
                target=ob.seed("output_1", cotangent=torch.ones(5, 5)),
                method="grad",
                reduce="sum",
            ),
            "read_target_invalid",
            recorded_shape=[2, 3],
        )
        assert "(2, 3)" in str(error)

    @_requires_gradient_edge
    def test_bare_site_without_index_or_cotangent(self) -> None:
        trace = _trace()
        _read_error(
            lambda: ob.read(trace, target=ob.seed("output_1"), method="grad", reduce="sum"),
            "read_target_invalid",
        )

    @_requires_gradient_edge
    def test_out_of_range_index_names_axis_and_shape(self) -> None:
        trace = _trace()
        _read_error(
            lambda: ob.read(
                trace,
                target=ob.seed("output_1", index=(0, 99)),
                method="grad",
                reduce="sum",
            ),
            "read_target_invalid",
            axis=1,
        )

    @_requires_gradient_edge
    def test_unknown_site_refuses(self) -> None:
        trace = _trace()
        _read_error(
            lambda: ob.read(trace, target=ob.seed("nonexistent_9_9"), method="grad", reduce="sum"),
            "read_target_invalid",
        )

    @_requires_gradient_edge
    def test_input_site_is_unaddressable(self) -> None:
        trace = _trace()
        _read_error(
            lambda: ob.read(
                trace,
                target=ob.seed("input_1", index=(0, 0)),
                method="grad",
                reduce="sum",
            ),
            "read_site_unaddressable",
            reason="no_grad_fn",
        )


class TestOptionVocabularies:
    """Closed vocabularies refuse with the whole menu."""

    def test_unknown_method(self) -> None:
        trace = _trace()
        _read_error(
            lambda: ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="magic"),
            "read_option_invalid",
            option="method",
        )

    def test_unknown_reduce(self) -> None:
        trace = _trace()
        _read_error(
            lambda: ob.read(
                trace,
                target=ob.seed("output_1", index=(0, 0)),
                method="grad",
                reduce="median",
            ),
            "read_option_invalid",
            option="reduce",
        )

    @_requires_gradient_edge
    def test_bad_batch_size(self) -> None:
        trace = _trace()
        _read_error(
            lambda: ob.read(
                trace,
                target=ob.seed("output_1", index=(0, 0)),
                method="grad",
                reduce="sum",
                target_batch_size=0,
            ),
            "read_option_invalid",
            option="target_batch_size",
        )

    @_requires_gradient_edge
    def test_activation_method_rejects_target_and_frozen(self) -> None:
        trace = _trace()
        _read_error(
            lambda: ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="activation"),
            "read_option_invalid",
            option="target",
        )
        _read_error(
            lambda: ob.read(trace, method="activation", frozen="gelu_1_2"),
            "read_option_invalid",
            option="frozen",
        )

    @_requires_gradient_edge
    def test_missing_target_for_gradient_method(self) -> None:
        trace = _trace()
        _read_error(
            lambda: ob.read(trace, method="grad", reduce="sum"),
            "read_target_invalid",
        )


class TestPopulationContract:
    """D9: explicit enumeration is a contract; implicit is a filter."""

    @_requires_gradient_edge
    def test_explicit_unretained_site_preflight_refusal(self) -> None:
        trace = _trace(capture=tl.options.CaptureOptions(layers_to_save=[]))
        error = _read_error(
            lambda: ob.read(
                trace,
                target=ob.seed("output_1", index=(0, 0)),
                method="activation_x_grad",
                reduce="sum",
                within="gelu_1_2",
            ),
            "read_payload_unretained",
        )
        assert "save=" in str(error), "the remedy must carry a concrete recapture recipe"

    @_requires_gradient_edge
    def test_explicit_unknown_site_refuses(self) -> None:
        trace = _trace()
        # A site absent from the trace is normally refused by the selection
        # algebra before the read sees it, so the read-level guard is pinned
        # directly through a hand-built population:
        from torchlens.attribution.onebackward._read import (
            _explicit_preflight,
            _SitePayloads,
        )

        index = ob.read_edge_index(trace)
        payloads = _SitePayloads(trace)
        with pytest.raises(ob.ReadError) as excinfo:
            _explicit_preflight(index, payloads, {"phantom_1_1:1": None}, "grad")
        assert excinfo.value.fields["code"] == "read_population_invalid"

    @_requires_gradient_edge
    def test_param_population_refuses_kind(self) -> None:
        trace = _trace()
        with pytest.raises(Exception) as excinfo:
            ob.read(
                trace,
                target=ob.seed("output_1", index=(0, 0)),
                method="grad",
                reduce="sum",
                within=tl.params("0.weight"),
            )
        assert getattr(excinfo.value, "fields", {}).get("code") == "selection_kind_incompatible"


class TestBudgetAndSaveModes:
    """Result-byte budget and save-mode honesty."""

    @_requires_gradient_edge
    def test_multi_target_element_grain_requires_budget(self) -> None:
        trace = _trace()
        targets = [ob.seed("output_1", index=(0, 0)), ob.seed("output_1", index=(0, 1))]
        error = _read_error(
            lambda: ob.read(trace, target=targets, method="grad", reduce=None),
            "read_result_budget_required",
        )
        assert error.fields["estimated_bytes"] > 0

    @_requires_gradient_edge
    def test_budget_exceeded_refuses_with_estimate(self) -> None:
        trace = _trace()
        targets = [ob.seed("output_1", index=(0, 0)), ob.seed("output_1", index=(0, 1))]
        _read_error(
            lambda: ob.read(
                trace, target=targets, method="grad", reduce=None, result_byte_budget=8
            ),
            "read_result_budget_required",
            budget=8,
        )

    @_requires_gradient_edge
    def test_view_save_mode_fails_closed_for_activation_methods(self) -> None:
        trace = _trace(save_mode="view")
        _read_error(
            lambda: ob.read(
                trace,
                target=ob.seed("output_1", index=(0, 0)),
                method="activation_x_grad",
                reduce="sum",
            ),
            "read_payload_untrustworthy",
            save_mode="view",
        )
        # method='grad' needs no payload and stays served on view captures.
        table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        assert table.status_counts()["ok"] >= 2


class TestTableRefusals:
    """Carrier-side typed refusals."""

    @_requires_gradient_edge
    def test_dense_table_not_portable(self, tmp_path) -> None:
        trace = _trace()
        table = ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce=None)
        _read_error(
            lambda: table.save(tmp_path / "dense.json"),
            "read_table_not_portable",
        )

    @_requires_gradient_edge
    def test_scalar_roundtrip_is_not_rescorable(self, tmp_path) -> None:
        trace = _trace()
        table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        path = table.save(tmp_path / "scalar.json")
        loaded = ob.load_read_table(path)
        assert len(loaded) == len(table)
        assert loaded.rescorable is False
        assert loaded.provenance.rescorable is False
        for key, row in table.items():
            assert loaded[key].score == row.score

    @_requires_gradient_edge
    def test_tampered_artifact_fails_closed(self, tmp_path) -> None:
        trace = _trace()
        table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        path = table.save(tmp_path / "scalar.json")
        text = path.read_text().replace('"ok"', '"blessed"')
        path.write_text(text)
        _read_error(
            lambda: ob.load_read_table(path),
            "read_table_artifact_invalid",
        )

    @_requires_gradient_edge
    def test_unknown_target_and_column(self) -> None:
        trace = _trace()
        table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        _read_error(lambda: table.for_target("t99"), "read_table_target_unknown")
        _read_error(lambda: table.column("nonexistent"), "read_table_column_unknown")
        _read_error(lambda: table.to_pandas(values="all"), "read_table_values_mode_invalid")

    @_requires_gradient_edge
    def test_table_is_immutable(self) -> None:
        trace = _trace()
        table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        with pytest.raises(AttributeError):
            table.provenance = None  # type: ignore[misc]


class TestTorchBandGate:
    """The capability gate refuses typed when GradientEdge is absent."""

    def test_refusal_when_flag_is_off(self, monkeypatch) -> None:
        from torchlens.attribution.onebackward import _accessor

        monkeypatch.setattr(_accessor, "get_gradient_edge_support", lambda: False)
        trace = _trace()
        _read_error(
            lambda: ob.read_edge_index(trace),
            "onebackward_torch_unsupported",
        )


class TestTypedDoorProvocations:
    """One direct assertion seam per new stable code (the provocation gate).

    The error-code coverage gate attributes a code to a test only when the
    literal flows into an assertion seam, so every code the helper-based
    tests exercise is ALSO provoked here with the plain two-line spelling.
    """

    def test_onebackward_torch_unsupported(self, monkeypatch) -> None:
        from torchlens.attribution.onebackward import _accessor

        monkeypatch.setattr(_accessor, "get_gradient_edge_support", lambda: False)
        trace = _trace()
        with pytest.raises(ob.ReadError) as excinfo:
            ob.read_edge_index(trace)
        assert excinfo.value.fields["code"] == "onebackward_torch_unsupported"

    def test_read_option_invalid(self) -> None:
        trace = _trace()
        with pytest.raises(ob.ReadError) as excinfo:
            ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="magic")
        assert excinfo.value.fields["code"] == "read_option_invalid"

    @_requires_gradient_edge
    def test_read_site_unaddressable(self) -> None:
        trace = _trace()
        with pytest.raises(ob.ReadError) as excinfo:
            ob.read(trace, target=ob.seed("input_1", index=(0, 0)), method="grad", reduce="sum")
        assert excinfo.value.fields["code"] == "read_site_unaddressable"

    @_requires_gradient_edge
    def test_read_payload_untrustworthy(self) -> None:
        trace = _trace(save_mode="view")
        with pytest.raises(ob.ReadError) as excinfo:
            ob.read(
                trace,
                target=ob.seed("output_1", index=(0, 0)),
                method="activation_x_grad",
                reduce="sum",
            )
        assert excinfo.value.fields["code"] == "read_payload_untrustworthy"

    @_requires_gradient_edge
    def test_read_frozen_selection_invalid(self) -> None:
        trace = _trace()
        with pytest.raises(ob.ReadError) as excinfo:
            ob.read(
                trace,
                target=ob.seed("output_1", index=(0, 0)),
                method="grad",
                reduce="sum",
                frozen=tl.params("0.weight"),
            )
        assert excinfo.value.fields["code"] == "read_frozen_selection_invalid"

    @_requires_gradient_edge
    def test_read_row_vocabulary_invalid(self) -> None:
        table = _grad_table(_trace())
        row = next(iter(table.rows()))
        from dataclasses import replace

        with pytest.raises(ob.ReadError) as excinfo:
            replace(row, status="blessed")
        assert excinfo.value.fields["code"] == "read_row_vocabulary_invalid"

    @_requires_gradient_edge
    def test_read_table_key_ambiguous(self) -> None:
        trace = _trace()
        table = ob.read(
            trace,
            target=[ob.seed("output_1", index=(0, 0)), ob.seed("output_1", index=(0, 1))],
            method="grad",
            reduce="sum",
        )
        with pytest.raises(ob.ReadError) as excinfo:
            table["linear_1_1:1"]
        assert excinfo.value.fields["code"] == "read_table_key_ambiguous"

    @_requires_gradient_edge
    def test_read_table_target_unknown(self) -> None:
        table = _grad_table(_trace())
        with pytest.raises(ob.ReadError) as excinfo:
            table.for_target("t99")
        assert excinfo.value.fields["code"] == "read_table_target_unknown"

    @_requires_gradient_edge
    def test_read_table_column_unknown(self) -> None:
        table = _grad_table(_trace())
        with pytest.raises(ob.ReadError) as excinfo:
            table.column("nonexistent")
        assert excinfo.value.fields["code"] == "read_table_column_unknown"

    @_requires_gradient_edge
    def test_read_table_values_mode_invalid(self) -> None:
        table = _grad_table(_trace())
        with pytest.raises(ob.ReadError) as excinfo:
            table.to_pandas(values="all")
        assert excinfo.value.fields["code"] == "read_table_values_mode_invalid"

    @_requires_gradient_edge
    def test_read_table_grain_unaggregatable(self) -> None:
        trace = _trace()
        table = ob.read(
            trace,
            target=[ob.seed("output_1", index=(0, 0)), ob.seed("output_1", index=(0, 1))],
            method="grad",
            reduce=None,
            result_byte_budget=1 << 20,
        )
        with pytest.raises(ob.ReadError) as excinfo:
            table.aggregate_targets(lambda scores: sum(scores))
        assert excinfo.value.fields["code"] == "read_table_grain_unaggregatable"

    @_requires_gradient_edge
    def test_read_table_not_portable(self, tmp_path) -> None:
        trace = _trace()
        table = ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce=None)
        with pytest.raises(ob.ReadError) as excinfo:
            table.save(tmp_path / "dense.json")
        assert excinfo.value.fields["code"] == "read_table_not_portable"

    @_requires_gradient_edge
    def test_read_table_artifact_invalid(self, tmp_path) -> None:
        table = _grad_table(_trace())
        path = table.save(tmp_path / "scalar.json")
        path.write_text(path.read_text().replace('"ok"', '"blessed"'))
        with pytest.raises(ob.ReadError) as excinfo:
            ob.load_read_table(path)
        assert excinfo.value.fields["code"] == "read_table_artifact_invalid"

    @_requires_gradient_edge
    def test_read_alias_conflict(self) -> None:
        from dataclasses import replace

        table = _grad_table(_trace())
        rows = {}
        corrupted_one = False
        for row in table.values():
            if row.alias_group is not None and row.score is not None and not corrupted_one:
                # Corrupt the FIRST alias member's score: the second member of
                # the same group then disagrees and the collapse must refuse.
                row = replace(row, score=row.score + 1.0)
                corrupted_one = True
            rows[row.key] = row
        assert corrupted_one, "the toy trace lost its alias group"
        corrupted = ob.ReadTable(rows, table.provenance, trace=table.trace)
        with pytest.raises(ob.ReadError) as excinfo:
            corrupted.collapse_alias_groups()
        assert excinfo.value.fields["code"] == "read_alias_conflict"

    def test_read_suppression_leak(self) -> None:
        trace = _trace()
        refs = trace.__dict__["_backward_gradfn_refs"]
        with pytest.raises(ob.ReadInternalError) as excinfo, ob.read_suppressed(trace):
            refs.append(object())
        assert excinfo.value.fields["code"] == "read_suppression_leak"
        refs.pop()

    @_requires_gradient_edge
    def test_read_engine_call_drift(self, monkeypatch) -> None:
        from torchlens.attribution.onebackward import _engine

        monkeypatch.setattr(_engine.math, "ceil", lambda value: 2)
        trace = _trace()
        with pytest.raises(ob.ReadInternalError) as excinfo:
            ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum")
        assert excinfo.value.fields["code"] == "read_engine_call_drift"

    def test_batched_attribution_unsupported(self) -> None:
        from torchlens.attribution.onebackward._engine import _translate_engine_error

        translated = _translate_engine_error(
            RuntimeError("Batching rule not implemented for aten::special_op"),
            batched=True,
            chunk_size=8,
        )
        assert translated is not None
        assert translated.fields["code"] == "batched_attribution_unsupported"
        assert "target_batch_size=1" in translated.fields["remedy"]
