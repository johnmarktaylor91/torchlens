"""Quickstart input-ladder unit tests (F17: rung law, grammar, synthesis, record).

Toy-model tier; the real-model rows live in ``test_quickstart_realmodel.py``.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.quickstart import (
    InputSpec,
    SynthesizedValueReadWarning,
    internal_read,
    is_gold,
    parse_input_size,
    provenance_from_record,
    provenance_to_record,
    require_gold,
    resolve_inputs,
    resolve_rung,
    trace_input_provenance,
)
from torchlens.user_funcs import render as _render

pytestmark = pytest.mark.smoke


def _cnn() -> nn.Module:
    """Small conv classifier with an adaptive pool (flexible spatial dims)."""

    return nn.Sequential(
        nn.Conv2d(3, 4, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)
    ).eval()


def _lm() -> nn.Module:
    """Tiny embedding-entry model (ids-recipe evidence)."""

    return nn.Sequential(nn.Embedding(50, 8), nn.Flatten(1), nn.Linear(8 * 4, 2)).eval()


class TestRungLaw:
    """Memo D2: real XOR input_size XOR nothing, identical on all verbs."""

    def test_mixed_rung_refuses(self) -> None:
        """A real input plus input_size= is a typed conflict, never a guess."""

        with pytest.raises(Exception) as excinfo:
            resolve_rung(torch.randn(1, 3), None, (1, 3))
        assert excinfo.value.fields["code"] == "input_rung_conflict"

    def test_trace_and_render_pin_identical_conflict_wording(self) -> None:
        """The XOR refusal is byte-identical across verbs (one resolver)."""

        model = _cnn()
        x = torch.randn(1, 3, 8, 8)
        messages = []
        for verb in (
            lambda: tl.trace(model, x, input_size=(1, 3, 8, 8)),
            lambda: tl.summary(model, x, input_size=(1, 3, 8, 8)),
            lambda: _render(model, x, input_size=(1, 3, 8, 8)),
        ):
            with pytest.raises(Exception) as excinfo:
                verb()
            assert excinfo.value.fields["code"] == "input_rung_conflict"
            messages.append(str(excinfo.value))
        assert len(set(messages)) == 1

    def test_kwargs_only_call_is_gold(self) -> None:
        """A keyword-only real call is rung 1, not rung 3."""

        assert resolve_rung(None, {"input_ids": torch.ones(1, 2)}, None) == 1


class TestGrammar:
    """Memo D4: SOL's grammar verbatim, refusing before capture."""

    @pytest.mark.parametrize(
        "bad",
        [(0, 3), (1, -2), (1, 2.5), ("a", 3), (), [], "224", 224, {3: (1, 2)}],
    )
    def test_malformed_sizes_refuse(self, bad: object) -> None:
        """Zero/negative/symbolic/empty/non-shape spellings refuse typed."""

        with pytest.raises(Exception) as excinfo:
            parse_input_size(bad, _cnn())
        assert excinfo.value.fields["code"] == "input_size_invalid"

    def test_bool_dimension_is_symbolic(self) -> None:
        """bool is not a size (True == 1 must not synthesize)."""

        with pytest.raises(Exception) as excinfo:
            parse_input_size((1, True, 4), _cnn())
        assert excinfo.value.fields["code"] == "input_size_invalid"

    def test_unknown_mapping_binding_refuses(self) -> None:
        """A mapping key the forward does not accept refuses with the names."""

        class OneArg(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        with pytest.raises(Exception) as excinfo:
            parse_input_size({"bogus": (1, 3)}, OneArg())
        assert excinfo.value.fields["code"] == "input_size_unknown_binding"

    def test_sequence_form_makes_positional_slots(self) -> None:
        """A sequence of shapes is multiple positional tensors."""

        parsed = parse_input_size([(1, 4), (1, 4)], nn.Linear(4, 2))
        assert len(parsed.positional) == 2
        assert parsed.positional[0].shape == (1, 4)

    def test_dtype_facts_embedding_entry_permits_ids(self) -> None:
        """An unambiguous embedding entry permits vocab-bounded int64 ids."""

        parsed = parse_input_size((1, 4), _lm())
        slot = parsed.positional[0]
        assert slot.dtype == torch.int64
        assert slot.recipe == "randint"
        assert slot.high <= 50

    def test_dtype_ambiguity_refuses_with_override_teach(self) -> None:
        """No static entry fact refuses; InputSpec is the taught override."""

        class Opaque(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        with pytest.raises(Exception) as excinfo:
            parse_input_size((1, 3), Opaque())
        assert excinfo.value.fields["code"] == "input_dtype_ambiguous"
        assert "InputSpec" in str(excinfo.value)
        parsed = parse_input_size(InputSpec((1, 3), dtype=torch.float32), Opaque())
        assert parsed.positional[0].dtype == torch.float32


class TestSynthesis:
    """Memo D4: local seed-0 generator; deterministic; RNG-neutral."""

    def test_synthesis_is_deterministic_and_rng_neutral(self) -> None:
        """Repeated resolution is byte-identical and never moves global RNG."""

        model = _cnn()
        state_before = torch.get_rng_state()
        first = resolve_inputs(model, input_size=(1, 3, 8, 8))
        second = resolve_inputs(model, input_size=(1, 3, 8, 8))
        assert torch.equal(first.plan.input_args[0], second.plan.input_args[0])
        assert torch.equal(state_before, torch.get_rng_state())

    def test_failed_declared_size_never_falls_through_to_inference(self) -> None:
        """Memo D2: a failing explicit size surfaces the forward's own error."""

        model = _cnn()
        with pytest.raises(Exception) as excinfo:
            tl.trace(model, input_size=(1, 5, 8, 8))  # wrong channel count
        # The refusal must NOT be an inference teach: the declared rung owns
        # its failure (any capture-side error is acceptable; the inference
        # code path is not).
        assert "input_inference_failed" not in repr(getattr(excinfo.value, "fields", {}))


class TestProvenanceRecord:
    """Memo D5: one persistent record; synthesized rungs disclose."""

    def test_declared_record_round_trips_through_base_record(self) -> None:
        """The rich view survives the ResolvedPreprocessing codec."""

        resolved = resolve_inputs(_cnn(), input_size=(2, 3, 8, 8))
        record = provenance_to_record(resolved.provenance)
        rehydrated = provenance_from_record(record)
        assert rehydrated is not None
        assert rehydrated.origin == "declared"
        assert rehydrated.rung == 2
        assert not rehydrated.values_semantic
        assert rehydrated.tensors[0].shape == (2, 3, 8, 8)
        assert rehydrated.recipes is not None
        assert rehydrated.caveats

    def test_gold_trace_has_no_quickstart_record_and_reads_gold(self) -> None:
        """Absent record = caller-authoritative history (shipped contract)."""

        log = tl.trace(_cnn(), torch.randn(1, 3, 8, 8))
        assert trace_input_provenance(log) is None
        assert is_gold(log)

    def test_declared_trace_carries_record(self) -> None:
        """input_size= traces persist the disclosure on the KEEP field."""

        log = tl.trace(_cnn(), input_size=(1, 3, 8, 8))
        provenance = trace_input_provenance(log)
        assert provenance is not None
        assert provenance.origin == "declared"
        assert not is_gold(log)


class TestCapabilityGate:
    """Memo D7: derived semantics refuse; first raw read warns once."""

    def test_require_gold_refuses_on_synthesized(self) -> None:
        """Derived-semantics claims hard-refuse with the real-input teach."""

        log = tl.trace(_cnn(), input_size=(1, 3, 8, 8))
        with pytest.raises(Exception) as excinfo:
            require_gold(log, "decoded labels")
        assert excinfo.value.fields["code"] == "nongold_semantics_unavailable"
        require_gold(tl.trace(_cnn(), torch.randn(1, 3, 8, 8)), "decoded labels")

    def test_first_raw_read_warns_once_per_trace(self) -> None:
        """Layer.out warns exactly once on a synthesized trace, never on gold."""

        synthesized = tl.trace(_cnn(), input_size=(1, 3, 8, 8))
        gold = tl.trace(_cnn(), torch.randn(1, 3, 8, 8))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = synthesized["conv2d_1_1"].out
            _ = synthesized["conv2d_1_1"].out
            _ = gold["conv2d_1_1"].out
        codes = [
            w.message.fields.get("code")
            for w in caught
            if isinstance(w.message, SynthesizedValueReadWarning)
        ]
        assert codes == ["nongold_raw_value_read"]

    def test_internal_readers_pass_the_ack(self) -> None:
        """Reads inside internal_read() never trip the user warning."""

        synthesized = tl.trace(_cnn(), input_size=(1, 3, 8, 8))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with internal_read():
                _ = synthesized["conv2d_1_1"].out
        assert not [w for w in caught if isinstance(w.message, SynthesizedValueReadWarning)]


class TestTypedRefusalProvocation:
    """Every F17 refusal code fires for real (error-code coverage gate)."""

    def test_nontorch_backend_with_input_size_refuses(self) -> None:
        """Synthesis/inference is torch-only; other backends need a real input."""

        with pytest.raises(Exception) as excinfo:
            tl.trace(_cnn(), input_size=(1, 3, 8, 8), backend="tf")
        assert excinfo.value.fields["code"] == "input_ladder_backend_unsupported"

    def test_render_pipeline_failure_refuses_typed(self) -> None:
        """A graphviz pipeline failure surfaces as one typed refusal."""

        from torchlens.quickstart._render import _render_dot_source

        with pytest.raises(Exception) as excinfo:
            _render_dot_source("digraph { a -> b }", "not_a_real_format")
        assert excinfo.value.fields["code"] == "render_engine_unavailable"

    def test_render_missing_graphviz_refuses_typed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """An absent graphviz package is the same typed refusal, with remedy."""

        import sys as _sys

        from torchlens.quickstart._render import _render_dot_source

        monkeypatch.setitem(_sys.modules, "graphviz", None)  # import raises ImportError
        with pytest.raises(Exception) as excinfo:
            _render_dot_source("digraph { a -> b }", "pdf")
        assert excinfo.value.fields["code"] == "render_engine_unavailable"
        assert "pip install graphviz" in excinfo.value.fields["remedy"]

    def test_open_viewer_windows_branch_uses_startfile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """The explicit view=True door routes through os.startfile on Windows."""

        import os as _os
        import sys as _sys

        from torchlens.quickstart._render import _open_viewer

        opened: list[str] = []
        monkeypatch.setattr(_sys, "platform", "win32")
        monkeypatch.setattr(_os, "name", "nt")
        monkeypatch.setattr(_os, "startfile", opened.append, raising=False)
        target = tmp_path / "graph.pdf"
        target.write_bytes(b"%PDF-")
        _open_viewer(target)
        assert opened == [str(target)]
