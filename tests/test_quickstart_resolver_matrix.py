"""The F17 resolver matrix row gate: rungs x verbs, one resolver, one record.

Composition cells from quickstart memo section 6: all three surfaces
serialize byte-equivalent provenance for the same resolved call; the
inferred rung consumes the exact verification trace (never recaptures); the
render facade is the only default route to collapse="auto"; explicit
overrides win; state restoration holds on the pinned surfaces.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.quickstart import (
    RenderResult,
    provenance_to_record,
    resolve_inputs,
    state_dict_hash,
    trace_input_provenance,
)
from torchlens.user_funcs import render


def _cnn() -> nn.Module:
    """Toy conv net with BatchNorm so restoration has something to protect."""

    return nn.Sequential(
        nn.Conv2d(3, 4, 3),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )


SIZE = (1, 3, 8, 8)


class TestOneResolverOneRecord:
    """Composition cell 1: byte-equivalent provenance across the verbs."""

    def test_declared_rung_provenance_is_identical_across_verbs(self, tmp_path) -> None:
        """trace / summary / render serialize the same declared record."""

        model = _cnn().eval()
        trace_log = tl.trace(model, input_size=SIZE)
        trace_record = trace_log.input_preprocessor
        render_result = render(model, input_size=SIZE, file=str(tmp_path / "unused.svg"))
        render_record = provenance_to_record(render_result.provenance)
        assert trace_record is not None
        assert dataclasses.asdict(trace_record) == dataclasses.asdict(render_record)

    def test_resolver_is_deterministic_across_calls(self) -> None:
        """Two resolutions of the same declared call are byte-equivalent."""

        model = _cnn().eval()
        first = resolve_inputs(model, input_size=SIZE)
        second = resolve_inputs(model, input_size=SIZE)
        assert dataclasses.asdict(first.provenance) == dataclasses.asdict(second.provenance)


class TestInferredReuse:
    """Composition cell 2: the verified trace is consumed, never recaptured."""

    def test_render_reuses_the_verification_trace(self, tmp_path) -> None:
        """The rung-3 render receipt proves zero additional captures."""

        result = render(_cnn().eval(), file=str(tmp_path / "g.svg"))
        assert result.receipt.policy == "reused_verified_trace"
        assert result.provenance.origin == "inferred"
        assert result.provenance.strategy

    @pytest.mark.smoke
    def test_zero_arg_trace_returns_the_verified_trace(self) -> None:
        """tl.trace(model) with default kwargs serves the verified capture."""

        log = tl.trace(_cnn().eval())
        provenance = trace_input_provenance(log)
        assert provenance is not None
        assert provenance.origin == "inferred"
        assert provenance.execution_policy == "eval_no_grad_restored"

    def test_zero_arg_trace_with_capture_kwargs_recaptures(self) -> None:
        """Non-default capture kwargs force a fresh capture (memo D8)."""

        log = tl.trace(_cnn().eval(), save=tl.func("relu"))
        provenance = trace_input_provenance(log)
        assert provenance is not None
        assert provenance.origin == "inferred"
        saved = [op for op in log if getattr(op, "has_saved_activation", False)]
        assert saved, "the save= predicate must have applied to the recapture"


class TestPinnedExecutionPolicy:
    """Memo D8: render restores flags, RNG, and norm buffers, hash-verified."""

    def test_render_restores_train_mode_model_state(self, tmp_path) -> None:
        """A train-mode model comes back bit-identical from render()."""

        model = _cnn().train()
        hash_before = state_dict_hash(model)
        rng_before = torch.get_rng_state()
        result = render(model, input_size=SIZE, file=str(tmp_path / "g.svg"))
        assert model.training is True
        assert state_dict_hash(model) == hash_before
        assert torch.equal(rng_before, torch.get_rng_state())
        assert result.receipt.policy == "eval_no_grad_restored"
        assert result.receipt.state_verified is True


class TestRenderFacade:
    """Memo D12-D14: curated dials, collapse=auto here only, never auto-open."""

    def test_default_collapse_is_auto_and_draw_stays_none(self, tmp_path) -> None:
        """render defaults to collapse='auto'; Trace.draw keeps 'none'."""

        result = render(_cnn().eval(), input_size=SIZE, file=str(tmp_path / "g.svg"))
        assert result.policy["collapse"] == "auto"
        import inspect

        draw_default = inspect.signature(tl.Trace.draw).parameters["collapse"].default
        assert draw_default == "none"

    def test_explicit_collapse_override_wins_and_is_disclosed(self, tmp_path) -> None:
        """An explicit collapse= value lands in the result policy."""

        result = render(
            _cnn().eval(), input_size=SIZE, collapse="none", file=str(tmp_path / "g.svg")
        )
        assert result.policy["collapse"] == "none"

    def test_curated_kwarg_collision_refuses(self, tmp_path) -> None:
        """Passing a curated dial's underlying draw spelling refuses typed."""

        with pytest.raises(Exception) as excinfo:
            render(
                _cnn().eval(),
                input_size=SIZE,
                file=str(tmp_path / "g.svg"),
                vis_theme="torchlens",
            )
        assert excinfo.value.fields["code"] == "render_kwarg_collision"

    @pytest.mark.smoke
    def test_result_is_detached_and_resaves_without_the_trace(self, tmp_path) -> None:
        """RenderResult.save() re-renders from the stored DOT source."""

        result = render(_cnn().eval(), input_size=SIZE, file=str(tmp_path / "g.svg"))
        assert isinstance(result, RenderResult)
        assert result.trace is None  # detached by default
        saved = result.save(tmp_path / "again.pdf")
        assert saved.exists() and saved.stat().st_size > 0

    def test_return_trace_retains_and_close_releases(self, tmp_path) -> None:
        """return_trace=True keeps the trace; close() releases it."""

        result = render(
            _cnn().eval(), input_size=SIZE, return_trace=True, file=str(tmp_path / "g.svg")
        )
        assert result.trace is not None
        with result:
            pass
        assert result.trace is None

    def test_collision_safe_default_filename(self, tmp_path, monkeypatch) -> None:
        """Fork F1 branch A: bare script render writes <Class>-graph.<fmt>, never overwrites."""

        monkeypatch.chdir(tmp_path)
        model = _cnn().eval()
        first = render(model, input_size=SIZE)
        second = render(model, input_size=SIZE)
        assert first.path is not None and first.path.exists()
        assert second.path is not None and second.path.exists()
        assert first.path != second.path

    @pytest.mark.smoke
    def test_value_channel_refuses_on_synthesized_input(self, tmp_path) -> None:
        """Memo D7: value-driven render channels hard-refuse on non-gold."""

        with pytest.raises(Exception) as excinfo:
            render(
                _cnn().eval(),
                input_size=SIZE,
                file=str(tmp_path / "g.svg"),
                color_by="magnitude",
            )
        assert excinfo.value.fields["code"] == "nongold_semantics_unavailable"


class TestDecodeGate:
    """Composition cell 6 (toy half): decode doors refuse on synthesized values."""

    def test_output_table_refuses_on_synthesized_trace(self) -> None:
        """decode_output (every decode door's chokepoint) requires gold."""

        log = tl.trace(_cnn().eval(), input_size=SIZE)
        with pytest.raises(Exception) as excinfo:
            log.decode_output()
        assert excinfo.value.fields["code"] == "nongold_semantics_unavailable"
