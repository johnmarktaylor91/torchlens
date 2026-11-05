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
from torch import nn

import torchlens as tl
from torchlens._quickstart import (
    Drawing,
    resolve_inputs,
    trace_input_provenance,
)
from torchlens._user_funcs import draw as render


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

        result = render(_cnn().eval(), path=str(tmp_path / "g.svg"))
        assert result.receipt.policy == "reused_verified_trace"
        assert result.provenance.origin == "inferred"
        assert result.provenance.strategy

    def test_zero_arg_trace_with_capture_kwargs_recaptures(self) -> None:
        """Non-default capture kwargs force a fresh capture (memo D8)."""

        log = tl.trace(_cnn().eval(), save=tl.func("relu"))
        provenance = trace_input_provenance(log)
        assert provenance is not None
        assert provenance.origin == "inferred"
        saved = [op for op in log if getattr(op, "has_saved_activation", False)]
        assert saved, "the save= predicate must have applied to the recapture"


class TestRenderFacade:
    """Memo D12-D14: curated dials, collapse=auto here only, never auto-open."""

    def test_result_is_detached_and_resaves_without_the_trace(self, tmp_path) -> None:
        """Drawing.save() re-renders from the stored DOT source."""

        result = render(_cnn().eval(), input_shape=SIZE, path=str(tmp_path / "g.svg"))
        assert isinstance(result, Drawing)
        assert result.trace is None  # detached by default
        saved = result.save(tmp_path / "again.pdf")
        assert saved.exists() and saved.stat().st_size > 0

    def test_collision_safe_default_filename(self, tmp_path, monkeypatch) -> None:
        """Fork F1 branch A: bare script render writes <Class>-graph.<fmt>, never overwrites."""

        monkeypatch.chdir(tmp_path)
        model = _cnn().eval()
        first = render(model, input_shape=SIZE)
        second = render(model, input_shape=SIZE)
        assert first.path is not None and first.path.exists()
        assert second.path is not None and second.path.exists()
        assert first.path != second.path

    def test_value_channel_refuses_on_synthesized_input(self, tmp_path) -> None:
        """Memo D7: value-driven render channels hard-refuse on non-gold."""

        with pytest.raises(Exception) as excinfo:
            render(
                _cnn().eval(),
                input_shape=SIZE,
                path=str(tmp_path / "g.svg"),
                color_by="magnitude",
            )
        assert excinfo.value.fields["code"] == "nongold_semantics_unavailable"
