"""The legacy summary spellings are gone: each refuses typed, naming its successor.

Clean break (no aliases): the eight legacy ``summary()`` keyword spellings and
the legacy ``level=`` preset names no longer route to a historical renderer.
Each one raises ``InvalidArgumentError`` whose message names the new-grammar
argument (or the public surface) that replaces it, on both doors: the
existing-capture door ``trace.summary()`` and the one-call door
``tl.summary(model, x)`` (which must refuse before running any capture).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError

#: removed keyword spelling -> (example value, token the refusal must name)
_REMOVED_KWARGS: dict[str, tuple[object, str]] = {
    "preset": ("overview", "view="),
    "fields": (["name"], "columns="),
    "show_ops": (True, "level='op'"),
    "include_ops": (True, "level='op'"),
    "mode": ("unrolled", "level='op'"),
    "print_to": (print, "report.print("),
    "count_fma_as_two": (False, "flop_convention="),
    "show_input_preprocessing_details": (True, "trace.provenance()"),
}

#: removed legacy level= preset -> token the refusal must name
_REMOVED_LEVELS: dict[str, str] = {
    "overview": "view='overview'",
    "compute": "view='compute'",
    "cost": "view='compute'",
    "graph": "trace.to_agent_json()",
    "memory": "trace.profile(sort_by='activation_memory')",
    "control_flow": "trace.conditional_records",
    "waterfall": "trace.profile(level='op')",
    "output": "trace.output_table()",
}


class _Toy(nn.Module):
    """One Linear."""

    def __init__(self) -> None:
        """One Linear."""

        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """fc."""

        return self.fc(x)


@pytest.fixture(scope="module")
def toy_trace():
    """One finished toy trace for the whole module."""

    trace = tl.trace(_Toy(), torch.randn(1, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.mark.smoke_cells(
    "test_removed_kwarg_refuses_typed_on_trace_summary[count_fma_as_two]",
    "test_removed_level_refuses_typed_on_trace_summary[graph]",
)
class TestRemovedSpellings:
    """Every removed spelling refuses typed and names its replacement."""

    @pytest.mark.parametrize("name", sorted(_REMOVED_KWARGS))
    def test_removed_kwarg_refuses_typed_on_trace_summary(self, toy_trace, name: str) -> None:
        """trace.summary(<old>=...) raises InvalidArgumentError naming the new spelling."""

        value, successor = _REMOVED_KWARGS[name]
        with pytest.raises(InvalidArgumentError) as excinfo:
            toy_trace.summary(**{name: value})
        assert excinfo.value.fields["code"] == "summary_option_invalid"
        assert f"{name}=" in str(excinfo.value)
        assert successor in str(excinfo.value)

    @pytest.mark.parametrize("level", sorted(_REMOVED_LEVELS))
    def test_removed_level_refuses_typed_on_trace_summary(self, toy_trace, level: str) -> None:
        """trace.summary(level=<legacy preset>) raises naming the replacement."""

        with pytest.raises(InvalidArgumentError) as excinfo:
            toy_trace.summary(level=level)
        assert excinfo.value.fields["code"] == "summary_level_invalid"
        assert _REMOVED_LEVELS[level] in str(excinfo.value)

    @pytest.mark.parametrize("name", sorted(_REMOVED_KWARGS))
    def test_removed_kwarg_refuses_before_capture_on_tl_summary(self, name: str) -> None:
        """tl.summary(model, x, <old>=...) refuses typed before any forward runs."""

        model = _Toy()
        calls: list[int] = []
        model.register_forward_hook(lambda *_: calls.append(1))
        value, successor = _REMOVED_KWARGS[name]
        with pytest.raises(InvalidArgumentError) as excinfo:
            tl.summary(model, torch.randn(1, 4), **{name: value})
        assert successor in str(excinfo.value)
        assert calls == [], "the refusal must fire before the capture runs"

    @pytest.mark.parametrize("level", sorted(_REMOVED_LEVELS))
    def test_removed_level_refuses_before_capture_on_tl_summary(self, level: str) -> None:
        """tl.summary(model, x, level=<legacy preset>) refuses typed before capture."""

        model = _Toy()
        calls: list[int] = []
        model.register_forward_hook(lambda *_: calls.append(1))
        with pytest.raises(InvalidArgumentError) as excinfo:
            tl.summary(model, torch.randn(1, 4), level=level)
        assert excinfo.value.fields["code"] == "summary_level_invalid"
        assert _REMOVED_LEVELS[level] in str(excinfo.value)
        assert calls == [], "the refusal must fire before the capture runs"
