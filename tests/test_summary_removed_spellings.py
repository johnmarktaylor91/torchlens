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

    @pytest.mark.parametrize("name", sorted(_REMOVED_KWARGS))
    def test_removed_kwarg_passed_as_none_refuses_up_front(self, toy_trace, name: str) -> None:
        """<old>=None refuses on both doors, and before the forward on tl.summary."""

        successor = _REMOVED_KWARGS[name][1]
        with pytest.raises(InvalidArgumentError) as excinfo:
            toy_trace.summary(**{name: None})
        assert successor in str(excinfo.value)

        model = _Toy()
        calls: list[int] = []
        model.register_forward_hook(lambda *_: calls.append(1))
        with pytest.raises(InvalidArgumentError) as excinfo:
            tl.summary(model, torch.randn(1, 4), **{name: None})
        assert excinfo.value.fields["code"] == "summary_option_invalid"
        assert successor in str(excinfo.value)
        assert calls == [], "a None-valued removed spelling must refuse before the capture"

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


def test_removed_spelling_tables_match_the_source() -> None:
    """This file's spelling tables cover exactly the source's removed tables."""

    from torchlens.report._summary_config import REMOVED_SUMMARY_LEVELS, REMOVED_SUMMARY_OPTIONS

    assert set(_REMOVED_KWARGS) == set(REMOVED_SUMMARY_OPTIONS)
    assert set(_REMOVED_LEVELS) == set(REMOVED_SUMMARY_LEVELS)


def test_grammar_options_match_what_resolve_config_consumes() -> None:
    """GRAMMAR_OPTIONS is exactly the option set resolve_config reads.

    _validate_summary_grammar drops None only for these names, so drift
    would make tl.summary(model, x, <new option>=None) refuse falsely.
    """

    import inspect
    import re

    from torchlens.report import _summary_config

    source = inspect.getsource(_summary_config.resolve_config)
    popped = set(re.findall(r'kwargs\.pop\("(\w+)"', source))
    closed = {axis for axis, _default, _valid in _summary_config._CLOSED_VOCABULARY_AXES}
    assert "kwargs.pop(axis" in source  # the closed axes are popped by name
    assert set(_summary_config.GRAMMAR_OPTIONS) == popped | closed
