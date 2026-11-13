"""Wrong drawing keywords raise a typed error that names the working spelling.

``Trace.draw`` takes every drawing setting as an explicit keyword; a 1.x ``vis_*``
keyword or a typo names its replacement instead of CPython's bare TypeError, and
``tl.trace`` sends drawing keywords and ``input_size`` to their real homes instead of
calling them model inputs.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._draw_keywords import LEGACY_DRAW_KEYWORDS


@pytest.fixture(scope="module")
def small_trace() -> Iterator[tl.Trace]:
    """A two-op capture to call ``draw`` on (it never renders: every call refuses)."""

    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    yield trace
    del trace


def test_legacy_draw_keyword_names_its_replacement(small_trace: tl.Trace) -> None:
    with pytest.raises(TypeError) as exc:
        small_trace.draw(vis_outpath="graph")
    assert exc.value.fields["code"] == "draw_keyword_unknown"
    assert exc.value.fields["unknown_keywords"] == ["vis_outpath"]
    assert "trace.draw(path=...)" in str(exc.value)


def test_vis_save_only_notes_the_inverted_value(small_trace: tl.Trace) -> None:
    with pytest.raises(TypeError, match="open_viewer=False"):
        small_trace.draw(vis_save_only=True)


def test_draw_typo_names_the_closest_keyword(small_trace: tl.Trace) -> None:
    with pytest.raises(TypeError) as exc:
        small_trace.draw(show_buffer="all")
    assert exc.value.fields["code"] == "draw_keyword_unknown"
    assert "trace.draw(show_buffers=...)" in str(exc.value)


def test_draw_signature_is_unchanged_and_every_replacement_exists() -> None:
    params = inspect.signature(tl.Trace.draw).parameters
    assert "path" in params and "show_buffers" in params
    assert set(LEGACY_DRAW_KEYWORDS.values()) <= set(params)


@pytest.mark.parametrize(
    ("keyword", "expected"),
    [
        ("vis_opt", "tl.trace(model, x).draw()"),
        ("vis_nesting_depth", "tl.trace(model, x).draw(depth=...)"),
        ("input_size", "input_shape="),
    ],
)
def test_trace_routes_drawing_keywords_and_input_size(keyword: str, expected: str) -> None:
    with pytest.raises(TypeError) as exc:
        tl.trace(nn.Linear(4, 4), torch.randn(1, 4), **{keyword: 1})
    assert exc.value.fields["code"] == "trace_forward_kwargs_unrouted"
    message = str(exc.value)
    assert expected in message
    assert "input_kwargs" not in message
