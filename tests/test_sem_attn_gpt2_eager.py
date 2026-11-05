"""A01 semantic-attention slice: GPT-2 EAGER per-head facets (real ops).

Lane A01 acceptance (megaplan row A01 / walkthrough A-I item 3): per-head
facets (scores/pattern/z/result) EXIST for GPT-2, transformers 4.x AND 5.x,
eager implementation -- anchored at REAL captured ops (graph-derived, never
class-name-keyed), with the per-head result a validated computed view.

Every assertion is version-adaptive: no transformers class names, no version
parsing -- the same file must pass under a 4.x and a 5.x install.
"""

from __future__ import annotations

import pytest

import torchlens as tl

pytest.importorskip("transformers")

from tests.real_model.r0.families import FAMILY_BY_NAME  # noqa: E402

ATTN_ADDRESS = "transformer.h.0.attn"


@pytest.fixture(scope="module")
def eager_capture():
    """One eager config-built GPT-2 capture shared by this module's tests."""

    spec = FAMILY_BY_NAME["gpt2"]
    model = spec.build("eager")
    kwargs = spec.input_kwargs()
    trace = tl.trace(model, (), input_kwargs=kwargs)
    yield model, trace
    tl.release_model(model)


def _view(trace):
    return trace.modules[ATTN_ADDRESS].facets


def test_unsaved_anchor_reports_needs_capture():
    """A selective capture reports unsaved anchors as actionable needs_capture."""

    spec = FAMILY_BY_NAME["gpt2"]
    model = spec.build("eager")
    trace = tl.trace(model, (), input_kwargs=spec.input_kwargs(), save=tl.func("softmax"))
    try:
        view = trace.modules[ATTN_ADDRESS].facets
        menu = view.declared()
        assert menu["pattern"].status == "available_now"
        assert menu["z"].status == "needs_capture"
        assert menu["z"].save_hint
        assert menu["scores"].status == "needs_capture"
    finally:
        tl.release_model(model)
