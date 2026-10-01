"""A-RAGGED-DOCS: pass-vs-step teaching honesty + the episode pass-axis disclosure.

Pins the foldB D11 docs-honesty fix (OPUS r3 R3-7): the shipped
``pass_variance`` example taught "variance explodes late in the sequence" as
``passes=[8, 9, 10]``, which is FALSE on any model whose layer is not called
at every step — pass indices are per-site OCCURRENCE counters (pass k of a
layer = the k-th time THAT layer ran), never episode steps. The RED fixture
here makes the false reading executable: a stepped model whose ``extra``
linear runs at only half the episode steps (occurrence 2 lands at episode
step 3, not step 2). GREEN is the corrected teaching in both cross-pass
producer docstrings, the MoE/grouping truth sentence in the episode-capture
doc, and the point-of-use disclosure: resolving a pass window on an episode
capture warns coded (``episode_pass_window_occurrence_axis``) with the
episode step count and each resolved layer's occurrence count. The typed
refusal for an explicit step axis is F-EPISODE's (``axis=``), not this
lane's.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.options import EpisodeSpec

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[1]

N_STEPS = 4


class RaggedLM(nn.Module):
    """Stepped model whose ``extra`` linear runs only on even-length inputs.

    Under a greedy loop from a 2-token prompt the input lengths are
    2, 3, 4, 5 — so ``extra`` runs at episode steps 0 and 2 only: its pass 2
    (second occurrence) lands at episode step 2, never step 1. This is the
    minimal executable form of the routed-MoE raggedness (an expert layer
    absent from some steps).
    """

    def __init__(self, vocab: int = 16, width: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.extra = nn.Linear(width, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.emb(ids).mean(1))
        if ids.shape[1] % 2 == 0:
            x = self.extra(x)
        return self.head(x)


class GreedyRunner(nn.Module):
    """Episode root: steps the inner model N times, greedy decode."""

    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            logits = self.model(current)
            next_token = logits.argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


def _extra_label(log: tl.Trace) -> str:
    """The layer label of the ragged ``extra`` linear (the 2-pass site)."""

    labels = {op.layer_label for op in log if op.func_name == "linear" and op.num_passes == 2}
    assert len(labels) == 1, f"expected exactly one 2-pass linear site, got {labels!r}"
    return labels.pop()


@pytest.fixture(scope="module")
def episode_log():
    """One ragged episode capture shared across the module's read tests."""

    torch.manual_seed(0)
    model = RaggedLM()
    runner = GreedyRunner(model, N_STEPS)
    trace = tl.trace(
        runner,
        torch.tensor([[1, 2]]),
        episode=EpisodeSpec(stepped_module=model, n_steps=N_STEPS),
    )
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def plain_log():
    """The same loop captured WITHOUT an episode declaration (control)."""

    torch.manual_seed(0)
    model = RaggedLM()
    runner = GreedyRunner(model, N_STEPS)
    trace = tl.trace(runner, torch.tensor([[1, 2]]))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_ragged_fixture_pins_the_false_reading(episode_log: tl.Trace) -> None:
    """RED pin: the extra site ran 2x across a 4-step episode — pass k != step k.

    This is the fact that made the shipped example false: on any capture
    where a layer's occurrence count differs from the episode step count,
    reading ``passes=[k]`` as "step k" is wrong.
    """

    label = _extra_label(episode_log)
    rows = episode_log.annotations["episode"]["rows"]
    assert len(rows) == N_STEPS
    assert episode_log[label].num_passes == 2
    assert episode_log[label].num_passes < len(rows)


def test_false_example_killed_in_docstrings() -> None:
    """GREEN pin: both cross-pass producer docstrings teach occurrences, not steps."""

    for producer in (tl.pass_variance, tl.stable_across_passes):
        doc = producer.__doc__ or ""
        assert "OCCURRENCE" in doc, f"{producer.__name__} lost the occurrence teaching"
        assert "episode" in doc.lower(), f"{producer.__name__} must name the episode disclosure"
    variance_doc = tl.pass_variance.__doc__ or ""
    assert "late in the sequence" not in variance_doc, (
        "the false 'late in the sequence' reading is back in pass_variance"
    )
    stable_doc = tl.stable_across_passes.__doc__ or ""
    assert '"all timesteps"' not in stable_doc, (
        "the false 'all timesteps' gloss is back in stable_across_passes"
    )


def test_docs_state_the_moe_grouping_truth() -> None:
    """GREEN pin: the episode-capture doc carries the MoE/grouping sentence."""

    doc = (_REPO_ROOT / "docs" / "reference" / "episode_capture.md").read_text(encoding="utf-8")
    assert "never on pass-interior graph equality" in doc
    assert "OCCURRENCE counters" in doc


def test_disclosure_fires_with_both_counts(episode_log: tl.Trace) -> None:
    """Resolving a pass window on an episode capture discloses k and n, coded."""

    label = _extra_label(episode_log)
    with pytest.warns(TorchLensWarning) as records:
        resolved = tl.pass_variance(above=-1.0, within=label).resolve(episode_log)
    coded = [
        r.message
        for r in records
        if getattr(r.message, "fields", {}).get("code") == "episode_pass_window_occurrence_axis"
    ]
    assert len(coded) == 1
    warning = coded[0]
    assert warning.fields["producer"] == "pass_variance"
    assert warning.fields["episode_steps"] == N_STEPS
    assert warning.fields["site_passes"] == {label: 2}
    assert warning.affected_sites == [label]
    # Compo memo D3: a disclosure attaches to a USABLE result — the resolve
    # still lands one entry per window pass of the site.
    entries = list(resolved)
    assert [entry.site_key for entry in entries] == [(label, 1), (label, 2)]


def test_disclosure_covers_stable_across_passes(episode_log: tl.Trace) -> None:
    """The sibling cross-pass producer makes the same claim and discloses too."""

    label = _extra_label(episode_log)
    with pytest.warns(TorchLensWarning) as records:
        tl.stable_across_passes(within=label, tol=1e9).resolve(episode_log)
    coded = [
        r.message
        for r in records
        if getattr(r.message, "fields", {}).get("code") == "episode_pass_window_occurrence_axis"
    ]
    assert len(coded) == 1
    assert coded[0].fields["producer"] == "stable_across_passes"


def test_no_disclosure_on_plain_captures(plain_log: tl.Trace) -> None:
    """A non-episode capture has no step axis to confuse: the resolve is silent."""

    label = _extra_label(plain_log)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.pass_variance(above=-1.0, within=label).resolve(plain_log)
    torchlens_warnings = [w for w in caught if isinstance(w.message, TorchLensWarning)]
    assert torchlens_warnings == []
