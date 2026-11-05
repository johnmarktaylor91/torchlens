"""Comparison reads see a module edit at the edited module's op.

A ``tl.module(addr)`` edit, post-capture (``fork.do``) or at capture time
(``intervene=tl.when``), records the edited value in its own
``interventionreplacement`` op after the module's last op; the last op keeps
its computed value and the unedited trace has no replacement op. Each read
below pairs ops across the two traces and must report the edit at the edited
module's last op (``first_divergence``, ``Bundle.compare``, ``rank_changes``,
the bundle-diff site deltas), not skip it and name a downstream op instead.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bundle._aligned_sites import site_row_key
from torchlens.bundle._deltas import site_deltas
from torchlens.debug._graph import _compare_rows

_REPLACEMENT = "intervention_replacement"


class _Chain(nn.Module):
    """``fc1 -> relu -> fc2 -> relu -> head``; ``fc2`` is the edited module."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 6)
        self.head = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def _replacements(trace: tl.Trace) -> list:
    return [op for op in trace._layer_list if op.func_name == _REPLACEMENT]


def _edited_op_label(clean: tl.Trace) -> str:
    """The fc2 linear op's layer label in the unedited trace."""

    return str(clean.modules["fc2"].calls[0].output_ops[0]).split(":")[0]


@pytest.fixture(params=["post_capture", "capture_time"])
def clean_and_edited(request) -> Iterator[tuple[tl.Trace, tl.Trace]]:
    # The model stays alive through the test: the Bundle gate reads the live reference.
    torch.manual_seed(3)
    model = _Chain().eval()
    x = torch.randn(2, 4)
    clean = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    if request.param == "post_capture":
        edited = clean.fork("edited").do(tl.module("fc2"), tl.zero_ablate())
    else:
        edited = tl.trace(model, x, intervene=tl.when(tl.module("fc2"), tl.zero_ablate()))
    yield clean, edited


def test_the_edit_is_recorded_as_a_replacement_op(clean_and_edited) -> None:
    clean, edited = clean_and_edited
    (replacement,) = _replacements(edited)
    assert torch.count_nonzero(replacement.out) == 0
    assert _replacements(clean) == []


def test_a_post_capture_edit_keeps_every_capture_label() -> None:
    torch.manual_seed(3)
    model = _Chain().eval()
    clean = tl.trace(model, torch.randn(2, 4), capture=tl.options.CaptureOptions(replayable=True))
    edited = clean.fork("edited").do(tl.module("fc2"), tl.zero_ablate())
    (replacement,) = _replacements(edited)
    # Later ops are not renumbered: every label-keyed read pairs them as before.
    assert [op.label for op in clean._layer_list] == [
        op.label for op in edited._layer_list if op is not replacement
    ]


def test_first_divergence_names_the_edited_module_op(clean_and_edited) -> None:
    clean, edited = clean_and_edited
    row = tl.debug.first_divergence(clean, edited)
    assert row is not None
    assert row["status"] == "present-in-both"
    assert row["allclose"] is False
    assert str(row["op"]).split(":")[0] == _edited_op_label(clean)
    # The replacement is read at the site it replaced, so it has no row of its own.
    rows, _attrs = _compare_rows(clean, edited)
    (replacement,) = _replacements(edited)
    assert replacement.label not in {row["op"] for row in rows}


def test_bundle_compare_scores_the_edited_module_op(clean_and_edited) -> None:
    clean, edited = clean_and_edited
    bundle = tl.Bundle({"clean": clean, "edited": edited}, baseline="clean")
    rows = dict(bundle.compare("clean", "edited", metric="relative_l2"))
    edited_label = _edited_op_label(clean)
    assert rows[edited_label] == pytest.approx(1.0)
    assert set(rows) == {site_row_key(op) for op in clean._layer_list}
    (replacement,) = _replacements(edited)
    # The replacement keys to the row of the op it replaced.
    assert site_row_key(replacement) == edited_label


def test_rank_changes_scores_the_edited_module_op(clean_and_edited) -> None:
    clean, edited = clean_and_edited
    bundle = tl.Bundle({"clean": clean, "edited": edited}, baseline="clean")
    ranked = dict(bundle.rank_changes(k=50, metric="relative_l2"))
    assert ranked[_edited_op_label(clean)] == pytest.approx(1.0)
    assert not any(key.startswith("interventionreplacement") for key in ranked)


def test_site_deltas_match_the_compare_rows_at_the_edit(clean_and_edited) -> None:
    clean, edited = clean_and_edited
    bundle = tl.Bundle({"clean": clean, "edited": edited}, baseline="clean")
    rows = dict(bundle.compare("clean", "edited", metric="relative_l2"))
    deltas = site_deltas(bundle, "relative_l2", "clean", "out", ["clean", "edited"])
    (replacement,) = _replacements(edited)
    edited_label = _edited_op_label(clean)
    assert deltas.value(replacement, "edited") == pytest.approx(rows[edited_label])
    assert deltas.value(replacement, "edited") == pytest.approx(1.0)
    assert deltas.value(replacement, "clean") == 0.0


def test_a_replacement_that_does_not_record_its_parent_keeps_its_own_row() -> None:
    torch.manual_seed(3)
    model = _Chain().eval()
    clean = tl.trace(model, torch.randn(2, 4), capture=tl.options.CaptureOptions(replayable=True))
    edited = clean.fork("edited").do(tl.module("fc2"), tl.zero_ablate())
    (replacement,) = _replacements(edited)
    replacement._internal_set("parents", ("no_such_op_1_99",))

    rows, _attrs = _compare_rows(clean, edited)
    by_label = {row["op"]: row for row in rows}
    assert by_label[replacement.label]["status"] == "only-b"
    assert site_row_key(replacement) == str(replacement.layer_label)
