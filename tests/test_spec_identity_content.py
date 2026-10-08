"""Spec rule identity follows the full, current content of tensor arguments (F4).

Rule ids, ``action_repr`` and ``spec_digest`` used to render tensor arguments
through ``repr``, which torch truncates above 1000 elements and which was
computed once at construction. Two model-width steering vectors differing in
one element therefore shared one rule id (a false ``spec_rules_duplicate`` on
merge), and an in-place edit of a staged steer vector changed results while
every record kept the construction-time identity.

The oracle in every test is a FRESH spec built over the expected content, never
"different from before".
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention.spec import InterventionSpec

_WIDTH = 4000  # above torch's 1000-element print threshold


class _ReluModel(nn.Module):
    """``y = 2 * relu(x)``: a relu steer of ``v`` gives ``2 * (relu(x) + v)``."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) * 2


def _steer_spec(vector: torch.Tensor) -> InterventionSpec:
    return tl.when(tl.func("relu"), tl.steer(vector, feature_axis=-1))


def _one_hot(index: int, value: float = 5.0) -> torch.Tensor:
    vector = torch.zeros(_WIDTH)
    vector[index] = value
    return vector


def test_large_vectors_differing_in_one_element_get_different_identity() -> None:
    """A middle-element difference (elided by torch's repr) changes every identity."""

    first = _steer_spec(torch.zeros(_WIDTH))
    second = _steer_spec(_one_hot(_WIDTH // 2))
    assert first.rules[0].rule_id != second.rules[0].rule_id
    assert first.rules[0].action_repr != second.rules[0].action_repr
    assert first.spec_digest != second.spec_digest


def test_equal_content_gives_equal_identity_across_objects_and_layouts() -> None:
    """Identity is content-derived: distinct objects and strided views agree."""

    base = torch.arange(2 * _WIDTH, dtype=torch.float32)
    strided_view = base[::2]
    contiguous_copy = strided_view.clone()
    assert not strided_view.is_contiguous()
    first = _steer_spec(strided_view)
    second = _steer_spec(contiguous_copy)
    assert first.spec_digest == second.spec_digest
    assert first.rules[0].rule_id == second.rules[0].rule_id
    assert first.rules[0].action_repr == second.rules[0].action_repr


def test_identity_distinguishes_dtype_and_shape_with_equal_bytes() -> None:
    """Equal raw bytes under a different dtype or shape are a different rule."""

    flat = torch.zeros(_WIDTH, dtype=torch.float32)
    as_int = torch.zeros(_WIDTH, dtype=torch.int32)
    reshaped = torch.zeros(2, _WIDTH // 2, dtype=torch.float32)
    digests = {
        _steer_spec(flat).spec_digest,
        _steer_spec(as_int).spec_digest,
        _steer_spec(reshaped).spec_digest,
    }
    assert len(digests) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_identity_is_device_independent() -> None:
    """The same values on CPU and CUDA name the same rule."""

    vector = torch.randn(_WIDTH)
    assert _steer_spec(vector).spec_digest == _steer_spec(vector.cuda()).spec_digest


def test_in_place_edit_after_staging_changes_digest_and_action_repr() -> None:
    """Records follow the current tensor content, matching a fresh spec over it."""

    vector = torch.zeros(_WIDTH)
    spec = _steer_spec(vector)
    staged_digest = spec.spec_digest
    staged_action = spec.rules[0].action_repr
    staged_rule_id = spec.rules[0].rule_id

    vector.add_(_one_hot(7, value=3.0))
    fresh = _steer_spec(_one_hot(7, value=3.0))
    assert spec.spec_digest != staged_digest
    assert spec.rules[0].action_repr != staged_action
    assert spec.rules[0].rule_id != staged_rule_id
    assert spec.spec_digest == fresh.spec_digest
    assert spec.rules[0].action_repr == fresh.rules[0].action_repr

    vector.sub_(_one_hot(7, value=3.0))
    assert spec.spec_digest == staged_digest
    assert spec.rules[0].rule_id == staged_rule_id


def test_untracked_data_edit_changes_spec_digest() -> None:
    """A ``.data`` write (no version-counter bump) still changes ``spec_digest``."""

    vector = torch.zeros(_WIDTH)
    spec = _steer_spec(vector)
    staged_digest = spec.spec_digest
    vector.data[123] = 1.0
    assert spec.spec_digest != staged_digest
    assert spec.spec_digest == _steer_spec(vector.clone()).spec_digest
    # The exact read refreshed the cached identity the other reads share.
    assert spec.rules[0].rule_id == _steer_spec(vector.clone()).rules[0].rule_id


def test_merge_of_distinct_large_vectors_no_longer_refuses() -> None:
    """Two different model-width vectors merge into a two-rule spec."""

    first = tl.when(tl.func("relu"), tl.steer(torch.zeros(_WIDTH), feature_axis=-1))
    second = tl.when(tl.func("sigmoid"), tl.steer(_one_hot(1), feature_axis=-1))
    third = tl.when(tl.func("relu"), tl.steer(_one_hot(_WIDTH // 2), feature_axis=-1))
    merged = first.merge(second)
    assert len(merged.rules) == 2
    # Same WHERE, different large vectors: distinct rules, so no duplicate refusal.
    assert len(first.merge(third).rules) == 2


def test_merge_of_true_duplicates_still_refuses() -> None:
    """Equal content in distinct tensor objects is still one rule."""

    first = _steer_spec(_one_hot(42))
    second = _steer_spec(_one_hot(42))
    with pytest.raises(InvalidArgumentError) as excinfo:
        first.merge(second)
    assert excinfo.value.fields["code"] == "spec_rules_duplicate"


def test_tensor_free_rule_identity_keeps_its_released_format() -> None:
    """Rules without tensor arguments render byte-identically to 2.36.0."""

    helper = tl.scale(2.0)
    spec = tl.when(tl.func("relu"), helper)
    expected = f"helper:{helper.helper_name}:{helper.args!r}:{helper.kwargs!r}"
    assert spec.rules[0].action_repr == expected


def test_bind_report_tracks_an_edit_between_calls() -> None:
    """Each bound call reports the identity it ran with, keys consistent."""

    model = _ReluModel()
    vector = torch.tensor([1.0])
    spec = _steer_spec(vector)
    bound = spec.bind(model)

    first_out = bound(torch.ones(1))
    first_report = bound.last_report
    vector.add_(9.0)
    second_out = bound(torch.ones(1))
    second_report = bound.last_report

    assert first_out.tolist() == [4.0]
    assert second_out.tolist() == [22.0]
    assert first_report.spec_digest != second_report.spec_digest
    assert second_report.spec_digest == _steer_spec(torch.tensor([10.0])).spec_digest
    for report in (first_report, second_report):
        assert set(report.rule_fire_counts) == set(report.resolved_static_targets)
        assert report.zero_fire_rule_ids == ()
        assert {fire["rule_id"] for fire in report.fires} == set(report.rule_fire_counts)


def test_capture_event_records_the_current_rule_identity() -> None:
    """A capture after an in-place edit records the edited content's rule id."""

    vector = torch.tensor([1.0])
    spec = _steer_spec(vector)
    vector.add_(9.0)
    trace = tl.trace(_ReluModel(), torch.ones(1), intervene=spec)
    expected_rule_id = _steer_spec(torch.tensor([10.0])).rules[0].rule_id
    events = [
        row
        for row in trace.state_history
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]
    assert events, "the capture recorded no intervention event"
    recorded = {rule["rule_id"] for event in events for rule in event.get("rules", ())}
    assert recorded == {expected_rule_id}
    assert all(not event.get("zero_fire_rule_ids") for event in events)
