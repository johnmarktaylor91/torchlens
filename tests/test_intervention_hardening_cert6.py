"""Regression tests for the cert6 (round-6) intervention hardening fixes.

Each test pins a specific silent-corruption / silent-empty-result bug found by the
cert5 inspector that previously lacked coverage:

* BLOCKER-1 -- differentiable-replay shallow fork aliased nested annotation dicts,
  so mutating the fork silently corrupted the parent trace.
* MAJOR-1 -- ``Trace._optimizer`` was declared ``share`` but silently deep-copied on
  fork because it is absent from ``MODEL_LOG_FIELD_ORDER``.
* MAJOR-2 -- intervention-spec save/load silently stringified non-string dict keys.
* MAJOR-3 -- ``DIRECTION_AGNOSTIC_KINDS`` omitted output/output_at/input_at, so a
  validly-composed forward-output & backward selector resolved to 0 sites.
* MAJOR-4 -- ``SuperOp.diff_pair`` silently truncated a vector member to element 0
  when compared against a scalar member.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.intervention._metrics import (
    cosine_distance,
    relative_l1_scalar,
)
from torchlens.intervention._super._base import _TensorBearing
from torchlens.intervention.errors import UnserializableDictKeyError
from torchlens.intervention.save import (
    SaveLevel,
    _SerializedState,
    _deserialize_value,
    _serialize_value,
)
from torchlens.options import ReplayOptions


def _identity_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Return the activation unchanged."""

    return out


def _new_state() -> _SerializedState:
    """Return a fresh empty serialization state."""

    return _SerializedState(tensor_entries=[], tensor_refs={})


class _ReluMul(torch.nn.Module):
    """Tiny model with an out-of-replay-cone input op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``relu(x) * 2`` summed to a scalar."""

        return (torch.relu(x) * 2).sum()


class _TupleOut(torch.nn.Module):
    """Model returning a two-element tuple output container."""

    def __init__(self) -> None:
        """Initialize two independent linear branches."""

        super().__init__()
        self.lin1 = torch.nn.Linear(4, 4)
        self.lin2 = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(lin1(x), lin2(x))``."""

        return self.lin1(x), self.lin2(x)


class _LinearSum(torch.nn.Module):
    """Model whose params can carry an attached optimizer."""

    def __init__(self) -> None:
        """Initialize a single linear layer."""

        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``lin(x)`` summed to a scalar."""

        return self.lin(x).sum()


def test_differentiable_replay_fork_annotation_is_independent_of_parent() -> None:
    """BLOCKER-1: annotating a diff-replay fork must not mutate the parent trace."""

    log = tl.trace(_ReluMul(), torch.randn(4), intervention_ready=True)
    log.annotate("input_1", data={"owner": "original"})
    parent_op = log.layer_dict_all_keys["input_1"]
    original_annotations = {"user": {"data": {"owner": "original"}}}
    assert parent_op.annotations == original_annotations

    fork = log.push(
        replay=ReplayOptions(hooks={tl.func("relu"): _identity_hook}, differentiable=True)
    )
    # ``input_1`` is outside the replay cone -> it goes through the shallow-fork path.
    fork.annotate("input_1", data={"owner": "fork-only"})

    # The parent's out-of-cone op must be untouched (previously it leaked).
    assert log.layer_dict_all_keys["input_1"].annotations == original_annotations
    # The fork carries the new annotation independently.
    assert fork.layer_dict_all_keys["input_1"].annotations == {
        "user": {"data": {"owner": "fork-only"}}
    }


def test_fork_shares_optimizer_aliasing_live_params() -> None:
    """MAJOR-1: a forked trace's ``_optimizer`` must alias the live model params."""

    model = _LinearSum()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    log = tl.trace(model, torch.randn(2, 4), intervention_ready=True, optimizer=opt)
    assert getattr(log, "_optimizer", None) is opt

    fork = log.fork()
    # Shared by reference, not deep-copied.
    assert getattr(fork, "_optimizer", None) is opt
    live_param = next(model.parameters())
    fork_param = fork._optimizer.param_groups[0]["params"][0]
    assert fork_param is live_param


def test_fork_policy_shares_optimizer_but_keeps_it_off_the_display_field_order() -> None:
    """MAJOR-1: ``_optimizer`` has a SHARE policy yet stays out of FIELD_ORDER."""

    from torchlens.constants import MODEL_LOG_FIELD_ORDER
    from torchlens.intervention.types import (
        MODEL_LOG_FIELD_FORK_POLICY,
        ForkFieldPolicy,
    )

    assert "_optimizer" not in MODEL_LOG_FIELD_ORDER
    assert MODEL_LOG_FIELD_FORK_POLICY["_optimizer"] is ForkFieldPolicy.FORK_SHARE


def test_spec_serialize_preserves_non_string_dict_keys() -> None:
    """MAJOR-2: int/float/tuple dict keys survive a spec save/load round-trip."""

    original = {0: "zero", 1: "one", 2.5: "two_point_five", (1, 2): "tuple_key", "s": "str_key"}
    serialized = _serialize_value(original, SaveLevel.AUDIT, _new_state())
    loaded = _deserialize_value(serialized, {})

    assert loaded == original
    assert [type(key).__name__ for key in loaded] == ["int", "int", "float", "tuple", "str"]
    # Downstream lookups by original key type keep working.
    assert loaded[0] == "zero"
    assert loaded[(1, 2)] == "tuple_key"

    # Nested non-string keys are preserved too.
    nested = {"outer": {3: "three"}}
    assert _deserialize_value(_serialize_value(nested, SaveLevel.AUDIT, _new_state()), {}) == nested

    # The common all-string case keeps the plain on-disk object format unchanged.
    plain = {"a": 1, "b": [2, 3]}
    assert _serialize_value(plain, SaveLevel.AUDIT, _new_state()) == plain


def test_spec_serialize_rejects_unpreservable_dict_key() -> None:
    """MAJOR-2: an unpreservable key raises rather than silently stringifying."""

    try:
        _serialize_value({frozenset({1}): "x"}, SaveLevel.AUDIT, _new_state())
    except UnserializableDictKeyError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("expected UnserializableDictKeyError for a frozenset dict key")


def test_output_at_composed_with_grad_input_resolves_nonzero_sites() -> None:
    """MAJOR-3: ``output_at(0) & grad_input()`` intersects meaningfully (not 0)."""

    model = _TupleOut()
    log = tl.trace(model, torch.randn(2, 4), intervention_ready=True)
    branch0 = log.find_sites(tl.output_at(0))
    assert branch0, "expected output_at(0) to match the first tuple branch"
    log.log_backward(branch0[0].out.sum())

    # A backward-only selector composed with a forward output-position selector
    # must bridge the GradFn site back to its paired forward op.
    composed = log.find_sites(tl.output_at(0) & tl.grad_input())
    assert len(composed) >= 1


@pytest.mark.parametrize(
    ("left", "right", "check_scalar_control"),
    [
        (
            torch.tensor(1.0),
            torch.tensor([1.0, 2.0, 3.0, 4.0, 100000.0]),
            True,
        ),
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0]), False),
    ],
    ids=["scalar_to_vector", "vector_to_singleton"],
)
def test_relative_l1_scalar_rejects_numel_mismatch(
    left: torch.Tensor,
    right: torch.Tensor,
    check_scalar_control: bool,
) -> None:
    """The scalar fallback rejects both mismatch orientations without truncation."""
    with pytest.raises(ValueError, match="equal element counts"):
        relative_l1_scalar(left, right)

    if check_scalar_control:
        assert float(relative_l1_scalar(torch.tensor(1.0), torch.tensor(2.0))) == 1.0


def test_diff_row_scalar_vs_vector_member_raises_not_truncates() -> None:
    """MAJOR-4: diff over a scalar ref and a vector member fails loudly."""

    # Scalar reference vs a longer-vector member -> previously silently returned
    # a truncated (element-0-only) distance; now raises via the real metric.
    tensors = [torch.tensor(1.0), torch.tensor([1.0, 2.0, 3.0, 4.0, 100000.0])]
    try:
        _TensorBearing._diff_row(tensors, 0, cosine_distance)
    except ValueError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("expected ValueError for a scalar-ref vs vector diff row")

    # Both-scalar members still use the scalar fallback and produce finite values.
    both_scalar = _TensorBearing._diff_row(
        [torch.tensor(1.0), torch.tensor(3.0)], 0, cosine_distance
    )
    assert both_scalar.tolist() == [[0.0, 2.0]]
    matrix = _TensorBearing._diff_matrix([torch.tensor(1.0), torch.tensor(2.0)], cosine_distance)
    assert matrix[0, 1] == 1.0
