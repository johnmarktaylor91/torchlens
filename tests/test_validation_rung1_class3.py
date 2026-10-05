"""Rung-1 class-3 false validation failures, fixed by narrow proofs or a stronger probe.

Each fix has a positive test (the false failure now validates end to end) and a
negative test (the neighbouring case outside the proof still fails
``perturbation_insensitive`` when its edge is frozen, the armed-proof pattern
of ``test_validation_exemption_tightening``):

- ScheduleNet: ``torch.distributions`` integer-support checks compute
  ``value % 1`` on an int64 sample, which is identically zero
  (``integer_mod_unit_divisor``).
- Neural Map: ``reshape_as``/``view_as`` read only the SHAPE of ``other``
  (``STRUCTURAL_ARG_POSITIONS``).
- KVAE: ``torch.empty(shape).normal_()`` and the other in-place RNG fills
  overwrite every destination element (``rng_probability_template``).
- BotRGCN: per-relation edge counts (``scatter_add_`` of ones) on balanced
  relations are invariant under the full index rotation, so the retry ladder
  now also moves a single in-domain index entry.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation import validate_forward_pass
from torchlens.validation._index_domain import index_domain_single_entry_values
from torchlens.validation.core import (
    _check_whether_func_on_saved_parents_yields_saved_tensor,
    _perturbation_retry_strategies,
)
from torchlens.validation.exemptions import (
    _integer_mod_unit_divisor_decision,
    _posthoc_structural_output_decision,
)


def _capture(model: nn.Module, args: Any, seed: int = 0) -> Any:
    """Capture a full-save trace the way ``validate_forward_pass`` does."""

    torch.manual_seed(seed)
    return tl.trace(
        model,
        args,
        capture=CaptureOptions(layers_to_save="all", save_arg_values=True, random_seed=seed),
    )


def _quiet_validate(model: nn.Module, args: Any) -> bool:
    """Public-path validation with provenance warnings silenced."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate_forward_pass(model, args, random_seed=0))


def _op_with_func_name(trace: Any, func_name: str) -> Any:
    """Return the first op in ``trace`` whose ``func_name`` matches."""

    return next(op for op in trace.layer_list if op.func_name == func_name)


def _freeze_op_replay(op: Any) -> None:
    """Freeze ``op``'s replay callable to return its saved out unconditionally."""

    saved = op.out.detach().clone()
    object.__setattr__(op, "func", lambda *args, **kwargs: saved.clone())


def _edge_result(trace: Any, op: Any, slot: int) -> Any:
    """Run the perturbation check on the parent recorded at positional ``slot``."""

    parent_label = op.parent_arg_positions["args"][slot]
    return _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[parent_label]
    )


def _assert_frozen_edge_fails(trace: Any, op: Any, slot: int) -> None:
    """Freeze ``op`` and assert the edge at ``slot`` fails ``perturbation_insensitive``."""

    _freeze_op_replay(op)
    result = _edge_result(trace, op, slot)
    assert result.decision == "failed", (result.decision, result.reason)
    assert result.reason == "perturbation_insensitive"


# ---------------------------------------------------------------------------
# ScheduleNet: integer % +-1
# ---------------------------------------------------------------------------


class _IntMod(nn.Module):
    """An integer index computed from the input, taken modulo ``divisor``."""

    def __init__(self, divisor: int | float) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 1)
        self.divisor = divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = self.lin(x).squeeze(-1).argmax().reshape(())
        return index % self.divisor


class _CategoricalLogProb(nn.Module):
    """ScheduleNet's policy tail: Categorical sample, then log_prob with support check."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = Categorical(probs=self.lin(x).squeeze(-1).softmax(0))
        return dist.log_prob(dist.sample())


def test_categorical_log_prob_support_check_validates() -> None:
    """The ``value % 1 == 0`` support check no longer fails validation."""

    torch.manual_seed(0)
    assert _quiet_validate(_CategoricalLogProb(), torch.randn(6, 8))


@pytest.mark.parametrize("divisor", [1, -1])
def test_integer_mod_unit_divisor_validates(divisor: int) -> None:
    """An argmax index modulo +-1 validates end to end."""

    torch.manual_seed(0)
    assert _quiet_validate(_IntMod(divisor), torch.randn(6, 8))


@pytest.mark.parametrize("divisor", [1, -1, 1.0])
def test_integer_mod_unit_divisor_dividend_is_exempt(divisor: int | float) -> None:
    """An integer dividend modulo +-1 is exempt, and only by the new proof."""

    torch.manual_seed(0)
    trace = _capture(_IntMod(divisor), torch.randn(6, 8))
    op = _op_with_func_name(trace, "__mod__")
    result = _edge_result(trace, op, 0)
    assert result.decision == "exempted", (result.decision, result.reason)
    assert result.reason == "integer_mod_unit_divisor"


def test_integer_mod_non_unit_divisor_still_fails_when_frozen() -> None:
    """An integer ``% 3`` dividend stays perturbation-tested."""

    torch.manual_seed(0)
    trace = _capture(_IntMod(3), torch.randn(6, 8))
    _assert_frozen_edge_fails(trace, _op_with_func_name(trace, "__mod__"), 0)


def test_float_mod_one_still_fails_when_frozen() -> None:
    """A float ``% 1`` dividend has a fractional part and stays tested."""

    class FloatMod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(8, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x) % 1

    torch.manual_seed(0)
    trace = _capture(FloatMod(), torch.randn(6, 8))
    _assert_frozen_edge_fails(trace, _op_with_func_name(trace, "__mod__"), 0)


def test_integer_mod_unit_divisor_refuses_a_divisor_parent() -> None:
    """The proof covers the dividend slot only, never a perturbed divisor."""

    dividend = torch.tensor([3, 4, 5])
    divisor = torch.ones(3, dtype=torch.long)
    layer = SimpleNamespace(
        func_name="remainder",
        out=torch.zeros(3, dtype=torch.long),
        parent_arg_positions={"args": {0: "dividend", 1: "divisor"}, "kwargs": {}},
    )
    args = (dividend, divisor)
    assert _integer_mod_unit_divisor_decision(layer, ["dividend"], args).exempt
    assert not _integer_mod_unit_divisor_decision(layer, ["divisor"], args).exempt
    same_parent = SimpleNamespace(
        func_name="remainder",
        out=torch.zeros(3, dtype=torch.long),
        parent_arg_positions={"args": {0: "both", 1: "both"}, "kwargs": {}},
    )
    assert not _integer_mod_unit_divisor_decision(same_parent, ["both"], args).exempt
    two = (dividend, torch.full((3,), 2))
    assert not _integer_mod_unit_divisor_decision(layer, ["dividend"], two).exempt
    floats = (dividend.float(), divisor)
    assert not _integer_mod_unit_divisor_decision(layer, ["dividend"], floats).exempt


class _ZeroMaskMod(nn.Module):
    """An all-zero integer mask taken modulo a divisor outside the +-1 proof."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x > 100).to(torch.uint8)
        if self.kind == "uint8_mod_neg1":
            return mask % -1
        if self.kind == "int64_mod_uint8_255":
            return mask.long() % torch.tensor([255], dtype=torch.uint8)
        if self.kind == "int64_mod_float16_one_out_float32":
            buf = torch.empty(mask.shape, dtype=torch.float32)
            return torch.remainder(mask.long(), torch.ones(1, dtype=torch.float16), out=buf)
        return mask.long() % torch.ones(1, dtype=torch.float16)


@pytest.mark.parametrize(
    "kind",
    [
        "uint8_mod_neg1",
        "int64_mod_uint8_255",
        "int64_mod_float16_one",
        "int64_mod_float16_one_out_float32",
    ],
)
def test_integer_mod_outside_unit_proof_still_fails_when_frozen(kind: str) -> None:
    """Wrapped -1 (uint8: ``% 255``) and float16 math (nan) stay tested.

    ``uint8 % -1`` computes ``% 255``, a uint8 ``255`` divisor compares equal to
    ``-1`` in its own dtype, and float16 math turns integers above 65504 into
    nan even when the result is cast into a float32 ``out=`` buffer, so in each
    case the dividend's values reach the output.
    """

    torch.manual_seed(0)
    trace = _capture(_ZeroMaskMod(kind), torch.randn(5))
    op = next(op for op in trace.layer_list if op.func_name in ("__mod__", "remainder"))
    _assert_frozen_edge_fails(trace, op, 0)


def test_integer_mod_unit_divisor_refuses_wrapped_and_narrow_float_operands() -> None:
    """Decision-level refusals for unsigned -1, a uint8 255 divisor and float16."""

    layer = SimpleNamespace(
        func_name="remainder",
        out=torch.zeros(3, dtype=torch.uint8),
        parent_arg_positions={"args": {0: "dividend"}, "kwargs": {}},
    )
    uint8_dividend = torch.zeros(3, dtype=torch.uint8)
    assert _integer_mod_unit_divisor_decision(layer, ["dividend"], (uint8_dividend, 1)).exempt
    assert not _integer_mod_unit_divisor_decision(layer, ["dividend"], (uint8_dividend, -1)).exempt
    int_layer = SimpleNamespace(
        func_name="remainder",
        out=torch.zeros(3, dtype=torch.long),
        parent_arg_positions={"args": {0: "dividend"}, "kwargs": {}},
    )
    dividend = torch.zeros(3, dtype=torch.long)
    wrapped = (dividend, torch.tensor([255], dtype=torch.uint8))
    assert not _integer_mod_unit_divisor_decision(int_layer, ["dividend"], wrapped).exempt
    uint8_one = (dividend, torch.tensor([1], dtype=torch.uint8))
    assert _integer_mod_unit_divisor_decision(int_layer, ["dividend"], uint8_one).exempt
    half_layer = SimpleNamespace(
        func_name="remainder",
        out=torch.zeros(3, dtype=torch.float16),
        parent_arg_positions={"args": {0: "dividend"}, "kwargs": {}},
    )
    half_one = (dividend, torch.ones(1, dtype=torch.float16))
    assert not _integer_mod_unit_divisor_decision(half_layer, ["dividend"], half_one).exempt
    float_layer = SimpleNamespace(
        func_name="remainder",
        out=torch.zeros(3, dtype=torch.float32),
        parent_arg_positions={"args": {0: "dividend"}, "kwargs": {}},
    )
    float_one = (dividend, torch.ones(1, dtype=torch.float32))
    assert _integer_mod_unit_divisor_decision(float_layer, ["dividend"], float_one).exempt
    # float16 computation cast into a float32 out= buffer is still float16 math.
    assert not _integer_mod_unit_divisor_decision(float_layer, ["dividend"], half_one).exempt


# ---------------------------------------------------------------------------
# Neural Map: reshape_as / view_as
# ---------------------------------------------------------------------------


class _ShapeAs(nn.Module):
    """Values from a linear layer, shaped like a tensor read only for its shape."""

    def __init__(self, method: str) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory = x.view(2, 3) + 1.0
        return getattr(self.lin(x), self.method)(memory)


@pytest.mark.parametrize("method", ["reshape_as", "view_as"])
def test_shape_template_as_validates(method: str) -> None:
    """``reshape_as(other)``/``view_as(other)`` no longer fail on ``other``."""

    torch.manual_seed(0)
    assert _quiet_validate(_ShapeAs(method), torch.randn(6))


@pytest.mark.parametrize("method", ["reshape_as", "view_as"])
def test_shape_template_as_values_still_fail_when_frozen(method: str) -> None:
    """Arg 0 supplies every output value and stays perturbation-tested."""

    torch.manual_seed(0)
    trace = _capture(_ShapeAs(method), torch.randn(6))
    _assert_frozen_edge_fails(trace, _op_with_func_name(trace, method), 0)


class _SameParentShapeAs(nn.Module):
    """``h.<method>(h)``: one parent fills the value slot and the shape slot."""

    def __init__(self, method: str) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return getattr(h, self.method)(h)


@pytest.mark.parametrize("method", ["reshape_as", "view_as", "expand_as", "type_as"])
def test_same_parent_in_value_and_shape_slot_still_fails_when_frozen(method: str) -> None:
    """``x.view_as(x)`` (the gradient-reversal idiom) keeps its value edge tested."""

    torch.manual_seed(0)
    trace = _capture(_SameParentShapeAs(method), torch.randn(6))
    op = next(op for op in trace.layer_list if op.func_name in (method, method.replace("_", "")))
    _assert_frozen_edge_fails(trace, op, 0)


@pytest.mark.parametrize("method", ["reshape_as", "view_as", "expand_as", "type_as"])
def test_same_parent_in_value_and_shape_slot_validates(method: str) -> None:
    """Unfrozen, the same-parent spelling validates by perturbation."""

    torch.manual_seed(0)
    assert _quiet_validate(_SameParentShapeAs(method), torch.randn(6))


# ---------------------------------------------------------------------------
# KVAE: in-place RNG fills overwrite their destination
# ---------------------------------------------------------------------------

_RNG_FILLS = {
    "normal_": (),
    "uniform_": (),
    "cauchy_": (),
    "log_normal_": (),
    "geometric_": (0.3,),
    "random_": (0, 7),
}


class _EmptyFill(nn.Module):
    """``torch.empty(shape).<fill>()`` added to a computed value, as in rsample."""

    def __init__(self, fill: str, fill_args: tuple[Any, ...], from_clone: bool) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.fill = fill
        self.fill_args = fill_args
        self.from_clone = from_clone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        destination = h.clone() if self.from_clone else torch.empty(h.shape)
        return h + getattr(destination, self.fill)(*self.fill_args)


@pytest.mark.parametrize("fill", sorted(_RNG_FILLS))
@pytest.mark.parametrize("from_clone", [False, True], ids=["empty", "clone"])
def test_rng_fill_destination_validates(fill: str, from_clone: bool) -> None:
    """The overwritten destination of an in-place RNG fill no longer fails."""

    torch.manual_seed(0)
    assert _quiet_validate(_EmptyFill(fill, _RNG_FILLS[fill], from_clone), torch.randn(3, 4))


def test_multivariate_normal_rsample_validates() -> None:
    """KVAE's ``MultivariateNormal(...).rsample()`` validates."""

    class Rsample(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            loc = self.lin(x)
            dist = torch.distributions.MultivariateNormal(loc, torch.eye(4))
            return dist.rsample()

    torch.manual_seed(0)
    assert _quiet_validate(Rsample(), torch.randn(3, 4))


@pytest.mark.parametrize("fill", sorted(_RNG_FILLS))
def test_rng_fill_template_refuses_a_non_destination_parent(fill: str) -> None:
    """Only a parent at the destination slot is exempt; any other slot falls through."""

    layer = SimpleNamespace(
        func_name=fill,
        saved_kwargs={},
        parent_arg_positions={"args": {0: "destination", 1: "other"}, "kwargs": {}},
    )
    args = (torch.zeros(3), torch.ones(3))
    assert _posthoc_structural_output_decision(layer, args, ["destination"]).exempt
    assert not _posthoc_structural_output_decision(layer, args, ["other"]).exempt
    assert not _posthoc_structural_output_decision(layer, args, ["destination", "other"]).exempt


def test_out_of_place_normal_mean_still_fails_when_frozen() -> None:
    """``torch.normal(mean, std)`` reads ``mean``'s values and stays tested."""

    class OutOfPlace(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.normal(self.lin(x), 1.0)

    torch.manual_seed(0)
    trace = _capture(OutOfPlace(), torch.randn(3, 4))
    _assert_frozen_edge_fails(trace, _op_with_func_name(trace, "normal"), 0)


# ---------------------------------------------------------------------------
# BotRGCN: balanced per-relation edge counts
# ---------------------------------------------------------------------------


class _RelationCounts(nn.Module):
    """PyG RGCNConv's per-relation edge count: ``scatter_add_`` of ones by edge type."""

    def __init__(self, num_relations: int = 2) -> None:
        super().__init__()
        self.num_relations = num_relations

    def forward(self, x: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(edge_type)
        index = edge_type.view(-1).expand_as(ones)
        counts = edge_type.new_zeros(self.num_relations).scatter_add_(0, index, ones)
        return x * counts.max()


def _balanced_edge_types() -> torch.Tensor:
    return torch.cat((torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.long)))


def test_balanced_relation_counts_validate() -> None:
    """A uniform edge-type histogram no longer defeats the index probe."""

    assert _quiet_validate(_RelationCounts(), [torch.randn(4), _balanced_edge_types()])


def test_balanced_relation_count_index_edge_is_verified_not_exempted() -> None:
    """The index edge is VERIFIED by a changed output; nothing is exempted."""

    trace = _capture(_RelationCounts(), [torch.randn(4), _balanced_edge_types()])
    op = _op_with_func_name(trace, "scatter_add_")
    result = _edge_result(trace, op, 2)
    assert result.decision == "validated", (result.decision, result.reason)
    assert result.reason == "perturbation_changed"
    assert "index_single_entry" in _perturbation_retry_strategies(op)


def test_spurious_index_edge_still_fails() -> None:
    """A frozen scatter_add_ (output truly independent of its index) still fails."""

    trace = _capture(_RelationCounts(), [torch.randn(4), _balanced_edge_types()])
    _assert_frozen_edge_fails(trace, _op_with_func_name(trace, "scatter_add_"), 2)


def test_single_entry_probe_moves_one_in_domain_index() -> None:
    """The probe changes exactly one in-domain entry and keeps sentinels."""

    layer = SimpleNamespace(
        func_name="scatter_add_",
        saved_args=(torch.zeros(3), 0, None, None),
        saved_kwargs={},
        parent_arg_positions={"args": {2: "index"}, "kwargs": {}},
    )
    index = torch.tensor([-1, 2, 0, 1, 2])
    probed = index_domain_single_entry_values(layer, "index", index)
    assert probed is not None
    assert probed.tolist() == [-1, 0, 0, 1, 2]
    assert index_domain_single_entry_values(layer, "not_index", index) is None
    assert index_domain_single_entry_values(layer, "index", torch.tensor([-1, 5])) is None


def test_bool_output_ladder_runs_before_the_index_single_entry_rung() -> None:
    """The index-only rung comes last, so a bool gather keeps its bool rungs."""

    class BoolGather(nn.Module):
        def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return torch.gather(x > 0, 0, index)

    trace = _capture(BoolGather(), [torch.randn(5), torch.tensor([0, 2, 4, 1])])
    op = _op_with_func_name(trace, "gather")
    strategies = _perturbation_retry_strategies(op)
    assert strategies[-1] == "index_single_entry"
    assert strategies.index("negate_values") < strategies.index("index_single_entry")
    assert _quiet_validate(BoolGather(), [torch.randn(5), torch.tensor([0, 2, 4, 1])])
