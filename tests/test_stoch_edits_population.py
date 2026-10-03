"""F02 population noun: origin contract, digests, datums, reducers (edits memo D2/D3/D11/D13).

Tier-A rows exercised here: A11 (reducer contract -- reduced-precision members
accumulate in float32, ``stochastic=False``, empty group refuses) plus the
population construction refusal roster.
"""

from __future__ import annotations

import pytest
import torch

from torchlens.intervention import OneDatum, PerRowDatums, reference


def test_origin_is_required_and_nonempty() -> None:
    """D2/DIS-3: a population without recorded provenance refuses typed."""

    with pytest.raises(Exception) as excinfo:
        reference(torch.randn(3, 2), origin="")
    assert excinfo.value.fields["code"] == "population_origin_required"
    with pytest.raises(Exception) as excinfo:
        reference(torch.randn(3, 2), origin="   ")
    assert excinfo.value.fields["code"] == "population_origin_required"


def test_tensor_stack_members_are_axis0_slices_with_content_digest() -> None:
    """The noun's documented stacking contract + the mandatory D11 digest."""

    stack = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    ref = reference(stack, origin="unit stack")
    assert len(ref) == 3
    assert ref.kind == "tensor"
    assert ref.digest_kind == "content"
    assert torch.equal(ref.members[1], stack[1])
    # Same bytes + same origin -> same identity; different origin -> different.
    twin = reference(stack.clone(), origin="unit stack")
    assert twin.population_identity == ref.population_identity
    other = reference(stack.clone(), origin="different provenance")
    assert other.population_identity != ref.population_identity


def test_member_sequence_heterogeneous_refuses() -> None:
    """A donor population is one homogeneous stack."""

    with pytest.raises(Exception) as excinfo:
        reference([torch.randn(3), torch.randn(4)], origin="x")
    assert excinfo.value.fields["code"] == "population_heterogeneous"
    with pytest.raises(Exception) as excinfo:
        reference([torch.randn(3), torch.randn(3).to(torch.float64)], origin="x")
    assert excinfo.value.fields["code"] == "population_heterogeneous"


def test_datum_count_mismatch_refuses() -> None:
    """Every member carries exactly one datum, aligned by order."""

    with pytest.raises(Exception) as excinfo:
        reference(torch.randn(3, 2), origin="x", data=["a", "b"])
    assert excinfo.value.fields["code"] == "population_datum_count_mismatch"


@pytest.mark.smoke
def test_empty_and_unsupported_sources_refuse() -> None:
    """Zero members and non-population sources refuse typed."""

    with pytest.raises(Exception) as excinfo:
        reference(torch.randn(0, 4), origin="x")
    assert excinfo.value.fields["code"] == "population_empty"
    with pytest.raises(Exception) as excinfo:
        reference(12345, origin="x")
    assert excinfo.value.fields["code"] == "population_source_invalid"


@pytest.mark.smoke
def test_reducers_are_deterministic_tensors() -> None:
    """D3: mean/std are reductions over the member stack, never draws."""

    stack = torch.stack([torch.zeros(4), torch.ones(4) * 2.0])
    ref = reference(stack, origin="means")
    assert torch.equal(ref.mean(), torch.ones(4))
    assert torch.allclose(ref.std(), torch.full((4,), 2.0**0.5))
    # over='all' reduces to a scalar; explicit axes keepdim-reduce.
    assert ref.mean("all").ndim == 0
    per_axis = reference(torch.arange(8, dtype=torch.float32).reshape(2, 4), origin="ax")
    reduced = per_axis.mean(0)
    assert tuple(reduced.shape) == (1,)


def test_reducer_accumulates_reduced_precision_in_float32() -> None:
    """A11: bf16 members accumulate in float32 and cast back."""

    members = torch.randn(64, 8).to(torch.bfloat16)
    ref = reference(members, origin="bf16 pop")
    result = ref.mean()
    assert result.dtype == torch.bfloat16
    oracle = members.to(torch.float32).mean(dim=0).to(torch.bfloat16)
    assert torch.equal(result, oracle)


@pytest.mark.smoke
def test_reducer_agreement_and_refusals() -> None:
    """Conditioned reductions honor agreement classes; empty classes refuse."""

    stack = torch.stack([torch.zeros(3), torch.ones(3), torch.ones(3) * 4.0])
    ref = reference(stack, origin="classes", data=["a", "b", "b"])
    conditioned = ref.mean(group_by=lambda d: d, matching=OneDatum("b"))
    assert torch.allclose(conditioned, torch.full((3,), 2.5))
    with pytest.raises(Exception) as excinfo:
        ref.mean(group_by=lambda d: d, matching=OneDatum("zzz"))
    assert excinfo.value.fields["code"] == "population_agreement_empty"
    # The ephemeral refusal teaches: requested key + available class sizes.
    assert excinfo.value.fields["class_sizes"] == {"'a'": 1, "'b'": 2}
    with pytest.raises(Exception) as excinfo:
        ref.mean(group_by=lambda d: d)
    assert excinfo.value.fields["code"] == "population_matching_invalid"
    with pytest.raises(Exception) as excinfo:
        ref.mean(group_by=lambda d: d, matching=OneDatum(PerRowDatums(["a", "b"])))
    assert excinfo.value.fields["code"] == "population_matching_invalid"
    with pytest.raises(Exception) as excinfo:
        ref.std(group_by=lambda d: d, matching=OneDatum("a"))
    assert excinfo.value.fields["code"] == "population_reduce_underdetermined"


def test_trace_backed_population_records_address_identity() -> None:
    """D11: trace-backed populations default to the ADDRESS digest kind."""

    class _TraceLike:
        trace_label = "toy"
        model_class_qualname = "Toy"
        random_seed = 7
        num_operations = 3
        layer_dict_all_keys: dict[str, object] = {}
        ops: dict[str, object] = {}
        layer_labels: tuple[str, ...] = ()

    ref = reference([_TraceLike(), _TraceLike()], origin="two runs")
    assert ref.kind == "trace"
    assert ref.digest_kind == "address"
    assert len(ref) == 2
