"""``torch.distributions`` outputs are structured containers: one output op per parameter.

Capture used to walk a ``Distribution`` by attribute search, keep the first tensor as
the single output op under an empty address and stamp the output lossy. These pins
cover the container contract that replaces it: parameter-named attribute paths, the
inert exact rebuild, the honest lossy flag for state the contract cannot record, the
module boundary, interventions on a parameter, replay and save/load.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.distributions import (
    Categorical,
    Independent,
    Normal,
    TransformedDistribution,
)
from torch.distributions.transforms import TanhTransform

import torchlens as tl
from torchlens._ir.container import (
    ContainerReconstructionError,
    ContainerSpec,
    NamedField,
    rebuild_container_from_spec,
)
from torchlens._ir.distribution_container import (
    DISTRIBUTION_CONTAINER_KIND,
    DISTRIBUTION_STATE_UNRECORDABLE,
    distribution_layout,
)
from torchlens._runnable import PathFaithfulness
from torchlens.backends.torch.ops import (
    _try_build_container_spec,
    _walk_output_tensors_with_paths,
    runnable_output_losslessness,
)


class NormalHead(nn.Module):
    """Two heads parameterize a diagonal Normal (the DIAYN policy shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.mu = nn.Linear(3, 2)
        self.log_std = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> Normal:
        """Build the distribution from the two heads."""

        return Normal(self.mu(x), self.log_std(x).clamp(-5, 2).exp())


class CategoricalHead(nn.Module):
    """A logits head returned as a ``Categorical``."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> Categorical:
        """Build the distribution from logits."""

        return Categorical(logits=self.lin(x))


class IndependentHead(nn.Module):
    """A diagonal Normal reinterpreted as one multivariate event."""

    def __init__(self) -> None:
        super().__init__()
        self.head = NormalHead()

    def forward(self, x: torch.Tensor) -> Independent:
        """Wrap the child module's Normal."""

        return Independent(self.head(x), 1)


class TanhHead(nn.Module):
    """A tanh-squashed Normal: the transform list is state the contract cannot record."""

    def __init__(self) -> None:
        super().__init__()
        self.head = NormalHead()

    def forward(self, x: torch.Tensor) -> TransformedDistribution:
        """Squash the child module's Normal."""

        return TransformedDistribution(self.head(x), [TanhTransform()])


class PolicyWithInnerDistribution(nn.Module):
    """Samples deterministically from a child module's distribution (the IQL/CQL shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.policy = NormalHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the distribution mean squashed."""

        dist = self.policy(x)
        return torch.tanh(dist.loc) * dist.scale


def _x() -> torch.Tensor:
    return torch.arange(6.0).view(2, 3) / 6


def _trace(model: nn.Module, **kwargs: Any) -> Any:
    torch.manual_seed(0)
    return tl.trace(model, _x(), **kwargs)


def _assert_same_distribution(rebuilt: Any, live: Any) -> None:
    assert type(rebuilt) is type(live)
    assert set(vars(rebuilt)) == set(vars(live))
    for name, value in vars(live).items():
        other = vars(rebuilt)[name]
        if isinstance(value, torch.Tensor):
            assert torch.equal(other.detach(), value.detach()), name
        elif isinstance(value, torch.distributions.Distribution):
            _assert_same_distribution(other, value)
        else:
            assert other == value, name


def test_normal_output_records_one_output_op_per_parameter() -> None:
    """Capture keeps ``loc`` and ``scale`` as two output ops at attribute paths."""

    model = NormalHead().eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = _trace(model)
    # The attribute-search fallback (and its advisory) is no longer taken.
    assert not [w for w in caught if "output traversal does not support" in str(w.message)]
    paths = [tuple(op.container_path) for op in trace.output_ops]
    assert paths == [(NamedField("loc"),), (NamedField("scale"),)]
    live = model(_x())
    assert torch.equal(trace.output_ops[0].out, live.loc.detach())
    assert torch.equal(trace.output_ops[1].out, live.scale.detach())


def test_normal_output_is_lossless_and_rebuilds_exactly() -> None:
    """The proof is positive and the spec rebuilds the same type with the same state."""

    model = NormalHead().eval()
    out = model(_x())
    entries = list(_walk_output_tensors_with_paths(out))
    proof = runnable_output_losslessness(out, entries)
    assert proof["lossless"] is True, proof
    assert proof["root_kind"] == "supported_container"
    spec = _try_build_container_spec(out)
    assert spec is not None and spec.kind == DISTRIBUTION_CONTAINER_KIND
    assert spec.fields == ("loc", "scale")
    assert spec.lossy_reconstruction is False
    rebuilt = rebuild_container_from_spec(spec, [tensor for tensor, _path, _spec in entries])
    _assert_same_distribution(rebuilt, out)


def test_categorical_records_the_constructed_parameter_and_its_alias() -> None:
    """``Categorical(logits=...)`` records ``logits`` once; ``_param`` is an alias of it."""

    out = CategoricalHead().eval()(_x())
    layout = distribution_layout(out)
    assert layout.fields == ("logits",)
    assert ("_param", "logits") in layout.aliases
    assert layout.lossy_reason == ""
    spec = _try_build_container_spec(out)
    assert spec is not None
    rebuilt = rebuild_container_from_spec(spec, [out.logits])
    assert rebuilt._param is rebuilt.logits
    _assert_same_distribution(rebuilt, out)
    assert torch.equal(rebuilt.probs, out.probs)


def test_independent_rebuilds_through_its_base_distribution() -> None:
    """A nested distribution is a child container at ``.base_dist``."""

    out = IndependentHead().eval()(_x())
    paths = [path for _tensor, path, _spec in _walk_output_tensors_with_paths(out)]
    assert paths == [
        (NamedField("base_dist"), NamedField("loc")),
        (NamedField("base_dist"), NamedField("scale")),
    ]
    entries = list(_walk_output_tensors_with_paths(out))
    assert runnable_output_losslessness(out, entries)["lossless"] is True
    spec = _try_build_container_spec(out)
    assert spec is not None
    rebuilt = rebuild_container_from_spec(spec, [tensor for tensor, _path, _spec in entries])
    _assert_same_distribution(rebuilt, out)


def test_transformed_distribution_stays_lossy_and_names_the_transforms() -> None:
    """The transform list is not recordable: lossy, reason named, parameters still addressable."""

    out = TanhHead().eval()(_x())
    entries = list(_walk_output_tensors_with_paths(out))
    assert [path for _tensor, path, _spec in entries] == [
        (NamedField("base_dist"), NamedField("loc")),
        (NamedField("base_dist"), NamedField("scale")),
    ]
    proof = runnable_output_losslessness(out, entries)
    assert proof["lossless"] is False
    assert (
        proof["reason"] == f"{DISTRIBUTION_STATE_UNRECORDABLE}:TransformedDistribution.transforms"
    )
    spec = _try_build_container_spec(out)
    assert spec is not None and spec.lossy_reconstruction is True


def test_distribution_output_settles_verified_on_run() -> None:
    """A live run rebuilds the Normal and settles VERIFIED (it used to be UNVERIFIABLE)."""

    model = NormalHead().eval()
    trace = _trace(model)
    result = trace.run(inputs=_x())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    _assert_same_distribution(result.output, model(_x()))


def test_lossy_distribution_output_settles_unverifiable_on_run() -> None:
    """The honest flag reaches the run verdict: never VERIFIED for a lossy rebuild."""

    model = TanhHead().eval()
    trace = _trace(model)
    result = trace.run(inputs=_x(), on_divergence="return")
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


def test_module_returning_a_distribution_exposes_parameter_outputs() -> None:
    """The module boundary walks the same container: ``policy`` outputs ``.loc`` and ``.scale``."""

    trace = _trace(PolicyWithInnerDistribution().eval())
    call = trace.module_calls["policy:1"]
    assert [tuple(path) for path in call.output_paths] == [
        (NamedField("loc"),),
        (NamedField("scale"),),
    ]


def test_output_replacement_targets_one_distribution_parameter() -> None:
    """Replacing ``.loc`` returns a new distribution; ``scale`` and the original are untouched."""

    from torchlens.backends.torch._ops_interventions import _replace_output_tensors_by_path

    live = NormalHead().eval()(_x())
    zeros = torch.zeros(2, 2)
    replaced = _replace_output_tensors_by_path(live, {(NamedField("loc"),): zeros})
    assert type(replaced) is Normal and replaced is not live
    assert replaced.loc is zeros
    assert replaced.scale is live.scale
    assert not torch.equal(live.loc, zeros)
    cat = CategoricalHead().eval()(_x())
    swapped = _replace_output_tensors_by_path(cat, {(NamedField("logits"),): zeros})
    assert swapped.logits is zeros and swapped._param is zeros


def test_assign_by_path_rebuilds_a_nested_distribution() -> None:
    """The generic path assigner follows ``.base_dist.scale`` and keeps aliases consistent."""

    from torchlens._utils.collections import assign_into_container_by_path

    live = IndependentHead().eval()(_x())
    ones = torch.ones(2, 2)
    updated = assign_into_container_by_path(
        live, (NamedField("base_dist"), NamedField("scale")), ones
    )
    assert type(updated) is Independent and type(updated.base_dist) is Normal
    assert updated.base_dist.scale is ones
    assert updated.base_dist.loc is live.base_dist.loc
    assert live.base_dist.scale is not ones


def test_distribution_output_round_trips_through_save_and_load(tmp_path: Path) -> None:
    """The spec (kind, fields, literal state, aliases) survives a portable bundle."""

    from torchlens.options import CaptureOptions

    model = CategoricalHead().eval()
    trace = _trace(model, capture=CaptureOptions(capture_container_structure=True))
    bundle = tmp_path / "categorical.tlspec"
    trace.save(bundle)
    loaded = tl.load(bundle)
    assert [tuple(op.container_path) for op in loaded.output_ops] == [(NamedField("logits"),)]
    spec = loaded.output_ops[0].container_spec
    assert isinstance(spec, ContainerSpec)
    assert spec.kind == DISTRIBUTION_CONTAINER_KIND
    rebuilt = rebuild_container_from_spec(spec, [op.out for op in loaded.output_ops])
    _assert_same_distribution(rebuilt, model(_x()))


def test_tampered_distribution_spec_refuses_typed() -> None:
    """A spec naming a non-Distribution type, or dunder state, is a typed refusal."""

    forged_type = ContainerSpec(
        kind=DISTRIBUTION_CONTAINER_KIND,
        length=1,
        fields=("loc",),
        type_module="collections",
        type_qualname="OrderedDict",
        aux_data=(("state", ()), ("aliases", ())),
    )
    with pytest.raises(ContainerReconstructionError):
        rebuild_container_from_spec(forged_type, [torch.zeros(1)])
    forged_state = ContainerSpec(
        kind=DISTRIBUTION_CONTAINER_KIND,
        length=1,
        fields=("loc",),
        type_module="torch.distributions.normal",
        type_qualname="Normal",
        aux_data=(("state", (("__class__", "x"),)), ("aliases", ())),
    )
    with pytest.raises(ContainerReconstructionError):
        rebuild_container_from_spec(forged_state, [torch.zeros(1)])


def test_forged_lossless_flag_cannot_bless_a_stateful_type() -> None:
    """Load-time recompute: a data-descriptor field makes the rebuild lossy whatever the flag."""

    from torchlens._runnable_execution import _spec_node_reconstruction_lossy

    spec = ContainerSpec(
        kind=DISTRIBUTION_CONTAINER_KIND,
        length=1,
        fields=("mean",),
        type_module="torch.distributions.normal",
        type_qualname="Normal",
        aux_data=(("state", ()), ("aliases", ())),
        lossy_reconstruction=False,
    )
    # ``Normal.mean`` is a property (a data descriptor): it cannot be restored inertly.
    assert _spec_node_reconstruction_lossy(spec) is True
    honest = ContainerSpec(
        kind=DISTRIBUTION_CONTAINER_KIND,
        length=2,
        fields=("loc", "scale"),
        type_module="torch.distributions.normal",
        type_qualname="Normal",
        aux_data=(("state", ()), ("aliases", ())),
        lossy_reconstruction=False,
    )
    assert _spec_node_reconstruction_lossy(honest) is False
