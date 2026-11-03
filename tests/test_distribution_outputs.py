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
from torch import distributions as D, nn
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
)
from torchlens._runnable import PathFaithfulness
from torchlens.backends.torch._ops_container_base import _try_build_container_spec
from torchlens.backends.torch._ops_containers import (
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


def test_output_replacement_targets_one_distribution_parameter() -> None:
    """Replacing ``.loc`` returns a new distribution; ``scale`` and the original are untouched."""

    from torchlens.backends.torch._ops_predicates import _replace_output_tensors_by_path

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

    from torchlens._ir._output_path_assign import assign_into_container_by_path

    live = IndependentHead().eval()(_x())
    ones = torch.ones(2, 2)
    updated = assign_into_container_by_path(
        live, (NamedField("base_dist"), NamedField("scale")), ones
    )
    assert type(updated) is Independent and type(updated.base_dist) is Normal
    assert updated.base_dist.scale is ones
    assert updated.base_dist.loc is live.base_dist.loc
    assert live.base_dist.scale is not ones


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

    from torchlens._runnable.call_outputs import _spec_node_reconstruction_lossy

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


def _families() -> list[tuple[str, Any, bool]]:
    """``(name, factory, lossless)`` for the standard families (the coverage table)."""

    p = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    loc, scale = torch.tensor([0.1, -0.2]), torch.tensor([1.5, 0.5])
    cov = torch.tensor([[2.0, 0.3], [0.3, 1.0]])
    return [
        ("Normal", lambda: D.Normal(loc, scale), True),
        ("Laplace", lambda: D.Laplace(loc, scale), True),
        ("Cauchy", lambda: D.Cauchy(loc, scale), True),
        ("Uniform", lambda: D.Uniform(loc - 1, loc + 1), True),
        ("Gamma", lambda: D.Gamma(scale, scale + 1), True),
        ("Exponential", lambda: D.Exponential(scale), True),
        ("Poisson", lambda: D.Poisson(scale), True),
        ("Beta", lambda: D.Beta(scale, scale + 1), True),
        ("Dirichlet", lambda: D.Dirichlet(p), True),
        ("StudentT", lambda: D.StudentT(scale + 2, loc, scale), True),
        ("Bernoulli(logits)", lambda: D.Bernoulli(logits=loc), True),
        ("Bernoulli(probs)", lambda: D.Bernoulli(probs=scale / 2), True),
        ("ContinuousBernoulli", lambda: D.ContinuousBernoulli(probs=scale / 2), True),
        ("Categorical(logits)", lambda: D.Categorical(logits=p.log()), True),
        ("Categorical(probs)", lambda: D.Categorical(probs=p), True),
        ("OneHotCategorical", lambda: D.OneHotCategorical(probs=p), True),
        ("Binomial", lambda: D.Binomial(5, probs=scale / 2), True),
        ("Multinomial", lambda: D.Multinomial(4, probs=p), True),
        ("Geometric", lambda: D.Geometric(probs=scale / 2), True),
        ("MultivariateNormal(cov)", lambda: D.MultivariateNormal(loc, cov), True),
        (
            "MultivariateNormal(scale_tril)",
            lambda: D.MultivariateNormal(loc, scale_tril=torch.linalg.cholesky(cov)),
            True,
        ),
        (
            "LowRankMultivariateNormal",
            lambda: D.LowRankMultivariateNormal(loc, torch.ones(2, 1), scale),
            True,
        ),
        ("Independent(Normal)", lambda: D.Independent(D.Normal(loc, scale), 1), True),
        (
            "MixtureSameFamily",
            lambda: D.MixtureSameFamily(D.Categorical(probs=p[0]), D.Normal(p[1], p[0] + 1)),
            True,
        ),
        ("LogNormal", lambda: D.LogNormal(loc, scale), False),
        ("Gumbel", lambda: D.Gumbel(loc, scale), False),
        ("RelaxedBernoulli", lambda: D.RelaxedBernoulli(torch.tensor(0.5), probs=scale / 2), False),
        (
            "TransformedDistribution(tanh)",
            lambda: D.TransformedDistribution(D.Normal(loc, scale), [TanhTransform()]),
            False,
        ),
    ]


@pytest.mark.parametrize(
    ("name", "factory", "lossless"), _families(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_standard_family_losslessness_table(name: str, factory: Any, lossless: bool) -> None:
    """Each family is lossless exactly when every instance attribute is recordable.

    A lossless family rebuilds to the same type with the same instance state and the
    same ``log_prob`` on a sample; a lossy family names the attribute it cannot record.
    """

    torch.manual_seed(0)
    live = factory()
    entries = list(_walk_output_tensors_with_paths(live))
    proof = runnable_output_losslessness(live, entries)
    assert proof["lossless"] is lossless, (name, proof["reason"])
    spec = _try_build_container_spec(live)
    assert spec is not None and spec.kind == DISTRIBUTION_CONTAINER_KIND
    if not lossless:
        assert proof["reason"].startswith(DISTRIBUTION_STATE_UNRECORDABLE)
        assert spec.lossy_reconstruction is True
        return
    assert spec.lossy_reconstruction is False
    rebuilt = rebuild_container_from_spec(spec, [tensor for tensor, _path, _spec in entries])
    _assert_same_distribution(rebuilt, live)
    sample = live.sample()
    assert torch.equal(rebuilt.log_prob(sample), live.log_prob(sample))


def _zero_fire_warnings(caught: list[warnings.WarningMessage], code: str) -> list[Any]:
    from torchlens.errors import TorchLensWarning

    return [
        w.message
        for w in caught
        if isinstance(w.message, TorchLensWarning) and w.message.fields.get("code") == code
    ]


def _policy_rule() -> Any:
    return tl.when(tl.module("policy"), tl.zero_ablate())


def test_capture_door_zero_fire_names_the_distribution_output() -> None:
    """``tl.trace(intervene=)`` on a module returning a Normal: zero fire, with the reason."""

    from torchlens.intervention._zero_fire import NON_TENSOR_MODULE_OUTPUT_REASON

    model = PolicyWithInnerDistribution().eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model, _x(), intervene=_policy_rule())
    (warning,) = _zero_fire_warnings(caught, "trace_zero_fire")
    assert warning.fields["reason"] == NON_TENSOR_MODULE_OUTPUT_REASON
    detail = warning.fields["reason_detail"]
    assert "module 'policy' output is a Normal, not a tensor" in detail
    assert "(output paths .loc, .scale)" in detail
    assert detail in str(warning)


def test_rerun_doors_zero_fire_name_the_distribution_output() -> None:
    """The capture-engine rerun warns with the reason; the fast rerun refuses with it."""

    from torchlens.errors import RunCapabilityUnavailableError
    from torchlens.intervention._zero_fire import NON_TENSOR_MODULE_OUTPUT_REASON

    model = PolicyWithInnerDistribution().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, _x(), intervene=_policy_rule())
    with pytest.raises(RunCapabilityUnavailableError) as refused:
        trace.run(inputs=_x() + 1, fast=True)
    assert refused.value.fields.get("reason") == NON_TENSOR_MODULE_OUTPUT_REASON
    assert "(output paths .loc, .scale)" in str(refused.value)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace.run(model, _x() + 1)
    (warning,) = _zero_fire_warnings(caught, "rerun_zero_fire")
    assert warning.fields["reason"] == NON_TENSOR_MODULE_OUTPUT_REASON
    assert "module 'policy' output is a Normal" in warning.fields["reason_detail"]


def test_bind_door_zero_fire_names_the_distribution_output() -> None:
    """``spec.bind`` agrees with the capture door: zero fire, the output untouched, the reason."""

    from torchlens.intervention._zero_fire import NON_TENSOR_MODULE_OUTPUT_REASON

    model = PolicyWithInnerDistribution().eval()
    with torch.no_grad():
        plain = model(_x())
    bound = _policy_rule().bind(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            out = bound(_x())
    assert torch.equal(out, plain)
    (warning,) = _zero_fire_warnings(caught, "bind_zero_fire")
    assert warning.fields["reason"] == NON_TENSOR_MODULE_OUTPUT_REASON
    assert "(output paths .loc, .scale)" in warning.fields["reason_detail"]


def test_runnable_save_round_trips_a_normal_output_verified(tmp_path: Path) -> None:
    """A runnable bundle replays the Normal output from its recorded parameters, VERIFIED."""

    from torchlens.options import CaptureOptions

    model = NormalHead().eval()
    trace = _trace(model, capture=CaptureOptions(replayable=True, capture_container_structure=True))
    path = tmp_path / "normal.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    result = tl.load(path).run(inputs=_x())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    _assert_same_distribution(result.output, model(_x()))


def test_runnable_save_never_blesses_a_lossy_distribution(tmp_path: Path) -> None:
    """A transform-carrying output refuses at save or loads UNVERIFIABLE; never VERIFIED."""

    from torchlens.options import CaptureOptions

    trace = _trace(
        TanhHead().eval(),
        capture=CaptureOptions(replayable=True, capture_container_structure=True),
    )
    path = tmp_path / "tanh.tlspec"
    try:
        trace.save(path, level="runnable", include_weights=True)
    except tl.errors.TorchLensError:
        return
    result = tl.load(path).run(inputs=_x(), on_divergence="return")
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


def test_zero_fire_reason_names_nested_parameter_leaves() -> None:
    """A nested distribution's reason names the leaf ops (``.base_dist.loc``), not the container."""

    from torchlens.intervention._zero_fire import live_distribution_leaf_paths

    assert live_distribution_leaf_paths(IndependentHead().eval()(_x())) == (
        ".base_dist.loc",
        ".base_dist.scale",
    )
    assert live_distribution_leaf_paths(NormalHead().eval()(_x())) == (".loc", ".scale")
