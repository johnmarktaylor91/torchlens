# SOURCE: vendored from stanford-iprl-lab/torchfilter @ master
# (torchfilter/filters/_particle_filter.py, torchfilter/base/_filter.py,
#  torchfilter/base/_dynamics_model.py,
#  torchfilter/base/_particle_filter_measurement_model.py,
#  torchfilter/base/_kalman_filter_measurement_model.py, torchfilter/types.py,
#  tests/_linear_system_models.py)
#
# "Differentiable Particle Filter" (queue rows: tu-rbo/differentiable-particle-filters
# [TF/Keras, the original RSS 2018 Jonschkowski/Hofmann/Brock repo] and
# stanford-iprl-lab/torchfilter [PyTorch] -- both queue rows name the same
# differentiable-particle-filter concept; torchfilter is the PyTorch
# implementation and is the one vendored here per the framework note. The
# TF/Keras tu-rbo repo is recorded as a dedup-skip pointing at this entry.)
#
# torchfilter is a general differentiable Bayesian-filtering library
# (particle filter / (square-root) unscented Kalman filter / extended Kalman
# filter). `ParticleFilter` (torchfilter/filters/_particle_filter.py) is
# vendored verbatim: propagate -> reweight-by-measurement-model ->
# (optionally differentiably) resample, exactly per Karkus et al.
# "Particle Filter Networks with Application to Visual Localization"
# (arXiv:1805.08975), which torchfilter's own soft_resample_alpha docstring
# cites. The dynamics/measurement submodules used to build a concrete
# instance are torchfilter's OWN test fixtures
# (tests/_linear_system_models.py: `LinearDynamicsModel`,
# `LinearParticleFilterMeasurementModel` wrapping `LinearKalmanFilterMeasurementModel`)
# -- the exact linear-Gaussian system torchfilter's own test suite exercises
# its ParticleFilter against, not an invented example.
#
# The only import-path change: the real repo's `fannypack.utils.SliceWrapper`
# (a generic list/tuple/tensor/dict slicing-and-mapping utility -- NOT part of
# the filtering architecture) is not one of the installed base libs here, so
# rather than adding an unlisted pip dependency for a pure data-plumbing
# helper, the minimal subset actually exercised by this module (`len()`,
# `.map()`, `__getitem__` over a plain `torch.Tensor` -- our concrete example
# only ever passes plain tensors as controls/observations, never dicts) is
# faithfully re-implemented below as `_SliceWrapper`, transcribed from
# fannypack==0.0.25's `fannypack/utils/_slice_wrapper.py` restricted to the
# torch.Tensor code path. No filtering/architecture logic was touched.
#
# Ref: https://github.com/stanford-iprl-lab/torchfilter/blob/master/torchfilter/filters/_particle_filter.py
# Ref: https://github.com/stanford-iprl-lab/torchfilter/blob/master/tests/_linear_system_models.py

import abc
from typing import Callable, Dict, NamedTuple, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# From torchfilter/types.py (semantic type aliases, verbatim subset).
# ---------------------------------------------------------------------------
StatesTorch = torch.Tensor
ObservationsNoDictTorch = torch.Tensor
ScaleTrilTorch = torch.Tensor
CovarianceTorch = torch.Tensor
TorchDict = Dict[str, torch.Tensor]
TorchTensorOrDict = Union[torch.Tensor, TorchDict]
ObservationsTorch = TorchTensorOrDict
ControlsTorch = TorchTensorOrDict


# ---------------------------------------------------------------------------
# Minimal faithful re-implementation of fannypack.utils.SliceWrapper,
# restricted to the plain-torch.Tensor code path (see header note). Ported
# from fannypack==0.0.25 (fannypack/utils/_slice_wrapper.py).
# ---------------------------------------------------------------------------
class _SliceWrapper:
    """Tensor-only subset of fannypack.utils.SliceWrapper: a thin wrapper
    providing __len__/__getitem__/.map()/.shape over a torch.Tensor so the
    vendored dynamics/filter code (originally written to also accept dict
    inputs) can be exercised unmodified with our concrete tensor example.
    """

    def __init__(self, data: torch.Tensor):
        assert isinstance(data, torch.Tensor)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, index):
        return self.data[index]

    def map(self, function: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return function(self.data)


# ---------------------------------------------------------------------------
# From torchfilter/base/_filter.py (Filter ABC, verbatim; `@overrides` decorator
# and fannypack.utils.SliceWrapper usages replaced 1:1 with the local
# _SliceWrapper defined above).
# ---------------------------------------------------------------------------
class Filter(nn.Module, abc.ABC):
    """Base class for a generic differentiable state estimator."""

    def __init__(self, *, state_dim: int):
        super().__init__()
        self.state_dim = state_dim

    @abc.abstractmethod
    def initialize_beliefs(self, *, mean: StatesTorch, covariance: CovarianceTorch) -> None: ...

    def forward(self, *, observations: ObservationsTorch, controls: ControlsTorch) -> StatesTorch:
        observations_wrapped = _SliceWrapper(observations)
        controls_wrapped = _SliceWrapper(controls)

        output = self.forward_loop(
            observations=observations_wrapped[None, ...],
            controls=controls_wrapped[None, ...],
        )
        assert output.shape[0] == 1
        return output[0]

    def forward_loop(
        self, *, observations: ObservationsTorch, controls: ControlsTorch
    ) -> StatesTorch:
        observations_wrapped = _SliceWrapper(observations)
        controls_wrapped = _SliceWrapper(controls)

        T, N = controls_wrapped.shape[:2]
        assert observations_wrapped.shape[:2] == (T, N)

        t = 0
        current_prediction = self(
            observations=observations_wrapped[t], controls=controls_wrapped[t]
        )
        state_predictions = current_prediction.new_zeros((T, N, self.state_dim))
        assert current_prediction.shape == (N, self.state_dim)
        state_predictions[t] = current_prediction

        for t in range(1, T):
            current_prediction = self(
                observations=observations_wrapped[t], controls=controls_wrapped[t]
            )
            assert current_prediction.shape == (N, self.state_dim)
            state_predictions[t] = current_prediction

        return state_predictions


# ---------------------------------------------------------------------------
# From torchfilter/base/_dynamics_model.py (DynamicsModel ABC, verbatim).
# ---------------------------------------------------------------------------
class DynamicsModel(nn.Module, abc.ABC):
    """Base class for a generic differentiable dynamics model, with additive
    white Gaussian noise."""

    def __init__(self, *, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim

    def forward(
        self,
        *,
        initial_states: StatesTorch,
        controls: ControlsTorch,
    ) -> Tuple[StatesTorch, ScaleTrilTorch]:
        controls_wrapped = _SliceWrapper(controls)

        predictions, scale_trils = self.forward_loop(
            initial_states=initial_states, controls=controls_wrapped[None, ...]
        )
        assert predictions.shape[0] == 1
        assert scale_trils.shape[0] == 1
        return predictions[0], scale_trils[0]

    def forward_loop(
        self, *, initial_states: StatesTorch, controls: ControlsTorch
    ) -> Tuple[StatesTorch, torch.Tensor]:
        controls_wrapped = _SliceWrapper(controls)

        T, N = controls_wrapped.shape[:2]
        assert initial_states.shape == (N, self.state_dim)
        assert T > 0

        predictions_list = []
        scale_trils_list = []

        constant_noise = True
        prediction = initial_states

        for t in range(T):
            prediction, scale_tril = self(initial_states=prediction, controls=controls_wrapped[t])

            if t >= 1 and (
                scale_tril.data_ptr() != scale_trils_list[-1].data_ptr()
                or scale_tril.stride() != scale_trils_list[-1].stride()
            ):
                constant_noise = False

            assert prediction.shape == (N, self.state_dim)
            assert scale_tril.shape == (N, self.state_dim, self.state_dim)
            predictions_list.append(prediction)
            scale_trils_list.append(scale_tril)

        predictions = torch.stack(predictions_list, dim=0)

        if constant_noise:
            scale_trils = scale_trils_list[0][None, :, :, :].expand(
                T, N, self.state_dim, self.state_dim
            )
        else:
            scale_trils = torch.stack(scale_trils_list, dim=0)

        assert predictions.shape == (T, N, self.state_dim)
        assert scale_trils.shape == (T, N, self.state_dim, self.state_dim)
        return predictions, scale_trils


# ---------------------------------------------------------------------------
# From torchfilter/base/_kalman_filter_measurement_model.py (verbatim).
# ---------------------------------------------------------------------------
class KalmanFilterMeasurementModel(abc.ABC, nn.Module):
    def __init__(self, *, state_dim, observation_dim):
        super().__init__()
        self.state_dim = state_dim
        self.observation_dim = observation_dim

    @abc.abstractmethod
    def forward(self, *, states: StatesTorch) -> Tuple[ObservationsNoDictTorch, ScaleTrilTorch]: ...


# ---------------------------------------------------------------------------
# From torchfilter/base/_particle_filter_measurement_model.py (verbatim).
# ---------------------------------------------------------------------------
class ParticleFilterMeasurementModel(abc.ABC, nn.Module):
    """Observation model base class for a generic differentiable particle
    filter; maps (state, observation) pairs to the log-likelihood of the
    observation given the state ( log p(z | x) )."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim

    @abc.abstractmethod
    def forward(self, *, states: StatesTorch, observations: ObservationsTorch) -> torch.Tensor: ...


class ParticleFilterMeasurementModelWrapper(ParticleFilterMeasurementModel):
    """Helper class for creating a particle filter measurement model (states,
    observations -> log-likelihoods) from a Kalman filter one (states ->
    observations)."""

    def __init__(self, kalman_filter_measurement_model: KalmanFilterMeasurementModel):
        super().__init__(state_dim=kalman_filter_measurement_model.state_dim)
        self.kalman_filter_measurement_model = kalman_filter_measurement_model

    def forward(self, *, states: StatesTorch, observations: ObservationsTorch) -> torch.Tensor:
        assert isinstance(observations, torch.Tensor), (
            "For wrapped Kalman filter measurement models, observations must be tensors."
        )
        observations = cast(torch.Tensor, observations)

        N, M, state_dim = states.shape
        N_alt, observation_dim = observations.shape
        assert observation_dim == self.kalman_filter_measurement_model.observation_dim
        assert N == N_alt

        pred_observations, observations_tril = self.kalman_filter_measurement_model(
            states=states.reshape((-1, state_dim))
        )
        assert pred_observations.shape == (N * M, observation_dim)
        assert observations_tril.shape == (N * M, observation_dim, observation_dim)
        pred_observations = pred_observations.reshape((N, M, observation_dim))
        observations_tril = observations_tril.reshape((N, M, observation_dim, observation_dim))

        observations = observations[:, None, :].expand((N, M, observation_dim))

        log_likelihoods = torch.distributions.MultivariateNormal(
            loc=pred_observations, scale_tril=observations_tril
        ).log_prob(observations)
        assert log_likelihoods.shape == (N, M)

        return log_likelihoods


# ---------------------------------------------------------------------------
# From torchfilter/filters/_particle_filter.py (ParticleFilter, verbatim;
# `@overrides` decorator and fannypack.utils.SliceWrapper usages replaced 1:1
# with the local _SliceWrapper defined above -- no filtering-algorithm logic
# touched).
# ---------------------------------------------------------------------------
class ParticleFilter(Filter):
    """Generic differentiable particle filter."""

    def __init__(
        self,
        *,
        dynamics_model: DynamicsModel,
        measurement_model: ParticleFilterMeasurementModel,
        num_particles: int = 100,
        resample=None,
        soft_resample_alpha: float = 1.0,
        estimation_method: str = "weighted_average",
    ):
        assert isinstance(dynamics_model, DynamicsModel)
        assert isinstance(measurement_model, ParticleFilterMeasurementModel)
        assert dynamics_model.state_dim == measurement_model.state_dim

        state_dim = dynamics_model.state_dim
        super().__init__(state_dim=state_dim)

        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model

        self.num_particles = num_particles
        self.resample = resample

        self.soft_resample_alpha = soft_resample_alpha
        assert estimation_method in ("weighted_average", "argmax")
        self.estimation_method = estimation_method

        self.particle_states: torch.Tensor
        self.particle_log_weights: torch.Tensor
        self._initialized = False

    def initialize_beliefs(self, *, mean: StatesTorch, covariance: CovarianceTorch) -> None:
        N = mean.shape[0]
        assert mean.shape == (N, self.state_dim)
        assert covariance.shape == (N, self.state_dim, self.state_dim)
        M = self.num_particles

        self.particle_states = (
            torch.distributions.MultivariateNormal(mean, covariance).sample((M,)).transpose(0, 1)
        )
        assert self.particle_states.shape == (N, M, self.state_dim)

        self.particle_log_weights = self.particle_states.new_full(
            (N, M), float(-np.log(M, dtype=np.float32))
        )
        assert self.particle_log_weights.shape == (N, M)

        self._initialized = True

    def forward(
        self,
        *,
        observations: ObservationsTorch,
        controls: ControlsTorch,
    ) -> StatesTorch:
        assert self._initialized, "Particle filter not initialized!"

        N, M, state_dim = self.particle_states.shape
        assert state_dim == self.state_dim
        assert len(_SliceWrapper(controls)) == N

        resample = self.resample
        if resample is None:
            resample = not self.training

        if not resample and self.num_particles != M:
            indices = self.particle_states.new_zeros((N, self.num_particles), dtype=torch.long)

            copy_count = (self.num_particles // M) * M
            if copy_count > 0:
                indices[:, :copy_count] = torch.arange(M).repeat(copy_count // M)[None, :]

            remaining_count = self.num_particles - copy_count
            assert remaining_count >= 0
            if remaining_count > 0:
                indices[:, copy_count:] = torch.randperm(M, device=indices.device)[
                    None, :remaining_count
                ]

            M = self.num_particles
            self.particle_states = self.particle_states.gather(
                1, indices[:, :, None].expand((N, M, state_dim))
            )
            self.particle_log_weights = self.particle_log_weights.gather(1, indices)
            assert self.particle_states.shape == (N, self.num_particles, state_dim)
            assert self.particle_log_weights.shape == (N, self.num_particles)

            self.particle_log_weights = self.particle_log_weights - torch.logsumexp(
                self.particle_log_weights, dim=1, keepdim=True
            )

        reshaped_states = self.particle_states.reshape(-1, self.state_dim)
        reshaped_controls = _SliceWrapper(controls).map(
            lambda tensor: torch.repeat_interleave(tensor, repeats=M, dim=0)
        )
        predicted_states, scale_trils = self.dynamics_model(
            initial_states=reshaped_states, controls=reshaped_controls
        )
        self.particle_states = (
            torch.distributions.MultivariateNormal(loc=predicted_states, scale_tril=scale_trils)
            .rsample()
            .view(N, M, self.state_dim)
        )
        assert self.particle_states.shape == (N, M, self.state_dim)

        self.particle_log_weights = self.particle_log_weights + self.measurement_model(
            states=self.particle_states,
            observations=observations,
        )

        self.particle_log_weights = self.particle_log_weights - torch.logsumexp(
            self.particle_log_weights, dim=1, keepdim=True
        )

        state_estimates: StatesTorch
        if self.estimation_method == "weighted_average":
            state_estimates = torch.sum(
                torch.exp(self.particle_log_weights[:, :, np.newaxis]) * self.particle_states,
                dim=1,
            )
        elif self.estimation_method == "argmax":
            best_indices = torch.argmax(self.particle_log_weights, dim=1)
            state_estimates = torch.gather(self.particle_states, dim=1, index=best_indices)
        else:
            assert False, "Unsupported estimation method!"

        if resample:
            self._resample()

        assert state_estimates.shape == (N, state_dim)
        assert self.particle_states.shape == (N, self.num_particles, state_dim)
        assert self.particle_log_weights.shape == (N, self.num_particles)

        return state_estimates

    def _resample(self) -> None:
        N, M, state_dim = self.particle_states.shape

        uniform_log_weights = self.particle_log_weights.new_full(
            (N, self.num_particles), float(-np.log(M, dtype=np.float32))
        )
        if self.soft_resample_alpha < 1.0:
            assert self.particle_log_weights.shape == (N, M)
            sample_logits = torch.logsumexp(
                torch.stack(
                    [
                        self.particle_log_weights + np.log(self.soft_resample_alpha),
                        uniform_log_weights + np.log(1.0 - self.soft_resample_alpha),
                    ],
                    dim=0,
                ),
                dim=0,
            )
            self.particle_log_weights = self.particle_log_weights - sample_logits
        else:
            sample_logits = self.particle_log_weights
            self.particle_log_weights = uniform_log_weights

        assert sample_logits.shape == (N, M)
        distribution = torch.distributions.Categorical(logits=sample_logits)
        state_indices = distribution.sample((self.num_particles,)).T
        assert state_indices.shape == (N, self.num_particles)

        self.particle_states = torch.gather(
            self.particle_states,
            dim=1,
            index=state_indices[:, :, None].expand((N, self.num_particles, state_dim)),
        )


# ---------------------------------------------------------------------------
# From tests/_linear_system_models.py (torchfilter's OWN test fixture: a
# linear-Gaussian system used to exercise ParticleFilter end-to-end).
# Verbatim except `torch.cholesky` deprecation -> `torch.linalg.cholesky`
# (functionally identical; only the un-used LinearVirtualSensorModel called
# the deprecated alias) and dropping the unused
# LinearVirtualSensorModel/get_trainable_model_error helpers, which are not
# needed to build a ParticleFilter.
# ---------------------------------------------------------------------------
_state_dim = 5
_control_dim = 3
_observation_dim = 7

torch.random.manual_seed(0)
_A = torch.empty(size=(_state_dim, _state_dim))
torch.nn.init.orthogonal_(_A, gain=1.0)

_B = torch.randn(size=(_state_dim, _control_dim))
_C = torch.randn(size=(_observation_dim, _state_dim))
_Q_tril = torch.eye(_state_dim) * 0.02
_R_tril = torch.eye(_observation_dim) * 0.05


class LinearDynamicsModel(DynamicsModel):
    """Forward model for our linear system. Maps (initial_states, controls)
    pairs to (predicted_state, uncertainty) pairs."""

    def __init__(self, trainable: bool = False):
        super().__init__(state_dim=_state_dim)
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(
        self,
        *,
        initial_states: StatesTorch,
        controls: ControlsTorch,
    ) -> Tuple[StatesTorch, ScaleTrilTorch]:
        assert isinstance(controls, torch.Tensor)
        controls = cast(torch.Tensor, controls)
        N, state_dim = initial_states.shape
        N_alt, control_dim = controls.shape
        assert _A.shape == (state_dim, state_dim)
        assert N == N_alt

        predicted_states = (_A[None, :, :] @ initial_states[:, :, None]).squeeze(-1) + (
            _B[None, :, :] @ controls[:, :, None]
        ).squeeze(-1)

        if self.trainable:
            predicted_states = predicted_states + self.output_bias

        return predicted_states, _Q_tril[None, :, :].expand((N, state_dim, state_dim))


class LinearKalmanFilterMeasurementModel(KalmanFilterMeasurementModel):
    """Kalman filter measurement model for our linear system. Maps states to
    (observation, uncertainty) pairs."""

    def __init__(self, trainable: bool = False):
        super().__init__(state_dim=_state_dim, observation_dim=_observation_dim)
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(self, *, states: StatesTorch) -> Tuple[ObservationsNoDictTorch, ScaleTrilTorch]:
        N = states.shape[0]
        assert states.shape == (N, _state_dim)

        observations = (_C[None, :, :] @ states[:, :, None]).squeeze(-1)
        scale_tril = _R_tril[None, :, :].expand((N, _observation_dim, _observation_dim))

        if self.trainable:
            observations = observations + self.output_bias

        return observations, scale_tril


class LinearParticleFilterMeasurementModel(ParticleFilterMeasurementModelWrapper):
    """Particle filter measurement model. Defined by wrapping our Kalman
    filter one."""

    def __init__(self, trainable: bool = False):
        super().__init__(
            kalman_filter_measurement_model=LinearKalmanFilterMeasurementModel(trainable=trainable)
        )


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo). Uses torchfilter's
# own linear-system test fixture (state_dim=5, control_dim=3,
# observation_dim=7) with a small num_particles for a fast CPU trace; the
# filter is put in eval() mode so its default `resample=None` behavior
# activates resampling (matches real deployment usage; train-mode disables
# resampling only to let gradients flow through time in a multi-step rollout,
# not relevant to a single forward-pass trace).
# ---------------------------------------------------------------------------
def build_torchfilter_particle_filter():
    torch.manual_seed(0)
    pf = ParticleFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearParticleFilterMeasurementModel(),
        num_particles=12,
    )
    pf.eval()
    mean = torch.zeros(1, _state_dim)
    covariance = torch.eye(_state_dim).unsqueeze(0) * 0.1
    pf.initialize_beliefs(mean=mean, covariance=covariance)
    return pf


def example_input_torchfilter_particle_filter():
    torch.manual_seed(0)
    observations = torch.randn(1, _observation_dim)
    controls = torch.randn(1, _control_dim)
    return (observations, controls)


class _ParticleFilterTraceWrapper(nn.Module):
    """Thin forward(*args)-style wrapper so tl.trace can call the filter
    positionally; the real ParticleFilter.forward is keyword-only
    (observations=, controls=), which this preserves exactly."""

    def __init__(self, pf: ParticleFilter):
        super().__init__()
        self.pf = pf

    def forward(self, observations: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        return self.pf(observations=observations, controls=controls)


def build_torchfilter_particle_filter_traceable():
    return _ParticleFilterTraceWrapper(build_torchfilter_particle_filter())


MENAGERIE_ENTRIES = [
    (
        "torchfilter.ParticleFilter (differentiable particle filter)",
        "build_torchfilter_particle_filter_traceable",
        "example_input_torchfilter_particle_filter",
        2018,
        MENAGERIE_ZOO,
    ),
]
