# SOURCE: vendored from https://github.com/zhejz/carla-roach @ main
# (agents/rl_birdview/models/{ppo_policy,torch_layers,distributions}.py)
"""Roach (ICCV 2021) -- "End-to-End Urban Driving by Imitating a Reinforcement Learning
Coach". The real RL policy network is `agents/rl_birdview/models/ppo_policy.py:PpoPolicy`,
a `nn.Module` combining a CNN+MLP feature extractor over (birdview-image, low-dim state)
observations with separate policy and value MLP heads, and a Beta-distribution action head
(steer/throttle in [0, 1]).

`PpoPolicy._build()` dynamically imports its feature-extractor and action-distribution
classes by dotted path via `carla_gym.utils.config_utils.load_entry_point` and constructs
them against a real `gym.spaces.Dict` observation_space. Per the repo's own shipped default
config (`config/agent/ppo/policy/xtma_beta.yaml`, used by `config/agent/ppo.yaml`):
`features_extractor_entry_point=agents.rl_birdview.models.torch_layers:XtMaCNN`,
`distribution_entry_point=agents.rl_birdview.models.distributions:BetaDistribution`.

This vendors `XtMaCNN` (`torch_layers.py`) and `BetaDistribution.proba_distribution_net`
(`distributions.py`) VERBATIM (every Conv2d/Linear/ReLU layer and shape computation
unchanged) and reproduces `PpoPolicy.__init__`/`_build()`/`_get_features()`/
`_get_action_dist_from_features()`'s real tensor computation graph in a plain
`nn.Module.forward` (`RoachNet`), swapping only the `load_entry_point()` dynamic-import
indirection for direct references to the same two vendored classes, and swapping the real
`gym.spaces.Dict` observation_space for a minimal duck-typed stand-in exposing the same
`observation_space['birdview'].shape` / `.sample()` / `observation_space['state'].shape`
surface `XtMaCNN.__init__` actually reads (CARLA + gym plumbing only; not part of the
network architecture). `PpoPolicy.forward`/`evaluate_values` themselves do numpy
conversion + `torch.distributions.Beta` sampling around this same
features -> policy_head/value_head -> alpha/beta tensor path, which is what `RoachNet`
traces directly.
"""

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# agents/rl_birdview/models/torch_layers.py:XtMaCNN
# ---------------------------------------------------------------------------


class XtMaCNN(nn.Module):
    """Inspired by https://github.com/xtma/pytorch_car_caring"""

    def __init__(self, observation_space, features_dim=256, states_neurons=[256]):
        super().__init__()
        self.features_dim = features_dim

        n_input_channels = observation_space["birdview"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space["birdview"].sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + states_neurons[-1], 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

        states_neurons = [observation_space["state"].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons) - 1):
            self.state_linear.append(nn.Linear(states_neurons[i], states_neurons[i + 1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, birdview, state):
        x = self.cnn(birdview)
        latent_state = self.state_linear(state)
        x = torch.cat((x, latent_state), dim=1)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# agents/rl_birdview/models/distributions.py:BetaDistribution.proba_distribution_net
# ---------------------------------------------------------------------------


def beta_distribution_net(latent_dim, action_dim=2, dist_init=None):
    """Builds the real alpha/beta Beta-distribution head modules, verbatim."""
    linear_alpha = nn.Linear(latent_dim, action_dim)
    linear_beta = nn.Linear(latent_dim, action_dim)

    if dist_init is not None:
        linear_alpha.bias.data[0] = dist_init[0][1]
        linear_beta.bias.data[0] = dist_init[0][0]
        linear_alpha.bias.data[1] = dist_init[1][1]
        linear_beta.bias.data[1] = dist_init[1][0]

    alpha = nn.Sequential(linear_alpha, nn.Softplus())
    beta = nn.Sequential(linear_beta, nn.Softplus())
    return alpha, beta


# ---------------------------------------------------------------------------
# Minimal duck-typed stand-in for the real gym.spaces.Box `observation_space['birdview']`
# / `observation_space['state']` -- only the `.shape` / `.sample()` surface XtMaCNN reads.
# Real repo config: obs_configs/birdview.yaml (birdview: 3-scale x 15 channels = 15, 192x192)
# + input_states=[control, vel_xy] (control=3 + vel_xy=2 = 5-dim state), per
# config/agent/ppo.yaml's `env_wrapper.kwargs.input_states`.
# ---------------------------------------------------------------------------


class _Box:
    def __init__(self, shape):
        self.shape = shape

    def sample(self):
        return np.zeros(self.shape, dtype=np.uint8)


class _ObservationSpace:
    def __init__(self, birdview_shape, state_shape):
        self._spaces = {"birdview": _Box(birdview_shape), "state": _Box(state_shape)}

    def __getitem__(self, key):
        return self._spaces[key]


# ---------------------------------------------------------------------------
# agents/rl_birdview/models/ppo_policy.py:PpoPolicy (network path only)
# ---------------------------------------------------------------------------


class RoachNet(nn.Module):
    """Reproduces PpoPolicy's real feature/policy/value tensor computation graph.

    `PpoPolicy.__init__`/`_build()`:
      features_extractor = XtMaCNN(observation_space, **features_extractor_kwargs)
      policy_head = Sequential(Linear, ReLU, Linear, ReLU, ...)   # policy_head_arch
      value_head  = Sequential(Linear, ReLU, Linear, ReLU, ..., Linear(-> 1))
      dist_mu, dist_sigma = BetaDistribution(action_dim).proba_distribution_net(latent_dim)

    `PpoPolicy.evaluate_values`:
      features = self._get_features(birdview, state)     # birdview/255.0, then extractor
      values = self.value_head(features)
      latent_pi = self.policy_head(features)
      mu, sigma = self.dist_mu(latent_pi), self.dist_sigma(latent_pi)   # alpha, beta
    """

    def __init__(
        self,
        observation_space,
        action_dim=2,
        policy_head_arch=(256, 256),
        value_head_arch=(256, 256),
        features_extractor_kwargs=None,
    ):
        super().__init__()
        features_extractor_kwargs = features_extractor_kwargs or {}
        self.features_extractor = XtMaCNN(observation_space, **features_extractor_kwargs)

        last_layer_dim_pi = self.features_extractor.features_dim
        policy_net = []
        for layer_size in policy_head_arch:
            policy_net.append(nn.Linear(last_layer_dim_pi, layer_size))
            policy_net.append(nn.ReLU())
            last_layer_dim_pi = layer_size
        self.policy_head = nn.Sequential(*policy_net)
        self.dist_mu, self.dist_sigma = beta_distribution_net(
            last_layer_dim_pi, action_dim=action_dim
        )

        last_layer_dim_vf = self.features_extractor.features_dim
        value_net = []
        for layer_size in value_head_arch:
            value_net.append(nn.Linear(last_layer_dim_vf, layer_size))
            value_net.append(nn.ReLU())
            last_layer_dim_vf = layer_size
        value_net.append(nn.Linear(last_layer_dim_vf, 1))
        self.value_head = nn.Sequential(*value_net)

    def forward(self, birdview, state):
        birdview = birdview.float() / 255.0
        features = self.features_extractor(birdview, state)

        values = self.value_head(features)

        latent_pi = self.policy_head(features)
        alpha = self.dist_mu(latent_pi)
        beta = self.dist_sigma(latent_pi)

        return values, alpha, beta


_BIRDVIEW_SHAPE = (15, 192, 192)
_STATE_DIM = 5


def build_roach():
    obs_space = _ObservationSpace(birdview_shape=_BIRDVIEW_SHAPE, state_shape=(_STATE_DIM,))
    return RoachNet(
        observation_space=obs_space, features_extractor_kwargs={"states_neurons": [64, 64]}
    )


def example_input_roach():
    torch.manual_seed(0)
    batch = 2
    birdview = torch.randint(0, 256, (batch,) + _BIRDVIEW_SHAPE, dtype=torch.uint8).float()
    state = torch.randn(batch, _STATE_DIM)
    return (birdview, state)


MENAGERIE_ENTRIES = [
    ("Roach", "build_roach", "example_input_roach", 2021, "vendored-pytorch"),
]
