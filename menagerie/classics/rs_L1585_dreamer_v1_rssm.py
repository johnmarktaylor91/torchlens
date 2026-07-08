# SOURCE: vendored from https://github.com/minhphd/PyDreamerV1 @ main
# (utils/models.py: RSSM, ConvEncoder, initialize_weights, lines 1-175)
#
# DreamerV1 (Hafner et al. 2019/2020, "Dream to Control: Learning Behaviors
# by Latent Imagination," arXiv:1912.01603). minhphd/PyDreamerV1 is a clean
# from-scratch PyTorch reimplementation of the Dreamer world model (the
# official reference implementation is TensorFlow 1.x, danijar/dreamer).
# Vendored here: the `RSSM` (Recurrent State-Space Model) -- the paper's
# defining architectural contribution, combining a deterministic GRU-driven
# recurrent path (`recurrent`) with stochastic latent states inferred two
# ways: the `representation` model (posterior, conditioned on the encoded
# observation embedding -- used during training/inference from real
# observations) and the `transition` model (prior, conditioned only on the
# deterministic state -- used for imagined latent rollouts without
# observations). Also vendored: `ConvEncoder`, the 4-layer strided-conv
# image encoder feeding the RSSM's observation embedding, and
# `initialize_weights`, the repo's real Kaiming-init helper applied to both.
# The catalog's existing `dreamer_rssm_gru` row is a quarantined generic
# conv-encoder placeholder (not a real RSSM); this staging module supersedes
# it with the actual GRU+stochastic-state recurrent model.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class RSSM(nn.Module):
    """Recurrent State Space Model (RSSM): the main model used to learn the
    latent dynamics of the environment."""

    def __init__(
        self,
        stochastic_size,
        obs_embed_size,
        deterministic_size,
        hidden_size,
        action_size,
        activation=nn.ELU,
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.action_size = action_size
        self.deterministic_size = deterministic_size
        self.obs_embed_size = obs_embed_size
        self.action_size = action_size

        # recurrent
        self.recurrent_linear = nn.Sequential(
            nn.Linear(stochastic_size + action_size, hidden_size),
            activation(),
        )
        self.gru_cell = nn.GRUCell(hidden_size, deterministic_size)

        # representation model, for calculating posterior
        self.representatio_model = nn.Sequential(
            nn.Linear(deterministic_size + obs_embed_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, stochastic_size * 2),
        )

        # transition model, for calculating prior, use for imagining trajectories
        self.transition_model = nn.Sequential(
            nn.Linear(deterministic_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, stochastic_size * 2),
        )

    def recurrent(self, stoch_state, action, deterministic):
        """The recurrent model: calculate the deterministic state given the
        prior stochastic state, the action, and the prior deterministic."""
        x = torch.cat((action, stoch_state), -1)
        out = self.recurrent_linear(x)
        out = self.gru_cell(out, deterministic)
        return out

    def representation(self, embed_obs, deterministic):
        """Calculate the posterior distribution of the stochastic state,
        conditioned on the embedded observation."""
        x = torch.cat((embed_obs, deterministic), -1)
        out = self.representatio_model(x)
        mean, std = torch.chunk(out, 2, -1)
        std = F.softplus(std) + 0.1

        post_dist = torch.distributions.Normal(mean, std)
        post = post_dist.rsample()

        return post_dist, post

    def transition(self, deterministic):
        """Calculate the prior distribution of the stochastic state, used
        for imagining trajectories without observations."""
        out = self.transition_model(deterministic)
        mean, std = torch.chunk(out, 2, -1)
        std = F.softplus(std) + 0.1

        prior_dist = torch.distributions.Normal(mean, std)
        prior = prior_dist.rsample()
        return prior_dist, prior

    def forward(self, prev_stoch, action, prev_deterministic, embed_obs):
        deterministic = self.recurrent(prev_stoch, action, prev_deterministic)
        post_dist, post = self.representation(embed_obs, deterministic)
        return post, deterministic


class ConvEncoder(nn.Module):
    def __init__(self, depth=32, input_shape=(3, 64, 64), activation=nn.ReLU):
        super().__init__()
        self.depth = depth
        self.input_shape = input_shape
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=depth * 1,
                kernel_size=4,
                stride=2,
                padding="valid",
            ),
            activation(),
            nn.Conv2d(
                in_channels=depth * 1,
                out_channels=depth * 2,
                kernel_size=4,
                stride=2,
                padding="valid",
            ),
            activation(),
            nn.Conv2d(
                in_channels=depth * 2,
                out_channels=depth * 4,
                kernel_size=4,
                stride=2,
                padding="valid",
            ),
            activation(),
            nn.Conv2d(
                in_channels=depth * 4,
                out_channels=depth * 8,
                kernel_size=4,
                stride=2,
                padding="valid",
            ),
            activation(),
        )
        self.conv_layer.apply(initialize_weights)

    def forward(self, x):
        batch_shape = x.shape[: -len(self.input_shape)]
        if not batch_shape:
            batch_shape = (1,)

        x = x.reshape(-1, *self.input_shape)

        out = self.conv_layer(x)

        # flatten output
        return out.reshape(*batch_shape, -1)


class DreamerV1RSSMNet(nn.Module):
    """Staging wrapper exercising the real RSSM + ConvEncoder together in a
    single traceable forward pass: encode an image observation, then run one
    RSSM step (recurrent transition + posterior representation), matching
    the per-timestep computation `algos/dreamer.py`'s rollout/training loop
    performs at every environment step."""

    def __init__(
        self,
        stochastic_size=16,
        deterministic_size=32,
        hidden_size=32,
        action_size=4,
        depth=4,
        image_size=32,
    ):
        super().__init__()
        self.encoder = ConvEncoder(depth=depth, input_shape=(3, image_size, image_size))
        obs_embed_size = self._probe_embed_size(depth, image_size)
        self.rssm = RSSM(
            stochastic_size=stochastic_size,
            obs_embed_size=obs_embed_size,
            deterministic_size=deterministic_size,
            hidden_size=hidden_size,
            action_size=action_size,
        )
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

    def _probe_embed_size(self, depth, image_size):
        probe = ConvEncoder(depth=depth, input_shape=(3, image_size, image_size))
        with torch.no_grad():
            out = probe(torch.zeros(1, 3, image_size, image_size))
        return out.shape[-1]

    def forward(self, image, action, prev_stoch, prev_deterministic):
        embed = self.encoder(image)
        post, deterministic = self.rssm(prev_stoch, action, prev_deterministic, embed)
        prior_dist, prior = self.rssm.transition(deterministic)
        return post, deterministic, prior


def build_dreamer_v1_rssm():
    return DreamerV1RSSMNet(
        stochastic_size=16,
        deterministic_size=32,
        hidden_size=32,
        action_size=4,
        depth=4,
        image_size=64,
    )


def example_input_dreamer_v1_rssm():
    image = torch.randn(2, 3, 64, 64)
    action = torch.zeros(2, 4)
    prev_stoch = torch.zeros(2, 16)
    prev_deterministic = torch.zeros(2, 32)
    return image, action, prev_stoch, prev_deterministic


MENAGERIE_ENTRIES = [
    (
        "DreamerV1 RSSM",
        "build_dreamer_v1_rssm",
        "example_input_dreamer_v1_rssm",
        2019,
        "vendored-pytorch",
    ),
]
