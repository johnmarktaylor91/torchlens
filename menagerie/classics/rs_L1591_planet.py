# SOURCE: vendored from https://github.com/Kaixhin/PlaNet @ 28c8491bc01e8f1b911300749e04c308c03db051
# (models.py: TransitionModel, VisualEncoder, VisualObservationModel,
# RewardModel, lines 1-190)
#
# PlaNet -- "Learning Latent Dynamics for Planning from Pixels" (Hafner et al.
# 2019, arXiv:1811.04551). The paper's own reference code
# (google-research/planet) is TensorFlow 1.x graph-mode (functional
# `tf.layers`/`tensorflow_probability` style, no installable OO model
# classes). Kaixhin/PlaNet is the well-known, widely-cited community PyTorch
# reimplementation of PlaNet's Recurrent State-Space Model (RSSM) and is the
# canonical PyTorch reference used across world-model reimplementations
# (DreamerV1/V2/V3 lineage cites it). The four model classes vendored here
# have no dependency beyond torch, so they are vendored verbatim EXCEPT for
# one minimal, non-architectural fix required by TorchLens's own anti-pattern
# policy (log the eager source module, not TorchScript artifacts): the
# original classes subclass `torch.jit.ScriptModule` and decorate `forward`
# with `@torch.jit.script_method` for training-time speed; those two
# decorations are stripped here (subclass plain `nn.Module`, `forward` is a
# regular Python method) with the method BODIES left untouched -- this is a
# de-scripting of the same real ops/control-flow, not an architecture
# rewrite. Architecture: `VisualEncoder` (4-layer strided Conv2d image
# encoder) -> `TransitionModel` (the RSSM: GRUCell deterministic belief +
# stochastic prior/posterior Gaussian latent state, unrolled over a time
# sequence with reparameterized sampling) -> `VisualObservationModel`
# (4-layer strided ConvTranspose2d image decoder) + `RewardModel` (MLP scalar
# reward head from belief+state). This is PlaNet's defining architectural
# contribution (the RSSM), not a generic conv autoencoder.

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- models.py (vendored, de-scripted: jit.ScriptModule -> nn.Module) ----
class TransitionModel(nn.Module):
    def __init__(
        self,
        belief_size,
        state_size,
        action_size,
        hidden_size,
        embedding_size,
        activation_function="relu",
        min_std_dev=0.1,
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    def forward(self, prev_state, actions, prev_belief, observations=None, nonterminals=None):
        T = actions.size(0) + 1
        (
            beliefs,
            prior_states,
            prior_means,
            prior_std_devs,
            posterior_states,
            posterior_means,
            posterior_std_devs,
        ) = (
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
        )
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        # Loop over time sequence
        for t in range(T - 1):
            _state = (
                prior_states[t] if observations is None else posterior_states[t]
            )  # Select appropriate previous state
            _state = (
                _state if nonterminals is None else _state * nonterminals[t]
            )  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(
                prior_means[t + 1]
            )
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(
                        torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)
                    )
                )
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(
                    self.fc_state_posterior(hidden), 2, dim=1
                )
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[
                    t + 1
                ] * torch.randn_like(posterior_means[t + 1])
        # Return new hidden states
        hidden = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        if observations is not None:
            hidden += [
                torch.stack(posterior_states[1:], dim=0),
                torch.stack(posterior_means[1:], dim=0),
                torch.stack(posterior_std_devs[1:], dim=0),
            ]
        return hidden


class VisualObservationModel(nn.Module):
    def __init__(self, belief_size, state_size, embedding_size, activation_function="relu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


class RewardModel(nn.Module):
    def __init__(self, belief_size, state_size, hidden_size, activation_function="relu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward


class VisualEncoder(nn.Module):
    def __init__(self, embedding_size, activation_function="relu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


# ---- end vendored models.py ----


class PlaNetRSSM(nn.Module):
    """Staging wrapper exercising PlaNet's real per-component construction in
    one traceable module: VisualEncoder observes a short image sequence,
    TransitionModel unrolls the RSSM (belief + prior/posterior latent state)
    over that sequence conditioned on actions, then VisualObservationModel and
    RewardModel decode the final belief/state -- matching main.py's real
    encoder -> transition-model -> observation/reward-model pipeline at
    reduced size (belief=32, state=16, hidden=32, embedding=64, T=3)."""

    def __init__(
        self, belief_size=32, state_size=16, action_size=4, hidden_size=32, embedding_size=64
    ):
        super().__init__()
        self.belief_size = belief_size
        self.state_size = state_size
        self.encoder = VisualEncoder(embedding_size)
        self.transition = TransitionModel(
            belief_size, state_size, action_size, hidden_size, embedding_size
        )
        self.observation_model = VisualObservationModel(belief_size, state_size, embedding_size)
        self.reward_model = RewardModel(belief_size, state_size, hidden_size)

    def forward(self, images, actions, prev_belief, prev_state):
        # images: (T, B, 3, 64, 64); actions: (T-1, B, action_size)
        T, B = images.size(0), images.size(1)
        flat_images = images.view(T * B, *images.shape[2:])
        embeddings = self.encoder(flat_images).view(T, B, -1)
        hidden = self.transition(prev_state, actions, prev_belief, embeddings)
        beliefs, posterior_states = hidden[0], hidden[4]
        final_belief, final_state = beliefs[-1], posterior_states[-1]
        recon = self.observation_model(final_belief, final_state)
        reward = self.reward_model(final_belief, final_state)
        return recon, reward


def build_planet_rssm():
    return PlaNetRSSM(
        belief_size=32, state_size=16, action_size=4, hidden_size=32, embedding_size=64
    )


def example_input_planet_rssm():
    T, B, action_size = 3, 2, 4
    images = torch.randn(T, B, 3, 64, 64)
    actions = torch.randn(T - 1, B, action_size)
    prev_belief = torch.zeros(B, 32)
    prev_state = torch.zeros(B, 16)
    return (images, actions, prev_belief, prev_state)


MENAGERIE_ENTRIES = [
    ("PlaNet (RSSM)", build_planet_rssm, example_input_planet_rssm, 2019, "vendored-pytorch"),
]
