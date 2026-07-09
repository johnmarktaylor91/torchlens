# SOURCE: vendored from https://github.com/facebookresearch/impact-driven-exploration @ main
# (src/models.py: MinigridPolicyNet, MinigridStateEmbeddingNet, MinigridForwardDynamicsNet,
#  MinigridInverseDynamicsNet, init; src/algos/ride.py: learn() composition, lines 1-46)
#
# RIDE / "Rewarding Impact-Driven Exploration for Procedurally-Generated Environments" (Raileanu &
# Rocktaschel, ICLR 2020). Official facebookresearch/impact-driven-exploration repo. RIDE's
# intrinsic-motivation signal rewards actions that produce a large change in a learned STATE
# EMBEDDING (rather than novelty/prediction-error alone), computed as the L2 "impact" distance
# between consecutive embedded states; the embedding network is trained jointly via forward- and
# inverse-dynamics losses (same self-supervised setup as ICM, but the *reward* is the embedding
# change, not the forward-model prediction error -- RIDE's defining architectural/algorithmic
# move). This module vendors the real minigrid-observation network stack used by RIDE's `learn()`:
#   - `MinigridPolicyNet`: the actor-critic -- 3-layer strided conv feature extractor over the
#     partial (7x7) grid observation, FC projection, a 2-layer LSTM core (with the training loop's
#     real per-timestep `notdone`-masked state reset baked into `forward`), and policy/baseline
#     heads.
#   - `MinigridStateEmbeddingNet`: the state-embedding tower RIDE's impact reward is computed from
#     (3-layer conv tower widened to 128 channels on the last layer).
#   - `MinigridForwardDynamicsNet` / `MinigridInverseDynamicsNet`: the self-supervised
#     forward/inverse dynamics heads used to train the state embedding (predict next-embedding
#     from (embedding, action); predict action from (embedding, next-embedding)).
# All four classes + the `init` weight-init helper are vendored verbatim from `src/models.py` (no
# import beyond `torch`/`numpy` in that file). The staging wrapper below composes them exactly as
# `ride.learn()` does: policy forward on partial_obs, state-embedding forward on the same
# partial_obs before/after one step, then forward- and inverse-dynamics heads consuming those
# embeddings + the taken action -- exercising the full real forward-pass graph in one trace.

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- src/models.py (vendored verbatim) ----
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MinigridPolicyNet(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(MinigridPolicyNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,  # noqa: E731
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        self.feat_extract = nn.Sequential(
            init_(
                nn.Conv2d(
                    in_channels=self.observation_shape[2],
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=2,
                    padding=1,
                )
            ),
            nn.ELU(),
            init_(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
            ),
            nn.ELU(),
            init_(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
            ),
            nn.ELU(),
        )

        self.fc = nn.Sequential(
            init_(nn.Linear(32, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        self.core = nn.LSTM(1024, 1024, 2)

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,  # noqa: E731
            lambda x: nn.init.constant_(x, 0),
        )

        self.policy = init_(nn.Linear(1024, self.num_actions))
        self.baseline = init_(nn.Linear(1024, 1))

    def initial_state(self, batch_size):
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs["partial_obs"]
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float()  # / 255.0

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        core_input = self.fc(x)

        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, action=action), core_state


class MinigridStateEmbeddingNet(nn.Module):
    def __init__(self, observation_shape):
        super(MinigridStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: (
                nn.init.  # noqa: E731
                constant_(x, 0)
            ),
            nn.init.calculate_gain("relu"),
        )

        self.feat_extract = nn.Sequential(
            init_(
                nn.Conv2d(
                    in_channels=self.observation_shape[2],
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=2,
                    padding=1,
                )
            ),
            nn.ELU(),
            init_(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
            ),
            nn.ELU(),
            init_(
                nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)
            ),
            nn.ELU(),
        )

    def forward(self, inputs):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float() / 255.0

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        state_embedding = x.view(T, B, -1)

        return state_embedding


class MinigridInverseDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MinigridInverseDynamicsNet, self).__init__()
        self.num_actions = num_actions

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: (
                nn.init.  # noqa: E731
                constant_(x, 0)
            ),
            nn.init.calculate_gain("relu"),
        )
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * 128, 256)),
            nn.ReLU(),
        )

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,  # noqa: E731
            lambda x: nn.init.constant_(x, 0),
        )
        self.id_out = init_(nn.Linear(256, self.num_actions))

    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits


class MinigridForwardDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MinigridForwardDynamicsNet, self).__init__()
        self.num_actions = num_actions

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: (
                nn.init.  # noqa: E731
                constant_(x, 0)
            ),
            nn.init.calculate_gain("relu"),
        )

        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(128 + self.num_actions, 256)),
            nn.ReLU(),
        )

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,  # noqa: E731
            lambda x: nn.init.constant_(x, 0),
        )

        self.fd_out = init_(nn.Linear(256, 128))

    def forward(self, state_embedding, action):
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = torch.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb


# ---- end vendored src/models.py ----


class RideAgent(nn.Module):
    """Staging wrapper exercising RIDE's real network composition from
    `src/algos/ride.py::learn()`: the policy net's actor-critic forward on
    `partial_obs`, plus the state-embedding/forward-dynamics/inverse-dynamics
    stack that produces RIDE's impact-driven intrinsic reward, in one traceable
    forward pass over a single unroll step."""

    def __init__(self, observation_shape=(7, 7, 3), num_actions=7):
        super().__init__()
        self.policy_net = MinigridPolicyNet(observation_shape, num_actions)
        self.state_embedding_net = MinigridStateEmbeddingNet(observation_shape)
        self.forward_dynamics_net = MinigridForwardDynamicsNet(num_actions)
        self.inverse_dynamics_net = MinigridInverseDynamicsNet(num_actions)
        self.num_actions = num_actions

    def forward(self, partial_obs, done, action):
        # partial_obs: (T+1, B, H, W, C); done: (T+1, B) bool; action: (T+1, B) long
        core_state = self.policy_net.initial_state(partial_obs.shape[1])
        policy_out, _ = self.policy_net({"partial_obs": partial_obs, "done": done}, core_state)

        state_emb = self.state_embedding_net(partial_obs[:-1])
        next_state_emb = self.state_embedding_net(partial_obs[1:])

        pred_next_state_emb = self.forward_dynamics_net(state_emb, action[1:])
        pred_action_logits = self.inverse_dynamics_net(state_emb, next_state_emb)

        control_rewards = torch.norm(next_state_emb - state_emb, dim=2, p=2)

        return (
            policy_out["policy_logits"],
            policy_out["baseline"],
            pred_next_state_emb,
            pred_action_logits,
            control_rewards,
        )


def build_ride():
    return RideAgent(observation_shape=(7, 7, 3), num_actions=7)


def example_input_ride():
    T, B = 4, 2
    partial_obs = torch.randint(0, 10, (T + 1, B, 7, 7, 3)).float()
    done = torch.zeros(T + 1, B, dtype=torch.bool)
    action = torch.randint(0, 7, (T + 1, B))
    return partial_obs, done, action


MENAGERIE_ENTRIES = [
    ("RIDE", "build_ride", "example_input_ride", 2020, "vendored-pytorch"),
]
