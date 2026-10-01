# SOURCE: vendored from https://github.com/facebookresearch/torchbeast @ main
# (torchbeast/monobeast.py: AtariNet, lines 545-635)
#
# IMPALA (Espeholt et al. 2018, "IMPALA: Scalable Distributed Deep-RL with Importance
# Weighted Actor-Learner Architectures"). The official DeepMind release
# (google-deepmind/scalable_agent) is TensorFlow 1.x. TorchBeast is Facebook AI Research's
# own official PyTorch reimplementation of IMPALA's single-machine ("monobeast") variant,
# published alongside the TorchBeast paper (arXiv:1910.03552). `AtariNet` is the real
# per-actor network used by monobeast's V-trace learner: a 3-layer CNN torso feeding an
# optional 2-layer LSTM core (conditioned on the last action and clipped last reward, as
# in the original IMPALA paper), with separate policy-logits and baseline (value) heads.
# Vendored verbatim from monobeast.py; only the module-level training/`act`/`learn`/`main`
# functions (not part of the architecture) are omitted.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- torchbeast/monobeast.py (AtariNet, vendored verbatim) ----
class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(inputs["last_action"].view(T * B), self.num_actions).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


# ---- staging wrapper ----
def build_impala_atarinet():
    # Small Atari-style observation (4-frame stack, 84x84), use_lstm=True to exercise the
    # full IMPALA recurrent core (as used by monobeast's default agent).
    return AtariNet(observation_shape=(4, 84, 84), num_actions=6, use_lstm=True)


def example_input_impala_atarinet():
    torch.manual_seed(0)
    T, B = 2, 1
    frame = torch.randint(0, 256, (T, B, 4, 84, 84), dtype=torch.uint8)
    reward = torch.zeros(T, B, dtype=torch.float32)
    done = torch.zeros(T, B, dtype=torch.bool)
    last_action = torch.zeros(T, B, dtype=torch.int64)
    inputs = dict(frame=frame, reward=reward, done=done, last_action=last_action)
    # AtariNet.forward's use_lstm branch needs a real (h, c) core_state, as monobeast's
    # `act()`/`learn()` supply via `model.initial_state(batch_size)`; the bare `()` default
    # only works when use_lstm=False.
    core_state = AtariNet(
        observation_shape=(4, 84, 84), num_actions=6, use_lstm=True
    ).initial_state(B)
    return (inputs, core_state)


MENAGERIE_ENTRIES = [
    (
        "IMPALA_AtariNet",
        "build_impala_atarinet",
        "example_input_impala_atarinet",
        2018,
        "vendored-pytorch",
    ),
]
