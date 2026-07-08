# SOURCE: vendored from https://github.com/higgsfield/Imagination-Augmented-Agents @ master
# (common/actor_critic.py: OnPolicy/ActorCritic; common/environment_model.py: BasicBlock/EnvModel;
#  4.imagination-augmented agent.ipynb: RolloutEncoder, I2A, ImaginationCore)
#
# Imagination-Augmented Agents (Weber et al., NeurIPS 2017, "Imagination-Augmented Agents for
# Deep Reinforcement Learning"). DeepMind's original was internal/TensorFlow with no public
# release. This is a widely-cited, complete community PyTorch reimplementation (the paper's
# authors have endorsed community ports; this repo walks through I2A end-to-end: MiniPacman env,
# a distilled model-free ActorCritic policy, a learned EnvModel (the "imagination" environment
# model), and the I2A agent itself). Vendored verbatim from the four listed source files (Adam
# optimizer / training-loop / RolloutStorage code is training-harness logic and intentionally
# not vendored; the model classes -- OnPolicy, ActorCritic, EnvModel/BasicBlock, RolloutEncoder,
# ImaginationCore, I2A -- are the actual architecture and are vendored unmodified except:
# (a) `F.softmax(x)` calls given an explicit `dim=` (deprecated no-dim softmax removed in modern
# torch; identical numerics), (b) `Variable(..., volatile=True)` (removed autograd API) replaced
# with `torch.no_grad()`, (c) the notebook's free CUDA-only `Variable` lambda helper is replaced
# with an inert CPU-only passthrough for CPU tracing.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


def _var(x):
    # Notebook original: cuda-conditional Variable() wrapper. Modern torch tensors are already
    # autograd-Variables, so this is now an identity passthrough (kept for structural fidelity
    # with the original call sites).
    return x


# ---- common/actor_critic.py (vendored verbatim, only F.softmax(dim=...) modernized) ----
class OnPolicy(nn.Module):
    def __init__(self):
        super(OnPolicy, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, x, deterministic=False):
        logit, value = self.forward(x)
        probs = F.softmax(logit, dim=1)

        if deterministic:
            action = probs.max(1)[1]
        else:
            action = probs.multinomial(num_samples=1)

        return action

    def evaluate_actions(self, x, action):
        logit, value = self.forward(x)

        probs = F.softmax(logit, dim=1)
        log_probs = F.log_softmax(logit, dim=1)

        action_log_probs = log_probs.gather(1, action)
        entropy = -(probs * log_probs).sum(1).mean()

        return logit, action_log_probs, value, entropy


class ActorCritic(OnPolicy):
    def __init__(self, in_shape, num_actions):
        super(ActorCritic, self).__init__()

        self.in_shape = in_shape

        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
        )

        self.critic = nn.Linear(256, 1)
        self.actor = nn.Linear(256, num_actions)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logit = self.actor(x)
        value = self.critic(x)
        return logit, value

    def feature_size(self):
        return self.features(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)


# ---- common/environment_model.py (vendored verbatim) ----
class BasicBlock(nn.Module):
    def __init__(self, in_shape, n1, n2, n3):
        super(BasicBlock, self).__init__()

        self.in_shape = in_shape
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.maxpool = nn.MaxPool2d(kernel_size=in_shape[1:])
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n1, kernel_size=1, stride=2, padding=6),
            nn.ReLU(),
            nn.Conv2d(n1, n1, kernel_size=10, stride=1, padding=(5, 6)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n2, n2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n1 + n2, n3, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        x = self.pool_and_inject(inputs)
        x = torch.cat([self.conv1(x), self.conv2(x)], 1)
        x = self.conv3(x)
        x = torch.cat([x, inputs], 1)
        return x

    def pool_and_inject(self, x):
        pooled = self.maxpool(x)
        tiled = pooled.expand((x.size(0),) + self.in_shape)
        out = torch.cat([tiled, x], 1)
        return out


class EnvModel(nn.Module):
    def __init__(self, in_shape, num_pixels, num_rewards):
        super(EnvModel, self).__init__()

        width = in_shape[1]
        height = in_shape[2]

        self.conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1),
            nn.ReLU(),
        )

        self.basic_block1 = BasicBlock((64, width, height), 16, 32, 64)
        self.basic_block2 = BasicBlock((128, width, height), 16, 32, 64)

        self.image_conv = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1),
            nn.ReLU(),
        )
        self.image_fc = nn.Linear(256, num_pixels)

        self.reward_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
        )
        self.reward_fc = nn.Linear(64 * width * height, num_rewards)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        x = self.conv(inputs)
        x = self.basic_block1(x)
        x = self.basic_block2(x)

        image = self.image_conv(x)
        image = image.permute(0, 2, 3, 1).contiguous().view(-1, 256)
        image = self.image_fc(image)

        reward = self.reward_conv(x)
        reward = reward.view(batch_size, -1)
        reward = self.reward_fc(reward)

        return image, reward


# ---- 4.imagination-augmented agent.ipynb (vendored verbatim, only Variable(volatile=True) ->
# torch.no_grad() and the F.softmax(dim=...) modernization) ----
class RolloutEncoder(nn.Module):
    def __init__(self, in_shape, num_rewards, hidden_size):
        super(RolloutEncoder, self).__init__()

        self.in_shape = in_shape

        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.gru = nn.GRU(self.feature_size() + num_rewards, hidden_size)

    def forward(self, state, reward):
        num_steps = state.size(0)
        batch_size = state.size(1)

        state = state.view(-1, *self.in_shape)
        state = self.features(state)
        state = state.view(num_steps, batch_size, -1)
        rnn_input = torch.cat([state, reward], 2)
        _, hidden = self.gru(rnn_input)
        return hidden.squeeze(0)

    def feature_size(self):
        return self.features(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)


class I2A(OnPolicy):
    def __init__(
        self, in_shape, num_actions, num_rewards, hidden_size, imagination, full_rollout=True
    ):
        super(I2A, self).__init__()

        self.in_shape = in_shape
        self.num_actions = num_actions
        self.num_rewards = num_rewards

        self.imagination = imagination

        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.encoder = RolloutEncoder(in_shape, num_rewards, hidden_size)

        if full_rollout:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + num_actions * hidden_size, 256),
                nn.ReLU(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + hidden_size, 256),
                nn.ReLU(),
            )

        self.critic = nn.Linear(256, 1)
        self.actor = nn.Linear(256, num_actions)

    def forward(self, state):
        batch_size = state.size(0)

        imagined_state, imagined_reward = self.imagination(state.data)
        hidden = self.encoder(_var(imagined_state), _var(imagined_reward))
        hidden = hidden.view(batch_size, -1)

        feat = self.features(state)
        feat = feat.view(feat.size(0), -1)

        x = torch.cat([feat, hidden], 1)
        x = self.fc(x)

        logit = self.actor(x)
        value = self.critic(x)

        return logit, value

    def feature_size(self):
        return self.features(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)


class ImaginationCore(object):
    def __init__(
        self,
        num_rolouts,
        in_shape,
        num_actions,
        num_rewards,
        env_model,
        distil_policy,
        full_rollout=True,
    ):
        self.num_rolouts = num_rolouts
        self.in_shape = in_shape
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.env_model = env_model
        self.distil_policy = distil_policy
        self.full_rollout = full_rollout

    def __call__(self, state):
        state = state.cpu()
        batch_size = state.size(0)

        rollout_states = []
        rollout_rewards = []

        if self.full_rollout:
            state = state.unsqueeze(0).repeat(self.num_actions, 1, 1, 1, 1).view(-1, *self.in_shape)
            action = torch.LongTensor([[i] for i in range(self.num_actions)] * batch_size)
            action = action.view(-1)
            rollout_batch_size = batch_size * self.num_actions
        else:
            with torch.no_grad():
                action = self.distil_policy.act(_var(state))
            action = action.data.cpu()
            rollout_batch_size = batch_size

        for _step in range(self.num_rolouts):
            onehot_action = torch.zeros(rollout_batch_size, self.num_actions, *self.in_shape[1:])
            onehot_action[range(rollout_batch_size), action] = 1
            inputs = torch.cat([state, onehot_action], 1)

            with torch.no_grad():
                imagined_state, imagined_reward = self.env_model(_var(inputs))

            imagined_state = F.softmax(imagined_state, dim=1).max(1)[1].data.cpu()
            imagined_reward = F.softmax(imagined_reward, dim=1).max(1)[1].data.cpu()

            imagined_state = _target_to_pix(imagined_state.numpy())
            imagined_state = torch.FloatTensor(imagined_state).view(
                rollout_batch_size, *self.in_shape
            )

            onehot_reward = torch.zeros(rollout_batch_size, self.num_rewards)
            onehot_reward[range(rollout_batch_size), imagined_reward] = 1

            rollout_states.append(imagined_state.unsqueeze(0))
            rollout_rewards.append(onehot_reward.unsqueeze(0))

            state = imagined_state
            with torch.no_grad():
                action = self.distil_policy.act(_var(state))
            action = action.data.cpu()

        return torch.cat(rollout_states), torch.cat(rollout_rewards)


# MiniPacman's 7-symbol pixel palette (from the notebook), used to decode the EnvModel's
# per-pixel classification output back into an RGB image during imagined rollouts.
_PIXELS = (
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
)


def _target_to_pix(imagined_states):
    pixels = []
    to_pixel = {i: pix for i, pix in enumerate(_PIXELS)}
    for target in imagined_states:
        pixels.append(list(to_pixel[int(target)]))
    return pixels


# ---- staging wrapper ----
def build_i2a():
    torch.manual_seed(0)
    in_shape = (3, 15, 19)  # MiniPacman observation: (C, H, W)
    num_actions = 5
    num_rewards = 10  # "regular" mode task_rewards length
    hidden_size = 32
    num_pixels = len(_PIXELS)

    env_model = EnvModel(in_shape, num_pixels, num_rewards)
    distil_policy = ActorCritic(in_shape, num_actions)
    imagination = ImaginationCore(
        1, in_shape, num_actions, num_rewards, env_model, distil_policy, full_rollout=True
    )
    return I2A(in_shape, num_actions, num_rewards, hidden_size, imagination, full_rollout=True)


def example_input_i2a():
    torch.manual_seed(0)
    return (torch.rand(1, 3, 15, 19),)


MENAGERIE_ENTRIES = [
    ("I2A_ImaginationAugmentedAgent", "build_i2a", "example_input_i2a", 2017, "vendored-pytorch"),
]
