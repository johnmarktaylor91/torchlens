# SOURCE: vendored from https://github.com/deligentfool/UNREAL_pytorch @ master
# (basic_net.py: conv_net, lstm_net, policy_net, value_net, pixel_control_net,
#  reward_prediction_net, lines 1-108; unreal.py: unreal.__init__ module
#  construction, lines 1-19)
#
# UNREAL (Jaderberg et al., ICLR 2017, "Reinforcement Learning with
# Unsupervised Auxiliary Tasks", arXiv:1611.05397). The original DeepMind
# reference implementation is Lua/Torch (not public in a runnable form); the
# widely-referenced community reimplementation 404akhan/unreal-implementation
# is TensorFlow-1.x (tf.contrib), also not runnable in a modern base env.
# deligentfool/UNREAL_pytorch is a clean, faithful PyTorch port matching the
# original paper's architecture: a shared conv+LSTM A3C trunk feeding (1) a
# policy head, (2) a value head, PLUS the paper's defining auxiliary-task
# heads -- (3) a pixel-control head (deconv duelling value/advantage over
# pixel-change patches) and (4) a reward-prediction head (from stacked conv
# features of consecutive frames). All six network classes have no dependency
# beyond torch, so they are vendored verbatim. The `unreal` class's
# loss-computation methods (`main_loss`/`pc_loss`/`rp_loss`/`vr_loss`,
# A3C/replay-buffer training logic) are omitted -- they are training-loop
# logic, not part of the traced network architecture; only the six
# `nn.Module` network definitions and their real construction (as in
# `unreal.__init__`) are vendored.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- basic_net.py (vendored verbatim) ----
class conv_net(nn.Module):
    def __init__(self, observation_dim):
        super(conv_net, self).__init__()
        self.observation_dim = observation_dim

        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.observation_dim[0], 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(nn.Linear(self.fc_dim(), 256), nn.ReLU())

    def fc_dim(self):
        tmp = torch.zeros(1, *self.observation_dim)
        return self.conv_layer(tmp).view(1, -1).size(1)

    def forward(self, observation):
        x = self.conv_layer(observation)
        x = x.view(x.size(0), -1)
        conv_feature = self.fc_layer(x)
        return conv_feature


class lstm_net(nn.Module):
    def __init__(self, action_dim):
        super(lstm_net, self).__init__()
        self.action_dim = action_dim

        self.lstm_layer = nn.LSTM(256 + self.action_dim + 1, 256, 1, batch_first=True)

    def forward(self, conv_feature, action, reward, hidden=None):
        conv_feature = torch.cat([conv_feature, action, reward], 1)
        conv_feature = conv_feature.unsqueeze(0)
        if not hidden:
            h0 = torch.zeros(conv_feature.size(0), 1, 256)
            c0 = torch.zeros(conv_feature.size(0), 1, 256)
            hidden = (h0, c0)
        lstm_feature, new_hidden = self.lstm_layer(conv_feature, hidden)
        return lstm_feature.squeeze(0), new_hidden


class policy_net(nn.Module):
    def __init__(self, action_dim):
        super(policy_net, self).__init__()
        self.action_dim = action_dim

        self.policy_layer = nn.Linear(256, self.action_dim)

    def forward(self, lstm_feature):
        prob = F.softmax(self.policy_layer(lstm_feature), 1)
        return prob


class value_net(nn.Module):
    def __init__(self):
        super(value_net, self).__init__()

        self.value_layer = nn.Linear(256, 1)

    def forward(self, lstm_feature):
        value = self.value_layer(lstm_feature)
        return value


class pixel_control_net(nn.Module):
    def __init__(self, action_dim):
        super(pixel_control_net, self).__init__()
        self.action_dim = action_dim

        self.deconv_fc_layer = nn.Sequential(nn.Linear(256, 32 * 9 * 9), nn.ReLU())
        self.value_deconv_layer = nn.ConvTranspose2d(32, 1, 4, 2)
        self.advan_deconv_layer = nn.ConvTranspose2d(32, self.action_dim, 4, 2)

    def forward(self, lstm_feature):
        x = self.deconv_fc_layer(lstm_feature)
        x = x.view(x.size(0), 32, 9, 9)
        value = self.value_deconv_layer(x)
        advan = self.advan_deconv_layer(x)
        return value + advan


class reward_prediction_net(nn.Module):
    def __init__(self, stack_num=3):
        super(reward_prediction_net, self).__init__()
        self.stack_num = stack_num
        self.reward_prediction_layer = nn.Sequential(
            nn.Linear(256 * self.stack_num, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, conv_feature):
        score = self.reward_prediction_layer(conv_feature)
        return score


# ---- end vendored basic_net.py ----


class UNREALNet(nn.Module):
    """Staging wrapper exercising the real UNREAL sub-network construction
    (as in unreal.unreal.__init__) as a single traceable module: shared
    conv+LSTM trunk feeding the policy head, value head, pixel-control
    auxiliary head, and reward-prediction auxiliary head in one forward
    pass -- the defining architectural move of the paper (auxiliary tasks
    riding on the same conv trunk as the base A3C agent)."""

    def __init__(self, observation_dim=(3, 84, 84), action_dim=4):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.conv_net = conv_net(observation_dim)
        self.lstm_net = lstm_net(action_dim)
        self.policy_net = policy_net(action_dim)
        self.value_net = value_net()
        self.pixel_control_net = pixel_control_net(action_dim)
        self.reward_prediction_net = reward_prediction_net(stack_num=3)

    def forward(self, observation, action_one_hot, reward, rp_stack):
        conv_feature = self.conv_net(observation)
        lstm_feature, _ = self.lstm_net(conv_feature, action_one_hot, reward)
        probs = self.policy_net(lstm_feature)
        value = self.value_net(lstm_feature)
        pc_q = self.pixel_control_net(lstm_feature)
        rp_score = self.reward_prediction_net(rp_stack)
        return probs, value, pc_q, rp_score


def build_unreal():
    return UNREALNet(observation_dim=(3, 84, 84), action_dim=4)


def example_input_unreal():
    batch = 2
    observation = torch.randn(batch, 3, 84, 84)
    action_one_hot = torch.zeros(batch, 4)
    action_one_hot[:, 0] = 1.0
    reward = torch.zeros(batch, 1)
    # reward-prediction head consumes 3 stacked conv-feature vectors (256 each)
    rp_stack = torch.randn(batch, 256 * 3)
    return observation, action_one_hot, reward, rp_stack


MENAGERIE_ENTRIES = [
    ("UNREAL", build_unreal, example_input_unreal, 2017, "vendored-pytorch"),
]
