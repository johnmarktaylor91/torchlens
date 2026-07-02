# SOURCE: vendored from facebookresearch/drqv2 @ main
# https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
# https://github.com/facebookresearch/drqv2/blob/main/utils.py (weight_init helper only)
#
# DrQ-v2: image-based continuous-control RL (data-augmented actor-critic with a
# shared conv encoder). The real model classes (RandomShiftsAug, Encoder, Actor,
# Critic) are reproduced verbatim from drqv2.py. `weight_init` is copied verbatim
# from utils.py -- the only piece of that file the model classes actually need
# (the rest of utils.py is training-loop machinery: schedules, timers, the
# TruncatedNormal distribution used only inside DrQV2Agent.act/update, which are
# not part of the traced nn.Module graph). No architectural change.
#
# DrQV2Agent itself (in drqv2.py) is a plain Python composite (not an nn.Module)
# that owns Encoder/Actor/Critic and drives them via update_critic/update_actor.
# For tracing we add a thin nn.Module, _DrQV2Trace, that wires the three real
# submodules together the same way DrQV2Agent.update does: encode obs -> actor
# forward (mean action, eval-mode path of act()) -> critic forward on the
# resulting action. This is new *glue*, not a new architecture -- every layer
# executed is the real Encoder/Actor/Critic code above.
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        # NOTE: repr_dim in the original is hardcoded for the paper's 84x84 input
        # (32 * 35 * 35). We keep the real conv stack unmodified and compute
        # repr_dim from a dry-run so the tiny example input used for tracing
        # produces a consistent flattened size -- purely a sizing constant, not
        # an architectural change.
        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            self.repr_dim = self.convnet(dummy).view(1, -1).shape[1]

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )

        self.apply(weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        # NOTE: the original returns a TruncatedNormal(mu, std) distribution
        # object (utils.py) so DrQV2Agent.act()/update_actor() can .sample()/
        # .log_prob()/.entropy() it. TorchLens traces tensor ops, not
        # distribution objects, so for the traced forward we return (mu, std)
        # directly -- exactly the two tensors the distribution wraps, computed
        # by the identical real network code above.
        return mu, std


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class _DrQV2Trace(nn.Module):
    """Thin tracing wrapper wiring the real Encoder/Actor/Critic together the
    way DrQV2Agent.update() does: augment -> encode -> actor(mu) -> critic(mu).
    """

    def __init__(
        self,
        obs_shape=(9, 84, 84),
        action_shape=(6,),
        feature_dim=50,
        hidden_dim=1024,
        std=0.1,
    ):
        super().__init__()
        self.aug = RandomShiftsAug(pad=4)
        self.encoder = Encoder(obs_shape)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim)
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim)
        self.std = std

    def forward(self, obs):
        obs = self.aug(obs.float())
        h = self.encoder(obs)
        mu, std = self.actor(h, self.std)
        q1, q2 = self.critic(h, mu)
        return mu, q1, q2


MENAGERIE_ZOO = "vendored-pytorch"


def build_drqv2():
    # Small obs (9x9x84x84 stacked-frame convention -> shrink spatial dims for
    # trace speed) and small hidden/feature dims; the real conv stack (4x
    # Conv2d+ReLU, stride-2 then stride-1x3) and MLP trunks are unmodified.
    return _DrQV2Trace(
        obs_shape=(9, 40, 40),
        action_shape=(6,),
        feature_dim=32,
        hidden_dim=64,
        std=0.1,
    ).eval()


def example_input_drqv2():
    # Real agent input: a stack of `frame_stack` RGB frames (3 * frame_stack
    # channels), HxW spatial, uint8-valued floats in [0, 255].
    return (torch.randint(0, 256, (1, 9, 40, 40)).float(),)


MENAGERIE_ENTRIES = [
    ("DrQ-v2", "build_drqv2", "example_input_drqv2", 2021, MENAGERIE_ZOO),
]
