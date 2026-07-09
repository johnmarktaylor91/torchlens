# SOURCE: vendored from https://github.com/mila-iqia/spr @ master
# (src/model_trainer.py: MCTSModel, RepNet, TransitionModel, FiLMTransitionModel, ConvFiLM,
#  FiLMResidualBlock, ResidualBlock, Conv2dSame, QNetwork, ValueNetwork, PolicyNetwork,
#  SmallEncoder, renormalize/transform/inverse_transform/to_categorical/from_categorical,
#  init/init_small)
#
# SPR: Self-Predictive Representations (Schwarzer, Anand, Goel, Hjelm, Courville, Bachman;
# "Data-Efficient Reinforcement Learning with Self-Predictive Representations", ICLR 2021).
# Official Mila repo. The real MCTSModel network (a MuZero-style latent-dynamics model: conv
# RepNet encoder -> residual-block TransitionModel dynamics -> categorical
# ValueNetwork/PolicyNetwork heads, trained with the SPR self-supervised consistency loss on
# top) is vendored verbatim below, with only two minimal import substitutions that carry no
# architectural change: (a) `NetworkOutput = namedarraytuple(...)` (rlpyt) replaced with a
# plain stdlib `collections.namedtuple` with the identical field names/semantics -- rlpyt is a
# full distributed-RL training framework and is not needed to construct or run the real
# nn.Module network; (b) `MCTSModel.forward` -- the buffer-sampling / distributed-training-loop
# method that samples from an rlpyt replay buffer and computes the multi-step SPR/NCE loss -- is
# NOT vendored (it is training-harness orchestration, not architecture, and it hard-depends on
# rlpyt buffer objects and an `args` config with dozens of training-only fields). The real
# forward INFERENCE path used at evaluation/acting time --
# `MCTSModel.initial_inference(obs, actions)` -- IS vendored and is what this staging module
# traces: encoder(RepNet) -> policy_model(PolicyNetwork) + value_model(ValueNetwork) +
# dynamics_model.reward_predictor(ValueNetwork). All classes below (SmallEncoder, ConvFiLM,
# FiLMTransitionModel, FiLMResidualBlock, QNetwork) are still vendored verbatim for
# completeness/fidelity even though the default (non-FiLM, non-q_learning, no_nce=True) config
# used by build_spr_mctsmodel() below does not instantiate them.

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import namedtuple

MENAGERIE_ZOO = "vendored-pytorch"

# Original: NetworkOutput = namedarraytuple('NetworkOutput', ['next_state', 'reward', 'policy_logits', 'value'])
# rlpyt's namedarraytuple adds array-indexing sugar on top of collections.namedtuple; the plain
# namedtuple below has identical field access semantics for the (non-indexed) use in this module.
NetworkOutput = namedtuple("NetworkOutput", ["next_state", "reward", "policy_logits", "value"])


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


init_small = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: (
        nn.init.  # noqa: E731
        constant_(x, 0)
    ),
    gain=0.01,
)
init_0 = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: (
        nn.init.  # noqa: E731
        constant_(x, 0)
    ),
)
init_relu = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: (
        nn.init.  # noqa: E731
        constant_(x, 0)
    ),
    nn.init.calculate_gain("relu"),
)


class SmallEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_size = args.hidden_size
        self.input_channels = 1 if args.grayscale else 3
        self.args = args
        init_ = lambda m: init(
            m,  # noqa: E731
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        self.main = nn.Sequential(
            init_(nn.Conv2d(self.input_channels, 32, 8, stride=2, padding=3)),  # 48x48
            nn.ReLU(),
            nn.BatchNorm2d(32),
            init_(nn.Conv2d(32, 64, 4, stride=2, padding=1)),  # 24x24
            nn.ReLU(),
            nn.BatchNorm2d(64),
            init_(nn.Conv2d(64, 128, 4, stride=2, padding=1)),  # 12 x 12
            nn.ReLU(),
            nn.BatchNorm2d(128),
            init_(nn.Conv2d(128, self.feature_size, 4, stride=2, padding=1)),  # 6 x 6
            nn.ReLU(),
            init_(nn.Conv2d(self.feature_size, self.feature_size, 1, stride=1, padding=0)),
            nn.ReLU(),
        )
        self.train()

    def forward(self, inputs):
        fmaps = self.main(inputs)
        return fmaps


class TransitionModel(nn.Module):
    def __init__(
        self,
        channels,
        num_actions,
        args=None,
        blocks=16,
        hidden_size=256,
        latent_size=36,
        action_dim=6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.args = args
        layers = [
            Conv2dSame(channels + action_dim, hidden_size, 3),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
        ]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size, hidden_size))
        layers.extend([Conv2dSame(hidden_size, channels, 3), nn.ReLU()])

        self.action_embedding = nn.Embedding(num_actions, latent_size * action_dim)

        self.network = nn.Sequential(*layers)
        self.reward_predictor = ValueNetwork(channels)
        self.train()

    def _make_layer(self, in_channels, depth):
        return nn.Sequential(
            Conv2dSame(in_channels, depth, 3),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            ResidualBlock(depth, depth),
            nn.ReLU(),
            ResidualBlock(depth, depth),
        )

    def forward(self, x, action):
        action_embedding = self.action_embedding(action).view(
            x.shape[0], -1, x.shape[-2], x.shape[-1]
        )
        stacked_image = torch.cat([x, action_embedding], 1)
        next_state = self.network(stacked_image)
        next_state = renormalize(next_state, 1)
        next_reward = self.reward_predictor(next_state)
        return next_state, next_reward


class ConvFiLM(nn.Module):
    def __init__(self, input_dim, cond_dim, bn=False, one_hot=True):
        super().__init__()
        if one_hot:
            self.embedding = nn.Embedding(cond_dim, cond_dim)
        else:
            self.embedding = nn.Identity()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.conditioning = nn.Linear(cond_dim, input_dim * 2)
        self.bn = nn.BatchNorm2d(input_dim, affine=False) if bn else nn.Identity()

    def forward(self, input, cond):
        cond = self.embedding(cond)
        conditioning = self.conditioning(cond)
        gamma = conditioning[..., : self.input_dim, None, None]
        beta = conditioning[..., self.input_dim :, None, None]
        input = self.bn(input)

        return input * gamma + beta


def renormalize(tensor, first_dim=1):
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min) / (max - min)

    return flat_tensor.view(*tensor.shape)


class FiLMTransitionModel(nn.Module):
    def __init__(
        self,
        channels,
        cond_size,
        args,
        blocks=16,
        hidden_size=256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.args = args
        layers = nn.ModuleList()
        layers.append(Conv2dSame(channels, hidden_size, 3))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(hidden_size))
        for _ in range(blocks):
            layers.append(FiLMResidualBlock(hidden_size, hidden_size, cond_size))
        layers.extend([Conv2dSame(hidden_size, channels, 3), nn.ReLU()])

        self.network = nn.Sequential(*layers)
        self.reward_predictor = ValueNetwork(channels)
        self.train()

    def forward(self, x, action):
        action = action.view(
            x.shape[0],
        )
        x = self.network[:3](x)
        for resblock in self.network[3:-2]:
            x = resblock(x, action)
        next_state = self.network[-1](x)
        next_state = renormalize(next_state, 1)
        next_reward = self.reward_predictor(next_state)
        return next_state, next_reward


class FiLMResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_size):
        super().__init__()
        self.film = ConvFiLM(out_channels, cond_size, bn=True)
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            Conv2dSame(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, a):
        residual = x
        out = self.film(x, a)
        out = self.block(out)
        out += residual
        out = F.relu(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            Conv2dSame(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class Conv2dSame(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d
    ):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class QNetwork(nn.Module):
    def __init__(self, input_channels, num_actions, hidden_size=128, pixels=36, limit=300):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [
            nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
            nn.Flatten(-3, -1),
            nn.Linear(pixels * hidden_size, 512),
            nn.ReLU(),
            init_small(nn.Linear(512, num_actions * (limit * 2 + 1))),
        ]
        self.network = nn.Sequential(*layers)
        self.num_actions = num_actions
        self.dist_size = limit * 2 + 1
        self.train()

    def forward(self, x):
        distributions = self.network(x).view(*(x.shape[:-3]), self.num_actions, self.dist_size)
        return distributions


class ValueNetwork(nn.Module):
    def __init__(self, input_channels, hidden_size=128, pixels=36, limit=300):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [
            nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
            nn.Flatten(-3, -1),
            nn.Linear(pixels * hidden_size, 256),
            nn.ReLU(),
            init_small(nn.Linear(256, limit * 2 + 1)),
        ]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)


class PolicyNetwork(nn.Module):
    def __init__(self, input_channels, num_actions, hidden_size=128, pixels=36):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [
            Conv2dSame(input_channels, hidden_size, 3),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
            nn.Flatten(-3, -1),
            init_small(nn.Linear(pixels * hidden_size, num_actions)),
        ]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)


class RepNet(nn.Module):
    def __init__(self, framestack=32, grayscale=False, actions=True):
        super().__init__()
        self.input_channels = framestack * (1 if grayscale else 3)
        self.actions = actions
        if self.actions:
            self.input_channels += framestack
        layers = nn.ModuleList()
        hidden_channels = 128
        layers.append(
            nn.Conv2d(self.input_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        )
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(hidden_channels))
        for _ in range(2):
            layers.append(ResidualBlock(hidden_channels, hidden_channels))
        layers.append(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1)
        )
        hidden_channels = hidden_channels * 2
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(hidden_channels))
        for _ in range(3):
            layers.append(ResidualBlock(hidden_channels, hidden_channels))
        layers.append(nn.AvgPool2d(2))
        for _ in range(3):
            layers.append(ResidualBlock(hidden_channels, hidden_channels))
        layers.append(nn.AvgPool2d(2))
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x, actions=None):
        if self.actions:
            actions = actions[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
            stacked_image = torch.cat([x, actions], 1)
        else:
            stacked_image = x
        latent = self.network(stacked_image)
        return renormalize(latent, 1)

    def conv_out_size(self, h, w):
        return (6, 6)


def transform(value, eps=0.001):
    value = value.float()  # Avoid any fp16 shenanigans
    value = torch.sign(value) * (torch.sqrt(torch.abs(value) + 1) - 1) + eps * value
    return value


def inverse_transform(value, eps=0.001):
    value = value.float()  # Avoid any fp16 shenanigans
    return torch.sign(value) * (
        ((torch.sqrt(1 + 4 * eps * (torch.abs(value) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
    )


def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit * 2 + 1), device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


def from_categorical(distribution, limit=300, logits=True):
    distribution = distribution.float()  # Avoid any fp16 shenanigans
    if logits:
        distribution = torch.softmax(distribution, -1)
    weights = torch.arange(-limit, limit + 1, device=distribution.device).float()
    return distribution @ weights


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        ctx.constant = constant
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.constant, None


class MCTSModel(nn.Module):
    def __init__(self, args, num_actions):
        super().__init__()
        self.args = args
        self.jumps = args.jumps
        self.multistep = args.multistep
        self.use_all_targets = args.use_all_targets
        self.no_nce = args.no_nce
        self.total_steps = 0

        self.batch_range = torch.arange(args.batch_size_per_worker).to(self.args.device)
        if args.film:
            self.dynamics_model = FiLMTransitionModel(
                channels=args.hidden_size,
                cond_size=num_actions,
                blocks=args.dynamics_blocks,
                args=args,
            )
        else:
            self.dynamics_model = TransitionModel(
                channels=args.hidden_size,
                num_actions=num_actions,
                blocks=args.dynamics_blocks,
                args=args,
            )
        if self.args.q_learning:
            self.value_model = QNetwork(args.hidden_size, num_actions)
        else:
            self.value_model = ValueNetwork(args.hidden_size)
            self.policy_model = PolicyNetwork(args.hidden_size, num_actions)
        self.encoder = RepNet(args.framestack, grayscale=args.grayscale, actions=False)
        if not self.no_nce:
            self.target_encoder = SmallEncoder(args)
            self.classifier = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size), nn.ReLU()
            )
            # BlockNCE (src/model_trainer.py) is training-loss-only machinery, not part of the
            # traced architecture below (default build config uses no_nce=True); not vendored.

        self.use_target_network = args.local_target_net
        if self.use_target_network:
            import copy

            self.target_repnet = copy.deepcopy(self.encoder)
            self.target_value_model = copy.deepcopy(self.value_model)

    def update_target_network(self, steps):
        if steps % self.args.target_update_interval == 0 and self.use_target_network:
            self.target_repnet.load_state_dict(self.encoder.state_dict())
            self.target_value_model.load_state_dict(self.value_model.state_dict())

    def encode(self, images, actions):
        return self.encoder(images, actions)

    def initial_inference(self, obs, actions=None, logits=False):
        if len(obs.shape) < 5:
            obs = obs.unsqueeze(0)
        obs = obs.flatten(1, 2)
        hidden_state = self.encoder(obs, actions)
        if not self.args.q_learning:
            policy_logits = self.policy_model(hidden_state)
        else:
            policy_logits = None
        value_logits = self.value_model(hidden_state)
        reward_logits = self.dynamics_model.reward_predictor(hidden_state)

        if logits:
            return NetworkOutput(hidden_state, reward_logits, policy_logits, value_logits)

        value = inverse_transform(from_categorical(value_logits, logits=True))
        reward = inverse_transform(from_categorical(reward_logits, logits=True))
        return NetworkOutput(hidden_state, reward, policy_logits, value)

    def value_target_network(self, obs, actions):
        if len(obs.shape) < 5:
            obs = obs.unsqueeze(0)
        obs = obs.flatten(1, 2)
        hidden_state = self.target_repnet(obs, actions)
        value_logits = self.target_value_model(hidden_state)
        value = inverse_transform(from_categorical(value_logits, logits=True))
        return value

    def inference(self, state, action):
        next_state, reward_logits, policy_logits, value_logits = self.step(state, action)
        value = inverse_transform(from_categorical(value_logits, logits=True))
        reward = inverse_transform(from_categorical(reward_logits, logits=True))

        return NetworkOutput(next_state, reward, policy_logits, value)

    def step(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        if not self.args.q_learning:
            policy_logits = self.policy_model(next_state)
        else:
            policy_logits = None
        value_logits = self.value_model(next_state)

        return next_state, reward_logits, policy_logits, value_logits

    # NOTE: the real MCTSModel.forward(self, buffer, trackers=None, step=True) samples from an
    # rlpyt replay buffer and computes the full multi-step SPR/NCE training loss -- that is
    # training-harness orchestration (not architecture) and hard-depends on rlpyt buffer
    # objects; it is intentionally not vendored. `initial_inference` above is the real forward
    # inference path (encoder -> policy/value heads + dynamics reward head) and is what this
    # staging module traces.
    def forward(self, obs, actions):
        return self.initial_inference(obs, actions, logits=True)


class _Args:
    """Minimal config namespace matching the real args.* fields MCTSModel.__init__ reads
    (jumps, multistep, use_all_targets, no_nce, batch_size_per_worker, device, film,
    hidden_size, dynamics_blocks, q_learning, framestack, grayscale, local_target_net).
    no_nce=True / local_target_net=False / film=False / q_learning=False select the base
    (non-FiLM, non-Q-learning, no-contrastive-loss) MCTSModel configuration used in the repo's
    default Atari 100k experiments -- the encoder/dynamics/policy/value architecture is
    identical to the paper's. hidden_size=256 matches RepNet's real (hardcoded, not
    hidden_size-parameterized) internal channel doubling 128->256 -- the repo's actual default
    config also uses hidden_size=256 for this reason (dynamics_model/value_model/policy_model
    input channel count must equal RepNet's real fixed output channel count)."""

    jumps = 5
    multistep = 1
    use_all_targets = False
    no_nce = True
    batch_size_per_worker = 2
    device = torch.device("cpu")
    film = False
    hidden_size = 256
    dynamics_blocks = 1
    q_learning = False
    framestack = 1
    grayscale = True
    local_target_net = False
    target_update_interval = 1


def build_spr_mctsmodel():
    torch.manual_seed(0)
    num_actions = 6
    return MCTSModel(_Args(), num_actions)


def example_input_spr_mctsmodel():
    torch.manual_seed(0)
    batch = 2
    # initial_inference expects obs already 5D as `states[0]` would be in the real training
    # loop: (B, framestack, C, H, W). Since it's already 5D, the `unsqueeze(0)` branch is
    # skipped and `flatten(1, 2)` merges (framestack, C) -> RepNet's expected
    # framestack*(1 if grayscale else 3) input channels.
    obs = torch.rand(batch, _Args.framestack, 1, 96, 96)
    return (obs, None)


MENAGERIE_ENTRIES = [
    (
        "SPR_SelfPredictiveRepresentations",
        "build_spr_mctsmodel",
        "example_input_spr_mctsmodel",
        2021,
        "vendored-pytorch",
    ),
]
