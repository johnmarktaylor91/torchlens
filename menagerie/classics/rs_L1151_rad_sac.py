# SOURCE: vendored from MishaLaskin/rad @ master
# https://github.com/MishaLaskin/rad/blob/master/encoder.py
# https://github.com/MishaLaskin/rad/blob/master/curl_sac.py
# "RAD: Reinforcement Learning with Augmented Data" (Laskin, Lee, Stooke,
# Pinto, Abbeel, Srinivas 2020). The `PixelEncoder` (shared pixel-observation
# CNN trunk used by both actor and critic), `Actor` (tanh-squashed Gaussian
# policy head), `QFunction`, and `Critic` (twin-Q SAC critic) classes are
# transcribed VERBATIM from the real repo's `encoder.py` / `curl_sac.py`.
# Only change: the training-only `RadSacAgent`/`CURL`/logging/replay-buffer
# machinery (which needs `gym`, `skimage`, and a live replay buffer) is
# dropped -- the network architecture itself (encoder -> actor/critic heads)
# is exercised end-to-end without them. `weight_init` is kept verbatim since
# it is applied in `Actor.__init__`/`Critic.__init__`.
import torch
import torch.nn as nn
import torch.nn.functional as F


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}
OUT_DIM_108 = {4: 47}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        if obs_shape[-1] == 108:
            assert num_layers in OUT_DIM_108
            out_dim = OUT_DIM_108[num_layers]
        elif obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        else:
            out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        if obs.max() > 1.0:
            obs = obs / 255.0

        self.outputs["obs"] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs["conv1"] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs["conv%s" % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs["fc"] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs["ln"] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs["tanh"] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in ("pixel",)
    return PixelEncoder(obs_shape, feature_dim, num_layers, num_filters, output_logits)


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * torch.log(torch.tensor(2 * torch.pi)) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        log_std_min,
        log_std_max,
        num_layers,
        num_filters,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type,
            obs_shape,
            encoder_feature_dim,
            num_layers,
            num_filters,
            output_logits=True,
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]),
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        self.outputs["mu"] = mu
        self.outputs["std"] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employs two q-functions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        num_layers,
        num_filters,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type,
            obs_shape,
            encoder_feature_dim,
            num_layers,
            num_filters,
            output_logits=True,
        )

        self.Q1 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2


MENAGERIE_ZOO = "vendored-pytorch"

_OBS_SHAPE = (9, 84, 84)
_ACTION_SHAPE = (6,)
_HIDDEN_DIM = 128
_ENCODER_FEATURE_DIM = 50
_NUM_LAYERS = 4
_NUM_FILTERS = 32


def build_rad_sac_critic():
    torch.manual_seed(0)
    model = Critic(
        obs_shape=_OBS_SHAPE,
        action_shape=_ACTION_SHAPE,
        hidden_dim=_HIDDEN_DIM,
        encoder_type="pixel",
        encoder_feature_dim=_ENCODER_FEATURE_DIM,
        num_layers=_NUM_LAYERS,
        num_filters=_NUM_FILTERS,
    )
    model.eval()
    return model


def example_input_rad_sac_critic():
    torch.manual_seed(0)
    obs = torch.rand(2, *_OBS_SHAPE)
    action = torch.rand(2, _ACTION_SHAPE[0])
    return (obs, action)


MENAGERIE_ENTRIES = [
    ("RAD-SAC-Critic", "build_rad_sac_critic", "example_input_rad_sac_critic", 2020, MENAGERIE_ZOO),
]
