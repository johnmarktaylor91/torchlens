# SOURCE: vendored from https://github.com/morim3/DeepKalmanFilter @ master
# (model/deep_kalman_filter.py)
#
# Deep Kalman Filter (Krishnan, Shalit, Sontag 2015, arXiv:1511.05121):
# a variational-recurrent state-space model that generalizes the classical
# Kalman filter with neural-network transition/emission/posterior functions.
# `Transition` produces the prior over the next latent state z_t from z_{t-1}
# (a two-layer MLP predicting a Gaussian mean/std), `Posterior` combines that
# prior with the current observation to produce the approximate-posterior
# Gaussian over z_t, `Emitter` decodes a latent sample z_t back to
# observation space, and `DeepKalmanFilter.loss` unrolls this
# transition -> posterior -> reparameterize -> emit recurrence across the
# full observation sequence (with an "overshooting" KL term between the
# posterior and multi-step-ahead transition predictions), exactly as in the
# paper. `Emitter`, `Transition`, `Posterior`, `DeepKalmanFilter.__init__`,
# `.reparametrization`, `.kl_div`, and `.loss` are copied verbatim below (no
# architecture change).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class Emitter(nn.Module):
    def __init__(self, z_dim, hidden_dim, obs_dim):
        super(Emitter, self).__init__()
        self.z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_loc = nn.Linear(hidden_dim, obs_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z):
        hidden1 = self.relu(self.z_to_hidden(z))
        hidden2 = self.relu(self.hidden_to_hidden(hidden1))
        loc = self.hidden_to_loc(hidden2)

        return loc


class Transition(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Transition, self).__init__()
        self.z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_loc = nn.Linear(hidden_dim, z_dim)
        self.hidden_to_sig = nn.Linear(hidden_dim, z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        hidden1 = self.relu(self.z_to_hidden(z_t_1))
        hidden2 = self.relu(self.hidden_to_hidden(hidden1))

        loc = self.hidden_to_loc(hidden2)
        sigma = self.softplus(self.hidden_to_sig(hidden2))

        return loc, sigma


class Posterior(nn.Module):
    def __init__(self, z_dim, hidden_dim, obs_dim):
        super(Posterior, self).__init__()
        self.z_obs_to_hidden = nn.Linear(2 * z_dim + obs_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)

        self.hidden_to_loc = nn.Linear(hidden_dim, z_dim)
        self.hidden_to_sig = nn.Linear(hidden_dim, z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_mu, z_sig, obs_t):
        hidden1 = self.relu(self.z_obs_to_hidden(torch.cat((z_mu, z_sig, obs_t), dim=-1)))
        hidden2 = self.relu(self.hidden_to_hidden(hidden1))

        loc = self.hidden_to_loc(hidden2)
        sig = self.softplus(self.hidden_to_sig(hidden2))

        return loc, sig


class DeepKalmanFilter(nn.Module):
    def __init__(self, config):
        super(DeepKalmanFilter, self).__init__()

        self.emitter = Emitter(config.z_dim, config.emit_hidden_dim, config.obs_dim)
        self.transition = Transition(config.z_dim, config.trans_hidden_dim)

        self.posterior = Posterior(config.z_dim, config.post_hidden_dim, config.obs_dim)

        self.z_q_0 = nn.Parameter(torch.zeros(config.z_dim))
        self.emit_log_sigma = nn.Parameter(config.emit_log_sigma * torch.ones(config.obs_dim))

        self.config = config

    @staticmethod
    def reparametrization(mu, sig):
        return mu + torch.randn_like(sig) * sig

    @staticmethod
    def kl_div(mu0, sig0, mu1, sig1):
        return -0.5 * torch.sum(
            1
            - 2 * sig1.log()
            + 2 * sig0.log()
            - (mu1 - mu0).pow(2) / sig1.pow(2)
            - (sig0 / sig1).pow(2)
        )

    def loss(self, obs):
        time_step = obs.size(1)
        batch_size = obs.size(0)
        overshoot_len = self.config.overshooting

        kl = torch.Tensor([0]).to(self.config.device)
        reconstruction = torch.Tensor([0]).to(self.config.device)

        emit_sig = self.emit_log_sigma.exp()

        for s in range(self.config.sampling_num):
            z_q_t = self.z_q_0.expand((batch_size, self.config.z_dim))

            for t in range(time_step):
                trans_loc, trans_sig = self.transition(z_q_t)

                post_loc, post_sig = self.posterior(trans_loc, trans_sig, obs[:, t])

                z_q_t = self.reparametrization(post_loc, post_sig)
                emit_loc = self.emitter(z_q_t)

                reconstruction += (
                    (emit_loc - obs[:, t]).pow(2).sum(dim=0) / 2 / emit_sig
                    + self.emit_log_sigma * batch_size / 2
                ).sum()
                if t > 0:
                    # `overshooting` is loop-carried Python state set at the
                    # end of the previous `t` iteration below (real repo
                    # pattern, verbatim); ruff's per-line scoping can't see
                    # the cross-iteration flow.
                    over_loc, over_sig = self.transition(overshooting[: overshoot_len - 1])  # noqa: F821
                    over_loc = torch.cat([trans_loc.unsqueeze(0), over_loc], dim=0)
                    over_sig = torch.cat([trans_sig.unsqueeze(0), over_sig], dim=0)
                else:
                    over_loc = trans_loc.unsqueeze(0)
                    over_sig = trans_sig.unsqueeze(0)

                overshooting = self.reparametrization(over_loc, over_sig)  # noqa: F841
                kl = kl + self.kl_div(
                    post_loc.expand_as(over_loc), post_sig.expand_as(over_sig), over_loc, over_sig
                ) / min(t + 1, self.config.overshooting)

        reconstruction = reconstruction / self.config.sampling_num
        kl = kl / self.config.sampling_num
        return reconstruction, kl


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo). The real repo
# drives `DeepKalmanFilter` via `config = AttrDict({...})`
# (experiment/sin_wave.ipynb); we reproduce that with a tiny stand-in config
# object exposing the same attributes, at the same values used in the real
# sin-wave experiment notebook (z_dim=2, hidden dims=50, obs_dim=1,
# sampling_num=5, overshooting=1), only shrinking hidden dims for a fast
# CPU trace. `loss()` is the real forward computation graph of the paper
# (the variational-RNN unroll); we trace it directly.
# ---------------------------------------------------------------------------
class _Config:
    def __init__(
        self,
        z_dim,
        emit_hidden_dim,
        trans_hidden_dim,
        post_hidden_dim,
        obs_dim,
        sampling_num,
        device,
        overshooting,
        emit_log_sigma=0.0,
    ):
        self.z_dim = z_dim
        self.emit_hidden_dim = emit_hidden_dim
        self.trans_hidden_dim = trans_hidden_dim
        self.post_hidden_dim = post_hidden_dim
        self.obs_dim = obs_dim
        self.sampling_num = sampling_num
        self.device = device
        self.overshooting = overshooting
        self.emit_log_sigma = emit_log_sigma


class DeepKalmanFilterLoss(nn.Module):
    """Thin adapter exposing DeepKalmanFilter.loss(obs) as a plain
    Tensor -> Tensor forward (torchlens traces forward(); the real loss()
    is called unmodified and we sum its two real return values into one
    scalar trace output)."""

    def __init__(self, dkf):
        super().__init__()
        self.dkf = dkf

    def forward(self, obs):
        reconstruction, kl = self.dkf.loss(obs)
        return reconstruction + kl


_BATCH = 4
_TIME_STEP = 5
_OBS_DIM = 1


def build_deepkf_deepkalmanfilter():
    torch.manual_seed(0)
    config = _Config(
        z_dim=2,
        emit_hidden_dim=8,
        trans_hidden_dim=8,
        post_hidden_dim=8,
        obs_dim=_OBS_DIM,
        sampling_num=2,
        device="cpu",
        overshooting=1,
    )
    dkf = DeepKalmanFilter(config)
    model = DeepKalmanFilterLoss(dkf)
    return model


def example_input_deepkf_deepkalmanfilter():
    torch.manual_seed(0)
    return torch.randn(_BATCH, _TIME_STEP, _OBS_DIM)


MENAGERIE_ENTRIES = [
    (
        "DeepKalmanFilter",
        "build_deepkf_deepkalmanfilter",
        "example_input_deepkf_deepkalmanfilter",
        2015,
        MENAGERIE_ZOO,
    ),
]
