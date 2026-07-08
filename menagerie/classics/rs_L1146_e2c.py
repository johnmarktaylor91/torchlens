# SOURCE: vendored from https://github.com/tung-nd/E2C-pytorch @ 8af375b58e
# Embed to Control (E2C): a locally-linear latent-dynamics VAE.
# Paper: "Embed to Control: A Locally Linear Latent Dynamics Model for
# Control from Raw Images" (Watter et al., NeurIPS 2015, arXiv:1506.07365).
#
# Vendored real repo code (networks.py + e2c_model.py + normal.py), combined
# into one file. Only non-architectural portability fixes applied:
#   - removed the hardcoded `.cuda()` call in Transition.forward so the model
#     traces on CPU (device-follow via `.to(z_bar_t.device)` instead).
#   - dropped the module-level `torch.set_default_dtype(torch.float64)` calls
#     from the original files (float64-everywhere is a training-script
#     convenience, not part of the architecture; using it here would silently
#     flip every recipe/model registered after this file is imported).
# No layer, head, or dataflow was changed.

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)


class NormalDistribution:
    """q(z|x) parameterization used by E2C; A = I + v^T r gives the locally
    linear covariance transform of the transition model."""

    def __init__(self, mean, logvar, v=None, r=None, A=None):
        self.mean = mean
        self.logvar = logvar
        self.v = v
        self.r = r

        sigma = torch.diag_embed(torch.exp(logvar))
        if A is None:
            self.cov = sigma
        else:
            self.cov = A.bmm(sigma.bmm(A.transpose(1, 2)))

    @staticmethod
    def KL_divergence(q_z_next_pred, q_z_next):
        mu_0 = q_z_next_pred.mean
        mu_1 = q_z_next.mean
        sigma_0 = torch.exp(q_z_next_pred.logvar)
        sigma_1 = torch.exp(q_z_next.logvar)
        v = q_z_next_pred.v
        r = q_z_next_pred.r
        k = float(q_z_next_pred.mean.size(1))

        def _sum(x):
            return torch.sum(x, dim=1)

        KL = 0.5 * torch.mean(
            _sum((sigma_0 + 2 * sigma_0 * v * r) / sigma_1)
            + _sum(r.pow(2) * sigma_0) * _sum(v.pow(2) / sigma_1)
            + _sum(torch.pow(mu_1 - mu_0, 2) / sigma_1)
            - k
            + 2 * (_sum(q_z_next.logvar - q_z_next_pred.logvar) - torch.log(1 + _sum(v * r)))
        )
        return KL


class Encoder(nn.Module):
    def __init__(self, net, obs_dim, z_dim):
        super(Encoder, self).__init__()
        self.net = net
        self.net.apply(weights_init)
        self.img_dim = obs_dim
        self.z_dim = z_dim

    def forward(self, x):
        return self.net(x).chunk(2, dim=1)  # first half mean, second half logvar


class Decoder(nn.Module):
    def __init__(self, net, z_dim, obs_dim):
        super(Decoder, self).__init__()
        self.net = net
        self.net.apply(weights_init)
        self.z_dim = z_dim
        self.obs_dim = obs_dim

    def forward(self, z):
        return self.net(z)


class Transition(nn.Module):
    def __init__(self, net, z_dim, u_dim):
        super(Transition, self).__init__()
        self.net = net
        self.net.apply(weights_init)
        self.h_dim = self.net[-3].out_features
        self.z_dim = z_dim
        self.u_dim = u_dim

        self.fc_A = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim * 2),  # v_t and r_t
            nn.Sigmoid(),
        )
        self.fc_A.apply(weights_init)

        self.fc_B = nn.Linear(self.h_dim, self.z_dim * self.u_dim)
        torch.nn.init.orthogonal_(self.fc_B.weight)
        self.fc_o = nn.Linear(self.h_dim, self.z_dim)
        torch.nn.init.orthogonal_(self.fc_o.weight)

    def forward(self, z_bar_t, q_z_t, u_t):
        h_t = self.net(z_bar_t)
        B_t = self.fc_B(h_t)
        o_t = self.fc_o(h_t)

        v_t, r_t = self.fc_A(h_t).chunk(2, dim=1)
        v_t = torch.unsqueeze(v_t, dim=-1)
        r_t = torch.unsqueeze(r_t, dim=-2)

        # NOTE (portability fix): original code hardcoded `.cuda()` here;
        # follow the input tensor's device instead so the model traces on CPU.
        A_t = torch.eye(self.z_dim, device=z_bar_t.device).repeat(
            z_bar_t.size(0), 1, 1
        ) + torch.bmm(v_t, r_t)

        B_t = B_t.view(-1, self.z_dim, self.u_dim)

        mu_t = q_z_t.mean

        mean = (
            A_t.bmm(mu_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + o_t
        )

        return mean, NormalDistribution(
            mean, logvar=q_z_t.logvar, v=v_t.squeeze(), r=r_t.squeeze(), A=A_t
        )


class PlanarEncoder(Encoder):
    def __init__(self, obs_dim=1600, z_dim=2):
        net = nn.Sequential(
            nn.Linear(obs_dim, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, z_dim * 2),
        )
        super(PlanarEncoder, self).__init__(net, obs_dim, z_dim)


class PlanarDecoder(Decoder):
    def __init__(self, z_dim=2, obs_dim=1600):
        net = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 1600),
            nn.Sigmoid(),
        )
        super(PlanarDecoder, self).__init__(net, z_dim, obs_dim)


class PlanarTransition(Transition):
    def __init__(self, z_dim=2, u_dim=2):
        net = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
        )
        super(PlanarTransition, self).__init__(net, z_dim, u_dim)


CONFIG = {
    "planar": (PlanarEncoder, PlanarDecoder, PlanarTransition),
}


def load_config(name):
    return CONFIG[name]


class E2C(nn.Module):
    """Embed-to-Control: VAE encoder/decoder plus a locally-linear latent
    transition model conditioned on an action `u`."""

    def __init__(self, obs_dim, z_dim, u_dim, env="planar"):
        super(E2C, self).__init__()
        enc, dec, trans = load_config(env)

        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.u_dim = u_dim

        self.encoder = enc(obs_dim=obs_dim, z_dim=z_dim)
        self.decoder = dec(z_dim=z_dim, obs_dim=obs_dim)
        self.trans = trans(z_dim=z_dim, u_dim=u_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def transition(self, z_bar, q_z, u):
        return self.trans(z_bar, q_z, u)

    def reparam(self, mean, logvar):
        sigma = (logvar / 2).exp()
        epsilon = torch.randn_like(sigma)
        return mean + torch.mul(epsilon, sigma)

    def forward(self, x, u, x_next):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)

        x_recon = self.decode(z)

        z_next, q_z_next_pred = self.transition(z, q_z, u)

        x_next_pred = self.decode(z_next)

        mu_next, logvar_next = self.encode(x_next)
        q_z_next = NormalDistribution(mean=mu_next, logvar=logvar_next)

        return x_recon, x_next_pred, q_z, q_z_next_pred, q_z_next


def build_e2c_planar():
    # Tiny (planar env) config: 1600-d flattened image observation, 2-d
    # latent, 2-d action -- the real repo's smallest registered environment.
    return E2C(obs_dim=1600, z_dim=2, u_dim=2, env="planar")


def example_input_e2c_planar():
    batch = 4
    x = torch.randn(batch, 1600)
    u = torch.randn(batch, 2)
    x_next = torch.randn(batch, 1600)
    return (x, u, x_next)


MENAGERIE_ENTRIES = [
    (
        "Embed to Control (E2C)",
        build_e2c_planar,
        example_input_e2c_planar,
        2015,
        MENAGERIE_ZOO,
    ),
]
