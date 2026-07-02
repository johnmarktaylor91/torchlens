# SOURCE: vendored from https://github.com/erichson/koopmanAE @ f6b7db79 (model.py)
#
# "Consistent Koopman Autoencoders" (Azencot, Erichson, Lin, Mahoney; ICML 2020).
# A dynamical-systems autoencoder: an MLP encoder lifts the raw state into a
# latent "Koopman" observable space, a single linear (bias-free) layer
# advances the latent state one Koopman-operator step at a time (rolled out
# for `steps` steps, matching the real repo's multi-step-prediction training
# objective), a second linear layer models the operator's (approximate)
# inverse for the backward/consistency loss, and an MLP decoder maps latent
# states back to reconstructed/predicted states. Real repo code (encoderNet,
# decoderNet, dynamics, dynamics_back, koopmanAE, gaussian_init_) copied
# unmodified aside from harmless per-call `m` shadowing already present in the
# original (loop variable named `m` inside `for m in self.modules()`, same as
# the constructor's `m` parameter -- present in the real repo, left as-is).
# No layer, initialization, or dataflow inside the model was changed.

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class encoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA=1):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, 16 * ALPHA)
        self.fc2 = nn.Linear(16 * ALPHA, 16 * ALPHA)
        self.fc3 = nn.Linear(16 * ALPHA, b)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


class decoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA=1):
        super(decoderNet, self).__init__()

        self.m = m
        self.n = n
        self.b = b

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(b, 16 * ALPHA)
        self.fc2 = nn.Linear(16 * ALPHA, 16 * ALPHA)
        self.fc3 = nn.Linear(16 * ALPHA, m * n)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = x.view(-1, 1, self.m, self.n)
        return x


class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

    def forward(self, x):
        x = self.dynamics(x)
        return x


class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())

    def forward(self, x):
        x = self.dynamics(x)
        return x


class koopmanAE(nn.Module):
    def __init__(self, m, n, b, steps, steps_back, alpha=1, init_scale=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back

        self.encoder = encoderNet(m, n, b, ALPHA=alpha)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoderNet(m, n, b, ALPHA=alpha)

    def forward(self, x, mode="forward"):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        if mode == "forward":
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous()))
            return out, out_back

        if mode == "backward":
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))

            out_back.append(self.decoder(z.contiguous()))
            return out, out_back


def build_koopman_ae():
    # Tiny config: state is a 1x4 "image" (m=1, n=4), latent (Koopman
    # observable) dimension b=6, 4 forward steps / 4 backward steps -- matches
    # the real repo's `koopmanAE(m, n, b, steps, steps_back, ...)` signature
    # used by `driver.py`/`train.py`, just at a much smaller scale.
    model = koopmanAE(m=1, n=4, b=6, steps=4, steps_back=4, alpha=1, init_scale=1)
    return model


def example_input_koopman_ae():
    # forward(x, mode='forward') only consumes x positionally; mode='forward'
    # is the default and matches the real repo's primary training path.
    return torch.randn(3, 1, 4)


MENAGERIE_ENTRIES = [
    (
        "Koopman-AE-Consistent",
        build_koopman_ae,
        example_input_koopman_ae,
        2020,
        MENAGERIE_ZOO,
    ),
]
