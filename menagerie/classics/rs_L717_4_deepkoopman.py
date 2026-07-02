# FAITHFUL PORT of BethanyL/DeepKoopman @ master (original framework: TensorFlow 1.x,
# networkarch.py `encoder`/`decoder`/`create_omega_net`/`varying_multiply`/
# `create_koopman_net`)
# DeepKoopman (Lusch, Kutz & Brunton, 2018, Nature Communications) learns a Koopman
# embedding for nonlinear dynamics: an encoder MLP maps state x to Koopman
# eigenfunction coordinates y, a state-dependent linear "auxiliary" (omega) network
# parameterizes complex-conjugate-pair rotation/scaling blocks (or real-eigenvalue
# scalings) that advance y one delta_t step, and a decoder MLP maps y back to x. The
# official repo is TF1 (`tf.placeholder`, `tf.Variable`, `tf.compat.v1.disable_eager_
# execution()`) with no PyTorch release, so this ports the forward (inference) path
# layer-for-layer into torch: same encoder/decoder MLP widths and activation choice,
# same per-eigenvalue omega auxiliary sub-networks, same `form_complex_conjugate_
# block`/`varying_multiply` state-dependent linear advance for one step. Training-only
# scaffolding (placeholders, multi-shift loss stacking, weight-distribution presets)
# is dropped since only the forward `create_koopman_net` computation graph is needed.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"

_ACTS = {"relu": torch.relu, "sigmoid": torch.sigmoid, "elu": torch.nn.functional.elu}


class MLPStack(nn.Module):
    """Stack of Linear layers with an activation between them and none on the final
    layer -- port of `encoder_apply_one_shift` / `decoder_apply`."""

    def __init__(self, widths, act_type="relu"):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)]
        )
        self.act = _ACTS[act_type]

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = self.act(x)
        return x


class OmegaNet(nn.Module):
    """Auxiliary network for one real eigenvalue or one complex-conjugate pair: maps
    the (squared-)radius of its y-coordinate(s) to the eigenvalue parameters used by
    `varying_multiply` -- port of `create_one_omega_net` / `omega_net_apply_one`."""

    def __init__(self, in_dim, hidden_widths, out_dim, act_type="relu"):
        super().__init__()
        widths = [in_dim] + list(hidden_widths) + [out_dim]
        self.net = MLPStack(widths, act_type)

    def forward(self, x):
        return self.net(x)


def form_complex_conjugate_block(omegas, delta_t):
    """Port of `form_complex_conjugate_block`: build the [batch, 2, 2] rotation-and-
    scaling matrix exp(mu*dt) * [[cos(w*dt), -sin(w*dt)], [sin(w*dt), cos(w*dt)]]."""
    scale = torch.exp(omegas[:, 1] * delta_t)
    entry11 = scale * torch.cos(omegas[:, 0] * delta_t)
    entry12 = scale * torch.sin(omegas[:, 0] * delta_t)
    row1 = torch.stack([entry11, -entry12], dim=1)
    row2 = torch.stack([entry12, entry11], dim=1)
    return torch.stack([row1, row2], dim=2)


def varying_multiply(y, omegas, delta_t, num_real, num_complex_pairs):
    """Port of `varying_multiply`: advance y-coordinates one delta_t step under the
    (state-dependent) block-diagonal Koopman operator L(y)."""
    parts = []
    for j in range(num_complex_pairs):
        ind = 2 * j
        ystack = torch.stack([y[:, ind : ind + 2], y[:, ind : ind + 2]], dim=2)
        l_stack = form_complex_conjugate_block(omegas[j], delta_t)
        parts.append((ystack * l_stack).sum(dim=1))
    for j in range(num_real):
        ind = 2 * num_complex_pairs + j
        temp = y[:, ind]
        parts.append((temp.unsqueeze(-1) * torch.exp(omegas[num_complex_pairs + j] * delta_t)))
    return torch.cat(parts, dim=1)


class DeepKoopman(nn.Module):
    """DeepKoopman (Lusch et al. 2018): encoder MLP -> state-dependent linear Koopman
    advance (real + complex-conjugate-pair eigenvalue blocks via per-eigenvalue omega
    auxiliary networks) -> decoder MLP. Forward pass returns (x_reconstructed,
    x_advanced_one_step), matching y[0] and y[1] of the original `create_koopman_net`."""

    def __init__(
        self,
        widths,
        hidden_widths_omega,
        num_real,
        num_complex_pairs,
        delta_t=0.02,
        act_type="relu",
    ):
        super().__init__()
        depth = (len(widths) - 4) // 2
        encoder_widths = widths[: depth + 2]
        decoder_widths = widths[depth + 2 :]

        self.encoder = MLPStack(encoder_widths, act_type)
        self.decoder = MLPStack(decoder_widths, act_type)

        self.num_real = num_real
        self.num_complex_pairs = num_complex_pairs
        self.delta_t = delta_t

        self.omega_complex = nn.ModuleList(
            [OmegaNet(1, hidden_widths_omega, 2, act_type) for _ in range(num_complex_pairs)]
        )
        self.omega_real = nn.ModuleList(
            [OmegaNet(1, hidden_widths_omega, 1, act_type) for _ in range(num_real)]
        )

    def _omega_apply(self, y):
        omegas = []
        for j in range(self.num_complex_pairs):
            ind = 2 * j
            pair = y[:, ind : ind + 2]
            radius = (pair**2).sum(dim=1, keepdim=True)
            omegas.append(self.omega_complex[j](radius))
        for j in range(self.num_real):
            ind = 2 * self.num_complex_pairs + j
            col = y[:, ind : ind + 1]
            omegas.append(self.omega_real[j](col))
        return omegas

    def forward(self, x):
        g = self.encoder(x)
        x_recon = self.decoder(g)

        omegas = self._omega_apply(g)
        g_advanced = varying_multiply(
            g, omegas, self.delta_t, self.num_real, self.num_complex_pairs
        )
        x_advanced = self.decoder(g_advanced)

        return x_recon, x_advanced


def build_deepkoopman():
    # Pendulum-experiment config (PendulumExperiment.py): n=2 state dims, k=2 Koopman
    # eigenfunction coords, 1 complex-conjugate eigenvalue pair (num_real=0), single
    # hidden layer of width 8 in encoder/decoder (shrunk from released ~10-20) and a
    # single hidden layer of width 8 in the omega auxiliary network.
    return DeepKoopman(
        widths=[2, 8, 2, 2, 8, 2],
        hidden_widths_omega=[8],
        num_real=0,
        num_complex_pairs=1,
        delta_t=0.02,
        act_type="relu",
    )


def example_input_deepkoopman():
    torch.manual_seed(0)
    return (torch.randn(4, 2),)


MENAGERIE_ENTRIES = [
    (
        "DeepKoopman",
        "build_deepkoopman",
        "example_input_deepkoopman",
        2018,
        "ported-pytorch",
    ),
]
