# SOURCE: vendored from simonalexanderson/StyleGestures @ bd906cf7d140
# Files: glow/models.py, glow/modules.py, glow/thops.py (unmodified except
# import-path fixes: relative `from . import X` -> local top-level import within
# this file) -- https://github.com/simonalexanderson/StyleGestures
"""MoGlow (Henter, Alexanderson & Beskow, SIGGRAPH Asia / ACM ToG 2020):
"Probabilistic and controllable motion synthesis using normalising flows".
The MoGlow project page (OFA-Sys/simonalexanderson/MoGlow) is a pointer-only
repo; the README explicitly directs to this sibling repo
(simonalexanderson/StyleGestures) as "Code and motion data are publicly
available on GitHub" -- this is the real, official MoGlow implementation:
an invertible Glow-style normalizing flow (actnorm + invertible 1x1 conv +
affine coupling) whose coupling-network is a stateful LSTM/GRU conditioned on
autoregressive control signal, giving frame-by-frame controllable motion
generation.
"""

import numpy as np
import scipy.linalg
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from glow/thops.py ----
def sum_(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


def mean_(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross3":
        return tensor[:, 0::3, ...], tensor[:, 1::3, ...], tensor[:, 2::3, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def cat_feature(tensor_a, tensor_b):
    return torch.cat((tensor_a, tensor_b), dim=1)


def timesteps(tensor):
    return int(tensor.size(2))


# ---- verbatim from glow/modules.py ----
def nan_throw(tensor, name="tensor"):
    # (diagnostics only; original nan/inf print-and-continue suppressed for staging --
    # upstream never raises here either, see glow/modules.py's commented-out raise)
    return


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.0):
        super().__init__()
        size = [1, num_features, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        assert input.device == self.bias.device
        with torch.no_grad():
            bias = mean_(input.clone(), dim=[0, 2], keepdim=True) * -1.0
            vars = mean_((input.clone() + bias) ** 2, dim=[0, 2], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, reverse=False):
        if not reverse:
            return input + self.bias
        else:
            return input - self.bias

    def _scale(self, input, logdet=None, reverse=False):
        logs = self.logs
        if not reverse:
            input = input * torch.exp(logs)
        else:
            input = input * torch.exp(-logs)
        if logdet is not None:
            dlogdet = sum_(logs) * timesteps(input)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)
        if not reverse:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)
        else:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 3
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCT`,"
            " channels should be {} rather than {}".format(self.num_features, input.size())
        )


class LinearZeroInit(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.weight.data.zero_()
        self.bias.data.zero_()


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = np.arange(self.num_channels - 1, -1, -1).astype(np.int64)
        self.indices_inverse = np.zeros((self.num_channels), dtype=np.int64)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        np.random.shuffle(self.indices)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 3
        if not reverse:
            return input[:, self.indices, :]
        else:
            return input[:, self.indices_inverse, :]


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer("p", torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer("sign_s", torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            timesteps_ = timesteps(input)
            dlogdet = torch.slogdet(self.weight)[1] * timesteps_
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1)
            else:
                weight = torch.inverse(self.weight.double()).float().view(w_shape[0], w_shape[1], 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye  # noqa: E741 (matches upstream LU-decomposition naming)
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(
                self.sign_s * torch.exp(self.log_s)
            )
            dlogdet = sum_(self.log_s) * timesteps(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()  # noqa: E741
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        weight, dlogdet = self.get_weight(input, reverse)
        nan_throw(weight, "weight")
        nan_throw(dlogdet, "dlogdet")

        if not reverse:
            z = F.conv1d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            nan_throw(input, "InConv input")
            z = F.conv1d(input, weight)
            nan_throw(z, "InConv z")
            nan_throw(logdet, "InConv logdet")
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.0):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = LinearZeroInit(self.hidden_dim, output_dim)
        self.do_init = True

    def init_hidden(self):
        self.do_init = True

    def forward(self, input):
        if self.do_init:
            lstm_out, self.hidden = self.lstm(input)
            self.do_init = False
        else:
            lstm_out, self.hidden = self.lstm(input, self.hidden)
        y_pred = self.linear(lstm_out)
        return y_pred


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.0):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = LinearZeroInit(self.hidden_dim, output_dim)
        self.do_init = True

    def init_hidden(self):
        self.do_init = True

    def forward(self, input):
        if self.do_init:
            gru_out, self.hidden = self.gru(input)
            self.do_init = False
        else:
            gru_out, self.hidden = self.gru(input, self.hidden)
        y_pred = self.linear(gru_out)
        return y_pred


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    def likelihood(self, x):
        return -0.5 * (((x) ** 2) + GaussianDiag.Log2PI)

    def logp(self, x):
        likelihood = self.likelihood(x)
        return sum_(likelihood, dim=[1, 2])

    def sample(self, z_shape, eps_std=None, device=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros(z_shape), std=torch.ones(z_shape) * eps_std)
        eps = eps.to(device)
        return eps


# ---- verbatim from glow/models.py ----
def f(in_channels, out_channels, hidden_channels, cond_channels, network_model, num_layers):
    if network_model == "LSTM":
        return LSTM(in_channels + cond_channels, hidden_channels, out_channels, num_layers)
    if network_model == "GRU":
        return GRU(in_channels + cond_channels, hidden_channels, out_channels, num_layers)
    if network_model == "FF":
        return nn.Sequential(
            nn.Linear(in_channels + cond_channels, hidden_channels),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=False),
            LinearZeroInit(hidden_channels, out_channels),
        )


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    NetworkModel = ["LSTM", "GRU", "FF"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(
        self,
        in_channels,
        hidden_channels,
        cond_channels,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="additive",
        network_model="LSTM",
        num_layers=2,
        LU_decomposed=False,
    ):
        assert flow_coupling in FlowStep.FlowCoupling
        assert network_model in FlowStep.NetworkModel
        assert flow_permutation in FlowStep.FlowPermutation
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.network_model = network_model
        # 1. actnorm
        self.actnorm = ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(
                in_channels // 2,
                in_channels - in_channels // 2,
                hidden_channels,
                cond_channels,
                network_model,
                num_layers,
            )
        elif flow_coupling == "affine":
            self.f = f(
                in_channels // 2,
                2 * (in_channels - in_channels // 2),
                hidden_channels,
                cond_channels,
                network_model,
                num_layers,
            )

    def init_lstm_hidden(self):
        if self.network_model == "LSTM" or self.network_model == "GRU":
            self.f.init_hidden()

    def forward(self, input, cond, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, cond, logdet)
        else:
            return self.reverse_flow(input, cond, logdet)

    def normal_flow(self, input, cond, logdet):
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](self, z, logdet, False)
        z1, z2 = split_feature(z, "split")
        z1_cond = torch.cat((z1, cond), dim=1)
        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1_cond)
        elif self.flow_coupling == "affine":
            h = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0) + 1e-6
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = sum_(torch.log(scale), dim=[1, 2]) + logdet

        z = cat_feature(z1, z2)
        return z, cond, logdet

    def reverse_flow(self, input, cond, logdet):
        z1, z2 = split_feature(input, "split")
        z1_cond = torch.cat((z1, cond), dim=1)

        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1_cond)
        elif self.flow_coupling == "affine":
            h = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
            shift, scale = split_feature(h, "cross")
            nan_throw(shift, "shift")
            nan_throw(scale, "scale")
            nan_throw(z2, "z2 unscaled")
            scale = torch.sigmoid(scale + 2.0) + 1e-6
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -sum_(torch.log(scale), dim=[1, 2]) + logdet

        z = cat_feature(z1, z2)
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](self, z, logdet, True)
        nan_throw(z, "z permute_" + str(self.flow_permutation))
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, cond, logdet


class FlowNet(nn.Module):
    def __init__(
        self,
        x_channels,
        hidden_channels,
        cond_channels,
        K,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="additive",
        network_model="LSTM",
        num_layers=2,
        LU_decomposed=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        N = cond_channels
        for _ in range(K):
            self.layers.append(
                FlowStep(
                    in_channels=x_channels,
                    hidden_channels=hidden_channels,
                    cond_channels=N,
                    actnorm_scale=actnorm_scale,
                    flow_permutation=flow_permutation,
                    flow_coupling=flow_coupling,
                    network_model=network_model,
                    num_layers=2,
                    LU_decomposed=LU_decomposed,
                )
            )
            self.output_shapes.append([-1, x_channels, 1])

    def init_lstm_hidden(self):
        for layer in self.layers:
            if isinstance(layer, FlowStep):
                layer.init_lstm_hidden()

    def forward(self, z, cond, logdet=0.0, reverse=False, eps_std=None):
        if not reverse:
            for layer in self.layers:
                z, cond, logdet = layer(z, cond, logdet, reverse=False)
            return z, logdet
        else:
            for i, layer in enumerate(reversed(self.layers)):
                z, cond, logdet = layer(z, cond, logdet=0, reverse=True)
            return z


class _TinyHparamsGlow:
    """Minimal stand-in for the `Glow` hparams sub-namespace the official Glow.__init__
    reads (hparams.Glow.*, hparams.Train.batch_size, hparams.Device.glow). The real
    training script builds this via glow/config.py's JsonConfig loading one of the
    shipped hparams/*.json files (e.g. hparams/preferred/locomotion.json); values below
    mirror that file's [Glow] block, with a tiny hidden_channels/K for fast tracing."""

    class Glow:
        hidden_channels = 8
        K = 2
        actnorm_scale = 1.0
        flow_permutation = "invconv"
        flow_coupling = "affine"
        network_model = "LSTM"
        num_layers = 2
        LU_decomposed = True
        distribution = "normal"

    class Train:
        batch_size = 2

    class Device:
        glow = ["cpu"]


class Glow(nn.Module):
    def __init__(self, x_channels, cond_channels, hparams):
        super().__init__()
        self.flow = FlowNet(
            x_channels=x_channels,
            hidden_channels=hparams.Glow.hidden_channels,
            cond_channels=cond_channels,
            K=hparams.Glow.K,
            actnorm_scale=hparams.Glow.actnorm_scale,
            flow_permutation=hparams.Glow.flow_permutation,
            flow_coupling=hparams.Glow.flow_coupling,
            network_model=hparams.Glow.network_model,
            num_layers=hparams.Glow.num_layers,
            LU_decomposed=hparams.Glow.LU_decomposed,
        )
        self.hparams = hparams

        # register prior hidden (get_proper_device inlined: hparams.Device.glow is
        # already a plain cpu-only device list in the staging hparams, so no GPU
        # probing is required here)
        num_device = len(hparams.Device.glow)
        assert hparams.Train.batch_size % num_device == 0
        self.z_shape = [hparams.Train.batch_size // num_device, x_channels, 1]
        if hparams.Glow.distribution == "normal":
            self.distribution = GaussianDiag()

    def init_lstm_hidden(self):
        self.flow.init_lstm_hidden()

    def forward(self, x=None, cond=None, z=None, eps_std=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, cond)
        else:
            return self.reverse_flow(z, cond, eps_std)

    def normal_flow(self, x, cond):
        n_timesteps = timesteps(x)
        logdet = torch.zeros_like(x[:, 0, 0])
        z, objective = self.flow(x, cond, logdet=logdet, reverse=False)
        objective += self.distribution.logp(z)
        nll = (-objective) / float(np.log(2.0) * n_timesteps)
        return z, nll

    def reverse_flow(self, z, cond, eps_std):
        with torch.no_grad():
            z_shape = self.z_shape
            if z is None:
                z = self.distribution.sample(z_shape, eps_std, device=cond.device)
            x = self.flow(z, cond, eps_std=eps_std, reverse=True)
        return x

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if m.__class__.__name__.find("ActNorm") >= 0:
                m.inited = inited

    @staticmethod
    def loss_generative(nll):
        return torch.mean(nll)


# ---- staging build/example helpers ----
class _MoGlowForward(nn.Module):
    """Thin forward-only wrapper: TorchLens traces a plain forward pass, and the real
    `Glow.forward(x=..., cond=..., reverse=False)` (normal_flow, i.e. the density/NLL
    direction used during training) needs both `x` and `cond` as separate positional
    tensors -- this wraps that call so the module can be traced with a single
    `model(x, cond)` call while exercising the exact official Glow graph
    (FlowNet -> K FlowSteps, each actnorm + invertible-1x1 + LSTM-conditioned affine
    coupling) with no architectural changes.
    """

    def __init__(self, glow: Glow):
        super().__init__()
        self.glow = glow
        self.glow.set_actnorm_init(False)  # first call initializes actnorm from data, as upstream

    def forward(self, x, cond):
        self.glow.init_lstm_hidden()
        z, nll = self.glow.normal_flow(x, cond)
        return z, nll


def build_moglow():
    torch.manual_seed(0)
    x_channels = 8  # must be even (split_feature halves the channel dim)
    cond_channels = 4
    hparams = _TinyHparamsGlow()
    glow = Glow(x_channels, cond_channels, hparams)
    glow.train()  # ActNorm data-dependent init only runs in training mode
    return _MoGlowForward(glow)


def example_input_moglow():
    torch.manual_seed(0)
    batch = 2  # must match hparams.Train.batch_size
    seqlen = 5
    x_channels = 8
    cond_channels = 4
    x = torch.randn(batch, x_channels, seqlen)
    cond = torch.randn(batch, cond_channels, seqlen)
    return (x, cond)


MENAGERIE_ENTRIES = [
    ("MoGlow", build_moglow, example_input_moglow, 2020, "vendored-pytorch"),
]
