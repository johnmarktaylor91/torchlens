# SOURCE: vendored from davidshih17/ANODE @ master (flows.py)
from __future__ import annotations

import math
import types

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == "input":
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == "output":
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, cond_in_features=None, bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features, bias=False)

        self.register_buffer("mask", mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


nn.MaskedLinear = MaskedLinear


class MADESplit(nn.Module):
    """An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(
        self,
        num_inputs,
        num_hidden,
        num_cond_inputs=None,
        s_act="tanh",
        t_act="relu",
        pre_exp_tanh=False,
    ):
        super(MADESplit, self).__init__()

        self.pre_exp_tanh = pre_exp_tanh

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs, num_inputs, mask_type="output")

        act_func = activations[s_act]
        self.s_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)

        self.s_trunk = nn.Sequential(
            act_func(),
            nn.MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            nn.MaskedLinear(num_hidden, num_inputs, output_mask),
        )

        act_func = activations[t_act]
        self.t_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)

        self.t_trunk = nn.Sequential(
            act_func(),
            nn.MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            nn.MaskedLinear(num_hidden, num_inputs, output_mask),
        )

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            h = self.s_joiner(inputs, cond_inputs)
            m = self.s_trunk(h)

            h = self.t_joiner(inputs, cond_inputs)
            a = self.t_trunk(h)

            if self.pre_exp_tanh:
                a = torch.tanh(a)

            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.s_joiner(x, cond_inputs)
                m = self.s_trunk(h)

                h = self.t_joiner(x, cond_inputs)
                a = self.t_trunk(h)

                if self.pre_exp_tanh:
                    a = torch.tanh(a)

                x[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class MADE(nn.Module):
    """An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(
        self, num_inputs, num_hidden, num_cond_inputs=None, act="relu", pre_exp_tanh=False
    ):
        super(MADE, self).__init__()

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type="output")

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)

        self.trunk = nn.Sequential(
            act_func(),
            nn.MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            nn.MaskedLinear(num_hidden, num_inputs * 2, output_mask),
        )

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(-1, keepdim=True)
        else:
            return torch.log(inputs / (1 - inputs)), -torch.log(inputs - inputs**2).sum(
                -1, keepdim=True
            )


class Logit(Sigmoid):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return super(Logit, self).forward(inputs, "inverse")
        else:
            return super(Logit, self).forward(inputs, "direct")


class BatchNormFlow(nn.Module):
    """An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_inputs))
        self.register_buffer("running_var", torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
                self.running_var.add_(self.batch_var.data * (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)


class ActNorm(nn.Module):
    """An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if not self.initialized:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == "direct":
            return (inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
                -1, keepdim=True
            ).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(-self.weight) + self.bias, -self.weight.sum(
                -1, keepdim=True
            ).unsqueeze(0).repeat(inputs.size(0), 1)


class InvertibleMM(nn.Module):
    """An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(InvertibleMM, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_inputs, num_inputs))
        nn.init.orthogonal_(self.W)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return inputs @ self.W, torch.slogdet(self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                inputs.size(0), 1
            )
        else:
            return inputs @ torch.inverse(self.W), -torch.slogdet(self.W)[-1].unsqueeze(
                0
            ).unsqueeze(0).repeat(inputs.size(0), 1)


class LUInvertibleMM(nn.Module):
    """An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(LUInvertibleMM, self).__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.from_numpy(sign_S)
        self.log_S = nn.Parameter(torch.from_numpy(log_S))

        self.I = torch.eye(self.L.size(0))

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if str(self.L_mask.device) != str(self.L.device):
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I = self.I.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == "direct":
            return inputs @ W, self.log_S.sum().unsqueeze(0).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(W), -self.log_S.sum().unsqueeze(0).unsqueeze(0).repeat(
                inputs.size(0), 1
            )


class Shuffle(nn.Module):
    """An implementation of a shuffling layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Shuffle, self).__init__()
        self.perm = np.random.permutation(num_inputs)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return inputs[:, self.perm], torch.zeros(inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(inputs.size(0), 1, device=inputs.device)


class Reverse(nn.Module):
    """An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return inputs[:, self.perm], torch.zeros(inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(inputs.size(0), 1, device=inputs.device)


class CouplingLayer(nn.Module):
    """An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(
        self, num_inputs, num_hidden, mask, num_cond_inputs=None, s_act="tanh", t_act="relu"
    ):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs

        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        mask = self.mask

        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        if mode == "direct":
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class FlowSequential(nn.Sequential):
    """A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode="direct", logdets=None):
        """Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ["direct", "inverse"]
        if mode == "direct":
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode="inverse")[0]
        return samples


def build_anode() -> FlowSequential:
    """Build a tiny ANODE normalizing-flow stack for TorchLens tracing."""
    model = FlowSequential(
        MADE(num_inputs=4, num_hidden=8, act="relu"),
        Reverse(num_inputs=4),
        BatchNormFlow(num_inputs=4),
    )
    model.eval()
    return model


def example_input_anode() -> torch.Tensor:
    """Return a small dense feature batch."""
    return torch.randn(3, 4)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [("ANODE", "build_anode", "example_input_anode", 2020, "CV1")]
