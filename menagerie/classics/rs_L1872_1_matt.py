# SOURCE: vendored from https://github.com/CECNL/MAtt @ main
# (mAtt/mAtt.py::mAtt_bci, mAtt/spd.py::{SPDTransform, SPDIncreaseDim,
#  SPDVectorize, SPDUnVectorize(Function), SPDTangentSpace(Function),
#  SPDUnTangentSpace(Function), SPDRectified(Function), SPDPowerFunction},
#  mAtt/utils.py::symmetric, mAtt/__init__.py::StiefelParameter). Class
# bodies are copied verbatim (only import paths adjusted to be
# self-contained in one file; no relative-package imports remain). MAtt
# ("Manifold Attention", NeurIPS 2022) is a Riemannian-geometry EEG decoder:
# raw signal -> temporal/spatial conv front-end -> per-window SPD
# (symmetric positive-definite) covariance matrices (E2R) -> a manifold
# self-attention block (AttentionManifold) that computes Q/K/V via learned
# Stiefel-manifold projections and attends using the log-Euclidean distance
# between SPD matrices -> SPD rectification -> log-Euclidean tangent-space
# flattening (R2E) -> linear classifier head. This has no analogue in any
# installed base lib (it is not standard multi-head attention; the
# attention score itself is a manifold distance and Q/K/V live on the SPD
# manifold), so it is vendored in full rather than approximated.
"""Vendored MAtt (Manifold Attention) EEG decoder model definition."""

import torch
import torch.nn as nn
from torch.autograd import Function

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# mAtt/utils.py
# ---------------------------------------------------------------------------


def symmetric(A):
    size = list(range(len(A.shape)))
    temp = size[-1]
    size.pop()
    size.insert(-1, temp)
    return 0.5 * (A + A.permute(*size))


# ---------------------------------------------------------------------------
# mAtt/__init__.py :: StiefelParameter
# ---------------------------------------------------------------------------


class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the
    space of Stiefel manifold."""

    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return "Parameter containing:" + self.data.__repr__()


# ---------------------------------------------------------------------------
# mAtt/spd.py
# ---------------------------------------------------------------------------


class SPDTransform(nn.Module):
    def __init__(self, input_size, output_size):
        super(SPDTransform, self).__init__()
        self.device = torch.device("cpu")
        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size
        self.weight = StiefelParameter(
            torch.FloatTensor(input_size, output_size).to(self.device), requires_grad=True
        )
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        output = input
        if self.increase_dim:
            output = self.increase_dim(output)
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1, 2), torch.bmm(output, weight))

        return output


class SPDIncreaseDim(nn.Module):
    def __init__(self, input_size, output_size):
        super(SPDIncreaseDim, self).__init__()
        self.device = torch.device("cpu")
        self.register_buffer("eye", torch.eye(output_size, input_size).to(self.device))
        add = torch.as_tensor(
            [0] * input_size + [1] * (output_size - input_size), dtype=torch.float32
        )
        add = add.to(self.device)
        self.register_buffer("add", torch.diag(add))

    def forward(self, input):
        eye = self.eye.unsqueeze(0)
        eye = eye.expand(input.size(0), -1, -1)
        add = self.add.unsqueeze(0)
        add = add.expand(input.size(0), -1, -1)

        output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1, 2)))

        return output


class SPDVectorize(nn.Module):
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        self.device = torch.device("cpu")
        row_idx, col_idx = torch.triu_indices(input_size, input_size)
        self.register_buffer("row_idx", torch.LongTensor(row_idx).to(self.device))
        self.register_buffer("col_idx", torch.LongTensor(col_idx).to(self.device))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output


class SPDUnVectorizeFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        n = int(-0.5 + 0.5 * torch.sqrt(1 + 8 * input.size(1)))
        output = input.new(len(input), n, n)
        output.fill_(0)
        mask_upper = torch.triu_indices(n, n)
        mask_diag = torch.diag_indices(n)
        for k, x in enumerate(input):
            output[k][mask_upper] = x
            output[k] = output[k] + output[k].t()
            output[k][mask_diag] /= 2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None
        if ctx.needs_input_grad[0]:
            n = int(-0.5 + 0.5 * torch.sqrt(1 + 8 * input.size(1)))
            grad_input = input.new(len(input), input.size(1))
            mask = torch.triu_indices(n, n)
            for k, g in enumerate(grad_output):
                grad_input[k] = g[mask]

        return grad_input


class SPDUnVectorize(nn.Module):
    def __init__(self):
        super(SPDUnVectorize, self).__init__()

    def forward(self, input):
        return SPDUnVectorizeFunction.apply(input)


class SPDTangentSpaceFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s.log_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1)
            eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                u, s, v = x.svd()

                g = symmetric(g)

                s_log_diag = s.log().diag()
                s_inv_diag = (1 / s).diag()

                dLdV = 2 * (g.mm(u.mm(s_log_diag)))
                dLdS = eye * (s_inv_diag.mm(u.t().mm(g.mm(u))))

                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0

                grad_input[k] = u.mm(symmetric(P.t() * (u.t().mm(dLdV))) + dLdS).mm(u.t())

        return grad_input


class SPDTangentSpace(nn.Module):
    def __init__(self, input_size, vectorize=True):
        super(SPDTangentSpace, self).__init__()
        self.vectorize = vectorize
        if vectorize:
            self.vec = SPDVectorize(input_size)

    def forward(self, input):
        output = SPDTangentSpaceFunction.apply(input)
        if self.vectorize:
            output = self.vec(output)

        return output


class SPDRectifiedFunction(Function):
    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s[s < epsilon[0]] = epsilon[0]

            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_variables
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1)
            eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, g in enumerate(grad_output):
                if len(g.shape) == 1:
                    continue

                g = symmetric(g)

                x = input[k]
                u, s, v = x.svd()

                max_mask = s > epsilon
                s_max_diag = s.clone()
                s_max_diag[~max_mask] = epsilon
                s_max_diag = s_max_diag.diag()
                Q = max_mask.float().diag()

                dLdV = 2 * (g.mm(u.mm(s_max_diag)))
                dLdS = eye * (Q.mm(u.t().mm(g.mm(u))))

                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0

                grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV)) + dLdS).mm(u.t())

        return grad_input, None


class SPDRectified(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.device = torch.device("cpu")
        self.register_buffer("epsilon", torch.FloatTensor([epsilon]).to(self.device))

    def forward(self, input):
        output = SPDRectifiedFunction.apply(input, self.epsilon)
        return output


# ---------------------------------------------------------------------------
# mAtt/mAtt.py
# ---------------------------------------------------------------------------


class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device("cpu")

    def forward(self, x):
        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        cov = x @ x.permute(0, 2, 1)
        cov = cov.to(self.dev)
        cov = cov / (x.shape[-1] - 1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov /= tra
        identity = (
            torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev)
            .to(self.dev)
            .repeat(x.shape[0], 1, 1)
        )
        cov = cov + (1e-5 * identity)
        return cov


class E2R(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()

    def patch_len(self, n, epochs):
        list_len = []
        base = n // epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base * epochs):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            return ValueError("check your epochs and axis should be split again")

    def forward(self, x):
        # x with shape[bs, ch, time]
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-1))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        return x


class AttentionManifold(nn.Module):
    def __init__(self, in_embed_size, out_embed_size):
        super(AttentionManifold, self).__init__()

        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.q_trans = SPDTransform(self.d_in, self.d_out).cpu()
        self.k_trans = SPDTransform(self.d_in, self.d_out).cpu()
        self.v_trans = SPDTransform(self.d_in, self.d_out).cpu()

    def tensor_log(self, t):  # 4dim
        u, s, v = torch.svd(t)
        return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)

    def tensor_exp(self, t):  # 4dim
        # condition: t is symmetric!
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 1, 3, 2)

    def log_euclidean_distance(self, A, B):
        inner_term = self.tensor_log(A) - self.tensor_log(B)
        inner_multi = inner_term @ inner_term.permute(0, 1, 3, 2)
        _, s, _ = torch.svd(inner_multi)
        final = torch.sum(s, dim=-1)
        return final

    def LogEuclideanMean(self, weight, cov):
        # cov:[bs, #p, s, s]
        # weight:[bs, #p, #p]
        bs = cov.shape[0]
        num_p = cov.shape[1]
        size = cov.shape[2]
        cov = self.tensor_log(cov).view(bs, num_p, -1)
        output = weight @ cov  # [bs, #p, -1]
        output = output.view(bs, num_p, size, size)
        return self.tensor_exp(output)

    def forward(self, x, shape=None):
        if len(x.shape) == 3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)
        x = x.to(torch.float)  # patch:[b, #patch, c, c]
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs * m, self.d_in, self.d_in)
        Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out)
        K = self.k_trans(x).view(bs, m, self.d_out, self.d_out)
        V = self.v_trans(x).view(bs, m, self.d_out, self.d_out)

        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)

        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1)
        K_expand = K_expand.view(
            K_expand.shape[0],
            K_expand.shape[1] * K_expand.shape[2],
            K_expand.shape[3],
            K_expand.shape[4],
        )

        atten_energy = self.log_euclidean_distance(Q_expand, K_expand).view(
            V.shape[0], V.shape[1], V.shape[1]
        )
        atten_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + atten_energy))).permute(
            0, 2, 1
        )  # now row is c.c.

        # calculate outputs(v_i') of attention module
        output = self.LogEuclideanMean(atten_prob, V)

        output = output.view(V.shape[0], V.shape[1], self.d_out, self.d_out)

        shape = list(output.shape[:2])
        shape.append(-1)

        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape


class mAtt_bci(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        # FE
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2 = nn.BatchNorm2d(20)

        # E2R
        self.ract1 = E2R(epochs=epochs)
        # riemannian part
        self.att2 = AttentionManifold(20, 18)
        self.ract2 = SPDRectified()

        # R2E
        self.tangent = SPDTangentSpace(18)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(9 * 19 * epochs, 4, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.ract1(x)
        x, shape = self.att2(x)
        x = self.ract2(x)

        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# Staging build/example helpers
# ---------------------------------------------------------------------------

_EPOCHS = 4  # repo default for BCI-competition-IV-2a (22-channel EEG)


def build_matt():
    return mAtt_bci(epochs=_EPOCHS)


def example_input_matt():
    # x: [batch, 1, channels=22, samples]; time axis must be long enough
    # for `epochs` non-degenerate covariance windows (>=2 samples/window).
    return (torch.randn(2, 1, 22, 256),)


MENAGERIE_ENTRIES = [
    ("MAtt", "build_matt", "example_input_matt", 2022, "vendored-pytorch"),
]
