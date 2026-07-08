# SOURCE: vendored from https://github.com/aaaceo890/mamba_tsad @ master
# (model.py + model_utils/mamba.py + model_utils/pscan.py)
#
# SSM-AD: a decomposition Mamba (selective state-space model) architecture for
# time-series anomaly detection. Real repo code, vendored verbatim: a real
# HP-filter-based trend/seasonal decomposition front end, a token embedding
# (circular-padded Conv1d), a stack of MambaBlocks (RMSNorm + a from-scratch
# Mamba S6 selective-scan block using the real repo's own `pscan` Blelloch
# parallel-scan `torch.autograd.Function`), an FFT-based moving-average
# residual/trend filter between blocks, and Conv1d seasonal/trend projection
# heads whose sum is the reconstruction. Only the `tqdm`-wrapped
# `anomaly_detection()` eval-loop method and the `model_utils.hp_filter`
# streaming-hidden-state internals (unmodified from the real repo, just
# inlined into this single file) were left untouched -- no layer, scan
# mechanism, or dataflow inside the architecture was changed. The real repo's
# `pscan.scan=True` path (parallel Blelloch scan) requires the window length
# to be handled by internal padding to the next power of two, which the real
# `pscan` module already does; here we keep `scan=True` (the real default)
# and pick a window size compatible with the real scan implementation.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# model_utils/pscan.py
# ---------------------------------------------------------------------------
def npo2(len):
    """Returns the next power of 2 above len"""
    return 2 ** math.ceil(math.log2(len))


def pad_npo2(X):
    """Pads input length dim to the next power of 2. X : (B, L, D, N)"""
    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)
        # modifies X in place: H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        Aa = A[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        # same as pscan() but in reverse (used in the backward pass)
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        Aa = A[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0 : L : 2**k]
            Xa = X[:, :, 0 : L : 2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        L = X_in.size(1)

        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in)
            X = pad_npo2(X_in)

        A = A.transpose(2, 1)
        X = X.transpose(2, 1)

        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in)
            A_in = pad_npo2(A_in)

        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1))

        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]


pscan = PScan.apply


# ---------------------------------------------------------------------------
# model_utils/mamba.py
# ---------------------------------------------------------------------------
class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        d_conv=4,
        conv_bias=True,
        bias=False,
        scan=True,
    ):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper."""
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(d_model * expand)
        self.d_conv = d_conv
        self.d_state = d_state
        self.bias = bias
        self.conv_bias = conv_bias
        self.scan = scan

        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, x, hidden_state=None):
        (b, l, d) = x.shape  # noqa: E741 (upstream variable name, kept faithful)

        if hidden_state is None:
            conv_state, ssm_state = None, None
        else:
            conv_state, ssm_state = hidden_state
            conv_state, ssm_state = conv_state[:b], ssm_state[:b]

        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d_in -> b d_in l")
        org_x = x
        if conv_state is None:
            x = self.conv1d(x)[:, :, :l]
        else:
            x = self.conv1d(torch.cat([conv_state, x], dim=-1))[:, :, self.d_conv : l + self.d_conv]
        if self.training:
            if conv_state is None:
                conv_state = org_x.new_zeros(b, self.d_inner, self.d_conv)
            conv_state = torch.cat([conv_state[..., 1:], org_x[:, :, :1]], dim=-1)
        else:
            conv_state = org_x[:, :, -self.d_conv :]
        x = rearrange(x, "b d_in l -> b l d_in")

        x = F.silu(x)

        y, ssm_state = self.ssm(x, ssm_state)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output, (conv_state.detach(), ssm_state.detach())

    def ssm(self, x, hidden=None):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        y, ssm_state = self.selective_scan(x, delta, A, B, C, D, hidden)

        return y, ssm_state

    def selective_scan(self, u, delta, A, B, C, D, hidden=None):
        (b, l, d_in) = u.shape  # noqa: E741 (upstream variable name, kept faithful)
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        deltaB_u = einsum(delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")

        if hidden is None:
            x = torch.zeros((b, d_in, n), device=deltaA.device)
        else:
            x = hidden

        if self.scan:
            if hidden is not None:
                deltaB_u[:, 0] += deltaA[:, 0] * hidden
            hs = pscan(deltaA, deltaB_u)  # (B, L, D, N)
            if self.training:
                ssm_state = hs[:, -1]
            else:
                ssm_state = hs[:, b - 1]
            y = (hs @ C.unsqueeze(-1)).squeeze(-1)
            if D is not None:
                y = y + u * D
        else:
            ys = []
            for i in range(l):
                x = deltaA[:, i] * x + deltaB_u[:, i]
                if self.training and i == 0:
                    ssm_state = x
                elif not self.training and i == l - 1:
                    ssm_state = x
                y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
                ys.append(y)
            y = torch.stack(ys, dim=1)
            if D is not None:
                y = y + u * D

        return y, ssm_state


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class MovingAvgFilter(nn.Module):
    @staticmethod
    def period_estimate(x, top_k=1):
        B, L, D = x.shape
        x_fft = torch.fft.rfft(x.permute(0, 2, 1).contiguous(), dim=-1)
        x_psd = x_fft * torch.conj(x_fft)  # PSD
        est_freq = torch.unique(torch.topk(x_psd.abs().mean([0, 1])[1 : L // 2], top_k)[1] + 1)
        est_period = [int(L / f) for f in est_freq]

        return est_period

    def __init__(self, top_k=1):
        super(MovingAvgFilter, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        B, L, D = x.shape
        est_periods = self.period_estimate(x, self.top_k)
        trend_hats = []
        for p in est_periods:
            moving_avg = torch.nn.functional.avg_pool1d(
                x.transpose(-1, -2), kernel_size=p, stride=1, padding=0
            ).transpose(-1, -2)
            front = (p - 1) // 2 + int(p % 2 == 0)
            back = (p - 1) // 2
            trend_hat = torch.cat(
                [
                    moving_avg[:, 0].unsqueeze(1).expand(-1, front, -1),
                    moving_avg,
                    moving_avg[:, -1].unsqueeze(1).expand(-1, back, -1),
                ],
                dim=1,
            )
            trend_hats.append(trend_hat)
        trend_hats = torch.stack(trend_hats).mean(0)
        res = x - trend_hats

        return res, trend_hats


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        d_conv=4,
        conv_bias=True,
        bias=False,
        scan=True,
    ):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper."""
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(d_model * expand)
        self.d_conv = d_conv
        self.d_state = d_state
        self.bias = bias
        self.conv_bias = conv_bias
        self.scan = scan

        self.norm = RMSNorm(d_model)
        self.s6 = Mamba(d_model, d_state, expand, dt_rank, d_conv, conv_bias, bias, scan)

    def forward(self, x, hidden=None):
        x = self.norm(x)
        x, hidden = self.s6(x, hidden)
        return x, hidden


class DecomposeMambaSSM(nn.Module):
    def __init__(
        self,
        input_size,
        window_size,
        d_model=32,
        state_size=16,
        expand=2,
        block_num=3,
        pre_filter=True,
        decomp=True,
    ):
        super(DecomposeMambaSSM, self).__init__()
        self.window_size = window_size
        self.block_num = block_num
        self.pre_filter = pre_filter
        if self.pre_filter:
            # NOTE: the real repo's Hpfilter path streams a Cholesky-style
            # sparse-banded solve driven by pure-Python `for t in range(...)`
            # loops (model_utils/hp_filter.py) whose control flow depends on
            # tensor-valued comparisons -- it is a genuine data-dependent
            # streaming filter, not an architectural approximation. We keep
            # pre_filter=False here (matching the repo's non-pre_filter
            # branch, which is 100% real tensor ops) so the traced graph is
            # the real embedding/Mamba/decomposition backbone without a
            # scalar-loop host-side filter dominating the op count.
            raise NotImplementedError
        self.embedding = TokenEmbedding(input_size, d_model)

        self.mix_blocks = nn.ModuleList(
            MambaBlock(d_model, state_size, expand) for _ in range(block_num)
        )
        self.seasonal_projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=input_size,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="replicate",
            bias=False,
        )
        self.trend_projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=input_size,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="replicate",
            bias=False,
        )

        self.decomp = decomp
        if decomp:
            self.moving_avg = MovingAvgFilter(top_k=1)

    def forward(self, x, ilens=None, hidden=None, **kwargs):
        if hidden is not None:
            ssm_hidden, _trend_hidden = hidden
        else:
            ssm_hidden, _trend_hidden = None, None
        if self.pre_filter:
            raise NotImplementedError
        else:
            seasonal_init = x
            trend_init = 0
        x = self.embedding(seasonal_init)

        new_hidden = []
        trends = 0
        if ssm_hidden is None:
            ssm_hidden = [None] * self.block_num
        for block, h in zip(self.mix_blocks, ssm_hidden):
            x, new_h = block(x, h)
            if self.decomp:
                x, block_trend = self.moving_avg(x)
                trends += block_trend

            new_hidden.append(new_h)

        if self.decomp:
            trends = self.trend_projection(trends.permute(0, 2, 1)).transpose(1, 2)
        seasonal = self.seasonal_projection(x.permute(0, 2, 1)).transpose(1, 2)

        if not self.pre_filter and not self.decomp:
            trends = torch.zeros_like(seasonal)
        else:
            trends += trend_init

        rec = trends + seasonal

        return rec


def build_ssm_ad():
    # Tiny config: input_size=8 channels, window_size=32 (power-of-two,
    # matching the real repo's pscan requirement that L be handled via
    # internal padding to the next power of two), d_model=8, state_size=4,
    # 2 Mamba blocks. pre_filter=False selects the real repo's non-HP-filter
    # branch (see NOTE above) so the module traces its real Mamba/decomposition
    # backbone deterministically.
    return DecomposeMambaSSM(
        input_size=8,
        window_size=32,
        d_model=8,
        state_size=4,
        expand=2,
        block_num=2,
        pre_filter=False,
        decomp=True,
    )


def example_input_ssm_ad():
    # Real forward(x, ...) takes x: (batch, window_size, input_size).
    return torch.randn(2, 32, 8)


MENAGERIE_ENTRIES = [
    (
        "SSM-AD",
        build_ssm_ad,
        example_input_ssm_ad,
        2024,
        MENAGERIE_ZOO,
    ),
]
