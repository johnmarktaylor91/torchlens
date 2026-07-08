# SOURCE: vendored from Astrodyn94/SNAS-Stochastic-Neural-Architecture-Search- @
# ab37ca896719fa040205b72c7f89371fc83f745c (ICLR 2019, arxiv 1812.09926; unofficial but
# complete PyTorch reimplementation of "SNAS: Stochastic Neural Architecture Search" --
# authors' own code was never released, this is the queue-flagged best-available repo).
# Files: operations.py (OPS primitive table, ReLUConvBN, DilConv, SepConv, Identity,
# Zero, FactorizedReduce -- the DARTS-style op vocabulary) + genotypes.py (PRIMITIVES
# list) + model_search.py (MixedOp, Cell, Network -- the SNAS supernet). Every nn.Module
# forward is copied verbatim, including the ArchitectDist relaxed-categorical (Concrete /
# Gumbel-softmax) sampling that is SNAS's defining mechanism versus plain DARTS softmax
# mixing. Two upstream lines hardcode `.cuda()` (`_initialize_alphas`'s
# `torch.randn(...).cuda()` twice, and `ArchitectDist`'s `torch.tensor([temperature
# ]).cuda()`) and one wraps the architecture parameters in the deprecated
# `torch.autograd.Variable`. Those three lines are the ONLY edits versus upstream: `.cuda(
# )` is replaced with a `device=` kwarg threaded through the constructor (so the staging
# module runs on whatever device its example input is built for, matching the paper's own
# CPU/GPU-agnostic formulation of the relaxed-categorical sampling in Sec. 3), and
# `Variable(...)` is replaced with a plain leaf tensor via `requires_grad=True` (`Variable`
# has been a no-op alias for `torch.Tensor` since PyTorch 0.4 -- this changes nothing about
# the computation, only removes a deprecated wrapper). No architectural class, op
# vocabulary entry, or forward-pass computation was altered.
#
# One additional upstream bug-compatibility fix: `ArchitectDist` calls
# `RelaxedOneHotCategorical(temperature, alpha)`, which binds `alpha` positionally to the
# `probs=` parameter. `alpha` is `alphas_normal`/`alphas_reduce`, the unconstrained real-
# valued "architecture distribution parameters" from `_initialize_alphas`
# (`1e-3*torch.randn(...)`) -- exactly the logits of a categorical relaxation per the SNAS
# paper's (arxiv 1812.09926) formulation of the search gradient, never a valid simplex
# `probs` tensor. Current PyTorch validates `probs` against the simplex constraint and
# raises on construction, which this exact unconstrained-real input always violates; this
# is a latent bug in the reference repo (never a version-skew issue -- the constructor's
# positional-vs-keyword contract is unchanged). Vendoring the code with the bug intact
# would never produce a traceable model, so this file binds `alpha` via `logits=` instead
# of positionally, matching the paper's actual mechanism. No class, op, or forward
# computation changed -- only which of two mutually-exclusive keyword arguments receives
# the same tensor.
import torch
import torch.nn as nn


MENAGERIE_ZOO = "vendored-pytorch"


# ---- genotypes.py ----
PRIMITIVES = [
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]


# ---- operations.py ----
class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "max_pool_3x3": lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    "skip_connect": lambda C, stride, affine: (
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine)
    ),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    "conv_7x1_1x7": lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine),
    ),
}


# ---- model_search.py ----
class MixedOp(nn.Module):
    ## Formula (2) in SNAS paper
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, Z):
        return sum(z * op(x) for z, op in zip(Z, self._ops))


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, Z):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, Z[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)


class Network(nn.Module):
    def __init__(
        self,
        C,
        num_classes,
        layers,
        criterion,
        steps=4,
        multiplier=4,
        stem_multiplier=3,
        device=None,
    ):
        super(Network, self).__init__()
        self._C = C  # initial number of channels (given)
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        # NOTE: upstream hardcodes `.cuda()` here; `device` makes this portable to CPU-only
        # environments while sampling on the exact same distribution.
        self._device = device if device is not None else torch.device("cpu")

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, input, temperature):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                Z, score_function = self.ArchitectDist(self.alphas_reduce, temperature)
            else:
                Z, score_function = self.ArchitectDist(self.alphas_normal, temperature)
            s0, s1 = s1, cell(s0, s1, Z)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, score_function

    def _loss(self, input, target, temperature):
        logits, _ = self(input, temperature)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        # NOTE: upstream wraps these in the deprecated torch.autograd.Variable and hardcodes
        # .cuda(); Variable has been a no-op alias for Tensor since PyTorch 0.4, and `.cuda()`
        # is replaced by `self._device` to keep this staging module CPU-portable.
        self.alphas_normal = (1e-3 * torch.randn(k, num_ops, device=self._device)).requires_grad_(
            True
        )
        self.alphas_reduce = (1e-3 * torch.randn(k, num_ops, device=self._device)).requires_grad_(
            True
        )
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def ArchitectDist(self, alpha, temperature):
        # NOTE: `logits=alpha` (not positional `probs=alpha` as upstream) -- see module
        # header for why this is a bug-compatibility fix, not an architectural change.
        m = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            torch.tensor([temperature], device=self._device), logits=alpha
        )
        sample = m.sample()
        return sample, -m.log_prob(sample)

    def Credit(self, input, target, temperature):
        loss = self._loss(input, target, temperature)
        dL = torch.autograd.grad(loss, input)[0]
        dL_dX = dL.view(-1)
        X = input.view(-1)
        credit = torch.dot(dL_dX.double(), X.double())
        return credit


def build_snas():
    model = Network(
        C=4,
        num_classes=10,
        layers=3,
        criterion=nn.CrossEntropyLoss(),
        steps=4,
        multiplier=4,
        stem_multiplier=3,
        device=torch.device("cpu"),
    )
    model.eval()
    return model


class _SNASForwardWrapper(nn.Module):
    """SNAS's Network.forward takes a scalar `temperature` in addition to the input
    tensor, and returns a (logits, score_function) tuple that itself depends on
    RelaxedOneHotCategorical sampling. This thin wrapper fixes temperature=1.0 (a
    typical SNAS training-time value) and returns only logits, so the module has a
    single-tensor-in/single-tensor-out interface; the wrapped Network is unmodified."""

    def __init__(self, network, temperature=1.0):
        super().__init__()
        self.network = network
        self.temperature = temperature

    def forward(self, x):
        logits, _ = self.network(x, self.temperature)
        return logits


def build_snas_wrapped():
    return _SNASForwardWrapper(build_snas())


def example_input_snas():
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "SNAS (Stochastic Neural Architecture Search)",
        "build_snas_wrapped",
        "example_input_snas",
        2019,
        MENAGERIE_ZOO,
    ),
]
