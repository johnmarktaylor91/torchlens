# SOURCE: vendored from deepmodeling/deepks-kit @ master
# Files combined below:
#   deepks/model/model.py (CorrNet + submodules: DenseNet, TraceEmbedding, ThermalEmbedding,
#     make_embedder, parse_actv_fn, mygelu, log_args, and the pad/mask helper functions it uses)
#   deepks/utils.py (only load_basis, get_shell_sec, load_elem_table, save_elem_table -- the
#     four helpers deepks/model/model.py imports)
#
# DeePKS (https://github.com/deepmodeling/deepks-kit) learns a machine-learned correction to
# DFT exchange-correlation energy (a "neural-network exchange-correlation functional") from
# local atomic descriptors; CorrNet is the real correction-energy network used throughout the
# package (deepks/scf/*, deepks/iterate/*). No architectural modification was made -- only
# imports were trimmed (deepks/utils.py's full module needs `ruamel.yaml`, which is installed,
# but only `load_basis`/`get_shell_sec`/`load_elem_table`/`save_elem_table` -- the four
# functions this file's imports actually use -- are reproduced here, verbatim, to keep the
# staging module import-light and dependency-free of pyscf, which is NOT installed and is only
# reached by load_basis() when given a pyscf basis-set *name* string; the default/None path
# used by this module's build function never reaches that branch).

import inspect
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from deepks/utils.py (only the four helpers deepks/model/model.py imports)
# ---------------------------------------------------------------------------

_zeta = 1.5 ** np.array([17, 13, 10, 7, 5, 3, 2, 1, 0, -1, -2, -3])
_coef = np.diag(np.ones(_zeta.size)) - np.diag(np.ones(_zeta.size - 1), k=1)
_table = np.concatenate([_zeta.reshape(-1, 1), _coef], axis=1)
DEFAULT_BASIS = [[0, *_table.tolist()], [1, *_table.tolist()], [2, *_table.tolist()]]
DEFAULT_SYMB = "Ne"


def load_basis(basis):
    if basis is None:
        return DEFAULT_BASIS
    elif isinstance(basis, np.ndarray) and basis.ndim == 2:
        return [[ll, *basis.tolist()] for ll in range(3)]
    elif not isinstance(basis, str):
        return basis
    elif basis.endswith(".npy"):
        table = np.load(basis)
        return [[ll, *table.tolist()] for ll in range(3)]
    elif basis.endswith(".npz"):
        all_tables = np.load(basis)
        return [
            [int(name.split("_L")[-1]) if "_L" in name else ii, *table.tolist()]
            for ii, (name, table) in enumerate(all_tables.items())
        ]
    else:
        from pyscf import gto  # local import matches upstream; only reached for named basis sets

        symb = DEFAULT_SYMB
        if "@" in basis:
            basis, symb = basis.split("@")
        return gto.basis.load(basis, symb=symb)


def get_shell_sec(basis):
    if not isinstance(basis, (list, tuple)):
        basis = load_basis(basis)
    shell_sec = []
    for l, c0, *cr in basis:
        nb = c0 if isinstance(c0, int) else (len(c0) - 1)
        shell_sec.extend([2 * l + 1] * nb)
    return shell_sec


def load_elem_table(filename):
    elem_list, elem_const = np.loadtxt(filename).T
    elem_list = elem_list.round().astype(int)
    return elem_list, elem_const


def save_elem_table(filename, elem_table):
    np.savetxt(filename, np.stack(elem_table).T, fmt=["%i", "%.16f"])


# ---------------------------------------------------------------------------
# from deepks/model/model.py
# ---------------------------------------------------------------------------

SCALE_EPS = 1e-8


def parse_actv_fn(code):
    if callable(code):
        return code
    assert type(code) is str
    lcode = code.lower()
    if lcode == "sigmoid":
        return torch.sigmoid
    if lcode == "tanh":
        return torch.tanh
    if lcode == "relu":
        return torch.relu
    if lcode == "softplus":
        return F.softplus
    if lcode == "silu":
        return F.silu
    if lcode == "gelu":
        return F.gelu
    if lcode == "mygelu":
        return mygelu
    raise ValueError(f"{code} is not a valid activation function")


def make_embedder(type, shell_sec, **kwargs):
    ltype = type.lower()
    if ltype in ("trace", "sum"):
        EmbdCls = TraceEmbedding
    elif ltype in ("thermal", "softmax"):
        EmbdCls = ThermalEmbedding
    else:
        raise ValueError(f"{type} is not a valid embedding type")
    embedder = EmbdCls(shell_sec, **kwargs)
    return embedder


def mygelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def log_args(name):
    def decorator(func):
        def warpper(self, *args, **kwargs):
            args_dict = inspect.getcallargs(func, self, *args, **kwargs)
            del args_dict["self"]
            setattr(self, name, args_dict)
            func(self, *args, **kwargs)

        return warpper

    return decorator


def make_shell_mask(shell_sec):
    lsize = len(shell_sec)
    msize = max(shell_sec)
    mask = torch.zeros(lsize, msize, dtype=bool)
    for l, m in enumerate(shell_sec):
        mask[l, :m] = 1
    return mask


def pad_lastdim(sequences, padding_value=0):
    max_size = sequences[0].size()
    front_dims = max_size[:-1]
    max_len = max([s.size(-1) for s in sequences])
    out_dims = front_dims + (len(sequences), max_len)
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(-1)
        out_tensor[..., i, :length] = tensor
    return out_tensor


def pad_masked(tensor, mask, padding_value=0):
    assert tensor.shape[-1] == mask.sum()
    new_shape = tensor.shape[:-1] + mask.shape
    return tensor.new_full(new_shape, padding_value).masked_scatter_(mask, tensor)


def unpad_lastdim(padded, length_list):
    return [padded[..., i, :length] for i, length in enumerate(length_list)]


def unpad_masked(padded, mask):
    new_shape = padded.shape[: -mask.ndim] + (mask.sum(),)
    return torch.masked_select(padded, mask).reshape(new_shape)


def masked_softmax(input, mask, dim=-1):
    exps = torch.exp(input - input.max(dim=dim, keepdim=True)[0])
    mexps = exps * mask.to(exps)
    msums = mexps.sum(dim=dim, keepdim=True).clamp(1e-10)
    return mexps / msums


class DenseNet(nn.Module):
    def __init__(self, sizes, actv_fn=torch.relu, use_resnet=True, with_dt=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])]
        )
        self.actv_fn = actv_fn
        self.use_resnet = use_resnet
        if with_dt:
            self.dts = nn.ParameterList(
                [nn.Parameter(torch.normal(torch.ones(out_f), std=0.01)) for out_f in sizes[1:]]
            )
        else:
            self.dts = None

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            tmp = layer(x)
            if i < len(self.layers) - 1:
                tmp = self.actv_fn(tmp)
            if self.use_resnet and layer.in_features == layer.out_features:
                if self.dts is not None:
                    tmp = tmp * self.dts[i]
                x = x + tmp
            else:
                x = tmp
        return x


class TraceEmbedding(nn.Module):
    def __init__(self, shell_sec):
        super().__init__()
        self.shell_sec = shell_sec
        self.ndesc = len(shell_sec)

    def forward(self, x):
        x_shells = x.split(self.shell_sec, dim=-1)
        tr_shells = [sx.sum(-1, keepdim=True) for sx in x_shells]
        return torch.cat(tr_shells, dim=-1)


class ThermalEmbedding(nn.Module):
    def __init__(self, shell_sec, embd_sizes=None, init_beta=5.0, momentum=None, max_memory=1000):
        super().__init__()
        self.shell_sec = shell_sec
        self.register_buffer("shell_mask", make_shell_mask(shell_sec), False)
        if embd_sizes is None:
            embd_sizes = shell_sec
        if isinstance(embd_sizes, int):
            embd_sizes = [embd_sizes] * len(shell_sec)
        assert len(embd_sizes) == len(shell_sec)
        self.embd_sizes = embd_sizes
        self.register_buffer("embd_mask", make_shell_mask(embd_sizes), False)
        self.ndesc = sum(embd_sizes)
        self.beta = nn.Parameter(
            pad_lastdim([torch.linspace(init_beta, -init_beta, ne) for ne in embd_sizes])
        )
        self.momentum = momentum
        self.max_memory = max_memory
        self.register_buffer("running_mean", torch.zeros(len(shell_sec)))
        self.register_buffer("running_var", torch.ones(len(shell_sec)))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x_padded = pad_masked(x, self.shell_mask, 0.0)
        if self.training:
            self.update_running_stats(x_padded)
        nx_padded = (
            (x_padded - self.running_mean.unsqueeze(-1))
            / (self.running_var.sqrt().unsqueeze(-1) + SCALE_EPS)
            * self.shell_mask.to(x_padded)
        )
        weight = masked_softmax(
            torch.einsum("...lm,lp->...lmp", nx_padded, -self.beta),
            self.shell_mask.unsqueeze(-1),
            dim=-2,
        )
        desc_padded = torch.einsum("...m,...mp->...p", x_padded, weight)
        return unpad_masked(desc_padded, self.embd_mask)

    def update_running_stats(self, x_padded):
        self.num_batches_tracked += 1
        if self.momentum is None and self.num_batches_tracked > self.max_memory:
            return
        exp_factor = 1.0 - 1.0 / float(self.num_batches_tracked)
        if self.momentum is not None:
            exp_factor = max(exp_factor, self.momentum)
        with torch.no_grad():
            fmask = self.shell_mask.to(x_padded)
            pad_portion = fmask.mean(-1)
            x_masked = x_padded * fmask
            reduced_dim = (*range(x_masked.ndim - 2), -1)
            batch_mean = x_masked.mean(reduced_dim) / pad_portion
            batch_var = x_masked.var(reduced_dim) / pad_portion
            self.running_mean[:] = exp_factor * self.running_mean + (1 - exp_factor) * batch_mean
            self.running_var[:] = exp_factor * self.running_var + (1 - exp_factor) * batch_var

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()


class CorrNet(nn.Module):
    """DeePKS's real machine-learned XC-energy correction network. Consumes projected
    atomic descriptors (nframes x natom x nfeature) and outputs a scalar correction energy."""

    @log_args("_init_args")
    def __init__(
        self,
        input_dim,
        hidden_sizes=(100, 100, 100),
        actv_fn="gelu",
        use_resnet=True,
        embedding=None,
        proj_basis=None,
        elem_table=None,
        input_shift=0,
        input_scale=1,
        output_scale=1,
    ):
        super().__init__()
        actv_fn = parse_actv_fn(actv_fn)
        self.input_dim = input_dim
        self._pbas = load_basis(proj_basis)
        self._init_args["proj_basis"] = self._pbas
        self.shell_sec = None
        if isinstance(elem_table, str):
            elem_table = load_elem_table(elem_table)
            self._init_args["elem_table"] = elem_table
        self.elem_table = elem_table
        self.elem_dict = None if elem_table is None else dict(zip(*elem_table))
        self.linear = nn.Linear(input_dim, 1).double()
        ndesc = input_dim
        self.embedder = None
        if embedding is not None:
            if isinstance(embedding, str):
                embedding = {"type": embedding}
            assert isinstance(embedding, dict)
            raw_shell_sec = get_shell_sec(self._pbas)
            self.shell_sec = raw_shell_sec * (input_dim // sum(raw_shell_sec))
            assert sum(self.shell_sec) == input_dim
            self.embedder = make_embedder(**embedding, shell_sec=self.shell_sec).double()
            self.linear.requires_grad_(False)
            ndesc = self.embedder.ndesc
        layer_sizes = [ndesc, *hidden_sizes, 1]
        self.densenet = DenseNet(layer_sizes, actv_fn, use_resnet).double()
        self.input_shift = nn.Parameter(
            torch.tensor(input_shift, dtype=torch.float64).expand(input_dim).clone(),
            requires_grad=False,
        )
        self.input_scale = nn.Parameter(
            torch.tensor(input_scale, dtype=torch.float64).expand(input_dim).clone(),
            requires_grad=False,
        )
        self.output_scale = nn.Parameter(
            torch.tensor(output_scale, dtype=torch.float64), requires_grad=False
        )
        self.energy_const = nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)

    def forward(self, x):
        # x: nframes x natom x nfeature
        x = (x - self.input_shift) / (self.input_scale + SCALE_EPS)
        l = self.linear(x)
        if self.embedder is not None:
            x = self.embedder(x)
        y = self.densenet(x)
        y = y / self.output_scale + l
        e = y.sum(-2) + self.energy_const
        return e

    def get_elem_const(self, elems):
        if self.elem_dict is None:
            return 0.0
        return sum(self.elem_dict[ee] for ee in elems)

    def set_normalization(self, shift=None, scale=None):
        dtype = self.input_scale.dtype
        if shift is not None:
            self.input_shift.data[:] = torch.tensor(shift, dtype=dtype)
        if scale is not None:
            self.input_scale.data[:] = torch.tensor(scale, dtype=dtype)

    def set_prefitting(self, weight, bias, trainable=False):
        dtype = self.linear.weight.dtype
        self.linear.weight.data[:] = torch.tensor(weight, dtype=dtype).reshape(-1)
        self.linear.bias.data[:] = torch.tensor(bias, dtype=dtype).reshape(-1)
        self.linear.requires_grad_(trainable)

    def set_energy_const(self, const):
        dtype = self.energy_const.dtype
        self.energy_const.data = torch.tensor(const, dtype=dtype).reshape([])


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_deepks():
    """Tiny DeePKS CorrNet using the real default (None -> DEFAULT_BASIS) projection basis
    and the 'trace' (sum) embedding, matching a typical deepks-kit training config."""
    shell_sec = get_shell_sec(None)  # [1]*12 + [3]*12 + [5]*12 -> sum = 108
    input_dim = sum(shell_sec)
    return CorrNet(
        input_dim=input_dim,
        hidden_sizes=(16, 16),
        actv_fn="gelu",
        use_resnet=True,
        embedding="trace",
        proj_basis=None,
    )


def example_input_deepks():
    g = torch.Generator().manual_seed(0)
    nframes, natom = 2, 3
    shell_sec = get_shell_sec(None)
    input_dim = sum(shell_sec)
    x = torch.rand(nframes, natom, input_dim, generator=g, dtype=torch.float64)
    return (x,)


MENAGERIE_ENTRIES = [
    ("DeePKS", build_deepks, example_input_deepks, 2021, MENAGERIE_ZOO),
]
