# SOURCE: vendored from openai/video-pre-training @ main (files: inverse_dynamics_model.py,
# lib/policy.py, lib/impala_cnn.py, lib/util.py, lib/xf.py, lib/mlp.py, lib/masked_attention.py,
# lib/misc.py, lib/torch_util.py, lib/tree_util.py -- fetched 2026-07-02).
#
# This vendors the REAL VPT (Video PreTraining) backbone architecture: ImpalaCNN visual encoder
# feeding into a stack of residual recurrent blocks (transformer self-attention with a banded
# causal cache, or multi-layer LSTM), exactly as defined in `lib/policy.py::MinecraftPolicy` /
# `InverseActionNet` and their support modules. This is the network used for both the VPT
# behavioral-cloning policy AND the Inverse Dynamics Model (IDM) that VPT trains on unlabeled
# YouTube video to pseudo-label actions (the "VoT" candidate in the queue is this IDM/backbone).
#
# NOT vendored (upstream gym3/gym-dependent, action-space plumbing only, no architecture):
# `lib/action_head.py`, `lib/action_mapping.py`, `lib/actions.py`, `agent.py`,
# `lib/scaled_mse_head.py`, `lib/normalize_ewma.py`, `lib/minecraft_util.py::store_args`
# (re-inlined below, functionally identical, to drop the gym3 import chain). The action/value
# heads attached in `MinecraftAgentPolicy`/`InverseActionPolicy` are thin `nn.Linear`-based
# heads over the backbone's `hidsize` output and add no architectural novelty; the traced
# subject here is the actual shared trunk (`MinecraftPolicy` in transformer-recurrence mode,
# the config VPT ships as its released checkpoints).
#
# Code below is the upstream source with only mechanical edits: gym3-dependent imports removed,
# cross-file imports flattened into this single module, unused kwargs/plumbing left intact.

import functools
import functools as _functools
import inspect as _inspect
import math
from copy import deepcopy
from typing import Dict, List, Optional

import torch as th
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# lib/minecraft_util.py::store_args (re-inlined; original imports lib.action_head -> gym3)
# ---------------------------------------------------------------------------
def store_args(method):
    """Stores provided method args as instance attributes."""
    argspec = _inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(zip(argspec.args[-len(argspec.defaults) :], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @_functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        args = defaults.copy()
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


# ---------------------------------------------------------------------------
# lib/misc.py
# ---------------------------------------------------------------------------
def intprod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def safezip(*args):
    args = [list(a) for a in args]
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(zip(*args))


def transpose(x, before, after):
    assert sorted(before) == sorted(after), f"cannot transpose {before} to {after}"
    assert x.ndim == len(before), (
        f"before spec '{before}' has length {len(before)} but x has {x.ndim} dimensions: {tuple(x.shape)}"
    )
    return x.permute(tuple(before.index(i) for i in after))


def transpose_undo(x, before, after, *, undo=None):
    return (
        transpose(x, before, after),
        compose_undo(undo, lambda x: transpose(x, before=after, after=before)),
    )


def compose_undo(u1, u2):
    assert u2 is not None
    if u1 is None:
        return u2

    def u(x):
        x = u2(x)
        x = u1(x)
        return x

    return u


NO_BIND = "__nobind"


def _parse_reshape_str(s, kind):
    assert kind in ("before", "after")
    result = []
    n_underscores = 0
    for i, part in enumerate(s.split(",")):
        part = part.strip()
        if part == "?" and kind == "before":
            result.append([f"__{i}"])
        elif part == "_":
            result.append([f"{NO_BIND}_{n_underscores}"])
            n_underscores += 1
        else:
            result.append([term.strip() for term in part.split("*")])
    return result


def _infer_part(part, concrete_dim, known, index, full_shape):
    if type(part) is int:
        return part
    assert isinstance(part, list), part
    lits = []
    syms = []
    for term in part:
        if type(term) is int:
            lits.append(term)
        elif type(term) is str:
            syms.append(term)
        else:
            raise TypeError(f"got {type(term)} but expected int or str")
    int_part = 1
    for x in lits:
        int_part *= x
    if len(syms) == 0:
        return int_part
    elif len(syms) == 1 and concrete_dim is not None:
        assert concrete_dim % int_part == 0, (
            f"{concrete_dim} % {int_part} != 0 (at index {index}, full shape is {full_shape})"
        )
        v = concrete_dim // int_part
        if syms[0] in known:
            assert known[syms[0]] == v, (
                f"known value for {syms[0]} is {known[syms[0]]} but found value {v} at index {index} (full shape is {full_shape})"
            )
        else:
            known[syms[0]] = v
        return concrete_dim
    else:
        for i in range(len(syms)):
            if syms[i] in known:
                syms[i] = known[syms[i]]
            else:
                try:
                    syms[i] = int(syms[i])
                except ValueError:
                    pass
        return lits + syms


def _infer_step(args):
    known, desc, shape = args
    new_known = known.copy()
    new_desc = desc.copy()
    for i in range(len(desc)):
        concrete_dim = None if shape is None else shape[i]
        new_desc[i] = _infer_part(
            part=desc[i], concrete_dim=concrete_dim, known=new_known, index=i, full_shape=shape
        )
    return new_known, new_desc, shape


def _infer(known, desc, shape):
    if shape is not None:
        assert len(desc) == len(shape), (
            f"desc has length {len(desc)} but shape has length {len(shape)} (shape={shape})"
        )
    known, desc, shape = fixed_point(_infer_step, (known, desc, shape))
    return desc, known


def _default_eq(a, b):
    return a == b


def fixed_point(f, x, eq=None):
    if eq is None:
        eq = _default_eq
    while True:
        new_x = f(x)
        if eq(x, new_x):
            return x
        else:
            x = new_x


def _infer_question_mark(x, total_product):
    try:
        question_mark_index = x.index(["?"])
    except ValueError:
        return x
    observed_product = 1
    for i in range(len(x)):
        if i != question_mark_index:
            assert type(x[i]) is int, (
                f"when there is a question mark, there can be no other unknown values (full list: {x})"
            )
            observed_product *= x[i]
    assert observed_product and total_product % observed_product == 0, (
        f"{total_product} is not divisible by {observed_product}"
    )
    value = total_product // observed_product
    x = x.copy()
    x[question_mark_index] = value
    return x


def _ground(x, known, infer_question_mark_with=None):
    x, known = _infer(known=known, desc=x, shape=None)
    if infer_question_mark_with:
        x = _infer_question_mark(x, infer_question_mark_with)
    for part in x:
        assert type(part) is int, f"cannot infer value of {part}"
    return x


def _handle_ellipsis(x, before, after):
    ell = ["..."]
    try:
        i = before.index(ell)
        length = len(x.shape) - len(before) + 1
        ellipsis_value = x.shape[i : i + length]
        ellipsis_value = list(ellipsis_value)
        before = before[:i] + ellipsis_value + before[i + 1 :]
    except ValueError:
        pass
    try:
        i = after.index(ell)
        after = after[:i] + ellipsis_value + after[i + 1 :]
    except ValueError:
        pass
    except UnboundLocalError as e:
        raise ValueError(
            "there cannot be an ellipsis in 'after' unless there is an ellipsis in 'before'"
        ) from e
    return before, after


def reshape_undo(inp, before, after, *, undo=None, known=None, **kwargs):
    if known:
        known = {**kwargs, **known}
    else:
        known = kwargs
    assert type(before) is type(after), f"{type(before)} != {type(after)}"
    assert isinstance(inp, th.Tensor), f"require tensor but got {type(inp)}"
    assert isinstance(before, (str, list)), f"require str or list but got {type(before)}"
    if isinstance(before, str):
        before = _parse_reshape_str(before, "before")
        after = _parse_reshape_str(after, "after")
        before, after = _handle_ellipsis(inp, before, after)
    before_saved, after_saved = before, after
    before, known = _infer(known=known, desc=before, shape=inp.shape)
    before = _ground(before, known, product(inp.shape))
    after = _ground(after, known, product(inp.shape))
    known = {k: v for k, v in known.items() if not k.startswith(NO_BIND)}
    assert tuple(inp.shape) == tuple(before), f"expected shape {before} but got shape {inp.shape}"
    assert product(inp.shape) == product(after), (
        f"cannot reshape {inp.shape} to {after} because the number of elements does not match"
    )
    return (
        inp.reshape(after),
        compose_undo(undo, lambda inp: reshape(inp, after_saved, before_saved, known=known)),
    )


def reshape(*args, **kwargs):
    x, _ = reshape_undo(*args, **kwargs)
    return x


def product(xs, one=1):
    result = one
    for x in xs:
        result = result * x
    return result


def exact_div(a, b):
    assert a % b == 0, f"{a} is not divisible by {b}"
    return a // b


def flatten_image(x):
    """Flattens last three dims"""
    *batch_shape, h, w, c = x.shape
    return x.reshape((*batch_shape, h * w * c))


def sequential(layers, x, *args, diag_name=None, use_checkpoint=False):
    for layer in layers:
        x = layer(x, *args)
    return x


# ---------------------------------------------------------------------------
# lib/torch_util.py (subset: dtype/device helpers + NormedLinear/LayerNorm)
# ---------------------------------------------------------------------------
DEFAULT_DEVICE = th.device("cpu")


def dev():
    return DEFAULT_DEVICE


def zeros(*args, **kwargs):
    return th.zeros(*args, **kwargs, device=dev())


def NormedLinear(*args, scale=1.0, dtype=th.float32, **kwargs):
    """nn.Linear but with normalized fan-in init"""
    dtype = parse_dtype(dtype)
    out = nn.Linear(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out


def LayerNorm(*args, dtype=th.float32, **kwargs):
    dtype = parse_dtype(dtype)
    out = nn.LayerNorm(*args, **kwargs)
    out.weight.no_scale = True
    return out


def parse_dtype(x):
    if isinstance(x, th.dtype):
        return x
    elif isinstance(x, str):
        table = {
            "float32": th.float32,
            "float": th.float32,
            "float64": th.float64,
            "double": th.float64,
            "float16": th.float16,
            "half": th.float16,
        }
        if x not in table:
            raise ValueError(f"cannot parse {x} as a dtype")
        return table[x]
    else:
        raise TypeError(f"cannot parse {type(x)} as dtype")


# ---------------------------------------------------------------------------
# lib/mlp.py
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, insize, nhidlayer, outsize, hidsize, hidactiv, dtype=th.float32):
        super().__init__()
        self.insize = insize
        self.nhidlayer = nhidlayer
        self.outsize = outsize
        in_sizes = [insize] + [hidsize] * nhidlayer
        out_sizes = [hidsize] * nhidlayer + [outsize]
        self.layers = nn.ModuleList(
            [
                NormedLinear(insize, outsize, dtype=dtype)
                for (insize, outsize) in safezip(in_sizes, out_sizes)
            ]
        )
        self.hidactiv = hidactiv

    def forward(self, x):
        *hidlayers, finallayer = self.layers
        for layer in hidlayers:
            x = layer(x)
            x = self.hidactiv(x)
        x = finallayer(x)
        return x

    @property
    def output_shape(self):
        return (self.outsize,)


# ---------------------------------------------------------------------------
# lib/xf.py (transformer / sparse-attention implementation)
# ---------------------------------------------------------------------------
SENTINEL = 0.1337


def attention(
    Q_bte,
    K_bTe,
    V_bTe,
    dtype,
    mask=True,
    extra_btT=None,
    maxlen=None,
    check_sentinel=False,
    use_muP_factor=False,
):
    assert Q_bte.dtype == K_bTe.dtype == dtype, (
        f"{Q_bte.dtype}, {K_bTe.dtype}, {dtype} must all match"
    )
    e = Q_bte.shape[2]
    if check_sentinel:
        invalid = (K_bTe == SENTINEL).int().sum(dim=-1) == e
        invalid = reshape(invalid, "b, T", "b, 1, T")
    if isinstance(mask, th.Tensor):
        bias = (~mask).float() * -1e9
    elif mask:
        bias = get_attn_bias_cached(
            Q_bte.shape[1], K_bTe.shape[1], maxlen=maxlen, device=Q_bte.device, dtype=th.float32
        )
    else:
        bias = Q_bte.new_zeros((), dtype=th.float32)
    if extra_btT is not None:
        bias = bias + extra_btT
    logit_btT = th.baddbmm(
        bias,
        Q_bte.float(),
        K_bTe.float().transpose(-1, -2),
        alpha=(1 / e) if use_muP_factor else (1 / math.sqrt(e)),
    )
    if check_sentinel:
        logit_btT = logit_btT - 1e9 * invalid.float()
    W_btT = th.softmax(logit_btT, dim=2).to(dtype)
    if callable(V_bTe):
        V_bTe = V_bTe()
    A_bte = th.einsum("btp,bpe->bte", W_btT, V_bTe)
    return A_bte


@functools.lru_cache()
def get_attn_bias_cached(t, T, maxlen, device, dtype):
    m = th.ones(t, T, dtype=bool)
    m.tril_(T - t)
    if maxlen is not None and maxlen < T:
        m.triu_(T - t - maxlen + 1)
    bias = (~m).to(device=device, dtype=dtype) * -1e9
    return bias


class Attn:
    def __init__(self, mask, maxlen):
        self.mask = mask
        self.maxlen = maxlen

    def preproc_qkv(self, Q_bte, K_bte, V_bte):
        raise NotImplementedError

    def preproc_r(self, R_btn):
        raise NotImplementedError


def split_heads(x_bte, h):
    b, t, e = x_bte.shape
    assert e % h == 0, "Embsize must be divisible by number of heads"
    q = e // h
    x_bthq = x_bte.reshape((b, t, h, q))
    x_bhtq = transpose(x_bthq, "bthq", "bhtq")
    x_Btq = x_bhtq.reshape((b * h, t, q))
    return x_Btq


class All2All(Attn):
    def __init__(self, nhead, maxlen, mask=True, head_dim=None):
        super().__init__(mask=mask, maxlen=maxlen)
        assert (nhead is None) != (head_dim is None), (
            "exactly one of nhead and head_dim must be specified"
        )
        self.h = nhead
        self.head_dim = head_dim

    def preproc_qkv(self, *xs):
        q = xs[0].shape[-1]
        for x in xs:
            assert x.shape[-1] == q, "embedding dimensions do not match"
        h = self.h or exact_div(q, self.head_dim)
        postproc = functools.partial(self.postproc_a, h=h)
        return (postproc, *tuple(split_heads(x, h) for x in xs))

    def preproc_r(self, R_btn):
        _, ret = self.preproc_qkv(R_btn)
        return ret

    def postproc_a(self, A_Btq, h):
        B, t, q = A_Btq.shape
        b = B // h
        A_bhtq = A_Btq.reshape((b, h, t, q))
        A_bthq = transpose(A_bhtq, "bhtq", "bthq")
        A_bte = A_bthq.reshape((b, t, h * q))
        return A_bte


Q_SCALE = 0.1
K_SCALE = 0.2
V_SCALE = 1.0
PROJ_SCALE = 1.0
MLP0_SCALE = 1.0
MLP1_SCALE = 1.0
R_SCALE = 0.1
B_SCALE = 0.2


def get_norm(name, d, dtype=th.float32):
    if name == "none":
        return lambda x: x
    elif name == "layer":
        return LayerNorm(d, dtype=dtype)
    else:
        raise NotImplementedError(name)


def bandify(b_nd, t, T):
    nbasis, bandsize = b_nd.shape
    b_nd = b_nd[:, th.arange(bandsize - 1, -1, -1)]
    if bandsize >= T:
        b_nT = b_nd[:, -T:]
    else:
        b_nT = th.cat([b_nd.new_zeros(nbasis, T - bandsize), b_nd], dim=1)
    D_tnT = _banded_repeat(b_nT, t)
    return D_tnT


def _banded_repeat(x, t):
    b, T = x.shape
    x = th.cat([x, x.new_zeros(b, t - 1)], dim=1)
    result = x.unfold(1, T, 1).flip(1)
    return result


class AttentionLayerBase(nn.Module):
    def __init__(
        self,
        *,
        attn,
        scale,
        x_size,
        c_size,
        qk_size,
        v_size,
        dtype,
        relattn=False,
        seqlens=None,
        separate=False,
    ):
        super().__init__()
        dtype = parse_dtype(dtype)
        self.attn = attn
        self.x_size = x_size
        self.c_size = c_size
        s = math.sqrt(scale)
        separgs = dict(seqlens=seqlens, separate=separate)
        self.q_layer = MultiscaleLinear(
            x_size, qk_size, name="q", scale=Q_SCALE, dtype=dtype, **separgs
        )
        self.k_layer = MultiscaleLinear(
            c_size, qk_size, name="k", scale=K_SCALE, bias=False, dtype=dtype, **separgs
        )
        self.v_layer = MultiscaleLinear(
            c_size, v_size, name="v", scale=V_SCALE * s, bias=False, dtype=dtype, **separgs
        )
        self.proj_layer = MultiscaleLinear(
            v_size, x_size, name="proj", scale=PROJ_SCALE * s, dtype=dtype, **separgs
        )
        self.relattn = relattn
        maxlen = attn.maxlen
        assert maxlen > 0 or not attn.mask
        if self.relattn:
            nbasis = 10
            self.r_layer = NormedLinear(x_size, nbasis * attn.h, scale=R_SCALE, dtype=dtype)
            self.b_nd = nn.Parameter(th.randn(nbasis, maxlen) * B_SCALE)
        self.maxlen = maxlen
        self.dtype = dtype

    def relattn_logits(self, X_bte, T):
        R_btn = self.r_layer(X_bte).float()
        R_btn = self.attn.preproc_r(R_btn)
        t = R_btn.shape[1]
        D_ntT = bandify(self.b_nd, t, T)
        extra_btT = th.einsum("btn,ntp->btp", R_btn, D_ntT)
        return extra_btT


class SelfAttentionLayer(AttentionLayerBase):
    """Residual attention layer: output = x + f(x)"""

    def __init__(
        self,
        x_size,
        attn,
        scale,
        dtype="float32",
        norm="layer",
        cache_keep_len=None,
        relattn=False,
        log_scope="sa",
        use_muP_factor=False,
        **kwargs,
    ):
        super().__init__(
            x_size=x_size,
            c_size=x_size,
            qk_size=x_size,
            v_size=x_size,
            attn=attn,
            scale=scale,
            relattn=relattn,
            dtype=dtype,
            **kwargs,
        )
        self.ln_x = get_norm(norm, x_size, dtype=dtype)
        if cache_keep_len is None:
            if hasattr(attn, "cache_keep_len"):
                cache_keep_len = attn.cache_keep_len
            else:
                stride = attn.stride if isinstance(attn, StridedAttn) else 1
                cache_keep_len = stride * attn.maxlen
        self.cache_keep_len = cache_keep_len
        self.log_scope = log_scope
        self.use_muP_factor = use_muP_factor

    def residual(self, X_bte, state):
        X_bte = self.ln_x(X_bte)
        Q_bte = self.q_layer(X_bte)
        K_bte = self.k_layer(X_bte)
        V_bte = self.v_layer(X_bte)
        if state:
            state, K_bte, V_bte = self.update_state(state, K_bte, V_bte)
        postproc_closure, Q_bte, K_bte, V_bte = self.attn.preproc_qkv(Q_bte, K_bte, V_bte)
        extra_btT = self.relattn_logits(X_bte, K_bte.shape[1]) if self.relattn else None
        A_bte = attention(
            Q_bte,
            K_bte,
            V_bte,
            mask=self.attn.mask,
            extra_btT=extra_btT,
            maxlen=self.maxlen,
            dtype=self.dtype,
            check_sentinel=isinstance(self.attn, StridedAttn),
            use_muP_factor=self.use_muP_factor,
        )
        A_bte = postproc_closure(A_bte)
        Aproj_bte = self.proj_layer(A_bte)
        return Aproj_bte, state

    def forward(self, X_bte, state):
        R_bte, state = self.residual(X_bte, state)
        return X_bte + R_bte, state

    def update_state(self, state, K_bte, V_bte):
        def append(prev, new):
            tprev = prev.shape[1]
            startfull = max(tprev - self.cache_keep_len, 0)
            full = th.cat([prev[:, startfull:], new], dim=1)
            outstate = full[:, max(full.shape[1] - (self.cache_keep_len), 0) :]
            return outstate, full

        instate_K, instate_V = state
        outstate_K, K_bte = append(instate_K, K_bte)
        outstate_V, V_bte = append(instate_V, V_bte)
        assert outstate_K.shape[-2] <= self.cache_keep_len
        return (outstate_K, outstate_V), K_bte, V_bte

    def initial_state(self, batchsize, initial_T=0):
        return (
            zeros((batchsize, initial_T, self.x_size), dtype=self.dtype),
            zeros((batchsize, initial_T, self.x_size), dtype=self.dtype),
        )


class StridedAttn(Attn):
    """Present for isinstance() checks inside SelfAttentionLayer; unused in the
    transformer-recurrence configuration traced here (VPT ships with All2All)."""

    def __init__(self, nhead, stride, maxlen, mask=True):
        super().__init__(mask=mask, maxlen=maxlen)
        self.h = nhead
        self.stride = stride


class PointwiseLayer(nn.Module):
    """Residual MLP applied at each timestep"""

    def __init__(self, x_size, scale, dtype, norm, actname="relu", mlp_ratio=2):
        super().__init__()
        s = math.sqrt(scale)
        self.ln = get_norm(norm, x_size, dtype=dtype)
        self.mlp = MLP(
            insize=x_size,
            nhidlayer=1,
            outsize=x_size,
            hidsize=int(x_size * mlp_ratio),
            hidactiv=functools.partial(act, actname),
            dtype=dtype,
        )
        self.mlp.layers[0].weight.data *= MLP0_SCALE * s
        self.mlp.layers[1].weight.data *= MLP1_SCALE * s

    def residual(self, x):
        x = self.ln(x)
        x = self.mlp(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


def quick_gelu(x):
    return x * th.sigmoid(1.702 * x)


def act(actname, x):
    if actname == "relu":
        return F.relu(x)
    elif actname == "gelu":
        return quick_gelu(x)
    elif actname == "none":
        return x
    else:
        raise NotImplementedError(actname)


def _is_separate(sep, name):
    if isinstance(sep, bool):
        return sep
    assert isinstance(sep, set)
    if name in sep:
        sep.remove(name)
        return True
    return False


def make_maybe_multiscale(make_fn, *args, seqlens, separate, name, **kwargs):
    if _is_separate(separate, name):
        modules = [make_fn(*args, **kwargs) for _ in seqlens]
        return SplitCallJoin(modules, seqlens)
    else:
        return make_fn(*args, **kwargs)


class SplitCallJoin(nn.Module):
    def __init__(self, mods, seqlens):
        super().__init__()
        self.mods = nn.ModuleList(mods)
        self.seqlens = seqlens

    def forward(self, x):
        tl = sum(self.seqlens)
        x, undo = reshape_undo(x, "..., z*tl, e", "..., z, tl, e", tl=tl)
        x = list(th.split(x, self.seqlens, dim=-2))
        new_x = []
        for x, mod in safezip(x, self.mods):
            x, this_undo = reshape_undo(x, "..., z, l, e", "..., z*l, e")
            x = mod(x)
            x = this_undo(x)
            new_x.append(x)
        x = th.cat(new_x, dim=-2)
        x = undo(x)
        return x


MultiscaleLinear = functools.partial(make_maybe_multiscale, NormedLinear)


# ---------------------------------------------------------------------------
# lib/tree_util.py (subset: tree_map for nested dict/list/tuple of tensors)
# ---------------------------------------------------------------------------
def tree_map(f, tree):
    if isinstance(tree, dict):
        return {k: tree_map(f, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(f, v) for v in tree)
    elif tree is None:
        return None
    else:
        return f(tree)


# ---------------------------------------------------------------------------
# lib/masked_attention.py
# ---------------------------------------------------------------------------
@functools.lru_cache()
def get_band_diagonal_mask(
    t: int, T: int, maxlen: int, batchsize: int, device: th.device
) -> th.Tensor:
    m = th.ones(t, T, dtype=bool)
    m.tril_(T - t)
    if maxlen is not None and maxlen < T:
        m.triu_(T - t - maxlen + 1)
    m_btT = m[None].repeat_interleave(batchsize, dim=0)
    m_btT = m_btT.to(device=device)
    return m_btT


def get_mask(first_b11, state_mask, t, T, maxlen, heads, device):
    b = first_b11.shape[0]
    if state_mask is None:
        state_mask = th.zeros((b, 1, T - t), dtype=bool, device=device)
    m_btT = get_band_diagonal_mask(t, T, maxlen, b, device).clone()
    not_first = ~first_b11.to(device=device)
    m_btT[:, :, :-t] &= not_first
    m_btT[:, :, :-t] &= state_mask
    m_bhtT = m_btT[:, None].repeat_interleave(heads, dim=1)
    m_btT = m_bhtT.reshape((b * heads), t, T)
    state_mask = th.cat(
        [
            state_mask[:, :, t:] & not_first,
            th.ones((b, 1, min(t, T - t)), dtype=bool, device=device),
        ],
        dim=-1,
    )
    return m_btT, state_mask


class MaskedAttention(nn.Module):
    """Transformer self-attention layer that removes frames from previous episodes
    from the hidden state under certain constraints (see upstream docstring)."""

    @store_args
    def __init__(
        self,
        input_size,
        memory_size: int,
        heads: int,
        timesteps: int,
        mask: str = "clipped_causal",
        init_scale=1,
        norm="none",
        log_scope="sa",
        use_muP_factor=False,
    ):
        super().__init__()
        assert mask in {"none", "clipped_causal"}
        assert memory_size >= 0
        self.maxlen = memory_size - timesteps
        if mask == "none":
            mask = None
        self.orc_attn = All2All(heads, self.maxlen, mask=mask is not None)
        self.orc_block = SelfAttentionLayer(
            input_size,
            self.orc_attn,
            scale=init_scale,
            relattn=True,
            cache_keep_len=self.maxlen,
            norm=norm,
            log_scope=log_scope,
            use_muP_factor=use_muP_factor,
        )

    def initial_state(self, batchsize: int, device=None):
        state = self.orc_block.initial_state(batchsize, initial_T=self.maxlen)
        state_mask = None
        if device is not None:
            state = tree_map(lambda x: x.to(device), state)
        return state_mask, state

    def forward(self, input_bte, first_bt, state):
        state_mask, xf_state = state
        t = first_bt.shape[1]
        if self.mask == "clipped_causal":
            new_mask, state_mask = get_mask(
                first_b11=first_bt[:, [[0]]],
                state_mask=state_mask,
                t=t,
                T=t + self.maxlen,
                maxlen=self.maxlen,
                heads=self.heads,
                device=input_bte.device,
            )
            self.orc_block.attn.mask = new_mask
        output, xf_state = self.orc_block(input_bte, xf_state)
        return output, (state_mask, xf_state)


# ---------------------------------------------------------------------------
# lib/util.py (subset: FanInInitReLULayer + ResidualRecurrentBlocks)
# ---------------------------------------------------------------------------
class FanInInitReLULayer(nn.Module):
    """Implements a slightly modified init that correctly produces std 1 outputs given ReLU activation"""

    @store_args
    def __init__(
        self,
        inchan: int,
        outchan: int,
        *layer_args,
        layer_type: str = "conv",
        init_scale: int = 1,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation=True,
        log_scope: Optional[str] = None,
        **layer_kwargs,
    ):
        super().__init__()
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
        self.layer = layer(inchan, outchan, bias=self.norm is None, *layer_args, **layer_kwargs)

        self.layer.weight.data *= init_scale / self.layer.weight.norm(
            dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        )
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x


class ResidualRecurrentBlocks(nn.Module):
    @store_args
    def __init__(
        self, n_block=2, recurrence_type="multi_layer_lstm", is_residual=True, **block_kwargs
    ):
        super().__init__()
        init_scale = n_block**-0.5 if is_residual else 1
        self.blocks = nn.ModuleList(
            [
                ResidualRecurrentBlock(
                    **block_kwargs,
                    recurrence_type=recurrence_type,
                    is_residual=is_residual,
                    init_scale=init_scale,
                    block_number=i,
                )
                for i in range(n_block)
            ]
        )

    def forward(self, x, first, state):
        state_out = []
        assert len(state) == len(self.blocks), (
            f"Length of state {len(state)} did not match length of blocks {len(self.blocks)}"
        )
        for block, _s_in in zip(self.blocks, state):
            x, _s_o = block(x, first, _s_in)
            state_out.append(_s_o)
        return x, state_out

    def initial_state(self, batchsize):
        if "lstm" in self.recurrence_type:
            return [None for b in self.blocks]
        else:
            return [b.r.initial_state(batchsize) for b in self.blocks]


class ResidualRecurrentBlock(nn.Module):
    @store_args
    def __init__(
        self,
        hidsize,
        timesteps,
        init_scale=1,
        recurrence_type="multi_layer_lstm",
        is_residual=True,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        attention_heads=8,
        attention_memory_size=2048,
        attention_mask_style="clipped_causal",
        log_scope="resblock",
        block_number=0,
    ):
        super().__init__()
        self.log_scope = f"{log_scope}{block_number}"
        s = init_scale
        if use_pointwise_layer:
            if is_residual:
                s *= 2**-0.5
            self.mlp0 = FanInInitReLULayer(
                hidsize,
                hidsize * pointwise_ratio,
                init_scale=1,
                layer_type="linear",
                layer_norm=True,
                log_scope=self.log_scope + "/ptwise_mlp0",
            )
            self.mlp1 = FanInInitReLULayer(
                hidsize * pointwise_ratio,
                hidsize,
                init_scale=s,
                layer_type="linear",
                use_activation=pointwise_use_activation,
                log_scope=self.log_scope + "/ptwise_mlp1",
            )

        self.pre_r_ln = nn.LayerNorm(hidsize)
        if recurrence_type in ["multi_layer_lstm", "multi_layer_bilstm"]:
            self.r = nn.LSTM(hidsize, hidsize, batch_first=True)
            nn.init.normal_(self.r.weight_hh_l0, std=s * (self.r.weight_hh_l0.shape[0] ** -0.5))
            nn.init.normal_(self.r.weight_ih_l0, std=s * (self.r.weight_ih_l0.shape[0] ** -0.5))
            self.r.bias_hh_l0.data *= 0
            self.r.bias_ih_l0.data *= 0
        elif recurrence_type == "transformer":
            self.r = MaskedAttention(
                input_size=hidsize,
                timesteps=timesteps,
                memory_size=attention_memory_size,
                heads=attention_heads,
                init_scale=s,
                norm="none",
                log_scope=log_scope + "/sa",
                use_muP_factor=True,
                mask=attention_mask_style,
            )

    def forward(self, x, first, state):
        residual = x
        x = self.pre_r_ln(x)
        x, state_out = recurrent_forward(
            self.r,
            x,
            first,
            state,
            reverse_lstm=self.recurrence_type == "multi_layer_bilstm"
            and (self.block_number + 1) % 2 == 0,
        )
        if self.is_residual and "lstm" in self.recurrence_type:
            x = x + residual
        if self.use_pointwise_layer:
            residual = x
            x = self.mlp1(self.mlp0(x))
            if self.is_residual:
                x = x + residual
        return x, state_out


def recurrent_forward(module, x, first, state, reverse_lstm=False):
    if isinstance(module, nn.LSTM):
        if state is not None:
            mask = 1 - first[:, 0, None, None].to(th.float)
            state = tree_map(lambda _s: _s * mask, state)
            state = tree_map(lambda _s: _s.transpose(0, 1), state)
        if reverse_lstm:
            x = th.flip(x, [1])
        x, state_out = module(x, state)
        if reverse_lstm:
            x = th.flip(x, [1])
        state_out = tree_map(lambda _s: _s.transpose(0, 1), state_out)
        return x, state_out
    else:
        return module(x, first, state)


# ---------------------------------------------------------------------------
# lib/impala_cnn.py
# ---------------------------------------------------------------------------
class CnnBasicBlock(nn.Module):
    """Residual basic block, as in ImpalaCNN. Preserves channel number and shape"""

    def __init__(
        self,
        inchan: int,
        init_scale: float = 1,
        log_scope="",
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.inchan = inchan
        s = math.sqrt(init_scale)
        self.conv0 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            log_scope=f"{log_scope}/conv0",
            **init_norm_kwargs,
        )
        self.conv1 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            log_scope=f"{log_scope}/conv1",
            **init_norm_kwargs,
        )

    def forward(self, x):
        x = x + self.conv1(self.conv0(x))
        return x


class CnnDownStack(nn.Module):
    """Downsampling stack from Impala CNN."""

    name = "Impala_CnnDownStack"

    def __init__(
        self,
        inchan: int,
        nblock: int,
        outchan: int,
        init_scale: float = 1,
        pool: bool = True,
        post_pool_groups: Optional[int] = None,
        log_scope: str = "",
        init_norm_kwargs: Dict = {},
        first_conv_norm=False,
        **kwargs,
    ):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        first_conv_init_kwargs = deepcopy(init_norm_kwargs)
        if not first_conv_norm:
            first_conv_init_kwargs["group_norm_groups"] = None
            first_conv_init_kwargs["batch_norm"] = False
        self.firstconv = FanInInitReLULayer(
            inchan,
            outchan,
            kernel_size=3,
            padding=1,
            log_scope=f"{log_scope}/firstconv",
            **first_conv_init_kwargs,
        )
        self.post_pool_groups = post_pool_groups
        if post_pool_groups is not None:
            self.n = nn.GroupNorm(post_pool_groups, outchan)
        self.blocks = nn.ModuleList(
            [
                CnnBasicBlock(
                    outchan,
                    init_scale=init_scale / math.sqrt(nblock),
                    log_scope=f"{log_scope}/block{i}",
                    init_norm_kwargs=init_norm_kwargs,
                    **kwargs,
                )
                for i in range(nblock)
            ]
        )

    def forward(self, x):
        x = self.firstconv(x)
        if self.pool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            if self.post_pool_groups is not None:
                x = self.n(x)
        x = sequential(self.blocks, x, diag_name=self.name)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if self.pool:
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.outchan, h, w)


class ImpalaCNN(nn.Module):
    """:param inshape: input image shape (height, width, channels)"""

    name = "ImpalaCNN"

    def __init__(
        self,
        inshape: List[int],
        chans: List[int],
        outsize: int,
        nblock: int,
        init_norm_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        first_conv_norm=False,
        **kwargs,
    ):
        super().__init__()
        h, w, c = inshape
        curshape = (c, h, w)
        self.stacks = nn.ModuleList()
        for i, outchan in enumerate(chans):
            stack = CnnDownStack(
                curshape[0],
                nblock=nblock,
                outchan=outchan,
                init_scale=math.sqrt(len(chans)),
                log_scope=f"downstack{i}",
                init_norm_kwargs=init_norm_kwargs,
                first_conv_norm=first_conv_norm if i == 0 else True,
                **kwargs,
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)

        self.dense = FanInInitReLULayer(
            intprod(curshape),
            outsize,
            layer_type="linear",
            log_scope="imapala_final_dense",
            init_scale=1.4,
            **dense_init_norm_kwargs,
        )
        self.outsize = outsize

    def forward(self, x):
        b, t = x.shape[:-3]
        x = x.reshape(b * t, *x.shape[-3:])
        x = transpose(x, "bhwc", "bchw")
        x = sequential(self.stacks, x, diag_name=self.name)
        x = x.reshape(b, t, *x.shape[1:])
        x = flatten_image(x)
        x = self.dense(x)
        return x


# ---------------------------------------------------------------------------
# lib/policy.py (subset: ImgPreprocessing, ImgObsProcess, MinecraftPolicy, InverseActionNet)
# ---------------------------------------------------------------------------
class ImgPreprocessing(nn.Module):
    """Normalize incoming images. scale by 1/255 (no img_statistics remote file)."""

    def __init__(self, img_statistics: Optional[str] = None, scale_img: bool = True):
        super().__init__()
        self.img_mean = None
        self.ob_scale = 255.0 if scale_img else 1.0

    def forward(self, img):
        x = img.to(dtype=th.float32)
        x = x / self.ob_scale
        return x


class ImgObsProcess(nn.Module):
    """ImpalaCNN followed by a linear layer."""

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.cnn = ImpalaCNN(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize, output_size, layer_type="linear", **dense_init_norm_kwargs
        )

    def forward(self, img):
        return self.linear(self.cnn(img))


class MinecraftPolicy(nn.Module):
    """VPT visual-recurrent trunk: ImpalaCNN encoder -> residual recurrent blocks
    (transformer self-attention or multi-layer LSTM) -> final dense+LN."""

    def __init__(
        self,
        recurrence_type="lstm",
        impala_width=1,
        impala_chans=(16, 32, 32),
        obs_processing_width=256,
        hidsize=512,
        single_output=False,
        img_shape=None,
        scale_input_img=True,
        only_img_input=False,
        init_norm_kwargs={},
        impala_kwargs={},
        input_shape=None,
        active_reward_monitors=None,
        img_statistics=None,
        first_conv_norm=False,
        diff_mlp_embedding=False,
        attention_mask_style="clipped_causal",
        attention_heads=8,
        attention_memory_size=2048,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        n_recurrence_layers=1,
        recurrence_is_residual=True,
        timesteps=None,
        use_pre_lstm_ln=True,
        **unused_kwargs,
    ):
        super().__init__()
        assert recurrence_type in [
            "multi_layer_lstm",
            "multi_layer_bilstm",
            "multi_masked_lstm",
            "transformer",
            "none",
        ]

        self.single_output = single_output
        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.hidsize = hidsize

        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        self.img_preprocess = ImgPreprocessing(
            img_statistics=img_statistics, scale_img=scale_input_img
        )
        self.img_process = ImgObsProcess(
            cnn_outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            **impala_kwargs,
        )

        self.pre_lstm_ln = nn.LayerNorm(hidsize) if use_pre_lstm_ln else None
        self.diff_obs_process = None
        self.recurrence_type = recurrence_type

        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type=recurrence_type,
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers,
        )

        self.lastlayer = FanInInitReLULayer(
            hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs
        )
        self.final_ln = th.nn.LayerNorm(hidsize)

    def output_latent_size(self):
        return self.hidsize

    def forward(self, ob, state_in, context):
        first = context["first"]
        x = self.img_preprocess(ob["img"])
        x = self.img_process(x)
        if self.diff_obs_process:
            processed_obs = self.diff_obs_process(ob["diff_goal"])
            x = processed_obs + x
        if self.pre_lstm_ln is not None:
            x = self.pre_lstm_ln(x)
        if self.recurrent_layer is not None:
            x, state_out = self.recurrent_layer(x, first, state_in)
        else:
            state_out = state_in
        x = F.relu(x, inplace=False)
        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x
        if self.single_output:
            return pi_latent, state_out
        return (pi_latent, vf_latent), state_out

    def initial_state(self, batchsize):
        if self.recurrent_layer:
            return self.recurrent_layer.initial_state(batchsize)
        else:
            return None


class InverseActionNet(MinecraftPolicy):
    """The Inverse Dynamics Model (IDM) trunk: same ImpalaCNN+recurrence backbone as the
    BC policy, with an optional Conv3D pre-layer over the raw video frames."""

    def __init__(self, hidsize=512, conv3d_params=None, **MCPoliy_kwargs):
        super().__init__(
            hidsize=hidsize, first_conv_norm=conv3d_params is not None, **MCPoliy_kwargs
        )
        self.conv3d_layer = None
        if conv3d_params is not None:
            conv3d_init_params = deepcopy(self.init_norm_kwargs)
            conv3d_init_params["group_norm_groups"] = None
            conv3d_init_params["batch_norm"] = False
            self.conv3d_layer = FanInInitReLULayer(
                layer_type="conv3d",
                log_scope="3d_conv",
                **conv3d_params,
                **conv3d_init_params,
            )

    def forward(self, ob, state_in, context):
        first = context["first"]
        x = self.img_preprocess(ob["img"])
        if self.conv3d_layer is not None:
            x = self._conv3d_forward(x)
        x = self.img_process(x)
        if self.recurrent_layer is not None:
            x, state_out = self.recurrent_layer(x, first, state_in)
        x = F.relu(x, inplace=False)
        pi_latent = self.lastlayer(x)
        pi_latent = self.final_ln(x)
        return (pi_latent, None), state_out

    def _conv3d_forward(self, x):
        x = transpose(x, "bthwc", "bcthw")
        new_x = []
        for mini_batch in th.split(x, 1):
            new_x.append(self.conv3d_layer(mini_batch))
        x = th.cat(new_x)
        x = transpose(x, "bcthw", "bthwc")
        return x


# ---------------------------------------------------------------------------
# Staging module trace wrapper
# ---------------------------------------------------------------------------
class VPTIDMTraceWrapper(nn.Module):
    """Wraps `InverseActionNet` (the real VPT Inverse Dynamics Model trunk) so it can be
    traced with a plain (video_batch,) tensor input instead of the dict-obs/state_in/context
    calling convention used by the training/inference harness (agent.py / IDMAgent)."""

    def __init__(self, net: InverseActionNet):
        super().__init__()
        self.net = net

    def forward(self, img):
        b, t = img.shape[0], img.shape[1]
        first = th.zeros((b, t), dtype=th.bool)
        state_in = self.net.initial_state(b)
        (pi_latent, _), _state_out = self.net({"img": img}, state_in, context={"first": first})
        return pi_latent


def build_vpt_idm():
    """Tiny-size real VPT IDM trunk (InverseActionNet, transformer recurrence)."""
    net = InverseActionNet(
        hidsize=32,
        img_shape=[16, 16, 3],
        impala_width=1,
        impala_chans=(4, 8, 8),
        impala_kwargs={"post_pool_groups": 1},
        init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
        recurrence_type="transformer",
        attention_heads=2,
        attention_memory_size=8,
        n_recurrence_layers=1,
        timesteps=4,
        use_pointwise_layer=True,
        pointwise_ratio=2,
    )
    return VPTIDMTraceWrapper(net)


def example_input_vpt_idm():
    # (batch, time, H, W, C) uint8-like video frames, matching ob["img"] convention.
    return th.zeros(1, 4, 16, 16, 3, dtype=th.float32)


MENAGERIE_ENTRIES = [
    (
        "VoT (Video Pretraining IDM)",
        build_vpt_idm,
        example_input_vpt_idm,
        2022,
        MENAGERIE_ZOO,
    ),
]
