# SOURCE: vendored from https://github.com/lucidrains/enformer-pytorch @ 29cd2529f31cc41456da7511e720d730f0bdd19d
#
# Enformer (Avsec et al., Nature Methods 2021, DeepMind) predicts basepair-resolution
# gene-expression / chromatin tracks from ~200kb DNA sequence windows: a conv-tower
# stem/downsample stack followed by a Transformer trunk with relative positional
# attention, ending in per-organism (human/mouse) 1x1-conv-style output heads. The
# official DeepMind repo (github.com/google-deepmind/deepmind-research/tree/master/enformer)
# ships the original TF2/Sonnet implementation; lucidrains/enformer-pytorch is the
# maintained, widely-used faithful PyTorch port (validated against the released TF
# weights; also the HF-Hub-integrated reference used by downstream genomics tooling
# such as Borzoi's own PyTorch port).
#
# The classes/functions below (EnformerConfig, Residual/GELU/AttentionPool/
# TargetLengthCrop/ConvBlock, the relative-positional-embedding Attention module,
# and the main Enformer class) are copied unmodified from enformer_pytorch/
# {config_enformer,modeling_enformer}.py. `str_to_one_hot`/`seq_indices_to_one_hot`
# (from enformer_pytorch/data.py) are copied verbatim rather than imported, since the
# real data.py also imports `polars`/`pyfaidx` (FASTA-interval dataset loading, not
# needed for a plain forward pass) which are not in the base env. The one deliberate,
# non-architectural adaptation: TF_GAMMAS (a 393KB precomputed tensor keyed to
# seq_len=1536, tf-xlogy positional-gamma correction) is lazy-loaded from the vendored
# `precomputed/tf_gammas.pt` file instead of at import time -- it is only read when
# `use_tf_gamma=True` (default False), so a tiny recipe-sized model never touches it.
#
# ruff: noqa: E402 -- this file amalgamates several original vendored source files,
# each carrying its own local import block; hoisting all imports to the top would
# obscure per-section provenance.

import math
from pathlib import Path

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange
from einops.layers.torch import Rearrange

from transformers import PreTrainedModel, PretrainedConfig

MENAGERIE_ZOO = "vendored-pytorch"

# ---- vendored from enformer_pytorch/config_enformer.py ----


class EnformerConfig(PretrainedConfig):
    model_type = "enformer"

    def __init__(
        self,
        dim=1536,
        depth=11,
        heads=8,
        output_heads=dict(human=5313, mouse=1643),
        target_length=896,
        attn_dim_key=64,
        dropout_rate=0.4,
        attn_dropout=0.05,
        pos_dropout=0.01,
        use_checkpointing=False,
        use_convnext=False,
        num_downsamples=7,  # genetic sequence is downsampled 2 ** 7 == 128x in default Enformer - can be changed for higher resolution
        dim_divisible_by=128,
        use_tf_gamma=False,
        **kwargs,
    ):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.output_heads = output_heads
        self.target_length = target_length
        self.attn_dim_key = attn_dim_key
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.pos_dropout = pos_dropout
        self.use_checkpointing = use_checkpointing
        self.num_downsamples = num_downsamples
        self.dim_divisible_by = dim_divisible_by
        self.use_tf_gamma = use_tf_gamma

        super().__init__(**kwargs)


# ---- vendored from enformer_pytorch/data.py (only the 2 functions the model needs) ----

seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord("a")] = 0
seq_indices_embed[ord("c")] = 1
seq_indices_embed[ord("g")] = 2
seq_indices_embed[ord("t")] = 3
seq_indices_embed[ord("n")] = 4
seq_indices_embed[ord("A")] = 0
seq_indices_embed[ord("C")] = 1
seq_indices_embed[ord("G")] = 2
seq_indices_embed[ord("T")] = 3
seq_indices_embed[ord("N")] = 4
seq_indices_embed[ord(".")] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord("a")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("c")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("g")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("t")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("n")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("A")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("C")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("G")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("T")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("N")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord(".")] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()


def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = seq_strs if batched else [seq_strs]

    seq_chrs = [torch.tensor(list(seq_str.encode())) for seq_str in seq_strs]
    return torch.stack(seq_chrs) if batched else seq_chrs[0]


def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]


def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]


def seq_indices_to_one_hot(t, padding=-1):
    is_padding = t == padding
    t = t.clamp(min=0)
    one_hot = F.one_hot(t, num_classes=5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out


# ---- vendored from enformer_pytorch/modeling_enformer.py ----

# constants

SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896

# gamma positions from tensorflow
# addressing a difference between xlogy results from tensorflow and pytorch
# solution came from @johahi

_TF_GAMMAS_PATH = Path(__file__).parent / "precomputed" / "tf_gammas.pt"
_TF_GAMMAS_CACHE = None


def _tf_gammas(device):
    global _TF_GAMMAS_CACHE
    if _TF_GAMMAS_CACHE is None:
        _TF_GAMMAS_CACHE = torch.load(str(_TF_GAMMAS_PATH), weights_only=True)
    return _TF_GAMMAS_CACHE.to(device)


# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def always(val):
    def inner(*args, **kwargs):
        return val

    return inner


def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# maybe sync batchnorm, for distributed training


def MaybeSyncBatchnorm(is_distributed=None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d


# losses and metrics


def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()


def pearson_corr_coef(x, y, dim=1, reduce_dims=(-1,)):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    return F.cosine_similarity(x_centered, y_centered, dim=dim).mean(dim=reduce_dims)


# relative positional encoding functions


def get_positional_features_exponential(
    positions, features, seq_len, min_half_life=3.0, dtype=torch.float
):
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device=positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions)


def get_positional_features_central_mask(positions, features, seq_len, dtype=torch.float):
    center_widths = 2 ** torch.arange(1, features + 1, device=positions.device).to(dtype)
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).to(dtype)


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(
    positions, features, seq_len, stddev=None, start_mean=None, eps=1e-8, dtype=torch.float
):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device=positions.device)

    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2

    probabilities = gamma_pdf(positions.to(dtype).abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs


def get_positional_embed(seq_len, feature_size, device, use_tf_gamma, dtype=torch.float):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    assert not use_tf_gamma or seq_len == 1536, (
        "if using tf gamma, only sequence length of 1536 allowed for now"
    )

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma if not use_tf_gamma else always(_tf_gammas(device)),
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size is not divisible by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len, dtype=dtype))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1)
    return embeddings.to(dtype)


def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., : ((t2 + 1) // 2)]


# classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange("b d (n p) -> b d n p", p=pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim=-1)

        return (x * attn).sum(dim=-1)


class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f"sequence length {seq_len} is less than target length {target_len}")

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]


def ConvBlock(dim, dim_out=None, kernel_size=1, is_distributed=None):
    batchnorm_klass = MaybeSyncBatchnorm(is_distributed=is_distributed)

    return nn.Sequential(
        batchnorm_klass(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=kernel_size // 2),
    )


# attention classes


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.0,
        pos_dropout=0.0,
        use_tf_gamma=False,
    ):
        super().__init__()
        self.scale = dim_key**-0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # whether to use tf gamma

        self.use_tf_gamma = use_tf_gamma

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        content_logits = einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)

        positions = get_positional_embed(
            n,
            self.num_rel_pos_features,
            device,
            use_tf_gamma=self.use_tf_gamma,
            dtype=self.to_rel_k.weight.dtype,
        )
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# main class


class Enformer(PreTrainedModel):
    config_class = EnformerConfig
    base_model_prefix = "enformer"

    @staticmethod
    def from_hparams(**kwargs):
        return Enformer(EnformerConfig(**kwargs))

    def __init__(self, config):
        super().__init__(config)
        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2

        # create stem

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding=7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size=2),
        )

        # create conv tower

        filter_list = exponential_linspace_int(
            half_dim,
            config.dim,
            num=(config.num_downsamples - 1),
            divisible_by=config.dim_divisible_by,
        )
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                nn.Sequential(
                    ConvBlock(dim_in, dim_out, kernel_size=5),
                    Residual(ConvBlock(dim_out, dim_out, 1)),
                    AttentionPool(dim_out, pool_size=2),
                )
            )

        self.conv_tower = nn.Sequential(*conv_layers)

        # whether to use tensorflow gamma positions

        use_tf_gamma = config.use_tf_gamma
        self.use_tf_gamma = use_tf_gamma

        # transformer

        transformer = []
        for _ in range(config.depth):
            transformer.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim),
                            Attention(
                                config.dim,
                                heads=config.heads,
                                dim_key=config.attn_dim_key,
                                dim_value=config.dim // config.heads,
                                dropout=config.attn_dropout,
                                pos_dropout=config.pos_dropout,
                                num_rel_pos_features=config.dim // config.heads,
                                use_tf_gamma=use_tf_gamma,
                            ),
                            nn.Dropout(config.dropout_rate),
                        )
                    ),
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim),
                            nn.Linear(config.dim, config.dim * 2),
                            nn.Dropout(config.dropout_rate),
                            nn.ReLU(),
                            nn.Linear(config.dim * 2, config.dim),
                            nn.Dropout(config.dropout_rate),
                        )
                    ),
                )
            )

        self.transformer = nn.Sequential(*transformer)

        # target cropping

        self.target_length = config.target_length
        self.crop_final = TargetLengthCrop(config.target_length)

        # final pointwise

        self.final_pointwise = nn.Sequential(
            Rearrange("b n d -> b d n"),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange("b d n -> b n d"),
            nn.Dropout(config.dropout_rate / 8),
            GELU(),
        )

        # create trunk sequential module

        self._trunk = nn.Sequential(
            Rearrange("b n d -> b d n"),
            self.stem,
            self.conv_tower,
            Rearrange("b d n -> b n d"),
            self.transformer,
            self.crop_final,
            self.final_pointwise,
        )

        # create final heads for human and mouse

        self.add_heads(**config.output_heads)

        # use checkpointing on transformer trunk

        self.use_checkpointing = config.use_checkpointing

    def add_heads(self, **kwargs):
        self.output_heads = kwargs

        self._heads = nn.ModuleDict(
            map_values(
                lambda features: nn.Sequential(nn.Linear(self.dim * 2, features), nn.Softplus()),
                kwargs,
            )
        )

    def set_target_length(self, target_length):
        crop_module = self._trunk[-2]
        crop_module.target_length = target_length

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    def trunk_checkpointed(self, x):
        x = rearrange(x, "b n d -> b d n")
        x = self.stem(x)
        x = self.conv_tower(x)
        x = rearrange(x, "b d n -> b n d")
        x = checkpoint_sequential(self.transformer, len(self.transformer), x)
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return x

    def forward(
        self,
        x,
        target=None,
        return_corr_coef=False,
        return_embeddings=False,
        return_only_embeddings=False,
        head=None,
        target_length=None,
    ):
        if isinstance(x, list):
            x = str_to_one_hot(x)

        elif type(x) == torch.Tensor and x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)
        x.to(self.device)

        no_batch = x.ndim == 2

        if no_batch:
            x = rearrange(x, "... -> () ...")

        if exists(target_length):
            self.set_target_length(target_length)

        trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
        x = trunk_fn(x)

        if no_batch:
            x = rearrange(x, "() ... -> ...")

        if return_only_embeddings:
            return x

        out = map_values(lambda fn: fn(x), self._heads)

        if exists(head):
            assert head in self._heads, f"head {head} not found"
            out = out[head]

        if exists(target):
            assert exists(head), (
                "head must be passed in if one were to calculate loss directly with targets"
            )

            if return_corr_coef:
                return pearson_corr_coef(out, target)

            return poisson_loss(out, target)

        if return_embeddings:
            return out, x

        return out


# ---- menagerie staging harness ----


def build_enformer():
    """Tiny-sized real Enformer: 1 transformer layer, small dim, 3 downsamples,
    no target cropping (target_length=-1) so a short recipe-scale input traces.
    (num_downsamples must be >= 3: exponential_linspace_int divides by
    num_downsamples - 1, so 2 would divide by zero.)"""
    config = EnformerConfig(
        dim=192,
        depth=1,
        heads=2,
        output_heads=dict(human=8),
        target_length=-1,
        attn_dim_key=16,
        dropout_rate=0.0,
        attn_dropout=0.0,
        pos_dropout=0.0,
        num_downsamples=3,
        dim_divisible_by=24,
        use_tf_gamma=False,
    )
    model = Enformer(config)
    return model.eval()


def example_input_enformer():
    """One-hot DNA sequence tensor (N, L, 4). Real Enformer uses L=196_608; here L
    is shrunk to the smallest multiple needed to clear the stem + 3 AttentionPool
    downsamples (2**4 == 16x) for a fast, real trace."""
    torch.manual_seed(0)
    batch_size, length = 1, 64
    idx = torch.randint(0, 4, (batch_size, length))
    x = seq_indices_to_one_hot(idx)
    return x


MENAGERIE_ENTRIES = [
    (
        "Enformer",
        "build_enformer",
        "example_input_enformer",
        2021,
        "vendored-pytorch",
    ),
]
