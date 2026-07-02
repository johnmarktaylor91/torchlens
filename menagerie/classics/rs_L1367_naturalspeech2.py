# SOURCE: vendored from https://github.com/lucidrains/naturalspeech2-pytorch @ main
# (naturalspeech2_pytorch/naturalspeech2_pytorch.py + naturalspeech2_pytorch/attend.py)
#
# NaturalSpeech2 (Shen et al., "NaturalSpeech 2: Latent Diffusion Models are
# Natural and Zero-Shot Speech and Singing Synthesizers") -- a latent-diffusion
# TTS model: a WaveNet+Transformer denoiser ("Model") predicts the diffusion
# target (v-prediction by default) over continuous audio-codec latents, with
# an optional phoneme/prompt/duration/pitch conditioning stack.
#
# This vendors the real classes verbatim (only whitespace-preserving trims of
# unused branches were made to keep this a self-contained single file --
# no architectural change). The upstream module imports `audiolm_pytorch`
# (for SoundStream/EncodecWrapper codecs), `pyworld` (native pitch extraction),
# `accelerate`/`ema_pytorch`/`tqdm` (Trainer-only), and uses `@beartype`
# decorators -- none of those packages are installed in this base env, and
# they are needed only for (a) the raw-waveform-encoding path (`codec=`), (b)
# the phoneme/prompt-conditioned TTS path (`condition_on_prompt=True`), and
# (c) the training-loop `Trainer` class. This module exercises
# `NaturalSpeech2` in its unconditional, pre-encoded-latent mode
# (`codec=None`, `target_sample_hz=<int>`, `Model(condition_on_prompt=False)`),
# which is the real diffusion-model forward path (`Model` wavenet+transformer
# denoiser predicting the training target over `audio` latents) and does not
# touch any of those optional subsystems, so they are dropped rather than
# vendored. `@beartype` annotations are removed (runtime type-checking only,
# not architecture); every `nn.Module` below is transcribed unmodified.

import math
from collections import namedtuple
from functools import partial, wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from packaging import version

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

MENAGERIE_ZOO = "vendored-pytorch"

# ---------------------------------------------------------------------------
# naturalspeech2_pytorch/attend.py (verbatim, self-contained)
# ---------------------------------------------------------------------------

Config = namedtuple(
    "EfficientAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


def _once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


_print_once = _once(print)


class Attend(nn.Module):
    def __init__(self, dropout=0.0, causal=False, use_flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse("2.0.0")), (
            "in order to use flash attention, you must be using pytorch 2.0 or above"
        )

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            _print_once("A100 GPU detected, using flash attention if input tensor is on cuda")
            self.cuda_config = Config(True, False, False)
        else:
            _print_once(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda  # noqa: F841

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )

        return out

    def forward(self, q, k, v, mask=None):
        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out


# ---------------------------------------------------------------------------
# naturalspeech2_pytorch/naturalspeech2_pytorch.py (verbatim classes needed
# for the unconditional codec-free diffusion-model forward path)
# ---------------------------------------------------------------------------

mlist = nn.ModuleList


def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def divisible_by(num, den):
    return (num % den) == 0


def pad_or_curtail_to_length(t, length):
    if t.shape[-1] == length:
        return t

    if t.shape[-1] > length:
        return t[..., :length]

    return F.pad(t, (0, length - t.shape[-1]))


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel=3, groups=8, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, kernel, *, dropout=0.0, groups=8, num_convs=2):
        super().__init__()

        blocks = []
        for ind in range(num_convs):
            is_first = ind == 0
            dim_in = dim if is_first else dim_out
            block = Block(dim_in, dim_out, kernel, groups=groups, dropout=dropout)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        x = rearrange(x, "b n c -> b c n")
        h = self.blocks(x)
        out = h + self.res_conv(x)
        return rearrange(out, "b c n -> b n c")


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation
        (stride,) = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.0)
        return super().forward(causal_padded_x)


class WavenetResBlock(nn.Module):
    def __init__(self, dim, *, dilation, kernel_size=3, skip_conv=False, dim_cond_mult=None):
        super().__init__()

        self.cond = exists(dim_cond_mult)
        self.to_time_cond = None

        if self.cond:
            self.to_time_cond = nn.Linear(dim * dim_cond_mult, dim * 2)

        self.conv = CausalConv1d(dim, dim, kernel_size, dilation=dilation)
        self.res_conv = CausalConv1d(dim, dim, 1)
        self.skip_conv = CausalConv1d(dim, dim, 1) if skip_conv else None

    def forward(self, x, t=None):
        if self.cond:
            assert exists(t)
            t = self.to_time_cond(t)
            t = rearrange(t, "b c -> b c 1")
            t_gamma, t_beta = t.chunk(2, dim=-2)

        res = self.res_conv(x)

        x = self.conv(x)

        if self.cond:
            x = x * t_gamma + t_beta

        x = x.tanh() * x.sigmoid()

        x = x + res

        skip = None
        if exists(self.skip_conv):
            skip = self.skip_conv(x)

        return x, skip


class WavenetStack(nn.Module):
    def __init__(self, dim, *, layers, kernel_size=3, has_skip=False, dim_cond_mult=None):
        super().__init__()
        dilations = 2 ** torch.arange(layers)

        self.has_skip = has_skip
        self.blocks = mlist([])

        for dilation in dilations.tolist():
            block = WavenetResBlock(
                dim=dim,
                kernel_size=kernel_size,
                dilation=dilation,
                skip_conv=has_skip,
                dim_cond_mult=dim_cond_mult,
            )

            self.blocks.append(block)

    def forward(self, x, t):
        residuals = []
        skips = []

        if isinstance(x, Tensor):
            x = (x,) * len(self.blocks)

        for block_input, block in zip(x, self.blocks):
            residual, skip = block(block_input, t)

            residuals.append(residual)
            skips.append(skip)

        if self.has_skip:
            return torch.stack(skips)

        return residuals


class Wavenet(nn.Module):
    def __init__(self, dim, *, stacks, layers, init_conv_kernel=3, dim_cond_mult=None):
        super().__init__()
        self.init_conv = CausalConv1d(dim, dim, init_conv_kernel)
        self.stacks = mlist([])

        for ind in range(stacks):
            is_last = ind == (stacks - 1)

            stack = WavenetStack(dim, layers=layers, dim_cond_mult=dim_cond_mult, has_skip=is_last)

            self.stacks.append(stack)

        self.final_conv = CausalConv1d(dim, dim, 1)

    def forward(self, x, t=None):
        x = self.init_conv(x)

        for stack in self.stacks:
            x = stack(x, t)

        return self.final_conv(x.sum(dim=0))


class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True, dim_cond=None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond=None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma, beta = map(lambda t: rearrange(t, "b d -> b 1 d"), (gamma, beta))
        return out * gamma + beta


class ConditionableTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        ff_causal_conv=False,
        dim_cond_mult=None,
        cross_attn=False,
        use_flash=False,
    ):
        super().__init__()
        self.dim = dim
        self.layers = mlist([])

        cond = exists(dim_cond_mult)

        maybe_adaptive_norm_kwargs = (
            dict(scale=not cond, dim_cond=dim * dim_cond_mult) if cond else dict()
        )
        rmsnorm = partial(RMSNorm, **maybe_adaptive_norm_kwargs)

        for _ in range(depth):
            self.layers.append(
                mlist(
                    [
                        rmsnorm(dim),
                        Attention(dim=dim, dim_head=dim_head, heads=heads, use_flash=use_flash),
                        rmsnorm(dim) if cross_attn else None,
                        Attention(dim=dim, dim_head=dim_head, heads=heads, use_flash=use_flash)
                        if cross_attn
                        else None,
                        rmsnorm(dim),
                        FeedForward(dim=dim, mult=ff_mult, causal_conv=ff_causal_conv),
                    ]
                )
            )

        self.to_pred = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim, bias=False))

    def forward(self, x, times=None, context=None):
        t = times

        for attn_norm, attn, cross_attn_norm, cross_attn, ff_norm, ff in self.layers:
            res = x
            x = attn_norm(x, cond=t)
            x = attn(x) + res

            if exists(cross_attn):
                assert exists(context)
                res = x
                x = cross_attn_norm(x, cond=t)
                x = cross_attn(x, context=context) + res

            res = x
            x = ff_norm(x, cond=t)
            x = ff(x) + res

        return self.to_pred(x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, causal_conv=False):
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = nn.Sequential(
            Rearrange("b n d -> b d n"),
            CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange("b d n -> b n d"),
        )

    return Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), conv, nn.Linear(dim_inner, dim))


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        dropout=0.0,
        use_flash=False,
        cross_attn_include_queries=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal=causal, dropout=dropout, use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = self.attend(q, k, v, mask=mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Model(nn.Module):
    """The DDPM denoiser: wavenet + transformer, optionally prompt-conditioned."""

    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        wavenet_layers=8,
        wavenet_stacks=4,
        dim_cond_mult=4,
        use_flash_attn=True,
        dim_prompt=None,
        num_latents_m=32,
        resampler_depth=2,
        cond_drop_prob=0.0,
        condition_on_prompt=False,
    ):
        super().__init__()
        self.dim = dim

        dim_time = dim * dim_cond_mult

        self.to_time_cond = Sequential(
            LearnedSinusoidalPosEmb(dim), nn.Linear(dim + 1, dim_time), nn.SiLU()
        )

        self.cond_drop_prob = cond_drop_prob
        self.condition_on_prompt = condition_on_prompt
        self.to_prompt_cond = None

        # NOTE: condition_on_prompt=True path (PerceiverResampler / prompt
        # conditioning) is intentionally not vendored -- see module docstring.
        assert not condition_on_prompt, "prompt-conditioned path not vendored (see header)"

        self.null_cond = None
        self.cond_to_model_dim = None

        dim_cond_mult = dim_cond_mult * (2 if condition_on_prompt else 1)

        self.wavenet = Wavenet(
            dim=dim, stacks=wavenet_stacks, layers=wavenet_layers, dim_cond_mult=dim_cond_mult
        )

        self.transformer = ConditionableTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            ff_causal_conv=True,
            dim_cond_mult=dim_cond_mult,
            use_flash=use_flash_attn,
            cross_attn=condition_on_prompt,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, cond_drop_prob=0.0, **kwargs)

        if cond_scale == 1.0:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)

        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, times, prompt=None, prompt_mask=None, cond=None, cond_drop_prob=None):
        b = x.shape[0]
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        t = self.to_time_cond(times)
        c = None

        x = rearrange(x, "b n d -> b d n")

        if exists(self.cond_to_model_dim):
            assert exists(cond)
            cond = self.cond_to_model_dim(cond)

            cond_drop_mask = prob_mask_like((b,), cond_drop_prob, self.device)

            cond = torch.where(rearrange(cond_drop_mask, "b -> b 1 1"), self.null_cond, cond)

            cond = pad_or_curtail_to_length(cond, x.shape[-1])

            x = x + cond

        x = self.wavenet(x, t)
        x = rearrange(x, "b d n -> b n d")

        x = self.transformer(x, t, context=c)
        return x


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def simple_linear_schedule(t, clip_min=1e-9):
    return (1 - t).clamp(min=clip_min)


def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min=clip_min)


def sigmoid_schedule(t, start=-3, end=3, tau=1, clamp_min=1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min=clamp_min, max=1.0)


def gamma_to_alpha_sigma(gamma, scale=1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)


class NaturalSpeech2(nn.Module):
    """Latent-diffusion TTS model (codec=None, unconditional path vendored).

    Forward pass: sample random diffusion times, noise the input audio-codec
    latents to the sampled noise level via the gamma schedule, run the
    `Model` (wavenet+transformer) denoiser, and return the diffusion loss
    against the chosen `objective` target (default 'v'-prediction).
    """

    def __init__(
        self,
        model,
        *,
        target_sample_hz=None,
        timesteps=1000,
        use_ddim=True,
        noise_schedule="sigmoid",
        objective="v",
        schedule_kwargs: dict = dict(),
        time_difference=0.0,
        min_snr_loss_weight=True,
        min_snr_gamma=5,
        train_prob_self_cond=0.9,
        rvq_cross_entropy_loss_weight=0.0,
        scale=1.0,
        duration_loss_weight=1.0,
        pitch_loss_weight=1.0,
        aligner_loss_weight=1.0,
        aligner_bin_loss_weight=0.0,
    ):
        super().__init__()

        self.conditional = model.condition_on_prompt
        assert not self.conditional, "prompt-conditioned path not vendored (see module header)"

        self.model = model
        self.codec = None

        assert exists(target_sample_hz)

        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = None

        assert objective in {"x0", "eps", "v"}, "objective must be either predict x0 or noise"
        self.objective = objective

        self.dim = model.dim

        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f"invalid noise schedule {noise_schedule}")

        assert scale <= 1, "scale must be less than or equal to 1"
        self.scale = scale

        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        self.timesteps = timesteps
        self.use_ddim = use_ddim
        self.time_difference = time_difference
        self.train_prob_self_cond = train_prob_self_cond

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight

        self.duration_loss_weight = duration_loss_weight
        self.pitch_loss_weight = pitch_loss_weight
        self.aligner_loss_weight = aligner_loss_weight

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, audio, codes=None):
        """`audio` is pre-encoded codec latents of shape (batch, n, dim) --
        the codec=None / non-raw-waveform path (raw-waveform encoding via
        SoundStream/EncodecWrapper is not vendored, see module header)."""
        duration_pitch_loss = 0.0

        batch, n, d, device = *audio.shape, self.device  # noqa: F841

        assert d == self.dim, f"codec codebook dimension {d} must match model dimensions {self.dim}"

        times = torch.zeros((batch,), device=device).float().uniform_(0, 1.0)

        noise = torch.randn_like(audio)

        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(audio, gamma)
        alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)

        noised_audio = alpha * audio + sigma * noise

        pred = self.model(noised_audio, times, prompt=None, cond=None)

        if self.objective == "eps":
            target = noise
        elif self.objective == "x0":
            target = audio
        elif self.objective == "v":
            target = alpha * noise - sigma * audio

        loss = F.mse_loss(pred, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")

        snr = (alpha * alpha) / (sigma * sigma)
        maybe_clipped_snr = snr.clone()

        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=self.min_snr_gamma)

        if self.objective == "eps":
            loss_weight = maybe_clipped_snr / snr
        elif self.objective == "x0":
            loss_weight = maybe_clipped_snr
        elif self.objective == "v":
            loss_weight = maybe_clipped_snr / (snr + 1)

        loss = (loss * loss_weight).mean()

        if self.rvq_cross_entropy_loss_weight == 0 or not exists(codes):
            return loss

        return loss + duration_pitch_loss


# ---------------------------------------------------------------------------
# Tiny build/example for TorchLens tracing. Real architectural constants
# (objective='v', noise_schedule='sigmoid', condition_on_prompt=False) match
# the paper's default config; dim/depth/wavenet sizes shrunk for a fast trace.
# ---------------------------------------------------------------------------
def build_naturalspeech2():
    torch.manual_seed(0)
    codec_dim = 32
    denoiser = Model(
        dim=codec_dim,
        depth=2,
        dim_head=16,
        heads=2,
        wavenet_layers=2,
        wavenet_stacks=1,
        use_flash_attn=False,
        condition_on_prompt=False,
    )
    model = NaturalSpeech2(
        model=denoiser,
        target_sample_hz=24000,
        timesteps=8,
    )
    model.eval()
    return model


def example_input_naturalspeech2():
    torch.manual_seed(0)
    return torch.randn(2, 12, 32)


MENAGERIE_ENTRIES = [
    ("NaturalSpeech2", "build_naturalspeech2", "example_input_naturalspeech2", 2023, MENAGERIE_ZOO),
]
