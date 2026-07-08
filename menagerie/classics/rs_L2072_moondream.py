# SOURCE: vendored from vikhyat/moondream @ main
# (moondream/torch/{config,layers,vision,text,rope,image_crops}.py)
# https://github.com/vikhyat/moondream/tree/main/moondream/torch
"""Moondream: a small (1.6B/1.93B) vision-language model (Vikhyat Korrapati,
vikhyat/moondream, Apache-2.0). Architecture: a SigLIP-style ViT vision
encoder (linear patch embed via reshape, not conv-based patchify) feeding a
gated-MLP resampler ("vision_projection") that fuses a global low-res crop
with reassembled high-res local crops, whose output tokens are concatenated
with text token embeddings and fed to a Phi-style causal transformer decoder
with a fixed-width (32-dim) partial-RoPE and (from some layer onward) a
mixture-of-experts GeGLU MLP.

This module vendors the REAL functional architecture code verbatim from
``moondream/torch/{config,layers,vision,text,rope,image_crops}.py``:
``vision_encoder``/``vision_projection``/``build_vision_model`` (vision.py),
``text_encoder``/``text_decoder``/``lm_head``/``build_text_model`` (text.py),
``attn``/``mlp``/``moe_mlp``/``layer_norm``/``QuantizedLinear`` (layers.py),
``precompute_freqs_cis``/``apply_rotary_emb`` (rope.py), and
``reconstruct_from_crops`` (image_crops.py). Dataclass configs are copied
from config.py unmodified.

Not vendored (non-architectural / requires extra deps or network fetch):
``moondream.torch.moondream.MoondreamModel`` itself (autoregressive
generation loop, KV-cache bookkeeping, ``torch.nn.attention.flex_attention``
decode-time block masking, and a ``Tokenizer.from_pretrained(...)`` HF Hub
fetch), the ``pyvips``/PIL image-loading path in ``overlap_crop_image``
(image I/O, not network architecture), and ``lora.py`` /
``region.py`` (LoRA adapter loading and the pointing/detection region head,
both optional heads not exercised by the core caption/VQA forward pass).
This staging module drives the exact same real functions
(``vision_encoder`` -> ``reconstruct_from_crops`` -> ``vision_projection``
-> ``text_encoder`` -> ``text_decoder`` -> ``lm_head``) that
``MoondreamModel._run_vision_encoder`` / ``._prefill`` / ``lm_head`` call,
using a single (non-tiled) image crop and pre-tokenized ``input_ids`` in
place of PIL image loading and the HF tokenizer, with tiny dims so the
graph traces quickly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# config.py (verbatim dataclasses, only field defaults referenced below are
# overridden per-instance for a tiny test-sized model)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TextMoeConfig:
    num_experts: int = 64
    start_layer: int = 4
    experts_per_token: int = 8
    expert_inner_dim: int = 1024


@dataclass(frozen=True)
class TextConfig:
    dim: int = 2048
    ff_dim: int = 8192
    n_layers: int = 24
    vocab_size: int = 51200
    max_context: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 32
    prefix_attn: int = 730
    group_size: Optional[int] = None
    moe: Optional[TextMoeConfig] = TextMoeConfig()


@dataclass(frozen=True)
class VisionConfig:
    enc_dim: int = 1152
    enc_patch_size: int = 14
    enc_n_layers: int = 27
    enc_ff_dim: int = 4304
    enc_n_heads: int = 16
    proj_out_dim: int = 2048
    crop_size: int = 378
    in_channels: int = 3
    max_crops: int = 12
    overlap_margin: int = 4
    proj_inner_dim: int = 8192


@dataclass(frozen=True)
class TokenizerConfig:
    bos_id: int = 0
    eos_id: int = 0
    answer_id: int = 3
    thinking_id: int = 4
    coord_id: int = 5
    size_id: int = 6
    start_ground_points_id: int = 7
    end_ground_id: int = 9
    templates: Dict[str, Optional[Dict[str, List[int]]]] = field(
        default_factory=lambda: {
            "caption": {
                "short": [1, 32708, 2, 12492, 3],
                "normal": [1, 32708, 2, 6382, 3],
                "long": [1, 32708, 2, 4059, 3],
            },
            "query": {"prefix": [1, 15381, 2], "suffix": [3]},
            "detect": {"prefix": [1, 7235, 476, 2], "suffix": [3]},
            "point": {"prefix": [1, 2581, 2], "suffix": [3]},
        }
    )


@dataclass(frozen=True)
class MoondreamConfig:
    text: TextConfig = TextConfig()
    vision: VisionConfig = VisionConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()


# --------------------------------------------------------------------------
# layers.py (verbatim; QuantizedLinear's torchao unpack path is unused since
# group_size=None routes to plain nn.Linear, but the class is kept intact)
# --------------------------------------------------------------------------

try:
    from torchao import quantize_
    from torchao.quantization import int4_weight_only
except ImportError:

    def quantize_(model, quant_mode):
        raise ImportError("torchao is not installed. Please install it with `pip install torchao`.")

    def int4_weight_only(group_size):
        raise ImportError("torchao is not installed. Please install it with `pip install torchao`.")


def gelu_approx(x):
    return F.gelu(x, approximate="tanh")


def dequantize_tensor(W_q, scale, zero, orig_shape, dtype=torch.bfloat16):
    _step = W_q.shape[0]
    W_r = torch.empty([2 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)
    W_r[:_step] = (W_q & 0b11110000) >> 4
    W_r[_step:] = W_q & 0b00001111
    W_r.sub_(zero).mul_(scale)
    return W_r.reshape(orig_shape)


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.ParameterDict(
            {
                "packed": nn.Parameter(
                    torch.empty(out_features * in_features // (128 * 2), 128, dtype=torch.uint8),
                    requires_grad=False,
                ),
                "scale": nn.Parameter(
                    torch.empty(out_features * in_features // 128, 1),
                    requires_grad=False,
                ),
                "zero_point": nn.Parameter(
                    torch.empty(out_features * in_features // 128, 1),
                    requires_grad=False,
                ),
            }
        )
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        self.unpacked = False

    def unpack(self):
        if self.unpacked:
            return

        self.weight = nn.Parameter(
            dequantize_tensor(
                self.weight["packed"],
                self.weight["scale"],
                self.weight["zero_point"],
                (self.out_features, self.in_features),
                torch.bfloat16,
            )
        )
        with torch.device("meta"):
            self.linear = nn.Linear(self.in_features, self.out_features, dtype=torch.bfloat16)
        self.linear.weight = self.weight
        self.linear.bias = nn.Parameter(self.bias.to(torch.bfloat16), requires_grad=False)

        del self.weight, self.bias
        quantize_(self, int4_weight_only(group_size=128))
        self.unpacked = True
        torch.cuda.empty_cache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.unpacked:
            self.unpack()
        return self.linear(x)


def layer_norm(x: torch.Tensor, w: nn.Module) -> torch.Tensor:
    return F.layer_norm(x, w.bias.shape, w.weight, w.bias)


def mlp(x: torch.Tensor, w: nn.Module, lora: Optional[dict] = None) -> torch.Tensor:
    x0 = w.fc1(x)
    if lora is not None:
        x1 = F.linear(F.linear(x, lora["fc1"]["A"]), lora["fc1"]["B"])
        x = x0 + x1
    else:
        x = x0

    x = gelu_approx(x)

    x0 = w.fc2(x)
    if lora is not None:
        x1 = F.linear(F.linear(x, lora["fc2"]["A"]), lora["fc2"]["B"])
        x = x0 + x1
    else:
        x = x0

    return x


def moe_mlp(x: torch.Tensor, mlp_module: nn.Module, experts_per_token: int) -> torch.Tensor:
    B, T, C = x.shape
    x = x.reshape(-1, C)

    # Router computation
    router_logits = mlp_module.router(x)
    topk_logits, topk_idxs = torch.topk(router_logits, experts_per_token, dim=-1)
    topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32).to(x.dtype)
    num_tokens, top_k = topk_idxs.shape

    if T == 1:
        w1_weight = mlp_module.fc1.weight
        w2_weight = mlp_module.fc2.weight

        flat_idxs = topk_idxs.view(-1)  # [T*A]
        flat_weights = topk_weights.view(-1)  # [T*A]

        w1_selected = w1_weight[flat_idxs]  # [T*A, H, D]
        w2_selected = w2_weight[flat_idxs]  # [T*A, D, H]

        x_expanded = x.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, C)  # [T*A, D]

        x1_full = torch.bmm(w1_selected, x_expanded.unsqueeze(-1)).squeeze(-1)  # [T*A, H]
        x1, g = x1_full.chunk(2, dim=-1)
        x1 = F.gelu(x1) * (g + 1)

        expert_outs = torch.bmm(w2_selected, x1.unsqueeze(-1)).squeeze(-1)  # [T*A, D]

        weighted_outs = expert_outs * flat_weights.unsqueeze(-1)  # [T*A, D]
        weighted_outs = weighted_outs.view(num_tokens, top_k, C)  # [T, A, D]

        mlp_out = weighted_outs.sum(dim=1)  # [T, D]
        mlp_out = mlp_out.view(B, T, C)

        return mlp_out
    else:
        out = x.new_zeros(x.size())

        for expert_id in range(mlp_module.fc1.weight.shape[0]):
            token_pos, which_k = (topk_idxs == expert_id).nonzero(as_tuple=True)
            if token_pos.numel() == 0:
                continue

            x_tok = x.index_select(0, token_pos)
            gate_tok = topk_weights[token_pos, which_k]

            h_full = F.linear(x_tok, mlp_module.fc1.weight[expert_id])
            h, g = h_full.chunk(2, dim=-1)
            h = F.gelu(h) * (g + 1)
            y = F.linear(h, mlp_module.fc2.weight[expert_id])

            y.mul_(gate_tok.unsqueeze(-1))
            out.index_add_(0, token_pos, y)

        return out.view(B, T, C)


def attn_vision(x: torch.Tensor, w: nn.Module, n_heads: int) -> torch.Tensor:
    """Vision self-attention (vision.py's ``attn``, renamed to avoid a name
    clash with text.py's ``attn`` in this single-file staging module)."""
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    q, k, v = [
        t.view(bsz, q_len, n_heads, head_dim).transpose(1, 2) for t in w.qkv(x).chunk(3, dim=-1)
    ]
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
    out = w.proj(out)
    return out


# --------------------------------------------------------------------------
# rope.py (verbatim)
# --------------------------------------------------------------------------


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 1500000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=dtype).unsqueeze(1)
    freqs = t * freqs.unsqueeze(0)
    freqs = torch.exp(1j * freqs)
    return torch.stack([freqs.real, freqs.imag], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads: int,
    rot_dim: int = 32,
    interleave: bool = False,
) -> torch.Tensor:
    assert rot_dim == freqs_cis.shape[-2] * 2
    assert num_heads == x.shape[1]

    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    if interleave:
        xq_r = x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)[..., 0]
        xq_i = x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)[..., 1]
    else:
        d_q = x_rot.shape[-1] // 2
        xq_r, xq_i = x_rot[..., :d_q], x_rot[..., d_q:]

    freqs_cos = freqs_cis[..., 0][position_ids, :].unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_cis[..., 1][position_ids, :].unsqueeze(0).unsqueeze(0)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(-2)

    return torch.cat([xq_out.to(x.dtype), x_pass], dim=-1)


# --------------------------------------------------------------------------
# image_crops.py (only the pure-tensor reassembly function; the PIL/pyvips
# resize-and-tile path in overlap_crop_image is image I/O, not network
# architecture, so it is not vendored -- callers below supply a single
# already-cropped tensor with tiling=(1, 1))
# --------------------------------------------------------------------------


def reconstruct_from_crops(
    crops: torch.Tensor,
    tiling: "tuple[int, int]",
    overlap_margin: int,
    patch_size: int = 14,
) -> torch.Tensor:
    tiling_h, tiling_w = tiling
    crop_height, crop_width = crops[0].shape[:2]
    margin_pixels = overlap_margin * patch_size

    output_h = (crop_height - 2 * margin_pixels) * tiling_h + 2 * margin_pixels
    output_w = (crop_width - 2 * margin_pixels) * tiling_w + 2 * margin_pixels

    reconstructed = torch.zeros(
        (output_h, output_w, crops[0].shape[2]),
        device=crops[0].device,
        dtype=crops[0].dtype,
    )

    for i, crop in enumerate(crops):
        tile_y = i // tiling_w
        tile_x = i % tiling_w

        x_start = 0 if tile_x == 0 else margin_pixels
        x_end = crop_width if tile_x == tiling_w - 1 else crop_width - margin_pixels
        y_start = 0 if tile_y == 0 else margin_pixels
        y_end = crop_height if tile_y == tiling_h - 1 else crop_height - margin_pixels

        out_x = tile_x * (crop_width - 2 * margin_pixels)
        out_y = tile_y * (crop_height - 2 * margin_pixels)

        reconstructed[out_y + y_start : out_y + y_end, out_x + x_start : out_x + x_end] = crop[
            y_start:y_end, x_start:x_end
        ]

    return reconstructed


# --------------------------------------------------------------------------
# vision.py (verbatim graph functions + builder)
# --------------------------------------------------------------------------


def create_patches(x, patch_size):
    B, C, H, W = x.shape
    P1 = P2 = patch_size

    x = x.reshape(B, C, H // P1, P1, W // P2, P2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, (H // P1) * (W // P2), C * P1 * P2)

    return x


def vision_encoder(input_BCHW: torch.Tensor, w: nn.Module, config: VisionConfig):
    x = create_patches(input_BCHW, config.enc_patch_size)

    x = w.patch_emb(x)
    x = x + w.pos_emb
    for block in w.blocks:
        x = x + attn_vision(layer_norm(x, block.ln1), block.attn, n_heads=config.enc_n_heads)
        x = x + mlp(layer_norm(x, block.ln2), block.mlp)
    x = layer_norm(x, w.post_ln)

    return x


def vision_projection(
    global_features: torch.Tensor,
    reconstructed: torch.Tensor,
    w: nn.Module,
    config: VisionConfig,
):
    reconstructed = reconstructed.permute(2, 0, 1)
    reconstructed = F.adaptive_avg_pool2d(
        reconstructed, output_size=(config.enc_n_layers, config.enc_n_layers)
    )
    reconstructed = reconstructed.permute(1, 2, 0).reshape(
        config.enc_n_layers * config.enc_n_layers, config.enc_dim
    )
    final_features = torch.cat([global_features, reconstructed], dim=-1)
    return mlp(final_features, w.proj_mlp)


def build_vision_model(config: VisionConfig, dtype: torch.dtype):
    patch_dim = config.enc_patch_size * config.enc_patch_size * config.in_channels
    grid_size = config.crop_size // config.enc_patch_size
    num_patches = grid_size * grid_size

    vision = nn.ModuleDict(
        {
            "patch_emb": nn.Linear(patch_dim, config.enc_dim, dtype=dtype),
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln1": nn.LayerNorm(config.enc_dim, dtype=dtype),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": nn.Linear(
                                        config.enc_dim, 3 * config.enc_dim, dtype=dtype
                                    ),
                                    "proj": nn.Linear(config.enc_dim, config.enc_dim, dtype=dtype),
                                }
                            ),
                            "ln2": nn.LayerNorm(config.enc_dim, dtype=dtype),
                            "mlp": nn.ModuleDict(
                                {
                                    "fc1": nn.Linear(
                                        config.enc_dim, config.enc_ff_dim, dtype=dtype
                                    ),
                                    "fc2": nn.Linear(
                                        config.enc_ff_dim, config.enc_dim, dtype=dtype
                                    ),
                                }
                            ),
                        }
                    )
                    for _ in range(config.enc_n_layers)
                ]
            ),
            "post_ln": nn.LayerNorm(config.enc_dim, dtype=dtype),
            "proj_mlp": nn.ModuleDict(
                {
                    "fc1": nn.Linear(config.enc_dim * 2, config.proj_inner_dim, dtype=dtype),
                    "fc2": nn.Linear(config.proj_inner_dim, config.proj_out_dim, dtype=dtype),
                }
            ),
        }
    )
    vision.pos_emb = nn.Parameter(torch.zeros(1, num_patches, config.enc_dim, dtype=dtype))
    return vision


# --------------------------------------------------------------------------
# text.py (verbatim graph functions + builder)
# --------------------------------------------------------------------------


def text_encoder(input_ids: torch.Tensor, w: nn.Module):
    return F.embedding(input_ids, w.wte)


def attn_text(
    x: torch.Tensor,
    w: nn.Module,
    freqs_cis: torch.Tensor,
    attn_mask: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    position_ids: torch.Tensor,
    lora: Optional[dict] = None,
):
    """Text decoder self-attention (text.py's ``attn``, renamed to avoid a
    name clash with vision.py's ``attn`` in this single-file staging
    module). KV-cache and flex_attention block-mask decode-time paths are
    omitted (they only matter for incremental autoregressive generation,
    not the architecture graph); this always takes the
    scaled_dot_product_attention "prefill" branch of the real function."""
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    qkv_out = w.qkv(x)
    if lora is not None:
        qkv_out += F.linear(F.linear(x, lora["qkv"]["A"]), lora["qkv"]["B"])
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)

    q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    if hasattr(w, "tau") and w.tau is not None:
        tok_feat = F.gelu(qkv_out)
        tok_q = torch.tanh(torch.matmul(tok_feat, w.tau["wq"].t())).permute(0, 2, 1)
        tok_v = torch.tanh(torch.matmul(tok_feat, w.tau["wv"].t())).permute(0, 2, 1)
        pos = position_ids.to(q.dtype) + 1
        tau_pos = 1 + (torch.sigmoid(w.tau["alpha"][:, None] * pos.log()) - 0.5)  # (H,S)
        tau_q = (tok_q + tau_pos[None]).unsqueeze(-1)  # (B,H,S,1)
        tau_v = (tok_v + tau_pos[None]).unsqueeze(-1)
        q = q * tau_q
        v = v * tau_v

    q = apply_rotary_emb(q, freqs_cis, position_ids, n_heads)
    k = apply_rotary_emb(k, freqs_cis, position_ids, n_kv_heads)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, enable_gqa=n_heads != n_kv_heads
    )
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)

    out0 = w.proj(out)
    if lora is not None:
        out1 = F.linear(F.linear(x, lora["proj"]["A"]), lora["proj"]["B"])
        out = out0 + out1
    else:
        out = out0

    return out


def text_decoder(
    x: torch.Tensor,
    w: nn.Module,
    attn_mask: torch.Tensor,
    position_ids: torch.Tensor,
    config: TextConfig,
    lora: Optional[dict] = None,
):
    for i, block in enumerate(w.blocks):
        if lora is not None:
            layer_lora = lora["text"]["blocks"][str(i)]
            mlp_lora = layer_lora["mlp"]
            attn_lora = layer_lora["attn"]
        else:
            mlp_lora = None
            attn_lora = None

        l_in = layer_norm(x, block.ln)
        l_attn = attn_text(
            l_in,
            block.attn,
            freqs_cis=w.freqs_cis,
            attn_mask=attn_mask,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            position_ids=position_ids,
            lora=attn_lora,
        )

        if config.moe is not None and i >= config.moe.start_layer:
            l_mlp = moe_mlp(l_in, block.mlp, config.moe.experts_per_token)
        else:
            l_mlp = mlp(l_in, block.mlp, lora=mlp_lora)

        x = x + l_attn + l_mlp

    return x


def lm_head(hidden_BTC: torch.Tensor, w: nn.Module, indices: Optional[torch.Tensor] = None):
    hidden_BC = hidden_BTC[:, -1, :]
    hidden_BC = layer_norm(hidden_BC, w.post_ln)
    if indices is not None:
        logits = hidden_BC @ w.lm_head.weight[indices].T + w.lm_head.bias[indices]
    else:
        logits = w.lm_head(hidden_BC)
    return logits


def build_dense_mlp(d_model, d_ffn, dtype, linear_cls):
    return nn.ModuleDict(
        {
            "fc1": linear_cls(d_model, d_ffn, dtype=dtype),
            "fc2": linear_cls(d_ffn, d_model, dtype=dtype),
        }
    )


def build_moe_mlp(d_model, d_ffn, n_experts, dtype):
    return nn.ModuleDict(
        {
            "router": nn.Linear(d_model, n_experts, dtype=dtype),
            "fc1": nn.ParameterDict(
                {"weight": nn.Parameter(torch.empty(n_experts, 2 * d_ffn, d_model, dtype=dtype))}
            ),
            "fc2": nn.ParameterDict(
                {"weight": nn.Parameter(torch.empty(n_experts, d_model, d_ffn, dtype=dtype))}
            ),
        }
    )


def build_text_model(config: TextConfig, dtype: torch.dtype) -> nn.Module:
    qkv_dim = int(config.dim * (1 + 2 * config.n_kv_heads / config.n_heads))
    linear_cls = QuantizedLinear if config.group_size is not None else nn.Linear

    text = nn.ModuleDict(
        {
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln": nn.LayerNorm(config.dim, dtype=dtype),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": linear_cls(config.dim, qkv_dim, dtype=dtype),
                                    "proj": linear_cls(config.dim, config.dim, dtype=dtype),
                                    "tau": nn.ParameterDict(
                                        {
                                            "wq": nn.Parameter(
                                                torch.empty(config.n_heads, qkv_dim, dtype=dtype)
                                            ),
                                            "wv": nn.Parameter(
                                                torch.empty(config.n_heads, qkv_dim, dtype=dtype)
                                            ),
                                            "alpha": nn.Parameter(
                                                torch.empty(config.n_heads, dtype=dtype)
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "mlp": (
                                build_moe_mlp(
                                    config.dim,
                                    config.moe.expert_inner_dim,
                                    config.moe.num_experts,
                                    dtype,
                                )
                                if config.moe is not None and layer_idx >= config.moe.start_layer
                                else build_dense_mlp(config.dim, config.ff_dim, dtype, linear_cls)
                            ),
                        }
                    )
                    for layer_idx in range(config.n_layers)
                ]
            ),
            "post_ln": nn.LayerNorm(config.dim, dtype=dtype),
            "lm_head": nn.Linear(config.dim, config.vocab_size, dtype=dtype),
        }
    )
    text.wte = nn.Parameter(torch.empty(config.vocab_size, config.dim, dtype=dtype))
    text.register_buffer(
        "freqs_cis",
        precompute_freqs_cis(config.dim // (2 * config.n_heads), config.max_context),
        persistent=False,
    )

    return text


# --------------------------------------------------------------------------
# Staging wrapper: drives the real functional pipeline end to end, mirroring
# MoondreamModel._run_vision_encoder (vision_encoder -> reconstruct_from_crops
# -> vision_projection) followed by MoondreamModel._prefill + lm_head
# (text_encoder -> concat image/text embeddings -> text_decoder -> lm_head).
# --------------------------------------------------------------------------


class MoondreamCore(nn.Module):
    """Tiny-sized real Moondream vision+text core (no tokenizer/PIL/LoRA/
    region-head machinery), built from the vendored functions above."""

    def __init__(self, config: MoondreamConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.vision = build_vision_model(config.vision, dtype)
        self.text = build_text_model(config.text, dtype)

    def forward(self, image_crops_bchw: torch.Tensor, input_ids: torch.Tensor):
        """``image_crops_bchw`` is ``(n_crops, C, crop_size, crop_size)`` with
        crop 0 the global (whole-image) crop and crops 1.. the (here: single,
        tiling=(1, 1)) local tile(s), matching the real
        ``MoondreamModel._run_vision_encoder`` batch-of-crops convention."""
        cfg = self.config

        # -- vision tower (real vision_encoder over the batch of crops) --
        vis_out = vision_encoder(image_crops_bchw, self.vision, cfg.vision)
        global_features = vis_out[0]
        local_features = vis_out[1:].reshape(
            -1, cfg.vision.enc_n_layers, cfg.vision.enc_n_layers, cfg.vision.enc_dim
        )

        reconstructed = reconstruct_from_crops(
            local_features,
            tiling=(1, 1),
            overlap_margin=0,
            patch_size=1,
        )
        image_tokens = vision_projection(global_features, reconstructed, self.vision, cfg.vision)
        image_tokens = image_tokens.unsqueeze(0)  # (1, n_img_tokens, dim)

        # -- text embeddings + fusion (real text_encoder) --
        text_tokens = text_encoder(input_ids, self.text)

        x = torch.cat([image_tokens, text_tokens], dim=1)
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device)
        causal_mask = torch.tril(
            torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # -- causal Phi-MoE decoder + LM head (real text_decoder / lm_head) --
        hidden = text_decoder(x, self.text, causal_mask, position_ids, cfg.text, lora=None)
        logits = lm_head(hidden.unsqueeze(0) if hidden.dim() == 2 else hidden, self.text)
        return logits


MENAGERIE_ZOO = "vendored-pytorch"

_TINY_VISION = VisionConfig(
    enc_dim=32,
    enc_patch_size=14,
    # Real Moondream keeps enc_n_layers (transformer depth) numerically equal
    # to crop_size // enc_patch_size (the per-crop patch-grid side), since
    # vision_projection's adaptive_avg_pool2d target size is `config.enc_n_layers`
    # patches per side. Mirror that invariant here: 2 blocks, 2x2 patch grid.
    enc_n_layers=2,
    enc_ff_dim=64,
    enc_n_heads=4,
    proj_out_dim=128,  # must match TextConfig.dim (image/text tokens are concatenated)
    crop_size=28,  # crop_size // enc_patch_size == enc_n_layers == 2
    in_channels=3,
    max_crops=1,
    overlap_margin=0,
    proj_inner_dim=64,
)
_TINY_TEXT = TextConfig(
    # apply_rotary_emb's rot_dim defaults to a fixed 32 (real Moondream head_dim
    # is always 64 >> 32). precompute_freqs_cis is called with
    # dim=config.dim // (2 * n_heads), and apply_rotary_emb requires
    # rot_dim == freqs_cis.shape[-2] * 2 == (dim // 2) * 2 == dim, so we need
    # config.dim // (2 * n_heads) == 32, i.e. config.dim == 64 * n_heads.
    dim=128,
    ff_dim=64,
    n_layers=2,
    vocab_size=64,
    max_context=64,
    n_heads=2,
    n_kv_heads=2,
    prefix_attn=5,
    group_size=None,
    moe=TextMoeConfig(num_experts=4, start_layer=1, experts_per_token=2, expert_inner_dim=16),
)
_TINY_CONFIG = MoondreamConfig(text=_TINY_TEXT, vision=_TINY_VISION)


def build_moondream_core():
    torch.manual_seed(0)
    model = MoondreamCore(_TINY_CONFIG, dtype=torch.float32)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)
            else:
                nn.init.zeros_(p)
    return model


def example_input_moondream_core():
    torch.manual_seed(0)
    # 2 crops: index 0 = global (whole-image) crop, index 1 = the single
    # local tile (tiling=(1, 1)), matching MoondreamModel's real
    # all_crops = [global_crop, *local_crops] batch-of-crops convention.
    image_crops = torch.randn(2, 3, _TINY_VISION.crop_size, _TINY_VISION.crop_size)
    input_ids = torch.randint(0, _TINY_TEXT.vocab_size, (1, 6))
    return (image_crops, input_ids)


MENAGERIE_ENTRIES = [
    (
        "Moondream",
        "build_moondream_core",
        "example_input_moondream_core",
        2024,
        MENAGERIE_ZOO,
    ),
]
