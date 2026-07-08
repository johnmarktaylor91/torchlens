# SOURCE: vendored from https://github.com/HIM-AIM/RNADiffFold @ master
#
# RNADiffFold (HIM-AIM) -- a generative RNA secondary-structure predictor built on
# multinomial (discrete) diffusion over binary contact maps, conditioned on (1) a
# frozen-style RNA-FM (ESM/RoBERTa-style RNA language model) sequence encoder's
# per-token embeddings + attention maps, and (2) a UFold-style U-Net conditioner over
# hand-engineered pairing features. The denoising network is a conditional 2D U-Net
# operating on the contact-map "image". Vendored verbatim (architecture-relevant
# classes only) from the repo's own files:
#   https://raw.githubusercontent.com/HIM-AIM/RNADiffFold/master/models/model.py
#   https://raw.githubusercontent.com/HIM-AIM/RNADiffFold/master/models/layers.py
#   https://raw.githubusercontent.com/HIM-AIM/RNADiffFold/master/models/diffusion_multinomial.py
#   https://raw.githubusercontent.com/HIM-AIM/RNADiffFold/master/models/condition/u_conditioner.py
#   https://raw.githubusercontent.com/HIM-AIM/RNADiffFold/master/models/condition/fm_conditioner/fm/model.py
#   https://raw.githubusercontent.com/HIM-AIM/RNADiffFold/master/models/condition/fm_conditioner/fm/modules.py
#   https://raw.githubusercontent.com/HIM-AIM/RNADiffFold/master/models/condition/fm_conditioner/fm/axial_attention.py
#   https://raw.githubusercontent.com/HIM-AIM/RNADiffFold/master/models/condition/fm_conditioner/fm/multihead_attention.py
#   https://raw.githubusercontent.com/HIM-AIM/RNADiffFold/master/models/condition/fm_conditioner/fm/data.py
#   https://raw.githubusercontent.com/HIM-AIM/RNADiffFold/master/models/condition/fm_conditioner/fm/constants.py
#
# What is kept: every architectural class unmodified -- Unet_conditioner (+conv_block/
# up_conv), the RNA-FM RNABertModel (+TransformerLayer/MultiheadAttention/RobertaLMHead/
# ContactPredictionHead/LearnedPositionalEmbedding/ESM1LayerNorm/ESM1bLayerNorm with the
# original apex-fallback), SegmentationUnet2DCondition (+ResnetBlock/LinearAttention/
# up/downsample blocks) as the diffusion denoiser, and MultinomialDiffusion's forward
# (loss-computing) pass. The `640`/`240` hardcoded conditioning dims inside
# SegmentationUnet2DCondition tie the model to the real RNA-FM-t12 config
# (embed_dim=640, layers=12, attention_heads=20 -> layers*heads=240), so that config is
# used verbatim rather than shrunk, to keep the vendored graph faithful; only the input
# RNA sequence length is kept tiny to make tracing fast.
#
# What is dropped (checkpoint I/O, not architecture): `pretrained.py`'s
# `load_model_and_alphabet_local`/`_hub` functions read `model_data["args"]` out of a
# downloaded/on-disk `.pth` checkpoint to build the `Namespace` fed to `RNABertModel`;
# there is no local checkpoint available in this environment, so the staging glue below
# builds the identical `Namespace(arch="roberta_large", ...)` + `Alphabet.from_architecture`
# directly (same architecture, freshly random-initialized, no weight loading) instead of
# calling `load_model_and_alphabet_local`. Similarly `Unet_conditioner`/`RNABertModel` are
# constructed fresh rather than `.load_state_dict`-restored from `ufold_train_alldata.pt` /
# `RNA-FM_pretrained.pth`. `DiffusionRNA2dPrediction.load_u_conditioner`'s post-hoc
# `Conv_1x1` channel-count swap (`cond_dim`) is preserved verbatim. The unused
# `import lightning.pytorch as pl` from the original `model.py` is dropped (never
# referenced -- `DiffusionRNA2dPrediction` subclasses `nn.Module`, not
# `pl.LightningModule`).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import math
from argparse import Namespace
from inspect import isfunction
from itertools import product
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum
from tqdm import tqdm

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from models/condition/fm_conditioner/fm/constants.py
# ---------------------------------------------------------------------------
rnaseq_toks = {
    "toks": ["A", "C", "G", "U", "R", "Y", "K", "M", "S", "W", "B", "D", "H", "V", "N", "-"]
}


# ---------------------------------------------------------------------------
# from models/condition/fm_conditioner/fm/data.py (Alphabet only)
# ---------------------------------------------------------------------------
class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    @classmethod
    def from_architecture(cls, name: str, theme="protein") -> "Alphabet":
        if name in ("ESM-1b", "roberta_large"):
            standard_toks = rnaseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)


# ---------------------------------------------------------------------------
# from models/condition/fm_conditioner/fm/multihead_attention.py (fairseq, MIT)
# ---------------------------------------------------------------------------
def utils_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


class MultiheadAttention(nn.Module):
    """Multi-headed attention. See "Attention Is All You Need" for more details."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.enable_torch_version = hasattr(F, "multi_head_attention_forward")

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        assert k is not None
        src_len = k.size(1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = (
                attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len)
                .type_as(attn)
                .transpose(1, 0)
            )
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


# ---------------------------------------------------------------------------
# from models/condition/fm_conditioner/fm/modules.py (fairseq/ESM, MIT)
# ---------------------------------------------------------------------------
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)
    normalized = x - avg
    return normalized


class ESM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)
except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm


class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn


class LearnedPositionalEmbedding(nn.Embedding):
    """Learns positional embeddings up to a fixed maximum size."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None):
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class ContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        prepend_bos: bool,
        append_eos: bool,
        bias=True,
        eos_idx: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attentions):
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx).to(attentions)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        attentions = attentions.to(next(self.parameters()))
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))


# ---------------------------------------------------------------------------
# from models/condition/fm_conditioner/fm/model.py (fairseq/ESM RNA-FM, MIT)
# ---------------------------------------------------------------------------
class RNABertModel(nn.Module):
    """RNA-FM: ESM/RoBERTa-style RNA language model used here as a frozen-style
    sequence conditioner (referenced from ProteinBertModel, fairseq-esm)."""

    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        if self.args.arch == "roberta_large":
            self.model_version = "ESM-1b"
            self._init_submodules_esm1b()
        else:
            self.model_version = "ESM-1"
            self._init_submodules_esm1()

    def _init_submodules_common(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads,
                    add_bias_kv=(self.model_version != "ESM-1b"),
                    use_esm1b_layer_norm=(self.model_version == "ESM-1b"),
                )
                for _ in range(self.args.layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.args.layers * self.args.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )

    def _init_submodules_esm1b(self):
        self._init_submodules_common()
        self.embed_scale = 1
        self.embed_positions = LearnedPositionalEmbedding(
            self.args.max_positions, self.args.embed_dim, self.padding_idx
        )
        self.emb_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.args.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(
        self,
        tokens,
        repr_layers=(),
        need_head_weights=False,
        return_contacts=False,
        masked_tokens=None,
    ):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)

        x = self.embed_scale * self.embed_tokens(tokens)
        x = x + self.embed_positions(tokens)

        x = self.emb_layer_norm_before(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        layer_idx = -1
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x, masked_tokens)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result


# ---------------------------------------------------------------------------
# from models/condition/u_conditioner.py
# ---------------------------------------------------------------------------
CH_FOLD = 1


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class Unet_conditioner(nn.Module):
    """UFold-style U-Net conditioner. data_seq: (batch, seq_len, 4); requires_channels: 17."""

    def __init__(self, img_ch=17, output_ch=1):
        super(Unet_conditioner, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=int(32 * CH_FOLD))
        self.Conv2 = conv_block(ch_in=int(32 * CH_FOLD), ch_out=int(64 * CH_FOLD))
        self.Conv3 = conv_block(ch_in=int(64 * CH_FOLD), ch_out=int(128 * CH_FOLD))
        self.Conv4 = conv_block(ch_in=int(128 * CH_FOLD), ch_out=int(256 * CH_FOLD))
        self.Conv5 = conv_block(ch_in=int(256 * CH_FOLD), ch_out=int(512 * CH_FOLD))

        self.Up5 = up_conv(ch_in=int(512 * CH_FOLD), ch_out=int(256 * CH_FOLD))
        self.Up_conv5 = conv_block(ch_in=int(512 * CH_FOLD), ch_out=int(256 * CH_FOLD))

        self.Up4 = up_conv(ch_in=int(256 * CH_FOLD), ch_out=int(128 * CH_FOLD))
        self.Up_conv4 = conv_block(ch_in=int(256 * CH_FOLD), ch_out=int(128 * CH_FOLD))

        self.Up3 = up_conv(ch_in=int(128 * CH_FOLD), ch_out=int(64 * CH_FOLD))
        self.Up_conv3 = conv_block(ch_in=int(128 * CH_FOLD), ch_out=int(64 * CH_FOLD))

        self.Up2 = up_conv(ch_in=int(64 * CH_FOLD), ch_out=int(32 * CH_FOLD))
        self.Up_conv2 = conv_block(ch_in=int(64 * CH_FOLD), ch_out=int(32 * CH_FOLD))

        self.Conv_1x1 = nn.Conv2d(int(32 * CH_FOLD), output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return torch.transpose(d1, -1, -2) * d1


# ---------------------------------------------------------------------------
# from models/layers.py (diffusion denoiser U-Net)
# ---------------------------------------------------------------------------
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample_new(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.upsample(x)


class Downsample_SP_conv(nn.Module):
    """https://arxiv.org/abs/2208.03641 SP-conv downsample (pixel-unshuffle style)."""

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.downsample = nn.Sequential(
            Rearrange("b c (h s1) (w s2) -> b (c s1 s2) h w", s1=2, s2=2),
            nn.Conv2d(dim * 4, dim_out, 1),
        )

    def forward(self, x):
        return self.downsample(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    def __init__(self, dim: int, scale: float = 1.0, flip_sin_to_cos=False):
        super().__init__()
        assert (dim % 2) == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(self.half_dim) * scale)
        self.flip_sin_to_cos = flip_sin_to_cos

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        if self.flip_sin_to_cos:
            fouriered = torch.cat(
                [fouriered[:, self.half_dim :], fouriered[:, : self.half_dim]], dim=-1
            )
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1), nn.GroupNorm(groups, dim_out), Mish()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, cond_dim=None, groups=8):
        super().__init__()

        self.time_mlp = None
        if exists(time_emb_dim):
            self.time_mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))

        self.cond_mlp = None
        if exists(cond_dim):
            self.cond_mlp = nn.Sequential(Mish(), nn.Linear(cond_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        h = self.block1(x)
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")

        h += time_emb

        if exists(self.cond_mlp) and exists(cond):
            cond = rearrange(cond, "b c h w -> b h w c")
            cond = self.cond_mlp(cond)
            cond = rearrange(cond, "b h w c -> b c h w")
            h += cond

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SegmentationUnet2DCondition(nn.Module):
    """Conditional 2D U-Net denoiser for the discrete diffusion process."""

    def __init__(
        self,
        num_classes,
        dim,
        cond_dim,
        num_steps,
        dim_mults=(1, 2, 4, 8),
        dropout=0.0,
        learned_time_emb=True,
        cat_cond=True,
        scale_skip_connection=False,
    ):
        super().__init__()
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.embedding = nn.Embedding(num_classes, dim)

        self.dim = dim
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.cat_cond = cat_cond
        self.skip_connect_scale = 1.0 if not scale_skip_connection else (2**-0.5)

        self.dropout = nn.Dropout(p=dropout)

        self.time_pos_emb = LearnedSinusoidalPosEmb(dim, scale=1.0, flip_sin_to_cos=False)

        self.to_time_cond = nn.Sequential(
            self.time_pos_emb,
            nn.Linear(self.dim + 1, 4 * self.dim),
            Mish(),
            nn.Linear(4 * self.dim, self.dim),
        )

        self.to_cond = nn.Sequential(
            nn.Linear(2 * self.dim + self.cond_dim, 4 * self.dim),
            Mish(),
            nn.Linear(4 * self.dim, self.dim),
        )

        # NOTE: 640 and 240 are hardcoded in the real repo, tied to the real RNA-FM-t12
        # config (embed_dim=640, layers*attention_heads=12*20=240) -- kept verbatim.
        self.fm_cond_1 = nn.Sequential(nn.Linear(640, 64), Mish(), nn.Linear(64, 8))

        self.fm_cond_2 = nn.Sequential(nn.Linear(240, 64), Mish(), nn.Linear(64, 8))

        self.fm_cond = nn.Sequential(nn.Linear(16, 48), Mish(), nn.Linear(48, 8))

        self.x_mlp = nn.Sequential(nn.Linear(48, 64), Mish(), nn.Linear(64, self.dim))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in, dim_out, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8
                        ),
                        ResnetBlock(
                            dim_out, dim_out, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8
                        ),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample_SP_conv(dim_out) if not is_last else nn.Identity(),
                        Downsample_SP_conv(self.cond_dim) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_blocks1 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8
        )
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out * 2, dim_out, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8
                        ),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Upsample_new(dim_out) if not is_last else nn.Identity(),
                        ResnetBlock(
                            dim_out, dim_in, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8
                        ),
                        ResnetBlock(
                            dim_in, dim_in, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8
                        ),
                    ]
                )
            )

        out_dim = num_classes
        self.res_conv = ResnetBlock(dim, dim, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8)
        self.out_conv = nn.Conv2d(dim, out_dim, 1)

    def forward(self, time, x, fm_condition, u_condition, seq_encoding):
        x_shape = x.shape[1:]
        if len(x.size()) == 3:
            x = x.unsqueeze(1)

        B, C, H, W = x.size()

        x = self.embedding(x)
        assert x.shape == (B, C, H, W, self.dim)
        x = x.permute(0, 1, 4, 2, 3)
        assert x.shape == (B, C, self.dim, H, W)

        x = x.reshape(B, C * self.dim, H, W)

        cond = None

        fm_embedding = self.fm_cond_1(fm_condition["fm_embedding"]).permute(0, 2, 1)
        fm_attention_map = self.fm_cond_2(
            fm_condition["fm_attention_map"].permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        cond_L = fm_embedding.size(-1)

        fm_out_cat = torch.cat(
            [
                fm_embedding.unsqueeze(-1).repeat(1, 1, 1, cond_L),
                fm_embedding.unsqueeze(-2).repeat(1, 1, cond_L, 1),
            ],
            dim=1,
        )
        seq_encoding = seq_encoding.permute(0, 2, 1)
        seq_out_cat = torch.cat(
            [
                seq_encoding.unsqueeze(-1).repeat(1, 1, 1, cond_L),
                seq_encoding.unsqueeze(-2).repeat(1, 1, cond_L, 1),
            ],
            dim=1,
        )

        x = self.x_mlp(
            torch.cat([x, fm_out_cat, fm_attention_map, seq_out_cat, u_condition], dim=1).permute(
                0, 2, 3, 1
            )
        ).permute(0, 3, 1, 2)

        t = self.to_time_cond(time)

        hiddens = []

        for ind, (resnet1, resnet2, attn, downsample, cond_downsample) in enumerate(self.downs):
            x = resnet1(x, t, cond)
            x = self.dropout(x)
            x = resnet2(x, t, cond)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x)

        x = self.mid_blocks1(x, t, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, cond)

        for ind, (resnet1, attn, upsample, resnet2, resnet3) in enumerate(self.ups):
            x = torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim=1)
            x = resnet1(x, t, cond)
            x = attn(x)
            x = upsample(x)
            x = resnet2(x, t, cond)
            x = resnet3(x, t, cond)

        x = self.res_conv(x, t, cond)
        final = self.out_conv(x).view(B, self.num_classes, *x_shape)
        return torch.transpose(final, -1, -2) * final


# ---------------------------------------------------------------------------
# from models/diffusion_multinomial.py
# ---------------------------------------------------------------------------
def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def index_to_log_onehot(x, K):
    assert x.max().item() < K, f"Error: {x.max().item()} >= {K}"
    x_onehot = F.one_hot(x, K)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_categorical(log_x_0, log_prob):
    return (log_x_0.exp() * log_prob).sum(dim=1)


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def beta_schedule(num_steps, schedule_name="cosine", s=0.01):
    """cosine schedule, as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
    t = torch.arange(0, num_steps + 1, dtype=torch.float64)
    if schedule_name == "cosine":
        f_t = torch.cos(((t / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    elif schedule_name == "sqrt":
        f_t = 1 - torch.sqrt(t / num_steps + 0.0001)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    alpha_bars = f_t / f_t[0]
    alphas = alpha_bars[1:] / alpha_bars[:-1]
    alphas = torch.clamp(alphas, min=0.001, max=0.999)
    alphas = torch.sqrt(alphas)
    return alphas


class MultinomialDiffusion(nn.Module):
    def __init__(self, num_classes, time_steps, denoise_fn):
        super(MultinomialDiffusion, self).__init__()
        self.K = num_classes
        self.time_steps = time_steps
        self._denoise_fn = denoise_fn

        alphas = beta_schedule(time_steps, schedule_name="cosine", s=0.01)
        log_alphas = torch.log(alphas)
        log_alpha_bars = torch.cumsum(log_alphas, dim=0)
        log_1_minus_alphas = torch.log(1 - torch.exp(log_alphas) + 1e-40)
        log_1_minus_alpha_bars = torch.log(1 - torch.exp(log_alpha_bars) + 1e-40)

        self.register_buffer("log_alphas", log_alphas.float())
        self.register_buffer("log_alpha_bars", log_alpha_bars.float())
        self.register_buffer("log_1_minus_alphas", log_1_minus_alphas.float())
        self.register_buffer("log_1_minus_alpha_bars", log_1_minus_alpha_bars.float())

        self.register_buffer("Lt_history", torch.zeros(self.time_steps))
        self.register_buffer("Lt_count", torch.zeros(self.time_steps))

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.K)
        return log_sample

    def q_pred_one_step(self, log_x_t, t):
        log_alphas_t = extract(self.log_alphas, t, log_x_t.shape)
        log_1_minus_alphas_t = extract(self.log_1_minus_alphas, t, log_x_t.shape)
        log_probs = log_add_exp(log_x_t + log_alphas_t, log_1_minus_alphas_t - math.log(self.K))
        return log_probs

    def q_pred(self, log_x_0, t):
        log_alpha_bars_t = extract(self.log_alpha_bars, t, log_x_0.shape)
        log_1_minus_alpha_bars_t = extract(self.log_1_minus_alpha_bars, t, log_x_0.shape)
        log_probs = log_add_exp(
            log_x_0 + log_alpha_bars_t, log_1_minus_alpha_bars_t - math.log(self.K)
        )
        return log_probs

    def q_posterior(self, log_x_t, log_x_0, t):
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_0, t_minus_1)

        num_axes = (1,) * (len(log_x_0.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_0)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_0, log_EV_qxtmin_x0)

        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_step(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - torch.logsumexp(
            unnormed_logprobs, dim=1, keepdim=True
        )

        return log_EV_xtmin_given_xt_given_xstart

    def predict_x_0(self, log_x_t, t, fm_condition, u_condition, seq_encoding):
        x_t = log_onehot_to_index(log_x_t)

        out = self._denoise_fn(t, x_t, fm_condition, u_condition, seq_encoding)

        log_pred = F.log_softmax(out, dim=1)

        return log_pred

    def p_pred(self, log_x_t, t, fm_condition, u_condition, seq_encoding):
        log_x_0_hat = self.predict_x_0(log_x_t, t, fm_condition, u_condition, seq_encoding)
        log_probs = self.q_posterior(log_x_t, log_x_0_hat, t)
        return log_probs

    def q_sample(self, log_x_0, t):
        log_EV_qxt_x0 = self.q_pred(log_x_0, t)
        log_x_t = self.log_sample_categorical(log_EV_qxt_x0)
        return log_x_t

    def kl_prior(self, log_x_0):
        b = log_x_0.size(0)
        device = log_x_0.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_0, t=(self.time_steps - 1) * ones)
        log_half_prob = -torch.log(self.K * torch.ones_like(log_qxT_prob))
        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(
        self,
        log_x_0,
        log_x_t,
        fm_condition,
        u_condition,
        seq_encoding,
        t,
        contact_masks,
        detach_mean=False,
    ):
        log_true_prob = self.q_posterior(log_x_t=log_x_t, log_x_0=log_x_0, t=t)

        log_model_prob = self.p_pred(
            log_x_t=log_x_t,
            t=t,
            fm_condition=fm_condition,
            u_condition=u_condition * contact_masks,
            seq_encoding=seq_encoding,
        )

        if detach_mean:
            log_model_prob = log_model_prob.detach()
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_0, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1.0 - mask) * kl

        return loss

    def sample_time(self, b, device, method="uniform"):
        t = torch.randint(0, self.time_steps, (b,), device=device).long()
        pt = torch.ones_like(t).float() / self.time_steps
        return t, pt

    def forward(self, x_0, fm_condition, u_condition, contact_masks, seq_encoding):
        batch, device = x_0.size(0), x_0.device

        t, pt = self.sample_time(batch, device, "uniform")
        log_x_0 = index_to_log_onehot(x_0, self.K)

        kl = self.compute_Lt(
            log_x_0=log_x_0,
            log_x_t=self.q_sample(log_x_0, t),
            fm_condition=fm_condition,
            u_condition=u_condition,
            seq_encoding=seq_encoding,
            t=t,
            contact_masks=contact_masks,
        )

        kl_prior = self.kl_prior(log_x_0)

        vb_loss = kl / pt + kl_prior

        return -vb_loss


# ---------------------------------------------------------------------------
# from models/model.py (top-level DiffusionRNA2dPrediction)
# ---------------------------------------------------------------------------
class DiffusionRNA2dPrediction(nn.Module):
    def __init__(
        self,
        num_classes,
        diffusion_dim,
        cond_dim,
        diffusion_steps,
        dp_rate,
        fm_layers,
        fm_embed_dim,
        fm_ffn_embed_dim,
        fm_attention_heads,
        fm_max_positions,
    ):
        super(DiffusionRNA2dPrediction, self).__init__()

        self.num_classes = num_classes
        self.diffusion_dim = diffusion_dim
        self.cond_dim = cond_dim
        self.diffusion_steps = diffusion_steps
        self.dp_rate = dp_rate

        # condition (RNA-FM built fresh from a plain Namespace + Alphabet, in place of
        # the original repo's checkpoint-driven `load_model_and_alphabet_local`)
        self.alphabet = Alphabet.from_architecture("roberta_large", theme="rna")
        fm_args = Namespace(
            arch="roberta_large",
            layers=fm_layers,
            embed_dim=fm_embed_dim,
            ffn_embed_dim=fm_ffn_embed_dim,
            attention_heads=fm_attention_heads,
            max_positions=fm_max_positions,
        )
        self.fm_conditioner = RNABertModel(fm_args, self.alphabet)

        self.u_conditioner = Unet_conditioner(img_ch=17, output_ch=1)
        condition_out = nn.Conv2d(
            int(32 * CH_FOLD), self.cond_dim, kernel_size=1, stride=1, padding=0
        )
        self.u_conditioner.Conv_1x1 = condition_out

        self.denoise_layer = SegmentationUnet2DCondition(
            num_classes=self.num_classes,
            dim=self.diffusion_dim,
            cond_dim=self.cond_dim,
            num_steps=self.diffusion_steps,
            dim_mults=(1, 2, 4, 8),
            dropout=self.dp_rate,
        )

        self.diffusion = MultinomialDiffusion(
            self.num_classes, self.diffusion_steps, self.denoise_layer
        )

    def get_alphabet(self):
        return self.alphabet

    def get_fm_embedding(self, data_seq_raw, set_max_len):
        device = data_seq_raw.device

        fm_condition = dict()

        backbone_result = self.fm_conditioner(
            data_seq_raw,
            need_head_weights=False,
            repr_layers=[self.fm_conditioner.args.layers],
            return_contacts=True,
        )
        fm_embedding = backbone_result["representations"][self.fm_conditioner.args.layers]
        fm_embedding = fm_embedding[:, 1:-1, :]

        fm_attention_map = backbone_result["attentions"]
        b, l, n, l1, l2 = fm_attention_map.shape  # noqa: E741 -- matches original repo variable naming
        fm_attention_map = fm_attention_map.reshape(b, l * n, l1, l2)[:, :, 1:-1, 1:-1]

        padding_value = 0
        padding_size = (
            0,
            set_max_len - fm_attention_map.shape[-2],
            0,
            set_max_len - fm_attention_map.shape[-1],
        )
        fm_embedding_pad = torch.zeros(
            fm_embedding.shape[0], set_max_len - fm_embedding.shape[1], fm_embedding.shape[2]
        ).to(device)
        fm_embedding = torch.cat([fm_embedding, fm_embedding_pad], dim=1)

        fm_attention_map = F.pad(fm_attention_map, padding_size, "constant", value=padding_value)

        fm_condition["fm_embedding"] = fm_embedding
        fm_condition["fm_attention_map"] = fm_attention_map

        return fm_condition

    def get_ufold_condition(self, data_fcn_2):
        return self.u_conditioner(data_fcn_2)

    def forward(self, x_0, data_fcn_2, data_seq_raw, contact_masks, set_max_len, data_seq_encoding):
        fm_condition = self.get_fm_embedding(data_seq_raw, set_max_len)
        u_condition = self.get_ufold_condition(data_fcn_2)

        loglik_bpd = self.diffusion(
            x_0, fm_condition, u_condition, contact_masks, data_seq_encoding
        )

        return loglik_bpd


# ---------------------------------------------------------------------------
# staging glue (not part of the original architecture)
# ---------------------------------------------------------------------------
def build_rnadifffold():
    # RNA-FM-t12-sized conditioner (embed_dim=640, layers=12, heads=20 -> 240) is kept
    # verbatim to match SegmentationUnet2DCondition's hardcoded 640/240 conditioning
    # dims; only the diffusion U-Net width/steps and sequence length are shrunk for a
    # fast trace.
    return DiffusionRNA2dPrediction(
        num_classes=2,
        diffusion_dim=8,
        cond_dim=8,
        diffusion_steps=8,
        dp_rate=0.0,
        fm_layers=12,
        fm_embed_dim=640,
        fm_ffn_embed_dim=5120,
        fm_attention_heads=20,
        fm_max_positions=64,
    )


def example_input_rnadifffold():
    # set_max_len must be >= 16 so the 4-stage Unet_conditioner (4x MaxPool2d) doesn't
    # collapse a spatial dim to 1 (BatchNorm2d requires >1 element per channel).
    set_max_len = 16
    batch = 2

    # x_0: binary contact map ground truth, (batch, L, L)
    x_0 = torch.randint(0, 2, (batch, set_max_len, set_max_len)).long()

    # data_fcn_2: UFold-style 17-channel pairing feature map, (batch, 17, L, L)
    data_fcn_2 = torch.randn(batch, 17, set_max_len, set_max_len)

    # data_seq_raw: RNA-FM tokens with <cls>/<eos> framing, (batch, L+2)
    alphabet = Alphabet.from_architecture("roberta_large", theme="rna")
    body = torch.randint(0, 4, (batch, set_max_len)).long()
    cls_col = torch.full((batch, 1), alphabet.cls_idx, dtype=torch.long)
    eos_col = torch.full((batch, 1), alphabet.eos_idx, dtype=torch.long)
    data_seq_raw = torch.cat([cls_col, body, eos_col], dim=1)

    # contact_masks: valid (i, j) contact region, (batch, 1, L, L)
    contact_masks = torch.ones(batch, 1, set_max_len, set_max_len)

    # data_seq_encoding: one-hot-ish sequence encoding, (batch, L, 4)
    data_seq_encoding = F.one_hot(torch.randint(0, 4, (batch, set_max_len)), num_classes=4).float()

    return (x_0, data_fcn_2, data_seq_raw, contact_masks, set_max_len, data_seq_encoding)


def _forward_rnadifffold(model, inputs):
    x_0, data_fcn_2, data_seq_raw, contact_masks, set_max_len, data_seq_encoding = inputs
    return model(x_0, data_fcn_2, data_seq_raw, contact_masks, set_max_len, data_seq_encoding)


MENAGERIE_ENTRIES = [
    ("RNADiffFold", "build_rnadifffold", "example_input_rnadifffold", 2024, "vendored-pytorch"),
]
