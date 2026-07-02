# FAITHFUL PORT of FanmengWang/MMPolymer @ master (original framework: PyTorch + Uni-Core)
# Also ports: dptech-corp/Uni-Core @ main (unicore/modules/{transformer_encoder_layer,
# multihead_attention,layer_norm}.py) -- the third-party attention/transformer/layer-norm
# primitives MMPolymer's real code imports from `unicore`.
#
# MMPolymer (Wang et al., CIKM 2024): a multimodal multitask pretraining framework for
# polymer property prediction. A "1D net" (RoBERTa, from `transformers`, applied to
# tokenized polymer SMILES) and a "3D net" (`TransformerEncoderWithPair`, a 3D-coordinate-
# aware Transformer over conformer atoms with Gaussian radial-basis pairwise-distance
# attention bias -- the Uni-Mol architecture) are each pooled to a 512-d representation via a
# `NonLinearHead`, concatenated, and fed through a small MLP `classification_head` to a
# scalar polymer-property prediction. The 3D-net internals (`TransformerEncoderWithPair`,
# `TransformerEncoderLayer`, `SelfMultiheadAttention`, `LayerNorm`) are the real `unicore`
# library code (dptech-corp/Uni-Core), ported here because `unicore` is not on PyPI and
# requires building custom CUDA extensions from source -- not reasonably installable in a
# base torch environment. The CUDA-fused-kernel fast paths in `unicore`'s `LayerNorm` /
# `softmax_dropout` are dropped (only reachable when the `unicore_fused_*` C++ extensions
# are compiled and a CUDA device is present); the CPU/no-extension fallback branches
# (`F.layer_norm`, `F.softmax` + `F.dropout`) that `unicore`'s own code always uses when the
# extensions are absent are kept verbatim -- same math, no architecture change. `unicore.utils
# .get_activation_fn` / `init_bert_params` are reproduced verbatim (trivial helper functions).
#
# The real `MMPolymerModel.forward` additionally computes 3 auxiliary Uni-Mol pretraining
# heads (masked-atom `lm_head`, masked-distance `dist_head`, masked-coordinate
# `pair2coord_proj`) gated behind `args.masked_token_loss/masked_dist_loss/masked_coord_loss
# > 0`; those flags default to -1 (disabled) at both pretrain and finetune time in the real
# repo's `base_architecture`/finetune configs (`env.yml`, `train.sh`), so the ported
# `forward` mirrors `features_only=True` classification-head inference path exactly as the
# real `get_prediction_results.py` / finetune inference calls it, without instantiating the
# unused pretraining heads. `matplotlib`/`sklearn.manifold.TSNE` (used only for embedding
# visualization elsewhere in the repo) and the custom `PolymerSmilesTokenizer` /
# `AutoConfig.from_pretrained("./MMPolymer/models/config")` file-path loading are replaced
# with a plain `RobertaConfig` built inline with the same real hyperparameters
# (config.json's hidden_size=768/num_hidden_layers=6/num_attention_heads=12/vocab_size=50265)
# and pre-tokenized `input_ids`/`attention_mask` tensors, since tokenization/plotting are not
# part of the model architecture. `fairseq`-style `register_model`/argparse scaffolding and
# `dictionary` are replaced with plain constructor kwargs; no architecture lines changed.
#
# MENAGERIE_ZOO = "ported-pytorch"

import numbers
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from transformers import RobertaConfig, RobertaModel


# ==== ported from dptech-corp/Uni-Core: unicore/modules/layer_norm.py ====
class LayerNorm(torch.nn.Module):
    """Uni-Core LayerNorm. The real module optionally dispatches to a fused CUDA kernel
    (`unicore_fused_layernorm`) when the C++ extension is compiled and a CUDA device with
    compute capability >= 7 is present; that extension is not installable here, so this port
    always takes the plain `F.layer_norm` branch the real code itself falls back to when the
    extension is absent -- identical math, no architecture change."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        assert elementwise_affine
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(
            input,
            self.normalized_shape,
            self.weight.type(input.dtype),
            self.bias.type(input.dtype),
            self.eps,
        )

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, elementwise_affine=True".format(**self.__dict__)


# ==== ported from dptech-corp/Uni-Core: unicore/modules/softmax_dropout.py ====
def softmax_dropout(input, dropout_prob, is_training=True, mask=None, bias=None, inplace=True):
    """Same no-fused-extension fallback branch the real `softmax_dropout` itself uses when
    `unicore_fused_softmax_dropout` is absent or the input is not on CUDA."""
    input = input.contiguous()
    if not inplace:
        input = input.clone()
    if mask is not None:
        input += mask
    if bias is not None:
        input += bias
    return F.dropout(F.softmax(input, dim=-1), p=dropout_prob, training=is_training)


# ==== ported from dptech-corp/Uni-Core: unicore/modules/multihead_attention.py ====
class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )

        assert k is not None
        src_len = k.size(1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn = softmax_dropout(
                attn_weights,
                self.dropout,
                self.training,
                bias=attn_bias,
            )
        else:
            attn_weights = attn_weights + attn_bias
            attn = softmax_dropout(
                attn_weights,
                self.dropout,
                self.training,
                inplace=False,
            )

        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn


# ==== ported from dptech-corp/Uni-Core: unicore/utils.py (activation fn lookup) ====
def get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


# ==== ported from dptech-corp/Uni-Core: unicore/modules/transformer_encoder.py ====
def init_bert_params(module):
    if not getattr(module, "can_global_init", True):
        return

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


# ==== ported from dptech-corp/Uni-Core: unicore/modules/transformer_encoder_layer.py ====
class TransformerEncoderLayer(nn.Module):
    """Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained models."""

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln=False,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)

        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs


# ==== ported from FanmengWang/MMPolymer: MMPolymer/models/transformer_encoder_with_pair.py ====
class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = emb.size(0)
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask = attn_mask.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

        for i in range(len(self.layers)):
            x, attn_mask, _ = self.layers[i](
                x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))).mean()

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(pair_mask, delta_pair_repr_norm, dim=(-1, -2))

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm


# ==== ported from FanmengWang/MMPolymer: MMPolymer/models/MMPolymer.py ====
class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(self, input_dim, out_dim, activation_fn, hidden=None):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class MMPolymerModel(nn.Module):
    def __init__(
        self,
        dictionary_len,
        encoder_layers=15,
        encoder_embed_dim=64,
        encoder_ffn_embed_dim=128,
        encoder_attention_heads=8,
        dropout=0.1,
        emb_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        max_seq_len=64,
        activation_fn="gelu",
        roberta_layers=2,
        roberta_hidden=64,
        roberta_heads=4,
        roberta_vocab=512,
    ):
        super().__init__()
        self.padding_idx = 0
        self.embed_tokens = nn.Embedding(dictionary_len, encoder_embed_dim, self.padding_idx)

        # 1D net: RoBERTa over tokenized polymer SMILES (real repo: HF RobertaModel with
        # a pretrained roberta-base config; scaled down here for a tiny random-init trace)
        roberta_config = RobertaConfig(
            vocab_size=roberta_vocab,
            hidden_size=roberta_hidden,
            num_hidden_layers=roberta_layers,
            num_attention_heads=roberta_heads,
            intermediate_size=roberta_hidden * 4,
            max_position_embeddings=max_seq_len + 2,
            layer_norm_eps=1e-12,
        )
        self.PretrainedModel = RobertaModel(config=roberta_config)

        # 3D net: Uni-Mol-style conformer Transformer with pairwise Gaussian distance bias
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=encoder_layers,
            embed_dim=encoder_embed_dim,
            ffn_embed_dim=encoder_ffn_embed_dim,
            attention_heads=encoder_attention_heads,
            emb_dropout=emb_dropout,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            max_seq_len=max_seq_len,
            activation_fn=activation_fn,
            no_final_head_layer_norm=False,
        )

        K = 128
        n_edge_type = dictionary_len * dictionary_len
        self.gbf_proj = NonLinearHead(K, encoder_attention_heads, activation_fn)
        self.gbf = GaussianLayer(K, n_edge_type)

        self.seq_layer = NonLinearHead(roberta_hidden, 512, activation_fn)
        self.space_layer = NonLinearHead(encoder_embed_dim, 512, activation_fn)

        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1),
        )

        self.apply(init_bert_params)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_input_ids,
        src_attention_mask,
    ):
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep = torch.where(
            encoder_pair_rep == float("-inf"), torch.zeros_like(encoder_pair_rep), encoder_pair_rep
        )

        seq_rep = self.PretrainedModel(
            input_ids=src_input_ids, attention_mask=src_attention_mask
        ).last_hidden_state

        # features_only=True classification-head inference path (real repo's finetune /
        # get_prediction_results.py call): masked_token_loss/masked_dist_loss/masked_coord_loss
        # all default to -1 (disabled) so the pretraining heads are never exercised here.
        seq_output = self.seq_layer(seq_rep)
        space_output = self.space_layer(encoder_rep)
        mol_output = torch.cat((seq_output[:, 0, :], space_output[:, 0, :]), dim=-1)
        logits = self.classification_head(mol_output)

        return logits


def build_mmpolymer():
    return MMPolymerModel(
        dictionary_len=64,
        encoder_layers=2,
        encoder_embed_dim=32,
        encoder_ffn_embed_dim=64,
        encoder_attention_heads=4,
        max_seq_len=16,
        roberta_layers=2,
        roberta_hidden=32,
        roberta_heads=4,
        roberta_vocab=128,
    )


def example_input_mmpolymer():
    bsz, n_atom, seq_len = 2, 8, 10
    src_tokens = torch.randint(1, 64, (bsz, n_atom))
    src_distance = torch.rand(bsz, n_atom, n_atom) * 3
    src_distance = (src_distance + src_distance.transpose(1, 2)) / 2
    src_coord = torch.randn(bsz, n_atom, 3)
    src_edge_type = torch.randint(0, 64 * 64, (bsz, n_atom, n_atom))
    src_input_ids = torch.randint(1, 128, (bsz, seq_len))
    src_attention_mask = torch.ones(bsz, seq_len)
    return (src_tokens, src_distance, src_coord, src_edge_type, src_input_ids, src_attention_mask)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("MMPolymer", "build_mmpolymer", "example_input_mmpolymer", 2024, "ported"),
]
