# SOURCE: vendored from HwwAncient/Pytorch-PLATO @ main
# https://raw.githubusercontent.com/HwwAncient/Pytorch-PLATO/main/plato/models/unified_transformer.py
# https://raw.githubusercontent.com/HwwAncient/Pytorch-PLATO/main/plato/models/model_base.py
# https://raw.githubusercontent.com/HwwAncient/Pytorch-PLATO/main/plato/modules/embedder.py
# https://raw.githubusercontent.com/HwwAncient/Pytorch-PLATO/main/plato/modules/feedforward.py
# https://raw.githubusercontent.com/HwwAncient/Pytorch-PLATO/main/plato/modules/functions.py
# https://raw.githubusercontent.com/HwwAncient/Pytorch-PLATO/main/plato/modules/multihead_attention.py
# https://raw.githubusercontent.com/HwwAncient/Pytorch-PLATO/main/plato/modules/transformer_block.py
#
# Bao et al. 2020 (ACL) "PLATO-2: Towards Building an Open-Domain Chatbot via
# Curriculum Learning" -- unified pre-training transformer with a discrete
# latent-variable coarse-grained generation model (Stage 1) refined by an
# evaluation/discriminator network (Stage 2). The official repo
# (PaddlePaddle/Knover) is PaddlePaddle-only; this vendors HwwAncient's
# community PyTorch reimplementation of the PLATO/PLATO-2 `UnifiedTransformer`
# architecture (`plato/models/unified_transformer.py` + its supporting
# `plato/modules/*` layers), which is a torch/numpy-only 1:1 port with no
# PaddlePaddle dependency at runtime.
#
# `UnifiedTransformer` is the model exactly as defined upstream: a shared
# transformer stack (`layers`: `TransformerBlock` -> `MultiheadAttention` +
# `FeedForward`) reused across three sub-networks driven by attention masking
# -- (1) `_posteriori_network` recognizes a discrete latent `z` from the full
# context+response via a special `[M]` mask-embedding prepended token,
# (2) `_generation_network` decodes the response conditioned on the sampled
# latent embedding using a joined bidirectional-context / autoregressive-
# response attention mask (`_create_mask` + `_join_mask`), and (3), when
# `use_discriminator=True`, `_discriminator_network` scores real vs.
# in-batch-shuffled (context, response) pairs for the PLATO discriminator
# loss. Latent selection at training time uses Gumbel-softmax
# (`plato/modules/functions.gumbel_softmax`-equivalent branch in
# `_forward`, `F.gumbel_softmax`); training-mode `_forward` is the traced
# entry point below (this exercises all three sub-networks + the BoW loss
# head), matching the real forward path of the model with
# `use_discriminator=True`.
#
# No architectural changes were made; only mechanical fixes for import
# isolation and tracing:
#   - Module paths `plato.args`, `plato.modules.embedder`,
#     `plato.models.model_base`, `plato.modules.transformer_block`,
#     `plato.modules.feedforward`, `plato.modules.multihead_attention`,
#     `plato.modules.functions` are flattened into this single file (their
#     contents are otherwise unchanged; `functions.py`'s standalone
#     `gumbel_softmax`/`equal`/`not_equal` helpers are vendored too since
#     `unified_transformer.py` imports the module as `F_alias`, even though
#     only `F_alias.unsqueeze` is actually called on the traced path).
#   - `UnifiedTransformer.add_cmdline_argument` uses `argparse`-only
#     `str2bool` from `plato/args.py`; that helper is vendored verbatim
#     alongside it (unused on the traced forward path, kept for fidelity).
#   - `ModelBase.__init__` reads `hparams.init_checkpoint`/`use_gpu`/`fp16`;
#     `build_plato2()` below supplies a `SimpleNamespace` hparams object
#     with `use_gpu=False`, `fp16=False`, matching the CPU-eager code path
#     already supported upstream (`if self.use_gpu: self.cuda()` is simply
#     not taken) -- not a code change.
#   - `_create_parameters`'s `self.sequence_mask` buffer construction via
#     `np.tri` + `torch.tensor` is unchanged; it is a plain (non-persistent,
#     non-`nn.Parameter`) attribute in the original code and is left that
#     way here.
#   - The real `_forward` returns a metrics-free `outputs` dict (not a
#     tensor); `PlatoForward` below is a *thin* `nn.Module` wrapper (no new
#     computation) that calls `model._forward(inputs, is_training=True)`
#     and returns `outputs["dec_probs"]` (the decoder token-probability
#     tensor) so the model has an ordinary tensor-in/tensor-out forward
#     for tracing -- this only selects which already-computed output to
#     return, it does not alter any layer.

import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# plato/args.py (only the piece unified_transformer.py imports)
# ---------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    elif v in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Unsupported value encountered.")


# ---------------------------------------------------------------------------
# plato/modules/functions.py
# ---------------------------------------------------------------------------
def unsqueeze(input, dims):
    """Implement multi-dimension unsqueeze function."""
    if isinstance(dims, (list, tuple)):
        dims = [dim if dim >= 0 else dim + len(input.shape) + 1 for dim in dims]
        dims = sorted(dims, reverse=True)
        shape = list(input.shape)
        for dim in dims:
            shape.insert(dim, 1)
        return torch.reshape(input, shape)
    elif isinstance(dims, int):
        return input.unsqueeze(dims)
    else:
        raise ValueError("Warning: type(dims) must in (list, tuple, int)!")


def gumbel_softmax(input, tau=1, eps=1e-10):
    """Basic implement of gumbel_softmax."""
    U = torch.tensor(np.random.rand(*input.shape))
    gumbel = 0.0 - torch.log(eps - torch.log(U + eps))
    y = input + gumbel
    return F.softmax(y / tau)


def equal(x, y, dtype=None):
    """Implement equal in dygraph mode. (paddle)"""
    if dtype is None:
        dtype = "float32"
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    out = np.equal(x, y).astype(dtype)
    return torch.tensor(out)


def not_equal(x, y, dtype=None):
    """Implement not_equal in dygraph mode. (paddle)"""
    return 1 - equal(x, y, dtype)


# ---------------------------------------------------------------------------
# plato/modules/multihead_attention.py
# ---------------------------------------------------------------------------
class MultiheadAttention(nn.Module):
    """
    Multi head attention layer.
    """

    def __init__(self, hidden_dim, num_heads, dropout):
        assert hidden_dim % num_heads == 0
        super(MultiheadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.linear_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        return

    def _split_heads(self, x, is_key=False):
        x = x.reshape(x.size(0), x.size(1), self.num_heads, self.head_dim)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)
        return x

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), self.hidden_dim)
        return x

    def _attn(self, query, key, value, mask):
        # shape: [batch_size, num_head, seq_len, seq_len]
        scores = torch.matmul(query, key)
        scores = scores * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, self.num_heads, 1, 1)
            scores.masked_fill_(mask.bool(), -1e10)  # scores = (1 - mask) * scores + mask * (-1e10)

        attn = self.softmax(scores)
        attn = self.dropout_layer(attn)

        if mask is not None:
            attn.masked_fill_(mask.bool(), 0.0)  # attn = (1 - mask) * attn

        out = torch.matmul(attn, value)
        return out

    def forward(self, inp, mask=None, cache=None):
        """Forward process of self attention."""
        # shape: [batch_size, seq_len, 3 * hidden_dim]
        qkv = self.linear_qkv(inp)
        query, key, value = torch.split(qkv, self.hidden_dim, dim=2)

        # shape: [batch_size, num_head, seq_len, head_dim]
        query = self._split_heads(query)
        # shape: [batch_size, num_head, head_dim, seq_len]
        key = self._split_heads(key, is_key=True)
        # shape: [batch_size, num_head, seq_len, head_dim]
        value = self._split_heads(value)

        if cache is not None:
            if "key" in cache and "value" in cache:
                key = torch.cat([cache["key"], key], dim=3)
                value = torch.cat([cache["value"], value], dim=2)
            cache["key"] = key
            cache["value"] = value

        out = self._attn(query, key, value, mask)
        out = self._merge_heads(out)
        out = self.linear_out(out)
        return out


# ---------------------------------------------------------------------------
# plato/modules/feedforward.py
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """
    Positional feed forward layer.
    """

    def __init__(self, hidden_dim, inner_dim, dropout):
        super(FeedForward, self).__init__()

        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.linear_hidden = nn.Sequential(nn.Linear(hidden_dim, inner_dim), nn.GELU())
        self.linear_out = nn.Linear(inner_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        return

    def forward(self, x):
        out = self.linear_hidden(x)
        out = self.dropout_layer(out)
        out = self.linear_out(out)
        return out


# ---------------------------------------------------------------------------
# plato/modules/transformer_block.py
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Transformer block module.
    """

    def __init__(self, hidden_dim, num_heads, dropout, attn_dropout, ff_dropout):
        super(TransformerBlock, self).__init__()

        self.attn = MultiheadAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=attn_dropout
        )
        self.attn_norm = nn.LayerNorm(
            normalized_shape=hidden_dim, eps=1e-12, elementwise_affine=True
        )
        self.ff = FeedForward(hidden_dim=hidden_dim, inner_dim=4 * hidden_dim, dropout=ff_dropout)
        self.ff_norm = nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-12, elementwise_affine=True)
        self.dropout_layer = nn.Dropout(p=dropout)
        return

    def forward(self, inp, mask=None, cache=None):
        attn_out = self.attn(inp, mask, cache)
        attn_out = self.dropout_layer(attn_out)
        attn_out = self.attn_norm(attn_out + inp)

        ff_out = self.ff(attn_out)
        ff_out = self.dropout_layer(ff_out)
        ff_out = self.ff_norm(ff_out + attn_out)

        return ff_out


# ---------------------------------------------------------------------------
# plato/modules/embedder.py
# ---------------------------------------------------------------------------
class Embedder(nn.Module):
    """
    Composite embedding layer.
    """

    def __init__(
        self,
        hidden_dim,
        num_token_embeddings,
        num_pos_embeddings,
        num_type_embeddings,
        num_turn_embeddings,
        padding_idx=None,
        dropout=0.1,
        pos_trainable=False,
    ):
        super(Embedder, self).__init__()

        self.token_embedding = nn.Embedding(num_token_embeddings, hidden_dim)
        self.pos_embedding = nn.Embedding(num_pos_embeddings, hidden_dim)
        self.pos_embedding.weight.requires_grad = pos_trainable
        self.type_embedding = nn.Embedding(num_type_embeddings, hidden_dim)
        self.turn_embedding = nn.Embedding(num_turn_embeddings, hidden_dim)
        self.dropout_layer = nn.Dropout(p=dropout)

        # follow the default xavier_uniform initializer in paddle version
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        nn.init.xavier_uniform_(self.turn_embedding.weight)
        return

    def forward(self, token_inp, pos_inp, type_inp, turn_inp):
        embed = (
            self.token_embedding(token_inp)
            + self.pos_embedding(pos_inp)
            + self.type_embedding(type_inp)
            + self.turn_embedding(turn_inp)
        )
        embed = self.dropout_layer(embed)
        return embed


# ---------------------------------------------------------------------------
# plato/models/model_base.py
# ---------------------------------------------------------------------------
class ModelBase(nn.Module):
    """
    Basic model wrapper for static graph and dygraph.
    """

    _registry = dict()

    @classmethod
    def register(cls, name):
        ModelBase._registry[name] = cls
        return

    @staticmethod
    def by_name(name):
        return ModelBase._registry[name]

    @staticmethod
    def create(hparams, *args, **kwargs):
        model_cls = ModelBase.by_name(hparams.model)
        return model_cls(hparams, *args, **kwargs)

    def __init__(self, hparams):
        super(ModelBase, self).__init__()
        self.init_checkpoint = hparams.init_checkpoint
        self.use_gpu = hparams.use_gpu
        self.fp16 = hparams.fp16
        return

    def _create_parameters(self):
        raise NotImplementedError

    def _forward(self, inputs, is_training):
        raise NotImplementedError

    def _collect_metrics(self, inputs, outputs):
        raise NotImplementedError

    def _optimize(self, loss):
        raise NotImplementedError

    def _infer(self, inputs):
        raise NotImplementedError

    def forward(self, inputs, is_training=False):
        if is_training:
            self.train()
        else:
            self.eval()

        outputs = self._forward(inputs, is_training)
        metrics = self._collect_metrics(inputs, outputs)
        loss = metrics["loss"]
        if is_training:
            self._optimize(loss)

        metrics = {k: v.cpu().detach().numpy() for k, v in metrics.items()}
        return metrics

    def infer(self, inputs):
        self.eval()
        results = self._infer(inputs)
        results = {name: results[name].cpu().detach().numpy() for name in results}
        return results


# ---------------------------------------------------------------------------
# plato/models/unified_transformer.py
# ---------------------------------------------------------------------------
class UnifiedTransformer(ModelBase):
    """
    Implement unified transformer.
    """

    def __init__(self, hparams, generator=None, dtype="float32"):
        super(UnifiedTransformer, self).__init__(hparams)
        self.generator = generator
        self.num_token_embeddings = hparams.num_token_embeddings
        self.num_pos_embeddings = hparams.num_pos_embeddings
        self.num_type_embeddings = hparams.num_type_embeddings
        self.num_turn_embeddings = hparams.num_turn_embeddings
        self.num_latent = hparams.num_latent
        self.tau = hparams.tau
        self.with_bow = hparams.with_bow
        self.hidden_dim = hparams.hidden_dim
        self.num_heads = hparams.num_heads
        self.num_layers = hparams.num_layers
        self.padding_idx = hparams.padding_idx
        self.dropout = hparams.dropout
        self.embed_dropout = hparams.embed_dropout
        self.attn_dropout = hparams.attn_dropout
        self.ff_dropout = hparams.ff_dropout
        self.use_discriminator = hparams.use_discriminator
        self.weight_sharing = hparams.weight_sharing
        self.pos_trainable = hparams.pos_trainable
        self.two_layer_predictor = hparams.two_layer_predictor
        self.bidirectional_context = hparams.bidirectional_context
        self.label_smooth = hparams.label_smooth
        self.initializer_range = hparams.initializer_range
        self._dtype = dtype

        self.embedder = Embedder(
            self.hidden_dim,
            self.num_token_embeddings,
            self.num_pos_embeddings,
            self.num_type_embeddings,
            self.num_turn_embeddings,
            padding_idx=self.padding_idx,
            dropout=self.embed_dropout,
            pos_trainable=self.pos_trainable,
        )
        self.embed_layer_norm = nn.LayerNorm(
            normalized_shape=self.hidden_dim, eps=1e-12, elementwise_affine=True
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    self.hidden_dim,
                    self.num_heads,
                    self.dropout,
                    self.attn_dropout,
                    self.ff_dropout,
                )
                for _ in range(hparams.num_layers)
            ]
        )

        if self.num_latent > 0:
            self.post_network = nn.Linear(self.hidden_dim, self.num_latent, bias=False)

            if self.use_discriminator:
                self.dis_ratio = hparams.dis_ratio
                self.discriminator = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())

        if self.two_layer_predictor:
            self.pre_predictor = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU()
            )
            if self.num_latent > 0 and self.with_bow:
                self.pre_bow_predictor = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU()
                )
        if not self.weight_sharing:
            self.predictor = nn.Linear(self.hidden_dim, self.num_token_embeddings, bias=False)
        if self.num_latent > 0 and self.with_bow:
            self.bow_predictor = nn.Linear(self.hidden_dim, self.num_token_embeddings, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self._create_parameters()

        self.nll_loss = nn.NLLLoss(ignore_index=self.padding_idx, reduction="none")

        self.max_grad_norm = hparams.max_grad_norm
        if self.max_grad_norm is not None:
            self.grad_clip = self.max_grad_norm
        else:
            self.grad_clip = None
        self.weight_decay = hparams.weight_decay
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=hparams.lr,
            weight_decay=self.weight_decay,
        )

        if self.use_gpu:
            self.cuda()

        return

    def _create_parameters(self):
        """Create model's paramters."""
        if self.num_latent > 0:
            self.mask_embed = nn.Parameter(torch.Tensor(1, 1, self.hidden_dim))
            self.latent_embeddings = nn.Parameter(torch.Tensor(self.num_latent, self.hidden_dim))
            nn.init.normal_(self.mask_embed, std=self.initializer_range)
            nn.init.normal_(self.latent_embeddings, std=self.initializer_range)

        sequence_mask = np.tri(self.num_pos_embeddings, self.num_pos_embeddings, dtype=self._dtype)
        self.sequence_mask = torch.tensor(sequence_mask)
        if self.use_gpu:
            self.sequence_mask = self.sequence_mask.cuda()
        return

    def _create_mask(self, input_mask, append_head=False, auto_regressive=False):
        seq_len = input_mask.shape[1]

        input_mask = input_mask.float()
        mask1 = input_mask.unsqueeze(-1).repeat(1, 1, seq_len)
        mask2 = mask1.permute(0, 2, 1)
        mask = mask1 * mask2

        if append_head:
            mask = torch.cat([mask[:, :1, :], mask], dim=1)
            mask = torch.cat([mask[:, :, :1], mask], dim=2)
            seq_len += 1

        if auto_regressive:
            seq_mask = self.sequence_mask[:seq_len, :seq_len]
            mask = mask * seq_mask

        mask = 1 - mask
        return mask

    def _join_mask(self, mask1, mask2):
        batch_size = mask1.shape[0]
        seq_len1 = mask1.shape[1]
        seq_len2 = mask2.shape[1]

        mask_lu = mask1
        mask_ru = torch.ones(batch_size, seq_len1, seq_len2)
        if self.use_gpu:
            mask_ru = mask_ru.cuda()
        mask3 = mask2[:, :, :1].repeat(1, 1, seq_len1)
        mask4 = mask1[:, :1].repeat(1, seq_len2, 1)
        mask_lb = mask3 + mask4 - mask3 * mask4
        mask_rb = mask2
        mask_u = torch.cat([mask_lu, mask_ru], dim=2)
        mask_b = torch.cat([mask_lb, mask_rb], dim=2)
        mask = torch.cat([mask_u, mask_b], dim=1)
        return mask

    def _posteriori_network(self, input_mask, embed, batch_size, src_len, tgt_len):
        """Basic posteriori network implement."""
        mask_embed = self.mask_embed
        mask_embed = mask_embed.repeat(batch_size, 1, 1)
        mask_embed = self.embed_layer_norm(mask_embed)
        post_embed = torch.cat([mask_embed, embed], dim=1)

        mask = self._create_mask(
            input_mask, auto_regressive=not self.bidirectional_context, append_head=True
        )

        for layer in self.layers:
            post_embed = layer(post_embed, mask, None)

        post_embed = post_embed[:, 0]
        post_logits = self.post_network(post_embed)
        post_probs = self.softmax(post_logits)
        post_logits = torch.log(post_probs)
        return post_embed, post_probs, post_logits

    def _discriminator_network(self, input_mask, embed, batch_size, src_len, tgt_len, pos_embed):
        """Basic discriminator network implement."""
        src_embed = embed[:, :src_len]
        tgt_embed = embed[:, src_len:]
        if batch_size > 1:
            neg_tgt_embed = torch.cat([tgt_embed[1:], tgt_embed[:1]], dim=0)
        else:
            neg_tgt_embed = tgt_embed
        neg_embed = torch.cat([src_embed, neg_tgt_embed], dim=1)

        src_mask = input_mask[:, :src_len]
        tgt_mask = input_mask[:, src_len:]
        if batch_size > 1:
            neg_tgt_mask = torch.cat([tgt_mask[1:], tgt_mask[:1]], dim=0)
        else:
            neg_tgt_mask = tgt_mask
        neg_mask = torch.cat([src_mask, neg_tgt_mask], dim=1)
        mask = self._create_mask(
            neg_mask, auto_regressive=not self.bidirectional_context, append_head=True
        )

        mask_embed = self.mask_embed
        mask_embed = mask_embed.repeat(batch_size, 1, 1)
        mask_embed = self.embed_layer_norm(mask_embed)
        neg_embed = torch.cat([mask_embed, neg_embed], dim=1)

        for layer in self.layers:
            neg_embed = layer(neg_embed, mask, None)

        neg_embed = neg_embed[:, 0]

        pos_probs = self.discriminator(pos_embed)
        neg_probs = self.discriminator(neg_embed)

        return pos_probs, neg_probs

    def _generation_network(self, input_mask, embed, batch_size, src_len, tgt_len, latent_embed):
        """Basic generation network implement."""
        if self.num_latent > 0:
            latent_embed = latent_embed.unsqueeze(1)
            latent_embed = self.embed_layer_norm(latent_embed)
            dec_embed = torch.cat([latent_embed, embed], dim=1)
        else:
            dec_embed = embed

        src_mask = input_mask[:, :src_len]
        tgt_mask = input_mask[:, src_len:]
        enc_mask = self._create_mask(
            src_mask,
            auto_regressive=not self.bidirectional_context,
            append_head=self.num_latent > 0,
        )
        dec_mask = self._create_mask(tgt_mask, auto_regressive=True)
        mask = self._join_mask(enc_mask, dec_mask)

        for layer in self.layers:
            dec_embed = layer(dec_embed, mask, None)

        if self.num_latent > 0:
            latent_embed = dec_embed[:, 0]
        else:
            latent_embed = None
        dec_embed = dec_embed[:, -tgt_len:]
        if self.two_layer_predictor:
            dec_embed = self.pre_predictor(dec_embed)
        if self.weight_sharing:
            token_embedding = self.embedder.token_embedding.weight
            dec_logits = torch.matmul(dec_embed, token_embedding.T)
        else:
            dec_logits = self.predictor(dec_embed)

        dec_probs = self.softmax(dec_logits)

        return latent_embed, dec_probs

    def _forward(self, inputs, is_training):
        """Real forward process of model in different mode(train/test)."""
        outputs = {}

        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]
        src_type = inputs["src_type"]
        src_turn = inputs["src_turn"]

        tgt_token = inputs["tgt_token"][:, :-1]
        tgt_mask = inputs["tgt_mask"][:, :-1]
        tgt_pos = inputs["tgt_pos"][:, :-1]
        tgt_type = inputs["tgt_type"][:, :-1]
        tgt_turn = inputs["tgt_turn"][:, :-1]

        input_mask = torch.cat([src_mask, tgt_mask], dim=1)
        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        tgt_embed = self.embedder(tgt_token, tgt_pos, tgt_type, tgt_turn)
        embed = torch.cat([src_embed, tgt_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        batch_size = src_token.shape[0]
        src_len = src_token.shape[1]
        tgt_len = tgt_token.shape[1]

        if self.num_latent > 0:
            post_embed, post_probs, post_logits = self._posteriori_network(
                input_mask, embed, batch_size, src_len, tgt_len
            )
            outputs["post_logits"] = post_logits

            if self.use_discriminator:
                pos_probs, neg_probs = self._discriminator_network(
                    input_mask, embed, batch_size, src_len, tgt_len, post_embed
                )
                outputs["pos_probs"] = pos_probs
                outputs["neg_probs"] = neg_probs

            if is_training:
                z = F.gumbel_softmax(logits=post_logits, tau=self.tau)
            else:
                indices = torch.argmax(post_logits, dim=1)
                z = F.one_hot(indices, num_classes=self.num_latent).float()
            latent_embeddings = self.latent_embeddings
            latent_embed = torch.matmul(z, latent_embeddings)
            outputs["latent_embed"] = latent_embed
        else:
            latent_embed = None

        latent_embed, dec_probs = self._generation_network(
            input_mask, embed, batch_size, src_len, tgt_len, latent_embed
        )
        outputs["dec_probs"] = dec_probs

        if self.num_latent > 0 and self.with_bow:
            if self.two_layer_predictor:
                latent_embed = self.pre_bow_predictor(latent_embed)
            bow_logits = self.bow_predictor(latent_embed)
            bow_probs = self.softmax(bow_logits)
            outputs["bow_probs"] = bow_probs

        return outputs

    def _collect_metrics(self, inputs, outputs):
        """Calculate loss function by using inputs and outputs."""
        metrics = {}

        tgt_len = torch.sum(torch.sum(inputs["tgt_mask"], dim=1) - 1)

        label = inputs["tgt_token"][:, 1:]
        nll = self.nll_loss(torch.log(outputs["dec_probs"] + 1e-12).permute(0, 2, 1), label)
        nll = torch.sum(nll, dim=1)
        token_nll = torch.sum(nll) / tgt_len
        nll = torch.mean(nll)
        metrics["nll"] = nll
        metrics["token_nll"] = token_nll
        loss = nll

        if self.num_latent > 0 and self.with_bow:
            bow_probs = outputs["bow_probs"].unsqueeze(1)
            bow_probs = bow_probs.repeat(1, label.shape[1], 1)
            bow = self.nll_loss(torch.log(bow_probs + 1e-12).permute(0, 2, 1), label)
            bow = torch.sum(bow, dim=1)
            token_bow = torch.sum(bow) / tgt_len
            bow = torch.mean(bow)
            metrics["bow"] = bow
            metrics["token_bow"] = token_bow
            loss = loss + bow

        if self.num_latent > 0 and self.use_discriminator:
            dis = 0.0 - (torch.log(outputs["pos_probs"]) + torch.log(1.0 - outputs["neg_probs"]))
            dis = torch.mean(dis)
            metrics["dis"] = dis
            loss = loss + dis * self.dis_ratio

        metrics["loss"] = loss
        metrics["token_num"] = tgt_len
        return metrics


UnifiedTransformer.register("UnifiedTransformer")


# ---------------------------------------------------------------------------
# Tracing wrapper (thin: selects an output tensor from the real forward, no
# new computation) + menagerie build/example functions.
# ---------------------------------------------------------------------------
class PlatoForward(nn.Module):
    """Thin tracing wrapper around UnifiedTransformer._forward (training mode)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        src_token,
        src_mask,
        src_pos,
        src_type,
        src_turn,
        tgt_token,
        tgt_mask,
        tgt_pos,
        tgt_type,
        tgt_turn,
    ):
        inputs = {
            "src_token": src_token,
            "src_mask": src_mask,
            "src_pos": src_pos,
            "src_type": src_type,
            "src_turn": src_turn,
            "tgt_token": tgt_token,
            "tgt_mask": tgt_mask,
            "tgt_pos": tgt_pos,
            "tgt_type": tgt_type,
            "tgt_turn": tgt_turn,
        }
        outputs = self.model._forward(inputs, is_training=True)
        return outputs["dec_probs"], outputs["bow_probs"]


def build_plato2():
    hparams = SimpleNamespace(
        num_token_embeddings=200,
        num_pos_embeddings=32,
        num_type_embeddings=2,
        num_turn_embeddings=8,
        num_latent=10,
        tau=0.67,
        with_bow=True,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        padding_idx=0,
        dropout=0.0,
        embed_dropout=0.0,
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_discriminator=True,
        dis_ratio=1.0,
        weight_sharing=True,
        pos_trainable=True,
        two_layer_predictor=False,
        bidirectional_context=True,
        label_smooth=0.0,
        initializer_range=0.02,
        lr=5e-5,
        weight_decay=0.0,
        max_grad_norm=None,
        init_checkpoint=None,
        use_gpu=False,
        fp16=False,
    )
    model = UnifiedTransformer(hparams, generator=None)
    model.eval()
    return PlatoForward(model)


def example_input_plato2():
    batch = 2
    src_len = 6
    tgt_len = 5  # tgt is sliced [:, :-1] inside _forward -> effective len 4

    src_token = torch.randint(1, 200, (batch, src_len))
    src_mask = torch.ones(batch, src_len)
    src_pos = torch.arange(src_len).unsqueeze(0).repeat(batch, 1)
    src_type = torch.zeros(batch, src_len, dtype=torch.int64)
    src_turn = torch.zeros(batch, src_len, dtype=torch.int64)

    tgt_token = torch.randint(1, 200, (batch, tgt_len))
    tgt_mask = torch.ones(batch, tgt_len)
    tgt_pos = torch.arange(tgt_len).unsqueeze(0).repeat(batch, 1)
    tgt_type = torch.ones(batch, tgt_len, dtype=torch.int64)
    tgt_turn = torch.zeros(batch, tgt_len, dtype=torch.int64)

    return (
        src_token,
        src_mask,
        src_pos,
        src_type,
        src_turn,
        tgt_token,
        tgt_mask,
        tgt_pos,
        tgt_type,
        tgt_turn,
    )


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("PLATO-2", "build_plato2", "example_input_plato2", 2020, "vendored"),
]
