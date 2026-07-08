# SOURCE: vendored from siat-nlp/GALAXY @ main
# (galaxy/models/unified_transformer.py, galaxy/models/model_base.py,
#  galaxy/modules/embedder.py, galaxy/modules/transformer_block.py,
#  galaxy/modules/multihead_attention.py, galaxy/modules/feedforward.py)
#
# GALAXY (He, Dai, Zhang, Zhang, Huang, Si, Sun, Li, "GALAXY: A Generative
# Pre-trained Model for Task-Oriented Dialog with Semi-Supervised Learning
# and Explicit Policy Injection", AAAI 2022, official siat-nlp repo). Real
# architecture: `UnifiedTransformer` is a from-scratch (not
# HuggingFace-derived) UniLM-style encoder-decoder Transformer -- a shared
# stack of pre-LN-free post-LN `TransformerBlock`s (self-attention +
# GELU feedforward, each with its own LayerNorm) operating over a single
# concatenated [context-tokens ; response-tokens] sequence, with a custom
# `_create_mask`/`_join_mask` pair building the UniLM-style block-diagonal
# attention mask (bidirectional over the source/context segment,
# autoregressive/causal over the target/response segment via a triangular
# `sequence_mask`), and an `Embedder` that additively composes
# token+position+segment-type+dialogue-turn embeddings. GALAXY's own
# contribution -- explicit dialog-act policy injection via a `[MASK]` latent
# token pooled through `act_classifier`, semi-supervised consistency
# regularization (R-Drop KL term) and OOD filtering gates -- lives in the
# `PretrainUnifiedTransformer` subclass on top of this shared body. This is
# fully bespoke Transformer code (own attention/embedder/mask-construction),
# not a thin usage-only wrapper of a base-library model, so it is vendored
# (rung 2) rather than recipe'd.
#
# Vendoring notes (imports/portability fixes only, architecture untouched):
#   - The original `UnifiedTransformer`/`PretrainUnifiedTransformer` take a
#     `hparams` argparse.Namespace (with an `add_cmdline_argument` classmethod
#     hierarchy) and a `generator` object (`galaxy.models.generator.Generator`,
#     used only by `_infer`, GALAXY's beam-search decoding path -- not part of
#     the traced forward computation). Reproduced here as a tiny local
#     `_HParams` container built directly with the constants this staged
#     module needs, and `generator=None` (never touched by `_forward`/
#     `_encoder_decoder_network`/`_dec_head`, the traced path).
#   - `from galaxy.args import str2bool` / `from galaxy.utils.eval import
#     DAEvaluation` / `from galaxy.utils.criterions import compute_kl_loss`
#     are argparse-CLI and post-hoc numpy-metric helpers used only inside
#     `add_cmdline_argument`/`_collect_metrics` (training-loop bookkeeping,
#     not part of the traced network graph) -- dropped along with those
#     methods; the traced entry only exercises `_forward`'s network path
#     (`_mask_encoder_decoder_network`/`_encoder_decoder_network` +
#     `_dec_head`), which is copied verbatim.
#   - `PretrainUnifiedTransformer.__init__` calls
#     `super().__init__(hparams, generator)` then reassigns
#     `self.act_classifier` (a plain `nn.Linear`, no `nn.Sigmoid` wrapper,
#     unlike the base class's) when `with_joint_act=True` -- reproduced
#     verbatim; the `_compute_gates`/`_collect_metrics`/`_optimize` training-
#     loss/OOD-filter bookkeeping methods are dropped (loss/metric
#     computation, not part of the traced forward network) but `_forward`
#     (GALAXY's actual joint-act-conditioned encoder-decoder pass) is
#     copied verbatim, including the joint-act classifier head.
#   - `MultiheadAttention._attn`'s two in-place `masked_fill_` calls
#     replaced with the equivalent non-mutating `masked_fill` (avoids
#     autograd in-place-write hazards under capture instrumentation);
#     identical masking values/semantics, no architectural change.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- galaxy/modules/embedder.py (verbatim) ---
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

        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        nn.init.xavier_uniform_(self.turn_embedding.weight)
        return

    def forward(self, token_inp, pos_inp=None, type_inp=None, turn_inp=None):
        embed = self.token_embedding(token_inp)
        if pos_inp is not None:
            embed = embed + self.pos_embedding(pos_inp)
        if type_inp is not None:
            embed = embed + self.type_embedding(type_inp)
        if turn_inp is not None:
            embed = embed + self.turn_embedding(turn_inp)
        embed = self.dropout_layer(embed)
        return embed


# --- galaxy/modules/multihead_attention.py (verbatim) ---
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
            scores = scores.masked_fill(mask.bool(), float("-inf"))

        attn = self.softmax(scores)
        attn = self.dropout_layer(attn)

        if mask is not None:
            attn = attn.masked_fill(mask.bool(), 0.0)

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


# --- galaxy/modules/feedforward.py (verbatim) ---
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


# --- galaxy/modules/transformer_block.py (verbatim) ---
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
        """
        Forward process on one transformer layer.
        """
        attn_out = self.attn(inp, mask, cache)
        attn_out = self.dropout_layer(attn_out)
        attn_out = self.attn_norm(attn_out + inp)

        ff_out = self.ff(attn_out)
        ff_out = self.dropout_layer(ff_out)
        ff_out = self.ff_norm(ff_out + attn_out)

        return ff_out


# --- galaxy/models/model_base.py (verbatim) ---
class ModelBase(nn.Module):
    """
    Basic model wrapper.
    """

    _registry = dict()

    @classmethod
    def register(cls, name):
        ModelBase._registry[name] = cls
        return

    @staticmethod
    def by_name(name):
        return ModelBase._registry[name]

    def __init__(self, hparams):
        super(ModelBase, self).__init__()
        self.init_checkpoint = hparams.init_checkpoint
        self.with_rdrop_act = hparams.with_rdrop_act
        self.use_gpu = hparams.use_gpu
        return

    def _create_parameters(self):
        """Create model's paramters."""
        raise NotImplementedError

    def _forward(self, inputs, is_training):
        """Real forward process of model in different mode(train/test)."""
        raise NotImplementedError


# --- a tiny stand-in for GALAXY's argparse.Namespace hparams (data-only,
#     not part of the traced network graph); mirrors the defaults declared
#     in UnifiedTransformer.add_cmdline_argument / model_base.py. ---
class _HParams:
    def __init__(
        self,
        num_token_embeddings,
        num_pos_embeddings=64,
        num_type_embeddings=2,
        num_turn_embeddings=16,
        num_act=20,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
        with_joint_act=False,
    ):
        self.init_checkpoint = None
        self.with_rdrop_act = False
        self.use_gpu = False
        self.num_token_embeddings = num_token_embeddings
        self.num_pos_embeddings = num_pos_embeddings
        self.num_type_embeddings = num_type_embeddings
        self.num_turn_embeddings = num_turn_embeddings
        self.num_act = num_act
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.padding_idx = 0
        self.dropout = 0.1
        self.embed_dropout = 0.0
        self.attn_dropout = 0.1
        self.ff_dropout = 0.1
        self.pos_trainable = True
        self.initializer_range = 0.02
        self.use_discriminator = False
        self.gradient_accumulation_steps = 1
        self.with_joint_act = with_joint_act
        self.with_rdrop_act = False
        self.token_loss = True
        self.dis_ratio = 1.0
        self.bce_ratio = 1.0
        self.max_len = num_pos_embeddings
        self.gpu = 0
        self.max_grad_norm = 5.0
        self.weight_decay = 0.0


# --- galaxy/models/unified_transformer.py (verbatim network path; the
#     add_cmdline_argument/_optimize/_collect_metrics/_infer training-loop
#     and CLI-arg methods are dropped -- see header notes) ---
class UnifiedTransformer(ModelBase):
    """
    Implement unified transformer for generation.
    """

    def __init__(self, hparams, generator=None, dtype="float32"):
        super(UnifiedTransformer, self).__init__(hparams)
        self.generator = generator
        self.num_token_embeddings = hparams.num_token_embeddings
        self.num_pos_embeddings = hparams.num_pos_embeddings
        self.num_type_embeddings = hparams.num_type_embeddings
        self.num_turn_embeddings = hparams.num_turn_embeddings
        self.num_act = hparams.num_act
        self.hidden_dim = hparams.hidden_dim
        self.num_heads = hparams.num_heads
        self.num_layers = hparams.num_layers
        self.padding_idx = hparams.padding_idx
        self.dropout = hparams.dropout
        self.embed_dropout = hparams.embed_dropout
        self.attn_dropout = hparams.attn_dropout
        self.ff_dropout = hparams.ff_dropout
        self.pos_trainable = hparams.pos_trainable
        self.initializer_range = hparams.initializer_range
        self.use_discriminator = hparams.use_discriminator
        self.gradient_accumulation_steps = hparams.gradient_accumulation_steps
        self.with_joint_act = hparams.with_joint_act
        self.with_rdrop_act = hparams.with_rdrop_act
        self.token_loss = hparams.token_loss
        self.dis_ratio = hparams.dis_ratio
        self.bce_ratio = hparams.bce_ratio
        self.max_len = hparams.max_len
        self.gpu = hparams.gpu
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

        if self.with_joint_act:
            self.act_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.num_act), nn.Sigmoid()
            )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self._create_parameters()

        self.nll_loss = nn.NLLLoss(ignore_index=self.padding_idx, reduction="none")
        self.bce_loss = nn.BCELoss()

        self.max_grad_norm = hparams.max_grad_norm
        if self.max_grad_norm is not None:
            self.grad_clip = self.max_grad_norm
        else:
            self.grad_clip = None
        self.weight_decay = hparams.weight_decay

        if self.use_gpu:
            self.cuda()

        return

    def _create_parameters(self):
        """Create model's extra parameters."""
        if self.with_joint_act:
            self.mask_embed = nn.Parameter(torch.Tensor(1, 1, self.hidden_dim))
            nn.init.normal_(self.mask_embed, std=self.initializer_range)

        sequence_mask = np.tri(self.num_pos_embeddings, self.num_pos_embeddings, dtype=self._dtype)
        self.sequence_mask = torch.tensor(sequence_mask)
        return

    def _create_mask(self, input_mask, append_head=False, auto_regressive=False):
        """
        Create attention mask.
        """
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
            seq_mask = seq_mask.to(mask.device)
            mask = mask * seq_mask

        mask = 1 - mask
        return mask

    def _join_mask(self, mask1, mask2):
        """
        Merge source attention mask and target attention mask.
        """
        batch_size = mask1.shape[0]
        seq_len1 = mask1.shape[1]
        seq_len2 = mask2.shape[1]

        mask_lu = mask1
        mask_ru = torch.ones(batch_size, seq_len1, seq_len2)
        mask_ru = mask_ru.to(mask_lu.device)
        mask3 = mask2[:, :, :1].repeat(1, 1, seq_len1)
        mask4 = mask1[:, :1].repeat(1, seq_len2, 1)
        mask_lb = mask3 + mask4 - mask3 * mask4
        mask_rb = mask2
        mask_u = torch.cat([mask_lu, mask_ru], dim=2)
        mask_b = torch.cat([mask_lb, mask_rb], dim=2)
        mask = torch.cat([mask_u, mask_b], dim=1)
        return mask

    def _dec_head(self, dec_embed):
        """Decoding head for response generation task."""
        dec_logits = torch.matmul(dec_embed, self.embedder.token_embedding.weight.T)
        dec_probs = self.softmax(dec_logits)
        return dec_probs

    def _encoder_decoder_network(
        self,
        src_token,
        src_mask,
        tgt_token,
        tgt_mask,
        src_pos=None,
        src_type=None,
        src_turn=None,
        tgt_pos=None,
        tgt_type=None,
        tgt_turn=None,
    ):
        """Unified encoder-decoder network for both understanding and generation."""

        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        tgt_embed = self.embedder(tgt_token, tgt_pos, tgt_type, tgt_turn)
        embed = torch.cat([src_embed, tgt_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        enc_mask = self._create_mask(src_mask, auto_regressive=False)
        dec_mask = self._create_mask(tgt_mask, auto_regressive=True)
        mask = self._join_mask(enc_mask, dec_mask)

        for layer in self.layers:
            embed = layer(embed, mask, None)

        tgt_len = tgt_token.shape[1]
        enc_embed = embed[:, :-tgt_len]
        dec_embed = embed[:, -tgt_len:]

        return enc_embed, dec_embed

    def _mask_encoder_decoder_network(
        self,
        src_token,
        src_mask,
        tgt_token,
        tgt_mask,
        src_pos=None,
        src_type=None,
        src_turn=None,
        tgt_pos=None,
        tgt_type=None,
        tgt_turn=None,
    ):
        """Unified mask-encoder-decoder network for both understanding and generation."""

        mask_embed = self.mask_embed.repeat(src_token.shape[0], 1, 1)
        mask_embed = self.embed_layer_norm(mask_embed)
        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        tgt_embed = self.embedder(tgt_token, tgt_pos, tgt_type, tgt_turn)
        embed = torch.cat([src_embed, tgt_embed], dim=1)
        embed = self.embed_layer_norm(embed)
        embed = torch.cat([mask_embed, embed], dim=1)

        enc_mask = self._create_mask(src_mask, auto_regressive=False, append_head=True)
        dec_mask = self._create_mask(tgt_mask, auto_regressive=True, append_head=False)
        mask = self._join_mask(enc_mask, dec_mask)

        for layer in self.layers:
            embed = layer(embed, mask, None)

        tgt_len = tgt_token.shape[1]
        latent_embed = embed[:, 0]
        enc_embed = embed[:, 1:-tgt_len]
        dec_embed = embed[:, -tgt_len:]

        return latent_embed, enc_embed, dec_embed

    def _forward(self, inputs, is_training=False):
        """Real forward process of model."""

        outputs = {}

        if self.with_joint_act:
            latent_embed, enc_embed, dec_embed = self._mask_encoder_decoder_network(
                src_token=inputs["src_token"],
                src_mask=inputs["src_mask"],
                tgt_token=inputs["tgt_token"][:, :-1],
                tgt_mask=inputs["tgt_mask"][:, :-1],
                src_pos=inputs["src_pos"],
                src_type=inputs["src_type"],
                src_turn=inputs["src_turn"],
                tgt_pos=inputs["tgt_pos"][:, :-1],
                tgt_type=inputs["tgt_type"][:, :-1],
                tgt_turn=inputs["tgt_turn"][:, :-1],
            )
            joint_act_probs = self.act_classifier(latent_embed)
            outputs["joint_act_probs"] = joint_act_probs
        else:
            enc_embed, dec_embed = self._encoder_decoder_network(
                src_token=inputs["src_token"],
                src_mask=inputs["src_mask"],
                tgt_token=inputs["tgt_token"][:, :-1],
                tgt_mask=inputs["tgt_mask"][:, :-1],
                src_pos=inputs["src_pos"],
                src_type=inputs["src_type"],
                src_turn=inputs["src_turn"],
                tgt_pos=inputs["tgt_pos"][:, :-1],
                tgt_type=inputs["tgt_type"][:, :-1],
                tgt_turn=inputs["tgt_turn"][:, :-1],
            )

        outputs["dec_probs"] = self._dec_head(dec_embed=dec_embed)
        return outputs

    def forward(self, inputs, is_training=False):
        """
        Forward process, include real forward (metrics/loss collection dropped;
        see header notes -- not part of the traced network graph).
        """
        return self._forward(inputs, is_training)


# --- galaxy/models/pretrain_unified_transformer.py (verbatim network path;
#     add_cmdline_argument/_compute_gates/_collect_metrics/_optimize
#     training-loop/CLI-arg methods are dropped -- see header notes) ---
class PretrainUnifiedTransformer(UnifiedTransformer):
    """
    Implement unified transformer for pre-training.
    """

    def __init__(self, hparams, generator=None):
        super(PretrainUnifiedTransformer, self).__init__(hparams, generator)
        self.with_filter = getattr(hparams, "with_filter", False)
        self.detach_filter = getattr(hparams, "detach_filter", True)
        self.filter_index = getattr(hparams, "filter_index", 1)
        self.kl_ratio = getattr(hparams, "kl_ratio", 1.0)

        if self.with_joint_act:
            self.act_classifier = nn.Linear(self.hidden_dim, self.num_act)
        return

    def _forward(self, inputs, is_training=False):
        """Real forward process of model."""
        outputs = {}

        latent_embed, enc_embed, dec_embed = self._mask_encoder_decoder_network(
            src_token=inputs["src_token"],
            src_mask=inputs["src_mask"],
            tgt_token=inputs["tgt_token"][:, :-1],
            tgt_mask=inputs["tgt_mask"][:, :-1],
            src_pos=inputs["src_pos"],
            src_type=inputs["src_type"],
            src_turn=inputs["src_turn"],
            tgt_pos=inputs["tgt_pos"][:, :-1],
            tgt_type=inputs["tgt_type"][:, :-1],
            tgt_turn=inputs["tgt_turn"][:, :-1],
        )

        if self.with_joint_act:
            joint_act_logits = self.act_classifier(latent_embed)
            outputs["joint_act_logits"] = joint_act_logits
            joint_act_probs = self.sigmoid(joint_act_logits)
            outputs["joint_act_probs"] = joint_act_probs

        outputs["dec_probs"] = self._dec_head(dec_embed=dec_embed)
        return outputs


_VOCAB = 200
_SEQ = 12
_TGT_SEQ = 8


class GalaxyWrapper(nn.Module):
    """Wraps GALAXY's dict-in `forward` into a single-tensor-friendly call so
    torchlens can trace it directly (UnifiedTransformer/PretrainUnifiedTransformer's
    real computation is unchanged)."""

    def __init__(self, with_joint_act):
        super().__init__()
        hparams = _HParams(
            num_token_embeddings=_VOCAB, num_pos_embeddings=32, with_joint_act=with_joint_act
        )
        self.model = PretrainUnifiedTransformer(hparams)

    def forward(self, dummy):
        torch.manual_seed(0)
        src_token = torch.randint(1, _VOCAB, (1, _SEQ))
        src_mask = torch.ones(1, _SEQ)
        tgt_token = torch.randint(1, _VOCAB, (1, _TGT_SEQ))
        tgt_mask = torch.ones(1, _TGT_SEQ)
        src_pos = torch.arange(_SEQ).unsqueeze(0)
        src_type = torch.zeros(1, _SEQ, dtype=torch.long)
        src_turn = torch.zeros(1, _SEQ, dtype=torch.long)
        tgt_pos = torch.arange(_TGT_SEQ).unsqueeze(0)
        tgt_type = torch.ones(1, _TGT_SEQ, dtype=torch.long)
        tgt_turn = torch.zeros(1, _TGT_SEQ, dtype=torch.long)

        inputs = {
            "src_token": src_token,
            "src_mask": src_mask,
            "tgt_token": tgt_token,
            "tgt_mask": tgt_mask,
            "src_pos": src_pos,
            "src_type": src_type,
            "src_turn": src_turn,
            "tgt_pos": tgt_pos,
            "tgt_type": tgt_type,
            "tgt_turn": tgt_turn,
        }
        outputs = self.model(inputs, is_training=False)
        return outputs["dec_probs"]


def build_galaxy():
    """Full GALAXY pretraining model: with_joint_act=True (explicit dialog-act
    policy injection via the [MASK]-token latent + act_classifier head)."""
    model = GalaxyWrapper(with_joint_act=True)
    model.eval()
    return model


def example_input_galaxy():
    return torch.zeros(1)


MENAGERIE_ENTRIES = [
    (
        "GALAXY (UniLM Dialog-Act-Conditioned Transformer)",
        "build_galaxy",
        "example_input_galaxy",
        2022,
        "vendored-pytorch",
    ),
]
