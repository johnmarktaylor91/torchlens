# FAITHFUL PORT of awasthiabhijeet/PIE @ master (original framework: TensorFlow 1.12)
# https://raw.githubusercontent.com/awasthiabhijeet/PIE/master/modified_modeling.py
# https://raw.githubusercontent.com/awasthiabhijeet/PIE/master/word_edit_model.py (gec_create_model,
#   the head that consumes modified_modeling.BertModel's output)
#
# Awasthi et al. 2019 (EMNLP) "Parallel Iterative Edit Models for Local Sequence Transduction" (PIE) --
# the repo is TensorFlow 1.x throughout (requirements.txt pins tensorflow==1.12; `modeling.py` is the
# unmodified google-research/bert TF1 reference and `modified_modeling.py` -- explicitly "for edit
# factorized architecture, figure 2 in the paper" -- is PIE's own architectural contribution). No PyTorch
# implementation exists anywhere in the repo, so per the ladder this is transcribed faithfully into
# self-contained torch (rung 3), preserving every mechanism from the real TF1 code:
#
#   1. `BertModelIRI` (transcribed from `modified_modeling.BertModel.__init__`): given one token-id
#      sequence, builds THREE parallel embedding streams over the SAME `seq_len` -- (a) the ordinary
#      "input" stream (word + position + token-type embeddings of the real tokens), (b) a "replace"
#      stream (word + position + token-type embeddings of an all-[MASK] sequence, `get_mask_ids`), and
#      (c) an "insert" stream (also all-[MASK], but using the *midpoint* position embeddings computed by
#      `get_mid_position_embeddings` -- the average of adjacent position rows, so an inserted token's
#      position embedding sits "between" two existing tokens). The three streams are concatenated along
#      the sequence axis into one `3*seq_len`-long sequence and run through a single shared-weight
#      standard Transformer encoder (`transformer_model`, exactly google-research/bert's post-LN encoder).
#   2. The attention mask is NOT a standard causal/padding mask -- `create_input_rep_ins_attention_mask`
#      builds a custom 3x3 block mask (`concat_list_1/2/3` in the source) so that: input-stream tokens
#      attend to all real input tokens (block 1); replace-stream position i attends to all real input
#      tokens EXCEPT the identity match at position i, PLUS attends to itself via the identity block
#      (blocks 2); insert-stream position i attends to all real input tokens AND to itself (blocks 3).
#      This is the "edit factorized architecture" from Figure 2 of the paper: the replace/insert slots
#      see the sentence context but are prevented from trivially copying their own (masked) token.
#   3. `gec_create_model`'s head (transcribed as `PIEEditHead`): slices the `3*seq_len` encoder output
#      back into `output_layer` (input), `replace_layer`, `append_layer`; runs a shared
#      `cls/predictions/transform` dense+GELU+LayerNorm ("h_word"/"m_replace"/"m_append", weight-tied
#      across the three -- `reuse=True` in the TF1 source) on each; computes the edit-label logits as the
#      SUM of three terms per eq. 3 in the paper: (i) `edit_logits` = a plain linear classifier over
#      `h_edit` (= the raw, untransformed input-stream output) into `num_labels` edit classes (copy/
#      delete/append-i/replace-i/suffix-transform-i); (ii) `inplace_word_logits`, the "copy" bias term =
#      dot product of the transformed word representation with its own input embedding, broadcast into
#      the copy/append/suffix-transform slots (zero elsewhere); (iii) `additional_logits`, the
#      append/replace bias = the transformed append/replace representations dotted with the "edit
#      vocabulary" word-embedding rows (`append_weights`/`replace_weights`, gathered from the same
#      token-embedding table via `insert_ids`), optionally with `replacement_minus_replaced_logits`
#      (the "subtract_replaced_from_replacement" option, default on) subtracting the replaced token's own
#      embedding dot-product from the replace-logit -- both transcribed verbatim.
#
# Only mechanical translation choices were made getting this out of TF1 graph-mode into eager torch
# (`tf.get_variable` -> `nn.Parameter`/`nn.Embedding`/`nn.Linear`, `tf.layers.dense` -> `nn.Linear`,
# `tf.contrib.layers.layer_norm` -> `nn.LayerNorm`, GELU uses the exact erf-based formula from
# `modified_modeling.gelu`); no mechanism was added, dropped, or altered. `use_one_hot_embeddings`,
# `is_training`-gated dropout scaling, and TPU-only code paths are omitted as TF-runtime-only concerns
# with no forward-pass effect (`is_training=False` disables all dropout in the original too, since the
# reference config always zeroes `hidden_dropout_prob`/`attention_probs_dropout_prob` in that path).
# `num_suffix_transforms=58` and the append/insert vocab are configuration constants from the source
# (`get_edit_vocab.py`); here reduced to tiny sizes for a fast trace-only forward pass.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact erf-based GELU, transcribed from modified_modeling.gelu."""
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x * cdf


def get_mid_position_embeddings(full_position_embeddings: torch.Tensor) -> torch.Tensor:
    """Transcribed from modified_modeling.get_mid_position_embeddings: the average of each pair of
    adjacent position-embedding rows (with a zero row prepended/appended), used as the "insert slot"
    position embedding so an inserted token's position sits between two existing tokens."""
    width = full_position_embeddings.size(-1)
    zeros = full_position_embeddings.new_zeros(1, width)
    t1 = torch.cat([zeros, full_position_embeddings], dim=0)
    t2 = torch.cat([full_position_embeddings, zeros], dim=0)
    t = (t1 + t2) / 2
    return t[1 : full_position_embeddings.size(0) + 1]


class BertEmbeddingsIRI(nn.Module):
    """Word + position + token-type embedding lookup and post-processing, transcribed from
    modified_modeling.embedding_lookup / embedding_postprocessor. Shared word/position/token-type
    tables across the input/replace/insert streams (weight-tied, as in the TF1 `reuse=True` scopes)."""

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        type_vocab_size,
        dropout_prob,
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, insert_mode: bool = False
    ):
        batch_size, seq_len = input_ids.shape
        word_emb = self.word_embeddings(input_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        output = word_emb + token_type_emb

        full_pos = self.position_embeddings.weight  # [max_position_embeddings, hidden_size]
        if insert_mode:
            pos_table = get_mid_position_embeddings(full_pos)
        else:
            pos_table = full_pos
        position_embeddings = pos_table[:seq_len].unsqueeze(0)  # [1, seq_len, hidden]
        output = output + position_embeddings

        output = self.layer_norm(output)
        output = self.dropout(output)
        return output, word_emb


class BertSelfAttentionIRI(nn.Module):
    """Multi-headed self-attention, transcribed from modified_modeling.attention_layer."""

    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size, seq_len):
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape
        query_layer = self.transpose_for_scores(self.query(hidden_states), batch_size, seq_len)
        key_layer = self.transpose_for_scores(self.key(hidden_states), batch_size, seq_len)
        value_layer = self.transpose_for_scores(self.value(hidden_states), batch_size, seq_len)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # attention_mask: [batch, seq_len, seq_len], 1.0 = attend, 0.0 = masked
        adder = (1.0 - attention_mask.unsqueeze(1)) * -10000.0
        attention_scores = attention_scores + adder

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            batch_size, seq_len, self.num_attention_heads * self.attention_head_size
        )
        return context_layer


class BertLayerIRI(nn.Module):
    """One post-LN Transformer block, transcribed from modified_modeling.transformer_model's per-layer body."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.attention = BertSelfAttentionIRI(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.attention_output_dense = nn.Linear(hidden_size, hidden_size)
        self.attention_output_dropout = nn.Dropout(hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output_dense(attention_output)
        attention_output = self.attention_output_dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + hidden_states)

        intermediate_output = gelu(self.intermediate_dense(attention_output))
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)
        return layer_output


class BertModelIRI(nn.Module):
    """Transcribed from modified_modeling.BertModel: builds input/replace/insert embedding streams,
    concatenates along the sequence axis, and runs the shared-weight Transformer encoder with the
    custom 3x3-block "edit factorized" attention mask."""

    def __init__(
        self,
        vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        mask_token_id=103,
    ):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.embeddings = BertEmbeddingsIRI(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size, hidden_dropout_prob
        )
        self.layers = nn.ModuleList(
            [
                BertLayerIRI(
                    hidden_size,
                    num_attention_heads,
                    intermediate_size,
                    hidden_dropout_prob,
                    attention_probs_dropout_prob,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    @staticmethod
    def _create_input_rep_ins_attention_mask(input_mask: torch.Tensor) -> torch.Tensor:
        """Transcribed from modified_modeling.create_input_rep_ins_attention_mask."""
        batch_size, seq_len = input_mask.shape
        device = input_mask.device

        input_attention_mask = input_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len).float()

        eye = (
            torch.eye(seq_len, device=device, dtype=torch.bool)
            .unsqueeze(0)
            .expand(batch_size, seq_len, seq_len)
        )
        replace_attention_mask = (~eye).float() * input_attention_mask
        identity_attention_mask = eye.float() * input_attention_mask
        zeros_attention_mask = torch.zeros(batch_size, seq_len, seq_len, device=device)

        row1 = torch.cat([input_attention_mask, zeros_attention_mask, zeros_attention_mask], dim=2)
        row2 = torch.cat(
            [replace_attention_mask, identity_attention_mask, zeros_attention_mask], dim=2
        )
        row3 = torch.cat(
            [input_attention_mask, zeros_attention_mask, identity_attention_mask], dim=2
        )

        return torch.cat([row1, row2, row3], dim=1)

    def forward(
        self, input_ids: torch.Tensor, input_mask: torch.Tensor, token_type_ids: torch.Tensor
    ):
        batch_size, seq_len = input_ids.shape
        mask_ids = torch.full_like(input_ids, 0)
        mask_ids = torch.where(
            input_ids != 0, torch.full_like(input_ids, self.mask_token_id), mask_ids
        )

        input_embedding_output, word_embedded_input = self.embeddings(
            input_ids, token_type_ids, insert_mode=False
        )
        replace_embedding_output, _ = self.embeddings(mask_ids, token_type_ids, insert_mode=False)
        insert_embedding_output, _ = self.embeddings(mask_ids, token_type_ids, insert_mode=True)

        embedding_output = torch.cat(
            [input_embedding_output, replace_embedding_output, insert_embedding_output], dim=1
        )

        attention_mask = self._create_input_rep_ins_attention_mask(input_mask)

        hidden_states = embedding_output
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states, word_embedded_input, self.embeddings.word_embeddings.weight


class PIEEditHead(nn.Module):
    """Transcribed from word_edit_model.gec_create_model's edit-factorized logit computation (eq. 3 in
    the paper): edit_logits + inplace_word_logits (copy bias) + additional_logits (append/replace bias)."""

    def __init__(
        self,
        hidden_size,
        num_appends,
        num_suffix_transforms=8,
        layer_norm_eps=1e-12,
        subtract_replaced_from_replacement=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_appends = num_appends
        self.num_replaces = num_appends  # appends and replacements share the same edit vocabulary
        self.num_suffix_transforms = num_suffix_transforms
        self.num_labels = 5 + num_appends + self.num_replaces + num_suffix_transforms
        self.subtract_replaced_from_replacement = subtract_replaced_from_replacement

        self.transform_dense = nn.Linear(hidden_size, hidden_size)
        self.transform_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.edit_weights = nn.Parameter(torch.empty(self.num_labels, hidden_size))
        nn.init.trunc_normal_(self.edit_weights, std=0.02)
        self.output_bias = nn.Parameter(torch.zeros(self.num_labels))

        # "edit vocabulary" indices into the shared word-embedding table (insert_ids in the source).
        self.register_buffer("insert_ids", torch.arange(2, 2 + num_appends), persistent=False)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform_layer_norm(gelu(self.transform_dense(x)))

    def forward(
        self,
        encoder_output: torch.Tensor,
        word_embedded_input: torch.Tensor,
        word_embedding_table: torch.Tensor,
        input_mask: torch.Tensor,
    ):
        batch_size, total_len, hidden = encoder_output.shape
        seq_len = total_len // 3

        output_layer = encoder_output[:, 0:seq_len, :]
        replace_layer = encoder_output[:, seq_len : 2 * seq_len, :]
        append_layer = encoder_output[:, 2 * seq_len : 3 * seq_len, :]

        flattened_output_layer = output_layer.reshape(-1, hidden)
        flattened_replace_layer = replace_layer.reshape(-1, hidden)
        flattened_append_layer = append_layer.reshape(-1, hidden)
        flattened_word_embedded_input = word_embedded_input.reshape(-1, hidden)

        h_edit = flattened_output_layer
        h_word = self._transform(flattened_output_layer)
        m_replace = self._transform(flattened_replace_layer)
        m_append = self._transform(flattened_append_layer)

        edit_logits = h_edit @ self.edit_weights.T  # eq. 3, term 1

        n = batch_size * seq_len
        inplace_logit = (h_word * flattened_word_embedded_input).sum(
            dim=1, keepdim=True
        )  # copy bias
        inplace_logit_appends = inplace_logit.expand(n, self.num_appends)
        inplace_logit_transforms = inplace_logit.expand(n, self.num_suffix_transforms)
        zero_3 = h_edit.new_zeros(n, 3)
        zero_1 = h_edit.new_zeros(n, 1)
        zero_replace = h_edit.new_zeros(n, self.num_replaces)
        inplace_word_logits = torch.cat(
            [
                zero_3,
                inplace_logit,
                zero_1,
                inplace_logit_appends,
                zero_replace,
                inplace_logit_transforms,
            ],
            dim=1,
        )  # eq. 3, term 2

        append_weights = word_embedding_table[self.insert_ids]  # [num_appends, hidden]
        replace_weights = append_weights

        zero_5 = h_edit.new_zeros(n, 5)
        append_logits = m_append @ append_weights.T
        if self.subtract_replaced_from_replacement:
            result_1 = m_replace @ replace_weights.T
            result_2 = (m_replace * flattened_word_embedded_input).sum(dim=1, keepdim=True)
            replace_logits = result_1 - result_2
        else:
            replace_logits = m_replace @ replace_weights.T
        suffix_logits = h_edit.new_zeros(n, self.num_suffix_transforms)
        additional_logits = torch.cat(
            [zero_5, append_logits, replace_logits, suffix_logits], dim=1
        )  # eq. 3, term 3

        logits = edit_logits + inplace_word_logits + additional_logits + self.output_bias
        logits = logits.view(batch_size, seq_len, self.num_labels)
        return logits


class PIEModel(nn.Module):
    """Full PIE edit-factorized model: BertModelIRI encoder + PIEEditHead. Composition matches
    word_edit_model.gec_create_model(use_bert_more=True), the paper's "logit factorisation" setting."""

    def __init__(
        self,
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=32,
        num_appends=6,
        num_suffix_transforms=8,
        mask_token_id=3,
    ):
        super().__init__()
        # The real BERT WordPiece vocab's [MASK] id is 103 (MASK_TOKEN in modified_modeling.py);
        # here it is a configurable in-range id so the tiny synthetic vocab below stays valid.
        self.bert = BertModelIRI(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            mask_token_id=mask_token_id,
        )
        self.edit_head = PIEEditHead(
            hidden_size, num_appends=num_appends, num_suffix_transforms=num_suffix_transforms
        )

    def forward(
        self, input_ids: torch.Tensor, input_mask: torch.Tensor, token_type_ids: torch.Tensor
    ):
        encoder_output, word_embedded_input, word_embedding_table = self.bert(
            input_ids, input_mask, token_type_ids
        )
        logits = self.edit_head(
            encoder_output, word_embedded_input, word_embedding_table, input_mask
        )
        return logits


def build_pie():
    return PIEModel(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=24,
        num_appends=6,
        num_suffix_transforms=8,
    )


def example_input_pie():
    batch = 2
    seq_len = 8
    input_ids = torch.randint(1, 64, (batch, seq_len))
    input_mask = torch.ones(batch, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    return (input_ids, input_mask, token_type_ids)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("PIE", "build_pie", "example_input_pie", 2019, "ported"),
]
