# FAITHFUL PORT of clovaai/som-dst @ master (model.py)
# https://raw.githubusercontent.com/clovaai/som-dst/master/model.py
# (original framework: PyTorch, but on the archived/EOL `pytorch_transformers`
# package -- not an installed base lib, and not being added: `pytorch_transformers`
# was superseded years ago by `transformers`, which is already installed, and
# installing the old archived package risks colliding with the modern one.)
#
# Kim et al. 2020 (ACL) "Efficient Dialogue State Tracking by Selectively
# Overwriting Memory" -- SOM-DST. `model.py` defines `SomDST` (the full model:
# a BERT `Encoder` that scores a state-operation per slot + a domain classifier,
# followed by a `Decoder` that selectively regenerates only the slots flagged
# "UPDATE" via a pointer-generator GRU with vocabulary softmax + a copy
# distribution over the input sequence). Every mechanism is transcribed
# faithfully:
#   - `SomDST.__init__`/`forward` -- preserved verbatim (encoder -> decoder
#     pipeline, same argument names/order, same return tuple
#     `(domain_scores, state_scores, gen_scores)`).
#   - `Encoder` -- preserved verbatim: BERT encode, gather per-slot state
#     token representations via `state_positions`, per-slot 4-way operation
#     classifier (`action_cls`), domain classifier over the pooled output
#     (`domain_cls`), then the exact gather/pad loop that assembles
#     `decoder_inputs` from only the state positions whose predicted (or
#     teacher-forced) op equals `update_id`.
#   - `Decoder` -- preserved verbatim: tied output embedding (shares BERT's
#     word-embedding weight matrix), a single-layer GRU stepped token-by-token
#     up to `max_len`, dot-product attention over the encoder sequence output,
#     a vocabulary-softmax distribution, a copy/pointer distribution built via
#     `scatter_add_` over the input token ids, and a learned sigmoid gate
#     (`w_gen`) that mixes the two distributions into `p_final` at every
#     decoding step -- greedy next-token selection (`w_idx`) feeds the next
#     GRU step unless `teacher` forcing ids are supplied.
#
# The ONLY substitution is `pytorch_transformers.modeling_bert.{BertModel,
# BertPreTrainedModel}` -> `transformers.{BertModel, BertPreTrainedModel}`
# (a straight drop-in: `transformers.BertModel` is the direct, API-compatible
# successor of the archived package's class this repo pins). Because the
# modern `transformers.BertModel.forward` takes `(input_ids, attention_mask,
# token_type_ids, ...)` positionally -- the *old* `pytorch_transformers`
# order was `(input_ids, token_type_ids, attention_mask)` -- `Encoder.forward`
# below calls BERT with explicit keyword arguments so the real inputs still
# land on the right parameters, and unpacks `last_hidden_state`/`pooler_output`
# (the modern equivalents of the old positional
# `sequence_output, pooled_output = bert_outputs[:2]`) instead of tuple
# indexing, since modern `transformers` returns a `ModelOutput` by default.
# No architectural code (layer composition, control flow, tensor shapes,
# attention/copy/gate mechanism) was changed.

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class SomDST(BertPreTrainedModel):
    def __init__(self, config, n_op, n_domain, update_id, exclude_domain=False):
        super(SomDST, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.encoder = Encoder(config, n_op, n_domain, update_id, exclude_domain)
        self.decoder = Decoder(config, self.encoder.bert.embeddings.word_embeddings.weight)
        self.post_init()

    def forward(
        self,
        input_ids,
        token_type_ids,
        state_positions,
        attention_mask,
        max_value,
        op_ids=None,
        max_update=None,
        teacher=None,
    ):
        enc_outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            state_positions=state_positions,
            attention_mask=attention_mask,
            op_ids=op_ids,
            max_update=max_update,
        )

        domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output = enc_outputs
        gen_scores = self.decoder(
            input_ids, decoder_inputs, sequence_output, pooled_output, max_value, teacher
        )

        return domain_scores, state_scores, gen_scores


class Encoder(nn.Module):
    def __init__(self, config, n_op, n_domain, update_id, exclude_domain=False):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.exclude_domain = exclude_domain
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(
            config.dropout if hasattr(config, "dropout") else config.hidden_dropout_prob
        )
        self.action_cls = nn.Linear(config.hidden_size, n_op)
        if self.exclude_domain is not True:
            self.domain_cls = nn.Linear(config.hidden_size, n_domain)
        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id

    def forward(
        self,
        input_ids,
        token_type_ids,
        state_positions,
        attention_mask,
        op_ids=None,
        max_update=None,
    ):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        sequence_output = bert_outputs.last_hidden_state
        pooled_output = bert_outputs.pooler_output
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        state_output = torch.gather(sequence_output, 1, state_pos)
        state_scores = self.action_cls(self.dropout(state_output))  # B,J,4
        if self.exclude_domain:
            domain_scores = torch.zeros(1, device=input_ids.device)  # dummy
        else:
            domain_scores = self.domain_cls(self.dropout(pooled_output))

        batch_size = state_scores.size(0)
        if op_ids is None:
            op_ids = state_scores.view(-1, self.n_op).max(-1)[-1].view(batch_size, -1)
        if max_update is None:
            max_update = op_ids.eq(self.update_id).sum(-1).max().item()

        gathered = []
        for b, a in zip(state_output, op_ids.eq(self.update_id)):  # update
            if a.sum().item() != 0:
                v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.hidden_size)
                n = v.size(1)
                gap = max_update - n
                if gap > 0:
                    zeros = torch.zeros(1, 1 * gap, self.hidden_size, device=input_ids.device)
                    v = torch.cat([v, zeros], 1)
            else:
                v = torch.zeros(1, max_update, self.hidden_size, device=input_ids.device)
            gathered.append(v)
        decoder_inputs = torch.cat(gathered)
        return (
            domain_scores,
            state_scores,
            decoder_inputs,
            sequence_output,
            pooled_output.unsqueeze(0),
        )


class Decoder(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(Decoder, self).__init__()
        self.pad_idx = 0
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.pad_idx)
        self.embed.weight = bert_model_embedding_weights
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, 1, batch_first=True)
        self.w_gen = nn.Linear(config.hidden_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(
            config.dropout if hasattr(config, "dropout") else config.hidden_dropout_prob
        )

        for n, p in self.gru.named_parameters():
            if "weight" in n:
                p.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, x, decoder_input, encoder_output, hidden, max_len, teacher=None):
        mask = x.eq(self.pad_idx)
        batch_size, n_update, _ = decoder_input.size()  # B,J',5 # long
        state_in = decoder_input
        all_point_outputs = torch.zeros(n_update, batch_size, max_len, self.vocab_size).to(x.device)
        for j in range(n_update):
            w = state_in[:, j].unsqueeze(1)  # B,1,D
            for k in range(max_len):
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B,D
                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                attn_e = attn_e.squeeze(-1).masked_fill(mask, -1e9)
                attn_history = nn.functional.softmax(attn_e, -1)  # B,T

                # B,D * D,V => B,V
                attn_v = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V
                attn_vocab = nn.functional.softmax(attn_v, -1)

                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D

                p_gen = self.sigmoid(
                    self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1))
                )  # B,1
                p_gen = p_gen.squeeze(-1)

                p_context_ptr = torch.zeros_like(attn_vocab).to(x.device)
                p_context_ptr.scatter_add_(1, x, attn_history)  # copy B,V
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
                _, w_idx = p_final.max(-1)
                if teacher is not None:
                    w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D
                all_point_outputs[j, :, k, :] = p_final

        return all_point_outputs.transpose(0, 1)


MENAGERIE_ZOO = "ported-pytorch"


def build_som_dst():
    from transformers import BertConfig

    config = BertConfig(
        vocab_size=210,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=2,
    )
    config.dropout = 0.1
    model = SomDST(config, n_op=4, n_domain=5, update_id=0, exclude_domain=False)
    model.eval()
    return model


def example_input_som_dst():
    batch_size = 1
    seq_len = 12
    n_slots = 3
    input_ids = torch.randint(1, 210, (batch_size, seq_len))
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    state_positions = torch.randint(0, seq_len, (batch_size, n_slots)).sort(dim=1).values
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    max_value = 5
    op_ids = torch.zeros(batch_size, n_slots, dtype=torch.long)  # all "update" (update_id=0)
    return (input_ids, token_type_ids, state_positions, attention_mask, max_value, op_ids)


MENAGERIE_ENTRIES = [
    ("SOM-DST", build_som_dst, example_input_som_dst, 2020, "ported-pytorch"),
]
