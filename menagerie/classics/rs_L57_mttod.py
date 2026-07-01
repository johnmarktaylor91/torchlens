# SOURCE: vendored from bepoetree/MTTOD @ main (commit at fetch time)
# https://raw.githubusercontent.com/bepoetree/MTTOD/main/model.py
#
# Lee 2021 (EMNLP) "Improving End-to-End Task-Oriented Dialog System with a
# Simple Auxiliary Task" (MTTOD) -- a T5 encoder-decoder that (1) adds a SECOND,
# independently-weight-initialized decoder stack + LM head (`resp_decoder`,
# `resp_lm_head`, selected via `decoder_type="resp"`) alongside the base T5
# decoder/`lm_head` so belief-state and response generation get separate decoder
# parameters over a shared encoder, and (2) in the `T5WithTokenSpan` subclass adds
# an auxiliary per-token span-classification head (`span_head`, a `Linear` over the
# encoder hidden states) trained jointly as the "simple auxiliary task" from the
# paper title. This is the actual `T5WithSpan`/`T5WithTokenSpan` architecture from
# `model.py` -- unchanged.
#
# `T5WithSpan` and `T5WithTokenSpan` are copied verbatim (module bodies unchanged)
# from `model.py`. No architectural code was rewritten; only these mechanical,
# import-isolation changes were made:
#   - Dropped two imports that are present in the original file but never referenced
#     anywhere in its body: `T5EncoderModel` (imported but unused upstream) and
#     `torch.nn.utils.rnn.pad_sequence` (imported but unused upstream), plus
#     `from utils import definitions` (a repo-local module of string constants for
#     the dialogue-state schema, also never referenced in `model.py`'s body -- only
#     used by the training/data-loading scripts this vendoring intentionally omits).
#   - `T5WithSpan.initialize_weights(self, modules)` -> `initialize_span_weights` (name
#     only, body unchanged). In 2021 (when this code was written) `PreTrainedModel` had
#     no method of that name; modern `transformers` (>=4.5x, installed here: 4.57.6)
#     added its own zero-arg `PreTrainedModel.initialize_weights()` hook called from
#     `post_init()`, which the same-named 2-arg override now shadows incompatibly
#     (`TypeError: missing 1 required positional argument: 'modules'` at construction
#     time). Renaming resolves the collision; the method's logic (init Linear/Embedding
#     weights normal(0, 0.02), zero LayerNorm bias + fill weight 1.0, zero Linear bias)
#     is untouched.
#   - `build_mttod()`/`example_input_mttod()` below are new (not in the original
#     file) and construct the real classes with a tiny random-init `T5Config` plus a
#     synthetic input batch, matching how `runner.py` upstream calls the model
#     (`decoder_type="resp"` selects the auxiliary response decoder branch;
#     `add_auxiliary_task=True` exercises the span-prediction path in
#     `T5WithTokenSpan`). `MTTODTraceWrapper` is a thin positional-args adapter
#     (see FiD's `FiDTraceWrapper` in `menagerie/classics/rs_L14_04_fid.py` for the
#     established pattern) around the keyword-heavy `forward()`; no architecture change.

import copy

import torch
from torch import nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


class T5WithSpan(T5ForConditionalGeneration):
    def __init__(self, config, num_span):
        super(T5WithSpan, self).__init__(config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        self.resp_decoder = type(self.decoder)(decoder_config, self.shared)
        self.resp_lm_head = type(self.lm_head)(config.d_model, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.dropout_rate)

    def initialize_additional_decoder(self):
        decoder_config = copy.deepcopy(self.config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        self.resp_decoder = type(self.decoder)(decoder_config, self.shared)
        self.resp_lm_head = type(self.lm_head)(
            self.config.d_model, self.config.vocab_size, bias=False
        )

        self.resp_decoder.load_state_dict(self.decoder.state_dict())
        self.resp_lm_head.load_state_dict(self.lm_head.state_dict())

    def initialize_span_weights(self, modules):
        for module in modules:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def predict_span(self, encoder_hidden_states, attention_mask, span_labels=None):
        span_loss, pred_spans, span_logits = 0, None, None

        return span_loss, pred_spans, span_logits

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "decoder_type": kwargs.get("decoder_type"),
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        span_labels=None,
        lm_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_only=None,
        add_auxiliary_task=None,
        decoder_type=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        span_loss, pred_spans, span_logits = 0, None, None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                return_dict=return_dict,
            )

            if return_dict:
                encoder_hidden_states = encoder_outputs.last_hidden_state
            else:
                encoder_hidden_states = encoder_outputs[0]

            hs = encoder_hidden_states * (self.model_dim**-0.5)

            if add_auxiliary_task:
                span_loss, pred_spans, span_logits = self.predict_span(
                    hs, attention_mask, span_labels
                )

        else:
            if isinstance(encoder_outputs, tuple):
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs.last_hidden_state

        if encoder_only:
            return (span_loss, pred_spans, span_logits), encoder_outputs

        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(lm_labels)

        if decoder_type == "resp":
            decoder = self.resp_decoder
            lm_head = self.resp_lm_head

        else:
            decoder = self.decoder
            lm_head = self.lm_head

        if past_key_values is not None:
            assert lm_labels is None, "Decoder should not use cached key value states when training"
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = lm_head(sequence_output)

        lm_loss = None
        if lm_labels is not None:
            lm_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            lm_loss = lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

        # for training
        if not return_dict:
            pred_lm = torch.argmax(lm_logits, dim=-1)
            outputs = (
                (
                    lm_loss,
                    pred_lm,
                )
                + (span_loss, pred_spans, span_logits, encoder_hidden_states)
                + decoder_outputs[1:]
            )

        # for prediction
        else:
            outputs = Seq2SeqLMOutput(
                loss=lm_loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                encoder_attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        return outputs


class T5WithTokenSpan(T5WithSpan):
    def __init__(self, config, num_span):
        super(T5WithTokenSpan, self).__init__(config, num_span)

        self.num_span_labels = num_span * 2 + 2
        self.span_head = nn.Linear(config.d_model, self.num_span_labels)

        self.initialize_span_weights([self.span_head])

    def predict_span(self, encoder_hidden_states, attention_mask, span_labels=None):
        span_head = self.span_head.to(encoder_hidden_states.device)

        span_logits = span_head(encoder_hidden_states)

        pred_spans = torch.argmax(span_logits, dim=-1)

        span_loss = 0
        if span_labels is not None:
            span_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            span_loss = span_loss_fct(
                span_logits.view(-1, self.num_span_labels), span_labels.view(-1)
            )

        return span_loss, pred_spans, span_logits


class MTTODTraceWrapper(nn.Module):
    """Thin positional-args adapter around T5WithTokenSpan for the trace-harness
    calling convention (the harness calls the example/build pair with plain
    positional args; the real `T5WithSpan.forward` takes many keyword-only args
    including flags like `decoder_type`/`add_auxiliary_task`). No architecture
    change -- forwards straight into the real module with the paper's actual
    training-time call signature (joint span + response-decoder auxiliary task)."""

    def __init__(self, mttod_model):
        super().__init__()
        self.mttod = mttod_model

    def forward(self, input_ids, attention_mask, lm_labels, span_labels):
        return self.mttod(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lm_labels=lm_labels,
            span_labels=span_labels,
            add_auxiliary_task=True,
            decoder_type="resp",
            return_dict=True,
        )


def build_mttod():
    from transformers import T5Config

    config = T5Config(
        d_model=32,
        d_ff=37,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_kv=8,
        vocab_size=100,
        dropout_rate=0.1,
        use_cache=False,
        return_dict=True,
        pad_token_id=0,
        decoder_start_token_id=0,
    )
    num_span = 10  # number of dialogue-state slot types (paper's auxiliary span task)
    model = T5WithTokenSpan(config, num_span)
    model.initialize_additional_decoder()
    return MTTODTraceWrapper(model)


def example_input_mttod():
    batch = 2
    src_len = 6
    tgt_len = 5
    input_ids = torch.randint(2, 100, (batch, src_len))
    attention_mask = torch.ones(batch, src_len, dtype=torch.int64)
    lm_labels = torch.randint(2, 100, (batch, tgt_len))
    span_labels = torch.randint(0, 22, (batch, src_len))
    return (input_ids, attention_mask, lm_labels, span_labels)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MTTOD", "build_mttod", "example_input_mttod", 2021, "vendored"),
]
