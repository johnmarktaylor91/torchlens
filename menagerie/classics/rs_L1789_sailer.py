# SOURCE: vendored from CSHaitao/SAILER @ main
# https://raw.githubusercontent.com/CSHaitao/SAILER/main/src/modeling.py
#
# Li, Ma, Fang, Wu, Zhou, Chen, Liu, Ye 2023 (SIGIR 2023) "SAILER: Structure-aware
# Pre-trained Language Model for Legal Case Retrieval" -- a structure-aware BERT
# pretraining scheme for legal case documents. The document is split into a "fact"
# span (span A) and two structurally-distinct spans that must be reconstructed from
# it: the "reasoning" (rationale) span and the "judgment" (decision) span. The model
# is a stock `BertForMaskedLM` encoder plus two lightweight asymmetric decoder heads
# (`reason_head`, `jud_head`), each a small stack of `BertLayer` transformer blocks.
# Each decoder head takes the encoder's [CLS] embedding of the fact span concatenated
# with the *embedding-only* (non-contextualized) tokens of its target span, runs them
# through its own `BertLayer` stack, and is trained with an auxiliary MLM loss on that
# span -- forcing the [CLS] embedding to encode enough structure to reconstruct both
# the legal reasoning and the judgment from the fact pattern alone. Total loss is the
# base encoder MLM loss plus both head MLM losses.
#
# `BertForSAILER` and `MaskedLMOutputWithLogs` are copied verbatim from the real
# `src/modeling.py` (only whitespace/comment cleanup; no architectural changes). It
# imports and subclasses only `transformers.BertForMaskedLM` / `BertLayer` -- no other
# repo-private modules are needed to construct and run the module in eval mode.
#
# `BertForSAILER.forward` only accepts `**model_input` (the real repo's calling
# convention -- it is always invoked as `model(**batch_dict)` by `src/trainer.py`).
# `SAILERTraceAdapter` below is a thin, architecture-free positional-arg wrapper that
# forwards named tensor args into that same real kwargs call, purely to fit a
# single-example-input tracer; the wrapped `.model` submodule IS the unmodified
# `BertForSAILER`.

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertLayer


@dataclass
class MaskedLMOutputWithLogs(MaskedLMOutput):
    logs: Optional[Dict[str, any]] = None


class BertForSAILER(BertForMaskedLM):
    def __init__(
        self,
        config,
        use_decoder_head: bool = True,
        n_head_layers: int = 2,
        enable_head_mlm: bool = True,
        head_mlm_coef: float = 1.0,
    ):
        super().__init__(config)
        if use_decoder_head:
            self.reason_head = nn.ModuleList([BertLayer(config) for _ in range(n_head_layers)])
            self.reason_head.apply(self._init_weights)

        self.jud_head = nn.ModuleList([BertLayer(config) for _ in range(n_head_layers)])
        self.jud_head.apply(self._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.use_decoder_head = use_decoder_head
        self.n_head_layers = n_head_layers
        self.enable_head_mlm = enable_head_mlm
        self.head_mlm_coef = head_mlm_coef

    def forward(self, **model_input):
        lm_out: MaskedLMOutput = super().forward(
            input_ids=model_input["input_ids"],
            attention_mask=model_input["attention_mask"],
            labels=model_input["labels"],
            output_hidden_states=True,
            return_dict=True,
        )

        cls_hiddens = lm_out.hidden_states[-1][:, 0]
        logs = dict()

        # add last layer mlm loss
        loss = lm_out.loss
        logs["encoder_mlm_loss"] = lm_out.loss.item()

        if self.use_decoder_head and self.enable_head_mlm:
            decoder_embedding_output = self.bert.embeddings(
                input_ids=model_input["reason_input_ids"]
            )
            decoder_attention_mask = self.get_extended_attention_mask(
                model_input["reason_attention_mask"],
                model_input["reason_attention_mask"].shape,
                model_input["reason_attention_mask"].device,
            )

            # Concat cls-hiddens of span A & embedding of span B
            hiddens = torch.cat([cls_hiddens.unsqueeze(1), decoder_embedding_output[:, 1:]], dim=1)
            for layer in self.reason_head:
                layer_out = layer(hiddens, decoder_attention_mask)
                hiddens = layer_out[0]

            # add head-layer mlm loss
            head_mlm_loss = (
                self.mlm_loss(hiddens, model_input["reason_labels"]) * self.head_mlm_coef
            )
            logs["reason_loss"] = head_mlm_loss.item()

            decoder_embedding_output = self.bert.embeddings(
                input_ids=model_input["judgment_input_ids"]
            )
            decoder_attention_mask = self.get_extended_attention_mask(
                model_input["judgment_attention_mask"],
                model_input["judgment_attention_mask"].shape,
                model_input["judgment_attention_mask"].device,
            )
            # Concat cls-hiddens of span A & embedding of span B
            hiddens = torch.cat([cls_hiddens.unsqueeze(1), decoder_embedding_output[:, 1:]], dim=1)

            for layer in self.jud_head:
                layer_out = layer(hiddens, decoder_attention_mask)
                hiddens = layer_out[0]

            cause_mlm_loss = (
                self.mlm_loss(hiddens, model_input["judgment_labels"]) * self.head_mlm_coef
            )
            logs["decision_loss"] = cause_mlm_loss.item()

            loss = loss + head_mlm_loss
            loss = loss + cause_mlm_loss

        return MaskedLMOutputWithLogs(
            loss=loss,
            logits=lm_out.logits,
            hidden_states=lm_out.hidden_states,
            attentions=lm_out.attentions,
            logs=logs,
        )

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.config.vocab_size), labels.view(-1)
        )
        return masked_lm_loss


class SAILERTraceAdapter(nn.Module):
    """Thin positional-arg adapter around the real, unmodified `BertForSAILER`.

    `BertForSAILER.forward` only accepts `**model_input` (the vendored calling
    convention above), but a single-tensor-input tracer needs an ordinary
    positional/tensor signature. This wrapper performs no architecture work of
    its own -- it just forwards named tensor args into the real model's kwargs
    call, unchanged.
    """

    def __init__(self, sailer_model: "BertForSAILER"):
        super().__init__()
        self.model = sailer_model

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        reason_input_ids,
        reason_attention_mask,
        reason_labels,
        judgment_input_ids,
        judgment_attention_mask,
        judgment_labels,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            reason_input_ids=reason_input_ids,
            reason_attention_mask=reason_attention_mask,
            reason_labels=reason_labels,
            judgment_input_ids=judgment_input_ids,
            judgment_attention_mask=judgment_attention_mask,
            judgment_labels=judgment_labels,
        )


def build_sailer():
    cfg = BertConfig(
        vocab_size=200,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    model = BertForSAILER(cfg, use_decoder_head=True, n_head_layers=2, enable_head_mlm=True)
    model.eval()
    return SAILERTraceAdapter(model)


def example_input_sailer():
    batch, seq_len = 1, 12
    vocab_size = 200

    def _ids():
        return torch.randint(1, vocab_size, (batch, seq_len), dtype=torch.long)

    def _labels():
        return torch.randint(0, vocab_size, (batch, seq_len), dtype=torch.long)

    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

    return (
        _ids(),
        attention_mask,
        _labels(),
        _ids(),
        attention_mask.clone(),
        _labels(),
        _ids(),
        attention_mask.clone(),
        _labels(),
    )


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SAILER", "build_sailer", "example_input_sailer", 2023, "vendored"),
]
