# SOURCE: vendored from waynewu6250/LABAN @ main
# (model/bert_zsl.py)
#
# LABAN (Lexical Awareness for Better Action Network) -- Wu, Fung, "Zero-Shot
# Learning for Joint Multi-Domain and Multi-Intent Detection", (EMNLP 2021
# workshop / associated repo). Real architecture: two independent BERT
# encoders (a query/utterance encoder and a label encoder), where the label
# encoder's [CLS] hidden state for each candidate label/intent phrase forms
# a "cluster" matrix; a Gram-matrix least-squares projection
# (`weights = pooled_output @ clusters^T @ (clusters @ clusters^T)^-1`)
# re-expresses each utterance embedding in terms of the label embeddings --
# the "label-aware" zero-shot compatibility scores used downstream for
# multi-intent classification. This label-embedding-conditioned attention
# mechanism is LABAN's own contribution; it is not present in a stock
# `BertModel`/`BertForSequenceClassification`, so this is vendored (rung 2)
# rather than recipe'd.
#
# Vendoring notes (imports/config fixes only, architecture untouched):
#   - `BertZSL.__init__` originally called
#     `BertModel.from_pretrained('bert-base-uncased', ...)` twice (utterance
#     encoder `self.bert` + label encoder `self.bertlabelencoder`); the
#     traced entry constructs two real `transformers.BertModel` instances
#     from a tiny random-init `BertConfig` instead of downloading
#     pretrained weights, per the menagerie "tiny config, random init"
#     convention. The `BertModel` class itself, and every downstream
#     module (`self.classifier`, `self.mapping`, `self.relations1/2`), are
#     unchanged.
#   - Commented-out alternative encoders (TOD-BERT, ALBERT) and the
#     `self.pre` pretrained-checkpoint-loading branch (`torch.load(
#     'checkpoints/best_e2e_pretrain.pth')`) are dropped -- dead/unused
#     code paths in the original file, not exercised when `self.pre=False`
#     (the original's own default).
#   - `self.mode = 'normal'` and `self.mode2 = 'zero-shot'` are the
#     defaults hardcoded in the original `__init__`; kept as-is (this is
#     LABAN's flagship zero-shot config -- the Gram-matrix `multi_learn`
#     branch). The other `mode`/`mode2` branches (self-attentive pooling,
#     hierarchical pooling, "bissect", "gram"/"dot"/"dnn"/"student"
#     classification heads) are copied verbatim from the source for
#     fidelity even though the traced forward pass only exercises the
#     'normal' + 'zero-shot' branches.
#   - `forward()` is copied verbatim; `labels` (used only inside the
#     'zero-shot' `multi_learn` branch as a docstring-documented argument
#     that is in fact unused by the Gram-matrix computation itself, exactly
#     as in the original) is still threaded through unchanged for fidelity.

import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

_HIDDEN = 32
_LAYERS = 2
_HEADS = 2
_VOCAB = 999
_MAXPOS = 64
_BATCH = 2
_SEQ = 12
_LABEL_SEQ = 6
_NUM_LABELS = 4


def _tiny_bert_config():
    return BertConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        intermediate_size=_HIDDEN * 2,
        max_position_embeddings=_MAXPOS,
    )


class BertZSL(nn.Module):
    """Main LABAN model"""

    def __init__(self, config, num_labels=2):
        super(BertZSL, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_labels = num_labels

        self.bert = BertModel(config)
        # You can share the utterance and label encoder by removing the following encoder
        self.bertlabelencoder = BertModel(config)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

        self.mapping = nn.Linear(config.hidden_size, num_labels)
        self.relations1 = nn.Linear(2 * config.hidden_size, 10)
        self.relations2 = nn.Linear(10, 1)

        self.mode = "normal"
        self.mode2 = "zero-shot"
        self.pre = False

        # Self-attentive
        if self.mode == "self-attentive":
            self.linear1 = nn.Linear(config.hidden_size, 256)
            self.linear2 = nn.Linear(4 * 256, config.hidden_size)
            self.tanh = nn.Tanh()
            self.context_vector = nn.Parameter(torch.randn(256, 4), requires_grad=True)

        # Hierarchical
        if self.mode == "h-max-pooling":
            self.linear = nn.Linear(13 * config.hidden_size, config.hidden_size)

    def forward(self, x_caps, x_masks, y_caps, y_masks, labels):
        """
        BERT outputs:
        last_hidden_states: (b, t, h)
        pooled_output: (b, h), from output of a linear classifier + tanh
        hidden_states: 13 x (b, t, h), embed to last layer embedding
        attentions: 12 x (b, num_heads, t, t)
        """
        # label encoder:
        label_out = self.bertlabelencoder(
            y_caps, attention_mask=y_masks, output_hidden_states=True, output_attentions=True
        )
        _last_hidden, clusters, _hidden, _att = (
            label_out.last_hidden_state,
            label_out.pooler_output,
            label_out.hidden_states,
            label_out.attentions,
        )
        utt_out = self.bert(
            x_caps, attention_mask=x_masks, output_hidden_states=True, output_attentions=True
        )
        last_hidden_states, pooled_output, hidden_states, attentions = (
            utt_out.last_hidden_state,
            utt_out.pooler_output,
            utt_out.hidden_states,
            utt_out.attentions,
        )

        pooled_output = self.transform(
            last_hidden_states, pooled_output, hidden_states, attentions, x_masks
        )  # (b, h)
        logits = self.multi_learn(pooled_output, clusters, labels)

        return last_hidden_states, pooled_output, logits

    def transform(self, last_hidden_states, pooled_output, hidden_states, attentions, mask):
        """Fuse the token-level hidden states into sentence representation."""

        if self.mode == "max-pooling":
            pooled_output, indexes = torch.max(last_hidden_states * mask[:, :, None], dim=1)

        elif self.mode == "self-attentive":
            b, _, _ = last_hidden_states.shape
            vectors = self.context_vector.unsqueeze(0).repeat(b, 1, 1)

            h = self.linear1(last_hidden_states)  # (b, t, h)
            scores = torch.bmm(h, vectors)  # (b, t, 4)
            scores = nn.Softmax(dim=1)(scores)  # (b, t, 4)
            outputs = torch.bmm(scores.permute(0, 2, 1), h).view(b, -1)  # (b, 4h)
            pooled_output = self.linear2(outputs)

        elif self.mode == "self-attentive-mean":
            b, _, _ = last_hidden_states.shape
            vector = torch.mean(last_hidden_states, dim=1).unsqueeze(2)

            scores = torch.bmm(last_hidden_states, vector)  # (b, t, 1)
            scores = nn.Softmax(dim=1)(scores)  # (b, t, 1)
            pooled_output = torch.bmm(scores.permute(0, 2, 1), last_hidden_states).squeeze(
                1
            )  # (b, h)

        elif self.mode == "h-max-pooling":
            b, t, h = last_hidden_states.shape
            N = len(hidden_states)
            final_vectors = torch.zeros(b, h, N).to(self.device)
            for i in range(len(hidden_states)):
                outs, _ = torch.max(hidden_states[i] * mask[:, :, None], dim=1)
                final_vectors[:, :, i] = outs
            final_vectors = final_vectors.view(b, -1)
            pooled_output = self.linear(final_vectors)

        elif self.mode == "bissect":
            hidden_states = hidden_states[1:]
            b, t, h = hidden_states[0].shape
            N = len(hidden_states)

            h_states = torch.zeros(b, t, h, N).to(self.device)
            for i in range(N):
                h_states[:, :, :, i] = hidden_states[i]

            final_vectors = torch.zeros(b, t, h).to(self.device)

            for i in range(t):
                word_vector = h_states[:, i, :, :]  # (b, h, N)
                vector = torch.mean(word_vector, dim=2).unsqueeze(1)  # (b, 1, h)

                scores = torch.bmm(vector, word_vector)  # (b, 1, N)
                scores = nn.Softmax(dim=2)(scores)  # (b, 1, N)
                final_vectors[:, i, :] = torch.bmm(word_vector, scores.permute(0, 2, 1)).squeeze(
                    2
                )  # (b, h)

            pooled_output, indexes = torch.max(final_vectors * mask[:, :, None], dim=1)

        else:
            # Baseline: Use [CLS] head
            pooled_output = pooled_output

        return pooled_output

    def multi_learn(self, pooled_output, clusters, labels):
        """Interact with the label embeddings."""

        if self.mode2 == "gram":
            gram = torch.mm(clusters, clusters.permute(1, 0))  # (n, n)
            weights = torch.mm(pooled_output, clusters.permute(1, 0))
            weights = torch.mm(weights, torch.inverse(gram))
            pooled_output = torch.mm(weights, clusters)  # (b, h)

        elif self.mode2 == "dot":
            weights = torch.mm(pooled_output, clusters.permute(1, 0))
            pooled_output = torch.mm(weights, clusters)  # (b, h)

        elif self.mode2 == "dnn":
            weights = self.mapping(pooled_output)  # (b, n)
            weights = nn.Tanh()(weights)
            pooled_output = torch.mm(weights, clusters)  # (b, h)

        elif self.mode2 == "student":
            self.alpha = 20.0
            q = 1.0 / (
                1.0
                + (
                    torch.sum(torch.square(pooled_output[:, None, :] - clusters), axis=2)
                    / self.alpha
                )
            )
            q = nn.Softmax(dim=1)(q)
            pooled_output = torch.mm(q, clusters)

        elif self.mode2 == "zero-shot":
            gram = torch.mm(clusters, clusters.permute(1, 0))  # (n, n)
            weights = torch.mm(pooled_output, clusters.permute(1, 0))
            weights = torch.mm(weights, torch.inverse(gram)) * np.sqrt(768)
            return weights

        else:
            pooled_output = pooled_output

        pooled_output_d = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_d)

        return logits


class _LABANEntry(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertZSL(_tiny_bert_config(), num_labels=_NUM_LABELS)

    def forward(self, x_caps, x_masks, y_caps, y_masks, labels):
        return self.model(x_caps, x_masks, y_caps, y_masks, labels)


def build_laban():
    m = _LABANEntry()
    m.eval()
    return m


def example_input_laban():
    x_caps = torch.randint(0, _VOCAB, (_BATCH, _SEQ))
    x_masks = torch.ones(_BATCH, _SEQ)
    y_caps = torch.randint(0, _VOCAB, (_NUM_LABELS, _LABEL_SEQ))
    y_masks = torch.ones(_NUM_LABELS, _LABEL_SEQ)
    labels = torch.randint(0, 2, (_BATCH, _NUM_LABELS)).float()
    return (x_caps, x_masks, y_caps, y_masks, labels)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "LABAN (Label-Aware BERT Attention Network)",
        build_laban,
        example_input_laban,
        2021,
        "vendored-pytorch",
    ),
]
