# SOURCE: vendored from CompEpigen/methylbert @ main
#   src/methylbert/network.py (MethylBertEmbeddedDMR)
#   src/methylbert/config.py  (MethylBERTConfig)
#   src/methylbert/function.py (FocalLoss)
"""MethylBERT: BERT-based read-level DNA methylation classifier + tumor
deconvolution (CompEpigen/methylbert, Nature Communications 2025).

The real architecture wraps an unmodified ``transformers.BertModel`` with a
DMR (differentially methylated region) label embedding that is concatenated
onto BERT's per-token hidden states, then flattened and fed through a small
MLP read-classifier head (``read_classifier``). This head + DMR-embedding
fusion is the paper's actual architectural contribution, so this is vendored
as a staging module (rung 2) rather than a bare rung-1 recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from transformers import BertModel, BertPreTrainedModel, BertConfig


# ---------------------------------------------------------------------------
# src/methylbert/config.py (verbatim, trimmed to the config class only)
# ---------------------------------------------------------------------------
class MethylBERTConfig(BertConfig):
    loss = "bce"
    num_labels = -1


# ---------------------------------------------------------------------------
# src/methylbert/function.py (verbatim: FocalLoss)
# ---------------------------------------------------------------------------
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    This code is from: https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


class FocalLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(input, target, reduction=self.reduction)


# ---------------------------------------------------------------------------
# src/methylbert/network.py (verbatim: MethylBertEmbeddedDMR)
# ---------------------------------------------------------------------------
class MethylBertEmbeddedDMR(BertPreTrainedModel):
    config_class = MethylBERTConfig
    base_model_prefix = "methylbert"

    def __init__(self, config, seq_len=150):
        super().__init__(config)
        self.num_labels = config.num_labels

        if config.loss not in ["bce", "focal_bce"]:
            raise ValueError(f"loss must be bce or focal_bce. {config.loss} is given.")

        self.loss = config.loss
        self.classification_loss_fct = self._setup_loss(self.loss)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.read_classifier = nn.Sequential(
            nn.Linear((config.hidden_size + 1) * (seq_len + 1), seq_len + 1),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.LayerNorm(seq_len + 1, eps=config.layer_norm_eps),
            nn.Linear(seq_len + 1, 2),
        )

        self.seq_len = seq_len

        self.dmr_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_labels, embedding_dim=seq_len + 1),
        )

        self.init_weights()

    def _setup_loss(self, loss):
        if loss == "bce":
            return nn.CrossEntropyLoss()
        elif loss == "focal_bce":
            return FocalLoss()

    def forward(
        self,
        step,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        ctype_label=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # DMR info
        encoded_dmr = self.dmr_encoder(labels.view(-1))

        sequence_output = torch.cat((sequence_output, encoded_dmr.unsqueeze(-1)), axis=-1)

        ctype_logits = self.read_classifier(
            sequence_output.view(-1, (self.seq_len + 1) * (self.config.hidden_size + 1))
        )

        loss = self.classification_loss_fct(
            ctype_logits.view(-1, 2),
            F.one_hot(ctype_label, num_classes=2).to(torch.float32).view(-1, 2),
        )
        ctype_logits = ctype_logits.softmax(dim=1)

        return {
            "loss": loss,
            "dmr_logits": sequence_output,
            "classification_logits": ctype_logits,
        }


# ---------------------------------------------------------------------------
# Tiny-scale staging wrapper (torchlens capture needs a single forward()
# call with a concrete example input; the real forward() takes a `step`
# scheduling arg plus several BERT inputs plus DMR labels).
# ---------------------------------------------------------------------------
_SEQ_LEN = 8  # small read length (real default is 150)
_NUM_DMR_LABELS = 2


class MethylBertTraceWrapper(nn.Module):
    """Wraps MethylBertEmbeddedDMR.forward with a fixed example input so
    TorchLens can capture a plain single-tensor-in forward pass."""

    def __init__(self, model: MethylBertEmbeddedDMR):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch = input_ids.shape[0]
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        ctype_label = torch.zeros(batch, dtype=torch.long)
        labels = torch.zeros(batch, dtype=torch.long)
        out = self.model(
            step=0,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            ctype_label=ctype_label,
        )
        return out["classification_logits"]


def build_methylbert() -> MethylBertTraceWrapper:
    config = MethylBERTConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=_SEQ_LEN + 2,
        layer_norm_eps=1e-12,
        loss="bce",
    )
    config.num_labels = _NUM_DMR_LABELS
    model = MethylBertEmbeddedDMR(config, seq_len=_SEQ_LEN + 1)
    model.eval()
    return MethylBertTraceWrapper(model)


def example_input_methylbert() -> torch.Tensor:
    return torch.randint(0, 32, (2, _SEQ_LEN + 2))


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "MethylBERT",
        "build_methylbert",
        "example_input_methylbert",
        2025,
        "vendored-pytorch",
    ),
]
