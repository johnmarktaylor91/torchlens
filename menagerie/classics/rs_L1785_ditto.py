# SOURCE: vendored from megagonlabs/ditto @ master (ditto_light/ditto.py)
"""Ditto: a Transformer-based entity matching (EM) model.

Vendored from https://github.com/megagonlabs/ditto (ditto_light/ditto.py).
Ditto wraps a pretrained transformer encoder (RoBERTa or DistilBERT) with a
2-way linear classification head over the [CLS] token to decide whether two
serialized entity records refer to the same real-world entity. The transformer
backbone is unmodified; the only architectural addition is the final ``fc``
linear layer, so this is vendored (not a bare rung-1 recipe).

For the menagerie we swap the real pretrained-checkpoint loading
(``AutoModel.from_pretrained(lm)``) for a tiny randomly-initialized
``DistilBertModel`` (same real ``transformers`` class used by the real model
when ``lm='distilbert'``) so the module builds and traces offline.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, DistilBertConfig, DistilBertModel

MENAGERIE_ZOO = "vendored-pytorch"

lm_mp = {"roberta": "roberta-base", "distilbert": "distilbert-base-uncased"}


class DittoModel(nn.Module):
    """A baseline model for EM. (verbatim from ditto_light/ditto.py, minus
    the pretrained-checkpoint download, which is swapped for a tiny random
    init backbone below in build_ditto()).
    """

    def __init__(self, device="cpu", lm="roberta", alpha_aug=0.8, bert=None):
        super().__init__()
        if bert is not None:
            # menagerie hook: caller supplies a tiny randomly-initialized
            # transformer backbone instead of downloading a pretrained one.
            self.bert = bert
        elif lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device)  # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device)  # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size]  # (batch_size, emb_size)
            enc2 = enc[batch_size:]  # (batch_size, emb_size)

            aug_lam = 0.5  # menagerie: fixed value, real code uses np.random.beta
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc)  # .squeeze() # .sigmoid()


def build_ditto():
    tiny_cfg = DistilBertConfig(
        vocab_size=256,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
    )
    backbone = DistilBertModel(tiny_cfg)
    return DittoModel(device="cpu", lm="distilbert", bert=backbone)


def example_input_ditto():
    return torch.randint(0, 256, (2, 16), dtype=torch.long)


MENAGERIE_ENTRIES = [
    ("Ditto", "build_ditto", "example_input_ditto", 2020, "SOURCE_AVAILABLE"),
]
