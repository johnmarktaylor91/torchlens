# SOURCE: vendored from https://github.com/ChangwenXu98/TransPolymer @ master (8399d4816ce7)
# (Downstream.py's DownstreamRegression)
#
# TransPolymer (Xu, Chen, Vlachos, "TransPolymer: a Transformer-based
# language model for polymer property predictions", npj Computational
# Materials 2023). A RoBERTa masked-LM encoder (pretrained on PSMILES
# polymer-string tokens via the repo's own PolymerSmilesTokenizer) with a
# custom downstream regression head: the [CLS]-token's last_hidden_state is
# projected through Dropout -> Linear(hidden, hidden) -> SiLU -> Linear(hidden, 1).
# This is a genuine architectural addition on top of RobertaModel (not a
# rung-1 case): the paper's contribution is the regression head + polymer
# tokenization scheme layered on the transformer, so the real class is
# vendored here as a module rather than emitted as a bare RobertaModel
# recipe. The repo's `DownstreamRegression.__init__` closes over a
# module-level `PretrainedModel`/`tokenizer` (set up in the script's
# `__main__` block from a yaml config + checkpoint path); that config/
# checkpoint plumbing is training-script wiring, not architecture, so here
# `PretrainedModel`/`vocab_size` are passed explicitly as constructor
# arguments instead of read from globals -- the Regressor head and forward
# pass are untouched.

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel

MENAGERIE_ZOO = "vendored-pytorch"


class DownstreamRegression(nn.Module):
    def __init__(self, pretrained_model: RobertaModel, vocab_size: int, drop_rate: float = 0.1):
        super(DownstreamRegression, self).__init__()
        self.PretrainedModel = pretrained_model
        self.PretrainedModel.resize_token_embeddings(vocab_size)

        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(
                self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size
            ),
            nn.SiLU(),
            nn.Linear(self.PretrainedModel.config.hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.Regressor(logits)
        return output


def build_transpolymer():
    torch.manual_seed(0)
    # Real repo "no pretrain" config path (Downstream.py __main__, else
    # branch): vocab_size=50265, max_position_embeddings=514,
    # num_attention_heads=12, num_hidden_layers=6. Shrunk to menagerie-recipe
    # scale here; RoBERTa architecture itself is untouched.
    config = RobertaConfig(
        vocab_size=100,
        hidden_size=32,
        max_position_embeddings=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    pretrained = RobertaModel(config=config)
    model = DownstreamRegression(pretrained_model=pretrained, vocab_size=100, drop_rate=0.1)
    model.eval()
    return model


def example_input_transpolymer():
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return (input_ids, attention_mask)


MENAGERIE_ENTRIES = [
    (
        "TransPolymer (RoBERTa + polymer-property regression head)",
        "build_transpolymer",
        "example_input_transpolymer",
        2023,
        MENAGERIE_ZOO,
    ),
]
