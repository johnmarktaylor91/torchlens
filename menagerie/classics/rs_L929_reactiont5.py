# SOURCE: vendored from https://github.com/sagawatatsuya/ReactionT5 @ main
# File: models.py (ReactionT5Yield)
#
# ReactionT5Yield wraps a pretrained T5ForConditionalGeneration (used only as
# an encoder-decoder backbone, base_model="sagawa/CompoundT5" in the real
# repo) with a custom fusion head: it runs the T5 encoder over the input
# reaction SMILES, feeds the encoder hidden states through the T5 decoder
# seeded with only the decoder-start token, then fuses the decoder's last
# hidden state (fc1) with the encoder's CLS-style first-token embedding
# (fc2) through a 2-layer regression head (fc3-fc5) to predict a scalar
# reaction yield. This decoder-seeded-fusion head is architecturally novel
# relative to stock T5ForConditionalGeneration (imports/module paths
# adjusted only; class body is unmodified from the real repo).

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, T5Config, T5ForConditionalGeneration

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from models.py ---
class ReactionT5Yield(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                self.cfg.pretrained_model_name_or_path, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)
        if pretrained:
            if "t5" in self.cfg.model:
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.cfg.pretrained_model_name_or_path
                )
            else:
                self.model = AutoModel.from_pretrained(self.cfg.pretrained_model_name_or_path)
        else:
            if "t5" in self.cfg.model:
                self.model = T5ForConditionalGeneration(self.config)
            else:
                self.model = AutoModel.from_config(self.config)
        self.model.resize_token_embeddings(len(self.cfg.tokenizer))
        self.fc_dropout1 = nn.Dropout(self.cfg.fc_dropout)
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
        self.fc_dropout2 = nn.Dropout(self.cfg.fc_dropout)

        self.fc2 = nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
        self.fc3 = nn.Linear(self.config.hidden_size // 2 * 2, self.config.hidden_size)
        self.fc4 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc5 = nn.Linear(self.config.hidden_size, 1)

        self._init_weights(self.fc1)
        self._init_weights(self.fc2)
        self._init_weights(self.fc3)
        self._init_weights(self.fc4)
        self._init_weights(self.fc5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        encoder_outputs = self.model.encoder(**inputs)
        encoder_hidden_states = encoder_outputs[0]
        outputs = self.model.decoder(
            input_ids=torch.full(
                (inputs["input_ids"].size(0), 1),
                self.config.decoder_start_token_id,
                dtype=torch.long,
                device=self.cfg.device,
            ),
            encoder_hidden_states=encoder_hidden_states,
        )
        last_hidden_states = outputs[0]
        output1 = self.fc1(self.fc_dropout1(last_hidden_states).view(-1, self.config.hidden_size))
        output2 = self.fc2(encoder_hidden_states[:, 0, :].view(-1, self.config.hidden_size))
        output = self.fc3(self.fc_dropout2(torch.hstack((output1, output2))))
        output = self.fc4(output)
        output = self.fc5(output)
        return output


# --- staging harness ---
class _TinyTokenizer:
    """Minimal stand-in exposing only __len__, matching real usage
    (`self.cfg.tokenizer`, only ever used via `len(self.cfg.tokenizer)`)."""

    def __init__(self, vocab_size):
        self._vocab_size = vocab_size

    def __len__(self):
        return self._vocab_size


class _TinyCfg:
    def __init__(self):
        self.pretrained_model_name_or_path = None
        self.model = "t5"
        self.tokenizer = _TinyTokenizer(96)
        self.fc_dropout = 0.1
        self.device = torch.device("cpu")


def build_reactiont5_yield():
    cfg = _TinyCfg()
    tiny_config = T5Config(
        vocab_size=96,
        d_model=32,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_kv=16,
        decoder_start_token_id=0,
        output_hidden_states=True,
    )

    model = ReactionT5Yield.__new__(ReactionT5Yield)
    nn.Module.__init__(model)
    model.cfg = cfg
    model.config = tiny_config
    model.model = T5ForConditionalGeneration(tiny_config)
    model.model.resize_token_embeddings(len(cfg.tokenizer))
    model.fc_dropout1 = nn.Dropout(cfg.fc_dropout)
    model.fc1 = nn.Linear(model.config.hidden_size, model.config.hidden_size // 2)
    model.fc_dropout2 = nn.Dropout(cfg.fc_dropout)
    model.fc2 = nn.Linear(model.config.hidden_size, model.config.hidden_size // 2)
    model.fc3 = nn.Linear(model.config.hidden_size // 2 * 2, model.config.hidden_size)
    model.fc4 = nn.Linear(model.config.hidden_size, model.config.hidden_size)
    model.fc5 = nn.Linear(model.config.hidden_size, 1)
    for lyr in (model.fc1, model.fc2, model.fc3, model.fc4, model.fc5):
        model._init_weights(lyr)
    return model


def example_input_reactiont5_yield():
    input_ids = torch.randint(0, 96, (2, 6), dtype=torch.long)
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    return ({"input_ids": input_ids, "attention_mask": attention_mask},)


MENAGERIE_ENTRIES = [
    (
        "ReactionT5_Yield",
        "build_reactiont5_yield",
        "example_input_reactiont5_yield",
        2023,
        "vendored-pytorch",
    ),
]
