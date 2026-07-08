# SOURCE: vendored from MolecularAI/Chemformer @ main
# (molbart/models/base_transformer.py::_AbsTransformerModel +
#  molbart/models/transformer_models.py::BARTModel +
#  molbart/models/util.py::PreNormEncoderLayer/PreNormDecoderLayer/FuncLR)
#
# Chemformer ("Chemformer: a pre-trained transformer for computational chemistry",
# Irwin, Dimitriadis, He, Bjerrum; Mach. Learn.: Sci. Technol. 2022). AstraZeneca
# MolecularAI. The model IS a real BART-style (pre-norm) Transformer encoder-decoder
# operating on tokenized SMILES strings -- the paper's contribution is the
# pre-training objective / SMILES tokenization / chemistry usage, not a novel
# architecture beyond BART with pre-norm sublayers, so this is vendored (not
# reimplemented) verbatim from the real repo.
#
# Only the ARCHITECTURE classes are vendored here (BARTModel + its base class +
# the pre-norm transformer layer helpers); the repo's data loading, tokenizer,
# and beam-search sampler machinery are irrelevant to tracing forward().
#
# Minimal signature-only compat fix (NOT an architecture change): the repo's
# `PreNormEncoderLayer.forward` / `PreNormDecoderLayer.forward` predate torch's
# `is_causal` / `tgt_is_causal` / `memory_is_causal` kwargs that
# `nn.TransformerEncoder`/`nn.TransformerDecoder` now pass through to sublayers
# on modern torch (>=2.x). Added those kwargs (accepted, unused, matching
# torch's own `nn.TransformerEncoderLayer.forward` / `nn.TransformerDecoderLayer.forward`
# base-class signatures) so the real code runs unmodified on current torch.
# `decode_sampler` is only used by `sample_molecules()` (beam/greedy decoding for
# inference), not by `forward()`, so it is passed as None here.

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

MENAGERIE_ZOO = "vendored-pytorch"


# molbart/models/util.py
class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


# molbart/models/util.py -- Use Pytorch implementation but with 'pre-norm' style
# layer normalisation. `is_causal` kwarg added for modern-torch signature compat
# (see module header); unused, matches torch's own base-class default.
class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self attention block
        att = self.norm1(src)
        att = self.self_attn(
            att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        att = src + self.dropout1(att)

        # Feedforward block
        out = self.norm2(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout2(out)
        return out


# molbart/models/util.py -- pre-norm decoder layer; `tgt_is_causal`/`memory_is_causal`
# kwargs added for modern-torch signature compat (see module header); unused.
class PreNormDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        # Self attention block
        query = self.norm1(tgt)
        query = self.self_attn(
            query,
            query,
            query,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        query = tgt + self.dropout1(query)

        # Context attention block
        att = self.norm2(query)
        att = self.multihead_attn(
            att,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        att = query + self.dropout2(att)

        # Feedforward block
        out = self.norm3(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout3(out)
        return out


# molbart/models/base_transformer.py::_AbsTransformerModel
class _AbsTransformerModel(pl.LightningModule):
    def __init__(
        self,
        pad_token_idx,
        vocabulary_size,
        d_model,
        num_layers,
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule,
        warm_up_steps,
        dropout=0.1,
        num_beams=10,
        **kwargs,
    ):
        super().__init__()

        self.pad_token_idx = pad_token_idx
        self.vocabulary_size = vocabulary_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation = activation
        self.num_steps = num_steps
        self.max_seq_len = max_seq_len
        self.schedule = schedule
        self.warm_up_steps = warm_up_steps
        self.dropout = dropout

        if self.schedule == "transformer":
            assert warm_up_steps is not None, (
                "A value for warm_up_steps is required for transformer LR schedule"
            )

        self.save_hyperparameters()

        self.sampler = None
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = num_beams
        self.n_unique_beams = num_beams

        self.emb = nn.Embedding(vocabulary_size, d_model, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", self._positional_embs())

    def forward(self, x):
        raise NotImplementedError()

    def configure_optimizers(self):
        params = self.parameters()
        optim = torch.optim.Adam(
            params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999)
        )

        if self.schedule == "const":
            const_sch = FuncLR(optim, lr_lambda=self._const_lr)
            sch = {"scheduler": const_sch, "interval": "step"}
        elif self.schedule == "cycle":
            from torch.optim.lr_scheduler import OneCycleLR

            cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
            sch = {"scheduler": cycle_sch, "interval": "step"}
        elif self.schedule == "transformer":
            trans_sch = FuncLR(optim, lr_lambda=self._transformer_lr)
            sch = {"scheduler": trans_sch, "interval": "step"}
        else:
            raise ValueError(f"Unknown schedule {self.schedule}")

        return [optim], [sch]

    def _transformer_lr(self, step):
        mult = self.d_model**-0.5
        step = 1 if step == 0 else step
        lr = min(step**-0.5, step * (self.warm_up_steps**-1.5))
        return self.lr * mult * lr

    def _const_lr(self, step):
        if self.warm_up_steps is not None and step < self.warm_up_steps:
            return (self.lr / self.warm_up_steps) * step
        return self.lr

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    def _positional_embs(self):
        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000**encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[: self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz, device="cpu"):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


# molbart/models/transformer_models.py::BARTModel
class BARTModel(_AbsTransformerModel):
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocabulary_size,
        d_model,
        num_layers,
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule="cycle",
        warm_up_steps=None,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(
            pad_token_idx,
            vocabulary_size,
            d_model,
            num_layers,
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            schedule,
            warm_up_steps,
            dropout,
            **kwargs,
        )

        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"

        self.encoder = nn.TransformerEncoder(
            PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.decoder = nn.TransformerDecoder(
            PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.loss_function = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)

        self.token_fc = nn.Linear(d_model, vocabulary_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        """Apply SMILES strings to model.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """
        encoder_input = x["encoder_input"]
        decoder_input = x["decoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)
        decoder_pad_mask = x["decoder_pad_mask"].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)
        decoder_embeddings = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embeddings.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=encoder_embs.device)

        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        model_output = self.decoder(
            decoder_embeddings,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=encoder_pad_mask.clone(),
        )

        token_output = self.token_fc(model_output)

        output = {"model_output": model_output, "token_output": token_output}

        return output

    def _calc_loss(self, batch_input, model_output):
        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]
        return self._calc_mask_loss(token_output, tokens, pad_mask)

    def _calc_mask_loss(self, token_output, target, target_mask):
        seq_len, batch_size = tuple(target.size())
        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_function(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))
        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens
        return loss


_D_MODEL = 32
_VOCAB_SIZE = 64
_MAX_SEQ_LEN = 32
_SEQ_LEN = 8
_BATCH_SIZE = 2


def build_chemformer():
    return BARTModel(
        decode_sampler=None,
        pad_token_idx=0,
        vocabulary_size=_VOCAB_SIZE,
        d_model=_D_MODEL,
        num_layers=2,
        num_heads=2,
        d_feedforward=64,
        lr=1e-4,
        weight_decay=0.0,
        activation="gelu",
        num_steps=1,
        max_seq_len=_MAX_SEQ_LEN,
    )


def example_input_chemformer():
    encoder_input = torch.randint(1, _VOCAB_SIZE, (_SEQ_LEN, _BATCH_SIZE))
    decoder_input = torch.randint(1, _VOCAB_SIZE, (_SEQ_LEN, _BATCH_SIZE))
    pad_mask = torch.zeros(_SEQ_LEN, _BATCH_SIZE, dtype=torch.bool)
    return (
        {
            "encoder_input": encoder_input,
            "encoder_pad_mask": pad_mask,
            "decoder_input": decoder_input,
            "decoder_pad_mask": pad_mask,
        },
    )


MENAGERIE_ENTRIES = [
    ("Chemformer", "build_chemformer", "example_input_chemformer", 2022, "vendored-pytorch"),
]
