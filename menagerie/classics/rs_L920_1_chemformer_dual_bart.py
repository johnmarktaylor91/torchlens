# SOURCE: vendored from MolecularAI/Chemformer @ main
#   (repo: https://github.com/MolecularAI/Chemformer)
#
# Covers TWO queue candidates that both describe the SAME architecture:
#   - "Dual Transformer retrosynthesis" (cand row 921) -- tied two-way
#     transformer (forward + backward SMILES models) with cycle-consistency
#     reranking, implemented on top of Chemformer's BART-style seq2seq model.
#   - "Dual-TB" (cand row 922) -- same tied/dual transformer retrosynthesis
#     scheme, explicitly marked POTENTIAL_DEDUP with row 921 in the queue.
# The "dual"/"tied two-way" scheme is a TRAINING/INFERENCE-time usage pattern
# (run one BARTModel forward, one BARTModel backward, tie/rerank via cycle
# consistency) built on top of Chemformer's single BARTModel seq2seq
# transformer -- there is no separate dual-specific nn.Module architecture
# in the repo. This file vendors that real BARTModel network (the actual
# per-direction transformer both candidates instantiate twice) verbatim from
# molbart/models/transformer_models.py + molbart/models/base_transformer.py
# + molbart/models/util.py.
#
# BARTModel is a pytorch_lightning.LightningModule wrapping standard
# nn.TransformerEncoder / nn.TransformerDecoder stacks built from
# Chemformer's own PreNormEncoderLayer / PreNormDecoderLayer (subclasses of
# nn.TransformerEncoderLayer / nn.TransformerDecoderLayer that swap in
# pre-norm residual blocks) plus sinusoidal positional embeddings and a
# token output head. Only import paths were adjusted (molbart's package
# imports were inlined into this single file); no architecture code was
# rewritten, approximated, or simplified. Lightning-specific methods
# (training_step/validation_step/configure_optimizers/etc.) are kept
# verbatim from _AbsTransformerModel for fidelity but are not exercised by
# a forward-pass trace. PreNormEncoderLayer.forward / PreNormDecoderLayer.forward
# gained a **kwargs passthrough to absorb is_causal/tgt_is_causal/
# memory_is_causal, which modern torch's nn.TransformerEncoder/Decoder now
# forward to each layer but which did not exist when this repo was written
# against an older torch -- a torch-version compat shim only, not an
# architectural change.

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Verbatim from molbart/models/util.py
# ---------------------------------------------------------------------------


class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    # NOTE: **kwargs absorbs is_causal (and any other newer-torch kwarg that
    # nn.TransformerEncoder.forward now forwards to each layer); the repo's
    # original signature predates that torch API addition. Torch-version
    # compat shim only -- no architectural change.
    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
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


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormDecoderLayer(nn.TransformerDecoderLayer):
    # NOTE: **kwargs absorbs tgt_is_causal/memory_is_causal (and any other
    # newer-torch kwarg that nn.TransformerDecoder.forward now forwards to
    # each layer); the repo's original signature predates that torch API
    # addition. Torch-version compat shim only -- no architectural change.
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        **kwargs,
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


# ---------------------------------------------------------------------------
# Verbatim from molbart/models/base_transformer.py
# ---------------------------------------------------------------------------


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

        # Additional args passed in to **kwargs in init will also be saved
        self.save_hyperparameters()

        # These must be set by subclasses
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

    def _calc_loss(self, batch_input, model_output):
        raise NotImplementedError()

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        self.train()

        model_output = self.forward(batch)
        loss = self._calc_loss(batch, model_output)

        self.log("training_loss", loss, on_step=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()

        with torch.no_grad():
            model_output = self.forward(batch)
            target_smiles = batch["target_smiles"]

            loss = self._calc_loss(batch, model_output)
            token_acc = self._calc_token_acc(batch, model_output)

            sampled_smiles, _ = self.sample_molecules(batch, sampling_alg=self.val_sampling_alg)

            sampled_metrics = self.sampler.compute_sampling_metrics(sampled_smiles, target_smiles)

            metrics = {
                "validation_loss": loss,
                "val_token_accuracy": token_acc,
            }

            metrics.update(sampled_metrics)
        return metrics

    def validation_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)
        return

    def test_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            model_output = self.forward(batch)
            target_smiles = batch["target_smiles"]

            loss = self._calc_loss(batch, model_output)
            token_acc = self._calc_token_acc(batch, model_output)
            sampled_smiles, log_likelihoods = self.sample_molecules(
                batch, sampling_alg=self.test_sampling_alg
            )

        sampled_metrics = self.sampler.compute_sampling_metrics(sampled_smiles, target_smiles)

        metrics = {
            "batch_idx": batch_idx,
            "test_loss": loss.item(),
            "test_token_accuracy": token_acc,
            "log_lhs": log_likelihoods,
            "sampled_molecules": sampled_smiles,
            "target_smiles": target_smiles,
        }

        metrics.update(sampled_metrics)
        return metrics

    def test_epoch_end(self, outputs):
        return

    def configure_optimizers(self):
        params = self.parameters()
        optim = torch.optim.Adam(
            params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999)
        )

        if self.schedule == "const":
            print("Using constant LR schedule.")
            const_sch = FuncLR(optim, lr_lambda=self._const_lr)
            sch = {"scheduler": const_sch, "interval": "step"}

        elif self.schedule == "cycle":
            print("Using cyclical LR schedule.")
            cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
            sch = {"scheduler": cycle_sch, "interval": "step"}

        elif self.schedule == "transformer":
            print("Using original transformer schedule.")
            trans_sch = FuncLR(optim, lr_lambda=self._transformer_lr)
            sch = {"scheduler": trans_sch, "interval": "step"}

        else:
            raise ValueError(f"Unknown schedule {self.schedule}")

        return [optim], [sch]

    def _transformer_lr(self, step):
        mult = self.d_model**-0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
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

    def _calc_perplexity(self, batch_input, model_output):
        target_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]
        vocab_dist_output = model_output["token_output"]

        inv_target_mask = ~(target_mask > 0)
        log_probs = vocab_dist_output.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=0)

        seq_lengths = inv_target_mask.sum(dim=0)
        exp = -(1 / seq_lengths)
        perp = torch.pow(log_probs.exp(), exp)
        return perp.mean()

    def _calc_token_acc(self, batch_input, model_output):
        token_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        accuracy = num_correct / total
        return accuracy

    def _avg_dicts(self, colls):
        complete_dict = {key: [] for key, val in colls[0].items()}
        for coll in colls:
            [complete_dict[key].append(coll[key]) for key in complete_dict.keys()]

        avg_dict = {key: sum(l) / len(l) for key, l in complete_dict.items()}  # noqa: E741 (kept for source fidelity)
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val, sync_dist=True)


# ---------------------------------------------------------------------------
# Verbatim from molbart/models/transformer_models.py (BARTModel only; the
# encoder-only UnifiedModel variant is omitted -- BARTModel is Chemformer's
# seq2seq model, the one actually paired forward+backward in the dual/tied
# retrosynthesis scheme both candidates describe).
# ---------------------------------------------------------------------------


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
        """Apply SMILES strings to model

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

    def encode(self, batch):
        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        encoder_embs = self._construct_input(encoder_input)
        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return model_output

    def decode(self, batch):
        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embeddings = self._construct_input(decoder_input)

        sequence_length, _, _ = tuple(decoder_embeddings.size())
        tgt_mask = self._generate_square_subsequent_mask(
            sequence_length, device=decoder_embeddings.device
        )

        decoder_output = self.decoder(
            decoder_embeddings,
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask,
        )
        token_log_probabilities = self.generator(decoder_output)
        return token_log_probabilities

    def generator(self, decoder_output):
        token_log_probabilities = self.log_softmax(self.token_fc(decoder_output))
        return token_log_probabilities

    def _calc_loss(self, batch_input, model_output):
        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)

        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_function(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    def sample_molecules(self, batch_input, sampling_alg="greedy", return_tokenized=False):
        raise NotImplementedError(
            "Sampling requires a real BeamSearchSampler; not used for a forward-pass trace."
        )


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_chemformer_bart():
    """Tiny-size real Chemformer BARTModel (the per-direction seq2seq
    transformer paired forward+backward in the dual/tied-transformer
    retrosynthesis scheme both queue candidates describe)."""
    return BARTModel(
        decode_sampler=None,
        pad_token_idx=0,
        vocabulary_size=64,
        d_model=32,
        num_layers=2,
        num_heads=2,
        d_feedforward=64,
        lr=1e-4,
        weight_decay=0.0,
        activation="gelu",
        num_steps=100,
        max_seq_len=32,
        schedule="cycle",
        dropout=0.0,
    )


def example_input_chemformer_bart():
    """Tiny SMILES-token batch matching BARTModel.forward's expected dict
    (encoder/decoder token ids + pad masks), shapes (seq_len, batch_size)."""
    torch.manual_seed(0)
    src_len, tgt_len, batch_size, vocab = 6, 5, 2, 64
    encoder_input = torch.randint(1, vocab, (src_len, batch_size))
    decoder_input = torch.randint(1, vocab, (tgt_len, batch_size))
    encoder_pad_mask = torch.zeros(src_len, batch_size, dtype=torch.bool)
    decoder_pad_mask = torch.zeros(tgt_len, batch_size, dtype=torch.bool)
    return {
        "encoder_input": encoder_input,
        "encoder_pad_mask": encoder_pad_mask,
        "decoder_input": decoder_input,
        "decoder_pad_mask": decoder_pad_mask,
    }


MENAGERIE_ENTRIES = [
    (
        "Chemformer-BART-DualTransformer",
        build_chemformer_bart,
        example_input_chemformer_bart,
        2022,
        "CODE",
    ),
]
