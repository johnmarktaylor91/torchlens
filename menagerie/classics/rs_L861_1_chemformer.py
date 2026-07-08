# SOURCE: vendored from MolecularAI/Chemformer @ main
# https://github.com/MolecularAI/Chemformer/blob/main/molbart/models/base_transformer.py
# https://github.com/MolecularAI/Chemformer/blob/main/molbart/models/transformer_models.py
# https://github.com/MolecularAI/Chemformer/blob/main/molbart/models/util.py
#
# Chemformer (AstraZeneca MolecularAI): a BART-style pre-norm Transformer seq2seq model
# operating on SMILES tokens, used for retrosynthesis / forward-synthesis prediction and
# reaction-condition ("ChemformerMapper" fine-tuning) tasks. This file inlines the real
# `_AbsTransformerModel` base class (base_transformer.py), the `BARTModel` encoder-decoder
# subclass (transformer_models.py), and the `PreNormEncoderLayer` / `PreNormDecoderLayer` /
# `FuncLR` helpers (util.py) verbatim. Only changes: `pytorch_lightning.LightningModule`
# training-loop methods (`training_step`, `validation_step`, `configure_optimizers`, etc.)
# are dropped since they are training-orchestration, not architecture -- the module still
# subclasses `nn.Module` (not `pl.LightningModule`, to avoid a hard `pytorch_lightning`
# runtime dependency for a pure architecture trace) and keeps every real layer: token
# embedding + sinusoidal positional embedding, pre-norm `nn.TransformerEncoder` /
# `nn.TransformerDecoder` stacks (using PyTorch's own `TransformerEncoderLayer` /
# `TransformerDecoderLayer` with pre-norm `forward` overridden exactly as upstream), the
# causal `_generate_square_subsequent_mask`, and the final `token_fc` + `log_softmax` head.
# No architectural change.
import math

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- util.py (verbatim pre-norm encoder/decoder layers) ---
# NOTE (torch-version compat, no architectural change): newer torch's
# nn.TransformerEncoder/TransformerDecoder containers pass an `is_causal` /
# `tgt_is_causal` / `memory_is_causal` kwarg down to the per-layer forward(); the
# upstream repo's signature predates that kwarg. Accept and ignore it via **_kwargs so
# the exact real forward-pass math below is unchanged.
class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, **_kwargs):
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


class PreNormDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        **_kwargs,
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


# --- base_transformer.py (verbatim architecture-relevant members of _AbsTransformerModel;
# pytorch_lightning training/validation/test hooks and optimizer scheduling dropped since
# they orchestrate training, not the forward architecture) ---
class _AbsTransformerModel(nn.Module):
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
        """Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000**encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[: self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz, device="cpu"):
        """
        Method copied from Pytorch nn.Transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode
        """

        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


# --- transformer_models.py (verbatim BARTModel encoder-decoder architecture) ---
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

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.

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


def build_chemformer():
    # Tiny menagerie-scale BARTModel config: small vocab, small d_model, 2 layers, 2 heads.
    # decode_sampler=None since forward() never touches self.sampler (only sample_molecules,
    # a generation-time helper not exercised by a plain forward trace).
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
        num_steps=1,
        max_seq_len=32,
    )


def example_input_chemformer():
    # Matches the real BARTModel.forward() input contract: (seq_len, batch_size) token-id
    # tensors plus matching (seq_len, batch_size) boolean pad masks for encoder and decoder.
    torch.manual_seed(0)
    src_len, tgt_len, batch_size, vocab_size = 10, 8, 2, 64
    encoder_input = torch.randint(1, vocab_size, (src_len, batch_size))
    decoder_input = torch.randint(1, vocab_size, (tgt_len, batch_size))
    encoder_pad_mask = torch.zeros(src_len, batch_size, dtype=torch.bool)
    decoder_pad_mask = torch.zeros(tgt_len, batch_size, dtype=torch.bool)
    x = {
        "encoder_input": encoder_input,
        "encoder_pad_mask": encoder_pad_mask,
        "decoder_input": decoder_input,
        "decoder_pad_mask": decoder_pad_mask,
    }
    return (x,)


MENAGERIE_ENTRIES = [
    (
        "ChemformerMapper",
        "build_chemformer",
        "example_input_chemformer",
        2022,
        MENAGERIE_ZOO,
    ),
]
