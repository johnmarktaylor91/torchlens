# SOURCE: vendored from Mathux/TEMOS @ master (temos/model/temos.py,
# temos/model/motionencoder/actor.py, temos/model/motiondecoder/actor.py,
# temos/model/textencoder/distilbert.py, temos/model/textencoder/distilbert_actor.py,
# temos/model/utils/positional_encoding.py, temos/data/tools/tensors.py)
# https://github.com/Mathux/TEMOS -- "TEMOS: Generating diverse human motions
# from textual descriptions" (Petrovich, Black, Varol, ECCV 2022 Oral). The
# `ActorAgnosticEncoder` / `ActorAgnosticDecoder` (motion VAE, transcribed
# verbatim from temos/model/motionencoder/actor.py and
# temos/model/motiondecoder/actor.py) and `DistilbertActorAgnosticEncoder` /
# `DistilbertEncoderBase` (text VAE encoder, transcribed verbatim from
# temos/model/textencoder/distilbert.py and distilbert_actor.py) are TEMOS's
# real architecture: a text-conditioned transformer VAE whose text branch
# projects a DistilBERT encoding into a shared latent space and whose motion
# branch is a transformer encoder/decoder pair operating on human-pose feature
# sequences. `PositionalEncoding` and `lengths_to_mask` are transcribed
# verbatim from temos/model/utils/positional_encoding.py and
# temos/data/tools/tensors.py. The top-level `TEMOS` module below reproduces
# TEMOS.text_to_motion_forward from temos/model/temos.py as a plain
# nn.Module -- only the PyTorch Lightning training scaffolding (BaseModel,
# losses/metrics/optimizer wiring, Hydra instantiate() config plumbing) was
# dropped, since that is training infrastructure, not architecture. The
# original `DistilbertEncoderBase` downloads a pretrained DistilBERT
# checkpoint via `AutoTokenizer.from_pretrained` / `AutoModel.from_pretrained`
# at train time; for a self-contained trace we instead construct a
# random-init `DistilBertConfig`/`DistilBertModel` at tiny size (same
# architecture class, no network fetch) fed by a minimal fixed-vocab
# tokenizer stub that reproduces `tokenizer(texts, return_tensors="pt",
# padding=True)`'s (input_ids, attention_mask) contract without hitting the
# HF hub.
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution
from transformers import DistilBertConfig, DistilBertModel

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from temos/data/tools/tensors.py ----
def lengths_to_mask(lengths, device):
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# ---- verbatim from temos/model/utils/positional_encoding.py ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


# ---- verbatim from temos/model/motionencoder/actor.py ----
class ActorAgnosticEncoder(nn.Module):
    def __init__(
        self,
        nfeats,
        vae,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        **kwargs,
    ):
        super().__init__()
        self.vae = vae

        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)

        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=num_layers)

    def forward(self, features: Tensor, lengths=None):
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        x = self.skel_embedding(x)
        x = x.permute(1, 0, 2)

        if self.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

            xseq = torch.cat((emb_token[None], x), 0)

            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            dist = torch.distributions.Normal(mu, std)
            return dist
        else:
            return final[0]


# ---- verbatim from temos/model/motiondecoder/actor.py ----
class ActorAgnosticDecoder(nn.Module):
    def __init__(
        self,
        nfeats,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        **kwargs,
    ):
        super().__init__()
        self.nfeats = nfeats

        output_feats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )

        self.seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer, num_layers=num_layers)

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z: Tensor, lengths):
        mask = lengths_to_mask(lengths, z.device)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape

        z = z[None]

        time_queries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        output = self.seqTransDecoder(tgt=time_queries, memory=z, tgt_key_padding_mask=~mask)

        output = self.final_layer(output)
        output[~mask.T] = 0
        feats = output.permute(1, 0, 2)
        return feats


# ---- adapted from temos/model/textencoder/distilbert.py: real DistilBertModel
# class, random-init tiny config instead of AutoModel.from_pretrained(hub id) ----
class _TinyFixedVocabTokenizer:
    """Minimal stand-in for the real DistilBERT AutoTokenizer's
    `tokenizer(texts, return_tensors="pt", padding=True)` call contract
    (batched whitespace tokenization -> padded input_ids + attention_mask),
    scoped to a tiny fixed vocabulary so DistilbertEncoderBase can run
    end-to-end without a network fetch."""

    def __init__(self, vocab_size, pad_token_id=0, cls_token_id=1, sep_token_id=2):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def __call__(self, texts, return_tensors="pt", padding=True):
        seqs = []
        for text in texts:
            words = text.split()
            ids = [self.cls_token_id]
            for w in words:
                ids.append(2 + (abs(hash(w)) % (self.vocab_size - 3)) + 1)
            ids.append(self.sep_token_id)
            seqs.append(ids)
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, : len(s)] = 1

        class _Batch(dict):
            def to(self, device):
                return {k: v.to(device) for k, v in self.items()}

        return _Batch(input_ids=input_ids, attention_mask=attention_mask)


class DistilbertEncoderBase(nn.Module):
    def __init__(
        self,
        vocab_size=99,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
        finetune=False,
    ):
        super().__init__()

        self.tokenizer = _TinyFixedVocabTokenizer(vocab_size=vocab_size)

        config = DistilBertConfig(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            max_position_embeddings=max_position_embeddings,
        )
        self.text_model = DistilBertModel(config)
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        self.text_encoded_dim = self.text_model.config.hidden_size

    def get_last_hidden_state(self, texts, return_mask=False):
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        device = next(self.text_model.parameters()).device
        moved = {k: v.to(device) for k, v in encoded_inputs.items()}
        output = self.text_model(**moved)
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, moved["attention_mask"].to(dtype=torch.bool)


# ---- verbatim from temos/model/textencoder/distilbert_actor.py ----
class DistilbertActorAgnosticEncoder(DistilbertEncoderBase):
    def __init__(
        self,
        vocab_size=99,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
        finetune=False,
        vae=True,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            max_position_embeddings=max_position_embeddings,
            finetune=finetune,
        )
        self.vae = vae

        encoded_dim = self.text_encoded_dim

        self.projection = nn.Sequential(nn.ReLU(), nn.Linear(encoded_dim, latent_dim))

        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=num_layers)

    def forward(self, texts) -> "Distribution | Tensor":
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask=True)

        x = self.projection(text_encoded)
        bs, nframes, _ = x.shape
        x = x.permute(1, 0, 2)

        if self.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

            xseq = torch.cat((emb_token[None], x), 0)

            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            dist = torch.distributions.Normal(mu, std)
            return dist
        else:
            return final[0]


# ---- adapted from temos/model/temos.py: TEMOS.text_to_motion_forward as a
# plain nn.Module (PyTorch Lightning BaseModel / losses / metrics / optimizer
# training scaffolding dropped -- not architecture) ----
class TEMOS(nn.Module):
    def __init__(
        self,
        nfeats=16,
        vae=True,
        latent_dim=32,
        vocab_size=99,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        ff_size=64,
        num_layers=2,
        num_heads=2,
    ):
        super().__init__()
        self.vae = vae

        self.textencoder = DistilbertActorAgnosticEncoder(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            vae=vae,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        self.motiondecoder = ActorAgnosticDecoder(
            nfeats=nfeats,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        self.sample_mean = False
        self.fact = None

    def sample_from_distribution(self, distribution, fact=None, sample_mean=False):
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        if fact is None:
            return distribution.rsample()

        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def forward(self, text_sentences, lengths):
        if self.vae:
            distribution = self.textencoder(text_sentences)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            latent_vector = self.textencoder(text_sentences)

        features = self.motiondecoder(latent_vector, lengths)
        return features


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_temos():
    torch.manual_seed(0)
    model = TEMOS(
        nfeats=16,
        vae=True,
        latent_dim=32,
        vocab_size=99,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        ff_size=64,
        num_layers=2,
        num_heads=2,
    )
    model.eval()
    return model


def example_input_temos():
    torch.manual_seed(0)
    # NOTE: `texts` is a *tuple* (not list) of strings on purpose -- TorchLens's
    # ergonomic string-input auto-tokenization path special-cases `list[str]`
    # inputs and would otherwise intercept this argument before it reaches
    # TEMOS's own (real, verbatim) internal tokenizer call inside
    # `DistilbertEncoderBase.get_last_hidden_state`.
    texts = ("a person walks forward then turns left", "someone waves with their right hand")
    lengths = [5, 7]
    return (texts, lengths)


MENAGERIE_ENTRIES = [
    ("TEMOS", build_temos, example_input_temos, 2022, "vendored-pytorch"),
]
