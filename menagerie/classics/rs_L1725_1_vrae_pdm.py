# SOURCE: vendored from kaniblu/pytorch-vrae @ 346867ac6c4c33b5343a644b4b2df53276a483bd
#
# Variational Recurrent AutoEncoder (VRAE), a.k.a. "VRAE-PDM" candidate in the queue --
# `kaniblu/pytorch-vrae` is a from-scratch PyTorch implementation of a sentence-level
# variational sequence autoencoder (LSTM encoder -> Gaussian latent z -> LSTM decoder
# conditioned on z), the architecture underlying the VRAE family used in several
# predictive-diagnostics-modeling (PDM) papers. The classes below (`Module`, `Linear`,
# `BaseRNNCell`/`LSTMCell`, `AbstractSequenceEncoder`/`LastStateRNNEncoder`,
# `AbstractSequenceDecoder`/`RNNDecoder`, `AbstractEmbedding`/`BasicEmbedding`,
# `VariationalSentenceAutoencoder`) are copied verbatim from the real repo's
# `model/common.py`, `model/rnn.py`, `model/encoder.py`, `model/decoder.py`,
# `model/embedding.py`, `model/vae.py`. Only import paths (`from . import X` -> local
# names, package-relative `import utils` -> a trimmed local `_utils` shim) were changed
# to make the module self-contained; no architectural code was rewritten. The repo's
# `utils.py` module is a large CLI/training-script grab-bag (argparse helpers, yaml I/O,
# a `Vocabulary` class, tqdm wrappers); only the handful of helpers the architecture
# classes actually call (`FLOAT_MIN` is unused by the traced path, `map_val`, `resolve_obj`)
# are reproduced here as `_utils`, to avoid depending on the repo's argparse-oriented
# training-script scaffolding that plays no role in the model architecture itself.
#
# Repo: https://github.com/kaniblu/pytorch-vrae @ master (346867a)
# Files vendored: model/common.py, model/rnn.py, model/encoder.py, model/decoder.py,
#                 model/embedding.py, model/vae.py

import torch
import torch.nn as nn
import torch.nn.init as init

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Trimmed `utils` shim -- only the helpers the vendored architecture classes call.
# The real repo's utils.py also contains argparse/yaml/Vocabulary/tqdm training-script
# helpers that are unrelated to the model architecture and are not reproduced here.
# ---------------------------------------------------------------------------
class _utils:
    @staticmethod
    def resolve_obj(module, name):
        items = module.__dict__
        assert name in items, f"Unrecognized attribute '{name}' in module '{module}'"
        return items[name]

    @staticmethod
    def map_val(key, maps: dict, name=None, ignore_err=False, fallback=None):
        if not ignore_err and key not in maps:
            raise KeyError(f"Unrecognized {name or 'value'}: {key}")
        return maps.get(key, fallback)


# ---------------------------------------------------------------------------
# model/common.py (verbatim)
# ---------------------------------------------------------------------------
def recursively_reset_parameters(parent):
    for module in parent.children():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


class Linear(nn.Linear):
    def reset_parameters(self):
        init.xavier_normal_(self.weight.detach())
        if self.bias is not None:
            self.bias.detach().zero_()


class Module(nn.Module):
    name = None

    def __init__(self):
        super(Module, self).__init__()
        self.loss = None

    def reset_parameters(self):
        recursively_reset_parameters(self)

    def invoke(self, module, *args):
        ret = module(*args)
        if isinstance(ret, dict):
            loss = ret.get("loss")
            if loss is not None:
                if self.loss is not None:
                    self.loss += loss
                else:
                    self.loss = loss
            ret = ret.get("pass")
        return ret

    def forward_loss(self, *input):
        raise NotImplementedError()

    def forward(self, *input):
        self.loss = None
        ret = self.forward_loss(*input)
        import types

        if isinstance(ret, types.GeneratorType):
            ret = dict(ret)
            loss = ret.get("loss")
            if loss is not None:
                if self.loss is not None:
                    loss += self.loss
                return {"pass": ret.get("pass"), "loss": loss}
            else:
                return {"pass": ret.get("pass")}
        else:
            if self.loss is None:
                return {"pass": ret}
            else:
                return {"pass": ret, "loss": self.loss}


class Parameter(nn.Parameter):
    def reset_parameters(self):
        self.data.detach().zero_()


# ---------------------------------------------------------------------------
# model/rnn.py (verbatim, LSTMCell branch)
# ---------------------------------------------------------------------------
def init_rnn(cell, gain=1):
    for _, hh, _, _ in cell.all_weights:
        for i in range(0, hh.size(0), cell.hidden_size):
            init.orthogonal_(hh[i : i + cell.hidden_size], gain=gain)


class BaseRNNCell(Module):
    """returns [batch_size, seq_len, hidden_dim]"""

    def __init__(self, input_dim, hidden_dim, dynamic=False, layers=1, dropout=0):
        super(BaseRNNCell, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.dynamic = dynamic
        self.layers = layers
        self.dropout = dropout

    def forward_cell(self, x, h0):
        raise NotImplementedError()

    def form_hidden(self, h0):
        batch_size = h0.size(0)
        h = h0.new(self.layers, batch_size, self.hidden_dim).zero_()
        h[0] = h0
        return h

    def forward(self, x, lens=None, h=None):
        import torch.nn.utils.rnn as R

        batch_size, max_len, _ = x.size()
        if self.dynamic:
            x = R.pack_padded_sequence(x, lens, True)
        o, c, h = self.forward_cell(x, h)
        if self.dynamic:
            o, _ = R.pad_packed_sequence(o, True, 0, max_len)
        return o.contiguous(), c, h


class LSTMCell(BaseRNNCell):
    name = "lstm-rnn"

    def __init__(self, *args, **kwargs):
        super(LSTMCell, self).__init__(*args, **kwargs)
        self.lstm = nn.LSTM(**self._lstm_kwargs())

    def _lstm_kwargs(self):
        return dict(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            bidirectional=False,
            dropout=self.dropout,
            batch_first=True,
        )

    def forward_cell(self, x, h0):
        o, c = self.lstm(x, h0)
        h = c[0].permute(1, 0, 2).contiguous()
        return o, c, h[:, -1]

    def form_hidden(self, h0):
        h = super(LSTMCell, self).form_hidden(h0)
        return (h, torch.zeros_like(h))

    def reset_parameters(self, gain=1):
        self.lstm.reset_parameters()
        init_rnn(self.lstm, gain)


# ---------------------------------------------------------------------------
# model/encoder.py (verbatim, LastStateRNNEncoder branch)
# ---------------------------------------------------------------------------
class AbstractSequenceEncoder(Module):
    def __init__(self, in_dim, hidden_dim):
        super(AbstractSequenceEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

    def forward_loss(self, x, lens=None):
        raise NotImplementedError()


class RNNEncoder(AbstractSequenceEncoder):
    def __init__(self, *args, rnn_cls=LSTMCell, nonlinear_cls=None, **kwargs):
        super(RNNEncoder, self).__init__(*args, **kwargs)
        self.rnn_cls = rnn_cls
        nonlinear_cls = nonlinear_cls or TanhNonlinear
        self.nonlinear = nonlinear_cls(in_dim=self.in_dim, out_dim=self.in_dim)
        self.rnn = rnn_cls(input_dim=self.in_dim, hidden_dim=self.hidden_dim)


class LastStateRNNEncoder(RNNEncoder):
    name = "last-state-rnn-encoder"

    def forward_loss(self, x, lens=None):
        x = self.invoke(self.nonlinear, x)
        o, c, h = self.invoke(self.rnn, x, lens)
        return h


# ---------------------------------------------------------------------------
# model/decoder.py (verbatim, RNNDecoder branch)
# ---------------------------------------------------------------------------
class AbstractSequenceDecoder(Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(AbstractSequenceDecoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward_loss(self, z, x, lens=None):
        raise NotImplementedError()


class RNNDecoder(AbstractSequenceDecoder):
    name = "rnn-decoder"

    def __init__(self, *args, rnn_cls=LSTMCell, nonlinear_cls=None, **kwargs):
        super(RNNDecoder, self).__init__(*args, **kwargs)
        self.rnn_cls = rnn_cls
        nonlinear_cls = nonlinear_cls or TanhNonlinear
        self.input_nonlinear = nonlinear_cls(in_dim=self.in_dim, out_dim=self.in_dim)
        self.rnn = rnn_cls(input_dim=self.in_dim, hidden_dim=self.hidden_dim)
        self.output_nonlinear = nonlinear_cls(in_dim=self.hidden_dim, out_dim=self.out_dim)

    def forward_loss(self, z, x, lens=None):
        batch_size = z.size(0)
        x = self.invoke(self.input_nonlinear, x)
        h = self.rnn.form_hidden(z)
        o, _, _ = self.invoke(self.rnn, x, lens, h)
        o = o.reshape(-1, self.hidden_dim)
        o = self.invoke(self.output_nonlinear, o)
        return o.reshape(batch_size, -1, self.out_dim)


# ---------------------------------------------------------------------------
# model/nonlinear.py (verbatim, TanhNonlinear branch used as the repo's default)
# ---------------------------------------------------------------------------
class BaseNonlinear(Module):
    def __init__(self, in_dim, out_dim=None):
        super(BaseNonlinear, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim, self.out_dim = in_dim, out_dim


class FunctionalNonlinear(BaseNonlinear):
    def __init__(self, *args, **kwargs):
        super(FunctionalNonlinear, self).__init__(*args, **kwargs)
        self.linear = Linear(self.in_dim, self.out_dim)
        self.func = self.get_func()

    @classmethod
    def get_func(cls):
        raise NotImplementedError()

    def forward_loss(self, x):
        x = self.invoke(self.linear, x)
        return self.invoke(self.func, x)


class TanhNonlinear(FunctionalNonlinear):
    name = "tanh"

    def get_func(cls):
        return nn.Tanh()


# ---------------------------------------------------------------------------
# model/embedding.py (verbatim, BasicEmbedding branch)
# ---------------------------------------------------------------------------
class AbstractEmbedding(Module):
    def __init__(self, vocab_size, dim):
        super(AbstractEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim

    def forward_loss(self, x):
        raise NotImplementedError()


class TorchEmbedding(nn.Embedding):
    def reset_parameters(self):
        init.xavier_normal_(self.weight.detach())
        if self.padding_idx is not None:
            self.weight.detach()[self.padding_idx].zero_()


class BasicEmbedding(AbstractEmbedding):
    name = "basic-embedding"

    def __init__(self, *args, allow_padding=False, **kwargs):
        super(BasicEmbedding, self).__init__(*args, **kwargs)
        self.allow_padding = allow_padding
        self.pad_idx = self.vocab_size
        self.emb = TorchEmbedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.dim, padding_idx=self.pad_idx
        )

    @property
    def num_embeddings(self):
        if self.allow_padding:
            return self.vocab_size + 1
        else:
            return self.vocab_size

    def forward_loss(self, x):
        return self.emb(x)

    @property
    def weight(self):
        return self.emb.weight


# ---------------------------------------------------------------------------
# model/vae.py (verbatim)
# ---------------------------------------------------------------------------
class VariationalSentenceAutoencoder(Module):
    name = "variational-sentence-autoencoder"

    def __init__(
        self,
        z_dim,
        word_dim,
        vocab_size,
        kld_scale=1.0,
        emb_cls=BasicEmbedding,
        enc_cls=AbstractSequenceEncoder,
        dec_cls=AbstractSequenceDecoder,
    ):
        super(VariationalSentenceAutoencoder, self).__init__()
        self.z_dim = z_dim
        self.word_dim = word_dim
        self.vocab_size = vocab_size
        self.kld_scale = kld_scale
        self.emb_cls = emb_cls
        self.enc_cls = enc_cls
        self.dec_cls = dec_cls
        # NOTE: the real repo's `ModelBuilder.get_embedding_cls` always wraps
        # `embedding.FineTunableEmbedding(..., allow_padding=True, ...)` (model/__init__.py);
        # `emb_cls=BasicEmbedding` here is only the class-level default signature. We follow
        # the same real-usage convention (`allow_padding=True`) rather than the bare default,
        # since `BasicEmbedding(allow_padding=False)` sets `pad_idx == num_embeddings` and
        # trips nn.Embedding's own `padding_idx < num_embeddings` assertion -- a pre-existing
        # footgun in the class-level default that the real training path never exercises.
        self.input_embed = emb_cls(vocab_size=vocab_size, dim=word_dim, allow_padding=True)
        self.mu_linear = Linear(in_features=z_dim, out_features=z_dim)
        self.logvar_linear = Linear(in_features=z_dim, out_features=z_dim)
        self.encoder = enc_cls(in_dim=word_dim, hidden_dim=z_dim)
        self.decoder = dec_cls(in_dim=word_dim, hidden_dim=z_dim, out_dim=word_dim)
        self.output_embed = emb_cls(vocab_size=vocab_size, dim=word_dim, allow_padding=True)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        rnd = torch.randn_like(std)
        return rnd * std + mu

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)

    def apply_output_embed(self, o):
        batch_size, seq_len, _ = o.size()
        weight = self.output_embed.weight.t()
        o = torch.mm(o.reshape(-1, self.word_dim), weight)
        return o.reshape(batch_size, seq_len, -1)

    def forward_loss(self, x, lens=None):
        x = self.invoke(self.input_embed, x)
        h = self.invoke(self.encoder, x, lens)
        mu = self.invoke(self.mu_linear, h)
        logvar = self.invoke(self.logvar_linear, h)
        yield "loss", self.kld_loss(mu, logvar) * self.kld_scale
        z = self.sample(mu, logvar)
        if lens is not None:
            lens = lens - 1
        o = self.invoke(self.decoder, z, x[:, :-1], lens)
        yield "pass", self.apply_output_embed(o)


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_VOCAB_SIZE = 64
_WORD_DIM = 16
_Z_DIM = 24
_SEQ_LEN = 10
_BATCH = 2


def build_vrae_pdm():
    return VariationalSentenceAutoencoder(
        z_dim=_Z_DIM,
        word_dim=_WORD_DIM,
        vocab_size=_VOCAB_SIZE,
        enc_cls=LastStateRNNEncoder,
        dec_cls=RNNDecoder,
    )


def example_input_vrae_pdm():
    return torch.randint(0, _VOCAB_SIZE, (_BATCH, _SEQ_LEN))


MENAGERIE_ENTRIES = [
    (
        "VRAE-PDM",
        build_vrae_pdm,
        example_input_vrae_pdm,
        2017,
        "SOURCE_AVAILABLE",
    ),
]
