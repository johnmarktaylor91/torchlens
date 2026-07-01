# SOURCE: vendored from guxd/DialogWAE @ master (models/dialogwae.py, modules.py)
#
# DialogWAE (Gu, Cho, Ha, Kim, "DialogWAE: Multimodal Response Generation
# with Conditional Wasserstein Auto-Encoder", ICLR 2019). Real architecture:
# a hierarchical dialogue encoder (a GRU `Encoder` per utterance, feeding a
# GRU `ContextEncoder` over the sequence of utterance encodings concatenated
# with a 2-dim speaker-floor one-hot), a Gaussian `Variation` net used twice
# (as the "posterior" net `q(e|c,x)` conditioned on context+response and the
# "prior" net `p(e|c)` conditioned on context only), two small MLP
# "generator" networks (`post_generator`/`prior_generator`) that push the
# Gaussian samples through the actual GAN-style adversarial transform (the
# WAE's core idea: the generators, not the Gaussian nets, define the true
# non-Gaussian latent code the discriminator is trained against), a
# `discriminator` (WGAN-GP critic scoring `[z, context]` pairs), and a GRU
# `Decoder` that autoregressively generates the response conditioned on
# `[z, context]`. This is the paper's real architecture (adversarially
# trained conditional VAE for dialogue), so it is vendored (rung 2) rather
# than reimplemented, using only `torch`/`numpy` (both base libs).
#
# Vendoring notes (imports/config fixes only, architecture untouched):
#   - The repo's `helper.py` `gVar`/`gData` unconditionally called
#     `tensor.cuda()` whenever `torch.cuda.is_available()` -- fine for GPU
#     training but irrelevant to CPU tracing; ported as plain
#     identity-on-CPU passthroughs (no cuda placement logic dropped, the
#     original also no-ops on CPU-only machines).
#   - `import torch.nn.init as weight_init` /
#     `from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence`
#     kept as in the original; `sys.path.insert(0, os.path.abspath(".."))`
#     (repo's cross-file import shim for its flat `models/`+top-level layout)
#     dropped since both files are merged into a single module here.
#   - `Encoder`/`ContextEncoder`/`Variation`/`Decoder`/`DialogWAE.__init__`
#     and `DialogWAE.sample_code_post`/`sample_code_prior` are copied
#     verbatim (unchanged compute, only whitespace/import cleanup).
#   - `MixVariation` (an alternate Gaussian-mixture prior used by the
#     repo's `dialogwae_gmp.py` variant, not by the base `DialogWAE` traced
#     here) is dropped as dead code for this entry point.
#   - The optimizer/scheduler construction in `DialogWAE.__init__`
#     (`optimizer_AE`/`optimizer_G`/`optimizer_D`/`lr_scheduler_AE`) and the
#     `train_AE`/`train_G`/`train_D`/`valid`/`adjust_lr` training-loop
#     methods (which call `.backward()`/`optimizer.step()`/`.item()`
#     directly, not appropriate for graph tracing) are dropped; the traced
#     entry point below (`DialogWAEWrapper.forward`) reimplements exactly
#     the forward-only computation of `train_AE` (context encode -> response
#     encode -> posterior-sample z -> decode logits), which is the model's
#     real generative forward pass, verbatim except it returns the decoder
#     logits instead of computing/backpropagating the masked
#     cross-entropy loss.
#   - `Decoder.sampling` (the autoregressive greedy/sample generation loop,
#     used only by `DialogWAE.sample`, not by the traced training forward
#     path) is dropped; it uses `np.int` which no longer exists in modern
#     numpy and is unrelated to the traced compute graph.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def gData(data):
    """CPU passthrough of the repo's `gData` (original moved tensors to
    cuda when available; tracing runs on CPU so this is a no-op here, same
    as the original on a CPU-only machine)."""
    return data


def gVar(data):
    return gData(data)


class Encoder(nn.Module):
    def __init__(
        self, embedder, input_size, hidden_size, bidirectional, n_layers, noise_radius=0.2
    ):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.noise_radius = noise_radius
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        assert type(self.bidirectional) == bool

        self.embedding = embedder
        self.rnn = nn.GRU(
            input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional
        )
        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, inputs, input_lens=None, noise=False):
        if self.embedding is not None:
            inputs = self.embedding(inputs)

        batch_size, seq_len, emb_size = inputs.size()
        inputs = F.dropout(inputs, 0.5, self.training)

        if input_lens is not None:
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)
            inputs = pack_padded_sequence(
                inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True
            )

        init_hidden = gVar(
            torch.zeros(self.n_layers * (1 + self.bidirectional), batch_size, self.hidden_size)
        )
        hids, h_n = self.rnn(inputs, init_hidden)
        if input_lens is not None:
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, (1 + self.bidirectional), batch_size, self.hidden_size)
        h_n = h_n[-1]
        enc = h_n.transpose(1, 0).contiguous().view(batch_size, -1)
        if noise and self.noise_radius > 0:
            gauss_noise = gVar(torch.normal(means=torch.zeros(enc.size()), std=self.noise_radius))
            enc = enc + gauss_noise

        return enc, hids


class ContextEncoder(nn.Module):
    def __init__(self, utt_encoder, input_size, hidden_size, n_layers=1, noise_radius=0.2):
        super(ContextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.noise_radius = noise_radius

        self.n_layers = n_layers

        self.utt_encoder = utt_encoder
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():  # initialize the gate weights with orthogonal
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, context, context_lens, utt_lens, floors, noise=False):
        batch_size, max_context_len, max_utt_len = context.size()
        utts = context.view(-1, max_utt_len)
        utt_lens = utt_lens.view(-1)
        utt_encs, _ = self.utt_encoder(utts, utt_lens)
        utt_encs = utt_encs.view(batch_size, max_context_len, -1)

        floor_one_hot = gVar(torch.zeros(floors.numel(), 2))
        floor_one_hot.data.scatter_(1, floors.view(-1, 1), 1)
        floor_one_hot = floor_one_hot.view(-1, max_context_len, 2)
        utt_floor_encs = torch.cat([utt_encs, floor_one_hot], 2)

        utt_floor_encs = F.dropout(utt_floor_encs, 0.25, self.training)
        context_lens_sorted, indices = context_lens.sort(descending=True)
        utt_floor_encs = utt_floor_encs.index_select(0, indices)
        utt_floor_encs = pack_padded_sequence(
            utt_floor_encs, context_lens_sorted.data.tolist(), batch_first=True
        )

        init_hidden = gVar(torch.zeros(1, batch_size, self.hidden_size))
        hids, h_n = self.rnn(utt_floor_encs, init_hidden)

        _, inv_indices = indices.sort()
        h_n = h_n.index_select(1, inv_indices)

        enc = h_n.transpose(1, 0).contiguous().view(batch_size, -1)

        if noise and self.noise_radius > 0:
            gauss_noise = gVar(torch.normal(means=torch.zeros(enc.size()), std=self.noise_radius))
            enc = enc + gauss_noise
        return enc


class Variation(nn.Module):
    def __init__(self, input_size, z_size):
        super(Variation, self).__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
        )
        self.context_to_mu = nn.Linear(z_size, z_size)
        self.context_to_logsigma = nn.Linear(z_size, z_size)

        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mu)
        self.init_weights(self.context_to_logsigma)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def forward(self, context):
        batch_size, _ = context.size()
        context = self.fc(context)
        mu = self.context_to_mu(context)
        logsigma = self.context_to_logsigma(context)
        std = torch.exp(0.5 * logsigma)

        epsilon = gVar(torch.randn([batch_size, self.z_size]))
        z = epsilon * std + mu
        return z, mu, logsigma


class Decoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, vocab_size, n_layers=1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = embedder
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)
        self.out.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.fill_(0)

    def forward(self, init_hidden, context=None, inputs=None, lens=None):
        batch_size, maxlen = inputs.size()
        if self.embedding is not None:
            inputs = self.embedding(inputs)
        if context is not None:
            repeated_context = context.unsqueeze(1).repeat(1, maxlen, 1)
            inputs = torch.cat([inputs, repeated_context], 2)
        inputs = F.dropout(inputs, 0.5, self.training)
        hids, h_n = self.rnn(inputs, init_hidden.unsqueeze(0))
        decoded = self.out(hids.contiguous().view(-1, self.hidden_size))
        decoded = decoded.view(batch_size, maxlen, self.vocab_size)
        return decoded


class DialogWAE(nn.Module):
    def __init__(self, config, vocab_size, PAD_token=0):
        super(DialogWAE, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen = config["maxlen"]
        self.clip = config["clip"]
        self.lambda_gp = config["lambda_gp"]
        self.temp = config["temp"]

        self.embedder = nn.Embedding(vocab_size, config["emb_size"], padding_idx=PAD_token)
        self.utt_encoder = Encoder(
            self.embedder,
            config["emb_size"],
            config["n_hidden"],
            True,
            config["n_layers"],
            config["noise_radius"],
        )
        self.context_encoder = ContextEncoder(
            self.utt_encoder,
            config["n_hidden"] * 2 + 2,
            config["n_hidden"],
            1,
            config["noise_radius"],
        )
        self.prior_net = Variation(config["n_hidden"], config["z_size"])  # p(e|c)
        self.post_net = Variation(config["n_hidden"] * 3, config["z_size"])  # q(e|c,x)

        self.post_generator = nn.Sequential(
            nn.Linear(config["z_size"], config["z_size"]),
            nn.BatchNorm1d(config["z_size"], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config["z_size"], config["z_size"]),
            nn.BatchNorm1d(config["z_size"], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config["z_size"], config["z_size"]),
        )
        self.post_generator.apply(self.init_weights)

        self.prior_generator = nn.Sequential(
            nn.Linear(config["z_size"], config["z_size"]),
            nn.BatchNorm1d(config["z_size"], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config["z_size"], config["z_size"]),
            nn.BatchNorm1d(config["z_size"], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config["z_size"], config["z_size"]),
        )
        self.prior_generator.apply(self.init_weights)

        self.decoder = Decoder(
            self.embedder,
            config["emb_size"],
            config["n_hidden"] + config["z_size"],
            vocab_size,
            n_layers=1,
        )

        self.discriminator = nn.Sequential(
            nn.Linear(config["n_hidden"] + config["z_size"], config["n_hidden"] * 2),
            nn.BatchNorm1d(config["n_hidden"] * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config["n_hidden"] * 2, config["n_hidden"] * 2),
            nn.BatchNorm1d(config["n_hidden"] * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config["n_hidden"] * 2, 1),
        )
        self.discriminator.apply(self.init_weights)

        self.criterion_ce = nn.CrossEntropyLoss()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def sample_code_post(self, x, c):
        e, _, _ = self.post_net(torch.cat((x, c), 1))
        z = self.post_generator(e)
        return z

    def sample_code_prior(self, c):
        e, _, _ = self.prior_net(c)
        z = self.prior_generator(e)
        return z


class DialogWAEWrapper(nn.Module):
    """Reimplements the forward-only compute of `DialogWAE.train_AE`
    (context encode -> response encode -> posterior-sample z -> decode
    logits) so torchlens can trace it directly. This is the model's real
    generative forward pass, unchanged except it returns decoder logits
    instead of computing/backpropagating the masked cross-entropy loss
    (`.backward()`/`optimizer.step()` are not part of the traced graph)."""

    VOCAB_SIZE = 300
    MAX_CONTEXT_LEN = 3
    MAX_UTT_LEN = 8
    RESP_LEN = 8

    def __init__(self):
        super().__init__()
        config = dict(
            maxlen=self.RESP_LEN,
            clip=5.0,
            lambda_gp=10.0,
            temp=1.0,
            emb_size=16,
            n_hidden=24,
            n_layers=1,
            noise_radius=0.2,
            z_size=12,
        )
        self.model = DialogWAE(config, vocab_size=self.VOCAB_SIZE, PAD_token=0)

    def forward(self, context, response):
        batch_size = context.size(0)
        context_lens = torch.full((batch_size,), self.MAX_CONTEXT_LEN, dtype=torch.long)
        utt_lens = torch.full(
            (batch_size, self.MAX_CONTEXT_LEN), self.MAX_UTT_LEN, dtype=torch.long
        )
        floors = torch.zeros(batch_size, self.MAX_CONTEXT_LEN, dtype=torch.long)
        res_lens = torch.full((batch_size,), self.RESP_LEN - 1, dtype=torch.long)

        c = self.model.context_encoder(context, context_lens, utt_lens, floors)
        x, _ = self.model.utt_encoder(response[:, 1:], res_lens)
        z = self.model.sample_code_post(x, c)
        output = self.model.decoder(torch.cat((z, c), 1), None, response[:, :-1], res_lens)
        return output


def build_dialogwae():
    return DialogWAEWrapper()


def example_input_dialogwae():
    torch.manual_seed(0)
    w = DialogWAEWrapper
    context = torch.randint(1, w.VOCAB_SIZE, (2, w.MAX_CONTEXT_LEN, w.MAX_UTT_LEN))
    response = torch.randint(1, w.VOCAB_SIZE, (2, w.RESP_LEN))
    return context, response


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DialogWAE (Conditional Wasserstein Auto-Encoder Dialogue Model)",
        build_dialogwae,
        example_input_dialogwae,
        2019,
        "vendored-pytorch",
    ),
]
