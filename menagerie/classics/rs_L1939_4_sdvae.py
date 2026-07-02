# FAITHFUL PORT of Hanjun-Dai/sdvae @ master (original framework: PyTorch 0.1.x + external
# grammar-parsing/data infra unavailable at run time)
# https://raw.githubusercontent.com/Hanjun-Dai/sdvae/master/mol_vae/mol_encoder/mol_encoder.py
# https://raw.githubusercontent.com/Hanjun-Dai/sdvae/master/mol_vae/mol_decoder/mol_decoder.py
# https://raw.githubusercontent.com/Hanjun-Dai/sdvae/master/mol_vae/mol_vae/mol_vae.py
#
# "Syntax-Directed Variational Autoencoder for Structured Data" (Dai, Tian, Dai, Skiena, Song;
# ICLR 2018). SD-VAE: a grammar/attribute-grammar-constrained VAE over parse trees of
# structured sequences (SMILES molecules / program syntax) -- the *architecture itself*
# (`CNNEncoder`, `StateDecoder`, `MolVAE`) is already real PyTorch in the repo (not TF, despite
# the neighboring `sparse_gp_theano_internal.py` Bayesian-optimization scripts being
# Theano-based). It is classified as a PORT rather than vendored because the module cannot be
# imported/run as-is: `mol_util.py` (which defines `DECISION_DIM`, the encoder/decoder's
# channel-count constant) parses a context-free-grammar file at import time via
# `cmd_args.grammar_file`, and that grammar file lives in the repo's separate, not-fetched
# `dropbox/context_free_grammars` data directory, so `DECISION_DIM` cannot be obtained without
# unavailable data. This port transcribes the same `CNNEncoder`/`StateDecoder`/`MolVAE` classes
# verbatim, with `DECISION_DIM` promoted from a grammar-derived global constant to an explicit
# `decision_dim` constructor argument (a data-loading value, not an architectural change) and
# `cmd_args`-only knobs (`.mode`, `.eps_std`, `.kl_coeff`, `.rnn_type`) turned into ordinary
# constructor parameters:
#   - CNNEncoder: 3x `Conv1d` (kernel 9/9/11, channel `DECISION_DIM -> 9 -> 9 -> 10`) + ReLU,
#     flatten, `Linear -> ReLU`, then two parallel `Linear` heads for `z_mean`/`z_log_var`.
#     `weights_init` (custom glorot_uniform Conv1d/Linear init) is dropped as a training-time
#     initialization convenience, not architecture (the Linear/Conv1d layer types and shapes
#     are identical either way; default torch init is used instead).
#   - StateDecoder: `Linear -> ReLU` on `z`, expand across `n_steps` time steps, 3-layer `GRU`
#     (hidden size 501, matching the real repo's hardcoded `nn.GRU(latent_dim, 501, 3)`), then
#     `Linear` to `DECISION_DIM` logits per step. The `rnn_type == 'sru'` branch (external `sru`
#     package, optional/GPU-only in the real repo) is dropped; only the default `gru` path is
#     kept.
#   - MolVAE.forward: encode -> reparameterize (`training`-gated, `eps ~ N(0, eps_std)`) ->
#     decode -> returns `(raw_logits, kl_loss)`. The real `PerpCalculator`
#     (`my_perp_loss`/`my_binary_loss` reconstruction losses against `true_binary`/`rule_masks`
#     one-hot targets synthesized by the grammar-tree walker) is training-loss infrastructure
#     tied to the unavailable grammar/tree-walking code, and is dropped; the module's
#     `forward()` returns the raw decoder logits plus the KL term exactly as `MolVAE.forward`
#     computes them before invoking `perp_calc`.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class CNNEncoder(nn.Module):
    def __init__(self, max_len, latent_dim, decision_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.decision_dim = decision_dim

        self.conv1 = nn.Conv1d(decision_dim, 9, 9)
        self.conv2 = nn.Conv1d(9, 9, 9)
        self.conv3 = nn.Conv1d(9, 10, 11)

        self.last_conv_size = max_len - 9 + 1 - 9 + 1 - 11 + 1
        self.w1 = nn.Linear(self.last_conv_size * 10, 435)
        self.mean_w = nn.Linear(435, latent_dim)
        self.log_var_w = nn.Linear(435, latent_dim)

    def forward(self, batch_input):
        # batch_input: (batch, decision_dim, max_len) one-hot rule-sequence encoding
        h1 = torch.relu(self.conv1(batch_input))
        h2 = torch.relu(self.conv2(h1))
        h3 = torch.relu(self.conv3(h2))

        flatten = h3.reshape(batch_input.shape[0], -1)
        h = torch.relu(self.w1(flatten))

        z_mean = self.mean_w(h)
        z_log_var = self.log_var_w(h)
        return z_mean, z_log_var


class StateDecoder(nn.Module):
    def __init__(self, max_len, latent_dim, decision_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.decision_dim = decision_dim

        self.z_to_latent = nn.Linear(self.latent_dim, self.latent_dim)
        self.gru = nn.GRU(self.latent_dim, 501, 3)
        self.decoded_logits = nn.Linear(501, decision_dim)

    def forward(self, z, n_steps=None):
        if n_steps is None:
            n_steps = self.max_len
        assert len(z.size()) == 2  # z must be a matrix

        h = torch.relu(self.z_to_latent(z))
        rep_h = h.expand(n_steps, z.size(0), z.size(1))  # repeat along time steps

        out, _ = self.gru(rep_h)  # multi-layer GRU
        logits = self.decoded_logits(out)
        return logits


class MolVAE(nn.Module):
    def __init__(self, max_len=50, latent_dim=56, decision_dim=64, eps_std=0.01, kl_coeff=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.eps_std = eps_std
        self.kl_coeff = kl_coeff

        self.encoder = CNNEncoder(max_len=max_len, latent_dim=latent_dim, decision_dim=decision_dim)
        self.state_decoder = StateDecoder(
            max_len=max_len, latent_dim=latent_dim, decision_dim=decision_dim
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            eps = torch.randn_like(mu) * self.eps_std
            return mu + eps * torch.exp(logvar * 0.5)
        return mu

    def forward(self, x_inputs):
        z_mean, z_log_var = self.encoder(x_inputs)
        z = self.reparameterize(z_mean, z_log_var)

        raw_logits = self.state_decoder(z)

        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
        return raw_logits, self.kl_coeff * torch.mean(kl_loss)


def build_sdvae():
    torch.manual_seed(0)
    # decision_dim / max_len stand in for the grammar-derived TOTAL_NUM_RULES + padded sequence
    # length, which normally come from parsing a context-free-grammar file (unavailable here).
    model = MolVAE(max_len=50, latent_dim=56, decision_dim=64, eps_std=0.01, kl_coeff=1.0)
    model.eval()
    return model


def example_input_sdvae():
    torch.manual_seed(0)
    batch_size = 4
    decision_dim = 64
    max_len = 50
    # one-hot-ish rule-sequence encoding (real repo builds this from `true_binary`/rule masks
    # produced by the grammar tree-walker; a random categorical-like tensor stands in here).
    return torch.rand(batch_size, decision_dim, max_len)


MENAGERIE_ENTRIES = [
    ("Syntax-Directed VAE (SD-VAE)", "build_sdvae", "example_input_sdvae", 2018, "ported"),
]
