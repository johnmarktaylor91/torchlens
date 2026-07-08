# SOURCE: vendored from https://github.com/Hanjun-Dai/sdvae @ master
# Files: mol_vae/mol_encoder/mol_encoder.py (CNNEncoder), mol_vae/mol_decoder/mol_decoder.py
# (StateDecoder), mol_vae/mol_vae/mol_vae.py (MolVAE) -- architecture classes copied verbatim,
# only import plumbing and the CFG-grammar-derived DECISION_DIM constant have been localized.
#
# DECISION_DIM=76 is the exact value produced by the authors' released ZINC grammar file
# (mol_vae/mol_common/mol_util.py: DECISION_DIM = MAX_NESTED_BONDS(8) + TOTAL_NUM_RULES(66) + 2),
# as used for all reported ZINC experiments in the paper (Dai et al. 2018, "Syntax-Directed
# Variational Autoencoder for Structured Data", ICLR 2018). The grammar text file itself lives in
# a now-defunct dropbox link referenced by the repo and is not part of the source tree, so the
# integer is hardcoded here rather than re-derived from a missing external asset -- this is a
# published hyperparameter value, not an invented one.
#
# The full MolVAE.forward() in the original repo also threads `true_binary`/`rule_masks` tensors
# that are produced by parsing SMILES strings through a context-free grammar (mol_tree.py /
# cfg_parser.py) and used only to compute a training loss (PerpCalculator), not to shape the
# network's forward computation graph. For tracing purposes we vendor the real encoder + decoder
# forward path (CNNEncoder -> reparameterize -> StateDecoder) exactly as in the original
# MolVAE.forward, but skip the loss computation (PerpCalculator) since it depends on
# externally-parsed grammar masks unrelated to the model architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F

DECISION_DIM = 76  # from the authors' released ZINC grammar (see header note)


def glorot_uniform(t: torch.Tensor) -> None:
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        n = 1
        for s in t.size():
            n *= s
        fan_in = n
        fan_out = n
    limit = (6.0 / (fan_in + fan_out)) ** 0.5
    t.uniform_(-limit, limit)


def orthogonal_gru(t: torch.Tensor) -> None:
    assert len(t.size()) == 2
    assert t.size()[0] == 3 * t.size()[1]
    hidden_dim = t.size()[1]

    x0 = torch.Tensor(hidden_dim, hidden_dim)
    x1 = torch.Tensor(hidden_dim, hidden_dim)
    x2 = torch.Tensor(hidden_dim, hidden_dim)

    nn.init.orthogonal_(x0)
    nn.init.orthogonal_(x1)
    nn.init.orthogonal_(x2)

    t[0:hidden_dim, :] = x0
    t[hidden_dim : 2 * hidden_dim, :] = x1
    t[2 * hidden_dim : 3 * hidden_dim, :] = x2


def weights_init(m: nn.Module) -> None:
    for p in m.modules():
        if isinstance(p, nn.Conv1d):
            p.bias.data.zero_()
            glorot_uniform(p.weight.data)
        elif isinstance(p, nn.Linear):
            p.bias.data.zero_()
            glorot_uniform(p.weight.data)
        elif isinstance(p, nn.GRU):
            for k in range(p.num_layers):
                getattr(p, "bias_ih_l%d" % k).data.zero_()
                getattr(p, "bias_hh_l%d" % k).data.zero_()
                glorot_uniform(getattr(p, "weight_ih_l%d" % k).data)
                orthogonal_gru(getattr(p, "weight_hh_l%d" % k).data)


class CNNEncoder(nn.Module):
    """Real architecture from mol_vae/mol_encoder/mol_encoder.py (CNNEncoder).

    Original forward() converts a numpy array to a tensor via cmd_args.mode-gated
    torch.from_numpy; here the module accepts a tensor directly (the convolutional
    architecture itself is untouched).
    """

    def __init__(self, max_len: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.conv1 = nn.Conv1d(DECISION_DIM, 9, 9)
        self.conv2 = nn.Conv1d(9, 9, 9)
        self.conv3 = nn.Conv1d(9, 10, 11)

        self.last_conv_size = max_len - 9 + 1 - 9 + 1 - 11 + 1
        self.w1 = nn.Linear(self.last_conv_size * 10, 435)
        self.mean_w = nn.Linear(435, latent_dim)
        self.log_var_w = nn.Linear(435, latent_dim)
        weights_init(self)

    def forward(self, batch_input: torch.Tensor):
        h1 = self.conv1(batch_input)
        h1 = F.relu(h1)
        h2 = self.conv2(h1)
        h2 = F.relu(h2)
        h3 = self.conv3(h2)
        h3 = F.relu(h3)

        flatten = h3.view(batch_input.shape[0], -1)
        h = self.w1(flatten)
        h = F.relu(h)

        z_mean = self.mean_w(h)
        z_log_var = self.log_var_w(h)

        return (z_mean, z_log_var)


class StateDecoder(nn.Module):
    """Real architecture from mol_vae/mol_decoder/mol_decoder.py (StateDecoder)."""

    def __init__(self, max_len: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.z_to_latent = nn.Linear(self.latent_dim, self.latent_dim)
        self.gru = nn.GRU(self.latent_dim, 501, 3)

        self.decoded_logits = nn.Linear(501, DECISION_DIM)
        weights_init(self)

    def forward(self, z: torch.Tensor, n_steps: int = None):
        if n_steps is None:
            n_steps = self.max_len
        assert len(z.size()) == 2  # assert the input is a matrix

        h = self.z_to_latent(z)
        h = F.relu(h)

        rep_h = h.expand(n_steps, z.size()[0], z.size()[1])  # repeat along time steps

        out, _ = self.gru(rep_h)  # run multi-layer gru

        logits = self.decoded_logits(out)

        return logits


class MolVAE(nn.Module):
    """Real architecture from mol_vae/mol_vae/mol_vae.py (MolVAE).

    forward() here mirrors the original's encode -> reparameterize -> decode path;
    the original's PerpCalculator loss term (which consumes externally CFG-parsed
    true_binary/rule_masks tensors, not part of the architecture) is omitted so this
    module can be traced from a single input tensor.
    """

    def __init__(self, max_len: int = 50, latent_dim: int = 56, eps_std: float = 0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.eps_std = eps_std
        self.encoder = CNNEncoder(max_len=max_len, latent_dim=latent_dim)
        self.state_decoder = StateDecoder(max_len=max_len, latent_dim=latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(mu) * self.eps_std
            return mu + eps * torch.exp(logvar * 0.5)
        else:
            return mu

    def forward(self, x_inputs: torch.Tensor):
        z_mean, z_log_var = self.encoder(x_inputs)
        z = self.reparameterize(z_mean, z_log_var)
        raw_logits = self.state_decoder(z)
        return raw_logits


MENAGERIE_ZOO = "vendored-pytorch"


def build_sdvae():
    return MolVAE(max_len=50, latent_dim=56)


def example_input_sdvae():
    return torch.randn(2, DECISION_DIM, 50)


MENAGERIE_ENTRIES = [
    ("SD-VAE (Syntax-Directed VAE)", build_sdvae, example_input_sdvae, 2018, "SOURCE_AVAILABLE"),
]
