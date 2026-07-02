# FAITHFUL PORT of stwisdom/dr-nmf @ master (original framework: Theano + Keras 1.x)
# https://raw.githubusercontent.com/stwisdom/dr-nmf/master/custom_layers.py
# https://raw.githubusercontent.com/stwisdom/dr-nmf/master/enhance.py
#
# Deep Recurrent NMF / "unfolded sparse NMF" (Wisdom, Powers, Hershey, Le Roux, Atlas,
# WASPAA 2017, "Building Deep Networks from Reusable Primitives: Application to Speech
# Enhancement"). The model is a deep-unfolded ISTA recursion for sparse NMF, expressed
# as a K_layers-deep recurrent network (`SimpleDeepRNN` in the original code) whose
# weights at every layer are NOT free parameters but are themselves computed from a
# small set of "alternate" SNMF parameters (`log_D`, `log_U1`, `log_Uk`, `log_alph`,
# `log_lam1`) via the closed-form maps in `build_alt()`. At each of the K deep layers
# k within a timestep: `h_k = relu(U_k @ h_{k-1} + S_{k-1} @ h_{k-1,cur} + W_k @ x_t +
# b_k)`, with `U_1 = I`, `U_{k>1} = 0` (SISTA-RNN-style: init from previous timestep's
# converged h, then k=2..K refine it), `S_k = [I - D^T D / alph]^T`, `W_k = [D^T /
# alph]^T`, `b_k = -lam1/alph` -- i.e. K unrolled proximal-gradient (ISTA) steps of
# sparse NMF with a nonnegativity-enforcing ReLU nonlinearity and a learnable
# per-layer-untied dictionary D. `flag_connect_input_to_layers=True` and
# `flag_nonnegative=True` (softplus-initialized trainable h0) match the repo's
# `build_unfolded_snmf()` config, which is the only place `SimpleDeepRNN` is
# instantiated in the codebase. The final hidden state's first/second halves (clean /
# noise NMF activations `H_clean`, `H_noise`) are each projected through a fixed
# (`DenseNonNegW`, i.e. `exp(log W)`-parameterized) linear reconstruction against the
# noisy-speech SNMF dictionary `W_noisy`, and the output is the ideal ratio mask
# `clean_est / (clean_est + noise_est)` (`divide_A_by_AplusB`), exactly as built in
# `enhance.py::build_unfolded_snmf`.
#
# This is a FAITHFUL PORT, not a from-scratch reimplementation: every mechanism above
# (the ISTA-derived per-layer W_k/S_k/b_k maps, the `U_1=I`/`U_{k>1}=0` layer-1-vs-rest
# split, the input-residual connection to every layer, the trainable nonneg h0, the
# masking-then-recurrence-then-DenseNonNegW-then-ratio-mask pipeline) is transcribed
# from the real Theano/Keras 1.x source (`custom_layers.py::SimpleDeepRNN` /
# `enhance.py::build_alt` + `build_unfolded_snmf`) into self-contained torch, because
# the original stack (Theano, Keras 1.x, Python 2, GPU-only sparse_nmf.m dictionary
# pretraining) cannot reasonably be installed alongside this repo's modern torch env.
# The masking layer (`keras.layers.Masking`) is dropped since it only affects loss
# computation over padded timesteps on variable-length batches (not the architecture
# traced per-timestep here); this port instead runs every timestep in the batch, which
# is the same computation Masking performs for any batch of equal-length sequences
# (as used by this staging module's single-length example input).

import torch
import torch.nn as nn


class UnfoldedSNMFCell(nn.Module):
    """Faithful port of `custom_layers.SimpleDeepRNN`, specialized to the
    `flag_connect_input_to_layers=True`, `flag_nonnegative=True`,
    `flag_return_all_hidden=False` configuration used by
    `enhance.py::build_unfolded_snmf` (the only call site in the repo).

    At every timestep t, runs K_layers deep-unfolded ISTA steps over the previous
    timestep's converged hidden state `prev_output`:
        preact_0 = U_1 @ prev_output + W_0 @ x_t
        hidden_0 = relu(preact_0 + b_0)
        preact_k = U_k @ prev_output + S_{k-1} @ hidden_{k-1} + W_k @ x_t   (k>0)
        hidden_k = relu(preact_k + b_k)
    and returns hidden_{K-1} as both the timestep output and the next `prev_output`.
    """

    def __init__(self, input_dim, hidden_dim, K_layers, W_noisy, alph, lam1, untie_alph=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K_layers = K_layers

        # --- alternate SNMF parameters (build_alt / build_unfolded_snmf) ---
        eps = 1e-7
        W_noisy_t = torch.as_tensor(W_noisy, dtype=torch.float32)  # (input_dim, hidden_dim)

        # log_D is untied per layer in the repo's canonical config
        # (params_untied: [log_D, log_alph]), so every layer gets its own D_k.
        log_D_init = torch.log(eps + W_noisy_t)
        self.log_D = nn.Parameter(log_D_init.unsqueeze(0).repeat(K_layers, 1, 1).clone())

        # log_alph is also untied per layer; alph may be scalar or per-hidden-unit.
        if untie_alph:
            alph_vec = torch.full((hidden_dim,), float(alph), dtype=torch.float32)
        else:
            alph_vec = torch.as_tensor(alph, dtype=torch.float32)
            if alph_vec.dim() == 0:
                alph_vec = alph_vec.repeat(hidden_dim)
        log_alph_init = torch.log(torch.tensor(eps, dtype=torch.float32) + alph_vec)
        self.log_alph = nn.Parameter(log_alph_init.unsqueeze(0).repeat(K_layers, 1).clone())

        # log_lam1 is tied across layers in the canonical config (not in params_untied).
        log_lam1_init = torch.log(torch.tensor(eps + float(lam1), dtype=torch.float32))
        self.log_lam1 = nn.Parameter(log_lam1_init.clone())

        # U_1 = I (fixed, non-trainable); U_{k>1} = 0 (fixed, non-trainable).
        self.register_buffer("U1", torch.eye(hidden_dim, dtype=torch.float32))
        self.register_buffer("I_hidden", torch.eye(hidden_dim, dtype=torch.float32))

        # Trainable nonnegative initial hidden state h0 = softplus(log_h0).
        self.log_h0 = nn.Parameter(torch.zeros(hidden_dim))

    def _Dk_normalized(self, k):
        # D_k / ||D_k||_2 (columnwise L2 normalization), matching build_alt's
        # repeated `K.exp(log_D) / sqrt(sum(exp(log_D)**2, axis=0))` pattern.
        Dk = torch.exp(self.log_D[k])
        norm = torch.sqrt(torch.sum(Dk**2, dim=0, keepdim=True))
        return Dk / norm

    def _Wk(self, k):
        # Wk = [ D_k^T / alph_k ]^T  ==  D_k / alph_k  (columnwise), shape (input_dim, hidden_dim)
        Dk_n = self._Dk_normalized(k)
        alph_k = torch.exp(self.log_alph[k])  # (hidden_dim,)
        return Dk_n / alph_k.unsqueeze(0)

    def _Sk(self, k):
        # Sk = [ I - (D_k^T D_k) / alph_k ]^T, for k = 1..K_layers-1 (indexing prev layer's D/alph)
        Dk_n = self._Dk_normalized(k)
        alph_k = torch.exp(self.log_alph[k])
        DtD = torch.matmul(Dk_n.t(), Dk_n) / alph_k.unsqueeze(0)
        return (self.I_hidden - DtD).t()

    def _bk(self, k):
        # bk = -lam1 / alph_k
        alph_k = torch.exp(self.log_alph[k])
        lam1 = torch.exp(self.log_lam1)
        return -lam1 / alph_k

    def initial_state(self, batch_size, device):
        h0 = torch.nn.functional.softplus(self.log_h0)
        return h0.unsqueeze(0).expand(batch_size, -1).to(device)

    def forward(self, x_t, prev_output):
        """One recurrence timestep.
        x_t: (batch, input_dim)
        prev_output: (batch, hidden_dim)
        returns: (batch, hidden_dim)
        """
        hidden = []
        for k in range(self.K_layers):
            Wk = self._Wk(k)
            bk = self._bk(k)
            if k == 0:
                preact = torch.matmul(prev_output, self.U1)
            else:
                # U_{k>1} = 0, so only the S_{k-1} and input terms contribute.
                Sk = self._Sk(k)
                preact = torch.matmul(hidden[k - 1], Sk)
            # residual connection from input to every deep layer (flag_connect_input_to_layers=True)
            preact = preact + torch.matmul(x_t, Wk)
            hidden.append(torch.relu(preact + bk))
        return hidden[-1]


class UnfoldedSNMF(nn.Module):
    """Faithful port of `enhance.py::build_unfolded_snmf`: masking layer -> the
    `SimpleDeepRNN` unfolded-ISTA recurrence over all timesteps -> split hidden state
    into clean/noise halves -> fixed nonnegative dictionary reconstruction
    (`DenseNonNegW`) -> ideal ratio mask (`divide_A_by_AplusB`).
    """

    def __init__(self, input_dim, hidden_dim, K_layers, W_noisy, alph=400.0, lam1=1.0):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must split evenly into clean/noise halves"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.r = hidden_dim // 2
        self.cell = UnfoldedSNMFCell(input_dim, hidden_dim, K_layers, W_noisy, alph, lam1)

        eps = 1e-7
        W_noisy_t = torch.as_tensor(W_noisy, dtype=torch.float32)
        W_clean = W_noisy_t[:, : self.r]
        W_noise = W_noisy_t[:, self.r :]
        # DenseNonNegW: output = input @ exp(kernel), kernel initialized to log(eps+W).T
        # so that exp(kernel) == W at init; use_bias=False as in the repo.
        self.log_W_clean = nn.Parameter(torch.log(eps + W_clean).clone())  # (input_dim, r)
        self.log_W_noise = nn.Parameter(torch.log(eps + W_noise).clone())  # (input_dim, r)

    def forward(self, x):
        """x: (batch, time, input_dim) -- masking is a no-op here since every batch
        in this staging module has equal-length, fully-valid timesteps."""
        batch, T, _ = x.shape
        device = x.device
        h = self.cell.initial_state(batch, device)

        outputs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outputs.append(h)
        H = torch.stack(outputs, dim=1)  # (batch, T, hidden_dim)

        H_clean = H[:, :, : self.r]
        H_noise = H[:, :, self.r :]

        # DenseNonNegW (TimeDistributed): output = H @ exp(log_W)
        clean_est = torch.matmul(H_clean, torch.exp(self.log_W_clean).t())
        noise_est = torch.matmul(H_noise, torch.exp(self.log_W_noise).t())

        eps = 1e-7
        irm_predicted = torch.exp(
            torch.log(eps + clean_est) - torch.log(eps + clean_est + noise_est)
        )
        return irm_predicted


def build_dr_nmf():
    torch.manual_seed(0)
    input_dim = 20
    r = 6
    hidden_dim = 2 * r
    K_layers = 3
    W_noisy = torch.rand(input_dim, hidden_dim).clamp_min(1e-3).numpy()
    model = UnfoldedSNMF(
        input_dim=input_dim, hidden_dim=hidden_dim, K_layers=K_layers, W_noisy=W_noisy
    )
    model.eval()
    return model


def example_input_dr_nmf():
    torch.manual_seed(0)
    # (batch, time, input_dim) log-magnitude-spectrogram-like frames, matching the
    # repo's Masking(input_shape=(maxseq, input_dim)) convention.
    return torch.rand(2, 5, 20)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("Deep-Recurrent-NMF", "build_dr_nmf", "example_input_dr_nmf", 2017, MENAGERIE_ZOO),
]
