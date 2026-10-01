# SOURCE: vendored from kit-cel/vae-equalizer @ main
# https://raw.githubusercontent.com/kit-cel/vae-equalizer/main/AWGN_channel/func_VAENN_MQAM.py
#
# "Communication VAE": a variational-autoencoder-based blind channel equalizer for
# wireless/optical constellation shaping, from the Karlsruhe Institute of Technology
# Communications Engineering Lab. The `Net` / `Net_BN` classes below are the exact
# receiver-network architecture (the "decoder" half of the VAE-EM equalizer described
# in the accompanying paper/README): a two-layer 1D convolutional network over the
# received I/Q samples that outputs a per-symbol soft amplitude-level distribution
# (softmax over `num_lev` levels, per I and Q rail), trained end-to-end against an
# ELBO-style loss (`loss_function` in the source, not needed for the forward pass
# traced here). Copied verbatim from `func_VAENN_MQAM.py`; only change is dropping
# the training-script imports (`matplotlib`, `torch.optim`) that are not needed by
# the model classes themselves.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class Net(nn.Module):
    def __init__(self, kernel_1, kernel_2, num_lev, sps):
        super(Net, self).__init__()
        self.fc1 = nn.Conv1d(2, 2 * num_lev, kernel_1, bias=True, padding=kernel_1 // 2)
        self.fc2 = nn.Conv1d(
            2 * num_lev, 2 * num_lev, kernel_2, bias=True, padding=kernel_2 // 2, stride=sps
        )
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)

    def forward(self, x):
        out = self.fc2(F.elu(self.fc1(x)))
        num_lev = out.shape[1] // 2
        N_out = out.shape[2]
        sps = x.shape[-1] // N_out
        x_res = torch.zeros(1, 2, N_out, device=x.device, dtype=torch.float32)
        for i in range(sps):
            x_res += x[:, :, i : sps * N_out : sps] / sps
        sm = nn.Softmax(dim=1)
        netout = torch.empty(1, 2 * num_lev, N_out, device=x.device, dtype=torch.float32)
        netout[0, :num_lev, :], netout[0, num_lev:, :] = (
            sm(out[:, :num_lev, :] + x_res[:, 0, :]),
            sm(out[:, num_lev:, :] + x_res[:, 1, :]),
        )
        return netout


class Net_BN(nn.Module):
    def __init__(self, kernel_1, kernel_2, num_lev, sps):
        super(Net_BN, self).__init__()
        self.fc1 = nn.Conv1d(2, 2 * num_lev, kernel_1, bias=True, padding=kernel_1 // 2)
        self.fc2 = nn.Conv1d(
            2 * num_lev, 2 * num_lev, kernel_2, bias=True, padding=kernel_2 // 2, stride=sps
        )
        self.batch1 = nn.BatchNorm1d(2 * num_lev)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)

    def forward(self, x):
        out = self.fc2(self.batch1(F.elu(self.fc1(x))))
        num_lev = out.shape[1] // 2
        N_out = out.shape[2]
        sps = x.shape[-1] // N_out
        x_res = torch.zeros(1, 2, N_out, device=x.device, dtype=torch.float32)
        for i in range(sps):
            x_res += x[:, :, i : sps * N_out : sps] / sps
        sm = nn.Softmax(dim=1)
        netout = torch.empty(1, 2 * num_lev, N_out, device=x.device, dtype=torch.float32)
        netout[0, :num_lev, :], netout[0, num_lev:, :] = (
            sm(out[:, :num_lev, :] + x_res[:, 0, :]),
            sm(out[:, num_lev:, :] + x_res[:, 1, :]),
        )
        return netout


def build_vae_equalizer():
    # Source default scenario: 16-QAM (num_lev = sqrt(16) = 4), sps=2 (samples per
    # symbol), kernel sizes as used by the AWGN_channel/Eval_run_shaping_vaele.py driver.
    model = Net_BN(kernel_1=17, kernel_2=17, num_lev=4, sps=2)
    model.eval()
    return model


def example_input_vae_equalizer():
    # Received complex baseband signal split into 2 real channels (I, Q), length must
    # be a multiple of sps (=2) so the strided fc2 conv divides evenly.
    torch.manual_seed(0)
    return (torch.randn(1, 2, 256),)


MENAGERIE_ENTRIES = [
    (
        "Communication VAE (VAE-equalizer, KIT-CEL)",
        "build_vae_equalizer",
        "example_input_vae_equalizer",
        2022,
        "vendored-pytorch",
    ),
]
