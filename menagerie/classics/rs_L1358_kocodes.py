# SOURCE: vendored from https://github.com/deepcomm/KOcodes @ c412e35b94879e6ad69ff2773d6f65a2b62b18a0
#   Vendored file:
#     - train_KO_m1_dumer.py -> `g_Full`, `f_Full` (the KO encoder/decoder MLP building blocks),
#       `power_constraint`, `repetition_code_matrices`, and the `encoder_full` /
#       `decoder_nn_full` composition logic (official implementation accompanying
#       "KO codes: Inventing Nonlinear Encoding and Decoding for Reliable Wireless
#       Communication via Deep-learning", Makkuva et al., ICML 2021, arXiv:2108.12920).
#
# KO codes learns a nonlinear generalization of the recursive Plotkin (|u|u+v|) construction
# used by Reed-Muller / polar codes. At each of the m recursion levels, the linear XOR
# combination v -> gnet(u, v) is replaced by a small feedforward "g_Full" network (bias +
# SELU MLP with a multiplicative skip term u*v), producing the KO(m,1) encoder as a tree of
# these learned nonlinear combiners. The paired neural decoder walks the same recursion in
# reverse with "f_Full" networks that refine the standard successive-cancellation ("Dumer")
# log-likelihood-ratio message passing. Architecture (the g_Full/f_Full nn.Module blocks and
# their exact recursive composition) reproduced verbatim from the training script; only the
# surrounding argparse/data-loader/training-loop scaffolding is dropped, and the per-level
# net dictionaries are wrapped in a single nn.Module (`KOCodeM1`) so the whole encoder+decoder
# pipeline is one traceable forward pass instead of free functions over dict-of-modules.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class g_Full(nn.Module):
    """Learned nonlinear Plotkin combiner used at each encoder recursion level."""

    def __init__(self, input_size, hidden_size, output_size):
        super(g_Full, self).__init__()

        self.input_size = input_size
        self.half_input_size = int(input_size / 2)
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def forward(self, y):
        x = F.selu(self.fc1(y))
        x = F.selu(self.fc2(x))

        x = F.selu(self.fc3(x))
        x = self.fc4(x) + y[:, : self.half_input_size] * y[:, self.half_input_size :]
        return x


class f_Full(nn.Module):
    """Learned nonlinear decoder-side refinement network used at each recursion level."""

    def __init__(self, input_size, hidden_size, output_size):
        super(f_Full, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def forward(self, y):
        x = F.selu(self.fc1(y))
        x = F.selu(self.fc2(x))

        x = F.selu(self.fc3(x))
        x = self.fc4(x)
        return x


def repetition_code_matrices(device, m=8):
    M_dict = {}
    for i in range(1, m):
        M_dict[i] = torch.ones(1, 2**i).to(device)
    return M_dict


def power_constraint_hard_block(codewords, m):
    """`hard_power_block` branch of the repo's `power_constraint` (deterministic, no running
    stats -- the `soft_power_*` branches instead depend on train/test-time buffers tracked on
    the top-level g_Full module, which is orthogonal to the architecture proper)."""
    return F.normalize(codewords, p=2, dim=1) * (2**m) ** 0.5


class KOCodeM1(nn.Module):
    """KO(m,1) encoder + Dumer-style neural decoder, composed as a single nn.Module.

    Reproduces `encoder_full` and `decoder_nn_full` from the official training script:
    the encoder recursively builds a length-2^m codeword from m+1 message bits via m
    learned `g_Full` combiners (one per recursion level), applies a hard power constraint,
    sends it through an AWGN channel, then the decoder recursively undoes the recursion via
    2m learned `f_Full` refinement networks combined with closed-form log-sum-exp message
    passing.
    """

    def __init__(self, m: int, hidden_size: int = 16, train_snr_db: float = 0.0):
        super().__init__()
        self.m = m
        self.train_snr_db = train_snr_db

        self.gnet_dict = nn.ModuleDict(
            {str(i): g_Full(2 * 2 ** (i - 1), hidden_size, 2 ** (i - 1)) for i in range(1, m + 1)}
        )
        fnet_dict = {}
        for i in range(1, m + 1):
            fnet_dict[str(2 * i - 1)] = f_Full(2 ** (m - i + 1), hidden_size, 1)
            fnet_dict[str(2 * i)] = f_Full(1 + 1 + 1, hidden_size, 1)
        self.fnet_dict = nn.ModuleDict(fnet_dict)

        for net in list(self.gnet_dict.values()) + list(self.fnet_dict.values()):
            net.apply(_weights_init)

        for i in range(1, m):
            self.register_buffer(f"repetition_{i}", torch.ones(1, 2**i))

    def _repetition(self, i):
        return getattr(self, f"repetition_{i}")

    def encode(self, msg_bits: torch.Tensor) -> torch.Tensor:
        """msg_bits: (batch, m+1) of +-1 valued bits -> codeword (batch, 2**m)."""
        u_level0 = msg_bits[:, 0:1]
        v_level0 = msg_bits[:, 1:2]

        for i in range(2, self.m + 1):
            u_level0 = torch.cat(
                [u_level0, self.gnet_dict[str(i - 1)](torch.cat([u_level0, v_level0], dim=1))],
                dim=1,
            )
            v_level0 = msg_bits[:, i : i + 1].mm(self._repetition(i - 1))

        u_levelm = torch.cat(
            [u_level0, self.gnet_dict[str(self.m)](torch.cat([u_level0, v_level0], dim=1))], dim=1
        )

        return power_constraint_hard_block(u_levelm, self.m)

    def channel(self, codewords: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """AWGN channel (noise passed in explicitly so the whole pipeline is deterministic
        and traceable; the repo instead samples `torch.randn_like` inline)."""
        noise_sigma = 10 ** (-self.train_snr_db / 20.0)
        return codewords + noise_sigma * noise

    def decode(self, corrupted_codewords: torch.Tensor) -> torch.Tensor:
        """corrupted_codewords: (batch, 2**m) -> decoded_llrs (batch, m+1)."""
        m = self.m
        Lu = corrupted_codewords
        decoded_llrs = torch.zeros(
            corrupted_codewords.shape[0], m + 1, dtype=corrupted_codewords.dtype
        )

        for i in range(m - 1, -1, -1):
            f_odd = self.fnet_dict[str(2 * (m - i) - 1)]
            f_even = self.fnet_dict[str(2 * (m - i))]

            lse = _log_sum_exp(
                torch.cat([Lu[:, : 2**i].unsqueeze(2), Lu[:, 2**i :].unsqueeze(2)], dim=2).permute(
                    0, 2, 1
                )
            ).sum(dim=1, keepdim=True)
            Lv = f_odd(Lu) + lse

            v_hat = torch.tanh(Lv / 2)
            decoded_llrs[:, i + 1] = v_hat.squeeze(1)

            Lu = (
                f_even(
                    torch.cat(
                        [
                            Lu[:, : 2**i].unsqueeze(2),
                            Lu[:, 2**i :].unsqueeze(2),
                            v_hat.unsqueeze(1).repeat(1, 2**i, 1),
                        ],
                        dim=2,
                    )
                ).squeeze(2)
                + Lu[:, : 2**i]
                + v_hat * Lu[:, 2**i :]
            )

        u_1_hat = torch.tanh(Lu / 2)
        decoded_llrs[:, 0] = u_1_hat.squeeze(1)

        return decoded_llrs

    def forward(self, msg_bits: torch.Tensor, channel_noise: torch.Tensor) -> torch.Tensor:
        codewords = self.encode(msg_bits)
        corrupted = self.channel(codewords, channel_noise)
        return self.decode(corrupted)


def _log_sum_exp(llr_vector: torch.Tensor) -> torch.Tensor:
    sum_vector = llr_vector.sum(dim=1, keepdim=True)
    sum_concat = torch.cat([sum_vector, torch.zeros_like(sum_vector)], dim=1)
    return torch.logsumexp(sum_concat, dim=1) - torch.logsumexp(llr_vector, dim=1)


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.0)


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_kocodes():
    """KO(m=3,1) encoder+decoder for tracing. m=3 is the smallest value that exercises the
    full recursive g_Full/f_Full tree (the repo's own experiments sweep m up to 8)."""
    model = KOCodeM1(m=3, hidden_size=16, train_snr_db=0.0)
    model.eval()
    return model


def example_input_kocodes():
    """(msg_bits, channel_noise): msg_bits are +-1 valued (batch=4, m+1=4); channel_noise is
    standard Gaussian of the codeword shape (batch=4, 2**m=8), matching the repo's
    `awgn_channel`'s `torch.randn_like(codewords)` call."""
    torch.manual_seed(0)
    msg_bits = 2 * (torch.rand(4, 4) < 0.5).float() - 1
    channel_noise = torch.randn(4, 8)
    return (msg_bits, channel_noise)


MENAGERIE_ENTRIES = [
    ("KO codes", "build_kocodes", "example_input_kocodes", 2021, MENAGERIE_ZOO),
]
