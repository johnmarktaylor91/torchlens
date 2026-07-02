# SOURCE: vendored from yihanjiang/turboae @ master
# https://github.com/yihanjiang/turboae (cnn_utils.py, interleavers.py, encoders.py::ENC_interCNN,
# decoders.py::DEC_LargeCNN)
# TurboAE: NeurIPS 2019 "Turbo Autoencoder" -- CNN encoder/decoder learned turbo-style
# channel code (Jiang, Kim, Asnani, Kannan, Oh, Viswanath). This vendors the canonical
# `TurboAE_rate3_cnn` encoder (ENC_interCNN) and iterative CNN turbo decoder (DEC_LargeCNN)
# verbatim from the official repo -- these are the real encoder/decoder classes used by
# `channel_ae.py::Channel_AE`, just called directly here (no channel-noise simulation,
# no commpy dependency) since TorchLens only needs the codec forward pass.
#
# NOTE: cand rows "Turbo-Autoencoder" and "TurboAE" in the discovery queue are the SAME
# NeurIPS 2019 paper / same repo (yihanjiang/turboae) -- POTENTIAL_DEDUP. Registered ONCE
# here under the canonical name "TurboAE" to avoid a duplicate architecture entry.
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# cnn_utils.py (verbatim, SameShapeConv1d only -- the piece ENC_interCNN/DEC_LargeCNN use)
# ---------------------------------------------------------------------------
class SameShapeConv1d(torch.nn.Module):
    def __init__(
        self, num_layer, in_channels, out_channels, kernel_size, activation="elu", no_act=False
    ):
        super(SameShapeConv1d, self).__init__()

        self.cnns = torch.nn.ModuleList()
        self.num_layer = num_layer
        self.no_act = no_act
        for idx in range(num_layer):
            if idx == 0:
                self.cnns.append(
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size // 2),
                        dilation=1,
                        groups=1,
                        bias=True,
                    )
                )
            else:
                self.cnns.append(
                    torch.nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size // 2),
                        dilation=1,
                        groups=1,
                        bias=True,
                    )
                )

        if activation == "elu":
            self.activation = F.elu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "selu":
            self.activation = F.selu
        elif activation == "prelu":
            self.activation = F.prelu
        else:
            self.activation = F.elu

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)
        x = inputs
        for idx in range(self.num_layer):
            if self.no_act:
                x = self.cnns[idx](x)
            else:
                x = self.activation(self.cnns[idx](x))

        outputs = torch.transpose(x, 1, 2)
        return outputs


# ---------------------------------------------------------------------------
# interleavers.py (verbatim, Interleaver / DeInterleaver)
# ---------------------------------------------------------------------------
class Interleaver(torch.nn.Module):
    def __init__(self, args, p_array):
        super(Interleaver, self).__init__()
        self.args = args
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

    def set_parray(self, p_array):
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        res = inputs[self.p_array]
        res = res.permute(1, 0, 2)

        return res


class DeInterleaver(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DeInterleaver, self).__init__()
        self.args = args

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(len(p_array))

    def set_parray(self, p_array):
        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(len(p_array))

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        res = inputs[self.reverse_p_array]
        res = res.permute(1, 0, 2)

        return res


# ---------------------------------------------------------------------------
# encoders.py::ENCBase + ENC_interCNN (verbatim; power_constraint/enc_act kept as-is)
# ---------------------------------------------------------------------------
class ENCBase(torch.nn.Module):
    def __init__(self, args):
        super(ENCBase, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.reset_precomp()

    def set_parallel(self):
        pass

    def set_precomp(self, mean_scalar, std_scalar):
        self.mean_scalar = mean_scalar.to(self.this_device)
        self.std_scalar = std_scalar.to(self.this_device)

    def reset_precomp(self):
        self.mean_scalar = torch.zeros(1).type(torch.FloatTensor).to(self.this_device)
        self.std_scalar = torch.ones(1).type(torch.FloatTensor).to(self.this_device)
        self.num_test_block = 0.0

    def enc_act(self, inputs):
        if self.args.enc_act == "tanh":
            return F.tanh(inputs)
        elif self.args.enc_act == "elu":
            return F.elu(inputs)
        elif self.args.enc_act == "relu":
            return F.relu(inputs)
        elif self.args.enc_act == "selu":
            return F.selu(inputs)
        elif self.args.enc_act == "sigmoid":
            return F.sigmoid(inputs)
        elif self.args.enc_act == "linear":
            return inputs
        else:
            return inputs

    def power_constraint(self, x_input):
        if self.args.no_code_norm:
            return x_input
        else:
            this_mean = torch.mean(x_input)
            this_std = torch.std(x_input)

            if self.args.precompute_norm_stats:
                self.num_test_block += 1.0
                self.mean_scalar = (
                    self.mean_scalar * (self.num_test_block - 1) + this_mean
                ) / self.num_test_block
                self.std_scalar = (
                    self.std_scalar * (self.num_test_block - 1) + this_std
                ) / self.num_test_block
                x_input_norm = (x_input - self.mean_scalar) / self.std_scalar
            else:
                x_input_norm = (x_input - this_mean) * 1.0 / this_std

            if self.args.enc_truncate_limit > 0:
                x_input_norm = torch.clamp(
                    x_input_norm, -self.args.enc_truncate_limit, self.args.enc_truncate_limit
                )

            return x_input_norm


class ENC_interCNN(ENCBase):
    def __init__(self, args, p_array):
        # turbofy only for code rate 1/3
        super(ENC_interCNN, self).__init__(args)
        self.args = args

        # Encoder (TurboAE_rate3_cnn branch -- SameShapeConv1d)
        self.enc_cnn_1 = SameShapeConv1d(
            num_layer=args.enc_num_layer,
            in_channels=args.code_rate_k,
            out_channels=args.enc_num_unit,
            kernel_size=args.enc_kernel_size,
        )

        self.enc_cnn_2 = SameShapeConv1d(
            num_layer=args.enc_num_layer,
            in_channels=args.code_rate_k,
            out_channels=args.enc_num_unit,
            kernel_size=args.enc_kernel_size,
        )

        self.enc_cnn_3 = SameShapeConv1d(
            num_layer=args.enc_num_layer,
            in_channels=args.code_rate_k,
            out_channels=args.enc_num_unit,
            kernel_size=args.enc_kernel_size,
        )

        self.enc_linear_1 = torch.nn.Linear(args.enc_num_unit, 1)
        self.enc_linear_2 = torch.nn.Linear(args.enc_num_unit, 1)
        self.enc_linear_3 = torch.nn.Linear(args.enc_num_unit, 1)

        self.interleaver = Interleaver(args, p_array)

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def set_parallel(self):
        self.enc_cnn_1 = torch.nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = torch.nn.DataParallel(self.enc_cnn_2)
        self.enc_cnn_3 = torch.nn.DataParallel(self.enc_cnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    def forward(self, inputs):
        # (is_variable_block_len branch omitted: args.is_variable_block_len=False by default)
        inputs = 2.0 * inputs - 1.0
        x_sys = self.enc_cnn_1(inputs)
        x_sys = self.enc_act(self.enc_linear_1(x_sys))

        x_p1 = self.enc_cnn_2(inputs)
        x_p1 = self.enc_act(self.enc_linear_2(x_p1))

        x_sys_int = self.interleaver(inputs)
        x_p2 = self.enc_cnn_3(x_sys_int)
        x_p2 = self.enc_act(self.enc_linear_3(x_p2))

        x_tx = torch.cat([x_sys, x_p1, x_p2], dim=2)

        codes = self.power_constraint(x_tx)

        return codes


# ---------------------------------------------------------------------------
# decoders.py::DEC_LargeCNN (verbatim)
# ---------------------------------------------------------------------------
class DEC_LargeCNN(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DEC_LargeCNN, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.interleaver = Interleaver(args, p_array)
        self.deinterleaver = DeInterleaver(args, p_array)

        self.dec1_cnns = torch.nn.ModuleList()
        self.dec2_cnns = torch.nn.ModuleList()
        self.dec1_outputs = torch.nn.ModuleList()
        self.dec2_outputs = torch.nn.ModuleList()

        CNNLayer = SameShapeConv1d

        for idx in range(args.num_iteration):
            self.dec1_cnns.append(
                CNNLayer(
                    num_layer=args.dec_num_layer,
                    in_channels=2 + args.num_iter_ft,
                    out_channels=args.dec_num_unit,
                    kernel_size=args.dec_kernel_size,
                )
            )

            self.dec2_cnns.append(
                CNNLayer(
                    num_layer=args.dec_num_layer,
                    in_channels=2 + args.num_iter_ft,
                    out_channels=args.dec_num_unit,
                    kernel_size=args.dec_kernel_size,
                )
            )
            self.dec1_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))

            if idx == args.num_iteration - 1:
                self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, 1))
            else:
                self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))

    def set_parallel(self):
        for idx in range(self.args.num_iteration):
            self.dec1_cnns[idx] = torch.nn.DataParallel(self.dec1_cnns[idx])
            self.dec2_cnns[idx] = torch.nn.DataParallel(self.dec2_cnns[idx])
            self.dec1_outputs[idx] = torch.nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = torch.nn.DataParallel(self.dec2_outputs[idx])

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)
        self.deinterleaver.set_parray(p_array)

    def forward(self, received):
        block_len = self.args.block_len

        received = received.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys = received[:, :, 0].view((self.args.batch_size, block_len, 1))
        r_sys_int = self.interleaver(r_sys)
        r_par1 = received[:, :, 1].view((self.args.batch_size, block_len, 1))
        r_par2 = received[:, :, 2].view((self.args.batch_size, block_len, 1))

        prior = torch.zeros((self.args.batch_size, block_len, self.args.num_iter_ft)).to(
            self.this_device
        )

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim=2)

            x_dec = self.dec1_cnns[idx](x_this_dec)
            x_plr = self.dec1_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int], dim=2)

            x_dec = self.dec2_cnns[idx](x_this_dec)

            x_plr = self.dec2_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys, r_par1, prior], dim=2)

        x_dec = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int], dim=2)

        x_dec = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        final = torch.sigmoid(self.deinterleaver(x_plr))

        return final


# ---------------------------------------------------------------------------
# channel_ae.py::Channel_AE, trimmed to the noiseless-passthrough codec (the AWGN-channel
# noise injection is training/eval-time stochastic simulation, not part of the encoder or
# decoder network architecture -- omitted here since TorchLens traces the codec module).
# ---------------------------------------------------------------------------
class TurboAE(torch.nn.Module):
    """TurboAE_rate3_cnn codec: real ENC_interCNN + DEC_LargeCNN composed as in
    channel_ae.py::Channel_AE.forward, with a fixed (identity) interleaver permutation
    and the AWGN channel modeled as a no-noise passthrough so the module is a pure
    torch.nn.Module suitable for architecture tracing."""

    def __init__(self, args):
        super(TurboAE, self).__init__()
        self.args = args
        rand_gen = np.random.RandomState(0)
        p_array = rand_gen.permutation(np.arange(args.block_len))
        self.enc = ENC_interCNN(args, p_array)
        self.dec = DEC_LargeCNN(args, p_array)

    def forward(self, inputs):
        codes = self.enc(inputs)
        x_dec = self.dec(codes)
        return x_dec


def build_turboae():
    args = SimpleNamespace(
        encoder="TurboAE_rate3_cnn",
        no_cuda=True,
        enc_act="elu",
        no_code_norm=False,
        precompute_norm_stats=False,
        enc_truncate_limit=0,
        code_rate_k=1,
        code_rate_n=3,
        enc_num_layer=2,
        enc_num_unit=100,
        enc_kernel_size=5,
        dec_num_layer=5,
        dec_num_unit=100,
        dec_kernel_size=5,
        num_iteration=6,
        num_iter_ft=5,
        extrinsic=1,
        block_len=100,
        batch_size=1,
    )
    return TurboAE(args)


def example_input_turboae():
    # (batch_size, block_len, code_rate_k) -- raw information bits
    return torch.randint(0, 2, (1, 100, 1)).float()


MENAGERIE_ENTRIES = [
    ("TurboAE", build_turboae, example_input_turboae, 2019, "SOURCE_AVAILABLE"),
]
