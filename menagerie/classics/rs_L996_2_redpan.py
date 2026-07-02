# SOURCE: vendored from tso1257771/RED-PAN @ main (commit c59534c940fdb1265403ae99e69fc21f05e3a2f4)
# https://raw.githubusercontent.com/tso1257771/RED-PAN/main/redpan/legacy/torch_mtan_ARRU.py
#
# Chin et al. 2022 (IEEE TGRS) "REal-time Decision fusion and Picking with Attention
# Network (RED-PAN)": a multi-task-attention residual-recurrent U-Net (`mtan_R2unet`)
# over 1D seismic waveforms that jointly predicts (1) P/S phase-pick probabilities
# (softmax over {P, S, noise}) and (2) an earthquake-detection mask (softmax over
# {earthquake, noise}), sharing a common recurrent-residual convolutional backbone
# with two lightweight task-specific attention sub-networks (`mtan_att_block`) that
# gate the backbone features at each encoder/decoder scale -- the actual
# multi-task-attention-network (MTAN, Liu et al. 2019) mechanism this repo implements
# for seismology. This is the only PyTorch model definition in the upstream repo (the
# production `redpan/models.py` used by `redpan/core.py`/`redpan/factory.py` at
# inference time is TensorFlow/Keras); this `torch_mtan_ARRU.py` legacy file is real,
# complete, standalone PyTorch source for the identical published architecture.
#
# Copied verbatim (module bodies, forward logic, and the published default channel
# schedule `n_chn = [3, 6, 12, 18, 24, 30, 36]` all unchanged). Only mechanical,
# import-isolation changes were made:
#   - Dropped the unused `import torch.optim as optim` (training-only, referenced only
#     in a commented-out line in upstream's own `if __name__ == "__main__":` block).
#   - Dropped the module-level `x = torch.randn(50, 3, 6000)` side-effect statement
#     that ran at import time in upstream (harmless numerically, but constructs a
#     throwaway tensor on import for no reason); the equivalent example tensor is
#     built by `example_input_redpan_mtan_r2unet()` below instead.
#   - Dropped the `if __name__ == "__main__":` smoke-test block (device selection +
#     print(model)); replaced by the `build_redpan_mtan_r2unet()`/
#     `example_input_redpan_mtan_r2unet()` staging pair.
# No architecture code (channel counts, kernel sizes, the RRconv recurrent-residual
# loop, the mtan attention gating, the encoder/decoder wiring) was altered.

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=None):
        super(conv_block, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.conv1d(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class RRconv_block(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding=1, stride=None, RRconv_time=3
    ):
        super(RRconv_block, self).__init__()

        self.RRconv_time = RRconv_time
        self.conv_block_init = conv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

        self.conv_1x1 = nn.Conv1d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1
        )

        self.conv_block_res = conv_block(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1
        )

    def forward(self, x):
        conv_out = self.conv_block_init(x)
        conv_out_1x1 = self.conv_1x1(conv_out)

        for i in range(self.RRconv_time):
            if i == 0:
                res_unit = conv_out
            res_unit += conv_out
            res_unit = self.conv_block_res(res_unit)
        RRconv_out = res_unit + conv_out_1x1
        return RRconv_out


class upconv_concat_RRblock(nn.Module):
    def __init__(self, in_channels, out_channels, upsize=5, RRconv_time=3):
        super(upconv_concat_RRblock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=upsize)
        self.conv_block = conv_block(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=1, stride=1
        )
        self.conv_skip_conncet = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        self.RRconv_block = RRconv_block(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            RRconv_time=RRconv_time,
        )

    def forward(self, target_layer=None, concat_layer=None):
        up_conv_out = self.upsample(target_layer)
        up_conv_out = self.conv_block(up_conv_out)
        skip_connect = torch.cat([up_conv_out[:, :, : concat_layer.shape[2]], concat_layer], dim=1)
        conca_out = self.conv_skip_conncet(skip_connect)
        RR_out = self.RRconv_block(conca_out)
        return RR_out


class att_layer(nn.Module):
    def __init__(self, channel):
        super(att_layer, self).__init__()

        self.batchnorm = nn.BatchNorm1d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1x1 = nn.Conv1d(
            in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.batchnorm(x)
        out = self.relu(out)
        out = self.conv_1x1(out)
        out = self.batchnorm(out)
        out = self.sigmoid(out)
        return out


class mtan_att_block(nn.Module):
    def __init__(self, in_channels, out_channels, upsize=5, stride=5, mode=None):
        super(mtan_att_block, self).__init__()

        # channels number of previous att/backbone layer
        self.in_chanels = in_channels
        # channels number of current backbone layer
        self.out_channels = out_channels
        self.mode = mode

        if mode == "up":
            self.init_conv = nn.Sequential(
                nn.Upsample(scale_factor=upsize),
                conv_block(
                    in_channels=in_channels * 2,  # due to concatenation
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=1,
                    stride=1,
                ),
            )
        if mode == "down":
            self.init_conv = nn.Conv1d(
                in_channels=in_channels * 2,  # due to concatenation
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                stride=stride,
            )

        self.att_layer = att_layer(channel=self.out_channels)

    def forward(self, pre_att_layer=None, pre_target_layer=None, target_layer=None):
        pre_layer_concat = torch.cat([pre_att_layer, pre_target_layer], dim=1)
        # attention layer
        pre_conv_init = self.init_conv(pre_layer_concat)
        att = self.att_layer(pre_conv_init)

        ## element-wise multiplication with target layer
        att_gate = torch.multiply(att[:, :, : target_layer.shape[2]], target_layer)

        return att_gate


class mtan_R2unet(nn.Module):
    def __init__(
        self,
        # input_size = (batch, 3, 6000),
        n_chn=[3, 6, 12, 18, 24, 30, 36],
        dropout_rate=0.1,
        kernel_size=5,
        stride=5,
        upsize=5,
        RRconv_time=3,
    ):
        super(mtan_R2unet, self).__init__()
        self.n_chn = n_chn
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.upsize = upsize
        self.RRconv_time = RRconv_time

        ## register encoder modules
        self.enc_conv_block = nn.ModuleList()
        self.enc_PS_att_block = nn.ModuleList()
        self.enc_eqM_att_block = nn.ModuleList()
        for i in range(len(self.n_chn) - 1):
            if i == 0:
                # initialize backbone
                self.enc_conv_block.append(
                    nn.Sequential(
                        conv_block(
                            in_channels=self.n_chn[i],
                            out_channels=self.n_chn[i + 1],
                            kernel_size=1,
                            padding=0,
                            stride=1,
                        ),
                        RRconv_block(
                            in_channels=self.n_chn[i + 1],
                            out_channels=self.n_chn[i + 1],
                            kernel_size=1,
                            padding=0,
                            stride=1,
                            RRconv_time=self.RRconv_time,
                        ),
                    )
                )
                # initialize sub-attention network: PS picking
                self.enc_PS_att_block.append(
                    mtan_att_block(
                        in_channels=self.n_chn[i + 1],
                        out_channels=self.n_chn[i + 1],
                        stride=1,
                        mode="down",
                    )
                )
                # initialize sub-attention network: EQ masking
                self.enc_eqM_att_block.append(
                    mtan_att_block(
                        in_channels=self.n_chn[i + 1],
                        out_channels=self.n_chn[i + 1],
                        stride=1,
                        mode="down",
                    )
                )
            else:
                # backbone
                self.enc_conv_block.append(
                    nn.Sequential(
                        conv_block(
                            in_channels=self.n_chn[i],
                            out_channels=self.n_chn[i + 1],
                            kernel_size=self.kernel_size,
                            padding=1,
                            stride=self.stride,
                        ),
                        RRconv_block(
                            in_channels=self.n_chn[i + 1],
                            out_channels=self.n_chn[i + 1],
                            kernel_size=1,
                            padding=0,
                            stride=1,
                            RRconv_time=self.RRconv_time,
                        ),
                    )
                )
                # sub-attention network: PS picking
                self.enc_PS_att_block.append(
                    mtan_att_block(
                        in_channels=self.n_chn[i],
                        out_channels=self.n_chn[i + 1],
                        stride=self.stride,
                        mode="down",
                    )
                )
                # sub-attention network: EQ masking
                self.enc_eqM_att_block.append(
                    mtan_att_block(
                        in_channels=self.n_chn[i],
                        out_channels=self.n_chn[i + 1],
                        stride=self.stride,
                        mode="down",
                    )
                )

        ## register decoder modules
        self.dec_conv_block = nn.ModuleList()
        self.dec_PS_att_block = nn.ModuleList()
        self.dec_eqM_att_block = nn.ModuleList()
        for j in range(len(self.n_chn) - 2):
            # backbone
            self.dec_conv_block.append(
                upconv_concat_RRblock(
                    in_channels=self.n_chn[-j - 1],
                    out_channels=self.n_chn[-j - 2],
                    upsize=self.upsize,
                )
            )

            # sub-attention network: PS picking
            self.dec_PS_att_block.append(
                mtan_att_block(
                    in_channels=self.n_chn[-j - 1],
                    out_channels=self.n_chn[-j - 2],
                    stride=self.stride,
                    mode="up",
                )
            )
            # sub-attention network: EQ masking
            self.dec_eqM_att_block.append(
                mtan_att_block(
                    in_channels=self.n_chn[-j - 1],
                    out_channels=self.n_chn[-j - 2],
                    stride=self.stride,
                    mode="up",
                )
            )

        # output conv layer
        self.PS_out_conv = nn.Conv1d(
            in_channels=self.n_chn[1], out_channels=3, stride=1, padding=0, kernel_size=1
        )
        self.eqM_out_conv = nn.Conv1d(
            in_channels=self.n_chn[1], out_channels=2, stride=1, padding=0, kernel_size=1
        )

    def forward(self, x):
        exp_Es = []
        PS_mtan_Es = []
        eqM_mtan_Es = []
        ## encoder
        for i in range(len(self.n_chn) - 1):
            if i == 0:
                # backbone
                exp_E = self.enc_conv_block[i](x)
                # sub-attention network: PS picking
                PS_mtan_E = self.enc_PS_att_block[i](
                    pre_att_layer=exp_E, pre_target_layer=exp_E, target_layer=exp_E
                )
                # sub-attention network: EQ masking
                eqM_mtan_E = self.enc_eqM_att_block[i](
                    pre_att_layer=exp_E, pre_target_layer=exp_E, target_layer=exp_E
                )
            else:
                # backbone
                exp_E = self.enc_conv_block[i](exp_Es[-1])
                # sub-attention network: PS picking
                PS_mtan_E = self.enc_PS_att_block[i](
                    pre_att_layer=PS_mtan_Es[-1], pre_target_layer=exp_Es[-1], target_layer=exp_E
                )
                # sub-attention network: EQ masking
                eqM_mtan_E = self.enc_eqM_att_block[i](
                    pre_att_layer=PS_mtan_Es[-1], pre_target_layer=exp_Es[-1], target_layer=exp_E
                )
            exp_Es.append(exp_E)
            PS_mtan_Es.append(PS_mtan_E)
            eqM_mtan_Es.append(eqM_mtan_E)

        ## decoder
        exp_Ds = []
        PS_mtan_Ds = []
        eqM_mtan_Ds = []
        for j in range(len(self.n_chn) - 2):
            if j == 0:
                # backbone
                exp_D = self.dec_conv_block[j](target_layer=exp_Es[-1], concat_layer=exp_Es[-j - 2])
                # sub-attention network: PS picking
                PS_mtan_D = self.dec_PS_att_block[j](
                    pre_att_layer=PS_mtan_Es[-1], pre_target_layer=exp_Es[-1], target_layer=exp_D
                )
                # sub-attention network: EQ masking
                eqM_mtan_D = self.dec_eqM_att_block[j](
                    pre_att_layer=PS_mtan_Es[-1], pre_target_layer=exp_Es[-1], target_layer=exp_D
                )
            else:
                # backbone
                exp_D = self.dec_conv_block[j](target_layer=exp_Ds[-1], concat_layer=exp_Es[-j - 2])
                # sub-attention network: PS picking
                PS_mtan_D = self.dec_PS_att_block[j](
                    pre_att_layer=PS_mtan_Ds[-1], pre_target_layer=exp_Ds[-1], target_layer=exp_D
                )
                # sub-attention network: EQ masking
                eqM_mtan_D = self.dec_eqM_att_block[j](
                    pre_att_layer=PS_mtan_Ds[-1], pre_target_layer=exp_Ds[-1], target_layer=exp_D
                )
            exp_Ds.append(exp_D)
            PS_mtan_Ds.append(PS_mtan_D)
            eqM_mtan_Ds.append(eqM_mtan_D)

        # outputs
        out_PS = self.PS_out_conv(PS_mtan_Ds[-1])
        out_eqM = self.eqM_out_conv(eqM_mtan_Ds[-1])

        out_PS_softmax = F.softmax(out_PS, dim=1)
        out_eqM_softmax = F.softmax(out_eqM, dim=1)

        return out_PS_softmax, out_eqM_softmax


def build_redpan_mtan_r2unet():
    return mtan_R2unet()


def example_input_redpan_mtan_r2unet():
    # batch_size, channel(3-component seismogram), data_length -- matches upstream's
    # own smoke-test tensor `torch.randn(50, 3, 6000)`, batch trimmed to 2 for a fast
    # menagerie trace.
    return (torch.randn(2, 3, 6000),)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "RED-PAN (mtan_R2unet)",
        "build_redpan_mtan_r2unet",
        "example_input_redpan_mtan_r2unet",
        2022,
        "vendored-pytorch",
    ),
]
