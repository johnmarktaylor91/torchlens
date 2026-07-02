# SOURCE: vendored from
# https://github.com/tonyzyl/Semisupervised-VAE-for-Regression-Application-on-Soft-Sensor
# @ d996cb5d (models/ssae.py)
#
# Semi-supervised Stacked AutoEncoder (SSAE) soft sensor for industrial
# quality-variable regression. A stack of shallow tanh autoencoders (`AE`,
# each `Linear(in, latent) -> tanh -> Linear(latent, in) -> tanh` for
# reconstruction plus a parallel `Linear(latent, 1)` head for the quality
# label) chained latent-to-input greedily -- the classic deep/stacked
# autoencoder soft-sensor architecture referenced by the queue's "DSAE Soft
# Sensor (Deep Supervised Autoencoder)" candidate. `SSAE.forward` in
# inference mode (`preTrain=False`, the path used here) runs the full stack
# encoder-only through every AE except the last (`decode=False`, discarding
# the reconstruction head) and returns the final AE's
# (latent, quality-estimate) pair from a full `decode=True` pass --
# unmodified real repo code.
#
# Only the training-loop / dataset / optimizer / pickle-based
# `SSAE_Trainer`/`MyDataset`/`loss_func` machinery was dropped (none of that
# is architecture) and the unused `from .scheduler import
# CosineAnnealingWarmupRestarts` relative import was removed since the
# scheduler class is never referenced by `AE` or `SSAE` themselves. No layer,
# activation, or dataflow inside `AE`/`SSAE` was changed.

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        self.y_layer = nn.Linear(latent_dim, 1, bias=True)

        nn.init.xavier_uniform_(self.encoder.weight.data)
        nn.init.xavier_uniform_(self.decoder.weight.data)
        nn.init.xavier_normal_(self.y_layer.weight.data)

    def forward(self, X, decode=True):
        # Return (latent/decode output, Y estimate by dense)
        H = torch.tanh(self.encoder(X))
        if decode:
            return torch.tanh(self.decoder(H)), self.y_layer(H)
        else:
            return H, self.y_layer(H)


class SSAE(nn.Module):
    def __init__(self, AE_list):
        super().__init__()
        self.num_AE = len(AE_list)
        self.SAE_list = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i in range(1, self.num_AE + 1):
            if i != self.num_AE:
                self.SAE_list.append(AE(AE_list[i - 1], AE_list[i]).to(self.device))
            else:
                self.SAE_list.append(AE(AE_list[-1], AE_list[-1]).to(self.device))
        # Real repo stores stage AEs in a plain python list (`self.SAE_list`)
        # populated post-`__init__`; register as an nn.ModuleList too so the
        # tiny-config parameters are discoverable/traceable as submodules
        # without altering the forward computation at all.
        self._SAE_modulelist = nn.ModuleList(self.SAE_list)

    def wgtFromList(self, wts_list):
        for i in range(self.num_AE):
            self.SAE_list[i].load_state_dict(wts_list[i])

    def forward(self, x, layer_idx, preTrain=False):
        # preTrain: previous layers' parameters are frozen
        # preTrain -> Return (input, AE_output, y estimate)
        # !preTrain -> Return last layer's (latent, estimate)
        output = x
        if preTrain:
            if layer_idx == 0:
                inputs = output
                output, y_estimate = self.SAE_list[layer_idx](output, decode=True)
                return inputs, output, y_estimate

            else:
                for i in range(layer_idx):
                    for param in self.SAE_list[i].parameters():
                        param.requires_grad = False
                    output, _ = self.SAE_list[i](output, decode=False)
                inputs = output
                output, y_estimate = self.SAE_list[layer_idx](output, decode=True)
                return inputs, output, y_estimate
        else:
            for i in range(self.num_AE - 1):
                for param in self.SAE_list[i].parameters():
                    param.requires_grad = True
                output, _ = self.SAE_list[i](output, decode=False)
            return self.SAE_list[-1](output, decode=False)


def build_dsae_soft_sensor():
    # Tiny config: 4 process-variable inputs -> stacked latent widths
    # [4, 8, 4, 2], matching the real repo's `AE_list` layer-width convention
    # consumed by `SSAE.__init__`.
    model = SSAE(AE_list=[4, 8, 4, 2])
    return model


def example_input_dsae_soft_sensor():
    # forward(x, layer_idx, preTrain=False) -- inference path used by the
    # real repo's finetune/eval loop (`self.model(inputs, self.model.num_AE-1,
    # preTrain=False)`), so layer_idx is fixed at the final stage index.
    x = torch.randn(3, 4)
    layer_idx = 3  # num_AE - 1 for AE_list=[4, 8, 4, 2] (num_AE=len(AE_list)=4)
    return (x, layer_idx)


MENAGERIE_ENTRIES = [
    (
        "DSAE-Soft-Sensor",
        build_dsae_soft_sensor,
        example_input_dsae_soft_sensor,
        2022,
        MENAGERIE_ZOO,
    ),
]
