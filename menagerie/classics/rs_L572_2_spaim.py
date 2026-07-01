# SOURCE: vendored from QSong-github/SpaIM @ main
# Original: src/model.py (mlp_simple, Imputation)
"""SpaIM: style-transfer autoencoder for spatial transcriptomics imputation.

SpaIM imputes missing spatial-transcriptomics (ST) gene expression from
paired single-cell RNA-seq (scRNA-seq) reference data using a style-transfer
framework: separate "content" and "style" encoders project ST and scRNA-seq
expression profiles into a shared latent space, then a shared decoder
reconstructs ST expression from a scRNA-seq content code combined with an ST
style code (a cross-modal content/style disentanglement, in the spirit of
AdaIN-style image style transfer, applied to gene-expression vectors).

Reference: https://github.com/QSong-github/SpaIM
"""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from src/model.py ---


class mlp_simple(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.l = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, use_norm=True):
        x = self.l(x)
        if use_norm:
            x = self.norm(x)
            x = self.relu(x)
        return x


class Imputation(nn.Module):
    def __init__(self, scdim, stdim, style_dim, hidden_dims):
        super().__init__()
        h2, h1 = hidden_dims
        self.st_enc1_cont = mlp_simple(stdim, h1)
        self.st_enc2_cont = mlp_simple(h1, h2)

        self.st_enc1_style = mlp_simple(stdim, h1)
        self.st_enc2_style = mlp_simple(h1, h2)

        self.st_dec2 = mlp_simple(h2, h1)
        self.st_dec1 = mlp_simple(h1, stdim)

        self.sc_enc2_cont = mlp_simple(scdim, h2)
        self.sc_enc1_cont = mlp_simple(h2, h1)

        self.enc_style2 = mlp_simple(style_dim, h2)
        self.enc_style1 = mlp_simple(style_dim, h1)

    def forward(self, sc, st, scstyle, ststyle, istrain=1):
        if istrain:
            # generate st cont
            st_cont1 = self.st_enc1_cont(st)
            st_cont2 = self.st_enc2_cont(st_cont1)

            # generate st style
            st_style1 = self.st_enc1_style(st)
            st_style2 = self.st_enc2_style(st_style1)

            # generate sc cont
            sc_cont2 = self.sc_enc2_cont(sc)
            sc_cont1 = self.sc_enc1_cont(sc_cont2)

            # generate fake style
            fake_style2 = self.enc_style2(ststyle)
            fake_style1 = self.enc_style1(ststyle)

            # real
            real_st_up2 = self.st_dec2(st_cont2 * st_style2)
            real_st_up1 = self.st_dec1(real_st_up2 + st_cont1 * st_style1, use_norm=False)

            # fake
            fake_st_up2 = self.st_dec2(sc_cont2 * fake_style2)
            fake_st_up1 = self.st_dec1(fake_st_up2 + sc_cont1 * fake_style1, use_norm=False)

            return {
                "st_cont1": st_cont1,
                "st_cont2": st_cont2,
                "sc_cont1": sc_cont1,
                "sc_cont2": sc_cont2,
                "st_style1": st_style1,
                "st_style2": st_style2,
                "fake_style1": fake_style1,
                "fake_style2": fake_style2,
                "st_real": real_st_up1,
                "st_fake": fake_st_up1,
            }

        else:
            # only have sc and ststyle
            # generate st_cont
            sc_cont2 = self.sc_enc2_cont(sc)
            sc_cont1 = self.sc_enc1_cont(sc_cont2)

            fake_style2 = self.enc_style2(ststyle)
            fake_style1 = self.enc_style1(ststyle)

            fake_st_up2 = self.st_dec2(sc_cont2 * fake_style2)
            fake_st_up1 = self.st_dec1(fake_st_up2 + sc_cont1 * fake_style1, use_norm=False)

            return {"st_fake": fake_st_up1}


# --- staging: tiny-size builder + example (multi-tensor) input ---

_SC_DIM = 20
_ST_DIM = 16
_STYLE_DIM = 8
_H1 = 12
_H2 = 6


def build_spaim():
    model = Imputation(
        scdim=_SC_DIM,
        stdim=_ST_DIM,
        style_dim=_STYLE_DIM,
        hidden_dims=(_H2, _H1),
    )
    model.eval()
    return model


def example_input_spaim():
    # (sc expression, st expression, sc style, st style); istrain=1 (the
    # training-mode forward) exercises both the content and style encoders
    # plus the shared decoder, matching the real Imputation.forward() call.
    sc = torch.rand(4, _SC_DIM)
    st = torch.rand(4, _ST_DIM)
    scstyle = torch.rand(4, _STYLE_DIM)
    ststyle = torch.rand(4, _STYLE_DIM)
    return sc, st, scstyle, ststyle, 1


MENAGERIE_ENTRIES = [
    (
        "SpaIM",
        "build_spaim",
        "example_input_spaim",
        2024,
        "vendored-pytorch",
    ),
]
