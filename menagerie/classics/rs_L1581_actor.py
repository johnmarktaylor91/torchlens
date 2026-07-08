# SOURCE: vendored from Mathux/ACTOR @ master (files: src/models/architectures/transformer.py,
# src/models/modeltype/cae.py, src/models/modeltype/cvae.py -- fetched 2026-07-02).
#
# This vendors the REAL ACTOR architecture (Petrovich, Black & Varol, ICCV 2021,
# "Action-Conditioned 3D Human Motion Synthesis with Transformer VAE"): a Transformer
# encoder (`Encoder_TRANSFORMER`) that maps a variable-length pose sequence to a Gaussian
# latent via learned mu/sigma query tokens prepended to the sequence, and a Transformer
# decoder (`Decoder_TRANSFORMER`) that reconstructs the pose sequence from the sampled latent
# (shifted by a learned per-action bias) attending over per-timestep positional queries. The
# `CVAE` wrapper reparameterizes the encoder's (mu, logvar) and wires encoder->decoder exactly
# as upstream.
#
# NOT vendored: `Rotation2xyz` (SMPL body-model conversion; requires external SMPL asset data
# not part of the network's own trainable weights -- traced here with outputxyz=False /
# pose_rep="rot6d" so the traced forward pass is the exact real encoder+decoder+reparameterize
# path, matching upstream's own `pose_rep != "xyz"` branch that also skips SMPL conversion).
#
# Code below is the upstream source with only mechanical edits: cross-file imports flattened
# into this single module, the CAE.__init__ kwargs trimmed to what the traced forward path
# needs (rotation2xyz / compute_loss / generate are training-time-only utilities, kept for
# completeness where cheap, dropped where they'd require SMPL assets), everything else
# untouched.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# src/models/architectures/transformer.py
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class Encoder_TRANSFORMER(nn.Module):
    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        num_frames,
        num_classes,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
        **kargs,
    ):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        if self.ablation == "average_encoder":
            self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.sigma_layer = nn.Linear(self.latent_dim, self.latent_dim)
        else:
            self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
            self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers
        )

    def forward(self, batch):
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        # embedding of the skeleton
        x = self.skelEmbedding(x)

        if self.ablation == "average_encoder":
            x = self.sequence_pos_encoder(x)
            final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
            z = final.mean(axis=0)
            mu = self.mu_layer(z)
            logvar = self.sigma_layer(z)
        else:
            # adding the mu and sigma queries
            xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)

            # add positional encoding
            xseq = self.sequence_pos_encoder(xseq)

            # create a bigger mask, to allow attend to mu and sigma
            muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
            maskseq = torch.cat((muandsigmaMask, mask), axis=1)

            final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
            mu = final[0]
            logvar = final[1]

        return {"mu": mu, "logvar": logvar}


class Decoder_TRANSFORMER(nn.Module):
    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        num_frames,
        num_classes,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        ablation=None,
        **kargs,
    ):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        if self.ablation == "zandtime":
            self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
        else:
            self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=activation,
        )
        self.seqTransDecoder = nn.TransformerDecoder(
            seqTransDecoderLayer, num_layers=self.num_layers
        )

        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, batch):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]  # noqa: F841 (lengths unused in this ablation branch, as upstream)

        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        if self.ablation == "zandtime":
            yoh = F.one_hot(y, self.num_classes)
            z = torch.cat((z, yoh), axis=1)
            z = self.ztimelinear(z)
            z = z[None]
        else:
            if self.ablation == "concat_bias":
                z = torch.stack((z, self.actionBiases[y]), axis=0)
            else:
                # shift the latent noise vector to be the action noise
                z = z + self.actionBiases[y]
                z = z[None]

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)

        output = self.seqTransDecoder(tgt=timequeries, memory=z, tgt_key_padding_mask=~mask)

        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)

        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)

        batch["output"] = output
        return batch


# ---------------------------------------------------------------------------
# src/models/modeltype/cae.py + cvae.py (subset: forward/reparameterize path,
# outputxyz/rotation2xyz branch dropped -- traced with pose_rep != "xyz")
# ---------------------------------------------------------------------------
class CVAE(nn.Module):
    """Transformer VAE: Encoder_TRANSFORMER -> reparameterize -> Decoder_TRANSFORMER,
    exactly `CVAE(CAE)`'s forward path from upstream (SMPL rot2xyz branch skipped since
    pose_rep != "xyz", matching the real conditional in upstream's own CAE.forward)."""

    def __init__(self, encoder, decoder, latent_dim, pose_rep):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.outputxyz = False

    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)
        eps = std.data.new(std.size()).normal_()
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, batch):
        # encode
        batch.update(self.encoder(batch))
        batch["z"] = self.reparameterize(batch)

        # decode
        batch.update(self.decoder(batch))

        return batch


# ---------------------------------------------------------------------------
# Staging module build/example-input functions
# ---------------------------------------------------------------------------
class ACTORTraceWrapper(nn.Module):
    """Wraps the real CVAE (transformer encoder+decoder) so it can be traced from plain
    positional tensor args instead of the dict-batch calling convention."""

    def __init__(self, cvae: CVAE):
        super().__init__()
        self.cvae = cvae

    def forward(self, x, y, mask, lengths):
        batch = {"x": x, "y": y, "mask": mask, "lengths": lengths}
        out = self.cvae(batch)
        return out["output"]


def build_actor():
    njoints, nfeats = 6, 6  # small "rot6d"-style joint feature layout
    num_classes = 4
    latent_dim = 32
    encoder = Encoder_TRANSFORMER(
        modeltype="cvae",
        njoints=njoints,
        nfeats=nfeats,
        num_frames=8,
        num_classes=num_classes,
        translation=True,
        pose_rep="rot6d",
        glob=True,
        glob_rot=True,
        latent_dim=latent_dim,
        ff_size=64,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
    )
    decoder = Decoder_TRANSFORMER(
        modeltype="cvae",
        njoints=njoints,
        nfeats=nfeats,
        num_frames=8,
        num_classes=num_classes,
        translation=True,
        pose_rep="rot6d",
        glob=True,
        glob_rot=True,
        latent_dim=latent_dim,
        ff_size=64,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
    )
    cvae = CVAE(encoder, decoder, latent_dim=latent_dim, pose_rep="rot6d")
    return ACTORTraceWrapper(cvae)


def example_input_actor():
    bs, njoints, nfeats, nframes = 2, 6, 6, 8
    x = torch.randn(bs, njoints, nfeats, nframes)
    y = torch.zeros(bs, dtype=torch.long)
    mask = torch.ones(bs, nframes, dtype=torch.bool)
    lengths = torch.full((bs,), nframes, dtype=torch.long)
    return (x, y, mask, lengths)


MENAGERIE_ENTRIES = [
    (
        "ACTOR (Transformer VAE for 3D Human Motion)",
        build_actor,
        example_input_actor,
        2021,
        MENAGERIE_ZOO,
    ),
]
