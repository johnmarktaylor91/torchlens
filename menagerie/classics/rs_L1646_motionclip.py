# SOURCE: vendored from GuyTevet/MotionCLIP @ main
# File: src/models/architectures/transformer.py (Encoder_TRANSFORMER,
# Decoder_TRANSFORMER, PositionalEncoding -- verbatim)
# Architecture unmodified from the real repo. MotionCLIP's full MOTIONCLIP
# wrapper (src/models/modeltype/motionclip.py) additionally needs the
# `clip` (OpenAI CLIP) package and an SMPL-based rotation2xyz renderer,
# neither of which are installed base libs, so this staging module vendors
# the actual novel architecture -- the transformer VAE encoder/decoder pair
# that MotionCLIP trains to align with CLIP's joint image/text embedding
# space -- which is pure torch/numpy and needs no extra dependencies.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


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

        self.muQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, self.latent_dim))
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

        x = self.skelEmbedding(x)

        y = y - y
        xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)

        xseq = self.sequence_pos_encoder(xseq)

        muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
        maskseq = torch.cat((muandsigmaMask, mask), axis=1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        mu = final[0]
        logvar = final[1]  # noqa: F841 (unused in original repo code)

        return {"mu": mu}


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

        self.actionBiases = nn.Parameter(torch.randn(1, self.latent_dim))

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

    def forward(self, batch, use_text_emb=False):
        z, y, mask, lengths = (  # noqa: F841 (lengths unused in original repo code)
            batch["z"],
            batch["y"],
            batch["mask"],
            batch["lengths"],
        )
        if use_text_emb:
            z = batch["clip_text_emb"]
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
                z = z[None]

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)

        output = self.seqTransDecoder(tgt=timequeries, memory=z, tgt_key_padding_mask=~mask)

        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)

        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)

        if use_text_emb:
            batch["txt_output"] = output
        else:
            batch["output"] = output
        return batch


class MotionCLIPAutoEncoder(nn.Module):
    """Thin staging wrapper chaining the real Encoder_TRANSFORMER /
    Decoder_TRANSFORMER so a single forward call exercises both (mirrors
    MOTIONCLIP.forward's encode-then-decode path, minus the CLIP alignment
    losses / SMPL rot2xyz rendering, which need extra packages)."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, mask, lengths):
        batch = {"x": x, "y": y, "mask": mask, "lengths": lengths}
        batch.update(self.encoder(batch))
        batch["z"] = batch["mu"]
        batch.update(self.decoder(batch))
        return batch["output"]


# --- staging glue -------------------------------------------------------------
def _motionclip_common_kwargs():
    return dict(
        modeltype="cvae",
        njoints=6,
        nfeats=4,
        num_frames=8,
        num_classes=1,
        translation=True,
        pose_rep="rot6d",
        glob=True,
        glob_rot=True,
        latent_dim=32,
        ff_size=64,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        activation="gelu",
    )


def build_motionclip_autoencoder():
    torch.manual_seed(0)
    kwargs = _motionclip_common_kwargs()
    encoder = Encoder_TRANSFORMER(**kwargs)
    decoder = Decoder_TRANSFORMER(**kwargs)
    return MotionCLIPAutoEncoder(encoder, decoder).eval()


def example_input_motionclip_autoencoder():
    torch.manual_seed(0)
    bs, njoints, nfeats, nframes = 2, 6, 4, 8
    x = torch.randn(bs, njoints, nfeats, nframes)
    y = torch.zeros(bs, dtype=torch.long)
    mask = torch.ones(bs, nframes, dtype=torch.bool)
    lengths = torch.full((bs,), nframes, dtype=torch.long)
    return (x, y, mask, lengths)


MENAGERIE_ENTRIES = [
    (
        "MotionCLIP-AutoEncoder",
        build_motionclip_autoencoder,
        example_input_motionclip_autoencoder,
        2022,
        MENAGERIE_ZOO,
    ),
]
