# SOURCE: vendored from nachiket92/conv-social-pooling @ master
# Files combined:
#   model.py (highwayNet)
#   utils.py (outputActivation -- the custom Graves-2015 output activation used at the end
#             of highwayNet.decode)
#
# This single repo/model.py file implements BOTH queue candidates:
#   - "Convolutional Social Pooling" (Deo & Trivedi, CVPRW 2018) -- highwayNet with
#     use_maneuvers=False: encoder LSTM + convolutional social-pooling grid + decoder LSTM,
#     uni-modal trajectory output.
#   - "CS-LSTM" (same paper, same repo) -- highwayNet with use_maneuvers=True: adds the
#     lateral/longitudinal maneuver classification heads (op_lat/op_lon) and concatenates the
#     maneuver one-hot encodings into the decoder input, producing a maneuver-conditioned
#     multi-modal trajectory distribution. This is a genuine architectural difference (extra
#     linear heads + wider decoder LSTM input), not a cosmetic config toggle, hence both are
#     staged as separate MENAGERIE_ENTRIES from the one vendored class.
#
# Import-only fix applied (no architectural change): `from utils import outputActivation`
# resolved by inlining the function body from the same repo's utils.py (the rest of
# utils.py is NGSIM dataset/loss-function plumbing, unrelated to the network architecture).
# forward()/decode() bodies are otherwise byte-for-byte the original.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# utils.py: outputActivation (Graves 2015 bivariate-Gaussian output activation)
# ---------------------------------------------------------------------------
def outputActivation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out


# ---------------------------------------------------------------------------
# model.py: highwayNet
# ---------------------------------------------------------------------------
class highwayNet(nn.Module):
    ## Initialization
    def __init__(self, args):
        super(highwayNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args["use_cuda"]

        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args["use_maneuvers"]

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args["train_flag"]

        ## Sizes of network layers
        self.encoder_size = args["encoder_size"]
        self.decoder_size = args["decoder_size"]
        self.in_length = args["in_length"]
        self.out_length = args["out_length"]
        self.grid_size = args["grid_size"]
        self.soc_conv_depth = args["soc_conv_depth"]
        self.conv_3x1_depth = args["conv_3x1_depth"]
        self.dyn_embedding_size = args["dyn_embedding_size"]
        self.input_embedding_size = args["input_embedding_size"]
        self.num_lat_classes = args["num_lat_classes"]
        self.num_lon_classes = args["num_lon_classes"]
        self.soc_embedding_size = (((args["grid_size"][0] - 4) + 1) // 2) * self.conv_3x1_depth

        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)

        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3, 1))
        self.soc_maxpool = torch.nn.MaxPool2d((2, 1), padding=(1, 0))

        # Decoder LSTM
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(
                self.soc_embedding_size
                + self.dyn_embedding_size
                + self.num_lat_classes
                + self.num_lon_classes,
                self.decoder_size,
            )
        else:
            self.dec_lstm = torch.nn.LSTM(
                self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size
            )

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size, 5)
        self.op_lat = torch.nn.Linear(
            self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes
        )
        self.op_lon = torch.nn.Linear(
            self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes
        )

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    ## Forward Pass
    def forward(self, hist, nbrs, masks, lat_enc, lon_enc):
        ## Forward pass hist:
        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = self.leaky_relu(
            self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2]))
        )

        ## Forward pass nbrs
        _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0, 3, 2, 1)

        ## Apply convolutional social pooling:
        soc_enc = self.soc_maxpool(
            self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc))))
        )
        soc_enc = soc_enc.view(-1, self.soc_embedding_size)

        ## Concatenate encodings:
        enc = torch.cat((soc_enc, hist_enc), 1)

        if self.use_maneuvers:
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):  # noqa: E741 (matches original repo variable name)
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc)
            return fut_pred

    def decode(self, enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "vendored-pytorch"

_GRID_SIZE = (13, 3)  # repo default grid_size (NGSIM social-context grid, 13 cols x 3 lanes)


def _base_args(use_maneuvers):
    # Mirrors the repo's train.py default `args` dict, with encoder/decoder widths shrunk
    # for fast tracing; grid_size and conv/pool topology are the repo defaults untouched.
    return {
        "use_cuda": False,
        "encoder_size": 16,
        "decoder_size": 16,
        "in_length": 16,
        "out_length": 25,
        "grid_size": _GRID_SIZE,
        "soc_conv_depth": 8,
        "conv_3x1_depth": 8,
        "dyn_embedding_size": 16,
        "input_embedding_size": 8,
        "num_lat_classes": 3,
        "num_lon_classes": 2,
        "use_maneuvers": use_maneuvers,
        "train_flag": True,
    }


def build_conv_social_pooling():
    """Convolutional Social Pooling (Deo & Trivedi, CVPRW 2018): highwayNet, use_maneuvers=False."""
    return highwayNet(_base_args(use_maneuvers=False))


def build_cs_lstm():
    """CS-LSTM (Deo & Trivedi, CVPRW 2018): highwayNet, use_maneuvers=True, train-mode decode."""
    return highwayNet(_base_args(use_maneuvers=True))


def _example_inputs(grid_size=_GRID_SIZE, in_length=16, batch=2):
    hist = torch.randn(in_length, batch, 2)
    # One neighbor slot per grid cell per batch element (repo's collate_fn packs nbrs
    # as [maxlen, total_active_neighbor_count, 2]; using grid_size[0]*grid_size[1]*batch
    # active neighbors keeps every social-pooling grid cell populated).
    n_nbrs = grid_size[0] * grid_size[1] * batch
    nbrs = torch.randn(in_length, n_nbrs, 2)
    # mask last dim must equal encoder_size (holds nbrs_enc, whose width is encoder_size,
    # per model.py forward(): soc_enc = zeros_like(masks); masked_scatter_(masks, nbrs_enc))
    masks = torch.ones(batch, grid_size[1], grid_size[0], 16, dtype=torch.bool)
    lat_enc = torch.zeros(batch, 3)
    lat_enc[:, 0] = 1
    lon_enc = torch.zeros(batch, 2)
    lon_enc[:, 0] = 1
    return hist, nbrs, masks, lat_enc, lon_enc


def example_input_conv_social_pooling():
    return _example_inputs()


def example_input_cs_lstm():
    return _example_inputs()


MENAGERIE_ENTRIES = [
    (
        "Convolutional Social Pooling",
        "build_conv_social_pooling",
        "example_input_conv_social_pooling",
        2018,
        MENAGERIE_ZOO,
    ),
    ("CS-LSTM", "build_cs_lstm", "example_input_cs_lstm", 2018, MENAGERIE_ZOO),
]
