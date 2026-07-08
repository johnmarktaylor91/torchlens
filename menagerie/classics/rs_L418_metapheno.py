# FAITHFUL PORT of nlapier2/metapheno @ master (original framework: Keras/TensorFlow)
# https://raw.githubusercontent.com/nlapier2/metapheno/master/classify.py
#
# LaPierre, Egan, Wang, Wang, Zhu, Zeng, Anderson, Eisen, Sun, Koslicki 2019
# (Frontiers in Microbiology) "MetaPheno: A Critical Evaluation of Deep Learning and
# Machine Learning in Metagenome-Based Disease Prediction" -- the paper's "autonn"
# pipeline (`run_autonn` in the real `classify.py`) pretrains a symmetric
# feedforward autoencoder on the metagenomic feature vector (MetaPhlAn taxon
# relative abundances or k-mer counts), keeps only the trained ENCODER half as a
# fixed nonlinear feature transform, then trains a separate deep MLP classifier
# (relu input layer -> `numlayers` tanh+dropout blocks with uniformly shrinking
# width -> sigmoid output) on the encoded features to predict a binary disease
# phenotype label. The real code is Keras (`keras.layers.Dense/Dropout`,
# `keras.models.Model/Sequential`) built and TRAINED (`.fit(...)`) inline inside
# `build_and_fit_autoencoder` / `build_and_fit_model`; Keras/TensorFlow are not
# available in this environment, so the two networks are faithfully transcribed
# here as `nn.Module`s in base torch, preserving every architectural choice from
# the real functions:
#   - `build_and_fit_autoencoder(x_train, layers, opt, learn_rate)`: encoder sizes
#     `[input_dim // 2**i for i in range(0, layers+1)]`, each encoder/decoder Dense
#     layer using relu activation (`kernel_initializer='random_normal'` in the
#     original -- weight-init choice, not an architectural change); this port
#     exposes both the full autoencoder (`Autoencoder`, matching `Model(inputs=
#     input_data, outputs=decoded)`) and the encoder-only forward path used by
#     `autoencoder_pretrain` to transform features for the downstream classifier.
#   - `build_and_fit_model(train_X, ..., numlayers=5, dropout=0.25)`: `Sequential`
#     of `Dense(layersize, relu)` input layer, then `numlayers` repeats of
#     `Dense(this_layersize, tanh)` + `Dropout(dropout)` with
#     `this_layersize = layersize - int(layersize * (layer_scale * (i+1)))` and
#     `layer_scale = 1.0 / (numlayers + 1)`, then a final `Dense(1, sigmoid)` head
#     (this port keeps the raw sigmoid output, matching `activation='sigmoid'`).
# Training/optimizer/data-loading code (keras `optimizers.adam/sgd/adagrad`,
# `.compile`/`.fit`/`.predict`, xgboost/gcforest/svm/randomforest baselines) is
# training plumbing, not part of either network's architecture, and is dropped.

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Faithful port of `build_and_fit_autoencoder`: a symmetric encoder/decoder
    stack of relu Dense layers, mirroring the real function's
    `encoded_layer_sizes = [int(input_dim / (2**i)) for i in range(0, layers+1)]`
    halving schedule for both the encoder and decoder halves."""

    def __init__(self, input_dim: int, layers: int = 2):
        super().__init__()
        sizes = [int(input_dim / (2**i)) for i in range(0, layers + 1)]

        encoder_layers = []
        prev = input_dim
        for i in range(1, layers + 1):
            encoder_layers.append(nn.Linear(prev, sizes[i]))
            encoder_layers.append(nn.ReLU())
            prev = sizes[i]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(1, layers):
            decoder_layers.append(nn.Linear(prev, sizes[layers - i]))
            decoder_layers.append(nn.ReLU())
            prev = sizes[layers - i]
        decoder_layers.append(nn.Linear(prev, input_dim))
        decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class MetaPhenoClassifier(nn.Module):
    """Faithful port of `build_and_fit_model`: relu input Dense layer, then
    `numlayers` (tanh Dense + Dropout) blocks with uniformly shrinking width
    (`layer_scale = 1.0 / (numlayers + 1)`), then a sigmoid Dense(1) output head."""

    def __init__(self, input_dim: int, numlayers: int = 5, dropout: float = 0.25):
        super().__init__()
        layersize = input_dim
        layer_scale = 1.0 / float(numlayers + 1)

        blocks = [nn.Linear(layersize, layersize), nn.ReLU()]
        prev = layersize
        for i in range(numlayers):
            this_layersize = layersize - int(layersize * (layer_scale * (i + 1)))
            blocks.append(nn.Linear(prev, this_layersize))
            blocks.append(nn.Tanh())
            blocks.append(nn.Dropout(dropout))
            prev = this_layersize
        blocks.append(nn.Linear(prev, 1))
        blocks.append(nn.Sigmoid())
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class MetaPhenoAutoNN(nn.Module):
    """The real `run_autonn` pipeline wired together as one traceable module:
    autoencoder-encode the input features, then classify the encoded
    representation, matching `run_autonn(train_X, test_X, ...)` ->
    `autoencoder_pretrain` (encoder-only feature transform) ->
    `build_and_fit_model` (MLP classifier on the transformed features)."""

    def __init__(
        self, input_dim: int = 128, auto_layers: int = 1, fc_layers: int = 5, dropout: float = 0.25
    ):
        super().__init__()
        self.autoencoder = Autoencoder(input_dim, layers=auto_layers)
        encoded_dim = int(input_dim / (2**auto_layers))
        self.classifier = MetaPhenoClassifier(encoded_dim, numlayers=fc_layers, dropout=dropout)

    def forward(self, x):
        encoded = self.autoencoder.encoder(x)
        return self.classifier(encoded)


def build_metapheno_autonn():
    # Real defaults from run_autonn(auto_layers=1, fc_layers=5, dropout=0.25);
    # input_dim kept small (128 MetaPhlAn-style taxon features) for a fast trace.
    return MetaPhenoAutoNN(input_dim=128, auto_layers=1, fc_layers=5, dropout=0.25)


def example_input_metapheno_autonn():
    return torch.randn(8, 128)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "MetaPheno-AutoNN",
        "build_metapheno_autonn",
        "example_input_metapheno_autonn",
        2019,
        "ported",
    ),
]
