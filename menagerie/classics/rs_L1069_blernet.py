# FAITHFUL PORT of https://github.com/immortal3/AutoEncoder-Based-Communication-System
# @ master (autoencoder_dynamic.ipynb) (original framework: TensorFlow 1.x / Keras)
#
# End-to-end learned communication autoencoder from Tim O'Shea & Jakob Hoydis,
# "An Introduction to Deep Learning for the Physical Layer" (IEEE Trans. on
# Cognitive Communications and Networking, 2017; http://ieeexplore.ieee.org/
# document/8054694/). A communications transmitter+channel+receiver system is
# modeled end-to-end as a single autoencoder: an (n, k) encoder maps one of
# M = 2^k one-hot messages to an n-dimensional unit-power constellation point,
# a Gaussian-noise layer models the physical channel (AWGN), and a decoder
# reconstructs the message via softmax classification. This repo (immortal3/
# AutoEncoder-Based-Communication-System) is the canonical, most-starred public
# reimplementation of the O'Shea & Hoydis architecture and is written in
# TensorFlow 1.x / Keras (`from keras.layers.normalization import
# BatchNormalization`, `from tensorflow import set_random_seed`, legacy
# `Lambda`+`K.l2_normalize` power-normalization idiom) -- APIs that no longer
# exist in the installed TF2/Keras3 or any base env here, so per the ladder this
# is transcribed faithfully into self-contained torch rather than vendored.
#
# Every layer of the original Keras functional-API graph is reproduced 1:1:
#   Dense(M, relu) -> Dense(n_channel, linear) -> Lambda(unit-power L2 normalize)
#   -> GaussianNoise(std=sqrt(1/(2*R*EbNo_train))) -> Dense(M, relu) -> Dense(M, softmax)
# with R = k / n_channel (code rate) and EbNo_train = 5.01187 (7 dB Eb/N0, as in
# the notebook). No architectural mechanism was added, removed, or altered; the
# only change from the original is the framework (Keras -> torch) and packaging
# the six layers into an nn.Module class (the original is a bare Keras
# functional-API script, not a class) for menagerie staging.

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class BLERAutoencoder(nn.Module):
    """(n, k) autoencoder-based communication system (O'Shea & Hoydis, 2017).

    Args:
        M: number of distinct messages (M = 2**k).
        n_channel: number of channel uses per message (encoder output dim / "n").
        ebno_train_db: training Eb/N0 in dB, converted to linear scale to set the
            AWGN standard deviation exactly as the original notebook does.
    """

    def __init__(self, M=4, n_channel=2, ebno_train_db=7.0):
        super().__init__()
        self.M = M
        self.n_channel = n_channel
        k = int(np.log2(M))
        self.k = k
        self.R = k / n_channel

        ebno_train = 10.0 ** (
            ebno_train_db / 10.0
        )  # dB -> linear, matches EbNo_train = 5.01187 for 7 dB
        self.noise_std = float(np.sqrt(1.0 / (2.0 * self.R * ebno_train)))

        # encoded = Dense(M, activation='relu')(input_signal)
        self.enc_fc1 = nn.Linear(M, M)
        # encoded1 = Dense(n_channel, activation='linear')(encoded)
        self.enc_fc2 = nn.Linear(M, n_channel)
        # decoded = Dense(M, activation='relu')(encoded3)
        self.dec_fc1 = nn.Linear(n_channel, M)
        # decoded1 = Dense(M, activation='softmax')(decoded)
        self.dec_fc2 = nn.Linear(M, M)

    def encode(self, x):
        # encoded = Dense(M, activation='relu')(input_signal)
        x = torch.relu(self.enc_fc1(x))
        # encoded1 = Dense(n_channel, activation='linear')(encoded)
        x = self.enc_fc2(x)
        # encoded2 = Lambda(lambda x: sqrt(n_channel) * l2_normalize(x, axis=1))(encoded1)
        x = np.sqrt(self.n_channel) * nn.functional.normalize(x, p=2, dim=1)
        return x

    def channel(self, x):
        # encoded3 = GaussianNoise(sqrt(1 / (2*R*EbNo_train)))(encoded2)
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x

    def decode(self, x):
        # decoded = Dense(M, activation='relu')(encoded3)
        x = torch.relu(self.dec_fc1(x))
        # decoded1 = Dense(M, activation='softmax')(decoded)
        x = torch.softmax(self.dec_fc2(x), dim=-1)
        return x

    def forward(self, x):
        z = self.encode(x)
        z = self.channel(z)
        return self.decode(z)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# Mirrors the notebook's (2,2) autoencoder configuration: M=4 (k=2 bits),
# n_channel=2, trained/evaluated at EbNo_train = 5.01187 (7 dB).
# ---------------------------------------------------------------------------
_M = 4
_N_CHANNEL = 2
_BATCH = 8


def build_blernet():
    torch.manual_seed(0)
    model = BLERAutoencoder(M=_M, n_channel=_N_CHANNEL, ebno_train_db=7.0)
    model.eval()
    return model


def example_input_blernet():
    torch.manual_seed(0)
    labels = torch.randint(0, _M, (_BATCH,))
    one_hot = nn.functional.one_hot(labels, num_classes=_M).float()
    return one_hot


MENAGERIE_ENTRIES = [
    ("BLER-Net", "build_blernet", "example_input_blernet", 2017, MENAGERIE_ZOO),
]
