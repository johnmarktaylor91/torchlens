# FAITHFUL PORT of bcbl-kaist/DeepLUCIA @ master (deeplucia_toolkit/make_model.py
# `seq_epi_20210105_v1_model`)
# (original framework: tensorflow.keras)
#
# DeepLUCIA (bcbl-kaist, Bioinformatics 2022, tissue-specific enhancer-promoter
# loop / chromatin-loop predictor from paired-anchor sequence + epigenomic
# tracks) is a two-modality dual-branch 1D-CNN: two 5000bp one-hot DNA-sequence
# anchors (seq_feature_one/two) are concatenated along the sequence axis and
# run through a Conv1D(128, k=40, relu, same) + MaxPool1D(50) branch; two
# 200-bin x 12-mark epigenomic-track anchors (epi_feature_one/two) are
# similarly concatenated and run through a parallel Conv1D(128, k=10, relu,
# same) + MaxPool1D(2) branch. The two branches are fused by concatenation
# along the FEATURE (channel) axis, then Dropout(0.5), a Conv1D(32, k=1,
# 'bottleneck') + Conv1D(512, k=10) refinement, MaxPool1D(100, 'same'),
# Flatten, Dropout(0.5), Dense(512, relu), Dense(1, sigmoid). Every
# layer/kernel-size/filter-count/pool-size below matches the real Keras
# functional graph; only the Keras -> torch layer translation (channels-last
# NLC -> torch's channels-first NCL Conv1d convention) and the CLI/training
# glue are new.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"

SEQ_LEN = 5000  # real: per-anchor one-hot DNA window length
SEQ_CH = 4  # real: one-hot ACGT channels
EPI_LEN = 200  # real: per-anchor epigenomic bin count
EPI_CH = 12  # real: number of epigenomic marks


class DeepLUCIASeqEpi(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, seq_ch=SEQ_CH, epi_len=EPI_LEN, epi_ch=EPI_CH):
        super().__init__()
        self.seq_len = seq_len
        self.epi_len = epi_len

        # keras: Conv1D(filters=128, kernel_size=40, padding='same', activation='relu') over
        # the seq-axis-concatenated pair (length 2*seq_len, channels=seq_ch).
        self.seq_conv = nn.Conv1d(seq_ch, 128, kernel_size=40, padding="same")
        self.seq_act = nn.ReLU()
        self.seq_pool = nn.MaxPool1d(kernel_size=50)

        # keras: Conv1D(filters=128, kernel_size=10, padding='same', activation='relu') over
        # the seq-axis-concatenated pair (length 2*epi_len, channels=epi_ch).
        self.epi_conv = nn.Conv1d(epi_ch, 128, kernel_size=10, padding="same")
        self.epi_act = nn.ReLU()
        self.epi_pool = nn.MaxPool1d(kernel_size=2)

        self.fuse_drop = nn.Dropout(0.5)

        # keras: Conv1D(filters=32, kernel_size=1, padding='same', activation='relu', name='bottleneck')
        # over the feature-axis (channel) concatenation of the two branches -> in_channels = 128+128.
        self.bottleneck = nn.Conv1d(128 + 128, 32, kernel_size=1, padding="same")
        self.bottleneck_act = nn.ReLU()

        # keras: Conv1D(filters=512, kernel_size=10, padding='same', activation='relu')
        self.refine_conv = nn.Conv1d(32, 512, kernel_size=10, padding="same")
        self.refine_act = nn.ReLU()

        # keras: MaxPooling1D(pool_size=100, padding='same')
        self.refine_pool = nn.MaxPool1d(kernel_size=100, padding=0, ceil_mode=True)

        self.head_drop = nn.Dropout(0.5)

        fused_len_seq = (2 * seq_len) // 50
        fused_len_epi = (2 * epi_len) // 2
        # keras `concatenate([seq_path, epi_path], axis=2)` fuses along the FEATURE
        # axis in channels-last (NLC) layout -- i.e. both branches must share the
        # same sequence length. The real defaults satisfy this: 2*5000/50 = 200 and
        # 2*200/2 = 200. Assert it holds for whatever seq_len/epi_len are passed in
        # so the port stays faithful (rather than silently mismatching lengths).
        assert fused_len_seq == fused_len_epi, (
            f"DeepLUCIA fusion requires matching branch lengths: seq branch -> {fused_len_seq}, "
            f"epi branch -> {fused_len_epi} (real defaults: seq_len=5000, epi_len=200)"
        )
        fused_len = fused_len_seq
        pooled_len = -(-fused_len // 100)  # ceil division, 'same'-style pooling
        self.flat_dim = 512 * pooled_len

        self.fc1 = nn.Linear(self.flat_dim, 512)
        self.fc1_act = nn.ReLU()
        self.fc_out = nn.Linear(512, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, seq_one, seq_two, epi_one, epi_two):
        # Inputs arrive channels-first (N, C, L) per torch Conv1d convention
        # (real Keras inputs are channels-last (N, L, C); the concat axis is
        # translated from axis=1 (length) in NLC to dim=-1 (length) in NCL).
        seq_path = torch.cat([seq_one, seq_two], dim=-1)
        seq_path = self.seq_act(self.seq_conv(seq_path))
        seq_path = self.seq_pool(seq_path)

        epi_path = torch.cat([epi_one, epi_two], dim=-1)
        epi_path = self.epi_act(self.epi_conv(epi_path))
        epi_path = self.epi_pool(epi_path)

        # keras `concatenate([seq_path, epi_path], axis=2)` fuses along the
        # feature (channel) axis in NLC layout -> channel-dim concat in NCL.
        combined = torch.cat([seq_path, epi_path], dim=1)
        combined = self.fuse_drop(combined)

        combined = self.bottleneck_act(self.bottleneck(combined))
        combined = self.refine_act(self.refine_conv(combined))
        combined = self.refine_pool(combined)

        combined = torch.flatten(combined, start_dim=1)
        combined = self.head_drop(combined)
        combined = self.fc1_act(self.fc1(combined))
        combined = self.fc_out(combined)
        return self.out_act(combined)


# ---------------------------------------------------------------------------
# menagerie staging entry point
# ---------------------------------------------------------------------------
# Use small synthetic seq_len/epi_len for trace speed (real model uses
# seq_len=5000, epi_len=200); the dual-branch conv+pool+fuse+bottleneck+dense
# architecture is identical. Scaled lengths preserve the real ratio
# (2*seq_len/50 == 2*epi_len/2) so the branch-fusion shape assertion holds.
_TINY_SEQ_LEN = 100  # -> seq branch len after pool: 2*100/50 = 4
_TINY_EPI_LEN = 4  # -> epi branch len after pool: 2*4/2 = 4


def build_deeplucia_seq_epi():
    return DeepLUCIASeqEpi(
        seq_len=_TINY_SEQ_LEN, seq_ch=SEQ_CH, epi_len=_TINY_EPI_LEN, epi_ch=EPI_CH
    )


def example_input_deeplucia_seq_epi():
    seq_one = torch.randn(2, SEQ_CH, _TINY_SEQ_LEN)
    seq_two = torch.randn(2, SEQ_CH, _TINY_SEQ_LEN)
    epi_one = torch.randn(2, EPI_CH, _TINY_EPI_LEN)
    epi_two = torch.randn(2, EPI_CH, _TINY_EPI_LEN)
    return (seq_one, seq_two, epi_one, epi_two)


MENAGERIE_ENTRIES = [
    (
        "DeepLUCIA-SeqEpi",
        build_deeplucia_seq_epi,
        example_input_deeplucia_seq_epi,
        2022,
        "SOURCE_AVAILABLE",
    ),
]
