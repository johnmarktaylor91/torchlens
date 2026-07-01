# FAITHFUL PORT of mrojascarulla/GeNet @ master (original framework: TensorFlow 1.x)
#
# GeNet (Rojas-Carulla, Tolstikhin, Luque, Youngblut, Ley & Scholkopf 2019) --
# hierarchical residual 1D-convolutional network for metagenomic taxonomic
# classification of raw DNA reads. The real repo (code/network.py, `GENEt` class)
# is TF1.x session/graph-mode code: `tf.get_variable`, `tf.layers.*`,
# `tf.variable_scope`, a `tf.data` iterator-fed input pipeline, and
# `tf.contrib.lookup` hash tables in its sibling util.py -- none of which run in
# a modern base-torch env (TF1.x graph mode, not installed, and incompatible with
# any installed TF2/Keras3 here). This file transcribes the REAL architecture
# from code/network.py + code/util.py's `get_conv_layer` faithfully into
# self-contained torch:
#   - token embedding + learned positional embedding + one-hot(vocab), summed
#     (exactly `self.input = one_hot(x) + x_embed + pos_embed`, then
#     unsqueezed to [batch, seq_len, vocab_size, 1] for the "image-shaped"
#     conv2d encoder)
#   - `build_conv_encoder`: an initial Conv2d over [region_size, vocab_size]
#     collapsing the vocab axis, followed by 2 fixed resnet blocks + `num_resnet_blocks`
#     more (each pair doubling channel width), each block matching
#     `resnet_block`'s exact structure: pre-activation BN+ReLU, downsample-by-2
#     average-pool when channel width changes, two conv branches, and a
#     1x1-projected skip connection when the channel width changes
#   - global average pool over the (seq, dummy-width) dims + BN + dense ->
#     `encoder_state`
#   - "sg" mode's hierarchical taxonomy head: one dense(+ReLU) per taxonomic
#     level, with `connect_softmax` cascading each level's logits into an
#     additive correction for the next level (`logits[i] = orig[i] + new[i]`,
#     `new[i] = dense(logits[i-1])`), exactly as in the original `GENEt.__init__`.
# `cnn` mode (single flat classification head, no taxonomy) is also faithfully
# ported and exposed as a second MENAGERIE entry since it is the model's other
# real, documented `mode`.
#
# Not ported: the TF-Dataset iterator front-end, hash-table label lookup, and
# training-loop code in util.py/genet_train.py -- those are I/O plumbing around
# the network, not part of the GENEt architecture itself.
#
# Upstream: no explicit license file in the repo; code used for faithful
# architecture reproduction per the paper (Rojas-Carulla et al. 2019,
# "GeNet: Deep Representations for Metagenomics").

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class ResnetBlock1D(nn.Module):
    """
    Faithful port of GENEt.resnet_block. Operates on tensors shaped
    [batch, seq_len, width, channels] (TF NHWC layout used by the original
    tf.layers.conv2d calls with kernel [region_size, 1]), kept as NHWC here
    (permuted internally to NCHW for torch's conv2d) to mirror the original
    downsample/skip-projection logic exactly.
    """

    def __init__(self, region_size: int, in_channels: int, out_channels: int):
        super().__init__()
        self.change_input_dim = out_channels != in_channels
        self.bn_in = nn.BatchNorm2d(in_channels)
        pad = region_size // 2
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(region_size, 1), stride=1, padding=(pad, 0)
        )
        self.bn_mid = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(region_size, 1), stride=1, padding=(pad, 0)
        )
        if self.change_input_dim:
            self.conv_project = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, seq_len, width]  (NCHW)
        if self.change_input_dim:
            # Downsample seq_len by 2, average pooling, matching
            # tf.nn.pool(input, [2, 1], strides=[2, 1], pooling_type='AVG')
            x = F.avg_pool2d(x, kernel_size=(2, 1), stride=(2, 1))

        input_tr = F.relu(self.bn_in(x))
        conv1 = self.conv1(input_tr)
        conv1 = F.relu(self.bn_mid(conv1))
        conv2 = self.conv2(conv1)

        if self.change_input_dim:
            conv_project = self.conv_project(x)
        else:
            conv_project = x

        return conv_project + conv2


class GeNetConvEncoder(nn.Module):
    """
    Faithful port of GENEt.build_conv_encoder.
    """

    def __init__(
        self,
        region_size: int,
        vocab_size: int,
        num_filters: int,
        num_resnet_blocks: int,
        fully_connected: int,
    ):
        super().__init__()
        self.region_size = region_size
        pad = region_size // 2
        # conv_project: kernel [region_size, vocab_size], collapses the vocab axis
        self.conv_project = nn.Conv2d(
            1,
            num_filters,
            kernel_size=(region_size, vocab_size),
            stride=(region_size, 1),
            padding=(pad, 0),
        )

        init_filters = num_filters
        self.block1 = ResnetBlock1D(region_size, init_filters, init_filters)
        self.block2 = ResnetBlock1D(region_size, init_filters, init_filters)

        extra_blocks = []
        for _l in range(num_resnet_blocks):
            extra_blocks.append(ResnetBlock1D(region_size, init_filters, 2 * init_filters))
            extra_blocks.append(ResnetBlock1D(region_size, 2 * init_filters, 2 * init_filters))
            init_filters = 2 * init_filters
        self.extra_blocks = nn.ModuleList(extra_blocks)

        self.bn_out = nn.BatchNorm2d(init_filters)
        self.bn_encoder = nn.BatchNorm1d(init_filters)
        self.dense_encoder = nn.Linear(init_filters, fully_connected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, vocab_size, 1] (NHWC) -> NCHW [batch, 1, seq_len, vocab_size]
        x = x.permute(0, 3, 1, 2).contiguous()
        conv = self.conv_project(x)
        conv = self.block1(conv)
        conv = self.block2(conv)
        for block in self.extra_blocks:
            conv = block(conv)

        conv = F.relu(self.bn_out(conv))
        # Average pool over spatial dims (seq_len, width), matching
        # tf.reduce_mean(conv, [1, 2]) on the NHWC tensor.
        pooled = conv.mean(dim=(2, 3))
        encoder_state = self.bn_encoder(pooled)
        encoder_state = F.relu(self.dense_encoder(encoder_state))
        return encoder_state


class GeNet(nn.Module):
    """
    Faithful port of GENEt (both `cnn` and `sg` modes).
    """

    def __init__(
        self,
        seq_length: int,
        vocab_size: int,
        num_labels: int,
        region_size: int = 3,
        num_filters: int = 16,
        fully_connected: int = 32,
        num_resnet_blocks: int = 1,
        mode: str = "sg",
        num_groups=None,
        connect_softmax: bool = True,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.mode = mode
        self.connect_softmax = connect_softmax

        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.pos_embedding = nn.Embedding(seq_length, vocab_size)

        self.encoder = GeNetConvEncoder(
            region_size=region_size,
            vocab_size=vocab_size,
            num_filters=num_filters,
            num_resnet_blocks=num_resnet_blocks,
            fully_connected=fully_connected,
        )

        if mode == "cnn":
            self.logits_head = nn.Linear(fully_connected, num_labels)
        elif mode == "sg":
            assert num_groups is not None and len(num_groups) > 0
            self.num_groups = list(num_groups)
            self.group_heads = nn.ModuleList(
                [nn.Linear(fully_connected, n) for n in self.num_groups]
            )
            if connect_softmax:
                self.cascade_heads = nn.ModuleList(
                    [
                        nn.Linear(self.num_groups[i - 1], self.num_groups[i])
                        for i in range(1, len(self.num_groups))
                    ]
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor):
        # x: [batch, seq_length] integer token ids in [0, vocab_size)
        batch_size = x.shape[0]
        x = x.long()

        x_embed = self.embedding(x)
        positions = (
            torch.arange(self.seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        pos_embed = self.pos_embedding(positions)

        one_hot = F.one_hot(x, num_classes=self.vocab_size).to(x_embed.dtype)
        inp = one_hot + x_embed + pos_embed
        inp = inp.unsqueeze(-1)  # [batch, seq_length, vocab_size, 1] (NHWC)

        encoder_state = self.encoder(inp)

        if self.mode == "cnn":
            return self.logits_head(encoder_state)

        # sg mode: hierarchical taxonomy heads with optional cascade
        logits = [head(encoder_state) for head in self.group_heads]
        logits = [F.relu(logit) for logit in logits]
        if self.connect_softmax:
            logits_add = [None] + [
                F.relu(self.cascade_heads[i - 1](logits[i - 1]))
                for i in range(1, len(self.num_groups))
            ]
            logits = [orig if new is None else orig + new for orig, new in zip(logits, logits_add)]
        return logits


def build_genet():
    model = GeNet(
        seq_length=12,
        vocab_size=6,
        num_labels=10,
        region_size=3,
        num_filters=8,
        fully_connected=16,
        num_resnet_blocks=1,
        mode="sg",
        num_groups=[3, 4, 5],
        connect_softmax=True,
    )
    model.eval()
    return model


def example_input_genet():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 12))
    return (x,)


def build_genet_cnn():
    model = GeNet(
        seq_length=12,
        vocab_size=6,
        num_labels=10,
        region_size=3,
        num_filters=8,
        fully_connected=16,
        num_resnet_blocks=1,
        mode="cnn",
    )
    model.eval()
    return model


def example_input_genet_cnn():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 12))
    return (x,)


MENAGERIE_ENTRIES = [
    ("GeNet", "build_genet", "example_input_genet", 2019, "ported-pytorch"),
    ("GeNet-CNN", "build_genet_cnn", "example_input_genet_cnn", 2019, "ported-pytorch"),
]
