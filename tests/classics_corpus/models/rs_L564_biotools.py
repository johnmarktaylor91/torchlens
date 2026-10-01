# SOURCE: vendored from https://github.com/YangLabHKUST/Portal @ 6bf61493
#
# Portal (Zhao, Cai, Zhang, Bao, Yang. 2022, "Adversarial domain translation networks
# for fast and accurate prediction of cellular responses to novel drugs", and the
# companion Portal single-cell atlas-integration paper). An adversarial domain
# translation autoencoder for atlas-level single-cell dataset integration: per-domain
# `encoder`/`generator` pairs (linear-relu-linear MLPs) plus a `discriminator` MLP that
# adversarially aligns the latent codes of two domains (batches A / B) so PCA-reduced
# expression profiles can be translated between domains. Vendored verbatim (byte-for-byte)
# from the repo's own network-definition file:
#   https://raw.githubusercontent.com/YangLabHKUST/Portal/main/portal/networks.py
#
# What is kept: the exact `encoder`/`generator`/`discriminator` nn.Module classes and
# their forward() computations, unmodified.
#
# What is dropped (data plumbing, not architecture): `portal/model.py` (training loop,
# scanpy/anndata preprocessing, PCA fitting, checkpointing) and `portal/utils.py`
# (rpy2/anndata2ri R-bridge evaluation metrics) are the surrounding pipeline, not part
# of the trainable network, and pull in scanpy/anndata/rpy2 deps we do not vendor here.
# The real constructor defaults (`Portal_pipeline.__init__`: `npcs=30`, `n_latent=20`)
# are preserved for the tiny build below (`portal/model.py:16-32`, instantiation sites
# `portal/model.py:261-266`).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Portal: vendored verbatim from portal/networks.py
# ---------------------------------------------------------------------------
class encoder(nn.Module):
    def __init__(self, n_input, n_latent):
        super(encoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(self.n_latent, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_latent).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        z = F.linear(h, self.W_2, self.b_2)
        return z


class generator(nn.Module):
    def __init__(self, n_input, n_latent):
        super(generator, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_latent).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(self.n_input, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_input).normal_(mean=0.0, std=0.1))

    def forward(self, z):
        h = F.relu(F.linear(z, self.W_1, self.b_1))
        x = F.linear(h, self.W_2, self.b_2)
        return x


class discriminator(nn.Module):
    def __init__(self, n_input):
        super(discriminator, self).__init__()
        self.n_input = n_input
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(n_hidden, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_3 = nn.Parameter(torch.Tensor(1, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_3 = nn.Parameter(torch.Tensor(1).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        h = F.relu(F.linear(h, self.W_2, self.b_2))
        score = F.linear(h, self.W_3, self.b_3)
        return torch.clamp(score, min=-50.0, max=50.0)


class PortalTranslation(nn.Module):
    """Staging wrapper composing the real Portal encoder/generator/discriminator
    modules into the single forward pass the pipeline performs when translating domain
    A profiles into domain B's latent/reconstruction space (encode with E_A, decode
    with G_B, then score with D_B) -- the actual cross-domain computation the adversarial
    training loop repeatedly runs (see portal/model.py train_integration loop)."""

    def __init__(self, npcs=30, n_latent=20):
        super().__init__()
        self.E_A = encoder(npcs, n_latent)
        self.G_B = generator(npcs, n_latent)
        self.D_B = discriminator(npcs)

    def forward(self, x_a):
        z_a = self.E_A(x_a)
        x_ab = self.G_B(z_a)
        score = self.D_B(x_ab)
        return x_ab, score


def build_portal():
    return PortalTranslation(npcs=30, n_latent=20)


def example_input_portal():
    # PCA-reduced expression profile, batch x npcs (real default npcs=30).
    return torch.randn(4, 30)


# ---------------------------------------------------------------------------
# SOURCE: vendored from https://github.com/kuixu/PrismNet @ afe40d47
#
# PrismNet (Sun, Xu, Zhang, Xu, Ouyang, Zhang, Zhang, Xie. 2021, "Predicting dynamic
# cellular protein-RNA interactions using deep learning and in vivo RNA structure").
# A squeeze-and-excitation-gated 2D/1D residual CNN over stacked RNA sequence + in-vivo
# icSHAPE structure features ("pu" mode: 5-channel one-hot sequence + structure track)
# that predicts protein-RNA binding. Vendored verbatim (byte-for-byte) from the repo's
# own model-definition files:
#   https://raw.githubusercontent.com/kuixu/PrismNet/master/prismnet/model/PrismNet.py
#   https://raw.githubusercontent.com/kuixu/PrismNet/master/prismnet/model/resnet.py
#   https://raw.githubusercontent.com/kuixu/PrismNet/master/prismnet/model/se.py
#
# What is kept: the exact `PrismNet` class (Conv2d stem -> SEBlock gate -> ResidualBlock2D
# -> avgpool over the feature axis -> ResidualBlock1D -> global pool -> Linear head) and
# its two residual-block / SE-block helper modules, unmodified.
#
# What is dropped (data plumbing, not architecture): `prismnet/loader.py` (h5-file
# dataset loading), `prismnet/engine/train_loop.py` (training loop), and
# `prismnet/model/smoothgrad.py` (post-hoc saliency-map tooling) are not part of the
# trainable network.
#
# MENAGERIE_ZOO = "vendored-pytorch"
# ---------------------------------------------------------------------------
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        relu=True,
        same_padding=False,
        bn=False,
    ):
        super(Conv2d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        padding = (p0, p1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class ResidualBlock1D(nn.Module):
    def __init__(self, planes, downsample=True):
        super(ResidualBlock1D, self).__init__()
        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm1d(planes)
        self.c2 = nn.Conv1d(planes, planes * 2, kernel_size=11, stride=1, padding=5, bias=False)
        self.b2 = nn.BatchNorm1d(planes * 2)
        self.c3 = nn.Conv1d(planes * 2, planes * 8, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm1d(planes * 8)
        self.downsample = nn.Sequential(
            nn.Conv1d(planes, planes * 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes * 8),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualBlock2D(nn.Module):
    def __init__(self, planes, kernel_size=(11, 5), padding=(5, 2), downsample=True):
        super(ResidualBlock2D, self).__init__()
        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(
            planes, planes * 2, kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )
        self.b2 = nn.BatchNorm2d(planes * 2)
        self.c3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm2d(planes * 4)
        self.downsample = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class PrismNet(nn.Module):
    def __init__(self, mode="pu"):
        super(PrismNet, self).__init__()
        self.mode = mode
        h_p, h_k = 2, 5
        if mode == "pu":
            self.n_features = 5
        elif mode == "seq":
            self.n_features = 4
            h_p, h_k = 1, 3
        elif mode == "str":
            self.n_features = 1
            h_p, h_k = 0, 1
        else:
            raise "mode error"

        base_channel = 8
        self.conv = Conv2d(1, base_channel, kernel_size=(11, h_k), bn=True, same_padding=True)
        self.se = SEBlock(base_channel)
        self.res2d = ResidualBlock2D(base_channel, kernel_size=(11, h_k), padding=(5, h_p))
        self.res1d = ResidualBlock1D(base_channel * 4)
        self.avgpool = nn.AvgPool2d((1, self.n_features))
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channel * 4 * 8, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        """[forward]

        Args:
            input ([tensor],N,C,W,H): input features
        """
        if self.mode == "seq":
            input = input[:, :, :, :4]
        elif self.mode == "str":
            input = input[:, :, :, 4:]
        x = self.conv(input)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(x * z)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x


def build_prismnet():
    return PrismNet(mode="pu")


def example_input_prismnet():
    # (N, C=1, W=seq_len, H=n_features); n_features=5 for "pu" mode (4-channel
    # one-hot sequence + 1 icSHAPE structure track), matching prismnet/loader.py.
    return torch.randn(2, 1, 101, 5)


# ---------------------------------------------------------------------------
# SOURCE: vendored from https://github.com/ZhangLab312/PROTRAIT @ 39313f69
#
# PROTRAIT (ZhangLab312, "PROTRAIT: an informer-based deep learning framework for
# predicting chromatin accessibility / regulatory element activity from DNA sequence
# across many cell types"). A convolutional-stem + ProbSparse-attention Informer-style
# encoder (6 encoder-layer/pooling-layer stages) that embeds a DNA sequence window,
# bottlenecks it to a 32-dim peak embedding, and predicts a per-cell-type sigmoid
# accessibility vector. Vendored verbatim (byte-for-byte) from the repo's own
# model-definition file:
#   https://raw.githubusercontent.com/ZhangLab312/PROTRAIT/main/public/model.py
#
# What is kept: the exact `StemLayer`/`StemEmbedding`/`BasePairEmbedding`/
# `PositionalEmbedding`/`SampleEmbedding`/`ProbAttention`/`SelfAttention`/`EncoderLayer`/
# `PoolingLayer`/`Encoder`/`BottleneckLayer`/`Prediction`/`Protrait` classes and their
# forward() computations, unmodified.
#
# What is dropped (data plumbing, not architecture): `train/dataset.py` (sequence/label
# loading), `public/train.py` / `public/earlyStopping.py` / `public/evaluation.py`
# (training loop, early-stopping, eval metrics), and `cluster/clustering.py`
# (post-hoc cell-embedding clustering) are not part of the trainable network. The real
# training entrypoint constructs `Protrait(seq_len=1344)` (public/train.py:35), which
# hits the class default `c_in=4, d_model=288, n_heads=8, n_cells=2034,
# seq_len_up_bound=1344`; the staging build below uses `seq_len=600` (one of the class's
# own supported `feature_map` keys, `public/model.py`'s `BottleneckLayer.feature_map`)
# with `seq_len_up_bound=1344` (so it takes the `BasePairEmbedding` branch rather than
# the deeper `StemEmbedding` cascade) and a tiny `n_cells=4` output head, for a fast
# trace -- every layer/mechanism is unchanged.
#
# MENAGERIE_ZOO = "vendored-pytorch"
# ---------------------------------------------------------------------------
import math  # noqa: E402


# 1. Embedding
class StemLayer(nn.Module):
    def __init__(self, number_input_features, number_output_features, kernel_size, pool_degree):
        super(StemLayer, self).__init__()
        self.stem_conv = nn.Conv1d(
            in_channels=number_input_features,
            out_channels=number_output_features,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="zeros",
        )
        self.stem_norm = nn.BatchNorm1d(num_features=number_output_features)
        self.stem_gelu = nn.GELU()
        self.stem_pool = nn.MaxPool1d(kernel_size=pool_degree)

    def forward(self, x):
        stem_conv = self.stem_conv(x)
        stem_norm = self.stem_norm(stem_conv)
        stem_gelu = self.stem_gelu(stem_norm)
        stem_pool = self.stem_pool(stem_gelu)

        return stem_pool


class StemEmbedding(nn.Module):
    def __init__(self, c_in=4, d_model=288):
        super(StemEmbedding, self).__init__()
        # 4 stem layer
        c_increase = (d_model - c_in) // 4

        self.stem_layer_1 = StemLayer(
            number_input_features=c_in,
            number_output_features=c_in + 1 * c_increase,
            kernel_size=5,
            pool_degree=2,
        )
        self.stem_layer_2 = StemLayer(
            number_input_features=c_in + 1 * c_increase,
            number_output_features=c_in + 2 * c_increase,
            kernel_size=5,
            pool_degree=4,
        )
        self.stem_layer_3 = StemLayer(
            number_input_features=c_in + 2 * c_increase,
            number_output_features=c_in + 3 * c_increase,
            kernel_size=5,
            pool_degree=4,
        )
        self.stem_layer_4 = StemLayer(
            number_input_features=c_in + 3 * c_increase,
            number_output_features=c_in + 4 * c_increase,
            kernel_size=5,
            pool_degree=4,
        )

    def forward(self, x):
        stem_layer_1 = self.stem_layer_1(x)
        stem_layer_2 = self.stem_layer_2(stem_layer_1)
        stem_layer_3 = self.stem_layer_3(stem_layer_2)
        stem_layer_4 = self.stem_layer_4(stem_layer_3)

        return stem_layer_4.permute(0, 2, 1)


class BasePairEmbedding(nn.Module):
    def __init__(self, c_in=4, d_model=288):
        super(BasePairEmbedding, self).__init__()
        self.base_pair_conv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.base_pair_gelu = nn.GELU()

    def forward(self, x):
        base_pair_conv = self.base_pair_conv(x)
        base_pair_gelu = self.base_pair_gelu(base_pair_conv)
        return base_pair_gelu.permute(0, 2, 1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = torch.zeros(
            size=(seq_len, d_model), requires_grad=False
        ).float()

        position = torch.arange(0, seq_len).unsqueeze(dim=1).float()
        div_value = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        self.positional_embedding[:, 0::2] = torch.sin(position * div_value)
        self.positional_embedding[:, 1::2] = torch.cos(position * div_value)

        self.positional_embedding = self.positional_embedding.unsqueeze(dim=0)

    def forward(self):
        positional_embedding = self.positional_embedding
        return positional_embedding


class SampleEmbedding(nn.Module):
    def __init__(self, c_in=4, d_model=288, seq_len=1344, seq_len_up_bound=1344):
        super(SampleEmbedding, self).__init__()
        self.seq_len = seq_len
        self.seq_len_up_bound = seq_len_up_bound

        if seq_len > seq_len_up_bound:
            self.stem_embedding = StemEmbedding(c_in=c_in, d_model=d_model)
            # pool 2 4 4 4
            seq_len = seq_len // (2 * 4 * 4 * 4)
        else:
            self.base_pair_embedding = BasePairEmbedding(c_in=c_in, d_model=d_model)

        self.positional_embedding = PositionalEmbedding(d_model=d_model, seq_len=seq_len)

    def forward(self, x):
        device = x.device
        positional_embedding = self.positional_embedding()
        positional_embedding = positional_embedding.to(device)
        if self.seq_len > self.seq_len_up_bound:
            stem_embedding = self.stem_embedding(x)
            embedding = stem_embedding + positional_embedding
        else:
            base_pair_embedding = self.base_pair_embedding(x)
            base_pair_embedding = base_pair_embedding.to(device)
            embedding = base_pair_embedding + positional_embedding

        return embedding


# 2. Encoder
class ProbAttention(nn.Module):
    def __init__(self):
        super(ProbAttention, self).__init__()

    def prob_q_k(self, query, key, down_sample_k, n_query):
        b, h, l_key, dim = key.shape
        _, _, l_query, _ = query.shape

        medium_key = key.unsqueeze(-3).expand(b, h, l_query, l_key, dim)

        index_key = torch.randint(high=l_key, size=(l_query, down_sample_k))

        medium_key = medium_key[:, :, torch.arange(l_query).unsqueeze(1), index_key, :]

        medium_query_key = torch.matmul(query.unsqueeze(-2), medium_key.transpose(-2, -1)).squeeze(
            -2
        )

        index_query = medium_query_key.max(-1).values - torch.div(medium_query_key.sum(-1), l_key)
        index_query = index_query.topk(k=n_query, sorted=False).indices

        query = query[
            torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], index_query, :
        ]

        query_key = torch.matmul(query, key.transpose(-2, -1))

        return query_key, index_query

    def _ini_value_uniform(self, value, l_query):
        b, h, l_value, dim = value.shape

        value_uniform = value.mean(dim=-2)
        value_uniform = value_uniform.unsqueeze(-2).expand(b, h, l_query, dim).clone()

        return value_uniform

    def prob_similarity_v(self, value_uniform, value, similarity, index):
        b, h, l_value, dim = value_uniform.shape

        a_map = torch.softmax(similarity, dim=-1)

        value_uniform[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], index, :] = (
            torch.matmul(a_map, value).type_as(value_uniform)
        )

        return value_uniform

    def forward(self, query, key, value):
        b, l_query, h, dim = query.shape
        _, l_key, _, _ = key.shape

        down_sample_k = l_key // 4
        down_sample_q = l_query // 4

        query = query.transpose(2, 1)
        key = key.transpose(2, 1)
        value = value.transpose(2, 1)

        similarity, query_index = self.prob_q_k(
            query=query, key=key, down_sample_k=down_sample_k, n_query=down_sample_q
        )
        scale = 1.0 / math.sqrt(dim)
        similarity = similarity * scale

        value_uniform = self._ini_value_uniform(value=value, l_query=l_query)
        q_k_v = self.prob_similarity_v(
            value=value, value_uniform=value_uniform, similarity=similarity, index=query_index
        )
        q_k_v = q_k_v.transpose(2, 1).contiguous()

        return q_k_v


class SelfAttention(nn.Module):
    def __init__(self, d_models=288, n_heads=8):
        super(SelfAttention, self).__init__()
        d_query = d_models // n_heads
        d_key = d_models // n_heads
        d_value = d_models // n_heads

        self.query_linear = nn.Linear(d_models, d_query * n_heads)
        self.key_linear = nn.Linear(d_models, d_key * n_heads)
        self.value_linear = nn.Linear(d_models, d_value * n_heads)

        self.prob_attention = ProbAttention()

        self.heads = n_heads

    def forward(self, query, key, value):
        b, l, _ = query.shape  # noqa: E741 (vendored verbatim)

        query = self.query_linear(query).view(b, l, self.heads, -1)
        key = self.key_linear(key).view(b, l, self.heads, -1)
        value = self.value_linear(value).view(b, l, self.heads, -1)

        prob_attention = self.prob_attention(query, key, value)
        prob_attention = prob_attention.view(b, l, -1)

        return prob_attention


class EncoderLayer(nn.Module):
    def __init__(self, d_models=288, n_heads=8):
        super(EncoderLayer, self).__init__()
        d_ff = d_models * 2

        self.self_attention = SelfAttention(d_models=d_models, n_heads=n_heads)

        self.conv_ffn_1 = nn.Conv1d(in_channels=d_models, out_channels=d_ff, kernel_size=1)
        self.gelu_ffn_1 = nn.GELU()
        self.conv_ffn_2 = nn.Conv1d(in_channels=d_ff, out_channels=d_models, kernel_size=1)
        self.gelu_ffn_2 = nn.GELU()

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=d_models)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=d_models)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        self_attention = self.self_attention(x, x, x)
        self_attention = x + self.dropout(self_attention)

        layer_norm_1 = self.layer_norm_1(self_attention)
        layer_norm_1 = layer_norm_1.transpose(-1, 1)

        conv_ffn_1 = self.conv_ffn_1(layer_norm_1)
        conv_ffn_1 = self.gelu_ffn_1(conv_ffn_1)

        conv_ffn_2 = self.conv_ffn_2(conv_ffn_1)
        conv_ffn_2 = self.gelu_ffn_2(conv_ffn_2)
        conv_ffn_2 = conv_ffn_2.transpose(-1, 1)

        ffn = conv_ffn_2 + layer_norm_1.transpose(-1, 1)
        layer_norm_2 = self.layer_norm_2(ffn)

        return layer_norm_2


class PoolingLayer(nn.Module):
    def __init__(self, c_in, pool_degree):
        super(PoolingLayer, self).__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=c_in, out_channels=c_in, kernel_size=5, padding=2, padding_mode="zeros"
        )
        self.norm = nn.BatchNorm1d(num_features=c_in)
        self.elu = nn.ELU()

        self.pooling = nn.MaxPool1d(kernel_size=pool_degree)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv = self.conv_layer(x)
        conv = self.norm(conv)
        conv = self.elu(conv)

        pooling = self.pooling(conv)
        pooling = pooling.permute(0, 2, 1)

        return pooling


class Encoder(nn.Module):
    def __init__(self, d_models=288, n_heads=8):
        super(Encoder, self).__init__()
        self.encoder_layer_1 = EncoderLayer(d_models=d_models, n_heads=n_heads)
        self.pooling_layer_1 = PoolingLayer(c_in=d_models, pool_degree=3)

        self.encoder_layer_2 = EncoderLayer(d_models=d_models, n_heads=n_heads)
        self.pooling_layer_2 = PoolingLayer(c_in=d_models, pool_degree=2)

        self.encoder_layer_3 = EncoderLayer(d_models=d_models, n_heads=n_heads)
        self.pooling_layer_3 = PoolingLayer(c_in=d_models, pool_degree=2)

        self.encoder_layer_4 = EncoderLayer(d_models=d_models, n_heads=n_heads)
        self.pooling_layer_4 = PoolingLayer(c_in=d_models, pool_degree=2)

        self.encoder_layer_5 = EncoderLayer(d_models=d_models, n_heads=n_heads)
        self.pooling_layer_5 = PoolingLayer(c_in=d_models, pool_degree=2)

        self.encoder_layer_6 = EncoderLayer(d_models=d_models, n_heads=n_heads)
        self.pooling_layer_6 = PoolingLayer(c_in=d_models, pool_degree=2)

    def forward(self, x):
        encoder_layer_1 = self.encoder_layer_1(x)
        pooling_layer_1 = self.pooling_layer_1(encoder_layer_1)

        encoder_layer_2 = self.encoder_layer_2(pooling_layer_1)
        pooling_layer_2 = self.pooling_layer_2(encoder_layer_2)

        encoder_layer_3 = self.encoder_layer_3(pooling_layer_2)
        pooling_layer_3 = self.pooling_layer_3(encoder_layer_3)

        encoder_layer_4 = self.encoder_layer_4(pooling_layer_3)
        pooling_layer_4 = self.pooling_layer_4(encoder_layer_4)

        encoder_layer_5 = self.encoder_layer_5(pooling_layer_4)
        pooling_layer_5 = self.pooling_layer_5(encoder_layer_5)

        encoder_layer_6 = self.encoder_layer_6(pooling_layer_5)
        pooling_layer_6 = self.pooling_layer_6(encoder_layer_6)

        return pooling_layer_6


# 3. Bottleneck
class BottleneckLayer(nn.Module):
    def __init__(self, seq_len, d_models=288):
        super(BottleneckLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=d_models, out_channels=d_models // 2, kernel_size=1)
        self.conv_norm = nn.BatchNorm1d(num_features=d_models // 2)
        self.conv_elu = nn.ELU()

        self.pooling = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten(start_dim=1)

        # 720 (131072, 1000), 1008 (1344), 432 (600)
        self.feature_map = {600: 432, 1000: 720, 1344: 1008, 131072: 720}
        self.linear = nn.Linear(in_features=self.feature_map[seq_len], out_features=32)
        self.linear_norm = nn.LayerNorm(normalized_shape=32)
        self.dropout = nn.Dropout(p=0.2)
        self.linear_elu = nn.ELU()

    def forward(self, x):
        conv = self.conv(x)
        conv = self.conv_norm(conv)
        conv = self.conv_elu(conv)

        pooling = self.pooling(conv)

        f = self.flatten(pooling)

        peak_embedding = self.linear(f)
        peak_embedding = self.linear_norm(peak_embedding)
        peak_embedding = self.dropout(peak_embedding)
        peak_embedding = self.linear_elu(peak_embedding)

        return peak_embedding


# 4. Prediction
class Prediction(nn.Module):
    def __init__(self, n_cells):
        super(Prediction, self).__init__()
        self.linear = nn.Linear(in_features=32, out_features=n_cells)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear(x)
        y = self.sigmoid(y)

        return y


# 5. Informer
class Protrait(nn.Module):
    def __init__(
        self, seq_len=1344, seq_len_up_bound=1344, c_in=4, d_model=288, n_heads=8, n_cells=2034
    ):
        super(Protrait, self).__init__()
        self.seq_len = seq_len
        self.embedding = SampleEmbedding(
            c_in=c_in, d_model=d_model, seq_len=seq_len, seq_len_up_bound=seq_len_up_bound
        )
        self.encoder = Encoder(d_models=d_model, n_heads=n_heads)

        self.bottle_neck = BottleneckLayer(d_models=d_model, seq_len=self.seq_len)

        self.prediction = Prediction(n_cells=n_cells)

    def forward(self, x):
        embedding = self.embedding(x)
        encoder = self.encoder(embedding)
        encoder = encoder.permute(0, 2, 1)

        bottle_neck = self.bottle_neck(encoder)

        y = self.prediction(bottle_neck)
        return y


def build_protrait():
    # Real training entrypoint (public/train.py:35) calls Protrait(seq_len=1344), which
    # hits the class defaults (c_in=4, d_model=288, n_heads=8, n_cells=2034). We use
    # seq_len=600 (one of BottleneckLayer's own supported feature_map keys) with
    # seq_len_up_bound=1344 (BasePairEmbedding branch, not the deeper StemEmbedding
    # cascade) and a tiny n_cells=4 output head for a fast trace; every mechanism is
    # otherwise identical to the real class.
    return Protrait(seq_len=600, seq_len_up_bound=1344, c_in=4, d_model=288, n_heads=8, n_cells=4)


def example_input_protrait():
    # (batch, c_in=4 one-hot bases, seq_len=600) DNA sequence window.
    return torch.randn(2, 4, 600)


MENAGERIE_ENTRIES = [
    ("Portal", "build_portal", "example_input_portal", 2022, "vendored-pytorch"),
    ("PrismNet", "build_prismnet", "example_input_prismnet", 2021, "vendored-pytorch"),
    ("PROTRAIT", "build_protrait", "example_input_protrait", 2023, "vendored-pytorch"),
]
