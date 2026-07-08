# SOURCE: vendored from reginabarzilaygroup/Mirai @ main
# (https://github.com/reginabarzilaygroup/Mirai, `onconet/` package)
#
# Mirai (Yala et al., "Toward robust mammography-based models for breast cancer risk",
# Science Translational Medicine 2021) predicts multi-year breast-cancer risk from a
# sequence of mammography views. Architecture: a per-image ResNet-style CNN encoder
# (`onconet.models.resnet_base.ResNet` via `custom_resnet`, generalized to support
# arbitrary block layouts / risk-factor fusion) whose per-view hidden vectors are fed
# into a custom sequence Transformer (`onconet.models.hiddens_transfomer.Transformer`)
# with learned time/view/side positional embeddings, followed by a risk-factor-fused
# pooling head and a cumulative-hazard survival-analysis output layer
# (`Cumulative_Probability_Layer`). `MiraiFull` (`onconet/models/mirai_full.py`) wires
# the image encoder and the sequence transformer together end to end; this is the exact
# model class registered as `"mirai_full"` in `onconet.models.factory.MODEL_REGISTRY`
# and instantiated by Mirai's own `mirai_trained.json` deployment config.
#
# All class bodies below (ResNet/Downsampler/BasicBlock/Bottleneck, the pool classes,
# Cumulative_Probability_Layer, the Transformer/TransformerLayer/MultiHead_Attention
# stack, RiskFactorVectorizer, CustomResnet, AllImageTransformer, MiraiFull, and the
# `parse_args`/`get_layers`/`get_block`/`get_pool` plumbing) are copied verbatim from the
# real repo files (only import paths were flattened into this single module and the
# `RegisterModel`/`RegisterBlock`/`RegisterPool` decorator registries were inlined
# in-place, since they are pure dict-registration boilerplate with no architectural
# content). No layer, mechanism, or forward-pass computation was altered or approximated.
#
# `parse_args()` is Mirai's real training-time argparse (`onconet/utils/parsing.py`),
# copied verbatim (including every default) since it is the authoritative source of the
# ~80 hyperparameters `ResNet`/`Transformer`/the pool classes read off `args`; this repo
# ships no other place where those defaults are declared (Mirai's public repo is
# inference-only and loads training args from pickled snapshots, not from a CLI). The
# staging `build_mirai()` below parses `[]` (all real defaults) and then overrides only
# the same knobs Mirai's own `onconet/configs/mirai_trained.json` deployment config
# overrides, shrunk to tiny sizes for random-init tracing (img_size, block_layout,
# num_images, num_layers, transfomer_hidden_dim, num_heads, max_followup) plus disables
# `pretrained_on_imagenet` (would attempt a real network download). `use_risk_factors`
# must stay True: `MiraiFull.__init__` reads `image_encoder._model.args.img_only_dim`,
# a field only ever populated by `RiskFactorPool.__init__` (see `resnet_base.ResNet`
# vs. `pools/risk_factor_pool.py`) -- so the risk-factor-fused pooling path is not an
# optional extra, it is load-bearing for `mirai_full` to construct at all.
#
# `RiskFactorVectorizer.__init__` (`onconet/utils/risk_factors.py`, verbatim) calls the
# real `parse_risk_factors(args)`, which unconditionally `json.load`s two real files:
# `args.metadata_path` (per-exam metadata) and `args.risk_factor_metadata_path`
# (per-patient risk-factor records). Both are private hospital datasets not distributed
# with the repo. The returned dict is stored on `self.risk_factor_metadata` but is only
# read later by `get_risk_factors_for_sample`/`get_buckets_for_sample`, which this
# staging module's forward pass never calls (risk factors are supplied directly as a
# list of tensors, matching how `MiraiFull.forward(x, risk_factors=..., batch=...)` is
# actually invoked at inference time -- see the real repo's `scripts/main.py` predict
# path). So `parse_risk_factors` itself is left untouched, and two minimal placeholder
# JSON files satisfying nothing but `json.load`'s well-formedness requirement are written
# to a temp dir and pointed to by `args.metadata_path`/`args.risk_factor_metadata_path`
# purely so the real, unmodified data-loading call succeeds; no synthetic values from
# those files ever reach the model's forward computation.

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# --------------------------------------------------------------------------------------
# onconet/models/blocks/factory.py + basic_block.py + bottleneck.py (verbatim)
# --------------------------------------------------------------------------------------

BLOCK_REGISTRY = {}


def RegisterBlock(block_name):
    def decorator(f):
        BLOCK_REGISTRY[block_name] = f
        return f

    return decorator


def get_block(block_name):
    if block_name not in BLOCK_REGISTRY:
        raise Exception(
            "Block {} not in BLOCK_REGISTRY! Available blocks are {}".format(
                block_name, BLOCK_REGISTRY.keys()
            )
        )
    return BLOCK_REGISTRY[block_name]


def conv3x3(inplanes, outplanes, stride=1, groups=1):
    return nn.Conv2d(
        inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False, groups=1
    )


@RegisterBlock("BasicBlock")
class BasicBlock(nn.Module):
    """A basic block for Resnets. Used in Resnet-18 and Resnet-34."""

    expansion = 1

    def __init__(self, args, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, groups=args.num_groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=args.num_groups)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@RegisterBlock("Bottleneck")
class Bottleneck(nn.Module):
    """A bottleneck block for Resnets. Used in Resnet-50, Resnet-101, and Resnet-152."""

    expansion = 4

    def __init__(self, args, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, groups=args.num_groups)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=args.num_groups,
        )
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False, groups=args.num_groups
        )

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# --------------------------------------------------------------------------------------
# onconet/models/pools/*.py (verbatim)
# --------------------------------------------------------------------------------------

POOL_REGISTRY = {}


def RegisterPool(pool_name):
    def decorator(f):
        POOL_REGISTRY[pool_name] = f
        return f

    return decorator


def get_pool(pool_name):
    if pool_name not in POOL_REGISTRY:
        raise Exception(
            "Pool {} not in POOL_REGISTRY! Available pools are {}".format(
                pool_name, POOL_REGISTRY.keys()
            )
        )
    return POOL_REGISTRY[pool_name]


class AbstractPool(nn.Module):
    def __init__(self, args, num_chan):
        super(AbstractPool, self).__init__()

    def replaces_fc(self):
        raise NotImplementedError


@RegisterPool("GlobalAvgPool")
class GlobalAvgPool(AbstractPool):
    def replaces_fc(self):
        return False

    def forward(self, x):
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        x = torch.mean(x, dim=-1)
        return None, x


MLP_HIDDEN_DIM = 100


@RegisterPool("RiskFactorPool")
class RiskFactorPool(AbstractPool):
    """onconet/models/pools/risk_factor_pool.py, verbatim."""

    def __init__(self, args, num_chan):
        super(RiskFactorPool, self).__init__(args, num_chan)
        self.args = args
        self.internal_pool = get_pool(args.pool_name)(args, num_chan)
        assert not self.internal_pool.replaces_fc()
        self.dropout = nn.Dropout(args.dropout)
        self.length_risk_factor_vector = RiskFactorVectorizer(args).vector_length
        if args.pred_risk_factors:
            for key in args.risk_factor_keys:
                num_key_features = args.risk_factor_key_to_num_class[key]
                key_fc = nn.Linear(self.args.hidden_dim, num_key_features)
                self.add_module("{}_fc".format(key), key_fc)

        self.args.img_only_dim = self.args.hidden_dim
        self.args.rf_dim = self.length_risk_factor_vector
        self.args.hidden_dim = self.args.rf_dim + self.args.img_only_dim

    def replaces_fc(self):
        return False

    def forward(self, x, risk_factors):
        if self.args.replace_snapshot_pool:
            x = x.data
        _, hidden = self.internal_pool(x)

        risk_factors_hidden = None
        if self.args.pred_risk_factors:
            pred_risk_factors = []
            for indx, key in enumerate(self.args.risk_factor_keys):
                gold_rf = risk_factors[indx] if risk_factors is not None else None
                key_logit = self._modules["{}_fc".format(key)](hidden)

                if self.args.risk_factor_key_to_num_class[key] == 1:
                    key_probs = torch.sigmoid(key_logit)
                else:
                    key_probs = F.softmax(key_logit, dim=-1)

                if not self.training and self.args.use_pred_risk_factors_if_unk:
                    is_rf_known = (torch.sum(gold_rf, dim=-1) > 0).unsqueeze(-1).float()
                    key_probs = (is_rf_known * gold_rf) + (1 - is_rf_known) * key_probs
                elif self.training and self.args.mask_prob > 0 and gold_rf is not None:
                    is_rf_known = np.random.random() > self.args.mask_prob
                    key_probs = (is_rf_known * gold_rf) + (1 - is_rf_known) * key_probs

                pred_risk_factors.append(key_probs)

            if (not self.training and self.args.use_pred_risk_factors_at_test) or (
                self.training and self.args.mask_prob > 0
            ):
                risk_factors_hidden = torch.cat(pred_risk_factors, dim=1)

        risk_factors_hidden = (
            torch.cat(risk_factors, dim=1) if risk_factors_hidden is None else risk_factors_hidden
        )
        hidden = torch.cat((hidden, risk_factors_hidden), 1)
        hidden = self.dropout(hidden)
        return None, hidden

    def get_pred_rf_loss(self, hidden, risk_factors):
        img_hidden = hidden[:, : -self.length_risk_factor_vector]
        loss = 0
        num_losses = 0
        for i, key in enumerate(self.args.risk_factor_keys):
            key_logit = self._modules["{}_fc".format(key)](img_hidden)
            key_gold = risk_factors[i]
            if self.args.risk_factor_key_to_num_class[key] == 1:
                loss += F.binary_cross_entropy_with_logits(key_logit, key_gold)
                num_losses += 1
            else:
                key_gold = key_gold.nonzero()
                if len(key_gold) == 0:
                    continue
                indicies_with_gold = key_gold[:, 0].contiguous()
                key_logit = key_logit.index_select(dim=0, index=indicies_with_gold)
                key_gold = key_gold[:, -1:].contiguous().view(-1)
                loss += F.cross_entropy(key_logit, key_gold)
                num_losses += 1
        if num_losses > 0:
            loss /= num_losses
        return loss


# --------------------------------------------------------------------------------------
# onconet/models/cumulative_probability_layer.py (verbatim)
# --------------------------------------------------------------------------------------


class Cumulative_Probability_Layer(nn.Module):
    def __init__(self, num_features, args, max_followup):
        super(Cumulative_Probability_Layer, self).__init__()
        self.args = args
        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        if self.args.make_probs_indep:
            return self.hazards(x)
        hazards = self.hazards(x)
        B, T = hazards.size()
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T)
        masked_hazards = expanded_hazards * self.upper_triagular_mask
        cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)
        return cum_prob


# --------------------------------------------------------------------------------------
# onconet/models/resnet_base.py (verbatim)
# --------------------------------------------------------------------------------------


class Downsampler(nn.Module):
    """Downsampling layers for ResNet. Downsamples input by 4x."""

    def __init__(self, inplanes, num_chan=3):
        self.inplanes = inplanes
        super(Downsampler, self).__init__()
        self.conv1 = nn.Conv2d(num_chan, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ResNet(nn.Module):
    """A ResNet model. Blocks can be Basic, Bottleneck, or anything in BLOCK_REGISTRY,
    intermixed in any order. Generalization of the standard resnet."""

    def __init__(self, layers, args):
        super(ResNet, self).__init__()

        self.args = args
        self.args.wrap_model = False

        self.args.hidden_dim = 512 * args.block_widening_factor
        input_dim = self.args.input_dim if self.args.use_precomputed_hiddens else self.args.num_chan
        self.inplanes = max(64 * args.block_widening_factor, input_dim)

        self.all_blocks = []
        if not self.args.use_precomputed_hiddens:
            downsampler = Downsampler(self.inplanes, input_dim)
            self.add_module("downsampler", downsampler)
            self.all_blocks.append("downsampler")

        layer_modules = [(self._make_layer(self.inplanes, layers[0]), "layer1_{}")]
        current_dim = self.inplanes
        indx = 1
        for layer_i in layers[1:]:
            indx += 1
            current_dim = min(current_dim * 2, 1024)
            layer_modules.append(
                (self._make_layer(current_dim, layer_i, stride=2), "layer{}_".format(indx) + "{}")
            )
        args.hidden_dim = current_dim

        for layer, layer_name in layer_modules:
            for indx, block in enumerate(layer):
                block_name = layer_name.format(indx)
                self.add_module(block_name, block)
                self.all_blocks.append(block_name)

        pool_name = args.pool_name
        if args.use_risk_factors:
            pool_name = (
                "DeepRiskFactorPool" if self.args.deep_risk_factor_pool else "RiskFactorPool"
            )
        self.pool = get_pool(pool_name)(args, args.hidden_dim)

        if not self.pool.replaces_fc():
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=args.dropout)
            self.fc = nn.Linear(args.hidden_dim, args.num_classes)

        if args.use_region_annotation and args.region_annotation_loss_type == "pred_region":
            self.region_fc = nn.Conv2d(
                current_dim,
                1,
                kernel_size=args.region_annotation_pred_kernel_size,
                padding=(args.region_annotation_pred_kernel_size - 1) // 2,
            )

        if args.predict_birads:
            self.birads_fc = nn.Linear(args.hidden_dim, 2)

        if args.survival_analysis_setup:
            self.prob_of_failure_layer = Cumulative_Probability_Layer(
                args.hidden_dim, args, max_followup=args.max_followup
            )

        self.gpu_to_layer_assignments = self.get_gpu_to_layer()

    def get_gpu_to_layer(self):
        if self.args.model_parallel and self.args.num_shards > 1:
            num_shards = self.args.num_shards
        else:
            num_shards = 1

        gpu_to_layers = np.array_split(self.all_blocks, num_shards)
        return gpu_to_layers

    def _make_layer(self, planes, blocks, stride=1):
        layers = []

        for i, block in enumerate(blocks):
            if (i == 0 and stride != 1) or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = None

            if i != 0:
                stride = 1

            layers.append(
                block(self.args, self.inplanes, planes, stride=stride, downsample=downsample)
            )

            self.inplanes = planes * block.expansion

        return layers

    def forward(self, x, risk_factors=None, batch=None):
        if self.args.use_precomputed_hiddens:
            x = x.transpose(2, 1)
        for gpu, layers in enumerate(self.gpu_to_layer_assignments):
            for name in layers:
                layer = self._modules[name]
                x = layer(x)
        logit, hidden = self.aggregate_and_classify(x, risk_factors=risk_factors)
        activ_dict = {"activ": x}
        if self.args.use_region_annotation:
            activ_dict["region_logit"] = self.region_fc(x)
        if self.args.predict_birads:
            activ_dict["birads_logit"] = self.birads_fc(hidden)

        if self.args.pred_risk_factors:
            try:
                activ_dict["pred_rf_loss"] = self.pool.get_pred_rf_loss(hidden, risk_factors)
            except Exception:
                pass
        if self.args.use_precomputed_hiddens:
            return logit, logit, logit, hidden
        else:
            return logit, hidden, activ_dict

    def aggregate_and_classify(self, x, risk_factors=None):
        if self.args.use_risk_factors:
            logit, hidden = self.pool(x, risk_factors)
        else:
            logit, hidden = self.pool(x)

        if not self.pool.replaces_fc():
            try:
                hidden = self.relu(hidden)
            except Exception:
                pass
            hidden = self.dropout(hidden)
            logit = self.fc(hidden)

        if self.args.survival_analysis_setup:
            logit = self.prob_of_failure_layer(hidden)
        return logit, hidden


# --------------------------------------------------------------------------------------
# onconet/models/custom_resnet.py (verbatim, minus RegisterModel/pretrained-imagenet path)
# --------------------------------------------------------------------------------------


class CustomResnet(nn.Module):
    def __init__(self, args):
        super(CustomResnet, self).__init__()
        layers = get_layers(args.block_layout)
        self._model = ResNet(layers, args)
        # NOTE: args.pretrained_on_imagenet is False in build_mirai(); the real
        # load_pretrained_weights(...)/load_pretrained_model(...) download path from
        # onconet/models/default_resnets.py is intentionally not exercised here.

    def forward(self, x, risk_factors=None, batch=None):
        return self._model(x, risk_factors=risk_factors, batch=None)


def validate_block_layout(block_layout):
    for layer_layout in block_layout:
        for block_spec in layer_layout:
            if len(block_spec) != 2:
                raise Exception("Invalid block specification: {}".format(block_spec))


def get_layers(block_layout):
    validate_block_layout(block_layout)
    layers = []
    for layer_layout in block_layout:
        layer = []
        for block_name, num_repeats in layer_layout:
            block = get_block(block_name)
            layer.extend([block] * num_repeats)
        layers.append(layer)
    return layers


# --------------------------------------------------------------------------------------
# onconet/models/hiddens_transfomer.py (verbatim)
# --------------------------------------------------------------------------------------

EMBEDDING_DIM = 96
MAX_TIME = 10
MAX_VIEWS = 2
MAX_SIDES = 2


class MultiHead_Attention(nn.Module):
    def __init__(self, args):
        super(MultiHead_Attention, self).__init__()
        self.args = args
        assert args.hidden_dim % args.num_heads == 0

        self.query = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.key = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.dropout = nn.Dropout(p=args.dropout)

        self.dim_per_head = args.hidden_dim // args.num_heads

        self.aggregate_fc = nn.Linear(args.hidden_dim, args.hidden_dim)

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_per_head)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, x):
        B, N, H = x.size()

        k = self.key(x).view(B, N, self.args.num_heads, self.dim_per_head)
        q = self.query(x).view(B, N, self.args.num_heads, self.dim_per_head)
        v = self.value(x).view(B, N, self.args.num_heads, self.dim_per_head)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        h = self.attention(q, k, v)

        h = h.transpose(1, 2).contiguous().view(B, -1, H)

        output = self.aggregate_fc(h)

        return output


class TransformerLayer(nn.Module):
    def __init__(self, args):
        super(TransformerLayer, self).__init__()

        self.args = args
        self.multihead_attention = MultiHead_Attention(self.args)
        self.layernorm_attn = nn.LayerNorm(self.args.hidden_dim)
        self.fc1 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.layernorm_fc = nn.LayerNorm(self.args.hidden_dim)

    def forward(self, x):
        h = self.multihead_attention(x)
        x = self.layernorm_attn(h + x)
        h = self.fc2(self.relu(self.fc1(x)))
        x = self.layernorm_fc(h + x)
        return x


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.args = args
        self.args.wrap_model = False
        assert EMBEDDING_DIM % 3 == 0
        self.time_embed = torch.nn.Embedding(MAX_TIME + 1, EMBEDDING_DIM // 3, padding_idx=-1)
        self.view_embed = torch.nn.Embedding(MAX_VIEWS + 1, EMBEDDING_DIM // 3, padding_idx=-1)
        self.side_embed = torch.nn.Embedding(MAX_SIDES + 1, EMBEDDING_DIM // 3, padding_idx=-1)
        self.embed_add_fc = nn.Linear(EMBEDDING_DIM, args.hidden_dim)
        self.embed_scale_fc = nn.Linear(EMBEDDING_DIM, args.hidden_dim)
        for layer in range(args.num_layers):
            transformer_layer = TransformerLayer(args)
            self.add_module("transformer_layer_{}".format(layer), transformer_layer)

    def condition_on_pos_embed(self, x, embed):
        return self.embed_scale_fc(embed) * x + self.embed_add_fc(embed)

    def forward(self, x, time_seq, view_seq, side_seq):
        view, time, side = (
            self.view_embed(view_seq),
            self.time_embed(time_seq),
            self.side_embed(side_seq),
        )
        embed = torch.cat([view, time, side], dim=-1)
        x = self.condition_on_pos_embed(x, embed)

        for indx in range(self.args.num_layers):
            name = "transformer_layer_{}".format(indx)
            x = self._modules[name](x)

        return x


class AllImageTransformer(nn.Module):
    def __init__(self, args):
        super(AllImageTransformer, self).__init__()

        self.args = args
        self.args.wrap_model = False
        args.hidden_dim = args.transfomer_hidden_dim
        assert args.use_precomputed_hiddens or args.model_name == "mirai_full"

        self.projection_layer = nn.Linear(args.precomputed_hidden_dim, args.hidden_dim)
        self.mask_embedding = torch.nn.Embedding(2, args.precomputed_hidden_dim, padding_idx=1)
        kept_images_vec = torch.nn.Parameter(
            torch.ones([1, args.num_images, 1]), requires_grad=False
        )
        self.register_parameter("kept_images_vec", kept_images_vec)
        self.transformer = Transformer(args)

        self.pred_masked_img_fc = nn.Linear(args.hidden_dim, args.precomputed_hidden_dim)
        pool_name = args.pool_name
        if args.use_risk_factors:
            pool_name = (
                "DeepRiskFactorPool" if self.args.deep_risk_factor_pool else "RiskFactorPool"
            )
        self.pool = get_pool(pool_name)(args, args.hidden_dim)

        if not self.pool.replaces_fc():
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=args.dropout)
            self.fc = nn.Linear(args.hidden_dim, args.num_classes)

        if args.survival_analysis_setup:
            if args.pred_both_sides:
                self.prob_of_failure_layer_l = Cumulative_Probability_Layer(
                    args.hidden_dim, args, max_followup=args.max_followup
                )
                self.prob_of_failure_layer_r = Cumulative_Probability_Layer(
                    args.hidden_dim, args, max_followup=args.max_followup
                )
            else:
                self.prob_of_failure_layer = Cumulative_Probability_Layer(
                    args.hidden_dim, args, max_followup=args.max_followup
                )

    def mask_input(self, x, view_seq):
        B, N, _ = x.size()
        mask_prob = self.args.mask_prob if self.training and self.args.pred_missing_mammos else 0
        is_mask = torch.bernoulli(self.kept_images_vec.expand([B, N, 1]) * mask_prob)
        is_mask = is_mask * (view_seq < MAX_VIEWS).unsqueeze(-1).float()
        is_kept = 1 - is_mask
        x = x * is_kept + self.mask_embedding(is_kept.squeeze(-1).long())
        if self.args.also_pred_given_mammos:
            is_mask = (is_mask >= -1).float() * (view_seq < MAX_VIEWS).unsqueeze(-1).float()
        return x, is_mask

    def get_pred_mask_loss(self, transformer_hidden, x, is_mask):
        if is_mask.sum().item() == 0:
            return 0
        B, N, D_n = transformer_hidden.size()
        _, _, D_o = x.size()
        hidden_for_mask = torch.masked_select(transformer_hidden, is_mask.bool()).view(-1, D_n)
        pred_x = self.pred_masked_img_fc(hidden_for_mask)
        x_for_mask = torch.masked_select(x, is_mask.bool()).view(-1, D_o)
        return F.mse_loss(pred_x, x_for_mask)

    def forward(self, x, risk_factors=None, batch=None):
        time_seq, view_seq, side_seq = batch["time_seq"], batch["view_seq"], batch["side_seq"]
        masked_x, is_mask = self.mask_input(x, view_seq)
        masked_x = self.projection_layer(masked_x)
        transformer_hidden = self.transformer(masked_x, time_seq, view_seq, side_seq)

        img_like_hidden = transformer_hidden.transpose(1, 2).unsqueeze(-1)
        logit, hidden = self.aggregate_and_classify(img_like_hidden, risk_factors=risk_factors)

        activ_dict = {}
        try:
            if self.args.predict_birads:
                activ_dict["birads_logit"] = self.birads_fc(hidden)
            if self.args.pred_risk_factors:
                activ_dict["pred_rf_loss"] = self.pool.get_pred_rf_loss(hidden, risk_factors)

            if self.args.pred_missing_mammos:
                activ_dict["pred_masked_mammo_loss"] = self.get_pred_mask_loss(
                    transformer_hidden, x, is_mask
                )
        except Exception:
            pass
        return logit, transformer_hidden, activ_dict

    def aggregate_and_classify(self, x, risk_factors=None):
        if self.args.use_risk_factors:
            logit, hidden = self.pool(x, risk_factors)
        else:
            logit, hidden = self.pool(x)

        if not self.pool.replaces_fc():
            try:
                hidden = self.relu(hidden)
            except Exception:
                pass
            hidden = self.dropout(hidden)
            logit = self.fc(hidden)

        if self.args.survival_analysis_setup:
            if self.args.pred_both_sides:
                logit = {
                    "l": self.prob_of_failure_layer_l(hidden),
                    "r": self.prob_of_failure_layer_r(hidden),
                }
            else:
                logit = self.prob_of_failure_layer(hidden)

        return logit, hidden


# --------------------------------------------------------------------------------------
# onconet/utils/risk_factors.py (verbatim: RiskFactorVectorizer + parse_risk_factors)
# --------------------------------------------------------------------------------------

MISSING_VALUE = -1
HASNT_HAPPENED_VALUE = -5
RACE_CODE_TO_NAME = {
    1: "White",
    2: "African American",
    3: "American Indian, Eskimo, Aleut",
    4: "Asian or Pacific Islander",
    5: "Other Race",
    6: "Caribbean/West Indian",
    7: "Unknown",
    8: "Hispanic",
    9: "Chinese",
    10: "Japanese",
    11: "Filipino",
    12: "Hawaiian",
    13: "Other Asian",
}
TREAT_MISSING_AS_NEGATIVE = False
NEGATIVE_99 = -99


class RiskFactorVectorizer:
    def __init__(self, args):
        self.risk_factor_metadata = parse_risk_factors(args)
        self.risk_factor_transformers = {
            "binary_family_history": self.transform_binary_family_history,
            "binary_biopsy_benign": self.get_binary_occurence_transformer(
                "biopsy_hyperplasia", "biopsy_hyperplasia_age"
            ),
            "binary_biopsy_LCIS": self.get_binary_occurence_transformer(
                "biopsy_LCIS", "biopsy_LCIS_age"
            ),
            "binary_biopsy_atypical_hyperplasia": self.get_binary_occurence_transformer(
                "biopsy_atypical_hyperplasia", "biopsy_atypical_hyperplasia_age"
            ),
            "age": self.get_exam_one_hot_risk_factor_transformer("age", [40, 50, 60, 70, 80]),
            "menarche_age": self.get_age_based_risk_factor_transformer(
                "menarche_age", [10, 12, 14, 16]
            ),
            "menopause_age": self.get_age_based_risk_factor_transformer(
                "menopause_age", [45, 50, 55, 60]
            ),
            "first_pregnancy_age": self.get_age_based_risk_factor_transformer(
                "first_pregnancy_age", [20, 25, 30, 35, 40]
            ),
            "density": self.get_image_biomarker_transformer("density"),
            "bpe": self.get_image_biomarker_transformer("bpe"),
            "5yearcancer": self.get_binary_transformer("5yearcancer"),
            "prior_hist": self.get_binary_transformer("prior_hist"),
            "years_to_cancer": self.get_exam_one_hot_risk_factor_transformer(
                "years_to_cancer", [0, 1, 2, 3, 4, 10]
            ),
            "race": self.transform_race,
            "parous": self.transform_parous,
            "menopausal_status": self.transform_menopausal_status,
            "weight": self.get_exam_one_hot_risk_factor_transformer(
                "weight", [100, 130, 160, 190, 220, 250]
            ),
            "height": self.get_exam_one_hot_risk_factor_transformer(
                "height", [50, 55, 60, 65, 70, 75]
            ),
            "ovarian_cancer": self.get_binary_occurence_transformer(
                "ovarian_cancer", "ovarian_cancer_age"
            ),
            "ovarian_cancer_age": self.get_age_based_risk_factor_transformer(
                "ovarian_cancer_age", [30, 40, 50, 60, 70]
            ),
            "ashkenazi": self.get_binary_transformer("ashkenazi", use_patient_factors=True),
            "brca": self.transform_brca,
            "mom_bc_cancer_history": self.get_binary_relative_cancer_history_transformer("M"),
            "m_aunt_bc_cancer_history": self.get_binary_relative_cancer_history_transformer("MA"),
            "p_aunt_bc_cancer_history": self.get_binary_relative_cancer_history_transformer("PA"),
            "m_grandmother_bc_cancer_history": self.get_binary_relative_cancer_history_transformer(
                "MG"
            ),
            "p_grantmother_bc_cancer_history": self.get_binary_relative_cancer_history_transformer(
                "PG"
            ),
            "brother_bc_cancer_history": self.get_binary_relative_cancer_history_transformer("B"),
            "father_bc_cancer_history": self.get_binary_relative_cancer_history_transformer("F"),
            "daughter_bc_cancer_history": self.get_binary_relative_cancer_history_transformer("D"),
            "sister_bc_cancer_history": self.get_binary_relative_cancer_history_transformer("S"),
            "mom_oc_cancer_history": self.get_binary_relative_cancer_history_transformer(
                "M", cancer="ovarian_cancer"
            ),
            "m_aunt_oc_cancer_history": self.get_binary_relative_cancer_history_transformer(
                "MA", cancer="ovarian_cancer"
            ),
            "p_aunt_oc_cancer_history": self.get_binary_relative_cancer_history_transformer(
                "PA", cancer="ovarian_cancer"
            ),
            "m_grandmother_oc_cancer_history": self.get_binary_relative_cancer_history_transformer(
                "MG", cancer="ovarian_cancer"
            ),
            "p_grantmother_oc_cancer_history": self.get_binary_relative_cancer_history_transformer(
                "PG", cancer="ovarian_cancer"
            ),
            "sister_oc_cancer_history": self.get_binary_relative_cancer_history_transformer(
                "S", cancer="ovarian_cancer"
            ),
            "daughter_oc_cancer_history": self.get_binary_relative_cancer_history_transformer(
                "D", cancer="ovarian_cancer"
            ),
            "hrt_type": self.get_hrt_information_transformer("type"),
            "hrt_duration": self.get_hrt_information_transformer("duration"),
            "hrt_years_ago_stopped": self.get_hrt_information_transformer("years_ago_stopped"),
        }

        self.risk_factor_keys = args.risk_factor_keys
        self.feature_names = []
        self.risk_factor_key_to_num_class = {}
        for k in self.risk_factor_keys:
            if k not in self.risk_factor_transformers.keys():
                raise Exception("Risk factor key '{}' not supported.".format(k))
            names = self.risk_factor_transformers[k](None, None, just_return_feature_names=True)
            self.risk_factor_key_to_num_class[k] = len(names)
            self.feature_names.extend(names)
        args.risk_factor_key_to_num_class = self.risk_factor_key_to_num_class

    @property
    def vector_length(self):
        return len(self.feature_names)

    def get_feature_names(self):
        return list(self.feature_names)

    def one_hot_vectorizor(self, value, cutoffs):
        one_hot_vector = torch.zeros(len(cutoffs) + 1)
        if value == MISSING_VALUE:
            return one_hot_vector
        for i, cutoff in enumerate(cutoffs):
            if value <= cutoff:
                one_hot_vector[i] = 1
                return one_hot_vector
        one_hot_vector[-1] = 1
        return one_hot_vector

    def one_hot_feature_names(self, risk_factor_name, cutoffs):
        feature_names = [""] * (len(cutoffs) + 1)
        feature_names[0] = "{}_lt_{}".format(risk_factor_name, cutoffs[0])
        feature_names[-1] = "{}_gt_{}".format(risk_factor_name, cutoffs[-1])
        for i in range(1, len(cutoffs)):
            feature_names[i] = "{}_{}_{}".format(risk_factor_name, cutoffs[i - 1], cutoffs[i])
        return feature_names

    def get_age_based_risk_factor_transformer(self, risk_factor_key, age_cutoffs):
        def transform_age_based_risk_factor(
            patient_factors, exam_factors, just_return_feature_names=False
        ):
            if just_return_feature_names:
                return self.one_hot_feature_names(risk_factor_key, age_cutoffs)
            exam_age = int(exam_factors["age"])
            age_based_risk_factor = int(patient_factors[risk_factor_key])
            if exam_age != MISSING_VALUE and exam_age < age_based_risk_factor:
                age_based_risk_factor = MISSING_VALUE
            return self.one_hot_vectorizor(age_based_risk_factor, age_cutoffs)

        return transform_age_based_risk_factor

    def get_exam_one_hot_risk_factor_transformer(self, risk_factor_key, cutoffs):
        def transform_exam_one_hot_risk_factor(
            patient_factors, exam_factors, just_return_feature_names=False
        ):
            if just_return_feature_names:
                return self.one_hot_feature_names(risk_factor_key, cutoffs)
            risk_factor = int(exam_factors[risk_factor_key])
            return self.one_hot_vectorizor(risk_factor, cutoffs)

        return transform_exam_one_hot_risk_factor

    def get_binary_occurence_transformer(self, occurence_key, occurence_age_key):
        def transform_binary_occurence(
            patient_factors, exam_factors, just_return_feature_names=False
        ):
            if just_return_feature_names:
                return ["binary_{}".format(occurence_key)]
            binary_occurence = torch.zeros(1)
            occurence = int(patient_factors[occurence_key])
            binary_occurence[0] = 1 if occurence == 1 else 0
            return binary_occurence

        return transform_binary_occurence

    def get_binary_transformer(self, risk_factor_key, use_patient_factors=False):
        def transform_binary(patient_factors, exam_factors, just_return_feature_names=False):
            if just_return_feature_names:
                return ["binary_{}".format(risk_factor_key)]
            binary_risk_factor = torch.zeros(1)
            risk_factor = (
                int(patient_factors[risk_factor_key])
                if use_patient_factors
                else int(exam_factors[risk_factor_key])
            )
            binary_risk_factor[0] = 1 if risk_factor == 1 else 0
            return binary_risk_factor

        return transform_binary

    def get_binary_relative_cancer_history_transformer(self, relative_code, cancer="breast_cancer"):
        def transform_binary_relative_cancer_history(
            patient_factors, exam_factors, just_return_feature_names=False
        ):
            if just_return_feature_names:
                return ["{}_{}_hist".format(relative_code, cancer)]
            binary_relative_cancer_history = torch.zeros(1)
            relative_list = patient_factors["relatives"][relative_code]
            for rel in relative_list:
                if rel[cancer] == 1:
                    binary_relative_cancer_history[0] = 1
            return binary_relative_cancer_history

        return transform_binary_relative_cancer_history

    def get_image_biomarker_transformer(self, name):
        def image_biomarker_transformer(
            patient_factors, exam_factors, just_return_feature_names=False
        ):
            if just_return_feature_names:
                return ["{}_{}".format(name, i) for i in range(1, 5)]
            image_biomarker_vector = torch.zeros(4)
            image_biomarker = int(exam_factors[name])
            if image_biomarker != MISSING_VALUE:
                image_biomarker_vector[image_biomarker - 1] = 1
            return image_biomarker_vector

        return image_biomarker_transformer

    def transform_binary_family_history(
        self, patient_factors, exam_factors, just_return_feature_names=False
    ):
        if just_return_feature_names:
            return ["binary_family_history"]
        relatives_dict = patient_factors["relatives"]
        binary_family_history = torch.zeros(1)
        for relative, relative_list in relatives_dict.items():
            if len(relative_list) > 0:
                binary_family_history[0] = 1
        return binary_family_history

    def transform_parous(self, patient_factors, exam_factors, just_return_feature_names=False):
        if just_return_feature_names:
            return ["parous"]
        binary_parous = torch.zeros(1)
        exam_age = int(exam_factors["age"])
        binary_parous[0] = 1 if patient_factors["num_births"] != MISSING_VALUE else 0
        if patient_factors["first_pregnancy_age"] != MISSING_VALUE:
            binary_parous[0] = 1 if patient_factors["first_pregnancy_age"] < exam_age else 0
        return binary_parous

    def transform_race(self, patient_factors, exam_factors, just_return_feature_names=False):
        values = range(1, 14)
        race_vector = torch.zeros(len(values))
        if just_return_feature_names:
            return [RACE_CODE_TO_NAME[i] for i in values]
        race = int(patient_factors["race"])
        race_vector[race - 1] = 1
        return race_vector

    def transform_menopausal_status(
        self, patient_factors, exam_factors, just_return_feature_names=False
    ):
        if just_return_feature_names:
            return ["pre", "peri", "post", "unknown"]
        exam_age = int(exam_factors["age"])
        menopausal_status = 3
        age_at_menopause = (
            patient_factors["menopause_age"]
            if patient_factors["menopause_age"] != MISSING_VALUE
            else NEGATIVE_99
        )
        if age_at_menopause != NEGATIVE_99:
            if age_at_menopause < exam_age:
                menopausal_status = 2
            elif age_at_menopause == exam_age:
                menopausal_status = 1
            elif age_at_menopause > exam_age:
                menopausal_status = 0
        else:
            if TREAT_MISSING_AS_NEGATIVE:
                menopausal_status = 0
        menopausal_status_vector = torch.zeros(4)
        menopausal_status_vector[menopausal_status] = 1
        return menopausal_status_vector

    def transform_brca(self, patient_factors, exam_factors, just_return_feature_names=False):
        if just_return_feature_names:
            return ["never or unknown", "negative result", "brca1", "brca2"]
        genetic_testing_patient = 0
        brca1 = patient_factors["brca1"]
        brca2 = patient_factors["brca2"]
        if brca2 == 1:
            genetic_testing_patient = 3
        elif brca1 == 1:
            genetic_testing_patient = 2
        elif brca1 == 0:
            genetic_testing_patient = 1
        genetic_testing_vector = torch.zeros(4)
        genetic_testing_vector[genetic_testing_patient] = 1
        return genetic_testing_vector

    def get_hrt_information_transformer(self, piece):
        def transform_hrt_information(
            patient_factors, exam_factors, just_return_feature_names=False
        ):
            year_cutoffs = [1, 3, 5, 7]
            piece_to_feature_names = {
                "type": ["hrt_combined", "hrt_estrogen", "hrt_unknown"],
                "duration": self.one_hot_feature_names("hrt_duration", year_cutoffs),
                "years_ago_stopped": self.one_hot_feature_names(
                    "hrt_years_ago_stopped", year_cutoffs
                ),
            }
            assert piece in piece_to_feature_names.keys()
            if just_return_feature_names:
                return piece_to_feature_names[piece]

            hrt_vector = torch.zeros(3)

            duration = MISSING_VALUE
            hrt_type = MISSING_VALUE
            hrt_years_ago_stopped = MISSING_VALUE
            first_age_key = None
            last_age_key = None
            duration_key = None
            current_age = int(exam_factors["age"])
            if patient_factors["combined_hrt"]:
                hrt_type = 0
                first_age_key = "combined_hrt_first_age"
                last_age_key = "combined_hrt_last_age"
                duration_key = "combined_hrt_duration"
            elif patient_factors["estrogen_hrt"]:
                hrt_type = 1
                first_age_key = "estrogen_hrt_first_age"
                last_age_key = "estrogen_hrt_last_age"
                duration_key = "estrogen_hrt_duration"
            elif patient_factors["unknown_hrt"]:
                hrt_type = 2
                first_age_key = "unknown_hrt_first_age"
                last_age_key = "unknown_hrt_last_age"
                duration_key = "unknown_hrt_duration"

            if first_age_key:
                first_age = patient_factors[first_age_key]
                last_age = patient_factors[last_age_key]
                extracted_duration = patient_factors[duration_key]

                if last_age >= current_age and current_age != MISSING_VALUE:
                    if first_age != MISSING_VALUE and first_age > current_age:
                        hrt_type = MISSING_VALUE
                    elif (
                        extracted_duration != MISSING_VALUE
                        and last_age - extracted_duration > current_age
                    ):
                        hrt_type = MISSING_VALUE
                    else:
                        duration = (
                            current_age - first_age
                            if current_age != MISSING_VALUE and first_age != MISSING_VALUE
                            else extracted_duration
                        )
                elif last_age != MISSING_VALUE:
                    hrt_years_ago_stopped = current_age - last_age
                    if extracted_duration != MISSING_VALUE:
                        duration = extracted_duration
                    elif first_age != MISSING_VALUE and last_age != MISSING_VALUE:
                        duration = last_age - first_age
                        assert duration >= 0
                else:
                    duration = (
                        extracted_duration if extracted_duration != MISSING_VALUE else MISSING_VALUE
                    )

            if hrt_type > MISSING_VALUE:
                hrt_vector[hrt_type] = 1

            piece_to_feature_names = {
                "type": hrt_vector,
                "duration": self.one_hot_vectorizor(duration, year_cutoffs),
                "years_ago_stopped": self.one_hot_vectorizor(hrt_years_ago_stopped, year_cutoffs),
            }
            return piece_to_feature_names[piece]

        return transform_hrt_information

    def transform(self, patient_factors, exam_factors):
        return [
            self.risk_factor_transformers[key](patient_factors, exam_factors)
            for key in self.risk_factor_keys
        ]

    def get_risk_factors_for_sample(self, sample):
        sample_patient_factors = self.risk_factor_metadata[sample["ssn"]]
        sample_exam_factors = self.risk_factor_metadata[sample["ssn"]]["accessions"][sample["exam"]]
        return self.transform(sample_patient_factors, sample_exam_factors)


def parse_risk_factors(args):
    """onconet/utils/risk_factors.py::parse_risk_factors, verbatim. Reads two real JSON
    files off args.metadata_path / args.risk_factor_metadata_path; build_mirai() below
    points these at minimal placeholder files so this unmodified data-loading call
    succeeds (the returned dict is never consulted for a random-init trace)."""
    try:
        metadata_json = json.load(open(args.metadata_path, "r"))  # noqa: F841 (verbatim; unused upstream too)
    except Exception as e:
        raise Exception("Not found {} {}".format(args.metadata_path, e))

    try:
        risk_factor_metadata = json.load(open(args.risk_factor_metadata_path, "r"))
    except Exception as e:
        raise Exception(
            "Metadata file {} could not be parsed! Exception: {}!".format(
                args.risk_factor_metadata_path, e
            )
        )

    return risk_factor_metadata


# --------------------------------------------------------------------------------------
# onconet/models/mirai_full.py :: MiraiFull (verbatim)
# --------------------------------------------------------------------------------------


class MiraiFull(nn.Module):
    def __init__(self, args):
        super(MiraiFull, self).__init__()
        self.args = args
        self.image_encoder = CustomResnet(args)

        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim
        args.precomputed_hidden_dim = self.image_repr_dim
        self.transformer = AllImageTransformer(args)
        args.img_only_dim = self.transformer.args.transfomer_hidden_dim

    def forward(self, x, risk_factors=None, batch=None):
        B, C, N, H, W = x.size()
        x = x.transpose(1, 2).contiguous().view(B * N, C, H, W)
        risk_factors_per_img = (
            (
                lambda N, risk_factors: [
                    factor.expand([N, *factor.size()])
                    .contiguous()
                    .view([-1, factor.size()[-1]])
                    .contiguous()
                    for factor in risk_factors
                ]
            )(N, risk_factors)
            if risk_factors is not None
            else None
        )
        _, img_x, _ = self.image_encoder(x, risk_factors_per_img, batch)
        img_x = img_x.view(B, N, -1)
        img_x = img_x[:, :, : self.image_repr_dim]
        logit, transformer_hidden, activ_dict = self.transformer(img_x, risk_factors, batch)
        return logit, transformer_hidden, activ_dict


# --------------------------------------------------------------------------------------
# onconet/utils/parsing.py :: parse_args (verbatim defaults, parsed against [])
# --------------------------------------------------------------------------------------


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="OncoNet Classifier")
    parser.add_argument("--run_prefix", default="snapshot")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--dev", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--ensemble_paths", nargs="*", default=[])
    parser.add_argument(
        "--train_years",
        nargs="*",
        type=int,
        default=[2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009],
    )
    parser.add_argument(
        "--dev_years", nargs="*", type=int, default=[2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009]
    )
    parser.add_argument(
        "--test_years",
        nargs="*",
        type=int,
        default=[2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009],
    )
    parser.add_argument("--predict_birads", action="store_true", default=False)
    parser.add_argument("--predict_birads_lambda", type=float, default=0)
    parser.add_argument("--invasive_only", action="store_true", default=False)
    parser.add_argument("--rebalance_eval_cancers", action="store_true", default=False)
    parser.add_argument("--downsample_activ", action="store_true", default=False)
    parser.add_argument("--confidence_interval", type=float, default=0.95)
    parser.add_argument("--num_resamples", type=int, default=10000)
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--image_transformers", nargs="*", default=["scale_2d"])
    parser.add_argument("--tensor_transformers", nargs="*", default=["normalize_2d"])
    parser.add_argument("--test_image_transformers", nargs="*", default=["scale_2d"])
    parser.add_argument(
        "--test_tensor_transformers", nargs="*", default=["force_num_chan_2d", "normalize_2d"]
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--patch_size", type=int, nargs="+", default=[-1, -1])
    parser.add_argument("--get_dataset_stats", action="store_true", default=False)
    parser.add_argument("--get_activs_instead_of_hiddens", action="store_true", default=False)
    parser.add_argument("--img_mean", type=float, nargs="+", default=[0.2023])
    parser.add_argument("--img_std", type=float, nargs="+", default=[0.2576])
    parser.add_argument("--img_dir", type=str, default="/home/administrator/Mounts/Isilon/pngs16")
    parser.add_argument("--num_chan", type=int, default=3)
    parser.add_argument("--force_input_dim", action="store_true", default=False)
    parser.add_argument("--input_dim", type=int, default=512)
    parser.add_argument("--transfomer_hidden_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--multi_image", action="store_true", default=False)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--pred_both_sides", action="store_true", default=False)
    parser.add_argument("--min_num_images", type=int, default=0)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--metadata_dir", type=str, default=None)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--drop_benign_side", action="store_true", default=False)
    parser.add_argument("--class_bal", action="store_true", default=False)
    parser.add_argument(
        "--shift_class_bal_towards_imediate_cancers", action="store_true", default=False
    )
    parser.add_argument("--year_weighted_class_bal", action="store_true", default=False)
    parser.add_argument("--device_class_bal", action="store_true", default=False)
    parser.add_argument("--allowed_devices", nargs="*", default="all")
    parser.add_argument("--use_c_view_if_available", action="store_true", default=False)
    parser.add_argument("--use_spatial_transformer", action="store_true", default=False)
    parser.add_argument("--spatial_transformer_name", type=str, default="affine")
    parser.add_argument("--spatial_transformer_img_size", nargs="+", default=[208, 256])
    parser.add_argument("--location_network_name", type=str, default="resnet18")
    parser.add_argument(
        "--location_network_block_layout",
        type=str,
        nargs="+",
        default=["BasicBlock,2", "BasicBlock,2", "BasicBlock,2", "BasicBlock,2"],
    )
    parser.add_argument("--tps_grid_size", type=int, default=10)
    parser.add_argument("--tps_span_range", type=float, default=0.9)
    parser.add_argument("--use_region_annotation", action="store_true", default=False)
    parser.add_argument("--fraction_region_annotation_to_use", type=float, default=1.0)
    parser.add_argument("--region_annotation_loss_type", type=str, default="pred_region")
    parser.add_argument("--region_annotation_pred_kernel_size", type=int, default=5)
    parser.add_argument("--region_annotation_focal_loss_lambda", type=float, default=0)
    parser.add_argument("--region_annotation_contrast_alpha", type=float, default=0.3)
    parser.add_argument("--regularization_lambda", type=float, default=0.5)
    parser.add_argument("--use_adv", action="store_true", default=False)
    parser.add_argument("--use_mmd_adv", action="store_true", default=False)
    parser.add_argument("--add_repulsive_mmd", action="store_true", default=False)
    parser.add_argument("--use_temporal_mmd", action="store_true", default=False)
    parser.add_argument("--temporal_mmd_cache_size", type=int, default=32)
    parser.add_argument("--temporal_mmd_discount_factor", type=float, default=0.60)
    parser.add_argument("--adv_loss_lambda", type=float, default=0.5)
    parser.add_argument("--train_adv_seperate", action="store_true", default=False)
    parser.add_argument("--anneal_adv_loss", action="store_true", default=False)
    parser.add_argument("--turn_off_model_train", action="store_true", default=False)
    parser.add_argument("--adv_on_logits_alone", action="store_true", default=False)
    parser.add_argument("--num_model_steps", type=int, default=1)
    parser.add_argument("--num_adv_steps", type=int, default=100)
    parser.add_argument("--wrap_model", action="store_true", default=False)
    parser.add_argument("--use_risk_factors", action="store_true", default=False)
    parser.add_argument("--pred_risk_factors", action="store_true", default=False)
    parser.add_argument("--pred_risk_factors_lambda", type=float, default=0.25)
    parser.add_argument("--use_pred_risk_factors_at_test", action="store_true", default=False)
    parser.add_argument("--use_pred_risk_factors_if_unk", action="store_true", default=False)
    parser.add_argument(
        "--risk_factor_keys",
        nargs="*",
        default=[
            "density",
            "binary_family_history",
            "binary_biopsy_benign",
            "binary_biopsy_LCIS",
            "binary_biopsy_atypical_hyperplasia",
            "age",
            "menarche_age",
            "menopause_age",
            "first_pregnancy_age",
            "prior_hist",
            "race",
            "parous",
            "menopausal_status",
            "weight",
            "height",
            "ovarian_cancer",
            "ovarian_cancer_age",
            "ashkenazi",
            "brca",
            "mom_bc_cancer_history",
            "m_aunt_bc_cancer_history",
            "p_aunt_bc_cancer_history",
            "m_grandmother_bc_cancer_history",
            "p_grantmother_bc_cancer_history",
            "sister_bc_cancer_history",
            "mom_oc_cancer_history",
            "m_aunt_oc_cancer_history",
            "p_aunt_oc_cancer_history",
            "m_grandmother_oc_cancer_history",
            "p_grantmother_oc_cancer_history",
            "sister_oc_cancer_history",
            "hrt_type",
            "hrt_duration",
            "hrt_years_ago_stopped",
        ],
    )
    parser.add_argument(
        "--risk_factor_metadata_path",
        type=str,
        default="/home/administrator/Mounts/Isilon/metadata/risk_factors_jul22_2018_mammo_and_mri.json",
    )
    parser.add_argument("--survival_analysis_setup", action="store_true", default=False)
    parser.add_argument("--make_probs_indep", action="store_true", default=False)
    parser.add_argument("--mask_mechanism", default="default")
    parser.add_argument("--eval_survival_on_risk", action="store_true", default=False)
    parser.add_argument("--max_followup", type=int, default=5)
    parser.add_argument("--eval_risk_survival", action="store_true", default=False)
    parser.add_argument("--mask_prob", type=float, default=0)
    parser.add_argument("--pred_missing_mammos", action="store_true", default=False)
    parser.add_argument("--also_pred_given_mammos", action="store_true", default=False)
    parser.add_argument("--pred_missing_mammos_lambda", type=float, default=0.25)
    parser.add_argument("--use_precomputed_hiddens", action="store_true", default=False)
    parser.add_argument("--zero_out_hiddens", action="store_true", default=False)
    parser.add_argument(
        "--use_precomputed_hiddens_in_get_hiddens", action="store_true", default=False
    )
    parser.add_argument(
        "--hiddens_results_path",
        type=str,
        default="/home/administrator/Mounts/Isilon/results/hiddens_from_best_dev_aug_29.results.json",
    )
    parser.add_argument("--use_dev_to_train_model_on_hiddens", action="store_true", default=False)
    parser.add_argument("--turn_off_init_projection", action="store_true", default=False)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--objective", type=str, default="cross_entropy")
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--turn_off_model_reset", action="store_true", default=False)
    parser.add_argument("--tuning_metric", type=str, default="loss")
    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument("--max_batches_per_train_epoch", type=int, default=10000)
    parser.add_argument("--max_batches_per_dev_epoch", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_splits", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--save_dir", type=str, default="snapshot")
    parser.add_argument("--results_path", type=str, default="logs/snapshot")
    parser.add_argument("--prediction_save_path", type=str, default=None)
    parser.add_argument("--no_tuning_on_dev", action="store_true", default=False)
    parser.add_argument("--lr_reduction_interval", type=int, default=1)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--ten_fold_cross_val", action="store_true", default=False)
    parser.add_argument("--ten_fold_cross_val_seed", type=int, default=1)
    parser.add_argument("--ten_fold_test_index", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--snapshot", type=str, default=None)
    parser.add_argument("--state_dict_path", type=str, default=None)
    parser.add_argument("--img_encoder_snapshot", type=str, default=None)
    parser.add_argument("--freeze_image_encoder", action="store_true", default=False)
    parser.add_argument("--transformer_snapshot", type=str, default=None)
    parser.add_argument("--calibrator_snapshot", type=str, default=None)
    parser.add_argument("--patch_snapshot", type=str, default=None)
    parser.add_argument("--pretrained_on_imagenet", action="store_true", default=False)
    parser.add_argument("--pretrained_imagenet_model_name", type=str, default="resnet18")
    parser.add_argument("--make_fc", action="store_true", default=False)
    parser.add_argument("--replace_bn_with_gn", action="store_true", default=False)
    parser.add_argument(
        "--block_layout",
        type=str,
        nargs="+",
        default=["BasicBlock,2", "BasicBlock,2", "BasicBlock,2", "BasicBlock,2"],
    )
    parser.add_argument("--block_widening_factor", type=int, default=1)
    parser.add_argument("--num_groups", type=int, default=1)
    parser.add_argument("--pool_name", type=str, default="GlobalAvgPool")
    parser.add_argument("--deep_risk_factor_pool", action="store_true", default=False)
    parser.add_argument("--replace_snapshot_pool", action="store_true", default=False)
    parser.add_argument("--is_ccds_server", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--data_parallel", action="store_true", default=False)
    parser.add_argument("--model_parallel", action="store_true", default=False)
    parser.add_argument("--plot_losses", action="store_true", default=False)
    parser.add_argument("--cluster_exams", action="store_true", default=False)
    parser.add_argument("--background_size", type=int, nargs="+", default=[1024, 1024])
    parser.add_argument("--noise", action="store_true", default=False)
    parser.add_argument("--noise_var", type=float, default=0.1)
    parser.add_argument("--use_permissive_cohort", action="store_true", default=True)
    parser.add_argument("--mammogram_type", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--ignore_warnings", action="store_true", default=False)

    args = parser.parse_args([])

    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = "cuda" if args.cuda else "cpu"

    args.optimizer_state = None
    args.current_epoch = None
    args.lr = None
    args.epoch_stats = None
    args.step_indx = 1

    return args


# --------------------------------------------------------------------------------------
# Staging build/example-input functions
# --------------------------------------------------------------------------------------


def _write_placeholder_metadata_files():
    """Two minimal, well-formed JSON files satisfying parse_risk_factors()'s
    json.load() calls. Content is never read by this staging module's forward pass
    (see the module docstring above); only file existence + JSON well-formedness
    matters for the real, unmodified parse_risk_factors(args) to return without
    raising."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="mirai_menagerie_"))
    metadata_path = tmp_dir / "metadata.json"
    risk_factor_metadata_path = tmp_dir / "risk_factors.json"
    metadata_path.write_text(json.dumps([]))
    risk_factor_metadata_path.write_text(json.dumps({}))
    return str(metadata_path), str(risk_factor_metadata_path)


class MiraiTracingWrapper(nn.Module):
    """Thin TorchLens-tracing wrapper: MiraiFull.forward(x, risk_factors, batch) takes a
    list-of-tensors (risk_factors) and a dict-of-long-tensors (batch) alongside the image
    stack x. This wrapper exposes a flat tensor-positional-args forward (x plus the
    per-risk-factor-key tensors, one positional arg per key) and reassembles the real
    risk_factors list / batch dict internally from constants (view/time/side sequencing
    is fixed metadata, not learned data, in this staging recipe) before calling the real,
    unmodified MiraiFull.forward."""

    def __init__(self, model: MiraiFull, num_images: int, risk_factor_keys):
        super().__init__()
        self.model = model
        self.risk_factor_keys = list(risk_factor_keys)
        batch = {
            "time_seq": torch.zeros(1, num_images, dtype=torch.long),
            "view_seq": torch.zeros(1, num_images, dtype=torch.long),
            "side_seq": torch.zeros(1, num_images, dtype=torch.long),
        }
        self.register_buffer("time_seq", batch["time_seq"])
        self.register_buffer("view_seq", batch["view_seq"])
        self.register_buffer("side_seq", batch["side_seq"])

    def forward(self, x, *risk_factor_tensors):
        risk_factors = list(risk_factor_tensors)
        batch = {"time_seq": self.time_seq, "view_seq": self.view_seq, "side_seq": self.side_seq}
        logit, transformer_hidden, activ_dict = self.model(
            x, risk_factors=risk_factors, batch=batch
        )
        return logit


def build_mirai():
    torch.manual_seed(0)
    args = parse_args()

    metadata_path, risk_factor_metadata_path = _write_placeholder_metadata_files()
    args.metadata_path = metadata_path
    args.risk_factor_metadata_path = risk_factor_metadata_path

    # Mirror onconet/configs/mirai_trained.json's real deployment overrides, shrunk to
    # tiny sizes for random-init tracing.
    args.model_name = "mirai_full"
    args.multi_image = True
    args.num_images = 2  # real config: 4 (L CC/MLO, R CC/MLO); shrunk for tracing
    args.min_num_images = 2
    args.use_risk_factors = True
    args.pred_risk_factors = True
    args.use_pred_risk_factors_at_test = True
    args.survival_analysis_setup = True
    args.max_followup = 3  # real config: 5
    args.num_chan = 3
    args.img_size = [64, 64]  # real config: [1664, 2048]
    # parse_args() already ran parse_block_layout() on the raw ["BasicBlock,2", ...]
    # strings by the time we get here, so overrides must use the same *parsed* format:
    # a length-4 list where each element is a list of (block_name, num_repeats) tuples.
    args.block_layout = [
        [("BasicBlock", 1)],
        [("BasicBlock", 1)],
        [("BasicBlock", 1)],
        [("BasicBlock", 1)],
    ]  # real default layout: BasicBlock,2 x4; shrunk depth
    args.block_widening_factor = 1
    args.num_classes = 2
    args.transfomer_hidden_dim = 96  # real default: 512; shrunk + kept divisible by num_heads
    args.num_heads = 4  # real default: 8
    args.num_layers = 2  # real default: 3
    args.pretrained_on_imagenet = False
    args.dropout = 0.0

    model = MiraiFull(args)
    return MiraiTracingWrapper(
        model, num_images=args.num_images, risk_factor_keys=args.risk_factor_keys
    )


def example_input_mirai():
    torch.manual_seed(0)
    # MiraiTracingWrapper.forward(x, *risk_factor_tensors):
    #   x: (B, C, N, H, W) stack of N mammography views
    #   risk_factor_tensors: one tensor per key in args.risk_factor_keys, matching
    #   RiskFactorVectorizer's per-key one-hot widths (verbatim transformer definitions
    #   above, 34 keys in the real default risk_factor_keys list). A zero one-hot row
    #   (all-missing) is a valid input for every transformer's declared output width.
    # Widths are derived from the real, unmodified RiskFactorVectorizer rather than
    # hand-transcribed, so this can never silently drift out of sync with the 34-key
    # default risk_factor_keys list above.
    B, C, N, H, W = 1, 3, 2, 64, 64
    x = torch.randn(B, C, N, H, W)

    probe_args = parse_args()
    metadata_path, risk_factor_metadata_path = _write_placeholder_metadata_files()
    probe_args.metadata_path = metadata_path
    probe_args.risk_factor_metadata_path = risk_factor_metadata_path
    vectorizer = RiskFactorVectorizer(probe_args)
    key_widths = [vectorizer.risk_factor_key_to_num_class[k] for k in probe_args.risk_factor_keys]
    risk_factors = tuple(torch.zeros(B, w) for w in key_widths)

    return (x, *risk_factors)


MENAGERIE_ENTRIES = [
    ("Mirai", build_mirai, example_input_mirai, 2021, MENAGERIE_ZOO),
]
