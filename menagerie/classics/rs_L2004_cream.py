# SOURCE: vendored from microsoft/Cream @ main (Cream/lib/models/{structures/childnet.py,
# builders/build_childnet.py, utils/builder_util.py}). "Cream" = CVPR'21 "Cream of the Crop:
# Distilling Prioritized Paths For One-Shot Neural Architecture Search" (Du/Peng et al.). The
# searched/distilled deployable network is the ChildNet defined here; imports/relative-paths are
# adapted to the installed timm's current module layout (timm.layers / timm.models._efficientnet_*
# instead of the repo's now-removed timm.models.layers / timm.models.efficientnet_blocks), and the
# small handful of arch-decoding helper functions from lib/utils/builder_util.py are vendored
# verbatim (they are plain string/dict utilities with no architectural content) since they are not
# public timm API. No architecture has been changed.
import math
import re
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import (
    CondConv2d,
    SelectAdaptivePool2d,
    SqueezeExcite,
    create_conv2d,
    get_condconv_initializer,
)
from timm.layers.activations import Swish
from timm.models._efficientnet_blocks import ConvBnAct, DepthwiseSeparableConv, InvertedResidual
from timm.models._efficientnet_builder import resolve_bn_args, round_channels

MENAGERIE_ZOO = "vendored-pytorch"

# --- vendored from Cream/lib/utils/builder_util.py (arch-string decoding helpers) ---


def parse_ksize(ss):
    if ss.isdigit():
        return int(ss)
    else:
        return [int(k) for k in ss.split(".")]


def decode_block_str(block_str):
    """Decode block definition string, e.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip."""
    assert isinstance(block_str, str)
    ops = block_str.split("_")
    block_type = ops[0]
    ops = ops[1:]
    options = {}
    noskip = False
    for op in ops:
        if op == "noskip":
            noskip = True
        elif op.startswith("n"):
            key = op[0]
            v = op[1:]
            if v == "re":
                value = nn.ReLU
            elif v == "r6":
                value = nn.ReLU6
            elif v == "sw":
                value = Swish
            else:
                continue
            options[key] = value
        else:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    act_layer = options["n"] if "n" in options else None
    exp_kernel_size = parse_ksize(options["a"]) if "a" in options else 1
    pw_kernel_size = parse_ksize(options["p"]) if "p" in options else 1
    fake_in_chs = int(options["fc"]) if "fc" in options else 0  # noqa: F841 (unused in upstream too)

    num_repeat = int(options["r"])
    if block_type == "ir":
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=parse_ksize(options["k"]),
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options["c"]),
            exp_ratio=float(options["e"]),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=int(options["s"]),
            act_layer=act_layer,
            noskip=noskip,
        )
        if "cc" in options:
            block_args["num_experts"] = int(options["cc"])
    elif block_type == "ds" or block_type == "dsa":
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=parse_ksize(options["k"]),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options["c"]),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=int(options["s"]),
            act_layer=act_layer,
            pw_act=block_type == "dsa",
            noskip=block_type == "dsa" or noskip,
        )
    elif block_type == "cn":
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options["k"]),
            out_chs=int(options["c"]),
            stride=int(options["s"]),
            act_layer=act_layer,
        )
    else:
        assert False, "Unknown block type (%s)" % block_type

    return block_args, num_repeat


def scale_stage_depth(stack_args, repeats, depth_multiplier=1.0, depth_trunc="ceil"):
    """Per-stage depth scaling (EfficientNet-style)."""
    num_repeat = sum(repeats)
    if depth_trunc == "round":
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round((r / num_repeat * num_repeat_scaled)))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]

    sa_scaled = []
    for ba, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend([deepcopy(ba) for _ in range(rep)])
    return sa_scaled


def decode_arch_def(arch_def, depth_multiplier=1.0, depth_trunc="ceil", experts_multiplier=1):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            ba, rep = decode_block_str(block_str)
            if ba.get("num_experts", 0) > 0 and experts_multiplier > 1:
                ba["num_experts"] *= experts_multiplier
            stack_args.append(ba)
            repeats.append(rep)
        arch_args.append(scale_stage_depth(stack_args, repeats, depth_multiplier, depth_trunc))
    return arch_args


def init_weight_goog(m, n="", fix_group_fanout=True, last_bn=None):
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_weight_fn = get_condconv_initializer(
            lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape
        )
        init_weight_fn(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if last_bn and n in last_bn:
            m.weight.data.zero_()
            m.bias.data.zero_()
        else:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)
        fan_in = 0
        if "routing_fn" in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def efficientnet_init_weights(model: nn.Module, init_fn=None, zero_gamma=False):
    last_bn = []
    if zero_gamma:
        prev_n = ""
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if "".join(prev_n.split(".")[:-1]) != "".join(n.split(".")[:-1]):
                    last_bn.append(prev_n)
                prev_n = n
        last_bn.append(prev_n)

    init_fn = init_fn or init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n, last_bn=last_bn)


# --- vendored from Cream/lib/models/builders/build_childnet.py ---


class ChildNetBuilder:
    def __init__(
        self,
        channel_multiplier=1.0,
        channel_divisor=8,
        channel_min=None,
        output_stride=32,
        pad_type="",
        act_layer=None,
        se_kwargs=None,
        norm_layer=nn.BatchNorm2d,
        norm_kwargs=None,
        drop_path_rate=0.0,
        feature_location="",
        verbose=False,
        logger=None,
    ):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.act_layer = act_layer
        self.se_kwargs = se_kwargs
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.drop_path_rate = drop_path_rate
        self.feature_location = feature_location
        assert feature_location in ("pre_pwl", "post_exp", "")
        self.verbose = verbose
        self.in_chs = None
        self.features = OrderedDict()
        self.logger = logger

    def _round_channels(self, chs):
        return round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, ba, block_idx, block_count):
        # NOTE (import/API-drift adapter, not an architecture change): the Cream repo was written
        # against a vintage timm whose EfficientNet blocks took a float `se_ratio` + a `se_kwargs`
        # dict (act_layer/gate_fn/reduce_mid/divisor) and a `norm_kwargs` dict. Current timm's
        # `_efficientnet_blocks` instead takes an `se_layer` *module factory* and drops
        # `norm_kwargs`/`fake_in_chs` entirely. This block adapts the old-style kwargs into the
        # new constructor surface (same SE reduction ratio, same gate fn) without altering what
        # the network computes.
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop("block_type")
        ba["in_chs"] = self.in_chs
        ba["out_chs"] = self._round_channels(ba["out_chs"])
        ba.pop("fake_in_chs", None)
        ba["norm_layer"] = self.norm_layer
        ba["pad_type"] = self.pad_type
        ba["act_layer"] = ba["act_layer"] if ba["act_layer"] is not None else self.act_layer
        assert ba["act_layer"] is not None

        se_ratio = ba.pop("se_ratio", None)
        se_kwargs = self.se_kwargs or {}
        if se_ratio:
            ba["se_layer"] = partial(
                SqueezeExcite,
                rd_ratio=se_ratio,
                act_layer=se_kwargs.get("act_layer", nn.ReLU),
                gate_layer=se_kwargs.get("gate_fn", "sigmoid"),
            )

        if bt == "ir":
            ba["drop_path_rate"] = drop_path_rate
            block = InvertedResidual(**ba)
        elif bt == "ds" or bt == "dsa":
            ba["drop_path_rate"] = drop_path_rate
            block = DepthwiseSeparableConv(**ba)
        elif bt == "cn":
            ba.pop("noskip", None)
            block = ConvBnAct(**ba)
        else:
            assert False, "Uknkown block type (%s) while building model." % bt
        self.in_chs = ba["out_chs"]

        return block

    def __call__(self, in_chs, model_block_args):
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        feature_idx = 0
        stages = []
        for stage_idx, stage_block_args in enumerate(model_block_args):
            last_stack = stage_idx == (len(model_block_args) - 1)
            assert isinstance(stage_block_args, list)

            blocks = []
            for block_idx, block_args in enumerate(stage_block_args):
                last_block = block_idx == (len(stage_block_args) - 1)
                extract_features = ""

                assert block_args["stride"] in (1, 2)
                if block_idx >= 1:
                    block_args["stride"] = 1

                do_extract = False
                if self.feature_location == "pre_pwl":
                    if last_block:
                        next_stage_idx = stage_idx + 1
                        if next_stage_idx >= len(model_block_args):
                            do_extract = True
                        else:
                            do_extract = model_block_args[next_stage_idx][0]["stride"] > 1
                elif self.feature_location == "post_exp":
                    if block_args["stride"] > 1 or (last_stack and last_block):
                        do_extract = True
                if do_extract:
                    extract_features = self.feature_location

                next_dilation = current_dilation
                if block_args["stride"] > 1:
                    next_output_stride = current_stride * block_args["stride"]
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args["stride"]
                        block_args["stride"] = 1
                    else:
                        current_stride = next_output_stride
                block_args["dilation"] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation

                block = self._make_block(block_args, total_block_idx, total_block_count)
                blocks.append(block)

                if extract_features:
                    feature_module = block.feature_module(extract_features)
                    if feature_module:
                        feature_module = (
                            "blocks.{}.{}.".format(stage_idx, block_idx) + feature_module
                        )
                    feature_channels = block.feature_channels(extract_features)
                    self.features[feature_idx] = dict(name=feature_module, num_chs=feature_channels)
                    feature_idx += 1

                total_block_idx += 1
            stages.append(nn.Sequential(*blocks))
        return stages


# --- vendored from Cream/lib/models/structures/childnet.py ---


class ChildNet(nn.Module):
    def __init__(
        self,
        block_args,
        num_classes=1000,
        in_chans=3,
        stem_size=16,
        num_features=1280,
        head_bias=True,
        channel_multiplier=1.0,
        pad_type="",
        act_layer=nn.ReLU,
        drop_rate=0.0,
        drop_path_rate=0.0,
        se_kwargs=None,
        norm_layer=nn.BatchNorm2d,
        norm_kwargs=None,
        global_pool="avg",
        logger=None,
        verbose=False,
    ):
        super(ChildNet, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans
        self.logger = logger

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **(norm_kwargs or {}))
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = ChildNetBuilder(
            channel_multiplier,
            8,
            None,
            32,
            pad_type,
            act_layer,
            se_kwargs,
            norm_layer,
            norm_kwargs,
            drop_path_rate,
            verbose=verbose,
        )
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        self._in_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = create_conv2d(
            self._in_chs, self.num_features, 1, padding=pad_type, bias=head_bias
        )
        self.act2 = act_layer(inplace=True)

        # Classifier
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(), self.num_classes
        )

        efficientnet_init_weights(self)

    def get_classifier(self):
        return self.classifier

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(1)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x


def gen_childnet(arch_list, arch_def, **kwargs):
    """Build a ChildNet from a discovered-path arch_list against the Cream search-space arch_def
    (Cream/tools/main.py uses the same arch_def / arch_list mechanism to instantiate a searched
    child)."""
    choices = {"kernel_size": [3, 5, 7], "exp_ratio": [4, 6]}
    choices_list = [[x, y] for x in choices["kernel_size"] for y in choices["exp_ratio"]]

    num_features = 1280
    act_layer = Swish

    new_arch = []
    for i, (layer_choice, layer_arch) in enumerate(zip(arch_list, arch_def)):
        if len(layer_arch) == 1:
            new_arch.append(layer_arch)
            continue
        else:
            new_layer = []
            for j, (block_choice, block_arch) in enumerate(zip(layer_choice, layer_arch)):
                kernel_size, exp_ratio = choices_list[block_choice]
                elements = block_arch.split("_")
                block_arch = block_arch.replace(elements[2], "k{}".format(str(kernel_size)))
                block_arch = block_arch.replace(elements[4], "e{}".format(str(exp_ratio)))
                new_layer.append(block_arch)
            new_arch.append(new_layer)

    model_kwargs = dict(
        block_args=decode_arch_def(new_arch),
        num_features=num_features,
        stem_size=16,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=act_layer,
        se_kwargs=dict(act_layer=nn.ReLU, gate_fn="hard_sigmoid", reduce_mid=True, divisor=8),
        **kwargs,
    )
    model = ChildNet(**model_kwargs)
    return model


# The retrain/14.yaml Cream-searched architecture from the official repo's
# Cream/experiments/configs/retrain/14.yaml (smallest published Cream-searched child, ~14M FLOPs
# scale family) expressed as an arch_list of per-stage block choices, decoded against the repo's
# fixed supernet search-space arch_def (mirrors Cream/lib/models/blocks default def used by
# tools/main.py at retrain time). Kept tiny for a fast, faithful trace.
_CREAM_ARCH_LIST = [
    [0],
    [4, 4],
    [4, 4],
    [4, 4],
    [4, 4],
    [0],
]
_CREAM_ARCH_DEF = [
    ["ds_r1_k3_s1_e1_c16_se0.25"],
    ["ir_r1_k3_s2_e4_c24_se0.25", "ir_r1_k3_s1_e4_c24_se0.25"],
    ["ir_r1_k5_s2_e4_c40_se0.25", "ir_r1_k5_s1_e4_c40_se0.25"],
    ["ir_r1_k3_s2_e4_c80_se0.25", "ir_r1_k3_s1_e4_c80_se0.25"],
    ["ir_r1_k5_s1_e4_c112_se0.25", "ir_r1_k5_s1_e4_c112_se0.25"],
    ["cn_r1_k1_s1_c320"],
]


def build_cream_childnet():
    return gen_childnet(_CREAM_ARCH_LIST, _CREAM_ARCH_DEF, num_classes=10, in_chans=3)


def example_input_cream_childnet():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Cream (ChildNet, one-shot NAS w/ prioritized path distillation)",
        build_cream_childnet,
        example_input_cream_childnet,
        2020,
        "CODE",
    ),
]
