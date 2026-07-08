# SOURCE: vendored from nightsnack/YOLObile @ master
# https://raw.githubusercontent.com/nightsnack/YOLObile/master/models.py
# https://raw.githubusercontent.com/nightsnack/YOLObile/master/utils/layers.py
# https://raw.githubusercontent.com/nightsnack/YOLObile/master/utils/parse_config.py
# https://raw.githubusercontent.com/nightsnack/YOLObile/master/cfg/csdarknet53s-panet-spp.cfg
#
# YOLObile (Cai, Li, Geng, Wu, Wang, AAAI 2021): "YOLObile: Real-Time Object Detection
# on Mobile Devices via Compression-Compilation Co-Design". A CSPDarknet53s backbone
# with a PANet neck and an SPP block, trained with the paper's block-punched pruning
# co-designed for mobile-compiler kernel scheduling (the pruning itself is a training-time
# procedure applied to this dense architecture; the architecture graph captured here is
# the dense CSPDarknet53s-PANet-SPP detector the pruning acts on -- identical to what
# `models.py::Darknet` builds from the repo's own
# `cfg/csdarknet53s-panet-spp.cfg`).
#
# The real repo builds the network from a Darknet-style `.cfg` text file parsed by
# `utils/parse_config.py::parse_model_cfg` and instantiated layer-by-layer by
# `models.py::create_modules`, exactly like the ultralytics/yolov3 lineage. All classes/
# functions below (`parse_model_cfg`, `FeatureConcat`, `RouteGroup`, `WeightedFeatureFusion`,
# `MixConv2d`, `Swish`/`Mish` activations, `YOLOLayer`, `create_modules`, `Darknet`,
# `get_yolo_layers`) are copied verbatim (only whitespace-preserving, no architectural
# edits) from `models.py` and `utils/layers.py`/`utils/parse_config.py`. The only
# omissions are training/export-only code paths that are dead at construction+forward
# time for this staging module: `load_darknet_weights` (weight-file I/O), `Darknet.fuse`
# (post-hoc Conv+BN fusion for deployment), augmented-inference (`augment=True` TTA path),
# and `ONNX_EXPORT` branches -- none of those affect the graph actually traced by a plain
# forward pass. `torch_utils.model_info`'s FLOPs printout (which imports the optional
# `thop` package) is reproduced as a minimal try/except-guarded stand-in so the real
# `Darknet.__init__` -> `self.info()` call path still runs without requiring `thop`.
#
# The real per-model architecture graph is defined by the repo's own
# `cfg/csdarknet53s-panet-spp.cfg` (157 layers: CSPDarknet53s backbone + PANet neck +
# SPP block + 3 YOLO detection heads over COCO's 80 classes), embedded verbatim below as
# `YOLOBILE_CFG_TEXT` so this module is self-contained (no network fetch, no repo checkout
# needed at trace time).
"""YOLObile (AAAI 2021): CSPDarknet53s backbone + PANet neck + SPP, vendored Darknet-cfg
model builder (`models.py`/`utils/layers.py`/`utils/parse_config.py`) driven by the repo's
real `cfg/csdarknet53s-panet-spp.cfg` architecture definition."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

ONNX_EXPORT = False


# ---------------------------------------------------------------------------------------
# utils/parse_config.py (verbatim, base-libs only: os + numpy)
# ---------------------------------------------------------------------------------------


def parse_model_cfg(path_or_lines):
    # Parse the yolo *.cfg file and return module definitions.
    # NOTE: original signature takes a filesystem `path`; this staging module instead
    # accepts the cfg text already split into lines (see `_parse_cfg_text` below) so the
    # embedded YOLOBILE_CFG_TEXT constant can drive it without touching disk. The parsing
    # logic itself (the loop body) is verbatim from the real `parse_model_cfg`.
    lines = path_or_lines
    lines = [x for x in lines if x and not x.startswith("#")]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for line in lines:
        if line.startswith("["):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]["type"] = line[1:-1].rstrip()
            if mdefs[-1]["type"] == "convolutional":
                mdefs[-1]["batch_normalize"] = (
                    0  # pre-populate with zeros (may be overwritten later)
                )
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == "anchors":  # return nparray
                mdefs[-1][key] = np.array([float(x) for x in val.split(",")]).reshape(
                    (-1, 2)
                )  # np anchors
            elif (key in ["from", "layers", "mask"]) or (
                key == "size" and "," in val
            ):  # return array
                mdefs[-1][key] = [int(x) for x in val.split(",")]
            else:
                val = val.strip()
                if val.isnumeric():  # return int or float
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string

    # Check all fields are supported
    supported = [
        "type",
        "batch_normalize",
        "filters",
        "size",
        "stride",
        "pad",
        "activation",
        "layers",
        "groups",
        "group_id",
        "resize",
        "from",
        "mask",
        "anchors",
        "classes",
        "num",
        "jitter",
        "ignore_thresh",
        "truth_thresh",
        "random",
        "stride_x",
        "stride_y",
        "weights_type",
        "weights_normalization",
        "scale_x_y",
        "beta_nms",
        "nms_kind",
        "iou_loss",
        "iou_normalizer",
        "cls_normalizer",
        "iou_thresh",
        "probability",
    ]

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(u), (
        "Unsupported fields %s. See https://github.com/ultralytics/yolov3/issues/631" % (u,)
    )

    return mdefs


def _parse_cfg_text(cfg_text):
    return parse_model_cfg(cfg_text.split("\n"))


# ---------------------------------------------------------------------------------------
# utils/layers.py (verbatim; the module only uses math/np/nn/torch/F from the real
# `from utils.utils import *` wildcard import, so those are imported directly above)
# ---------------------------------------------------------------------------------------


def make_divisible(v, divisor):
    # Function ensures all layers have a channel number that is divisible by 8
    return math.ceil(v / divisor) * divisor


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    def forward(self, x):
        return x.view(x.size(0), -1)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class RouteGroup(nn.Module):
    def __init__(self, layers, groups, group_id):
        super(RouteGroup, self).__init__()
        self.layers = layers
        self.multi = len(layers) > 1
        self.groups = groups
        self.group_id = group_id

    def forward(self, x, outputs):
        if self.multi:
            outs = []
            for layer in self.layers:
                out = torch.chunk(outputs[layer], self.groups, dim=1)
                outs.append(out[self.group_id])
            return torch.cat(outs, dim=1)
        else:
            out = torch.chunk(outputs[self.layers[0]], self.groups, dim=1)
            return out[self.group_id]


class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return (
            torch.cat([outputs[i] for i in self.layers], 1)
            if self.multiple
            else outputs[self.layers[0]]
        )


class WeightedFeatureFusion(
    nn.Module
):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = (
                outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]
            )  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class MixConv2d(
    nn.Module
):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(
        self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method="equal_params"
    ):
        super(MixConv2d, self).__init__()

        groups = len(k)
        if method == "equal_ch":  # equal channels per group
            i = torch.linspace(0, groups - 1e-6, out_ch).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = (
                np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)
            )  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=ch[g],
                    kernel_size=k[g],
                    stride=stride,
                    padding=k[g] // 2,  # 'same' pad
                    dilation=dilation,
                    bias=bias,
                )
                for g in range(groups)
            ]
        )

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class HardSwish(nn.Module):  # https://arxiv.org/pdf/1905.02244.pdf
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0.0, 6.0, True) / 6.0


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * F.softplus(x).tanh()


# ---------------------------------------------------------------------------------------
# models.py (verbatim: create_modules, YOLOLayer, Darknet, get_yolo_layers).
# `load_darknet_weights`, `Darknet.fuse`, and the ONNX_EXPORT/augment branches (all dead
# at plain-forward construction time) are omitted per the header note above.
# ---------------------------------------------------------------------------------------


def create_modules(module_defs, img_size, cfg):
    # Constructs module list of layer blocks from module configuration in module_defs

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1
    upsample_index = 0

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef["type"] == "convolutional":
            bn = mdef["batch_normalize"]
            filters = mdef["filters"]
            k = mdef["size"]  # kernel size
            stride = mdef["stride"] if "stride" in mdef else (mdef["stride_y"], mdef["stride_x"])
            if isinstance(k, int):  # single-size conv
                modules.add_module(
                    "Conv2d",
                    nn.Conv2d(
                        in_channels=output_filters[-1],
                        out_channels=filters,
                        kernel_size=k,
                        stride=stride,
                        padding=k // 2 if mdef["pad"] else 0,
                        groups=mdef["groups"] if "groups" in mdef else 1,
                        bias=not bn,
                    ),
                )
            else:  # multiple-size conv
                modules.add_module(
                    "MixConv2d",
                    MixConv2d(
                        in_ch=output_filters[-1], out_ch=filters, k=k, stride=stride, bias=not bn
                    ),
                )

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters, momentum=0.03, eps=1e-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            elif mdef["activation"] == "swish":
                modules.add_module("activation", Swish())
            elif mdef["activation"] == "mish":
                modules.add_module("activation", Mish())

        elif mdef["type"] == "BatchNorm2d":
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1e-4)
            if i == 0 and filters == 3:  # normalize RGB image
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef["type"] == "maxpool":
            k = mdef["size"]  # kernel size
            stride = mdef["stride"]
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module("ZeroPad2d", nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module("MaxPool2d", maxpool)
            else:
                modules = maxpool

        elif mdef["type"] == "upsample":
            if ONNX_EXPORT:
                g = (upsample_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
                upsample_index = upsample_index + 1
            else:
                modules = nn.Upsample(scale_factor=mdef["stride"])

        elif mdef["type"] == "route":  # nn.Sequential() placeholder for 'route' layer
            layers = mdef["layers"]
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])  # noqa: E741
            routs.extend([i + l if l < 0 else l for l in layers])  # noqa: E741
            modules = FeatureConcat(layers=layers)
            if "groups" in mdef:
                groups = mdef["groups"]
                group_id = mdef["group_id"]
                modules = RouteGroup(layers=layers, groups=groups, group_id=group_id)
                filters //= groups
        elif mdef["type"] == "shortcut":  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef["from"]
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])  # noqa: E741
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        elif mdef["type"] == "reorg3d":  # yolov3-spp-pan-scale
            pass

        elif mdef["type"] == "yolo":
            yolo_index += 1
            stride = [32, 16, 8]  # P5, P4, P3 strides
            if any(x in cfg for x in ["panet", "yolov4", "cd53"]):  # stride order reversed
                stride = list(reversed(stride))
            layers = mdef["from"] if "from" in mdef else []
            modules = YOLOLayer(
                anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                nc=mdef["classes"],  # number of classes
                img_size=img_size,  # (416, 416)
                yolo_index=yolo_index,  # 0, 1, 2...
                layers=layers,  # output layers
                stride=stride[yolo_index],
            )

            # Initialize preceding Conv2d() bias
            try:
                j = layers[yolo_index] if "from" in mdef else -1
                if module_list[j].__class__.__name__ == "Dropout":
                    j -= 1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[: modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                bias[:, 4] += -4.5  # obj
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls
                module_list[j][0].bias = torch.nn.Parameter(
                    bias_, requires_grad=bias_.requires_grad
                )
            except Exception:
                print("WARNING: smart bias initialization failure.")

        elif mdef["type"] == "dropout":
            perc = float(mdef["probability"])
            modules = nn.Dropout(p=perc)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))

    def create_grids(self, ng=(13, 13), device="cpu"):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid(
                [torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)]
            )
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        ASFF = False
        if ASFF:
            i, n = self.index, self.nl
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)

            p = out[self.layers[i]][:, :-n] * w[:, i : i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j : j + 1] * F.interpolate(
                        out[self.layers[j]][:, :-n],
                        size=[ny, nx],
                        mode="bilinear",
                        align_corners=False,
                    )

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        p = (
            p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()
        )  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            m = self.na * self.nx * self.ny
            ng = 1.0 / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = (
                torch.sigmoid(p[:, 4:5])
                if self.nc == 1
                else torch.sigmoid(p[:, 5 : self.no]) * torch.sigmoid(p[:, 4:5])
            )  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]).to(p.device) + self.grid.to(p.device)  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]).to(p.device) * self.anchor_wh.to(p.device)  # wh
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


def _model_info(model, verbose=False):
    # Minimal stand-in for utils/torch_utils.py::model_info (only the FLOPs-via-`thop`
    # optional path, guarded exactly like the real function). Real `Darknet.__init__`
    # calls `self.info(verbose)`, which calls this; behavior (print + return (n_p, macs))
    # is preserved so the real construction code path runs unmodified.
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    macs = None
    try:  # FLOPS
        from thop import profile

        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 320, 320),), verbose=False)
        fs = ", %.1f GFLOPS" % (macs / 1e9 * 2)
    except Exception:
        fs = ""
    print(
        "Model Summary: %g layers, %g parameters, %g gradients%s"
        % (len(list(model.parameters())), n_p, n_g, fs)
    )
    return n_p, macs


class Darknet(nn.Module):
    # YOLOv3-lineage object detection model (YOLObile's CSPDarknet53s-PANet-SPP config)

    def __init__(self, cfg_text, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()

        self.module_defs = _parse_cfg_text(cfg_text)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg_text)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header
        self.version = np.array([0, 2, 5], dtype=np.int32)
        self.seen = np.array([0], dtype=np.int64)
        self.info(verbose) if not ONNX_EXPORT else None

    def forward(self, x, augment=False, verbose=False):
        return self.forward_once(x)

    def forward_once(self, x, augment=False, verbose=False):
        yolo_out, out = [], []
        if verbose:
            print("0", x.shape)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat", "RouteGroup"]:  # sum, concat
                x = module(x, out)
            elif name == "YOLOLayer":
                yolo_out.append(module(x, out))
            else:  # run module directly
                x = module(x)

            out.append(x if self.routs[i] else [])

        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            return x, p

    def info(self, verbose=False):
        _model_info(self, verbose)


def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == "YOLOLayer"]


# ---------------------------------------------------------------------------------------
# cfg/csdarknet53s-panet-spp.cfg (embedded verbatim -- the real YOLObile architecture
# definition: CSPDarknet53s backbone + PANet neck + SPP block + 3 YOLO heads, COCO 80cls)
# ---------------------------------------------------------------------------------------

YOLOBILE_CFG_TEXT = """[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch=64
subdivisions=16
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500500
policy=steps
steps=400000,450000
scales=.1,.1

#23:104x104 54:52x52 85:26x26 104:13x13 for 416



[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

# Downsample

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

#[convolutional]
#batch_normalize=1
#filters=64
#size=1
#stride=1
#pad=1
#activation=leaky

#[route]
#layers = -2

#[convolutional]
#batch_normalize=1
#filters=64
#size=1
#stride=1
#pad=1
#activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

#[convolutional]
#batch_normalize=1
#filters=64
#size=1
#stride=1
#pad=1
#activation=leaky

#[route]
#layers = -1,-7

#[convolutional]
#batch_normalize=1
#filters=64
#size=1
#stride=1
#pad=1
#activation=leaky

# Downsample

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -1,-10

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

# Downsample

[convolutional]
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -1,-28

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

# Downsample

[convolutional]
batch_normalize=1
filters=512
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -1,-28

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

# Downsample

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -2

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -1,-16

[convolutional]
batch_normalize=1
filters=1024
size=1
stride=1
pad=1
activation=leaky

##########################

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

### SPP ###
[maxpool]
stride=1
size=5

[route]
layers=-2

[maxpool]
stride=1
size=9

[route]
layers=-4

[maxpool]
stride=1
size=13

[route]
layers=-1,-3,-5,-6
### End SPP ###

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = 79

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -1, -3

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = 48

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -1, -3

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

##########################

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[route]
layers = -4

[convolutional]
batch_normalize=1
size=3
stride=2
pad=1
filters=256
activation=leaky

[route]
layers = -1, -16

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[route]
layers = -4

[convolutional]
batch_normalize=1
size=3
stride=2
pad=1
filters=512
activation=leaky

[route]
layers = -1, -37

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
"""


def build_yolobile():
    torch.manual_seed(0)
    model = Darknet(YOLOBILE_CFG_TEXT, img_size=(256, 256), verbose=False)
    model.eval()
    return model


def example_input_yolobile():
    # 256x256 (multiple of the network's max stride 32) keeps the trace light while using
    # the real, unmodified cfg architecture (img_size is a construction parameter of
    # Darknet, not a cfg edit -- the paper's own default is 416x416).
    torch.manual_seed(0)
    return (torch.randn(1, 3, 256, 256),)


MENAGERIE_ENTRIES = [
    ("YOLObile", "build_yolobile", "example_input_yolobile", 2021, "vendored"),
]
