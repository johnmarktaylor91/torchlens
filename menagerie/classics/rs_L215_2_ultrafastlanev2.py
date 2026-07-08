# SOURCE: vendored from cfzd/Ultra-Fast-Lane-Detection-v2 @ master
#
# https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2
# https://raw.githubusercontent.com/cfzd/Ultra-Fast-Lane-Detection-v2/master/model/model_culane.py
# https://raw.githubusercontent.com/cfzd/Ultra-Fast-Lane-Detection-v2/master/model/backbone.py
# https://raw.githubusercontent.com/cfzd/Ultra-Fast-Lane-Detection-v2/master/model/seg_model.py
# https://raw.githubusercontent.com/cfzd/Ultra-Fast-Lane-Detection-v2/master/utils/common.py
#
# Ultra-Fast-Lane-Detection-v2 (Qin, Zhang, Chen, Chen, Feng, Wang. TPAMI 2022,
# "Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification").
# Row-and-column anchor-driven ordinal-classification lane detector. This vendors the real
# `parsingNet` class from `model/model_culane.py` (the canonical variant -- `model_tusimple.py`
# simply re-imports it, and `model_curvelanes.py` is a near-identical earlier-arg-list
# variant), the real `resnet`/`vgg16bn` backbone-wrapper classes from `model/backbone.py`
# (thin wrappers around the real `torchvision.models.resnet18/34/50/101/152` and
# `torchvision.models.vgg16_bn`), the real `SegHead`/`conv_bn_relu` classes from
# `model/seg_model.py` (the auxiliary segmentation branch, `use_aux=True`), and the real
# `initialize_weights`/`real_init_weights` functions from `utils/common.py` -- copied verbatim
# (only import-path fixes: relative imports `from model.backbone import resnet` etc. become
# local references since everything is in one file; the `'34fca'` backbone option, which loads
# `torch.hub.load('cfzd/FcaNet', ...)`, a network fetch, is kept as dead code for fidelity but
# never selected by `build_ultrafastlanev2`).
#
# Pure `torch`/`torchvision` only -- no custom CUDA ops, no OpenMMLab registry dependency,
# so this is a straight vendor (rung 2), not a port.

import torch


# ---------------------------------------------------------------------------
# utils/common.py::initialize_weights / real_init_weights
# ---------------------------------------------------------------------------


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print("unkonwn module", m)


# ---------------------------------------------------------------------------
# model/backbone.py
# ---------------------------------------------------------------------------


class vgg16bn(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(vgg16bn, self).__init__()
        import torchvision

        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33] + model[34:43]
        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained=False):
        super(resnet, self).__init__()
        import torchvision

        if layers == "18":
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == "34":
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == "50":
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == "101":
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == "152":
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == "50next":
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == "101next":
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == "50wide":
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == "101wide":
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        elif layers == "34fca":
            # NOTE (vendoring deviation): upstream loads a third-party hub model here
            # (network fetch); never selected by build_ultrafastlanev2, kept for fidelity.
            model = torch.hub.load("cfzd/FcaNet", "fca34", pretrained=True)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


# ---------------------------------------------------------------------------
# model/seg_model.py
# ---------------------------------------------------------------------------


class conv_bn_relu(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False
    ):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SegHead(torch.nn.Module):
    def __init__(self, backbone, num_lanes):
        super(SegHead, self).__init__()

        self.aux_header2 = torch.nn.Sequential(
            conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1)
            if backbone in ["34", "18"]
            else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128, 128, 3, padding=1),
            conv_bn_relu(128, 128, 3, padding=1),
            conv_bn_relu(128, 128, 3, padding=1),
        )
        self.aux_header3 = torch.nn.Sequential(
            conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1)
            if backbone in ["34", "18"]
            else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128, 128, 3, padding=1),
            conv_bn_relu(128, 128, 3, padding=1),
        )
        self.aux_header4 = torch.nn.Sequential(
            conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1)
            if backbone in ["34", "18"]
            else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128, 128, 3, padding=1),
        )
        self.aux_combine = torch.nn.Sequential(
            conv_bn_relu(384, 256, 3, padding=2, dilation=2),
            conv_bn_relu(256, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=4, dilation=4),
            torch.nn.Conv2d(128, num_lanes + 1, 1),
            # output : n, num_of_lanes+1, h, w
        )

        initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

    def forward(self, x2, x3, fea):
        x2 = self.aux_header2(x2)
        x3 = self.aux_header3(x3)
        x3 = torch.nn.functional.interpolate(x3, scale_factor=2, mode="bilinear")
        x4 = self.aux_header4(fea)
        x4 = torch.nn.functional.interpolate(x4, scale_factor=4, mode="bilinear")
        aux_seg = torch.cat([x2, x3, x4], dim=1)
        aux_seg = self.aux_combine(aux_seg)
        return aux_seg


# ---------------------------------------------------------------------------
# model/model_culane.py::parsingNet
# ---------------------------------------------------------------------------


class parsingNet(torch.nn.Module):
    def __init__(
        self,
        pretrained=True,
        backbone="50",
        num_grid_row=None,
        num_cls_row=None,
        num_grid_col=None,
        num_cls_col=None,
        num_lane_on_row=None,
        num_lane_on_col=None,
        use_aux=False,
        input_height=None,
        input_width=None,
        fc_norm=False,
    ):
        super(parsingNet, self).__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.use_aux = use_aux
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        mlp_mid_dim = 2048
        self.input_dim = input_height // 32 * input_width // 32 * 8

        self.model = resnet(backbone, pretrained=pretrained)

        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_mid_dim, self.total_dim),
        )
        self.pool = (
            torch.nn.Conv2d(512, 8, 1)
            if backbone in ["34", "18", "34fca"]
            else torch.nn.Conv2d(2048, 8, 1)
        )
        if self.use_aux:
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)
        initialize_weights(self.cls)

    def forward(self, x):
        x2, x3, fea = self.model(x)
        if self.use_aux:
            seg_out = self.seg_head(x2, x3, fea)
        fea = self.pool(fea)

        fea = fea.view(-1, self.input_dim)
        out = self.cls(fea)

        pred_dict = {
            "loc_row": out[:, : self.dim1].view(
                -1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row
            ),
            "loc_col": out[:, self.dim1 : self.dim1 + self.dim2].view(
                -1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col
            ),
            "exist_row": out[:, self.dim1 + self.dim2 : self.dim1 + self.dim2 + self.dim3].view(
                -1, 2, self.num_cls_row, self.num_lane_on_row
            ),
            "exist_col": out[:, -self.dim4 :].view(-1, 2, self.num_cls_col, self.num_lane_on_col),
        }
        if self.use_aux:
            pred_dict["seg_out"] = seg_out

        return pred_dict


# ---------------------------------------------------------------------------
# Tiny-scale build
# ---------------------------------------------------------------------------


def build_ultrafastlanev2():
    return parsingNet(
        pretrained=False,
        backbone="18",
        num_grid_row=4,
        num_cls_row=3,
        num_grid_col=4,
        num_cls_col=3,
        num_lane_on_row=2,
        num_lane_on_col=2,
        use_aux=True,
        input_height=64,
        input_width=64,
        fc_norm=False,
    )


def example_input_ultrafastlanev2():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Ultra-Fast-Lane-Detection-v2",
        "build_ultrafastlanev2",
        "example_input_ultrafastlanev2",
        2022,
        "vendored",
    ),
]
