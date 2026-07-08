# SOURCE: vendored from flashcp/TextBoxes_plusplus_Pytorch_for_mtwi @ master
# https://raw.githubusercontent.com/flashcp/TextBoxes_plusplus_Pytorch_for_mtwi/master/ssd.py
# https://raw.githubusercontent.com/flashcp/TextBoxes_plusplus_Pytorch_for_mtwi/master/layers/modules/l2norm.py
# https://raw.githubusercontent.com/flashcp/TextBoxes_plusplus_Pytorch_for_mtwi/master/layers/functions/prior_box.py
# https://raw.githubusercontent.com/flashcp/TextBoxes_plusplus_Pytorch_for_mtwi/master/data/config.py
#
# Liao, Shi, Bai 2018 "TextBoxes++: A Single-Shot Oriented Scene Text Detector" (AAAI
# 2018 / TIP 2018). SSD-style single-shot detector with a VGG16 backbone, extra feature
# layers, and multibox loc/conf heads emitting 12-value (quadrilateral + axis-aligned
# box) location offsets per prior instead of SSD's usual 4. `SSD` (renamed `TextBoxesPP`
# here), `vgg()`, `add_extras()`, `multibox()`, `L2Norm`, and `PriorBox` are copied
# verbatim from ssd.py / layers/modules/l2norm.py / layers/functions/prior_box.py; the
# `mtwi384` anchor-config dict is copied verbatim from data/config.py. Built in "train"
# phase (returns raw loc/conf/prior tensors) so the forward pass does not require the
# `Detect` NMS postprocessing layer or the OpenCV-dependent `data` package (dataset
# loaders, box_utils) that the original `layers`/`data` packages pull in transitively.
"""TextBoxes++: SSD-style oriented scene-text detector (VGG16 backbone + 12-value
quadrilateral multibox heads)."""

from math import sqrt as sqrt
from itertools import product as product

import torch
import torch.nn as nn
import torch.nn.init as init

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from data/config.py ---
mtwi384 = {
    "num_classes": 2,
    "lr_steps": (80000, 100000, 120000),
    "max_iter": 90100,
    "feature_maps": [48, 24, 12, 6, 4, 2],
    "min_dim": 384,
    "steps": [8, 16, 32, 64, 100, 200],
    "min_sizes": [38, 76, 142, 207, 272, 337],
    "max_sizes": [76, 142, 207, 272, 337, 403],
    "aspect_ratios": [[2, 3], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3], [2, 3]],
    "variance": [0.1, 0.2],
    "clip": True,
    "name": "mtwi",
}


# --- vendored from layers/modules/l2norm.py ---
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


# --- vendored from layers/functions/prior_box.py ---
class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg["min_dim"]
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg["aspect_ratios"])
        self.variance = cfg["variance"] or [0.1]
        self.feature_maps = cfg["feature_maps"]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.clip = cfg["clip"]
        self.version = cfg["name"]
        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


# --- vendored from ssd.py ---
class TextBoxesPP(nn.Module):
    """Single Shot Multibox Architecture (TextBoxes++ variant of SSD).
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions (12 values: quad + box)
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf (SSD) and the TextBoxes++ paper for details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(TextBoxesPP, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = mtwi384
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,384,384].

        Return (train phase):
            tuple of (loc, conf, priors):
                loc: Shape [batch, num_priors*12]
                conf: Shape [batch, num_priors, num_classes]
                priors: Shape [num_priors, 4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = torch.relu(v(x))
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for x, l, c in zip(sources, self.loc, self.conf):  # noqa: E741 (kept as in upstream source)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        output = (
            loc.view(loc.size(0), -1, 12),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors,
        )
        return output


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "C":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != "S":
            if v == "S":
                layers += [
                    nn.Conv2d(
                        in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1
                    )
                ]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [
            nn.Conv2d(vgg[v].out_channels, cfg[k] * 12, kernel_size=(3, 5), padding=(1, 2))
        ]
        conf_layers += [
            nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=(3, 5), padding=(1, 2))
        ]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 12, kernel_size=(3, 5), padding=(1, 2))]
        conf_layers += [
            nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=(3, 5), padding=(1, 2))
        ]
    return vgg, extra_layers, (loc_layers, conf_layers)


base_cfg = {
    "384": [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512],
}
extras_cfg = {
    "384": [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
}
mbox_cfg = {
    "384": [6, 8, 8, 8, 6, 6],  # number of boxes per feature map location
}


def build_textboxes_pp(phase="train", size=384, num_classes=2):
    base_, extras_, head_ = multibox(
        vgg(base_cfg[str(size)], 3),
        add_extras(extras_cfg[str(size)], 1024),
        mbox_cfg[str(size)],
        num_classes,
    )
    model = TextBoxesPP(phase, size, base_, extras_, head_, num_classes)
    model.eval()
    return model


def example_input_textboxes_pp():
    torch.manual_seed(0)
    # real repo's mtwi384 config: 384x384 input.
    return (torch.randn(1, 3, 384, 384),)


MENAGERIE_ENTRIES = [
    ("TextBoxes++", "build_textboxes_pp", "example_input_textboxes_pp", 2018, "vendored"),
]
