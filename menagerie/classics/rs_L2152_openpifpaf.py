# SOURCE: vendored from openpifpaf/openpifpaf @ 9539c071014f02c10e260b2fc1aafca0bc7263b7 (main)
# https://raw.githubusercontent.com/openpifpaf/openpifpaf/main/src/openpifpaf/network/nets.py
# https://raw.githubusercontent.com/openpifpaf/openpifpaf/main/src/openpifpaf/network/basenetworks.py
# https://raw.githubusercontent.com/openpifpaf/openpifpaf/main/src/openpifpaf/network/heads.py
# https://raw.githubusercontent.com/openpifpaf/openpifpaf/main/src/openpifpaf/headmeta.py
#
# Kreiss, Bertoni, Alahi 2021 "OpenPifPaf: Composite Fields for Semantic Keypoint
# Detection and Spatio-Temporal Association" (real-time multi-person pose estimation,
# arXiv:2103.02440; official openpifpaf/openpifpaf repo, `pip install openpifpaf`
# available but not in the installed base-lib set here, so the real source is vendored
# directly). `Shell` (nets.py), `BaseNetwork`/`Resnet` (basenetworks.py),
# `index_field_torch`/`HeadNetwork`/`CompositeField3` (heads.py), and the
# `Base`/`Cif`/`Caf` head-meta dataclasses (headmeta.py) are copied verbatim -- these
# are the real CIF (Composite Intensity Field, i.e. PIF) + CAF (Composite Association
# Field, i.e. PAF) head architecture over a torchvision ResNet backbone that gives
# OpenPifPaf its name. Only the argparse-based `cli()`/`configure()` classmethods and
# `LOG.debug`/`LOG.info` calls are dropped (CLI wiring is not architecture) and the
# resnet is constructed with `pretrained=False` (random-init probe, no weight
# download). `factory.py`'s `resnet18` base-network lambda
# (`basenetworks.Resnet('resnet18', torchvision.models.resnet18, 512)`) and a
# `Cif`+`Caf` head pairing (the classic PIF/PAF single-scale pose configuration) are
# used to instantiate the model, matching how `openpifpaf.network.factory.factory()`
# assembles a `Shell` from a base-network name + head metas.
"""OpenPifPaf: ResNet backbone + CompositeField3 CIF (keypoint intensity) and CAF
(pairwise association) heads for real-time multi-person pose estimation (Kreiss et al.
2021)."""

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Optional, Tuple

import torch
import torchvision.models

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from headmeta.py ---
@dataclass
class Base:
    name: str
    dataset: str

    head_index: Optional[int] = field(default=None, init=False)
    base_stride: Optional[int] = field(default=None, init=False)
    upsample_stride: int = field(default=1, init=False)

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 1
    n_scales: ClassVar[int] = 1
    vector_offsets: ClassVar[List[bool]] = [True]

    @property
    def stride(self) -> Optional[int]:
        if self.base_stride is None:
            return None
        return self.base_stride // self.upsample_stride

    @property
    def n_fields(self) -> int:
        raise NotImplementedError


# --- vendored from headmeta.py ---
@dataclass
class Cif(Base):
    """Head meta data for a Composite Intensity Field (CIF)."""

    keypoints: List[str]
    sigmas: List[float]
    pose: Any = None
    draw_skeleton: Optional[List[Tuple[int, int]]] = None
    score_weights: Optional[List[float]] = None

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 1
    n_scales: ClassVar[int] = 1
    vector_offsets: ClassVar[List[bool]] = [True]

    decoder_min_scale = 0.0
    decoder_seed_mask: Optional[List[int]] = None

    training_weights: Optional[List[float]] = None

    @property
    def n_fields(self) -> int:
        return len(self.keypoints)


# --- vendored from headmeta.py ---
@dataclass
class Caf(Base):
    """Head meta data for a Composite Association Field (CAF)."""

    keypoints: List[str]
    sigmas: List[float]
    skeleton: List[Tuple[int, int]]
    pose: Any = None
    sparse_skeleton: Optional[List[Tuple[int, int]]] = None
    dense_to_sparse_radius: float = 2.0
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2
    vector_offsets: ClassVar[List[bool]] = [True, True]

    decoder_min_distance = 0.0
    decoder_max_distance = float("inf")
    decoder_confidence_scales: Optional[List[float]] = None

    training_weights: Optional[List[float]] = None

    @property
    def n_fields(self) -> int:
        return len(self.skeleton)


# --- vendored from network/basenetworks.py ---
class BaseNetwork(torch.nn.Module):
    """Common base network."""

    def __init__(self, name: str, *, stride: int, out_features: int):
        super().__init__()
        self.name = name
        self.stride = stride
        self.out_features = out_features

    def forward(self, x):
        raise NotImplementedError


# --- vendored from network/basenetworks.py (cli()/configure() argparse classmethods
# dropped; not architecture) ---
class Resnet(BaseNetwork):
    pretrained = False
    pool0_stride = 0
    input_conv_stride = 2
    input_conv2_stride = 0
    remove_last_block = False
    block5_dilation = 1

    def __init__(self, name, torchvision_resnet, out_features=2048):
        modules = list(torchvision_resnet(self.pretrained).children())
        stride = 32

        input_modules = modules[:4]

        # input pool
        if self.pool0_stride:
            if self.pool0_stride != 2:
                input_modules[3].stride = torch.nn.modules.utils._pair(self.pool0_stride)
                stride = int(stride * 2 / self.pool0_stride)
        else:
            input_modules.pop(3)
            stride //= 2

        # input conv
        if self.input_conv_stride != 2:
            input_modules[0].stride = torch.nn.modules.utils._pair(self.input_conv_stride)
            stride = int(stride * 2 / self.input_conv_stride)

        # optional use a conv in place of the max pool
        if self.input_conv2_stride:
            assert not self.pool0_stride
            channels = input_modules[0].out_channels
            conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(channels),
                torch.nn.ReLU(inplace=True),
            )
            input_modules.append(conv2)
            stride *= 2

        # block 5
        block5 = modules[7]
        if self.remove_last_block:
            block5 = None
            stride //= 2
            out_features //= 2

        if self.block5_dilation != 1:
            stride //= 2
            for m in block5.modules():
                if not isinstance(m, torch.nn.Conv2d):
                    continue
                m.stride = torch.nn.modules.utils._pair(1)
                if m.kernel_size[0] == 1:
                    continue
                m.dilation = torch.nn.modules.utils._pair(self.block5_dilation)
                padding = (m.kernel_size[0] - 1) // 2 * self.block5_dilation
                m.padding = torch.nn.modules.utils._pair(padding)

        super().__init__(name, stride=stride, out_features=out_features)
        self.input_block = torch.nn.Sequential(*input_modules)
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]
        self.block5 = block5

    def forward(self, x):
        x = self.input_block(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


# --- vendored from network/heads.py ---
def index_field_torch(shape, device, unsqueeze=(0, 0)):
    assert len(shape) == 2
    xy = torch.empty((2, shape[0], shape[1]), device=device)
    xy[0] = torch.arange(shape[1], device=device)
    xy[1] = torch.arange(shape[0], device=device).unsqueeze(1)

    for dim in unsqueeze:
        xy = torch.unsqueeze(xy, dim)

    return xy


# --- vendored from network/heads.py ---
class HeadNetwork(torch.nn.Module):
    """Base class for head networks."""

    def __init__(self, meta: Base, in_features: int):
        super().__init__()
        self.meta = meta
        self.in_features = in_features

    def forward(self, x):
        raise NotImplementedError


# --- vendored from network/heads.py (cli()/configure() argparse classmethods dropped;
# not architecture) ---
class CompositeField3(HeadNetwork):
    dropout_p = 0.0
    inplace_ops: bool = True

    def __init__(self, meta: Base, in_features, *, kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)

        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)

        # convolution
        out_features = meta.n_fields * (meta.n_confidences + meta.n_vectors * 3 + meta.n_scales)
        self.conv = torch.nn.Conv2d(
            in_features,
            out_features * (meta.upsample_stride**2),
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        # upsample
        assert meta.upsample_stride >= 1
        self.upsample_op = None
        if meta.upsample_stride > 1:
            self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)

    @property
    def sparse_task_parameters(self):
        return [self.conv.weight]

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)
        x = self.conv(x)
        # upscale
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.meta.upsample_stride - 1) // 2
            high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
            if self.training:
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                x = x[
                    :, :, low_cut : int(x.shape[2]) - high_cut, low_cut : int(x.shape[3]) - high_cut
                ]

        # Extract some shape parameters once.
        x_size = x.size()
        batch_size = x_size[0]
        feature_height = int(x_size[2])
        feature_width = int(x_size[3])

        x = x.view(
            batch_size,
            self.meta.n_fields,
            self.meta.n_confidences + self.meta.n_vectors * 3 + self.meta.n_scales,
            feature_height,
            feature_width,
        )

        if not self.training and self.inplace_ops:
            # classification
            classes_x = x[:, :, 0 : self.meta.n_confidences]
            torch.sigmoid_(classes_x)

            # regressions x: add index
            if self.meta.n_vectors > 0:
                index_field = index_field_torch((feature_height, feature_width), device=x.device)
                first_reg_feature = self.meta.n_confidences
                for i, do_offset in enumerate(self.meta.vector_offsets):
                    if not do_offset:
                        continue
                    reg_x = x[:, :, first_reg_feature + i * 2 : first_reg_feature + (i + 1) * 2]
                    reg_x.add_(index_field)

            # scale
            first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
            scales_x = x[:, :, first_scale_feature : first_scale_feature + self.meta.n_scales]
            scales_x[:] = torch.nn.functional.softplus(scales_x)

            # remove width in the middle and add one to the front (v4 style)
            first_width_feature = self.meta.n_confidences + self.meta.n_vectors * 2
            x = torch.cat(
                [
                    x[:, :, first_width_feature : first_width_feature + 1],
                    x[:, :, :first_width_feature],
                    x[:, :, self.meta.n_confidences + self.meta.n_vectors * 3 :],
                ],
                dim=2,
            )
        elif not self.training and not self.inplace_ops:
            x = torch.transpose(x, 1, 2)

            classes_x = x[:, 0 : self.meta.n_confidences]
            classes_x = torch.sigmoid(classes_x)

            first_reg_feature = self.meta.n_confidences
            regs_x = [
                x[:, first_reg_feature + i * 2 : first_reg_feature + (i + 1) * 2]
                for i in range(self.meta.n_vectors)
            ]
            index_field = index_field_torch(
                (feature_height, feature_width), device=x.device, unsqueeze=(1, 0)
            )
            index_field = torch.from_numpy(index_field.numpy())
            regs_x = [
                reg_x + index_field if do_offset else reg_x
                for reg_x, do_offset in zip(regs_x, self.meta.vector_offsets)
            ]

            first_reglogb_feature = self.meta.n_confidences + self.meta.n_vectors * 2
            single_reg_logb = x[:, first_reglogb_feature : first_reglogb_feature + 1]

            first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
            scales_x = x[:, first_scale_feature : first_scale_feature + self.meta.n_scales]
            scales_x = torch.nn.functional.softplus(scales_x)

            x = torch.cat([single_reg_logb, classes_x, *regs_x, scales_x], dim=1)
            x = torch.transpose(x, 1, 2)

        return x


# --- vendored from network/nets.py ---
class Shell(torch.nn.Module):
    def __init__(self, base_net, head_nets, *, process_input=None, process_heads=None):
        super().__init__()

        self.base_net = base_net
        self.head_nets = None
        self.process_input = process_input
        self.process_heads = process_heads

        self.set_head_nets(head_nets)

    @property
    def head_metas(self):
        if self.head_nets is None:
            return None
        return [hn.meta for hn in self.head_nets]

    def set_head_nets(self, head_nets):
        if not isinstance(head_nets, torch.nn.ModuleList):
            head_nets = torch.nn.ModuleList(head_nets)

        for hn_i, hn in enumerate(head_nets):
            hn.meta.head_index = hn_i
            hn.meta.base_stride = self.base_net.stride

        self.head_nets = head_nets

    def forward(self, image_batch, head_mask=None):
        if self.process_input is not None:
            image_batch = self.process_input(image_batch)

        x = self.base_net(image_batch)
        if head_mask is not None:
            head_outputs = tuple(hn(x) if m else None for hn, m in zip(self.head_nets, head_mask))
        else:
            head_outputs = tuple(hn(x) for hn in self.head_nets)

        if self.process_heads is not None:
            head_outputs = self.process_heads(head_outputs)

        return head_outputs


def build_openpifpaf():
    # real factory-style assembly: resnet18 base network (factory.py's
    # `basenetworks.Resnet('resnet18', torchvision.models.resnet18, 512)`, pretrained
    # random-init) + one Cif head (keypoint intensity, i.e. PIF) + one Caf head
    # (pairwise association, i.e. PAF) over a small toy 4-keypoint / 3-edge skeleton --
    # the classic single-scale OpenPifPaf pose-estimation head pairing.
    base_net = Resnet("resnet18", torchvision.models.resnet18, out_features=512)

    keypoints = ["a", "b", "c", "d"]
    sigmas = [0.1, 0.1, 0.1, 0.1]
    skeleton = [(1, 2), (2, 3), (3, 4)]

    cif_meta = Cif(name="cif", dataset="toy", keypoints=keypoints, sigmas=sigmas)
    caf_meta = Caf(name="caf", dataset="toy", keypoints=keypoints, sigmas=sigmas, skeleton=skeleton)

    head_nets = [
        CompositeField3(cif_meta, base_net.out_features),
        CompositeField3(caf_meta, base_net.out_features),
    ]

    model = Shell(base_net, head_nets)
    model.eval()
    return model


def example_input_openpifpaf():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 64, 64),)


MENAGERIE_ENTRIES = [
    ("OpenPifPaf", "build_openpifpaf", "example_input_openpifpaf", 2021, "vendored"),
]
