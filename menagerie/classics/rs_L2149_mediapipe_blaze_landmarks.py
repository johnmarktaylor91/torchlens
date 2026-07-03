# SOURCE: vendored from positive666/mediapipe_PoseEstimation_pytorch @ main
# https://raw.githubusercontent.com/positive666/mediapipe_PoseEstimation_pytorch/main/blazebase.py
# https://raw.githubusercontent.com/positive666/mediapipe_PoseEstimation_pytorch/main/blazeface_landmark.py
# https://raw.githubusercontent.com/positive666/mediapipe_PoseEstimation_pytorch/main/blazehand_landmark.py
#
# Google's MediaPipe Face Mesh (468 3D face landmarks) and MediaPipe Hands landmark model
# (21 3D hand keypoints + handedness) are shipped upstream only as TFLite graphs; this repo
# is a faithful PyTorch re-hosting of the same BlazeFace-style landmark networks (originally
# reverse-engineered layer-by-layer from the released .tflite weights by
# https://github.com/zmurez/MediaPipePyTorch, of which this repo is a fork/adaptation).
# BlazeBlock is copied verbatim from blazebase.py (the shared depthwise-separable residual
# block used by every Blaze* network in the repo); BlazeFaceLandmark is copied verbatim from
# blazeface_landmark.py; BlazeHandLandmark is copied verbatim from blazehand_landmark.py.
# Only the `from blazebase import BlazeLandmark, BlazeBlock` inlining is changed: BlazeBlock
# is inlined directly and the `BlazeLandmark`/`BlazeBase` parent classes (whose extract_roi /
# denormalize_landmarks / load_weights helpers require `cv2` for camera-frame pre/post-
# processing and are irrelevant to the forward pass and not part of the architecture) are
# replaced with a plain `nn.Module` base so this stays a base-torch-only single-file module.
"""MediaPipe BlazeFace-style landmark networks: Face Mesh (468 3D landmarks) and Hands
(21 3D keypoints + handedness), PyTorch port of Google's MediaPipe TFLite graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from blazebase.py ---
class BlazeBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, act="relu", skip_proj=False
    ):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        if skip_proj:
            self.skip_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            self.skip_proj = None

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "prelu":
            self.act = nn.PReLU(out_channels)
        else:
            raise NotImplementedError("unknown activation %s" % act)

    def forward(self, x):
        if self.stride == 2:
            if self.kernel_size == 3:
                h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            else:
                h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.skip_proj is not None:
            x = self.skip_proj(x)
        elif self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


# --- vendored from blazeface_landmark.py (MediaPipe Face Mesh, 468 3D landmarks) ---
class BlazeFaceLandmark(nn.Module):
    """The face landmark model from MediaPipe."""

    def __init__(self):
        super(BlazeFaceLandmark, self).__init__()

        # size of ROIs used for input
        self.resolution = 192

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True
            ),
            nn.PReLU(16),
            BlazeBlock(16, 16, 3, act="prelu"),
            BlazeBlock(16, 16, 3, act="prelu"),
            BlazeBlock(16, 32, 3, 2, act="prelu"),
            BlazeBlock(32, 32, 3, act="prelu"),
            BlazeBlock(32, 32, 3, act="prelu"),
            BlazeBlock(32, 64, 3, 2, act="prelu"),
            BlazeBlock(64, 64, 3, act="prelu"),
            BlazeBlock(64, 64, 3, act="prelu"),
            BlazeBlock(64, 128, 3, 2, act="prelu"),
            BlazeBlock(128, 128, 3, act="prelu"),
            BlazeBlock(128, 128, 3, act="prelu"),
            BlazeBlock(128, 128, 3, 2, act="prelu"),
            BlazeBlock(128, 128, 3, act="prelu"),
            BlazeBlock(128, 128, 3, act="prelu"),
        )

        # facial_landmark head
        self.backbone2a = nn.Sequential(
            BlazeBlock(128, 128, 3, 2, act="prelu"),
            BlazeBlock(128, 128, 3, act="prelu"),
            BlazeBlock(128, 128, 3, act="prelu"),
            nn.Conv2d(128, 32, 1, padding=0, bias=True),
            nn.PReLU(32),
            BlazeBlock(32, 32, 3, act="prelu"),
            nn.Conv2d(32, 1404, 3, padding=0, bias=True),
        )

        self.backbone2b = nn.Sequential(
            BlazeBlock(128, 128, 3, 2, act="prelu"),
            nn.Conv2d(128, 32, 1, padding=0, bias=True),
            nn.PReLU(32),
            BlazeBlock(32, 32, 3, act="prelu"),
            nn.Conv2d(32, 1, 3, padding=0, bias=True),
        )

    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros((0,)), torch.zeros((0, 468, 3))

        x = nn.ReflectionPad2d((1, 0, 1, 0))(x)

        x = self.backbone1(x)
        landmarks = self.backbone2a(x).view(-1, 468, 3) / 192
        flag = self.backbone2b(x).sigmoid().view(-1)

        return flag, landmarks


# --- vendored from blazehand_landmark.py (MediaPipe Hands, 21 3D keypoints) ---
class BlazeHandLandmark(nn.Module):
    """The hand landmark model from MediaPipe."""

    def __init__(self):
        super(BlazeHandLandmark, self).__init__()

        # size of ROIs used for input
        self.resolution = 256

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=0, bias=True
            ),
            nn.ReLU(inplace=True),
            BlazeBlock(24, 24, 5),
            BlazeBlock(24, 24, 5),
            BlazeBlock(24, 48, 5, 2),
        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 96, 5, 2),
        )

        self.backbone3 = nn.Sequential(
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5, 2),
        )

        self.backbone4 = nn.Sequential(
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5, 2),
        )

        self.blaze5 = BlazeBlock(96, 96, 5)
        self.blaze6 = BlazeBlock(96, 96, 5)
        self.conv7 = nn.Conv2d(96, 48, 1, bias=True)

        self.backbone8 = nn.Sequential(
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 96, 5, 2),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
        )

        self.hand_flag = nn.Conv2d(288, 1, 2, bias=True)
        self.handed = nn.Conv2d(288, 1, 2, bias=True)
        self.landmarks = nn.Conv2d(288, 63, 2, bias=True)

    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0, 21, 3))

        x = F.pad(x, (0, 1, 0, 1), "constant", 0)

        x = self.backbone1(x)
        y = self.backbone2(x)
        z = self.backbone3(y)
        w = self.backbone4(z)

        z = z + F.interpolate(w, scale_factor=2, mode="bilinear")
        z = self.blaze5(z)

        y = y + F.interpolate(z, scale_factor=2, mode="bilinear")
        y = self.blaze6(y)
        y = self.conv7(y)

        x = x + F.interpolate(y, scale_factor=2, mode="bilinear")

        x = self.backbone8(x)

        hand_flag = self.hand_flag(x).view(-1).sigmoid()
        handed = self.handed(x).view(-1).sigmoid()
        landmarks = self.landmarks(x).view(-1, 21, 3) / 256

        return hand_flag, handed, landmarks


def build_blaze_face_mesh():
    return BlazeFaceLandmark()


def example_input_blaze_face_mesh():
    torch.manual_seed(0)
    return (torch.rand(1, 3, 192, 192),)


def build_blaze_hands():
    return BlazeHandLandmark()


def example_input_blaze_hands():
    torch.manual_seed(0)
    return (torch.rand(1, 3, 256, 256),)


MENAGERIE_ENTRIES = [
    (
        "MediaPipe Face Mesh (BlazeFace landmark)",
        "build_blaze_face_mesh",
        "example_input_blaze_face_mesh",
        2019,
        "vendored",
    ),
    (
        "MediaPipe Hands (BlazeHand landmark)",
        "build_blaze_hands",
        "example_input_blaze_hands",
        2020,
        "vendored",
    ),
]
