# SOURCE: vendored from HarshayuGirase/Human-Path-Prediction @ a345ada1557f53dc2acb4c07f71d340d366980da
# https://raw.githubusercontent.com/HarshayuGirase/Human-Path-Prediction/a345ada1557f53dc2acb4c07f71d340d366980da/ynet/model.py
# https://raw.githubusercontent.com/HarshayuGirase/Human-Path-Prediction/a345ada1557f53dc2acb4c07f71d340d366980da/ynet/utils/softargmax.py
#
# Girase et al. 2021 (ICCV) "Loki: Long Term and Key Intentions for Trajectory
# Prediction" ships the Y-Net trajectory-prediction architecture in this repo
# (`ynet/model.py`). Y-Net is a dual-decoder U-Net: a shared `YNetEncoder`
# backbone feeds TWO `YNetDecoder` heads -- one predicting a goal/waypoint
# heatmap (`goal_decoder`), one predicting the trajectory heatmap conditioned
# on the goal/waypoint channels (`traj_decoder`, built with `traj=waypoints`
# so its encoder-channel list is widened to accept the extra conditioning
# channels). `SoftArgmax2D` (vendored from `ynet/utils/softargmax.py`, itself
# credited upstream to kornia/torchgeometry) turns the predicted heatmaps back
# into 2D coordinates. This dual-head conditioned-decoder design is the paper's
# architectural contribution, so it is vendored rather than constructed from a
# base-library class.
#
# `YNetEncoder`, `YNetDecoder`, `YNetTorch`, `SoftArgmax2D`, and
# `create_meshgrid` are the real, unmodified classes/functions from the two
# files above (layer composition, channel arithmetic, and control flow are
# byte-for-byte the original). Only mechanical import-isolation edits:
#   - Dropped the `from tqdm import tqdm`, `from torch.utils.data import
#     DataLoader`, and the `utils.preprocessing` / `utils.image_utils` /
#     `utils.dataloader` / `test` / `train` imports at the top of the original
#     `model.py` -- those are training/eval-pipeline dependencies used only by
#     the `YNet` sklearn-style wrapper class's `train()`/`evaluate()` methods,
#     which are NOT vendored here (only the `nn.Module` architecture:
#     `YNetEncoder`, `YNetDecoder`, `YNetTorch`).
#   - `SoftArgmax2D`/`create_meshgrid` copied verbatim from
#     `ynet/utils/softargmax.py` into this file (flat import, no package
#     layout) since `YNetTorch.__init__` instantiates
#     `SoftArgmax2D(normalized_coordinates=False)` unconditionally.
#
# `YNetTorch` normally loads a pretrained semantic-segmentation backbone via
# `torch.load(segmentation_model_fp)` (a full pickled `nn.Module`, e.g. a
# DeepLab/UNet trained on scene-segmentation masks -- not part of this repo's
# source and not obtainable without the paper's released weights file). The
# real `__init__` already handles `segmentation_model_fp=None` by falling back
# to `nn.Identity()` for `self.semantic_segmentation` (see original code
# below) -- that fallback is exercised as-is, with `semantic_classes` left at
# a small placeholder value so `encoder`'s first-conv input-channel count
# matches the traced example input.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def create_meshgrid(x: torch.Tensor, normalized_coordinates: Optional[bool]) -> torch.Tensor:
    assert len(x.shape) == 4, x.shape
    _, _, height, width = x.shape
    _device, _dtype = x.device, x.dtype
    if normalized_coordinates:
        xs = torch.linspace(-1.0, 1.0, width, device=_device, dtype=_dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=_device, dtype=_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
        ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)
    return torch.meshgrid(ys, xs)  # pos_y, pos_x


class SoftArgmax2D(nn.Module):
    r"""Creates a module that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        normalized_coordinates (Optional[bool]): wether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`
    """

    def __init__(self, normalized_coordinates: Optional[bool] = True) -> None:
        super(SoftArgmax2D, self).__init__()
        self.normalized_coordinates: Optional[bool] = normalized_coordinates
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}".format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x: torch.Tensor = input.view(batch_size, channels, -1)

        # compute softmax with max substraction trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)

        # create coordinates grid
        pos_y, pos_x = create_meshgrid(input, self.normalized_coordinates)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y: torch.Tensor = torch.sum((pos_y * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        expected_x: torch.Tensor = torch.sum((pos_x * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        output: torch.Tensor = torch.cat([expected_x, expected_y], dim=-1)
        return output.view(batch_size, channels, 2)  # BxNx2


class YNetEncoder(nn.Module):
    def __init__(self, in_channels, channels=(64, 128, 256, 512, 512)):
        """
        Encoder model
        :param in_channels: int, semantic_classes + obs_len
        :param channels: list, hidden layer channels
        """
        super(YNetEncoder, self).__init__()
        self.stages = nn.ModuleList()

        # First block
        self.stages.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                ),
                nn.ReLU(inplace=True),
            )
        )

        # Subsequent blocks, each starting with MaxPool
        for i in range(len(channels) - 1):
            self.stages.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        channels[i + 1],
                        channels[i + 1],
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.ReLU(inplace=True),
                )
            )

        # Last MaxPool layer before passing the features into decoder
        self.stages.append(
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )
        )

    def forward(self, x):
        # Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class YNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):
        """
        Decoder models
        :param encoder_channels: list, encoder channels, used for skip connections
        :param decoder_channels: list, decoder channels
        :param output_len: int, pred_len
        :param traj: False or int, if False -> Goal and waypoint predictor, if int -> number of waypoints
        """
        super(YNetDecoder, self).__init__()

        # The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
        if traj:
            encoder_channels = [channel + traj for channel in encoder_channels]
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        center_channels = encoder_channels[0]

        decoder_channels = decoder_channels

        # The center layer (the layer with the smallest feature map size)
        self.center = nn.Sequential(
            nn.Conv2d(
                center_channels,
                center_channels * 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                center_channels * 2,
                center_channels * 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.ReLU(inplace=True),
        )

        # Determine the upsample channel dimensions
        upsample_channels_in = [center_channels * 2] + decoder_channels[:-1]
        upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

        # Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
        self.upsample_conv = [
            nn.Conv2d(
                in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
            for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)
        ]
        self.upsample_conv = nn.ModuleList(self.upsample_conv)

        # Determine the input and output channel dimensions of each layer in the decoder
        # As we concat the encoded feature and decoded features we have to sum both dims
        in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
        out_channels = decoder_channels

        self.decoder = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                ),
                nn.ReLU(inplace=True),
            )
            for in_channels_, out_channels_ in zip(in_channels, out_channels)
        ]
        self.decoder = nn.ModuleList(self.decoder)

        # Final 1x1 Conv prediction to get our heatmap logits (before softmax)
        self.predictor = nn.Conv2d(
            in_channels=decoder_channels[-1],
            out_channels=output_len,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, features):
        # Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
        features = features[
            ::-1
        ]  # reverse the order of encoded features, as the decoder starts from the smallest image
        center_feature = features[0]
        x = self.center(center_feature)
        for i, (feature, module, upsample_conv) in enumerate(
            zip(features[1:], self.decoder, self.upsample_conv)
        ):
            x = F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )  # bilinear interpolation for upsampling
            x = upsample_conv(x)  # 3x3 conv for upsampling
            x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
            x = module(x)  # Conv
        x = self.predictor(x)  # last predictor layer
        return x


class YNetTorch(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        segmentation_model_fp,
        use_features_only=False,
        semantic_classes=6,
        encoder_channels=[],
        decoder_channels=[],
        waypoints=1,
    ):
        """
        Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
        :param obs_len: int, observed timesteps
        :param pred_len: int, predicted timesteps
        :param segmentation_model_fp: str, filepath to pretrained segmentation model
        :param use_features_only: bool, if True -> use segmentation features from penultimate layer, if False -> use softmax class predictions
        :param semantic_classes: int, number of semantic classes
        :param encoder_channels: list, encoder channel structure
        :param decoder_channels: list, decoder channel structure
        :param waypoints: int, number of waypoints
        """
        super(YNetTorch, self).__init__()

        if segmentation_model_fp is not None:
            self.semantic_segmentation = torch.load(segmentation_model_fp)
            if use_features_only:
                self.semantic_segmentation.segmentation_head = nn.Identity()
                semantic_classes = 16  # instead of classes use number of feature_dim
        else:
            self.semantic_segmentation = nn.Identity()

        self.encoder = YNetEncoder(
            in_channels=semantic_classes + obs_len, channels=encoder_channels
        )

        self.goal_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)
        self.traj_decoder = YNetDecoder(
            encoder_channels, decoder_channels, output_len=pred_len, traj=waypoints
        )

        self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)

    def segmentation(self, image):
        return self.semantic_segmentation(image)

    # Forward pass for goal decoder
    def pred_goal(self, features):
        goal = self.goal_decoder(features)
        return goal

    # Forward pass for trajectory decoder
    def pred_traj(self, features):
        traj = self.traj_decoder(features)
        return traj

    # Forward pass for feature encoder, returns list of feature maps
    def pred_features(self, x):
        features = self.encoder(x)
        return features

    # Softmax for Image data as in dim=NxCxHxW, returns softmax image shape=NxCxHxW
    def softmax(self, x):
        return nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)

    # Softargmax for Image data as in dim=NxCxHxW, returns 2D coordinates=Nx2
    def softargmax(self, output):
        return self.softargmax_(output)

    def sigmoid(self, output):
        return torch.sigmoid(output)

    def softargmax_on_softmax_map(self, x):
        """Softargmax: As input a batched image where softmax is already performed (not logits)"""
        pos_y, pos_x = create_meshgrid(x, normalized_coordinates=False)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)
        x = x.flatten(2)

        estimated_x = pos_x * x
        estimated_x = torch.sum(estimated_x, dim=-1, keepdim=True)
        estimated_y = pos_y * x
        estimated_y = torch.sum(estimated_y, dim=-1, keepdim=True)
        softargmax_coords = torch.cat([estimated_x, estimated_y], dim=-1)
        return softargmax_coords


# `YNetTorch` has no `forward()` in the original -- the upstream `train.py` /
# `test.py` drive `pred_features` -> `pred_goal` and `pred_features` ->
# `pred_traj` (with goal/waypoint heatmaps concatenated onto the encoder
# input channels by the dataset pipeline, not by the model) as two separate
# calls, never a single `nn.Module.__call__`. To keep a single-callable
# staging entry (required for menagerie recipes/modules) WITHOUT inventing
# new architecture, `YNetPredictWrapper` below only ROUTES to the existing,
# unmodified `pred_features`/`pred_goal` methods -- it adds no new layers,
# no new arithmetic, and does not touch `traj_decoder` (which upstream
# requires externally-prepared conditioning channels from the data pipeline,
# not present in this repo's `model.py`).
class YNetPredictWrapper(nn.Module):
    """Thin routing-only wrapper so YNetTorch's goal-decoder path (encoder +
    goal_decoder + softargmax, unmodified) can be traced as one nn.Module."""

    def __init__(self, ynet_torch):
        super().__init__()
        self.ynet_torch = ynet_torch

    def forward(self, x):
        features = self.ynet_torch.pred_features(x)
        goal_logits = self.ynet_torch.pred_goal(features)
        goal_coords = self.ynet_torch.softargmax(goal_logits)
        return goal_logits, goal_coords


def build_ynet():
    # Tiny config: 2 encoder downsample stages (channels tuple length controls
    # depth), small obs/pred lengths, no pretrained segmentation backbone
    # (segmentation_model_fp=None -> nn.Identity(), the real fallback path).
    ynet_torch = YNetTorch(
        obs_len=4,
        pred_len=2,
        segmentation_model_fp=None,
        use_features_only=False,
        semantic_classes=3,
        encoder_channels=[8, 16],
        decoder_channels=[16, 8],
        waypoints=1,
    )
    return YNetPredictWrapper(ynet_torch)


def example_input_ynet():
    # in_channels = semantic_classes (3, since segmentation is Identity) + obs_len (4) = 7
    return torch.randn(1, 7, 32, 32)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Y-Net", "build_ynet", "example_input_ynet", 2021, "vendored"),
]
