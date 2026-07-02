# SOURCE: vendored from robodhruv/visualnav-transformer @ main
#   train/vint_train/models/gnm/gnm.py
#   train/vint_train/models/gnm/modified_mobilenetv2.py
#   train/vint_train/models/base_model.py
# GNM (General Navigation Model): goal-conditioned visual navigation model using twin
# MobileNetV2 encoders (one for the observation-context stack, one for the
# observation+goal stack) that fuse into distance and trajectory-action predictors.
"""Vendored GNM model definition (robot visual navigation, RSS 2023 / robodhruv)."""

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.ops.misc import ConvNormActivation


class BaseModel(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
    ) -> None:
        super(BaseModel, self).__init__()
        self.context_size = context_size
        self.learn_angle = learn_angle
        self.len_trajectory_pred = len_traj_pred
        if self.learn_angle:
            self.num_action_params = 4
        else:
            self.num_action_params = 2

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z

    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MobileNetEncoder(nn.Module):
    def __init__(
        self,
        num_images: int = 1,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            ConvNormActivation(
                num_images * 3,
                input_channel,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        ]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel
        features.append(
            ConvNormActivation(
                input_channel,
                self.last_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        )
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class GNM(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
    ) -> None:
        """
        GNM main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(GNM, self).__init__(context_size, len_traj_pred, learn_angle)
        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = mobilenet.features
        self.obs_encoding_size = obs_encoding_size
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )
        stacked_mobilenet = MobileNetEncoder(
            num_images=2 + self.context_size
        )  # stack the goal and the current observation
        self.goal_mobilenet = stacked_mobilenet.features
        self.goal_encoding_size = goal_encoding_size
        self.compress_goal = nn.Sequential(
            nn.Linear(stacked_mobilenet.last_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.goal_encoding_size),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(self.goal_encoding_size + self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)

        obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        goal_encoding = self.goal_mobilenet(obs_goal_input)
        goal_encoding = self.flatten(goal_encoding)
        goal_encoding = self.compress_goal(goal_encoding)

        z = torch.cat([obs_encoding, goal_encoding], dim=1)
        z = self.linear_layers(z)
        dist_pred = self.dist_predictor(z)
        action_pred = self.action_predictor(z)

        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred


class _GNMTraceWrapper(nn.Module):
    """Tuple-in/tuple-out wrapper so TorchLens sees a single positional call."""

    def __init__(self, gnm: GNM) -> None:
        super().__init__()
        self.gnm = gnm

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        obs_img, goal_img = inputs
        return self.gnm(obs_img, goal_img)


_CONTEXT_SIZE = 2


def build_gnm() -> nn.Module:
    gnm = GNM(
        context_size=_CONTEXT_SIZE,
        len_traj_pred=3,
        learn_angle=True,
        obs_encoding_size=32,
        goal_encoding_size=32,
    )
    gnm.eval()
    return _GNMTraceWrapper(gnm)


def example_input_gnm() -> Tuple[torch.Tensor, torch.Tensor]:
    batch = 1
    # obs stack: (1 + context_size) RGB frames stacked on channel dim
    obs_img = torch.randn(batch, 3 * (1 + _CONTEXT_SIZE), 64, 64)
    # goal image: a single RGB frame; forward() concatenates it onto obs_img to
    # build the (2 + context_size) * 3 channel stack fed to goal_mobilenet
    goal_img = torch.randn(batch, 3, 64, 64)
    return (obs_img, goal_img)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    ("GNM", build_gnm, example_input_gnm, 2023, MENAGERIE_ZOO),
]
