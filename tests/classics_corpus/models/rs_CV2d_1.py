# SOURCE: vendored from KalmanNet/KalmanNet_TSP @ main (KNet/KalmanNet_nn.py);
# vendored from nicofarr/brainnetcnnVis_pytorch @ master (BrainNetCnnGoldMSI.py);
# vendored from CBICA/BrainMaGe @ master (BrainMaGe/models/networks.py,
# BrainMaGe/models/seg_modules.py);
# vendored from nabeelre/BTSbot @ main (btsbot/architectures.py);
# vendored from gram-ai/capsule-networks @ master (capsule_network.py).
"""CV2d vendored TorchLens menagerie staging models."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class KalmanNetNN(torch.nn.Module):
    """KalmanNet recurrent Kalman-gain estimator from the source repository."""

    def __init__(self) -> None:
        """Initialize an unbuilt KalmanNet module."""
        super().__init__()

    def NNBuild(self, SysModel: SimpleNamespace, args: SimpleNamespace) -> None:
        """Build the network from system dynamics and source-style args.

        Parameters
        ----------
        SysModel
            Namespace carrying dynamics, observation, dimensions, and priors.
        args
            Namespace carrying source KalmanNet width and device options.
        """
        if args.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    def InitKGainNet(
        self,
        prior_Q: torch.Tensor,
        prior_Sigma: torch.Tensor,
        prior_S: torch.Tensor,
        args: SimpleNamespace,
    ) -> None:
        """Initialize Kalman-gain recurrent and fully connected layers.

        Parameters
        ----------
        prior_Q
            Prior process covariance.
        prior_Sigma
            Prior posterior covariance.
        prior_S
            Prior innovation covariance.
        args
            Source KalmanNet args namespace.
        """
        self.seq_len_input = 1
        self.batch_size = args.n_batch

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m**2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet
        self.d_hidden_Sigma = self.m**2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        self.d_input_S = self.n**2 + 2 * self.n * args.in_mult_KNet
        self.d_hidden_S = self.n**2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n**2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU(),
        ).to(self.device)

        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2),
        ).to(self.device)

        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m**2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.ReLU(),
        ).to(self.device)

        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU(),
        ).to(self.device)

        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.ReLU(),
        ).to(self.device)

        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.ReLU(),
        ).to(self.device)

        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.ReLU(),
        ).to(self.device)

    def InitSystemDynamics(self, f: object, h: object, m: int, n: int) -> None:
        """Initialize system dynamics functions and dimensions.

        Parameters
        ----------
        f
            State evolution callable.
        h
            Observation callable.
        m
            State dimension.
        n
            Observation dimension.
        """
        self.f = f
        self.m = m
        self.h = h
        self.n = n

    def InitSequence(self, M1_0: torch.Tensor, T: int) -> None:
        """Initialize source-style per-sequence filtering state.

        Parameters
        ----------
        M1_0
            Initial posterior first moment.
        T
            Sequence length.
        """
        self.T = T
        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

    def step_prior(self) -> None:
        """Compute prior state and observation moments."""
        self.m1x_prior = self.f(self.m1x_posterior)
        self.m1y = self.h(self.m1x_prior)

    def step_KGain_est(self, y: torch.Tensor) -> None:
        """Estimate Kalman gain from normalized innovation features.

        Parameters
        ----------
        y
            Current observation tensor.
        """
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)
        fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(
            self.m1x_posterior_previous,
            2,
        )
        fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(
            self.m1x_prior_previous,
            2,
        )

        obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = F.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = F.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = F.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)

        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    def KNet_step(self, y: torch.Tensor) -> torch.Tensor:
        """Run one KalmanNet filtering step.

        Parameters
        ----------
        y
            Current observation tensor.

        Returns
        -------
        torch.Tensor
            Updated posterior state estimate.
        """
        self.step_prior()
        self.step_KGain_est(y)
        dy = y - self.m1y
        inov = torch.bmm(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + inov
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y
        return self.m1x_posterior

    def KGain_step(
        self,
        obs_diff: torch.Tensor,
        obs_innov_diff: torch.Tensor,
        fw_evol_diff: torch.Tensor,
        fw_update_diff: torch.Tensor,
    ) -> torch.Tensor:
        """Run the recurrent Kalman-gain network.

        Parameters
        ----------
        obs_diff
            Observation difference features.
        obs_innov_diff
            Innovation difference features.
        fw_evol_diff
            Forward evolution difference features.
        fw_update_diff
            Forward update difference features.

        Returns
        -------
        torch.Tensor
            Flattened Kalman-gain estimate.
        """

        def expand_dim(x: torch.Tensor) -> torch.Tensor:
            """Add the source sequence dimension.

            Parameters
            ----------
            x
                Batch feature tensor.

            Returns
            -------
            torch.Tensor
                Expanded sequence-first tensor.
            """
            expanded = torch.empty(
                self.seq_len_input,
                self.batch_size,
                x.shape[-1],
                device=self.device,
            )
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        out_FC5 = self.FC5(fw_update_diff)
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)
        out_FC6 = self.FC6(fw_evol_diff)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(torch.cat((out_Q, out_FC6), 2), self.h_Sigma)
        out_FC1 = self.FC1(out_Sigma)
        out_FC7 = self.FC7(torch.cat((obs_diff, obs_innov_diff), 2))
        out_S, self.h_S = self.GRU_S(torch.cat((out_FC1, out_FC7), 2), self.h_S)
        out_FC2 = self.FC2(torch.cat((out_Sigma, out_S), 2))
        out_FC3 = self.FC3(torch.cat((out_S, out_FC2), 2))
        out_FC4 = self.FC4(torch.cat((out_Sigma, out_FC3), 2))
        self.h_Sigma = out_FC4
        return out_FC2

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Run one KalmanNet forward step.

        Parameters
        ----------
        y
            Observation tensor.

        Returns
        -------
        torch.Tensor
            Posterior state estimate.
        """
        y = y.to(self.device)
        return self.KNet_step(y)

    def init_hidden_KNet(self) -> None:
        """Initialize recurrent hidden states from covariance priors."""
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = (
            self.prior_S.flatten()
            .reshape(1, 1, -1)
            .repeat(
                self.seq_len_input,
                self.batch_size,
                1,
            )
        )
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = (
            self.prior_Sigma.flatten()
            .reshape(1, 1, -1)
            .repeat(
                self.seq_len_input,
                self.batch_size,
                1,
            )
        )
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = (
            self.prior_Q.flatten()
            .reshape(1, 1, -1)
            .repeat(
                self.seq_len_input,
                self.batch_size,
                1,
            )
        )


class E2EBlock(torch.nn.Module):
    """BrainNetCNN edge-to-edge block."""

    def __init__(
        self, in_planes: int, planes: int, example: torch.Tensor, bias: bool = False
    ) -> None:
        """Initialize an edge-to-edge block.

        Parameters
        ----------
        in_planes
            Input channel count.
        planes
            Output channel count.
        example
            Example connectivity tensor used by the source code for graph size.
        bias
            Whether convolutions use bias.
        """
        super().__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply edge-to-edge row and column convolutions.

        Parameters
        ----------
        x
            Connectivity tensor.

        Returns
        -------
        torch.Tensor
            Edge-to-edge features.
        """
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class BrainNetCNN(torch.nn.Module):
    """BrainNetCNN from the source notebook export."""

    def __init__(self, example: torch.Tensor, num_classes: int = 10) -> None:
        """Initialize BrainNetCNN.

        Parameters
        ----------
        example
            Example connectivity tensor used by the source code for graph size.
        num_classes
            Unused source argument retained for provenance compatibility.
        """
        super().__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)
        self.num_classes = num_classes
        self.e2econv1 = E2EBlock(1, 32, example, bias=True)
        self.e2econv2 = E2EBlock(32, 64, example, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run BrainNetCNN prediction.

        Parameters
        ----------
        x
            Connectivity tensor.

        Returns
        -------
        torch.Tensor
            Prediction logits.
        """
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        return F.leaky_relu(self.dense3(out), negative_slope=0.33)


class InConv(nn.Module):
    """BrainMaGe input convolution module."""

    def __init__(self, input_channels: int, output_channels: int, res: bool = False) -> None:
        """Initialize the input convolution module.

        Parameters
        ----------
        input_channels
            Input channel count.
        output_channels
            Output channel count.
        res
            Whether to use the source residual path.
        """
        super().__init__()
        self.residual = res
        self.dropout_p = 0.3
        self.leakiness = 1e-2
        self.lrelu_inplace = True
        self.dropout = nn.Dropout3d(self.dropout_p)
        self.in_0 = nn.InstanceNorm3d(output_channels, affine=True, track_running_stats=True)
        self.in_1 = nn.InstanceNorm3d(output_channels, affine=True, track_running_stats=True)
        self.conv0 = nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the input convolution module.

        Parameters
        ----------
        x
            Input volume.

        Returns
        -------
        torch.Tensor
            Encoded volume.
        """
        x = self.conv0(x)
        if self.residual:
            skip = x
        x = F.leaky_relu(self.in_0(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv1(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = F.leaky_relu(self.in_1(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv2(x)
        if self.residual:
            x = x + skip
        return x


class DownsamplingModule(nn.Module):
    """BrainMaGe strided downsampling module."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        """Initialize a downsampling module.

        Parameters
        ----------
        input_channels
            Input channel count.
        output_channels
            Output channel count.
        """
        super().__init__()
        self.leakiness = 1e-2
        self.lrelu_inplace = True
        self.in_0 = nn.InstanceNorm3d(output_channels, affine=True, track_running_stats=True)
        self.conv0 = nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample an encoded volume.

        Parameters
        ----------
        x
            Input volume.

        Returns
        -------
        torch.Tensor
            Downsampled volume.
        """
        return F.leaky_relu(
            self.in_0(self.conv0(x)),
            negative_slope=self.leakiness,
            inplace=self.lrelu_inplace,
        )


class EncodingModule(nn.Module):
    """BrainMaGe residual encoding module."""

    def __init__(self, input_channels: int, output_channels: int, res: bool = False) -> None:
        """Initialize an encoding module.

        Parameters
        ----------
        input_channels
            Input channel count.
        output_channels
            Output channel count.
        res
            Whether to use the source residual path.
        """
        super().__init__()
        self.res = res
        self.dropout_p = 0.3
        self.leakiness = 1e-2
        self.lrelu_inplace = True
        self.dropout = nn.Dropout3d(self.dropout_p)
        self.in_0 = nn.InstanceNorm3d(output_channels, affine=True, track_running_stats=True)
        self.in_1 = nn.InstanceNorm3d(output_channels, affine=True, track_running_stats=True)
        self.conv0 = nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run an encoding module.

        Parameters
        ----------
        x
            Input volume.

        Returns
        -------
        torch.Tensor
            Encoded volume.
        """
        if self.res:
            skip = x
        x = F.leaky_relu(self.in_0(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv0(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = F.leaky_relu(self.in_1(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv1(x)
        if self.res:
            x = x + skip
        return x


class UpsamplingModule(nn.Module):
    """BrainMaGe transpose-convolution upsampling module."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        """Initialize an upsampling module.

        Parameters
        ----------
        input_channels
            Input channel count.
        output_channels
            Output channel count.
        """
        super().__init__()
        self.conv0 = nn.ConvTranspose3d(input_channels, output_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample an encoded volume.

        Parameters
        ----------
        x
            Input volume.

        Returns
        -------
        torch.Tensor
            Upsampled volume.
        """
        return self.conv0(x)


class DecodingModule(nn.Module):
    """BrainMaGe decoder module with skip concatenation."""

    def __init__(self, input_channels: int, output_channels: int, res: bool = False) -> None:
        """Initialize a decoder module.

        Parameters
        ----------
        input_channels
            Input channel count after concatenation.
        output_channels
            Output channel count.
        res
            Whether to use the source residual path.
        """
        super().__init__()
        self.res = res
        self.dropout_p = 0.3
        self.leakiness = 1e-2
        self.lrelu_inplace = True
        self.dropout = nn.Dropout3d(self.dropout_p)
        self.in_0 = nn.InstanceNorm3d(input_channels, affine=True, track_running_stats=True)
        self.in_1 = nn.InstanceNorm3d(output_channels, affine=True, track_running_stats=True)
        self.conv0 = nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Decode and merge with a skip tensor.

        Parameters
        ----------
        x1
            Upsampled tensor.
        x2
            Skip tensor.

        Returns
        -------
        torch.Tensor
            Decoded tensor.
        """
        x = torch.cat([x1, x2], dim=1)
        if self.res:
            x = F.leaky_relu(
                self.in_0(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace
            )
            skip = self.conv0(x)
            x = skip
        else:
            x = F.leaky_relu(
                self.in_0(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace
            )
            x = self.conv0(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = F.leaky_relu(self.in_1(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv1(x)
        if self.res:
            x = x + skip
        return x


class OutConv(nn.Module):
    """BrainMaGe output convolution module."""

    def __init__(self, input_channels: int, output_channels: int, res: bool = False) -> None:
        """Initialize output convolution.

        Parameters
        ----------
        input_channels
            Input channel count after concatenation.
        output_channels
            Output channel count.
        res
            Whether to use the source residual path.
        """
        super().__init__()
        self.res = res
        self.dropout_p = 0.3
        self.leakiness = 1e-2
        self.lrelu_inplace = True
        self.dropout = nn.Dropout3d(self.dropout_p)
        self.in_0 = nn.InstanceNorm3d(input_channels, affine=True, track_running_stats=True)
        self.in_1 = nn.InstanceNorm3d(16, affine=True, track_running_stats=True)
        self.conv0 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, output_channels, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Run output convolution with the first skip tensor.

        Parameters
        ----------
        x1
            Upsampled tensor.
        x2
            Skip tensor.

        Returns
        -------
        torch.Tensor
            Sigmoid segmentation tensor.
        """
        x = torch.cat([x1, x2], dim=1)
        if self.res:
            x = F.leaky_relu(
                self.in_0(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace
            )
            skip = self.conv0(x)
            x = skip
        else:
            x = F.leaky_relu(
                self.in_0(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace
            )
            x = self.conv0(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = F.leaky_relu(self.in_1(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv1(x)
        if self.res:
            x = x + skip
        x = self.conv2(x)
        return torch.sigmoid(x)


class ResUNet(nn.Module):
    """BrainMaGe ResUNet segmentation network."""

    def __init__(self, n_channels: int, n_classes: int, base_filters: int = 16) -> None:
        """Initialize BrainMaGe ResUNet.

        Parameters
        ----------
        n_channels
            Input channel count.
        n_classes
            Number of source classes including background.
        base_filters
            Base filter count; source configs use 16.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.ins = InConv(self.n_channels, base_filters, res=True)
        self.ds_0 = DownsamplingModule(base_filters, base_filters * 2)
        self.en_1 = EncodingModule(base_filters * 2, base_filters * 2, res=True)
        self.ds_1 = DownsamplingModule(base_filters * 2, base_filters * 4)
        self.en_2 = EncodingModule(base_filters * 4, base_filters * 4, res=True)
        self.ds_2 = DownsamplingModule(base_filters * 4, base_filters * 8)
        self.en_3 = EncodingModule(base_filters * 8, base_filters * 8, res=True)
        self.ds_3 = DownsamplingModule(base_filters * 8, base_filters * 16)
        self.en_4 = EncodingModule(base_filters * 16, base_filters * 16, res=True)
        self.us_3 = UpsamplingModule(base_filters * 16, base_filters * 8)
        self.de_3 = DecodingModule(base_filters * 16, base_filters * 8, res=True)
        self.us_2 = UpsamplingModule(base_filters * 8, base_filters * 4)
        self.de_2 = DecodingModule(base_filters * 8, base_filters * 4, res=True)
        self.us_1 = UpsamplingModule(base_filters * 4, base_filters * 2)
        self.de_1 = DecodingModule(base_filters * 4, base_filters * 2, res=True)
        self.us_0 = UpsamplingModule(base_filters * 2, base_filters)
        self.out = OutConv(base_filters * 2, self.n_classes - 1, res=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run BrainMaGe ResUNet.

        Parameters
        ----------
        x
            Input 3D image tensor.

        Returns
        -------
        torch.Tensor
            Segmentation probabilities.
        """
        x1 = self.ins(x)
        x2 = self.ds_0(x1)
        x2 = self.en_1(x2)
        x3 = self.ds_1(x2)
        x3 = self.en_2(x3)
        x4 = self.ds_2(x3)
        x4 = self.en_3(x4)
        x5 = self.ds_3(x4)
        x5 = self.en_4(x5)

        x = self.us_3(x5)
        x = self.de_3(x, x4)
        x = self.us_2(x)
        x = self.de_2(x, x3)
        x = self.us_1(x)
        x = self.de_1(x, x2)
        x = self.us_0(x)
        return self.out(x, x1)


class BTSbotUMCNN(nn.Module):
    """BTSbot unimodal CNN architecture."""

    def __init__(self, config: dict[str, int | float]) -> None:
        """Initialize the BTSbot source CNN.

        Parameters
        ----------
        config
            Source-style architecture configuration.
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                3,
                int(config["conv1_channels"]),
                kernel_size=int(config["conv_kernel"]),
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                int(config["conv1_channels"]),
                int(config["conv1_channels"]),
                kernel_size=int(config["conv_kernel"]),
                padding="same",
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(float(config["conv_dropout1"])),
            nn.Conv2d(
                int(config["conv1_channels"]),
                int(config["conv2_channels"]),
                kernel_size=int(config["conv_kernel"]),
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                int(config["conv2_channels"]),
                int(config["conv2_channels"]),
                kernel_size=int(config["conv_kernel"]),
                padding="same",
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout2d(float(config["conv_dropout2"])),
            nn.Flatten(),
        )

        conv_feature_dim = (
            int(config["conv2_channels"]) * (int(config.get("image_size", 63)) // 8) ** 2
        )
        self.head = nn.Sequential(
            nn.Linear(conv_feature_dim, int(config["fc1_neurons"])),
            nn.ReLU(),
            nn.Linear(int(config["fc1_neurons"]), int(config["fc2_neurons"])),
            nn.ReLU(),
            nn.Dropout(float(config["dropout"])),
            nn.Linear(int(config["fc2_neurons"]), 1),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Run BTSbot CNN inference.

        Parameters
        ----------
        input_data
            Three-channel image tensor.

        Returns
        -------
        torch.Tensor
            Binary classification logit.
        """
        features = self.conv_layers(input_data)
        return self.head(features)


def capsule_softmax(input_tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Run the source capsule softmax helper.

    Parameters
    ----------
    input_tensor
        Input tensor.
    dim
        Dimension to normalize.

    Returns
    -------
    torch.Tensor
        Softmax-normalized tensor.
    """
    transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
    softmaxed_output = F.softmax(
        transposed_input.contiguous().view(-1, transposed_input.size(-1)),
        dim=-1,
    )
    return softmaxed_output.view(*transposed_input.size()).transpose(
        dim,
        len(input_tensor.size()) - 1,
    )


class CapsuleLayer(nn.Module):
    """Gram.AI dynamic-routing capsule layer."""

    def __init__(
        self,
        num_capsules: int,
        num_route_nodes: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | None = None,
        stride: int | None = None,
        num_iterations: int = 3,
    ) -> None:
        """Initialize a capsule layer.

        Parameters
        ----------
        num_capsules
            Number of capsules.
        num_route_nodes
            Number of route nodes, or -1 for convolutional capsules.
        in_channels
            Input channel count.
        out_channels
            Output channel count.
        kernel_size
            Convolution kernel size for primary capsules.
        stride
            Convolution stride for primary capsules.
        num_iterations
            Dynamic routing iteration count.
        """
        super().__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(
                torch.randn(num_capsules, num_route_nodes, in_channels, out_channels),
            )
        else:
            self.capsules = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=0,
                    )
                    for _ in range(num_capsules)
                ],
            )

    def squash(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Apply capsule squash nonlinearity.

        Parameters
        ----------
        tensor
            Tensor to squash.
        dim
            Dimension over which to compute norm.

        Returns
        -------
        torch.Tensor
            Squashed tensor.
        """
        squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run capsule routing or primary convolution capsules.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Capsule outputs.
        """
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = torch.zeros(*priors.size(), device=x.device, dtype=x.dtype)
            for i in range(self.num_iterations):
                probs = capsule_softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        return outputs


class CapsuleNet(nn.Module):
    """Gram.AI dynamic-routing CapsuleNet."""

    def __init__(self) -> None:
        """Initialize CapsuleNet."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(
            num_capsules=8,
            num_route_nodes=-1,
            in_channels=256,
            out_channels=32,
            kernel_size=9,
            stride=2,
        )
        self.digit_capsules = CapsuleLayer(
            num_capsules=10,
            num_route_nodes=32 * 6 * 6,
            in_channels=8,
            out_channels=16,
        )
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run CapsuleNet classification and reconstruction.

        Parameters
        ----------
        x
            MNIST-like image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Class probabilities and reconstruction.
        """
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        classes = (x**2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        _, max_length_indices = classes.max(dim=1)
        y = torch.eye(10, device=x.device, dtype=x.dtype).index_select(
            dim=0,
            index=max_length_indices.data,
        )
        reconstructions = self.decoder((x * y[:, :, None]).contiguous().view(x.size(0), -1))
        return classes, reconstructions


def _identity_state(x: torch.Tensor) -> torch.Tensor:
    """Return the identity system-dynamics tensor.

    Parameters
    ----------
    x
        State tensor.

    Returns
    -------
    torch.Tensor
        Unchanged state tensor.
    """
    return x


def build_braingate_kalmannet() -> KalmanNetNN:
    """Build the tiny trace-gated KalmanNet entry.

    Returns
    -------
    KalmanNetNN
        Initialized KalmanNet module.
    """
    sys_model = SimpleNamespace(
        f=_identity_state,
        h=_identity_state,
        m=2,
        n=2,
        prior_Q=torch.eye(2),
        prior_Sigma=torch.eye(2),
        prior_S=torch.eye(2),
    )
    args = SimpleNamespace(use_cuda=False, n_batch=1, in_mult_KNet=2, out_mult_KNet=2)
    model = KalmanNetNN()
    model.NNBuild(sys_model, args)
    model.InitSequence(torch.zeros(1, 2, 1), 1)
    model.init_hidden_KNet()
    return model


def example_input_braingate_kalmannet() -> torch.Tensor:
    """Return a tiny KalmanNet observation tensor.

    Returns
    -------
    torch.Tensor
        Example observation tensor.
    """
    return torch.randn(1, 2, 1)


def build_brainnetcnn() -> BrainNetCNN:
    """Build the tiny trace-gated BrainNetCNN entry.

    Returns
    -------
    BrainNetCNN
        BrainNetCNN module.
    """
    return BrainNetCNN(torch.zeros(1, 1, 8, 8)).eval()


def example_input_brainnetcnn() -> torch.Tensor:
    """Return a tiny BrainNetCNN connectivity tensor.

    Returns
    -------
    torch.Tensor
        Example connectivity tensor.
    """
    return torch.randn(1, 1, 8, 8)


def build_brainmage_resunet() -> ResUNet:
    """Build the trace-gated BrainMaGe ResUNet entry.

    Returns
    -------
    ResUNet
        BrainMaGe ResUNet module.
    """
    return ResUNet(1, 2, 16).eval()


def example_input_brainmage_resunet() -> torch.Tensor:
    """Return a small 3D image tensor.

    Returns
    -------
    torch.Tensor
        Example 3D image tensor.
    """
    return torch.randn(1, 1, 32, 32, 32)


def build_btsbot_um_cnn() -> BTSbotUMCNN:
    """Build the trace-gated BTSbot unimodal CNN entry.

    Returns
    -------
    BTSbotUMCNN
        BTSbot CNN module.
    """
    config = {
        "conv1_channels": 4,
        "conv2_channels": 8,
        "conv_kernel": 3,
        "conv_dropout1": 0.0,
        "conv_dropout2": 0.0,
        "image_size": 64,
        "fc1_neurons": 16,
        "fc2_neurons": 8,
        "dropout": 0.0,
    }
    return BTSbotUMCNN(config).eval()


def example_input_btsbot_um_cnn() -> torch.Tensor:
    """Return a tiny BTSbot image tensor.

    Returns
    -------
    torch.Tensor
        Example three-channel image tensor.
    """
    return torch.randn(1, 3, 64, 64)


def build_capsule_networks_part_whole() -> CapsuleNet:
    """Build the trace-gated dynamic-routing CapsuleNet entry.

    Returns
    -------
    CapsuleNet
        CapsuleNet module.
    """
    return CapsuleNet().eval()


def example_input_capsule_networks_part_whole() -> torch.Tensor:
    """Return a small CapsuleNet batch.

    Returns
    -------
    torch.Tensor
        Example MNIST-like image batch.
    """
    return torch.randn(2, 1, 28, 28)


MENAGERIE_ENTRIES = [
    (
        "BrainGate KalmanNet",
        "build_braingate_kalmannet",
        "example_input_braingate_kalmannet",
        2021,
        "CV2d",
    ),
    ("BrainNetCNN", "build_brainnetcnn", "example_input_brainnetcnn", 2017, "CV2d"),
    ("BrainMaGe", "build_brainmage_resunet", "example_input_brainmage_resunet", 2021, "CV2d"),
    ("BTSbot", "build_btsbot_um_cnn", "example_input_btsbot_um_cnn", 2024, "CV2d"),
    (
        "Capsule models of ventral-stream part-whole coding",
        "build_capsule_networks_part_whole",
        "example_input_capsule_networks_part_whole",
        2017,
        "CV2d",
    ),
    (
        "Capsule networks for cortical grouping",
        "build_capsule_networks_part_whole",
        "example_input_capsule_networks_part_whole",
        2017,
        "CV2d",
    ),
]
