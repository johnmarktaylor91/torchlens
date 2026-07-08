# FAITHFUL PORT of zach-gousseau/sifnet_public @ main (original framework: TensorFlow/Keras)
#
# Sea Ice Forecasting Network (SIFNet), "Spatial Feature Pyramid, Hidden
# State" variant -- the model the repo's own `ice_presence_experiment.py`
# docstring calls "the suggested model without future channel"
# (`spatial_feature_pyramid_net_hiddenstate_ND` in
# sifnet/medium_term_ice_forecasting/ice_presence/model.py, attributed to
# Matthew King, NRC, Nov 2019). Gousseau et al., "A Convolutional Neural
# Network for Sea Ice Concentration Forecasting", Cryosphere 2022.
#
# Upstream is a `tensorflow.keras` functional-API model; this repo's declared
# base env for this task does not include TensorFlow, so this is a faithful,
# op-for-op PyTorch transcription of the real Keras code in:
#   - ice_presence/model.py: spatial_feature_pyramid_net_hiddenstate_ND()
#   - utilities/model_utilities.py: spatial_feature_pyramid(),
#     res_step_decoder_HS_functional()
#
# Every mechanism in the real graph has a 1:1 counterpart here:
#   - spatial_feature_pyramid: TimeDistributed(Conv2D) base + a log2(8)=3-level
#     average-pool/Conv2D pyramid, then a top-down Conv2DTranspose "upflow"
#     refinement identical to an FPN top-down pathway (Lin et al. 2017), all
#     applied per-timestep (encoded independently per input day).
#   - ConvLSTM2D: Keras's default gate/cell equations (activation='tanh',
#     recurrent_activation='hard_sigmoid', unit_forget_bias=True) -- the
#     standard Shi et al. 2015 convolutional LSTM cell, run non-stateful,
#     return_sequences=False (only the final hidden state is kept).
#   - res_step_decoder_HS_functional: a residual step-decoder that
#     extrapolates `output_steps` (default 30) daily states from a single
#     encoded state using two SeparableConv2D-based residual predictors (one
#     for the visible state, one for a parallel hidden state), each step
#     conditioned ("anchored") on the original encoded state via
#     channel-concat, exactly as upstream's per-step Python loop.
#   - Final TimeDistributed 1x1-conv "network-in-a-network" stack
#     (48->32->16 LeakyReLU, then 8->1 sigmoid) matches the real head.
#
# Defaults mirror train.py / model.py kwargs: input_shape=(3,160,300,8)
# (days=3, H=160, W=300, channels=8), output_steps=30, l2reg=0.001 (unused
# here -- L2 weight regularization changes the loss, not the forward
# architecture, so it is faithfully omitted from a pure forward-pass port),
# leaky_relu_alpha=0.01. The staging example below uses a much smaller
# spatial/temporal size (kept architecturally identical) so tracing is fast.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hard_sigmoid(x):
    # Keras's default ConvLSTM2D recurrent_activation: clip(0.2*x + 0.5, 0, 1)
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class TimeDistributedConv2D(nn.Module):
    """kl.TimeDistributed(kl.Conv2D(...)) -- apply the same 2D conv
    independently to every timestep of a (B, T, C, H, W) tensor."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same"):
        super().__init__()
        if padding == "same":
            padding = (
                kernel_size // 2
                if isinstance(kernel_size, int)
                else tuple(k // 2 for k in kernel_size)
            )
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b * t, *x.shape[2:])
        x = self.conv(x)
        return x.reshape(b, t, *x.shape[1:])


class TimeDistributedConvTranspose2D(nn.Module):
    """kl.TimeDistributed(kl.Conv2DTranspose(filters, (4,4), (2,2),
    padding='same')) -- Keras 'same' Conv2DTranspose with stride 2 and
    kernel 4 exactly doubles the spatial size; PyTorch's ConvTranspose2d
    with padding=1 (== (kernel-stride)//2 for k=4,s=2) matches that."""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2):
        super().__init__()
        padding = (kernel_size - stride) // 2
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b * t, *x.shape[2:])
        x = self.conv(x)
        return x.reshape(b, t, *x.shape[1:])


class TimeDistributedAvgPool2D(nn.Module):
    """kl.TimeDistributed(kl.AveragePooling2D((k,k), strides=(k,k),
    padding='same'))."""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b * t, *x.shape[2:])
        # padding='same' with stride==kernel in Keras just pads up to a
        # multiple of the factor before pooling (ceil_mode reproduces this
        # for the common case where H, W are already multiples of factor,
        # which holds for our staging example sizes).
        x = F.avg_pool2d(x, kernel_size=self.factor, stride=self.factor, ceil_mode=True)
        return x.reshape(b, t, *x.shape[1:])


class SpatialFeaturePyramid(nn.Module):
    """spatial_feature_pyramid() in model_utilities.py -- per-timestep
    multi-scale feature extractor with an FPN-style top-down refinement
    pass. Operates on channels-first (B, T, C, H, W) tensors."""

    def __init__(self, in_channels, full_res_features, kernel_size, max_downsampling_factor, alpha):
        super().__init__()
        self.alpha = alpha
        self.full_res_features = full_res_features

        self.base_conv = TimeDistributedConv2D(in_channels, full_res_features, kernel_size)

        exp2 = int(torch.log2(torch.tensor(float(max_downsampling_factor))).item())
        self.downsampling_factors = [2**x for x in range(1, exp2 + 1)]

        self.downsamplers = nn.ModuleList(
            [TimeDistributedAvgPool2D(f) for f in self.downsampling_factors]
        )
        self.downsample_convs = nn.ModuleList(
            [
                TimeDistributedConv2D(in_channels, max(1, full_res_features // f), kernel_size)
                for f in self.downsampling_factors
            ]
        )

        # Top-down refinement (index: level 1..len(feature_maps)-1, matching
        # upstream's `for level in range(len(feature_maps)-1, 0, -1)`).
        n_levels = 1 + len(self.downsampling_factors)
        self.upflow_convs = nn.ModuleList()
        for level in range(n_levels - 1, 0, -1):
            features_at_n = max(1, full_res_features // (2 ** (level - 1)))
            # in_channels of feature_maps[level] before this refinement step:
            in_ch = max(1, full_res_features // (2**level))
            self.upflow_convs.append(
                TimeDistributedConvTranspose2D(in_ch, features_at_n, kernel_size=4, stride=2)
            )

    def forward(self, x):
        # x: (B, T, C, H, W)
        base = F.leaky_relu(self.base_conv(x), self.alpha)
        feature_maps = [base]

        for downsampler, conv in zip(self.downsamplers, self.downsample_convs):
            down = downsampler(x)
            down = conv(down)
            down = F.leaky_relu(down, self.alpha)
            feature_maps.append(down)

        upflow_idx = 0
        for level in range(len(feature_maps) - 1, 0, -1):
            f = feature_maps[level]
            n = feature_maps[level - 1]

            fp = self.upflow_convs[upflow_idx](f)
            upflow_idx += 1
            fp = F.leaky_relu(fp, self.alpha)

            # Crop fp down to n's spatial size if AveragePooling2D's 'same'
            # padding produced an odd-sized downsample (matches upstream's
            # Cropping3D fallback).
            if fp.shape[-2:] != n.shape[-2:]:
                fp = fp[..., : n.shape[-2], : n.shape[-1]]

            n = n + fp
            feature_maps[level - 1] = n

        return feature_maps[0]  # full-resolution only (return_all=False)


class ConvLSTM2DCell(nn.Module):
    """A single Keras-semantics ConvLSTM2D step: input-to-state and
    state-to-state convolutions, standard LSTM gate equations with
    activation='tanh' and recurrent_activation='hard_sigmoid'."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        # Combined conv producing the 4 gates (i, f, c, o) at once.
        self.conv_x = nn.Conv2d(
            in_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=True
        )
        self.conv_h = nn.Conv2d(
            hidden_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=False
        )

    def forward(self, x_t, h_prev, c_prev):
        gates = self.conv_x(x_t) + self.conv_h(h_prev)
        i, f, c_tilde, o = torch.split(gates, self.hidden_channels, dim=1)
        i = _hard_sigmoid(i)
        # unit_forget_bias=True: Keras initializes the forget gate bias to 1,
        # a training-time init detail with no effect on this random-init
        # forward-pass architecture (bias values are still learnable params).
        f = _hard_sigmoid(f)
        o = _hard_sigmoid(o)
        c_tilde = torch.tanh(c_tilde)
        c = f * c_prev + i * c_tilde
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM2D(nn.Module):
    """kl.ConvLSTM2D(filters, kernel_size, padding='same', activation='selu',
    return_sequences=False) as used for the 'State_Encoder' layer.

    Note: upstream passes activation='selu' to ConvLSTM2D, but Keras's
    ConvLSTM2D only ever applies `activation` to the cell-state candidate
    in place of tanh when the layer's `activation` kwarg is set -- so here
    we honor that: the cell-candidate nonlinearity is SELU (matching the
    call `activation='selu'`), while the recurrent (gate) activation stays
    hard_sigmoid (the Keras default, since `recurrent_activation` is not
    overridden in the call)."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv_x = nn.Conv2d(
            in_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=True
        )
        self.conv_h = nn.Conv2d(
            hidden_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=False
        )

    def forward(self, x):
        # x: (B, T, C, H, W) -> returns final hidden state (B, hidden, H, W)
        b, t, _, h, w = x.shape
        h_t = x.new_zeros(b, self.hidden_channels, h, w)
        c_t = x.new_zeros(b, self.hidden_channels, h, w)
        for step in range(t):
            gates = self.conv_x(x[:, step]) + self.conv_h(h_t)
            i, f, c_tilde, o = torch.split(gates, self.hidden_channels, dim=1)
            i = _hard_sigmoid(i)
            f = _hard_sigmoid(f)
            o = _hard_sigmoid(o)
            c_tilde = F.selu(c_tilde)
            c_t = f * c_t + i * c_tilde
            h_t = o * F.selu(c_t)
        return h_t


class SeparableConv2DBlock(nn.Module):
    """kl.SeparableConv2D(filters, kernel_size, padding='same',
    depth_multiplier=depth_multiplier) -- depthwise conv (with the given
    depth multiplier) followed by a 1x1 pointwise conv, exactly matching
    Keras's SeparableConv2D decomposition."""

    def __init__(self, in_channels, out_channels, kernel_size=3, depth_multiplier=2):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels * depth_multiplier, out_channels, kernel_size=1, bias=True
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResStepDecoderHS(nn.Module):
    """res_step_decoder_HS_functional() in model_utilities.py -- residual
    step-decoder with an auxiliary hidden state, `anchored=True` (each
    step's residual predictors see the original encoded state via
    channel-concat, matching the upstream default call)."""

    def __init__(
        self,
        in_channels,
        filters,
        hidden_filters,
        upsampled_filters,
        output_steps,
        kernel_size=3,
        depth_multiplier=2,
        alpha=0.03,
    ):
        super().__init__()
        self.output_steps = output_steps
        self.alpha = alpha

        self.initial_state_conv = SeparableConv2DBlock(
            in_channels, filters, kernel_size, depth_multiplier
        )
        self.hidden_state_conv = SeparableConv2DBlock(
            in_channels, hidden_filters, kernel_size, depth_multiplier
        )

        self.upsampler = nn.Conv2d(filters, upsampled_filters, kernel_size=1)
        self.hidden_upsampler = nn.Conv2d(hidden_filters, upsampled_filters, kernel_size=1)

        combined_channels = upsampled_filters + in_channels + upsampled_filters
        self.res_pred = SeparableConv2DBlock(
            combined_channels, filters, kernel_size, depth_multiplier
        )
        self.hidden_res_pred = SeparableConv2DBlock(
            combined_channels, hidden_filters, kernel_size, depth_multiplier
        )

    def forward(self, encoded_state):
        initial_state = F.leaky_relu(self.initial_state_conv(encoded_state), self.alpha)
        hidden_state = F.leaky_relu(self.hidden_state_conv(encoded_state), self.alpha)

        daily_states = []
        incoming_state = initial_state
        for _ in range(self.output_steps):
            upsampled = F.leaky_relu(self.upsampler(incoming_state), self.alpha)
            hidden_upsampled = self.hidden_upsampler(hidden_state)

            combined_state = torch.cat([upsampled, encoded_state, hidden_upsampled], dim=1)
            residual = self.res_pred(combined_state)
            hidden_residual = self.hidden_res_pred(combined_state)

            next_state = incoming_state + residual
            hidden_state = hidden_state + hidden_residual

            daily_states.append(next_state)
            incoming_state = next_state

        # return_sequence=True: stack along the time axis.
        return torch.stack(daily_states, dim=1)  # (B, output_steps, filters, H, W)


class SpatialFeaturePyramidHiddenStateND(nn.Module):
    """spatial_feature_pyramid_net_hiddenstate_ND() in
    ice_presence/model.py -- the repo's suggested model for ice-presence
    forecasting without future forcing channels."""

    def __init__(self, input_shape=(3, 160, 300, 8), output_steps=30, alpha=0.01):
        super().__init__()
        days_in, height, width, channels = input_shape
        self.days_in = days_in
        self.output_steps = output_steps
        self.alpha = alpha

        n_features = 24
        self.pyramid = SpatialFeaturePyramid(
            in_channels=channels,
            full_res_features=n_features,
            kernel_size=3,
            max_downsampling_factor=8,
            alpha=alpha,
        )

        encoder_hidden = 48 - channels
        self.state_encoder = ConvLSTM2D(
            in_channels=n_features, hidden_channels=encoder_hidden, kernel_size=3
        )

        encoded_channels = encoder_hidden + channels
        self.decoder = ResStepDecoderHS(
            in_channels=encoded_channels,
            filters=16,
            hidden_filters=16,
            upsampled_filters=48,
            output_steps=output_steps,
            kernel_size=3,
            depth_multiplier=2,
            alpha=alpha,
        )

        self.nin1 = TimeDistributedConv2D(16, 48, 1)
        self.nin2 = TimeDistributedConv2D(48, 32, 1)
        self.nin3 = TimeDistributedConv2D(32, 16, 1)
        self.sigmoid_pre_out = TimeDistributedConv2D(16, 8, 1)
        self.sigmoid_out = TimeDistributedConv2D(8, 1, 1)

    def forward(self, x):
        # x: (B, days_in, C, H, W) channels-first equivalent of Keras's
        # (B, days_in, H, W, C) `input_shape`.
        full_res_map = self.pyramid(x)
        encoded_state = self.state_encoder(full_res_map)

        input_last_day_only = x[:, -1]  # Cropping3D((days_in-1,0),...) + squeeze time axis
        encoded_state = torch.cat([encoded_state, input_last_day_only], dim=1)

        out = self.decoder(encoded_state)

        out = F.leaky_relu(self.nin1(out), self.alpha)
        out = F.leaky_relu(self.nin2(out), self.alpha)
        out = F.leaky_relu(self.nin3(out), self.alpha)
        out = torch.sigmoid(self.sigmoid_pre_out(out))
        out = torch.sigmoid(self.sigmoid_out(out))
        return out


MENAGERIE_ZOO = "ported-pytorch"


def build_sifnet_spatial_feature_pyramid_hs():
    # Small staging size: days_in=2, H=16, W=16, channels=4, output_steps=2
    # (architecturally identical to the real default (3,160,300,8)/30 -- just
    # scaled down so the per-step decoder loop and pyramid trace quickly).
    return SpatialFeaturePyramidHiddenStateND(
        input_shape=(2, 16, 16, 4), output_steps=2, alpha=0.01
    )


def example_input_sifnet_spatial_feature_pyramid_hs():
    return (torch.randn(1, 2, 4, 16, 16),)


MENAGERIE_ENTRIES = [
    (
        "SIFNet (spatial feature pyramid, hidden-state decoder)",
        "build_sifnet_spatial_feature_pyramid_hs",
        "example_input_sifnet_spatial_feature_pyramid_hs",
        2022,
        "ported-pytorch",
    ),
]
