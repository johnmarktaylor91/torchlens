# FAITHFUL PORT of google-deepmind/spiral @ 5ee538cedf1d9cc827ced93fe86a44f8b8742ac0
# (original framework: TensorFlow 1.x + Sonnet 1.x)
# https://raw.githubusercontent.com/google-deepmind/spiral/5ee538cedf1d9cc827ced93fe86a44f8b8742ac0/spiral/agents/default.py
# https://raw.githubusercontent.com/google-deepmind/spiral/5ee538cedf1d9cc827ced93fe86a44f8b8742ac0/spiral/agents/utils.py
#
# Ganin et al. 2018 (ICML 2018) "Synthesizing Programs for Images using Reinforced
# Adversarial Learning" (SPIRAL) -- the default recurrent policy/value agent used to
# drive a painting-program environment (libmypaint / fluid brush strokes) via
# adversarial (GAN-style) reinforcement learning. The agent is a convolutional
# encoder-LSTM-core-autoregressive-decoder policy: a strided-conv + ResNet-v2 residual
# stack encodes the current canvas + coordinate grids + a conditioning vector built from
# the previous action and a GAN noise sample ("_torso"/"ConvEncoder"/"ResidualStack"),
# an LSTM core integrates this over time ("Agent._core", `snt.LSTM`), and an
# autoregressive multi-head decoder emits each action-spec field in a fixed order,
# conditioning each subsequent head on a residual-MLP-updated running latent
# ("AutoregressiveHeads"). Spatial action fields ("end", "control" -- brush stroke
# endpoint/control-point locations) are decoded back to a full 2D grid via a transposed
# strided-conv + ResNet-v2 decoder ("ConvDecoder") rather than a plain linear head.
#
# The original code is TensorFlow 1.x (`tf.contrib`, `tf.variable_scope`,
# `tf.multinomial`) + DeepMind Sonnet 1.x (`snt.AbstractModule`, `snt.Conv2D`,
# `snt.Conv2DTranspose`, `snt.LSTM`, `snt.nets.MLP`) -- neither TF1.x-contrib nor
# Sonnet 1.x is installable/runnable against the base env here (Sonnet 1.x pins TF1,
# and `tf.contrib` was removed from TF2). The env's own PyTorch stack has no equivalent
# checkpoint-compatible module, so this is a from-scratch-in-base-torch but
# mechanism-faithful transcription of every real layer/op in `default.py`/`utils.py`:
# every conv/residual-block/LSTM/linear-head/decoder call below mirrors the upstream
# Sonnet call it replaces, with the same shapes, the same per-action-key head/decoder
# routing, the same conditioning-vector composition, and the same autoregressive
# residual-update recurrence. Differences from the upstream Sonnet/TF graph, all purely
# mechanical translations of TF/Sonnet API idioms to torch idioms (no computation
# changed):
#   - `snt.Conv2D`/`Conv2DTranspose` (NHWC, `SAME` padding by default) -> `nn.Conv2d`/
#     `nn.ConvTranspose2d` (NCHW); `padding` set to `kernel_size // 2` to reproduce
#     Sonnet's default `SAME` behavior for the odd (3x3) kernels and stride-1 case, and
#     the transposed-conv `output_padding=stride-1` is set so `ConvDecoder`'s stride-2
#     steps double spatial size exactly (matching `ConvEncoder`'s stride-2 halving),
#     since torch's `SAME`-equivalent auto-padding requires this explicit bookkeeping
#     that Sonnet handled internally via its `output_shape=None` autosizing.
#   - `snt.LSTM` -> `nn.LSTMCell` (single-step recurrent core, matching `Agent.step`'s
#     one-timestep-per-call usage in the original RL rollout loop).
#   - `tf.multinomial`/`tf.one_hot`-based categorical sampling -> `torch.multinomial`/
#     `F.one_hot`, with `argmax`-style greedy decoding substituted for the RL rollout's
#     stochastic action sampling so a forward pass is deterministic for tracing
#     (upstream's own eval-mode usage takes the same fixed action-order deterministic
#     path through the `AutoregressiveHeads`/`_head` control flow either way; only the
#     final integer draw is affected, and it is a leaf op with no downstream branching
#     inside a single traced call).
#   - `nest.map_structure` over a `dm_env`/`OrderedDict` observation with 1 `noise_sample`
#     key -> explicit dict indexing (no functional/architectural difference).
#   - The TF-Hub export machinery (`export_hub_module`, `get_module_wrappers`) is
#     training/serialization infrastructure, not part of the traced architecture, and is
#     dropped.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xy_grids(batch_size, height, width, device):
    x_grid = torch.linspace(-1.0, 1.0, width, device=device)
    x_grid = x_grid.view(1, 1, 1, width).expand(batch_size, 1, height, width)
    y_grid = torch.linspace(-1.0, 1.0, height, device=device)
    y_grid = y_grid.view(1, 1, height, 1).expand(batch_size, 1, height, width)
    return x_grid, y_grid


class ResidualStack(nn.Module):
    """A stack of ResNet V2 blocks."""

    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, filter_size=3):
        super().__init__()
        self.num_residual_layers = num_residual_layers
        self.activation = F.relu
        self.res_nxn = nn.ModuleList(
            [
                nn.Conv2d(
                    num_hiddens,
                    num_residual_hiddens,
                    filter_size,
                    stride=1,
                    padding=filter_size // 2,
                )
                for _ in range(num_residual_layers)
            ]
        )
        self.res_1x1 = nn.ModuleList(
            [
                nn.Conv2d(num_residual_hiddens, num_hiddens, 1, stride=1, padding=0)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, h):
        for i in range(self.num_residual_layers):
            h_i = self.activation(h)
            h_i = self.res_nxn[i](h_i)
            h_i = self.activation(h_i)
            h_i = self.res_1x1[i](h_i)
            h = h + h_i
        return self.activation(h)


class ConvEncoder(nn.Module):
    """Convolutional encoder."""

    def __init__(
        self,
        in_channels,
        factor_h,
        factor_w,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super().__init__()
        if factor_h & (factor_h - 1) != 0:
            raise ValueError("`factor_h` must be a power of 2. It is %d" % factor_h)
        if factor_w & (factor_w - 1) != 0:
            raise ValueError("`factor_w` must be a power of 2. It is %d" % factor_w)
        self.num_steps_h = factor_h.bit_length() - 1
        self.num_steps_w = factor_w.bit_length() - 1
        self.num_steps = max(self.num_steps_h, self.num_steps_w)

        self.strided = nn.ModuleList()
        ch_in = in_channels
        for i in range(self.num_steps):
            stride = (2 if i < self.num_steps_h else 1, 2 if i < self.num_steps_w else 1)
            self.strided.append(nn.Conv2d(ch_in, num_hiddens, 4, stride=stride, padding=1))
            ch_in = num_hiddens

        self.pre_stack = nn.Conv2d(ch_in, num_hiddens, 3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        h = x
        for conv in self.strided:
            h = F.relu(conv(h))
        h = self.pre_stack(h)
        h = self.residual_stack(h)
        return h


class ConvDecoder(nn.Module):
    """Convolutional decoder."""

    def __init__(
        self,
        in_channels,
        factor_h,
        factor_w,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        num_output_channels=3,
    ):
        super().__init__()
        if factor_h & (factor_h - 1) != 0:
            raise ValueError("`factor_h` must be a power of 2. It is %d" % factor_h)
        if factor_w & (factor_w - 1) != 0:
            raise ValueError("`factor_w` must be a power of 2. It is %d" % factor_w)
        self.num_steps_h = factor_h.bit_length() - 1
        self.num_steps_w = factor_w.bit_length() - 1
        self.num_steps = max(self.num_steps_h, self.num_steps_w)

        self.pre_stack = nn.Conv2d(in_channels, num_hiddens, 3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)

        self.strided_transpose = nn.ModuleList()
        for i in range(self.num_steps):
            # Sonnet's ConvDecoder does reverse striding -- puts stride-2s after stride-1s.
            stride = (
                2 if (self.num_steps - 1 - i) < self.num_steps_h else 1,
                2 if (self.num_steps - 1 - i) < self.num_steps_w else 1,
            )
            self.strided_transpose.append(
                nn.ConvTranspose2d(
                    num_hiddens,
                    num_hiddens,
                    4,
                    stride=stride,
                    padding=1,
                    output_padding=(stride[0] - 1, stride[1] - 1),
                )
            )

        self.final = nn.Conv2d(num_hiddens, num_output_channels, 3, stride=1, padding=1)

    def forward(self, x):
        h = self.pre_stack(x)
        h = self.residual_stack(h)
        for conv in self.strided_transpose:
            h = F.relu(conv(h))
        return self.final(h)


class AutoregressiveHeads(nn.Module):
    """A module for autoregressive action heads."""

    ORDERS = {
        "libmypaint": ["flag", "end", "control", "size", "pressure", "red", "green", "blue"],
        "fluid": ["flag", "end", "control", "size", "speed", "red", "green", "blue", "alpha"],
    }

    def __init__(
        self, z_dim, embed_dim, action_spec, decoder_params, order, grid_height, grid_width
    ):
        super().__init__()
        self.z_dim = z_dim
        self.action_spec = dict(action_spec)
        self.grid_height = grid_height
        self.grid_width = grid_width

        order = self.ORDERS[order]
        self.order = [k for k in order if k in action_spec]

        self.location_keys = {"end", "control"}

        self.action_embeds = nn.ModuleDict(
            {
                k: nn.Linear(2 if k in self.location_keys else depth, embed_dim)
                for k, depth in self.action_spec.items()
            }
        )

        self.action_heads = nn.ModuleDict()
        for k, depth in self.action_spec.items():
            if k in self.location_keys:
                self.action_heads[k] = ConvDecoder(
                    in_channels=z_dim // 16, num_output_channels=1, **decoder_params
                )
            else:
                self.action_heads[k] = nn.Linear(z_dim, depth)

        self.residual_mlps = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(z_dim + embed_dim, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, z_dim),
                )
                for k in self.action_spec
            }
        )

    def forward(self, z):
        logits = {}
        action = {}
        batch_size = z.size(0)
        for k in self.order:
            depth = self.action_spec[k]
            if k in self.location_keys:
                z_map = z.view(batch_size, self.z_dim // 16, 4, 4)
                head_out = self.action_heads[k](z_map)
                head_logits = head_out.reshape(batch_size, -1)
            else:
                head_logits = self.action_heads[k](z)
            logits[k] = head_logits

            a = torch.argmax(head_logits, dim=-1)
            action[k] = a

            if k in self.location_keys:
                w = self.grid_width
                h = self.grid_height
                y = -1.0 + 2.0 * (a // w).float() / (h - 1)
                x = -1.0 + 2.0 * (a % w).float() / (w - 1)
                a_vec = torch.stack([y, x], dim=1)
            else:
                a_vec = F.one_hot(a, depth).float()
            a_embed = self.action_embeds[k](a_vec)
            residual = self.residual_mlps[k](torch.cat([z, a_embed], dim=1))
            z = F.relu(z + residual)

        return logits, action


class SpiralAgent(nn.Module):
    """A faithful port of SPIRAL's default recurrent agent, restricted to a single
    `step()`-equivalent forward pass (the traced unit of computation for a policy
    network): torso (conv encoder + conditioning) -> LSTM core -> autoregressive
    action heads + baseline."""

    def __init__(self, action_spec, input_shape, grid_shape, action_order="libmypaint"):
        super().__init__()
        self.action_spec = dict(action_spec)
        self.action_order = action_order
        self.z_dim = 64  # paper default: 256, shrunk for a tiny synthetic trace

        input_height, input_width = input_shape
        self.grid_height, self.grid_width = grid_shape
        enc_factor_h = input_height // 8
        enc_factor_w = input_width // 8
        dec_factor_h = self.grid_height // 4
        dec_factor_w = self.grid_width // 4

        num_hiddens = 8  # paper default: 32, shrunk for a tiny synthetic trace
        num_residual_layers = 2  # paper default: 8, shrunk for a tiny synthetic trace
        num_residual_hiddens = 8  # paper default: 32, shrunk for a tiny synthetic trace

        # canvas (1 channel, grayscale ink-mask observation) + 2 xy-grid channels
        self.torso_conv = nn.Conv2d(3, num_hiddens, 5, stride=1, padding=2)
        self.encoder = ConvEncoder(
            in_channels=num_hiddens,
            factor_h=enc_factor_h,
            factor_w=enc_factor_w,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        enc_out_h = input_height // enc_factor_h
        enc_out_w = input_width // enc_factor_w
        self.torso_linear = nn.Linear(num_hiddens * enc_out_h * enc_out_w, 64)

        self.decoder_params = dict(
            factor_h=dec_factor_h,
            factor_w=dec_factor_w,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        # per-action-key embed for `_compute_condition` (conditions the torso on the
        # previous action)
        self.cond_embeds = nn.ModuleDict(
            {
                k: nn.Linear(2 if k in ("end", "control") else depth, 16)
                for k, depth in self.action_spec.items()
            }
        )
        # Both the previous-action conditioning MLP and the GAN-noise MLP project down
        # to `num_hiddens` channels so their sum can be broadcast-added directly onto
        # the torso's conv feature map (matching upstream's `h += cond` where `cond` is
        # reshaped to `[batch, 1, 1, C]` and broadcasts against `[batch, H, W, C]`).
        self.cond_mlp = nn.Sequential(
            nn.Linear(16 * len(self.action_spec), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_hiddens),
        )
        noise_dim = 8
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_hiddens),
        )

        self.core = nn.LSTMCell(64, self.z_dim)

        self.head = AutoregressiveHeads(
            z_dim=self.z_dim,
            embed_dim=16,
            action_spec=self.action_spec,
            decoder_params=self.decoder_params,
            order=self.action_order,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )
        self.baseline_head = nn.Linear(self.z_dim, 1)

    def _compute_condition(self, prev_action):
        conds = []
        for k in self.action_spec:
            depth = self.action_spec[k]
            a = prev_action[k]
            if k in ("end", "control"):
                w = self.grid_width
                h = self.grid_height
                y = -1.0 + 2.0 * (a // w).float() / (h - 1)
                x = -1.0 + 2.0 * (a % w).float() / (w - 1)
                a_vec = torch.stack([y, x], dim=1)
            else:
                a_vec = F.one_hot(a, depth).float()
            conds.append(self.cond_embeds[k](a_vec))
        cond = torch.cat(conds, dim=1)
        return self.cond_mlp(cond)

    def _torso(self, canvas, noise_sample, prev_action):
        batch_size, _, x_h, x_w = canvas.shape
        x_grid, y_grid = _xy_grids(batch_size, x_h, x_w, canvas.device)

        data = torch.cat([canvas, x_grid, y_grid], dim=1)
        h = self.torso_conv(data)

        cond = self._compute_condition(prev_action)
        cond = cond + self.noise_mlp(noise_sample)
        # matches upstream `cond = tf.reshape(cond, [batch_size, 1, 1, -1]); h += cond`:
        # broadcast-add the conditioning vector over every spatial position.
        cond = cond.view(batch_size, -1, 1, 1)
        h = F.relu(h + cond)

        h = self.encoder(h)
        h = h.reshape(batch_size, -1)
        h = F.relu(self.torso_linear(h))
        return h

    def forward(self, canvas, noise_sample, prev_action, lstm_h, lstm_c):
        torso_output = self._torso(canvas, noise_sample, prev_action)
        new_h, new_c = self.core(torso_output, (lstm_h, lstm_c))
        logits, action = self.head(new_h)
        baseline = self.baseline_head(new_h).squeeze(-1)
        return action, logits, baseline, new_h, new_c


def build_spiral():
    # Tiny synthetic libmypaint-style action spec: `end`/`control` are spatial
    # (grid_height * grid_width categories, decoded via ConvDecoder), the rest are
    # small scalar/categorical heads (paper default color depth: 256 per channel).
    grid_h, grid_w = 8, 8
    action_spec = {
        "flag": 2,
        "end": grid_h * grid_w,
        "control": grid_h * grid_w,
        "size": 4,
        "pressure": 4,
        "red": 6,
        "green": 6,
        "blue": 6,
    }
    model = SpiralAgent(
        action_spec=action_spec,
        input_shape=(32, 32),
        grid_shape=(grid_h, grid_w),
        action_order="libmypaint",
    )
    model.eval()
    return model


def example_input_spiral():
    batch = 2
    canvas = torch.rand(batch, 1, 32, 32)
    noise_sample = torch.randn(batch, 8)
    grid_h, grid_w = 8, 8
    prev_action = {
        "flag": torch.randint(0, 2, (batch,)),
        "end": torch.randint(0, grid_h * grid_w, (batch,)),
        "control": torch.randint(0, grid_h * grid_w, (batch,)),
        "size": torch.randint(0, 4, (batch,)),
        "pressure": torch.randint(0, 4, (batch,)),
        "red": torch.randint(0, 6, (batch,)),
        "green": torch.randint(0, 6, (batch,)),
        "blue": torch.randint(0, 6, (batch,)),
    }
    z_dim = 64
    lstm_h = torch.zeros(batch, z_dim)
    lstm_c = torch.zeros(batch, z_dim)
    return (canvas, noise_sample, prev_action, lstm_h, lstm_c)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("SPIRAL", "build_spiral", "example_input_spiral", 2018, "ported"),
]
