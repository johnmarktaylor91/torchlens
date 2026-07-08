# SOURCE: vendored from facebookresearch/nle @ main (nle/agent/agent.py)
#
# NetHackNet is the real self-contained baseline actor-critic policy/value network
# shipped in the NetHack Learning Environment (NLE) reference MonoBeast/torchbeast
# agent (`nle/agent/agent.py`). It fuses a glyph-embedding CNN over the full dungeon
# map, a second CNN over an ego-centric cropped glyph window (via a differentiable
# `grid_sample`-based Crop module), and a small MLP over the game's numeric
# "blstats" (bottom-line stats), then an optional LSTM core feeding policy/baseline
# heads. Vendored verbatim except: (1) the `nethack.MAX_GLYPH` C-extension import is
# replaced with the literal constant (5976, read directly from the compiled NLE
# `_pynethack` extension for nle==1.3.0 -- MAX_GLYPH is not exposed anywhere in pure
# Python source, only inside the game's C glyph table) so this module needs no NLE
# install; (2) the `RandomNet`/training-loop/CLI code around it is dropped, this
# file keeps only the network itself. Everything inside `NetHackNet`/`Crop` is
# unmodified real NLE code (down to the odd double `blstats.view` line and the
# `# TODO ???` comment).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# nle.nethack.MAX_GLYPH for nle==1.3.0 (compiled constant, see header note above).
MAX_GLYPH = 5976


def _step_to_range(delta: float, num_steps: int) -> torch.Tensor:
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)


class Crop(nn.Module):
    """Helper class for NetHackNet below."""

    def __init__(self, height: int, width: int, height_target: int, width_target: int) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[None, :].expand(
            self.height_target, -1
        )
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[:, None].expand(
            -1, self.width_target
        )

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Calculates centered crop around given x,y coordinates.

        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return torch.round(F.grid_sample(inputs, grid, align_corners=True)).squeeze(1).long()


class NetHackNet(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_actions,
        use_lstm,
        embedding_dim=32,
        crop_dim=9,
        num_layers=5,
    ):
        super().__init__()

        self.glyph_shape = observation_shape["glyphs"].shape
        self.blstats_size = observation_shape["blstats"].shape[0]

        self.num_actions = num_actions
        self.use_lstm = use_lstm

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = 512

        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F_ = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F_, F_),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # CNN crop model.
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F_, F_),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        out_dim = self.k_dim
        # CNN over full glyph map
        out_dim += self.H * self.W * Y

        # CNN crop model.
        out_dim += self.crop_dim**2 * Y

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        if self.use_lstm:
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=1)

        self.policy = nn.Linear(self.h_dim, self.num_actions)
        self.baseline = nn.Linear(self.h_dim, 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2)
        )

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs, core_state):
        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]

        # -- [T x B x F]
        blstats = env_outputs["blstats"]

        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # -- [B' x F]
        blstats = blstats.view(T * B, -1).float()

        # -- [B x H x W]
        glyphs = glyphs.long()
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]
        # TODO ???
        # coordinates[:, 0].add_(-1)

        # -- [B x F]
        blstats = blstats.view(T * B, -1).float()
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)

        assert blstats_emb.shape[0] == T * B

        reps = [blstats_emb]

        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)

        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)

        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb)

        # -- [B x K']
        crop_rep = crop_rep.view(T * B, -1)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)

        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st = self.fc(st)

        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~env_outputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = st

        # -- [B x A]
        policy_logits = self.policy(core_output)
        # -- [B x A]
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


class _ShapeStub:
    """Minimal stand-in for a gym.spaces.Box entry: exposes only `.shape`.

    Real NLE observation spaces are `gym.spaces.Dict` of `gym.spaces.Box`; NetHackNet
    only ever reads `.shape` off them, so this staging-only stub avoids requiring the
    `nle`/`gym` packages just to build the network.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def build_nethacknet() -> nn.Module:
    """Build a small NetHackNet at the real NLE default sizes (embedding_dim=32,
    crop_dim=9, num_layers=5), for NetHackScore-v0-shaped observations."""

    observation_shape = {
        "glyphs": _ShapeStub((21, 79)),  # nle DUNGEON_SHAPE
        "blstats": _ShapeStub((26,)),  # nle BLSTATS_SHAPE
    }
    model = NetHackNet(observation_shape, num_actions=23, use_lstm=True)
    model.eval()
    return model


def example_input_nethacknet():
    """Real NetHackNet forward signature: (env_outputs dict, core_state tuple)."""

    T, B = 1, 1
    env_outputs = {
        "glyphs": torch.randint(0, MAX_GLYPH, (T, B, 21, 79)),
        "blstats": torch.randn(T, B, 26),
        "done": torch.zeros(T, B, dtype=torch.bool),
    }
    core_state = (torch.zeros(1, B, 512), torch.zeros(1, B, 512))
    return (env_outputs, core_state)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "NLE NetHackNet (glyph+crop CNN, blstats MLP, LSTM core actor-critic)",
        "build_nethacknet",
        "example_input_nethacknet",
        "2020",
        "DC",
    ),
]
