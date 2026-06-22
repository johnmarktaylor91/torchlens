"""Compact faithful LeRobot policy families.

Paper: LeRobot model zoo; ACT (Zhao et al., 2023), Diffusion Policy (Chi et al.,
2023), VQ-BeT (Lee et al., 2024), TD-MPC (Hansen et al., 2022), pi0/pi0.5
(Black et al., 2024/2025), SmolVLA (Hugging Face, 2025), X-VLA (2025), and
VLA-JEPA (2026).

The real checkpoints are dependency-gated and large.  These compact random-init
reconstructions preserve the distinctive policy primitives used by the LeRobot
entries: ACT's transformer CVAE action chunker, Diffusion Policy's
observation-conditioned denoising UNet, VQ-BeT's discrete behavior tokens plus
offset head, TD-MPC's latent dynamics/value/policy model, pi0-family VLM
conditioning with flow-matching action experts, pi0-FAST's frequency-domain
action tokenization, SmolVLA's lightweight VLM plus non-autoregressive action
expert, X-VLA's embodiment soft prompts, and VLA-JEPA's video-world-model latent
prediction feeding a flow DiT action head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_DIM = 7
STATE_DIM = 8
TEXT_VOCAB = 128


class TinyVisionEncoder(nn.Module):
    """Small patch-style vision encoder used by LeRobot VLA policies."""

    def __init__(self, d_model: int = 48, views: int = 2) -> None:
        """Initialize the multi-view convolutional patch projector.

        Parameters
        ----------
        d_model:
            Width of the visual tokens.
        views:
            Number of camera views concatenated as tokens.
        """

        super().__init__()
        self.views = views
        self.stem = nn.Sequential(
            nn.Conv2d(3, d_model // 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into a short token sequence.

        Parameters
        ----------
        images:
            Tensor of shape ``(batch, views, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Visual tokens of shape ``(batch, views * patches, d_model)``.
        """

        batch, views, channels, height, width = images.shape
        x = images.reshape(batch * views, channels, height, width)
        feat = self.stem(x).flatten(2).transpose(1, 2)
        feat = self.proj(feat[:, :4])
        return feat.reshape(batch, views * feat.shape[1], feat.shape[2])


class TextStateFusion(nn.Module):
    """Fuse language tokens and robot proprioception with visual context."""

    def __init__(self, d_model: int = 48) -> None:
        """Initialize text, state, and CLS projections.

        Parameters
        ----------
        d_model:
            Shared token width.
        """

        super().__init__()
        self.text = nn.Embedding(TEXT_VOCAB, d_model)
        self.state = nn.Linear(STATE_DIM, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=96, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(
        self, image_tokens: torch.Tensor, text_ids: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Return fused context tokens.

        Parameters
        ----------
        image_tokens:
            Encoded camera tokens.
        text_ids:
            Integer instruction tokens.
        state:
            Current robot state vector.

        Returns
        -------
        torch.Tensor
            Fused context sequence.
        """

        batch = image_tokens.shape[0]
        cls = self.cls.expand(batch, -1, -1)
        state_token = self.state(state).unsqueeze(1)
        tokens = torch.cat([cls, image_tokens, self.text(text_ids), state_token], dim=1)
        return self.encoder(tokens)


class ACTPolicy(nn.Module):
    """Action Chunking Transformer with CVAE latent conditioning."""

    def __init__(self, d_model: int = 48, chunk: int = 6, latent_dim: int = 12) -> None:
        """Initialize a compact ACT policy.

        Parameters
        ----------
        d_model:
            Transformer width.
        chunk:
            Number of future actions predicted together.
        latent_dim:
            CVAE latent dimension.
        """

        super().__init__()
        self.vision = TinyVisionEncoder(d_model)
        self.fusion = TextStateFusion(d_model)
        self.action_in = nn.Linear(ACTION_DIM, d_model)
        self.latent_stats = nn.Linear(d_model, 2 * latent_dim)
        self.latent_to_token = nn.Linear(latent_dim, d_model)
        self.query = nn.Parameter(torch.randn(1, chunk, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=96, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, ACTION_DIM)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        text_ids: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict an action chunk from cameras, language, state, and past actions."""

        image_tokens = self.vision(images)
        context = self.fusion(image_tokens, text_ids, state)
        action_summary = self.action_in(past_actions).mean(dim=1)
        stats = self.latent_stats(action_summary)
        mean, logvar = stats.chunk(2, dim=-1)
        z = mean + torch.exp(0.5 * logvar) * torch.tanh(mean)
        memory = context + self.latent_to_token(z).unsqueeze(1)
        query = self.query.expand(images.shape[0], -1, -1)
        return self.head(self.decoder(query, memory))


class ConditionalUnet1D(nn.Module):
    """Observation-conditioned 1-D UNet used by Diffusion Policy."""

    def __init__(self, d_model: int = 48, horizon: int = 8) -> None:
        """Initialize the denoising network.

        Parameters
        ----------
        d_model:
            Conditioning width.
        horizon:
            Action trajectory length.
        """

        super().__init__()
        self.horizon = horizon
        self.down = nn.Conv1d(ACTION_DIM, d_model, kernel_size=3, padding=1)
        self.mid = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.up = nn.Conv1d(d_model, ACTION_DIM, kernel_size=3, padding=1)
        self.cond = nn.Linear(d_model + 1, 2 * d_model)

    def forward(
        self, noisy_actions: torch.Tensor, context: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Denoise an action trajectory with FiLM conditioning."""

        pooled = context[:, 0]
        film = self.cond(torch.cat([pooled, sigma[:, None]], dim=-1))
        scale, shift = film.chunk(2, dim=-1)
        x = self.down(noisy_actions.transpose(1, 2))
        x = F.gelu(x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1))
        skip = x
        x = F.avg_pool1d(x, kernel_size=2)
        x = F.gelu(self.mid(x))
        x = F.interpolate(x, size=self.horizon, mode="linear", align_corners=False)
        x = x + skip
        return self.up(x).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    """LeRobot Diffusion Policy with visual conditioning and denoising steps."""

    def __init__(self, d_model: int = 48, horizon: int = 8) -> None:
        """Initialize a compact diffusion policy."""

        super().__init__()
        self.vision = TinyVisionEncoder(d_model)
        self.fusion = TextStateFusion(d_model)
        self.unet = ConditionalUnet1D(d_model, horizon)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        text_ids: torch.Tensor,
        noisy_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Run a short deterministic denoising chain."""

        context = self.fusion(self.vision(images), text_ids, state)
        actions = noisy_actions
        for sigma_value in (1.0, 0.5, 0.25):
            sigma = actions.new_full((actions.shape[0],), sigma_value)
            actions = actions - sigma[:, None, None] * self.unet(actions, context, sigma)
        return actions


class VQBeTPolicy(nn.Module):
    """Behavior Transformer with vector-quantized action tokens and offsets."""

    def __init__(self, d_model: int = 48, codebook: int = 16, horizon: int = 6) -> None:
        """Initialize a compact VQ-BeT policy."""

        super().__init__()
        self.vision = TinyVisionEncoder(d_model)
        self.fusion = TextStateFusion(d_model)
        self.queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.codes = nn.Parameter(torch.randn(codebook, ACTION_DIM))
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=96, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.logits = nn.Linear(d_model, codebook)
        self.offset = nn.Linear(d_model, ACTION_DIM)

    def forward(
        self, images: torch.Tensor, state: torch.Tensor, text_ids: torch.Tensor
    ) -> torch.Tensor:
        """Select discrete behavior codes and add continuous residual offsets."""

        context = self.fusion(self.vision(images), text_ids, state)
        query = self.queries.expand(images.shape[0], -1, -1)
        hidden = self.decoder(query, context)
        probs = torch.softmax(self.logits(hidden), dim=-1)
        coarse = probs @ self.codes
        return coarse + 0.1 * torch.tanh(self.offset(hidden))


class TDMPCPolicy(nn.Module):
    """TD-MPC style latent dynamics, reward/value, and policy heads."""

    def __init__(self, d_model: int = 48, latent: int = 32, horizon: int = 5) -> None:
        """Initialize compact temporal-difference model predictive control."""

        super().__init__()
        self.horizon = horizon
        self.vision = TinyVisionEncoder(d_model)
        self.fusion = TextStateFusion(d_model)
        self.enc = nn.Linear(d_model, latent)
        self.dynamics = nn.GRUCell(latent + ACTION_DIM, latent)
        self.actor = nn.Linear(latent, ACTION_DIM)
        self.reward = nn.Linear(latent, 1)
        self.value = nn.Linear(latent, 1)

    def forward(
        self, images: torch.Tensor, state: torch.Tensor, text_ids: torch.Tensor
    ) -> torch.Tensor:
        """Roll latent dynamics under the learned actor and score the rollout."""

        context = self.fusion(self.vision(images), text_ids, state)
        latent = torch.tanh(self.enc(context[:, 0]))
        actions = []
        score = latent.new_zeros(latent.shape[0], 1)
        for _ in range(self.horizon):
            action = torch.tanh(self.actor(latent))
            latent = self.dynamics(torch.cat([latent, action], dim=-1), latent)
            score = score + self.reward(latent) + 0.1 * self.value(latent)
            actions.append(action + 0.01 * score)
        return torch.stack(actions, dim=1)


class FlowActionExpert(nn.Module):
    """Non-autoregressive transformer/DiT action-flow head."""

    def __init__(self, d_model: int = 48, horizon: int = 6, use_fast_tokens: bool = False) -> None:
        """Initialize the action expert.

        Parameters
        ----------
        d_model:
            Token width.
        horizon:
            Action chunk length.
        use_fast_tokens:
            Whether to inject DCT-style frequency tokens, as in pi0-FAST.
        """

        super().__init__()
        self.horizon = horizon
        self.use_fast_tokens = use_fast_tokens
        self.query = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.time = nn.Linear(1, d_model)
        self.noisy_action = nn.Linear(ACTION_DIM, d_model)
        self.fast = nn.Linear(ACTION_DIM, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=96, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.velocity = nn.Linear(d_model, ACTION_DIM)

    def forward(
        self, context: torch.Tensor, noisy_actions: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predict flow-matching velocity for an action chunk."""

        query = self.query.expand(noisy_actions.shape[0], -1, -1)
        query = query + self.noisy_action(noisy_actions) + self.time(t[:, None]).unsqueeze(1)
        if self.use_fast_tokens:
            basis = torch.cos(
                torch.arange(self.horizon, device=noisy_actions.device).float()
                * torch.arange(1, ACTION_DIM + 1, device=noisy_actions.device).float()[:, None]
                * torch.pi
                / max(1, self.horizon)
            )
            freq = torch.matmul(noisy_actions.transpose(1, 2), basis.T).transpose(1, 2)
            query = query + self.fast(freq[:, : self.horizon])
        return self.velocity(self.decoder(query, context))


class FASTActionTokenizer(nn.Module):
    """Frequency-space action tokenizer with discrete BPE-like action codes."""

    def __init__(self, horizon: int = 6, action_dim: int = ACTION_DIM, vocab: int = 32) -> None:
        """Initialize DCT projection, vector codebook, and token embeddings.

        Parameters
        ----------
        horizon:
            Number of action timesteps in a chunk.
        action_dim:
            Per-timestep action dimension.
        vocab:
            Size of the compact discrete action vocabulary.
        """

        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        steps = torch.arange(horizon).float()
        freqs = torch.arange(horizon).float().unsqueeze(1)
        basis = torch.cos(torch.pi / horizon * (steps + 0.5) * freqs)
        basis[0] = basis[0] / horizon**0.5
        basis[1:] = basis[1:] * (2.0 / horizon) ** 0.5
        self.register_buffer("dct_basis", basis)
        self.codebook = nn.Parameter(torch.randn(vocab, action_dim) * 0.02)
        self.token = nn.Embedding(vocab, 48)

    def forward(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert continuous action chunks into soft discrete FAST tokens.

        Parameters
        ----------
        actions:
            Continuous actions shaped ``(batch, horizon, action_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Embedded soft action tokens and token logits.
        """

        coeffs = torch.matmul(self.dct_basis, actions)
        distances = (
            (coeffs.unsqueeze(-2) - self.codebook.view(1, 1, -1, self.action_dim)).pow(2).sum(-1)
        )
        logits = -distances
        weights = torch.softmax(logits, dim=-1)
        token_emb = torch.matmul(weights, self.token.weight)
        return token_emb, logits


class Pi0FASTPolicy(nn.Module):
    """pi0-FAST policy with discrete DCT/BPE-style autoregressive action tokens."""

    def __init__(self, d_model: int = 48, horizon: int = 6) -> None:
        """Initialize VLM context, FAST tokenizer, and autoregressive decoder."""

        super().__init__()
        self.vision = TinyVisionEncoder(d_model)
        self.fusion = TextStateFusion(d_model)
        self.fast = FASTActionTokenizer(horizon=horizon)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=96, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.bpe_merge = nn.Conv1d(d_model, d_model, kernel_size=2, padding=1, groups=4)
        self.out = nn.Linear(d_model, self.fast.codebook.shape[0])

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        text_ids: torch.Tensor,
        noisy_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict autoregressive discrete FAST action-token logits."""

        context = self.fusion(self.vision(images), text_ids, state)
        token_emb, token_logits = self.fast(noisy_actions)
        prefix = torch.cat([torch.zeros_like(token_emb[:, :1]), token_emb[:, :-1]], dim=1)
        merged = self.bpe_merge(prefix.transpose(1, 2))[..., : prefix.shape[1]].transpose(1, 2)
        hidden = self.decoder(prefix + merged, context)
        return self.out(hidden) + token_logits


class VLAJEPAWorldPolicy(nn.Module):
    """VLA-JEPA with leakage-free future-latent target prediction and action tuning."""

    def __init__(self, d_model: int = 48, horizon: int = 6) -> None:
        """Initialize student/target encoders, predictor, and action head."""

        super().__init__()
        self.student_vision = TinyVisionEncoder(d_model, views=1)
        self.target_vision = TinyVisionEncoder(d_model, views=1)
        self.fusion = TextStateFusion(d_model)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.stage = nn.Linear(d_model * 2, d_model)
        self.expert = FlowActionExpert(d_model, horizon)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        text_ids: torch.Tensor,
        noisy_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict future target latents from current context and produce actions."""

        current = images[:, :1]
        future = images[:, -1:]
        student_tokens = self.student_vision(current)
        context = self.fusion(student_tokens, text_ids, state)
        with torch.no_grad():
            target = self.target_vision(future).mean(dim=1)
        predicted_future = self.predictor(context[:, 0])
        world_token = self.stage(torch.cat([context[:, 0], predicted_future], dim=-1)).unsqueeze(1)
        action_context = torch.cat([context, world_token], dim=1)
        t = noisy_actions.new_full((noisy_actions.shape[0],), 0.5)
        velocity = self.expert(action_context, noisy_actions, t)
        jepa_error = (predicted_future - target.detach()).pow(2).mean(dim=-1, keepdim=True)
        return velocity, jepa_error


class VLAPolicy(nn.Module):
    """Shared compact VLM plus action expert for pi0, SmolVLA, X-VLA, and VLA-JEPA."""

    def __init__(
        self,
        d_model: int = 48,
        horizon: int = 6,
        *,
        fast_tokens: bool = False,
        soft_prompt: bool = False,
        jepa_world: bool = False,
        system_two: bool = False,
    ) -> None:
        """Initialize a compact vision-language-action policy."""

        super().__init__()
        self.soft_prompt = soft_prompt
        self.jepa_world = jepa_world
        self.system_two = system_two
        self.vision = TinyVisionEncoder(d_model)
        self.fusion = TextStateFusion(d_model)
        self.prompt = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        self.jepa_predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.planner = nn.GRU(d_model, d_model, batch_first=True)
        self.expert = FlowActionExpert(d_model, horizon, use_fast_tokens=fast_tokens)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        text_ids: torch.Tensor,
        noisy_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict an action-flow velocity from multimodal context."""

        context = self.fusion(self.vision(images), text_ids, state)
        if self.soft_prompt:
            context = torch.cat([self.prompt.expand(images.shape[0], -1, -1), context], dim=1)
        if self.jepa_world:
            predicted_next = self.jepa_predictor(context[:, 1:5])
            context = torch.cat([context, predicted_next], dim=1)
        if self.system_two:
            plan, _ = self.planner(context[:, :4])
            context = torch.cat([context, plan], dim=1)
        t = noisy_actions.new_full((noisy_actions.shape[0],), 0.5)
        return self.expert(context, noisy_actions, t)


def _example_vla() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tiny VLA example tensors."""

    return (
        torch.randn(1, 2, 3, 32, 32),
        torch.randn(1, STATE_DIM),
        torch.randint(0, TEXT_VOCAB, (1, 5)),
        torch.randn(1, 6, ACTION_DIM),
    )


def example_act() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ACT example tensors."""

    return _example_vla()


def example_diffusion() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Diffusion Policy example tensors."""

    images, state, text_ids, _ = _example_vla()
    return images, state, text_ids, torch.randn(1, 8, ACTION_DIM)


def example_vqbet() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return VQ-BeT example tensors."""

    images, state, text_ids, _ = _example_vla()
    return images, state, text_ids


def example_tdmpc() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return TD-MPC example tensors."""

    images, state, text_ids, _ = _example_vla()
    return images, state, text_ids


def example_vla() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flow-VLA example tensors."""

    return _example_vla()


def build_act() -> nn.Module:
    """Build a compact ACT policy."""

    return ACTPolicy()


def build_diffusion() -> nn.Module:
    """Build a compact Diffusion Policy."""

    return DiffusionPolicy()


def build_vqbet() -> nn.Module:
    """Build a compact VQ-BeT policy."""

    return VQBeTPolicy()


def build_tdmpc() -> nn.Module:
    """Build a compact TD-MPC policy."""

    return TDMPCPolicy()


def build_pi0() -> nn.Module:
    """Build a compact pi0 flow-matching VLA."""

    return VLAPolicy()


def build_pi0fast() -> nn.Module:
    """Build a compact pi0-FAST VLA with discrete autoregressive FAST tokens."""

    return Pi0FASTPolicy()


def build_pi05() -> nn.Module:
    """Build a compact pi0.5 VLA with planner-style context recurrence."""

    return VLAPolicy(system_two=True)


def build_smolvla() -> nn.Module:
    """Build a compact SmolVLA policy."""

    return VLAPolicy(d_model=40)


def build_xvla() -> nn.Module:
    """Build a compact X-VLA policy with embodiment soft prompts."""

    return VLAPolicy(soft_prompt=True)


def build_vla_jepa() -> nn.Module:
    """Build a compact VLA-JEPA policy with latent world prediction."""

    return VLAJEPAWorldPolicy()


def _entry(
    name: str, build: str, example: str, year: str, primitive: str
) -> tuple[str, str, str, str, str]:
    """Create a menagerie entry tuple.

    Parameters
    ----------
    name:
        Catalog model name.
    build:
        Builder function name.
    example:
        Example-input function name.
    year:
        Publication or release year.
    primitive:
        Compact era/category code.

    Returns
    -------
    tuple[str, str, str, str, str]
        Self-declaring registry entry.
    """

    return (name, build, example, year, primitive)


MENAGERIE_ENTRIES = [
    _entry("lerobot_act_aloha_sim_insertion_human", "build_act", "example_act", "2023", "RL"),
    _entry("lerobot_act_aloha_sim_transfer_cube_human", "build_act", "example_act", "2023", "RL"),
    _entry("lerobot_diffusion_pusht", "build_diffusion", "example_diffusion", "2023", "RL"),
    _entry(
        "lerobot_diffusion_pusht_keypoints", "build_diffusion", "example_diffusion", "2023", "RL"
    ),
    _entry("lerobot_vqbet_pusht", "build_vqbet", "example_vqbet", "2024", "RL"),
    _entry("LeRobot_VQBeT", "build_vqbet", "example_vqbet", "2024", "RL"),
    _entry("LeRobot_TDMPC", "build_tdmpc", "example_tdmpc", "2022", "RL"),
    _entry("pi0_base_lerobot", "build_pi0", "example_vla", "2024", "RL"),
    _entry("pi0_libero_base_lerobot", "build_pi0", "example_vla", "2024", "RL"),
    _entry("pi0_libero_finetuned_v044", "build_pi0", "example_vla", "2024", "RL"),
    _entry("Pi0_lerobot", "build_pi0", "example_vla", "2024", "RL"),
    _entry("pi0fast_base_lerobot", "build_pi0fast", "example_vla", "2024", "RL"),
    _entry("pi0fast_libero_lerobot", "build_pi0fast", "example_vla", "2024", "RL"),
    _entry("pi05_base_lerobot", "build_pi05", "example_vla", "2025", "RL"),
    _entry("pi05_droid_lerobot", "build_pi05", "example_vla", "2025", "RL"),
    _entry("pi05_libero_base_lerobot", "build_pi05", "example_vla", "2025", "RL"),
    _entry("smolvla_base", "build_smolvla", "example_vla", "2025", "RL"),
    _entry("smolvla_libero", "build_smolvla", "example_vla", "2025", "RL"),
    _entry("smolvla_robocasa", "build_smolvla", "example_vla", "2025", "RL"),
    _entry("smolvla_robocerebra", "build_smolvla", "example_vla", "2025", "RL"),
    _entry("smolvla_robotwin", "build_smolvla", "example_vla", "2025", "RL"),
    _entry("SmolVLA_lerobot", "build_smolvla", "example_vla", "2025", "RL"),
    _entry("lerobot_xvla", "build_xvla", "example_vla", "2025", "RL"),
    _entry("xvla_agibot_world_lerobot", "build_xvla", "example_vla", "2025", "RL"),
    _entry("xvla_base_lerobot", "build_xvla", "example_vla", "2025", "RL"),
    _entry("xvla_google_robot_lerobot", "build_xvla", "example_vla", "2025", "RL"),
    _entry("xvla_widowx_lerobot", "build_xvla", "example_vla", "2025", "RL"),
    _entry("vla_jepa_libero_lerobot", "build_vla_jepa", "example_vla", "2026", "RL"),
    _entry("vla_jepa_pretrain_lerobot", "build_vla_jepa", "example_vla", "2026", "RL"),
    _entry("vla_jepa_simplerenv_lerobot", "build_vla_jepa", "example_vla", "2026", "RL"),
]
