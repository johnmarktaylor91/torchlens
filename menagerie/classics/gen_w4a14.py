"""Multi-agent trajectory / motion-forecasting classics (build queue rows 85-90).

Sources checked (repo_url / desc_source from build_queue.tsv, architecture study only --
no clone, no pip install, faithful compact reimplementation from scratch in base-env torch):

- SIMPL: https://github.com/HKUST-Aerial-Robotics/SIMPL, arxiv:2402.02519 (RA-L 2024).
  Polyline actor/map encoders -> symmetric fusion transformer (directed message passing,
  single feed-forward pass for all agents) -> Bernstein-polynomial (Bezier control point)
  continuous trajectory decoder.
- SMART: https://github.com/rainmaker22/SMART, arxiv:2405.15677 (NeurIPS 2024).
  GPT-style decoder-only transformer performing autoregressive NEXT-TOKEN prediction over
  a shared discrete vocabulary of tokenized agent-motion and roadgraph tokens (causal
  self-attention over an interleaved agent+map token sequence).
- Social-GAN: https://github.com/agrimgupta92/sgan, arxiv:1803.10892 (CVPR 2018).
  LSTM encoder-decoder trajectory generator with a POOLING MODULE (relative-position MLP
  + max-pool over neighboring agents' encoded states) injected into the decoder's initial
  hidden state, trained adversarially against an LSTM-based discriminator.
- Social-LSTM: https://github.com/quancore/social-lstm,
  http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf (CVPR 2016).
  Per-pedestrian LSTMs coupled via a SOCIAL POOLING GRID: each agent's hidden state is
  scattered into neighbors' spatial occupancy grids and summed, then fed back into the
  LSTM input at the next step ("Social pooling layer", Fig. 2 of the paper).
- Social-STGCNN: https://github.com/abduallahmohamed/Social-STGCNN, arxiv:2002.11927
  (CVPR 2020). Spatio-Temporal Graph Convolutional Neural Network (ST-GCNN) with a
  kernel-weighted, distance-based adjacency (no learned attention) over the pedestrian
  graph, followed by a Temporal eXtrapolator Convolutional Neural Network (TXP-CNN) that
  convolves directly along the time axis to extrapolate future frames.
- Sophie: https://github.com/coolsunxu/sophie, arxiv:1806.01482 (CVPR 2019).
  Attentive GAN: an LSTM encoder per agent feeds two attention modules -- PHYSICAL
  attention over a CNN scene-context feature map, and SOCIAL attention over the other
  agents' encoded states -- whose concatenated context drives an LSTM decoder trained
  adversarially (GAN discriminator over generated trajectories).

All six are trajectory / motion-forecasting architectures; kept intentionally tiny
(few agents, short horizons, small hidden dims) since this is an architecture catalog,
not a trained-weights zoo.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# SIMPL: polyline encoders + symmetric fusion transformer + Bernstein decoder
# ---------------------------------------------------------------------------


class PolylineEncoder(nn.Module):
    """Per-polyline point-net style encoder (shared MLP + max-pool over points)."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, poly: torch.Tensor) -> torch.Tensor:
        """Encode polylines.

        Parameters
        ----------
        poly : torch.Tensor
            Shape ``(N, P, in_dim)`` -- N polylines, P points each.

        Returns
        -------
        torch.Tensor
            Shape ``(N, hidden)`` pooled polyline features.
        """
        feat = self.mlp(poly)  # (N, P, hidden)
        return feat.max(dim=1).values


class SymmetricFusionLayer(nn.Module):
    """Directed message-passing fusion (SIMPL's symmetric fusion transformer block)."""

    def __init__(self, dim: int, n_heads: int = 2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens


class BernsteinTrajectoryDecoder(nn.Module):
    """Decodes a trajectory as Bezier control points evaluated via Bernstein basis."""

    def __init__(self, dim: int, n_ctrl: int = 5, n_steps: int = 10) -> None:
        super().__init__()
        self.n_ctrl = n_ctrl
        self.n_steps = n_steps
        self.to_ctrl = nn.Linear(dim, n_ctrl * 2)
        t = torch.linspace(0.0, 1.0, n_steps)
        k = torch.arange(n_ctrl)
        n = n_ctrl - 1
        # Bernstein basis matrix B[t_idx, k] = C(n,k) t^k (1-t)^(n-k)
        binom = torch.tensor([math.comb(n, int(kk)) for kk in k], dtype=torch.float32)
        basis = binom[None, :] * (t[:, None] ** k[None, :]) * ((1 - t[:, None]) ** (n - k[None, :]))
        self.register_buffer("basis", basis)  # (n_steps, n_ctrl)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Decode continuous trajectories from fused agent features.

        Parameters
        ----------
        feat : torch.Tensor
            Shape ``(N, dim)`` fused per-agent features.

        Returns
        -------
        torch.Tensor
            Shape ``(N, n_steps, 2)`` positions evaluated along the Bezier curve.
        """
        ctrl = self.to_ctrl(feat).view(-1, self.n_ctrl, 2)  # (N, n_ctrl, 2)
        return torch.einsum("tk,nkc->ntc", self.basis, ctrl)


class SIMPL(nn.Module):
    """SIMPL: polyline encoding + symmetric fusion transformer + Bernstein decoder."""

    def __init__(self, dim: int = 32, n_fusion_layers: int = 2) -> None:
        super().__init__()
        self.actor_enc = PolylineEncoder(in_dim=4, hidden=dim)
        self.map_enc = PolylineEncoder(in_dim=4, hidden=dim)
        self.fusion = nn.ModuleList([SymmetricFusionLayer(dim) for _ in range(n_fusion_layers)])
        self.decoder = BernsteinTrajectoryDecoder(dim, n_ctrl=5, n_steps=10)

    def forward(self, actors: torch.Tensor, lanes: torch.Tensor) -> torch.Tensor:
        """Predict continuous future trajectories for each actor.

        Parameters
        ----------
        actors : torch.Tensor
            Shape ``(N_actor, P, 4)`` actor history polylines (x, y, dx, dy).
        lanes : torch.Tensor
            Shape ``(N_lane, P, 4)`` map polylines.

        Returns
        -------
        torch.Tensor
            Shape ``(N_actor, n_steps, 2)`` predicted trajectories.
        """
        actor_feat = self.actor_enc(actors)  # (N_actor, dim)
        lane_feat = self.map_enc(lanes)  # (N_lane, dim)
        tokens = torch.cat([actor_feat, lane_feat], dim=0).unsqueeze(0)  # (1, N, dim)
        for layer in self.fusion:
            tokens = layer(tokens)
        fused_actor_feat = tokens[0, : actor_feat.shape[0]]
        return self.decoder(fused_actor_feat)


def build_simpl() -> nn.Module:
    """Build a tiny SIMPL motion-prediction model."""
    return SIMPL(dim=32, n_fusion_layers=2).eval()


def example_input_simpl() -> tuple[torch.Tensor, torch.Tensor]:
    """Example (actor polylines, lane polylines) input for SIMPL."""
    return torch.randn(4, 6, 4), torch.randn(8, 6, 4)


# ---------------------------------------------------------------------------
# SMART: GPT-style decoder-only next-token prediction over motion tokens
# ---------------------------------------------------------------------------


class CausalTransformerBlock(nn.Module):
    """Standard pre-norm causal self-attention transformer block."""

    def __init__(self, dim: int, n_heads: int = 2) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class SMART(nn.Module):
    """SMART: decoder-only transformer, next-token prediction over motion/map tokens.

    A shared vocabulary embeds both discretized agent-motion tokens and roadgraph
    tokens into one interleaved sequence; a causal transformer autoregressively
    predicts the next token (agent displacement bin) at each position.
    """

    def __init__(
        self, vocab_size: int = 64, dim: int = 32, n_layers: int = 2, n_heads: int = 2
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(64, dim)
        self.blocks = nn.ModuleList([CausalTransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict next-token logits over the shared agent+map vocabulary.

        Parameters
        ----------
        tokens : torch.Tensor
            Shape ``(B, T)`` integer token ids (interleaved agent-motion + map tokens).

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, vocab_size)`` next-token logits.
        """
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
        causal_mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.ln_f(x)
        return self.head(x)


def build_smart() -> nn.Module:
    """Build a tiny SMART next-token motion-generation model."""
    return SMART(vocab_size=64, dim=32, n_layers=2, n_heads=2).eval()


def example_input_smart() -> torch.Tensor:
    """Example token sequence (B=2, T=16) for SMART."""
    return torch.randint(0, 64, (2, 16))


# ---------------------------------------------------------------------------
# Social-GAN: LSTM encoder-decoder + social pooling module + GAN discriminator
# ---------------------------------------------------------------------------


class PoolingModule(nn.Module):
    """Social-GAN pooling module: relative-position MLP + max-pool over neighbors."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.rel_mlp = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU())

    def forward(self, h_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Pool neighbor encodings relative to each agent's own position.

        Parameters
        ----------
        h_states : torch.Tensor
            Shape ``(N, hidden)`` encoder hidden states, one per agent.
        positions : torch.Tensor
            Shape ``(N, 2)`` current (x, y) position per agent.

        Returns
        -------
        torch.Tensor
            Shape ``(N, hidden)`` pooled social context per agent.
        """
        n = h_states.shape[0]
        rel_pos = positions[None, :, :] - positions[:, None, :]  # (N_self, N_other, 2)
        h_rep = h_states[None, :, :].expand(n, -1, -1)  # (N_self, N_other, hidden)
        combined = torch.cat([h_rep, rel_pos], dim=-1)
        pooled = self.rel_mlp(combined)  # (N_self, N_other, hidden)
        return pooled.max(dim=1).values


class SGANGenerator(nn.Module):
    """Social-GAN generator: LSTM encoder -> social pooling -> LSTM decoder."""

    def __init__(self, hidden: int = 24, noise_dim: int = 8) -> None:
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Linear(2, hidden)
        self.encoder = nn.LSTMCell(hidden, hidden)
        self.pool = PoolingModule(hidden)
        self.noise_dim = noise_dim
        self.decoder_init = nn.Linear(hidden * 2 + noise_dim, hidden)
        self.decoder = nn.LSTMCell(hidden, hidden)
        self.out = nn.Linear(hidden, 2)

    def forward(self, traj: torch.Tensor, noise: torch.Tensor, n_pred: int = 4) -> torch.Tensor:
        """Encode observed trajectories and roll out socially-aware future positions.

        Parameters
        ----------
        traj : torch.Tensor
            Shape ``(N, T_obs, 2)`` observed trajectories for N agents.
        noise : torch.Tensor
            Shape ``(N, noise_dim)`` per-agent latent noise vector.
        n_pred : int
            Number of future steps to roll out.

        Returns
        -------
        torch.Tensor
            Shape ``(N, n_pred, 2)`` predicted future trajectories.
        """
        n, t_obs, _ = traj.shape
        h = torch.zeros(n, self.hidden, device=traj.device)
        c = torch.zeros(n, self.hidden, device=traj.device)
        for step in range(t_obs):
            emb = self.embed(traj[:, step])
            h, c = self.encoder(emb, (h, c))

        pooled = self.pool(h, traj[:, -1])
        dec_h = self.decoder_init(torch.cat([h, pooled, noise], dim=-1))
        dec_c = torch.zeros_like(dec_h)
        last_pos = traj[:, -1]
        preds = []
        dec_input = self.embed(last_pos)
        for _ in range(n_pred):
            dec_h, dec_c = self.decoder(dec_input, (dec_h, dec_c))
            delta = self.out(dec_h)
            last_pos = last_pos + delta
            preds.append(last_pos)
            dec_input = self.embed(last_pos)
        return torch.stack(preds, dim=1)


class SGANDiscriminator(nn.Module):
    """Social-GAN discriminator: LSTM encoder over full trajectory -> real/fake score."""

    def __init__(self, hidden: int = 24) -> None:
        super().__init__()
        self.embed = nn.Linear(2, hidden)
        self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        emb = self.embed(traj)
        _, (h_n, _) = self.encoder(emb)
        return self.classifier(h_n[-1])


class SocialGAN(nn.Module):
    """Wraps the Social-GAN generator + discriminator into a single traceable module."""

    def __init__(self, hidden: int = 24, noise_dim: int = 8) -> None:
        super().__init__()
        self.generator = SGANGenerator(hidden=hidden, noise_dim=noise_dim)
        self.discriminator = SGANDiscriminator(hidden=hidden)

    def forward(self, traj: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Generate future trajectories, then score the full trajectory as real/fake.

        Parameters
        ----------
        traj : torch.Tensor
            Shape ``(N, T_obs, 2)`` observed trajectories.
        noise : torch.Tensor
            Shape ``(N, noise_dim)`` latent noise for the generator.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 1)`` discriminator realism score for the full (obs + pred) trajectory.
        """
        pred = self.generator(traj, noise, n_pred=4)
        full = torch.cat([traj, pred], dim=1)
        return self.discriminator(full)


def build_social_gan() -> nn.Module:
    """Build a tiny Social-GAN model (generator + discriminator)."""
    return SocialGAN(hidden=24, noise_dim=8).eval()


def example_input_social_gan() -> tuple[torch.Tensor, torch.Tensor]:
    """Example (observed trajectories, noise) input for Social-GAN."""
    return torch.randn(5, 6, 2), torch.randn(5, 8)


# ---------------------------------------------------------------------------
# Social-LSTM: per-pedestrian LSTMs coupled via a spatial social-pooling grid
# ---------------------------------------------------------------------------


class SocialPoolingGrid(nn.Module):
    """Scatters neighbor hidden states into a spatial occupancy grid around each agent."""

    def __init__(self, hidden: int, grid_size: int = 4, neighborhood: float = 4.0) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.neighborhood = neighborhood
        self.embed = nn.Linear(hidden * grid_size * grid_size, hidden)

    def forward(self, h_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Build the per-agent social hidden-state tensor (Social-LSTM pooling layer).

        Parameters
        ----------
        h_states : torch.Tensor
            Shape ``(N, hidden)`` current LSTM hidden states.
        positions : torch.Tensor
            Shape ``(N, 2)`` current (x, y) positions.

        Returns
        -------
        torch.Tensor
            Shape ``(N, hidden)`` pooled social-grid embedding per agent.
        """
        n, hidden = h_states.shape
        g = self.grid_size
        rel = positions[None, :, :] - positions[:, None, :]  # (N_self, N_other, 2)
        # bin the relative offsets into an integer grid cell in [0, g)
        bin_idx = ((rel / self.neighborhood + 0.5) * g).long().clamp(0, g - 1)
        in_range = (rel.abs() <= self.neighborhood).all(dim=-1)  # (N_self, N_other)
        not_self = ~torch.eye(n, dtype=torch.bool, device=h_states.device)
        mask = in_range & not_self

        grid = torch.zeros(n, g, g, hidden, device=h_states.device)
        flat_cell = bin_idx[..., 0] * g + bin_idx[..., 1]  # (N_self, N_other)
        for self_idx in range(n):
            valid = mask[self_idx]
            if valid.any():
                cells = flat_cell[self_idx][valid]
                vals = h_states[valid]
                grid_flat = grid[self_idx].view(g * g, hidden)
                grid_flat.index_add_(0, cells, vals)
        return self.embed(grid.view(n, -1))


class SocialLSTM(nn.Module):
    """Social-LSTM: per-agent LSTMs coupled at every step by a social-pooling grid."""

    def __init__(self, hidden: int = 16, grid_size: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Linear(2, hidden)
        self.pool = SocialPoolingGrid(hidden, grid_size=grid_size)
        self.cell = nn.LSTMCell(hidden * 2, hidden)
        self.out = nn.Linear(hidden, 2)

    def forward(self, traj: torch.Tensor, n_pred: int = 4) -> torch.Tensor:
        """Roll out socially-pooled future trajectories for a scene of pedestrians.

        Parameters
        ----------
        traj : torch.Tensor
            Shape ``(N, T_obs, 2)`` observed trajectories for N pedestrians.
        n_pred : int
            Number of future steps to roll out.

        Returns
        -------
        torch.Tensor
            Shape ``(N, n_pred, 2)`` predicted future positions.
        """
        n, t_obs, _ = traj.shape
        h = torch.zeros(n, self.hidden, device=traj.device)
        c = torch.zeros(n, self.hidden, device=traj.device)
        pos = traj[:, 0]
        for step in range(t_obs):
            pos = traj[:, step]
            emb = self.embed(pos)
            social = self.pool(h, pos)
            h, c = self.cell(torch.cat([emb, social], dim=-1), (h, c))

        preds = []
        for _ in range(n_pred):
            social = self.pool(h, pos)
            emb = self.embed(pos)
            h, c = self.cell(torch.cat([emb, social], dim=-1), (h, c))
            delta = self.out(h)
            pos = pos + delta
            preds.append(pos)
        return torch.stack(preds, dim=1)


def build_social_lstm() -> nn.Module:
    """Build a tiny Social-LSTM model."""
    return SocialLSTM(hidden=16, grid_size=4).eval()


def example_input_social_lstm() -> torch.Tensor:
    """Example observed trajectories (N=5 pedestrians, T_obs=6) for Social-LSTM."""
    return torch.randn(5, 6, 2)


# ---------------------------------------------------------------------------
# Social-STGCNN: ST-GCNN (distance-weighted graph conv) + TXP-CNN extrapolator
# ---------------------------------------------------------------------------


class STGCNNLayer(nn.Module):
    """Spatio-temporal graph conv: kernel-weighted adjacency + per-timestep 1x1 conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, feat: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply one spatial graph-conv step followed by temporal-channel mixing.

        Parameters
        ----------
        feat : torch.Tensor
            Shape ``(1, C, T, N)`` node features (channels, time, nodes).
        adjacency : torch.Tensor
            Shape ``(T, N, N)`` per-timestep distance-kernel adjacency.

        Returns
        -------
        torch.Tensor
            Shape ``(1, out_ch, T, N)`` graph-convolved features.
        """
        _, c, t, n = feat.shape
        # graph convolution: for each timestep, propagate features along adjacency
        out = torch.einsum("bctn,tnm->bctm", feat, adjacency)
        out = self.conv(out)
        out = self.bn(out)
        return F.relu(out)


class TXPCNN(nn.Module):
    """Temporal eXtrapolator CNN: convolves directly along the time axis."""

    def __init__(self, t_obs: int, t_pred: int, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(t_obs, t_pred, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(t_pred, t_pred, kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Extrapolate future timesteps from the ST-GCNN output.

        Parameters
        ----------
        feat : torch.Tensor
            Shape ``(1, C, T_obs, N)`` node features.

        Returns
        -------
        torch.Tensor
            Shape ``(1, C, T_pred, N)`` extrapolated future node features.
        """
        x = feat.permute(0, 2, 1, 3)  # (1, T_obs, C, N)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x.permute(0, 2, 1, 3)  # (1, C, T_pred, N)


class SocialSTGCNN(nn.Module):
    """Social-STGCNN: ST-GCNN (distance-kernel graph conv) + TXP-CNN extrapolator."""

    def __init__(self, hidden: int = 8, t_obs: int = 6, t_pred: int = 4) -> None:
        super().__init__()
        self.gcn1 = STGCNNLayer(2, hidden)
        self.gcn2 = STGCNNLayer(hidden, hidden)
        self.txp = TXPCNN(t_obs=t_obs, t_pred=t_pred, channels=hidden)
        self.out = nn.Conv2d(hidden, 2, kernel_size=1)

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """Predict future pedestrian positions via graph conv + temporal extrapolation.

        Parameters
        ----------
        traj : torch.Tensor
            Shape ``(N, T_obs, 2)`` observed pedestrian trajectories.

        Returns
        -------
        torch.Tensor
            Shape ``(N, T_pred, 2)`` predicted future positions.
        """
        n, t_obs, _ = traj.shape
        feat = traj.permute(2, 1, 0).unsqueeze(0)  # (1, 2, T_obs, N)
        # distance-based kernel adjacency per timestep (Social-STGCNN's weighted graph)
        pos_t = traj.permute(1, 0, 2)  # (T_obs, N, 2)
        dist = torch.cdist(pos_t, pos_t)  # (T_obs, N, N)
        adjacency = torch.exp(-dist)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        h = self.gcn1(feat, adjacency)
        h = self.gcn2(h, adjacency)
        h = self.txp(h)  # (1, hidden, T_pred, N)
        out = self.out(h)  # (1, 2, T_pred, N)
        return out.squeeze(0).permute(2, 1, 0)  # (N, T_pred, 2)


def build_social_stgcnn() -> nn.Module:
    """Build a tiny Social-STGCNN model."""
    return SocialSTGCNN(hidden=8, t_obs=6, t_pred=4).eval()


def example_input_social_stgcnn() -> torch.Tensor:
    """Example observed trajectories (N=5 pedestrians, T_obs=6) for Social-STGCNN."""
    return torch.randn(5, 6, 2)


# ---------------------------------------------------------------------------
# Sophie: attentive GAN with physical (scene-CNN) + social attention modules
# ---------------------------------------------------------------------------


class PhysicalAttention(nn.Module):
    """Attention over a CNN scene-context feature map (Sophie's physical attention)."""

    def __init__(self, hidden: int, feat_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden, feat_dim)

    def forward(self, h_state: torch.Tensor, scene_feat: torch.Tensor) -> torch.Tensor:
        """Attend over spatial scene-context features conditioned on agent hidden state.

        Parameters
        ----------
        h_state : torch.Tensor
            Shape ``(N, hidden)`` per-agent LSTM hidden states.
        scene_feat : torch.Tensor
            Shape ``(L, feat_dim)`` flattened CNN scene-context features (L locations).

        Returns
        -------
        torch.Tensor
            Shape ``(N, feat_dim)`` attended physical context per agent.
        """
        q = self.query(h_state)  # (N, feat_dim)
        scores = q @ scene_feat.t()  # (N, L)
        weights = F.softmax(scores, dim=-1)
        return weights @ scene_feat  # (N, feat_dim)


class SocialAttention(nn.Module):
    """Attention over other agents' encoded states (Sophie's social attention)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)

    def forward(self, h_states: torch.Tensor) -> torch.Tensor:
        """Attend each agent over every other agent's hidden state.

        Parameters
        ----------
        h_states : torch.Tensor
            Shape ``(N, hidden)`` per-agent LSTM hidden states.

        Returns
        -------
        torch.Tensor
            Shape ``(N, hidden)`` attended social context per agent.
        """
        q = self.query(h_states)
        k = self.key(h_states)
        scores = q @ k.t() / math.sqrt(h_states.shape[-1])
        weights = F.softmax(scores, dim=-1)
        return weights @ h_states


class SceneCNN(nn.Module):
    """Small CNN producing a flattened scene-context feature map."""

    def __init__(self, feat_dim: int = 16) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        feat = self.conv(scene)  # (1, feat_dim, H', W')
        c = feat.shape[1]
        return feat.view(c, -1).t()  # (L, feat_dim)


class SophieGenerator(nn.Module):
    """Sophie generator: LSTM encoder + physical & social attention -> LSTM decoder."""

    def __init__(self, hidden: int = 16, feat_dim: int = 16, noise_dim: int = 8) -> None:
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Linear(2, hidden)
        self.encoder = nn.LSTMCell(hidden, hidden)
        self.scene_cnn = SceneCNN(feat_dim=feat_dim)
        self.physical_attn = PhysicalAttention(hidden, feat_dim)
        self.social_attn = SocialAttention(hidden)
        self.decoder_init = nn.Linear(hidden + feat_dim + hidden + noise_dim, hidden)
        self.decoder = nn.LSTMCell(hidden, hidden)
        self.out = nn.Linear(hidden, 2)
        self.noise_dim = noise_dim

    def forward(
        self, traj: torch.Tensor, scene: torch.Tensor, noise: torch.Tensor, n_pred: int = 4
    ) -> torch.Tensor:
        """Roll out future trajectories fusing physical scene and social agent context.

        Parameters
        ----------
        traj : torch.Tensor
            Shape ``(N, T_obs, 2)`` observed trajectories.
        scene : torch.Tensor
            Shape ``(1, 3, H, W)`` RGB scene image.
        noise : torch.Tensor
            Shape ``(N, noise_dim)`` per-agent latent noise.
        n_pred : int
            Number of future steps.

        Returns
        -------
        torch.Tensor
            Shape ``(N, n_pred, 2)`` predicted future trajectories.
        """
        n, t_obs, _ = traj.shape
        h = torch.zeros(n, self.hidden, device=traj.device)
        c = torch.zeros(n, self.hidden, device=traj.device)
        for step in range(t_obs):
            emb = self.embed(traj[:, step])
            h, c = self.encoder(emb, (h, c))

        scene_feat = self.scene_cnn(scene)
        phys_ctx = self.physical_attn(h, scene_feat)
        soc_ctx = self.social_attn(h)
        dec_h = self.decoder_init(torch.cat([h, phys_ctx, soc_ctx, noise], dim=-1))
        dec_c = torch.zeros_like(dec_h)
        last_pos = traj[:, -1]
        preds = []
        dec_input = self.embed(last_pos)
        for _ in range(n_pred):
            dec_h, dec_c = self.decoder(dec_input, (dec_h, dec_c))
            delta = self.out(dec_h)
            last_pos = last_pos + delta
            preds.append(last_pos)
            dec_input = self.embed(last_pos)
        return torch.stack(preds, dim=1)


class Sophie(nn.Module):
    """Sophie: physical + social attention GAN generator wrapped for tracing."""

    def __init__(self, hidden: int = 16, feat_dim: int = 16, noise_dim: int = 8) -> None:
        super().__init__()
        self.generator = SophieGenerator(hidden=hidden, feat_dim=feat_dim, noise_dim=noise_dim)

    def forward(self, traj: torch.Tensor, scene: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.generator(traj, scene, noise, n_pred=4)


def build_sophie() -> nn.Module:
    """Build a tiny Sophie attentive-GAN motion-prediction model."""
    return Sophie(hidden=16, feat_dim=16, noise_dim=8).eval()


def example_input_sophie() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Example (trajectories, scene image, noise) input for Sophie."""
    return torch.randn(5, 6, 2), torch.randn(1, 3, 32, 32), torch.randn(5, 8)


MENAGERIE_ENTRIES = [
    ("SIMPL", "build_simpl", "example_input_simpl", "2024", "SEQ"),
    ("SMART", "build_smart", "example_input_smart", "2024", "SEQ"),
    ("Social-GAN", "build_social_gan", "example_input_social_gan", "2018", "SEQ"),
    ("Social-LSTM", "build_social_lstm", "example_input_social_lstm", "2016", "SEQ"),
    ("Social-STGCNN", "build_social_stgcnn", "example_input_social_stgcnn", "2020", "SEQ"),
    ("Sophie", "build_sophie", "example_input_sophie", "2019", "SEQ"),
]
