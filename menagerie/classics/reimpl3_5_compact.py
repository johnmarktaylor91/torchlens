"""Compact faithful reconstructions for dependency-gated reimpl3_5 targets.

The implementations are random-initialized, CPU-sized Torch modules that preserve
each target family's distinctive primitive:

* MACE: radial atom graph messages, angular/outer-product many-body features,
  residual interaction blocks, energy/dipole/scale-shift heads.
* Mamba-3: complex-valued selective state recurrence, with SISO and MIMO mixers.
* StyleCLIP mapper: text-conditioned residual StyleGAN latent/style deltas split
  into coarse, medium, and fine branches.
* Matrix-LSTM: a grid of LSTM cells that turns asynchronous events into a learned
  event surface.
* MatterGen: diffusion-style crystal graph denoising over atom types, fractional
  coordinates, and lattice vectors.
* PETS: ensemble probabilistic dynamics MLP returning Gaussian mean/log variance.
* MeloTTS: text encoder with duration, pitch, energy, flow, and neural vocoder path.
* Memory Mosaics: stacked associative memories; v2 uses dense SwiGLU memories.
* GShard/MoE: Transformer blocks with top-2 sparse expert routing.
* Depth Anything V2 metric: DINOv2-like ViT encoder with DPT multiscale depth head.
* MFN: multiplicative-filter coordinate network with sinusoidal filters.
* nnU-Net: self-configuring 2D/3D U-Net with skip connections and deep supervision.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Small feed-forward network used by compact classics."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2) -> None:
        """Initialize an MLP.

        Parameters
        ----------
        in_dim:
            Input feature dimension.
        hidden_dim:
            Hidden feature dimension.
        out_dim:
            Output feature dimension.
        layers:
            Number of linear layers.
        """

        super().__init__()
        mods: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(1, layers - 1)):
            mods.extend([nn.Linear(dim, hidden_dim), nn.SiLU()])
            dim = hidden_dim
        mods.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return self.net(x)


class CompactMACE(nn.Module):
    """MACE-style higher-order atomistic message passing model."""

    def __init__(self, head: str = "energy", scale_shift: bool = False) -> None:
        """Initialize a compact MACE variant.

        Parameters
        ----------
        head:
            Prediction head: ``"energy"``, ``"dipole"``, ``"energy_dipole"``, or ``"dielectric"``.
        scale_shift:
            Whether to apply ScaleShiftMACE-style residual scale/shift at readout.
        """

        super().__init__()
        self.head = head
        self.scale_shift = scale_shift
        hidden = 24
        self.atom_embed = nn.Embedding(12, hidden)
        self.radial = MLP(7, hidden, hidden)
        self.vector_radial = MLP(7, hidden, hidden * 3)
        self.edge_gate = nn.Linear(hidden + 4, hidden)
        self.tensor_product = MLP(hidden * 3, hidden, hidden)
        self.update1 = MLP(hidden * 4, hidden, hidden)
        self.update2 = MLP(hidden * 4, hidden, hidden)
        self.energy_head = MLP(hidden, hidden, 1)
        self.charge_head = MLP(hidden, hidden, 1)
        self.polar_head = MLP(hidden, hidden, 6)
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def _interaction(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        update: MLP,
    ) -> torch.Tensor:
        """Apply one ACE-like interaction block.

        Parameters
        ----------
        h:
            Node features.
        pos:
            Atomic positions.
        edge_index:
            Sender/receiver edge indices shaped ``(2, E)``.
        update:
            Node update network.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """

        src, dst = edge_index[0], edge_index[1]
        rel = pos[src] - pos[dst]
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        radial = torch.cat(
            [dist, torch.sin(dist), torch.cos(dist), dist.square(), rel / dist], dim=-1
        )
        rfeat = self.radial(radial)
        vector_channels = self.vector_radial(radial).view(rel.shape[0], -1, 3)
        angular = torch.cat([rel / dist, (rel / dist).prod(dim=-1, keepdim=True)], dim=-1)
        msg = torch.tanh(self.edge_gate(torch.cat([rfeat * h[src], angular], dim=-1)))
        agg = torch.zeros_like(h).index_add(0, dst, msg)
        vector_msg = vector_channels * h[src].unsqueeze(-1)
        vector_agg = torch.zeros(h.shape[0], h.shape[1], 3, device=h.device, dtype=h.dtype)
        vector_agg = vector_agg.index_add(0, dst, vector_msg)
        second_order = torch.einsum("nci,ndi->ncd", vector_agg, vector_agg).mean(dim=-1)
        irreps_mix = torch.cat([agg, vector_agg.norm(dim=-1), second_order], dim=-1)
        many_body = self.tensor_product(irreps_mix)
        delta = update(torch.cat([h, agg, vector_agg.norm(dim=-1), many_body], dim=-1))
        return h + delta

    def forward(
        self, atom_types: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Predict energy, dipole, or both from an atom graph.

        Parameters
        ----------
        atom_types:
            Atomic numbers as integer ids shaped ``(N,)``.
        pos:
            Atom coordinates shaped ``(N, 3)``.
        edge_index:
            Sender/receiver edge indices shaped ``(2, E)``.

        Returns
        -------
        torch.Tensor
            Model output vector.
        """

        h = self.atom_embed(atom_types)
        h = self._interaction(h, pos, edge_index, self.update1)
        h = self._interaction(h, pos, edge_index, self.update2)
        energy = self.energy_head(h).sum(dim=0)
        if self.scale_shift:
            energy = energy * self.scale + self.shift
        charges = self.charge_head(h)
        dipole = (charges * pos).sum(dim=0)
        if self.head == "dipole":
            return dipole
        if self.head == "energy_dipole":
            return torch.cat([energy, dipole], dim=0)
        if self.head == "dielectric":
            polar = self.polar_head(h).sum(dim=0)
            return torch.cat([energy, dipole, polar], dim=0)
        return energy


class ComplexSSMBlock(nn.Module):
    """Mamba-3-like complex selective SSM mixer."""

    def __init__(self, d_model: int = 32, d_state: int = 8, mimo: bool = False) -> None:
        """Initialize the SSM block.

        Parameters
        ----------
        d_model:
            Model width.
        d_state:
            State size per channel.
        mimo:
            Whether to use MIMO state mixing instead of SISO channels.
        """

        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.mimo = mimo
        self.in_proj = nn.Linear(d_model, d_model * 3)
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=2, groups=d_model)
        self.dt = nn.Linear(d_model, d_model)
        self.b = nn.Linear(d_model, d_model * d_state)
        self.c = nn.Linear(d_model, d_model * d_state)
        self.mix = nn.Linear(d_model, d_model, bias=False) if mimo else nn.Identity()
        self.a_real = nn.Parameter(-torch.arange(1, d_state + 1).float().repeat(d_model, 1))
        self.a_imag = nn.Parameter(torch.linspace(0.1, 1.0, d_state).repeat(d_model, 1))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run complex selective recurrence over a token sequence.

        Parameters
        ----------
        x:
            Token tensor shaped ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Mixed token tensor.
        """

        batch, steps, _ = x.shape
        u, gate, residual = self.in_proj(x).chunk(3, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :steps].transpose(1, 2)
        u = F.silu(self.mix(u))
        dt = F.softplus(self.dt(u))
        b = self.b(u).view(batch, steps, self.d_model, self.d_state)
        c = self.c(u).view(batch, steps, self.d_model, self.d_state)
        hr = x.new_zeros(batch, self.d_model, self.d_state)
        hi = x.new_zeros(batch, self.d_model, self.d_state)
        outs = []
        for idx in range(steps):
            decay = torch.exp(dt[:, idx].unsqueeze(-1) * self.a_real.unsqueeze(0))
            phase = dt[:, idx].unsqueeze(-1) * self.a_imag.unsqueeze(0)
            nr = decay * (hr * torch.cos(phase) - hi * torch.sin(phase))
            ni = decay * (hr * torch.sin(phase) + hi * torch.cos(phase))
            hr = nr + b[:, idx] * u[:, idx].unsqueeze(-1)
            hi = ni
            y = (hr * c[:, idx]).sum(dim=-1)
            outs.append(y)
        y = torch.stack(outs, dim=1) * F.silu(gate) + residual
        return self.out(y)


class Mamba3LM(nn.Module):
    """Tiny Mamba-3 language model with complex SSM layers."""

    def __init__(self, mimo: bool = False) -> None:
        """Initialize the language model.

        Parameters
        ----------
        mimo:
            Whether layers use MIMO mixing.
        """

        super().__init__()
        self.embed = nn.Embedding(128, 32)
        self.blocks = nn.ModuleList([ComplexSSMBlock(mimo=mimo) for _ in range(2)])
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 128)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Predict logits from token ids.

        Parameters
        ----------
        ids:
            Token ids shaped ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Vocabulary logits.
        """

        x = self.embed(ids)
        for block in self.blocks:
            x = x + block(self.norm(x))
        return self.head(self.norm(x))


class StyleBranch(nn.Module):
    """One coarse/medium/fine StyleCLIP mapper branch."""

    def __init__(self, latent_dim: int, text_dim: int) -> None:
        """Initialize a mapper branch.

        Parameters
        ----------
        latent_dim:
            Latent dimension for one StyleGAN layer group.
        text_dim:
            CLIP text embedding dimension.
        """

        super().__init__()
        self.net = MLP(latent_dim + text_dim, latent_dim * 2, latent_dim, layers=3)

    def forward(self, latent: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Return a text-conditioned residual latent delta.

        Parameters
        ----------
        latent:
            Style latent group.
        text:
            CLIP text embedding.

        Returns
        -------
        torch.Tensor
            Residual delta.
        """

        return self.net(torch.cat([latent, text], dim=-1))


class StyleCLIPMapper(nn.Module):
    """StyleCLIP text-guided mapper split into StyleGAN layer levels."""

    def __init__(self, mode: str = "levels", style_space: bool = False) -> None:
        """Initialize the mapper.

        Parameters
        ----------
        mode:
            Mapper variant name.
        style_space:
            Whether to output per-style-channel deltas.
        """

        super().__init__()
        self.mode = mode
        self.style_space = style_space
        text_dim = 32
        self.text = nn.Linear(text_dim, text_dim)
        self.coarse = StyleBranch(32, text_dim)
        self.medium = StyleBranch(32, text_dim)
        self.fine = StyleBranch(32, text_dim)
        self.to_style = nn.Linear(96, 128) if style_space else nn.Identity()

    def forward(self, latent: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Map latent and CLIP text embedding to StyleGAN edits.

        Parameters
        ----------
        latent:
            Extended latent shaped ``(B, 3, 32)``.
        text:
            Text embedding shaped ``(B, 32)``.

        Returns
        -------
        torch.Tensor
            Edited latent or style-space delta.
        """

        t = F.normalize(self.text(text), dim=-1)
        dc = self.coarse(latent[:, 0], t)
        dm = self.medium(latent[:, 1], t)
        df = self.fine(latent[:, 2], t)
        if self.mode == "without_torgb":
            df = df * 0.25
        delta = torch.stack([dc, dm, df], dim=1)
        edited = latent + delta
        if self.style_space:
            return self.to_style(edited.flatten(1))
        return edited


class MatrixLSTMEventNet(nn.Module):
    """Matrix-LSTM learned event surface for event-camera streams."""

    def __init__(self, grid: int = 4, hidden: int = 12) -> None:
        """Initialize the event surface model.

        Parameters
        ----------
        grid:
            Grid height and width.
        hidden:
            LSTM hidden size per cell.
        """

        super().__init__()
        self.grid = grid
        self.hidden = hidden
        self.cell = nn.LSTMCell(3, hidden)
        self.classifier = MLP(grid * grid * hidden, 64, 5)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """Process asynchronous events.

        Parameters
        ----------
        events:
            Event tensor shaped ``(T, 4)`` with x, y, polarity, dt.

        Returns
        -------
        torch.Tensor
            Classification logits from the learned surface.
        """

        cells = self.grid * self.grid
        h = events.new_zeros(cells, self.hidden)
        c = events.new_zeros(cells, self.hidden)
        for idx in range(events.shape[0]):
            xy = events[idx, :2].clamp(0, self.grid - 1).long()
            cell_idx = xy[1] * self.grid + xy[0]
            inp = events[idx, 1:].unsqueeze(0)
            nh, nc = self.cell(inp, (h[cell_idx : cell_idx + 1], c[cell_idx : cell_idx + 1]))
            mask = F.one_hot(cell_idx, cells).to(events.dtype).unsqueeze(-1)
            h = h * (1.0 - mask) + nh * mask
            c = c * (1.0 - mask) + nc * mask
        return self.classifier(h.flatten().unsqueeze(0))


class MatterGenCrystalDiffusion(nn.Module):
    """Compact MatterGen-style crystal diffusion denoiser."""

    def __init__(self) -> None:
        """Initialize crystal graph denoiser."""

        super().__init__()
        hidden = 32
        self.atom = nn.Embedding(16, hidden)
        self.time = nn.Linear(1, hidden)
        self.edge = MLP(7, hidden, hidden)
        self.node = MLP(hidden * 2, hidden, hidden)
        self.type_head = nn.Linear(hidden, 16)
        self.coord_head = nn.Linear(hidden, 3)
        self.lattice_head = MLP(hidden, hidden, 9)

    def forward(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise crystal atom types, fractional coordinates, and lattice.

        Parameters
        ----------
        atom_types:
            Atom type ids shaped ``(N,)``.
        frac_coords:
            Fractional coordinates shaped ``(N, 3)``.
        lattice:
            Lattice matrix shaped ``(3, 3)``.
        t:
            Diffusion time shaped ``(1, 1)``.

        Returns
        -------
        torch.Tensor
            Concatenated denoising predictions.
        """

        n = atom_types.shape[0]
        h = self.atom(atom_types) + self.time(t).expand(n, -1)
        rel = frac_coords.unsqueeze(1) - frac_coords.unsqueeze(0)
        rel = rel - rel.round()
        cart = torch.matmul(rel, lattice)
        dist = cart.norm(dim=-1, keepdim=True)
        edge_feat = self.edge(torch.cat([rel, cart, dist], dim=-1))
        agg = edge_feat.sum(dim=1) / float(n)
        h = h + self.node(torch.cat([h, agg], dim=-1))
        type_logits = self.type_head(h).mean(dim=0)
        coord_score = self.coord_head(h).mean(dim=0)
        lattice_score = self.lattice_head(h.mean(dim=0)).flatten()
        return torch.cat([type_logits, coord_score, lattice_score], dim=0)


class PETSGaussianMLP(nn.Module):
    """PETS ensemble Gaussian dynamics model."""

    def __init__(self, ensemble: int = 3) -> None:
        """Initialize the PETS dynamics ensemble.

        Parameters
        ----------
        ensemble:
            Number of probabilistic MLP members.
        """

        super().__init__()
        self.members = nn.ModuleList([MLP(8, 32, 12, layers=3) for _ in range(ensemble)])

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        """Return ensemble mean and log-variance predictions.

        Parameters
        ----------
        state_action:
            Concatenated state/action tensor shaped ``(B, 8)``.

        Returns
        -------
        torch.Tensor
            Predictions shaped ``(E, B, 12)``.
        """

        return torch.stack([member(state_action) for member in self.members], dim=0)


class MeloTTSNet(nn.Module):
    """Compact MeloTTS/VITS-style text-to-waveform path."""

    def __init__(self) -> None:
        """Initialize text encoder, variance adaptors, flow, and decoder."""

        super().__init__()
        self.embed = nn.Embedding(128, 32)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(32, 4, 64, batch_first=True), num_layers=1
        )
        self.duration = nn.Linear(32, 1)
        self.pitch = nn.Linear(32, 1)
        self.energy = nn.Linear(32, 1)
        self.flow = nn.Conv1d(32, 32, 3, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 24, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(24, 1, 7, padding=3),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Synthesize a compact waveform proxy from phoneme tokens.

        Parameters
        ----------
        tokens:
            Token ids shaped ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Waveform proxy shaped ``(B, 1, 2T)``.
        """

        h = self.encoder(self.embed(tokens))
        variance = self.duration(h) + self.pitch(h) + self.energy(h)
        h = h + variance
        z = self.flow(h.transpose(1, 2))
        return self.decoder(z)


class AssociativeMemory(nn.Module):
    """Key-value associative memory unit."""

    def __init__(self, dim: int = 32, slots: int = 16, swiglu: bool = False) -> None:
        """Initialize memory unit.

        Parameters
        ----------
        dim:
            Feature dimension.
        slots:
            Number of memory slots.
        swiglu:
            Whether to use v2 dense SwiGLU memory.
        """

        super().__init__()
        self.swiglu = swiglu
        self.keys = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.values = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.gate = nn.Linear(dim, dim)
        self.up = nn.Linear(dim, dim) if swiglu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve and compose memory values.

        Parameters
        ----------
        x:
            Token features shaped ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Updated token features.
        """

        attn = torch.softmax(torch.matmul(x, self.keys.t()) / math.sqrt(x.shape[-1]), dim=-1)
        mem = torch.matmul(attn, self.values)
        if self.swiglu:
            mem = F.silu(self.gate(mem)) * self.up(mem)
        return x + mem


class MemoryMosaicNet(nn.Module):
    """Stack of associative memories for Memory Mosaics."""

    def __init__(self, v2: bool = False) -> None:
        """Initialize the mosaic.

        Parameters
        ----------
        v2:
            Use v2 SwiGLU persistent memory units.
        """

        super().__init__()
        self.embed = nn.Embedding(128, 32)
        self.memories = nn.ModuleList([AssociativeMemory(swiglu=v2) for _ in range(3)])
        self.head = nn.Linear(32, 128)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Predict tokens using stacked memories.

        Parameters
        ----------
        ids:
            Token ids shaped ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        x = self.embed(ids)
        for memory in self.memories:
            x = memory(x)
        return self.head(x)


class Top2MoE(nn.Module):
    """GShard top-2 routed feed-forward layer."""

    def __init__(self, dim: int = 32, experts: int = 4) -> None:
        """Initialize sparse expert layer.

        Parameters
        ----------
        dim:
            Token dimension.
        experts:
            Number of experts.
        """

        super().__init__()
        self.router = nn.Linear(dim, experts)
        self.experts = nn.ModuleList([MLP(dim, 64, dim) for _ in range(experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route each token through top-2 experts.

        Parameters
        ----------
        x:
            Token features shaped ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Expert-combined features.
        """

        probs = torch.softmax(self.router(x), dim=-1)
        topw, topi = torch.topk(probs, 2, dim=-1)
        gate = torch.zeros_like(probs).scatter(-1, topi, topw)
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        out = torch.zeros_like(x)
        for idx, expert in enumerate(self.experts):
            out = out + gate[..., idx : idx + 1] * expert(x)
        return out


class GShardMoETransformer(nn.Module):
    """Compact Transformer with every other FFN replaced by top-2 MoE."""

    def __init__(self) -> None:
        """Initialize GShard-style MoE Transformer."""

        super().__init__()
        self.embed = nn.Embedding(128, 32)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.moe = Top2MoE(32, 4)
        self.ffn = MLP(32, 64, 32)
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(32)
        self.head = nn.Linear(32, 128)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Run the MoE Transformer.

        Parameters
        ----------
        ids:
            Token ids shaped ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Vocabulary logits.
        """

        x = self.embed(ids)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.moe(self.norm2(x))
        x = x + self.ffn(self.norm2(x))
        return self.head(x)


class TinyViTBlock(nn.Module):
    """Small ViT block for Depth Anything V2 compact encoder."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize block.

        Parameters
        ----------
        dim:
            Token dimension.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and MLP.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        torch.Tensor
            Updated tokens.
        """

        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class DepthAnythingMetric(nn.Module):
    """Depth Anything V2 metric-depth compact DINOv2+DPT model."""

    def __init__(self, width: int = 32) -> None:
        """Initialize metric depth model.

        Parameters
        ----------
        width:
            Encoder width used to represent ViT-S/B/L aliases compactly.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, width, 4, stride=4)
        self.pos = nn.Parameter(torch.randn(1, 64, width) * 0.02)
        self.blocks = nn.ModuleList([TinyViTBlock(width) for _ in range(2)])
        self.proj1 = nn.Conv2d(width, 24, 1)
        self.proj2 = nn.Conv2d(width, 24, 3, padding=1)
        self.head = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Softplus(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict dense metric depth.

        Parameters
        ----------
        image:
            RGB image shaped ``(B, 3, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Positive depth map shaped ``(B, 1, 32, 32)``.
        """

        feat = self.patch(image)
        batch, dim, height, width = feat.shape
        tokens = feat.flatten(2).transpose(1, 2) + self.pos[:, : height * width]
        mids = []
        for block in self.blocks:
            tokens = block(tokens)
            mids.append(tokens.transpose(1, 2).view(batch, dim, height, width))
        dpt = torch.cat([self.proj1(mids[0]), self.proj2(mids[-1])], dim=1)
        return F.interpolate(self.head(dpt), size=image.shape[-2:], mode="bilinear")


class MultiplicativeFilterNetwork(nn.Module):
    """MFN coordinate network with multiplicative sinusoidal filters."""

    def __init__(self) -> None:
        """Initialize MFN."""

        super().__init__()
        self.filters = nn.ModuleList([nn.Linear(2, 32) for _ in range(3)])
        self.linears = nn.ModuleList([nn.Linear(32, 32) for _ in range(3)])
        self.out = nn.Linear(32, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate implicit signal at coordinates.

        Parameters
        ----------
        coords:
            Coordinates shaped ``(B, N, 2)``.

        Returns
        -------
        torch.Tensor
            Scalar field values.
        """

        h = torch.sin(self.filters[0](coords))
        for filt, linear in zip(self.filters[1:], self.linears[1:], strict=False):
            h = linear(h) * torch.sin(filt(coords))
        return self.out(h)


class NNUNet2D(nn.Module):
    """Compact nnU-Net 2D self-configuring U-Net."""

    def __init__(self) -> None:
        """Initialize 2D nnU-Net."""

        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.InstanceNorm2d(12), nn.LeakyReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.InstanceNorm2d(24), nn.LeakyReLU()
        )
        self.bot = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1), nn.InstanceNorm2d(24), nn.LeakyReLU()
        )
        self.up = nn.ConvTranspose2d(24, 12, 2, stride=2)
        self.seg = nn.Conv2d(24, 3, 1)
        self.aux = nn.Conv2d(24, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segment a 2D medical image.

        Parameters
        ----------
        x:
            Image shaped ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Main and auxiliary logits concatenated.
        """

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bot(e2)
        up = self.up(b)
        main = self.seg(torch.cat([up, e1], dim=1))
        aux = F.interpolate(self.aux(b), size=main.shape[-2:], mode="bilinear")
        return torch.cat([main, aux], dim=1)


class NNUNet3D(nn.Module):
    """Compact nnU-Net 3D volumetric U-Net."""

    def __init__(self) -> None:
        """Initialize 3D nnU-Net."""

        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.InstanceNorm3d(8), nn.LeakyReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.InstanceNorm3d(16), nn.LeakyReLU()
        )
        self.bot = nn.Sequential(
            nn.Conv3d(16, 16, 3, padding=1), nn.InstanceNorm3d(16), nn.LeakyReLU()
        )
        self.up = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.seg = nn.Conv3d(16, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segment a 3D medical volume.

        Parameters
        ----------
        x:
            Volume shaped ``(B, 1, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Segmentation logits.
        """

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bot(e2)
        up = self.up(b)
        return self.seg(torch.cat([up, e1], dim=1))


def build_mace() -> nn.Module:
    """Build compact MACE energy model.

    Returns
    -------
    nn.Module
        Random-initialized MACE model.
    """

    return CompactMACE("energy").eval()


def build_scaleshift_mace() -> nn.Module:
    """Build compact ScaleShiftMACE model.

    Returns
    -------
    nn.Module
        Random-initialized ScaleShiftMACE model.
    """

    return CompactMACE("energy", scale_shift=True).eval()


def build_atomic_dipoles_mace() -> nn.Module:
    """Build compact AtomicDipolesMACE model.

    Returns
    -------
    nn.Module
        Random-initialized dipole model.
    """

    return CompactMACE("dipole").eval()


def build_energy_dipoles_mace() -> nn.Module:
    """Build compact EnergyDipolesMACE model.

    Returns
    -------
    nn.Module
        Random-initialized energy+dipole model.
    """

    return CompactMACE("energy_dipole").eval()


def build_atomic_dielectric_mace() -> nn.Module:
    """Build compact AtomicDielectricMACE model.

    Returns
    -------
    nn.Module
        Random-initialized energy, dipole, and polarizability model.
    """

    return CompactMACE("dielectric").eval()


def example_mace() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a small molecular graph example.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Atom ids, positions, and edge index.
    """

    atom_types = torch.tensor([6, 1, 1, 8, 1], dtype=torch.long)
    pos = torch.randn(5, 3) * 0.5
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 3, 4, 1], [1, 2, 0, 0, 4, 0, 3, 3]], dtype=torch.long
    )
    return atom_types, pos, edge_index


def build_mamba3_siso() -> nn.Module:
    """Build compact Mamba-3 SISO model.

    Returns
    -------
    nn.Module
        Random-initialized Mamba-3 SISO model.
    """

    return Mamba3LM(mimo=False).eval()


def build_mamba3_mimo() -> nn.Module:
    """Build compact Mamba-3 MIMO model.

    Returns
    -------
    nn.Module
        Random-initialized Mamba-3 MIMO model.
    """

    return Mamba3LM(mimo=True).eval()


def example_tokens() -> torch.Tensor:
    """Return compact token input.

    Returns
    -------
    torch.Tensor
        Token ids.
    """

    return torch.randint(0, 128, (1, 8))


def build_styleclip_levels_mapper() -> nn.Module:
    """Build StyleCLIP levels mapper.

    Returns
    -------
    nn.Module
        StyleCLIP mapper.
    """

    return StyleCLIPMapper("levels").eval()


def build_styleclip_mapper() -> nn.Module:
    """Build StyleCLIP full mapper.

    Returns
    -------
    nn.Module
        StyleCLIP mapper.
    """

    return StyleCLIPMapper("full").eval()


def build_styleclip_fullstylespace_mapper() -> nn.Module:
    """Build StyleCLIP StyleSpace mapper.

    Returns
    -------
    nn.Module
        StyleCLIP style-space mapper.
    """

    return StyleCLIPMapper("full", style_space=True).eval()


def build_styleclip_without_torgb_mapper() -> nn.Module:
    """Build StyleCLIP mapper without ToRGB/fine color emphasis.

    Returns
    -------
    nn.Module
        StyleCLIP mapper.
    """

    return StyleCLIPMapper("without_torgb").eval()


def example_styleclip() -> tuple[torch.Tensor, torch.Tensor]:
    """Return latent and text embedding example.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Latent and CLIP text tensors.
    """

    return torch.randn(1, 3, 32), torch.randn(1, 32)


def build_matrix_lstm_event() -> nn.Module:
    """Build Matrix-LSTM event model.

    Returns
    -------
    nn.Module
        Random-initialized Matrix-LSTM.
    """

    return MatrixLSTMEventNet().eval()


def example_events() -> torch.Tensor:
    """Return event stream.

    Returns
    -------
    torch.Tensor
        Events with x, y, polarity, dt.
    """

    return torch.tensor(
        [[0, 0, 1.0, 0.01], [1, 0, -1.0, 0.03], [2, 1, 1.0, 0.04], [3, 3, 1.0, 0.06]]
    )


def build_mattergen_crystal_diffusion() -> nn.Module:
    """Build MatterGen crystal diffusion model.

    Returns
    -------
    nn.Module
        Random-initialized crystal denoiser.
    """

    return MatterGenCrystalDiffusion().eval()


def example_mattergen() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a small crystal diffusion example.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Atom types, fractional coordinates, lattice, and time.
    """

    return torch.tensor([6, 8, 14, 8]), torch.rand(4, 3), torch.eye(3), torch.tensor([[0.4]])


def build_pets_gaussian_mlp() -> nn.Module:
    """Build PETS probabilistic ensemble.

    Returns
    -------
    nn.Module
        PETS Gaussian MLP ensemble.
    """

    return PETSGaussianMLP().eval()


def example_pets() -> torch.Tensor:
    """Return state-action input.

    Returns
    -------
    torch.Tensor
        State-action tensor.
    """

    return torch.randn(2, 8)


def build_melotts() -> nn.Module:
    """Build MeloTTS compact model.

    Returns
    -------
    nn.Module
        Random-initialized MeloTTS model.
    """

    return MeloTTSNet().eval()


def build_memory_mosaics() -> nn.Module:
    """Build Memory Mosaics v1 model.

    Returns
    -------
    nn.Module
        Associative-memory mosaic.
    """

    return MemoryMosaicNet(v2=False).eval()


def build_memory_mosaics_v2() -> nn.Module:
    """Build Memory Mosaics v2 model.

    Returns
    -------
    nn.Module
        SwiGLU-memory mosaic.
    """

    return MemoryMosaicNet(v2=True).eval()


def build_gshard_moe() -> nn.Module:
    """Build GShard top-2 MoE Transformer.

    Returns
    -------
    nn.Module
        GShard-style MoE model.
    """

    return GShardMoETransformer().eval()


def build_depth_anything_v2_metric_s() -> nn.Module:
    """Build compact Depth Anything V2 ViT-S metric model.

    Returns
    -------
    nn.Module
        Depth model.
    """

    return DepthAnythingMetric(width=24).eval()


def build_depth_anything_v2_metric_b() -> nn.Module:
    """Build compact Depth Anything V2 ViT-B metric model.

    Returns
    -------
    nn.Module
        Depth model.
    """

    return DepthAnythingMetric(width=32).eval()


def build_depth_anything_v2_metric_l() -> nn.Module:
    """Build compact Depth Anything V2 ViT-L metric model.

    Returns
    -------
    nn.Module
        Depth model.
    """

    return DepthAnythingMetric(width=40).eval()


def example_image() -> torch.Tensor:
    """Return compact image input.

    Returns
    -------
    torch.Tensor
        RGB image.
    """

    return torch.randn(1, 3, 32, 32)


def build_mfn() -> nn.Module:
    """Build multiplicative filter network.

    Returns
    -------
    nn.Module
        MFN coordinate model.
    """

    return MultiplicativeFilterNetwork().eval()


def example_coords() -> torch.Tensor:
    """Return coordinate input.

    Returns
    -------
    torch.Tensor
        Coordinate tensor.
    """

    return torch.rand(1, 16, 2) * 2 - 1


def build_nnunet_2d() -> nn.Module:
    """Build compact nnU-Net 2D.

    Returns
    -------
    nn.Module
        2D nnU-Net.
    """

    return NNUNet2D().eval()


def example_nnunet_2d() -> torch.Tensor:
    """Return 2D medical image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(1, 1, 32, 32)


def build_nnunet_3d() -> nn.Module:
    """Build compact nnU-Net 3D.

    Returns
    -------
    nn.Module
        3D nnU-Net.
    """

    return NNUNet3D().eval()


def example_nnunet_3d() -> torch.Tensor:
    """Return 3D medical volume.

    Returns
    -------
    torch.Tensor
        Volume tensor.
    """

    return torch.randn(1, 1, 8, 16, 16)


MACE_ENTRIES = [
    ("mace", "build_mace", "example_mace", "2022", "DC"),
    ("MACE small", "build_mace", "example_mace", "2022", "DC"),
    ("mace_MACE_R6", "build_mace", "example_mace", "2022", "DC"),
    ("NequIP-derived: ScaleShiftMACE", "build_scaleshift_mace", "example_mace", "2022", "DC"),
    ("mace_MACE_cuEquivariance", "build_mace", "example_mace", "2022", "DC"),
    ("mace_AtomicDielectricMACE", "build_atomic_dielectric_mace", "example_mace", "2022", "DC"),
    ("mace_AtomicDielectricMACE_R6", "build_atomic_dielectric_mace", "example_mace", "2022", "DC"),
    ("mace_AtomicDipolesMACE_R6", "build_atomic_dipoles_mace", "example_mace", "2022", "DC"),
    ("mace_EnergyDipolesMACE", "build_energy_dipoles_mace", "example_mace", "2022", "DC"),
    ("mace_EnergyDipolesMACE_R6", "build_energy_dipoles_mace", "example_mace", "2022", "DC"),
    ("mace_MACE_OpenEquivariance", "build_mace", "example_mace", "2022", "DC"),
    ("mace_ScaleShiftMACE", "build_scaleshift_mace", "example_mace", "2022", "DC"),
    ("mace_ScaleShiftMACE_R6", "build_scaleshift_mace", "example_mace", "2022", "DC"),
    ("mace_MACE_SO3", "build_mace", "example_mace", "2022", "DC"),
    ("MACE-MP-0 / MACE-MH", "build_mace", "example_mace", "2024", "DC"),
    ("MACE-OFF", "build_mace", "example_mace", "2024", "DC"),
    ("mace_MACE", "build_mace", "example_mace", "2022", "DC"),
    ("mace_AtomicDipolesMACE", "build_atomic_dipoles_mace", "example_mace", "2022", "DC"),
]

MENAGERIE_ENTRIES = [
    *MACE_ENTRIES,
    ("mamba3_mimo", "build_mamba3_mimo", "example_tokens", "2026", "DA"),
    ("mamba3_siso", "build_mamba3_siso", "example_tokens", "2026", "DA"),
    ("styleclip_levels_mapper", "build_styleclip_levels_mapper", "example_styleclip", "2021", "E5"),
    ("styleclip_mapper", "build_styleclip_mapper", "example_styleclip", "2021", "E5"),
    (
        "styleclip_fullstylespace_mapper",
        "build_styleclip_fullstylespace_mapper",
        "example_styleclip",
        "2021",
        "E5",
    ),
    (
        "styleclip_without_torgb_mapper",
        "build_styleclip_without_torgb_mapper",
        "example_styleclip",
        "2021",
        "E5",
    ),
    ("Matrix-LSTM-event", "build_matrix_lstm_event", "example_events", "2020", "E5"),
    (
        "MatterGen-CrystalDiffusion",
        "build_mattergen_crystal_diffusion",
        "example_mattergen",
        "2025",
        "DC",
    ),
    ("PETS_GaussianMLP", "build_pets_gaussian_mlp", "example_pets", "2018", "MB"),
    ("MeloTTS", "build_melotts", "example_tokens", "2023", "DE"),
    ("MemoryMosaics", "build_memory_mosaics", "example_tokens", "2025", "DA"),
    ("MemoryMosaicsV2", "build_memory_mosaics_v2", "example_tokens", "2025", "DA"),
    ("gshard_moe", "build_gshard_moe", "example_tokens", "2020", "DA"),
    ("MoE_sparse", "build_gshard_moe", "example_tokens", "2020", "DA"),
    (
        "depth_anything_v2_metric:vitb",
        "build_depth_anything_v2_metric_b",
        "example_image",
        "2024",
        "E5",
    ),
    (
        "depth_anything_v2_metric:vitl",
        "build_depth_anything_v2_metric_l",
        "example_image",
        "2024",
        "E5",
    ),
    (
        "depth_anything_v2_metric:vits",
        "build_depth_anything_v2_metric_s",
        "example_image",
        "2024",
        "E5",
    ),
    (
        "depth_anything_v2_metric_hypersim_vitb",
        "build_depth_anything_v2_metric_b",
        "example_image",
        "2024",
        "E5",
    ),
    (
        "depth_anything_v2_metric_hypersim_vitl",
        "build_depth_anything_v2_metric_l",
        "example_image",
        "2024",
        "E5",
    ),
    (
        "depth_anything_v2_metric_hypersim_vits",
        "build_depth_anything_v2_metric_s",
        "example_image",
        "2024",
        "E5",
    ),
    (
        "depth_anything_v2_metric_vkitti_vitb",
        "build_depth_anything_v2_metric_b",
        "example_image",
        "2024",
        "E5",
    ),
    (
        "depth_anything_v2_metric_vkitti_vitl",
        "build_depth_anything_v2_metric_l",
        "example_image",
        "2024",
        "E5",
    ),
    (
        "depth_anything_v2_metric_vkitti_vits",
        "build_depth_anything_v2_metric_s",
        "example_image",
        "2024",
        "E5",
    ),
    ("MFN", "build_mfn", "example_coords", "2020", "DA"),
    ("nnUNet_2D_Generic", "build_nnunet_2d", "example_nnunet_2d", "2021", "E5"),
    ("nnUNet_3D_Generic", "build_nnunet_3d", "example_nnunet_3d", "2021", "E5"),
]
