# SOURCE: vendored from https://github.com/mengmengliu1998/GATraj @ main
# (basemodel.py, laplace_decoder.py, models.py)
# GATraj: A Graph- and Attention-based Multi-Agent Trajectory Prediction Model. ISPRS 2023.
"""GATraj: graph-attention social-refinement encoder + Laplacian mixture-density decoder.

Vendored verbatim from mengmengliu1998/GATraj `basemodel.py` + `laplace_decoder.py` +
`models.py`. Architecture is unmodified; only this header/build/example wrapper were
added for menagerie staging. Two minimal fixes for a self-contained modern-Python
module: (1) the original files use unqualified `from utils import *` / `from
basemodel import *` / `from laplace_decoder import *` star-imports across three
files -- inlined here into one file in dependency order instead; (2) `basemodel.py`
imports `from fractions import gcd`, a Python-2-era symbol removed from the stdlib in
Python 3.9+ (PEP 570-era `fractions.gcd` deprecation) -- the import is dead code (never
referenced anywhere in the class bodies below) so it is dropped rather than "fixed",
since fixing a completely unused import would be a no-op with extra risk.
"""

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# basemodel.py
# ---------------------------------------------------------------------------
def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)  # initializing the lstm bias with zeros
        else:
            print(m, "************")


class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP_gate(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP_gate, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Temperal_Encoder(nn.Module):
    """Construct the sequence model"""

    def __init__(self, args):
        super(Temperal_Encoder, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        if args.input_mix:
            self.conv1d = nn.Conv1d(4, self.hidden_size, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1d = nn.Conv1d(2, self.hidden_size, kernel_size=3, stride=1, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.args.x_encoder_head,
            dim_feedforward=self.hidden_size,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.args.x_encoder_layers
        )
        self.mlp1 = MLP(self.hidden_size)
        self.mlp = MLP(self.hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        initialize_weights(self.conv1d.modules())

    def forward(self, x):
        self.x_dense = self.conv1d(x).permute(0, 2, 1)  # [N, H, dim]
        self.x_dense = self.mlp1(self.x_dense) + self.x_dense  # [N, H, dim]
        self.x_dense_in = self.transformer_encoder(self.x_dense) + self.x_dense  # [N, H, D]
        output, (hn, cn) = self.lstm(self.x_dense_in)
        self.x_state, cn = hn.squeeze(0), cn.squeeze(0)  # [N, D]
        self.x_endoced = self.mlp(self.x_state) + self.x_state  # [N, D]
        return self.x_endoced, self.x_state, cn


class Global_interaction(nn.Module):
    def __init__(self, args):
        super(Global_interaction, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        # Motion gate
        self.ngate = MLP_gate(self.hidden_size * 3, self.hidden_size)  # sigmoid
        # Relative spatial embedding layer
        self.relativeLayer = MLP(2, self.hidden_size)
        # Attention
        self.WAr = MLP(self.hidden_size * 3, 1)
        self.weight = MLP(self.hidden_size)

    def forward(self, corr_index, nei_index, nei_num, hidden_state, cn):
        """
        States Refinement process
        Params:
            corr_index: relative coords of each pedestrian pair [N, N, D]
            nei_index: neighbor exsists flag [N, N]
            nei_num: neighbor number [N]
            hidden_state: output states of GRU [N, D]
        Return:
            Refined states
        """
        device = hidden_state.device
        self_h = hidden_state
        self.N = corr_index.shape[0]
        self.D = self.hidden_size
        nei_inputs = self_h.repeat(self.N, 1)  # [N, N, D]
        nei_index_t = nei_index.view(self.N * self.N)  # [N*N]
        corr_t = corr_index.contiguous().view((self.N * self.N, -1))  # [N*N, D]
        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch
            return hidden_state, cn
        r_t = self.relativeLayer(corr_t[nei_index_t > 0])  # [N*N, D]
        inputs_part = nei_inputs[nei_index_t > 0].float()
        hi_t = (
            nei_inputs.view((self.N, self.N, self.hidden_size))
            .permute(1, 0, 2)
            .contiguous()
            .view(-1, self.hidden_size)
        )  # [N*N, D]
        tmp = torch.cat((r_t, hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)  # [N*N, 3*D]
        # Motion Gate
        nGate = self.ngate(tmp).float()  # [N*N, D]
        # Attention
        Pos_t = torch.full((self.N * self.N, 1), 0, device=device).view(-1).float()
        tt = (
            self.WAr(torch.cat((r_t, hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1))
            .view(-1)
            .float()
        )  # [N*N, 1]
        # have bug if there's any zero value in tt
        Pos_t[nei_index_t > 0] = tt
        Pos = Pos_t.view((self.N, self.N))
        Pos[Pos == 0] = -10000
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)
        # Message Passing
        H = torch.full((self.N * self.N, self.D), 0, device=device).float()
        H[nei_index_t > 0] = inputs_part * nGate
        H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(
            self.D, 1
        ).transpose(0, 1)
        H = H.view(self.N, self.N, -1)  # [N, N, D]
        H_sum = self.weight(torch.sum(H, 1))  # [N, D]
        # Update hidden states
        C = H_sum + cn  # [N, D]
        H = hidden_state + F.tanh(C)  # [N, D]
        return H, C


class Laplacian_Decoder(nn.Module):
    def __init__(self, args):
        super(Laplacian_Decoder, self).__init__()
        self.args = args
        if args.mlp_decoder:
            self._decoder = MLPDecoder(args)
        else:
            self._decoder = GRUDecoder(args)

    def forward(self, x_encode, hidden_state, cn, epoch):
        mdn_out = self._decoder(x_encode, hidden_state, cn)
        loc, scale, pi = mdn_out  # [F, N, H, 2], [F, N, H, 2], [N, F]
        return (loc, scale, pi)


# ---------------------------------------------------------------------------
# laplace_decoder.py
# ---------------------------------------------------------------------------
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif "weight_hh" in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif "weight_hr" in name:
                nn.init.xavier_uniform_(param)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
            elif "bias_hh" in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif "weight_hh" in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
            elif "bias_hh" in name:
                nn.init.zeros_(param)


class GRUDecoder(nn.Module):
    def __init__(self, args) -> None:
        super(GRUDecoder, self).__init__()
        min_scale: float = 1e-3
        self.args = args
        self.input_size = self.args.hidden_size
        self.hidden_size = self.args.hidden_size
        self.future_steps = args.pred_length
        self.num_modes = args.final_mode
        self.min_scale = min_scale
        self.args = args
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False,
        )
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2),
        )
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2),
        )
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1),
        )
        self.multihead_proj_global = nn.Sequential(
            nn.Linear(self.input_size, self.num_modes * self.hidden_size),
            nn.LayerNorm(self.num_modes * self.hidden_size),
            nn.ReLU(inplace=True),
        )
        self.apply(init_weights)

    def forward(self, global_embed: torch.Tensor, hidden_state, cn):
        global_embed = self.multihead_proj_global(global_embed).view(
            -1, self.num_modes, self.hidden_size
        )  # [N, F, D]
        global_embed = global_embed.transpose(0, 1)  # [F, N, D]
        local_embed = hidden_state.repeat(self.num_modes, 1, 1)  # [F, N, D]
        cn = cn.repeat(self.num_modes, 1, 1)  # [F, N, D]
        pi = self.pi(torch.cat((local_embed, global_embed), dim=-1)).squeeze(-1).t()  # [N, F]
        global_embed = global_embed.reshape(-1, self.hidden_size)  # [F x N, D]
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.reshape(-1, self.input_size).unsqueeze(0)  # [1, F x N, D]
        cn = cn.reshape(-1, self.input_size).unsqueeze(0)  # [1, F x N, D]
        out, _ = self.lstm(global_embed, (local_embed, cn))
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out)  # [F x N, H, 2]
        scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
        loc = loc.view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        scale = scale.view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        return (loc, scale, pi)  # [F, N, H, 2], [F, N, H, 2], [N, F]


class MLPDecoder(nn.Module):
    def __init__(self, args) -> None:
        super(MLPDecoder, self).__init__()
        min_scale: float = 1e-3
        self.args = args
        self.input_size = self.args.hidden_size
        self.hidden_size = self.args.hidden_size
        self.future_steps = args.pred_length
        self.num_modes = args.final_mode
        self.min_scale = min_scale
        self.args = args
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2),
        )
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2),
        )
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1),
        )
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
        )
        self.multihead_proj_global = nn.Sequential(
            nn.Linear(self.input_size, self.num_modes * self.hidden_size),
            nn.LayerNorm(self.num_modes * self.hidden_size),
            nn.ReLU(inplace=True),
        )
        self.apply(init_weights)

    def forward(self, x_encode: torch.Tensor, hidden_state, cn):
        x_encode = self.multihead_proj_global(x_encode).view(
            -1, self.num_modes, self.hidden_size
        )  # [N, F, D]
        x_encode = x_encode.transpose(0, 1)  # [F, N, D]
        local_embed = hidden_state.repeat(self.num_modes, 1, 1)  # [F, N, D]
        pi = self.pi(torch.cat((local_embed, x_encode), dim=-1)).squeeze(-1).t()  # [N, F]
        out = self.aggr_embed(torch.cat((x_encode, local_embed), dim=-1))
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        scale = (
            F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
        )
        scale = scale + self.min_scale  # [F, N, H, 2]
        return (loc, scale, pi)  # [F, N, H, 2], [F, N, H, 2], [N, F]


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------
class SoftTargetCrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == "mean":
            return cross_entropy.mean()
        elif self.reduction == "sum":
            return cross_entropy.sum()
        elif self.reduction == "none":
            return cross_entropy
        else:
            raise ValueError("{} is not a valid value for reduction".format(self.reduction))


class LaplaceNLLLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "none":
            return nll
        else:
            raise ValueError("{} is not a valid value for reduction".format(self.reduction))


class GaussianNLLLoss(nn.Module):
    """https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html"""

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = 0.5 * (torch.log(scale**2) + torch.abs(target - loc) ** 2 / scale**2)
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "none":
            return nll
        else:
            raise ValueError("{} is not a valid value for reduction".format(self.reduction))


class GATraj(nn.Module):
    def __init__(self, args):
        super(GATraj, self).__init__()
        self.args = args
        self.Temperal_Encoder = Temperal_Encoder(self.args)
        self.Laplacian_Decoder = Laplacian_Decoder(self.args)
        if self.args.SR:
            message_passing = []
            for i in range(self.args.pass_time):
                message_passing.append(Global_interaction(args))
            self.Global_interaction = nn.ModuleList(message_passing)
        if self.args.ifGaussian:
            self.reg_loss = GaussianNLLLoss(reduction="mean")
        else:
            self.reg_loss = LaplaceNLLLoss(reduction="mean")
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction="mean")

    def forward(self, inputs, epoch, iftest=False):
        device = next(self.parameters()).device
        batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split = (
            inputs  # [H, N, 2], [H, N, 2], [B, H, N, N], [N, H], [B, 2]
        )
        self.batch_norm_gt = batch_norm_gt
        if self.args.input_offset:
            train_x = (
                batch_norm_gt[1 : self.args.obs_length, :, :]
                - batch_norm_gt[: self.args.obs_length - 1, :, :]
            )  # [H, N, 2]
        elif self.args.input_mix:
            offset = (
                batch_norm_gt[1 : self.args.obs_length, :, :]
                - batch_norm_gt[: self.args.obs_length - 1, :, :]
            )  # [H, N, 2]
            position = batch_norm_gt[: self.args.obs_length, :, :]  # [H, N, 2]
            pad_offset = torch.zeros_like(position).to(device)
            pad_offset[1:, :, :] = offset
            train_x = torch.cat((position, pad_offset), dim=2)
        elif self.args.input_position:
            train_x = batch_norm_gt[: self.args.obs_length, :, :]  # [H, N, 2]
        train_x = train_x.permute(1, 2, 0)  # [N, 2, H]
        train_y = batch_norm_gt[self.args.obs_length :, :, :].permute(1, 2, 0)  # [N, 2, H]
        self.pre_obs = batch_norm_gt[1 : self.args.obs_length]
        self.x_encoded_dense, self.hidden_state_unsplited, cn = self.Temperal_Encoder.forward(
            train_x
        )  # [N, D], [N, D]
        self.hidden_state_global = torch.ones_like(self.hidden_state_unsplited, device=device)
        cn_global = torch.ones_like(cn, device=device)
        if self.args.SR:
            for b in range(len(nei_list_batch)):
                left, right = batch_split[b][0], batch_split[b][1]
                element_states = self.hidden_state_unsplited[left:right]  # [N, D]
                cn_state = cn[left:right]  # [N, D]
                if element_states.shape[0] != 1:
                    corr = batch_abs_gt[self.args.obs_length - 1, left:right, :2].repeat(
                        element_states.shape[0], 1, 1
                    )  # [N, N, D]
                    corr_index = corr.transpose(0, 1) - corr  # [N, N, D]
                    nei_num = nei_num_batch[left:right, self.args.obs_length - 1]  # [N]
                    nei_index = (
                        nei_list_batch[b][self.args.obs_length - 1].clone().detach().to(device)
                    )  # [N, N]
                    for i in range(self.args.pass_time):
                        element_states, cn_state = self.Global_interaction[i](
                            corr_index, nei_index, nei_num, element_states, cn_state
                        )
                    self.hidden_state_global[left:right] = element_states
                    cn_global[left:right] = cn_state
                else:
                    self.hidden_state_global[left:right] = element_states
                    cn_global[left:right] = cn_state
        else:
            self.hidden_state_global = self.hidden_state_unsplited
            cn_global = cn
        mdn_out = self.Laplacian_Decoder.forward(
            self.x_encoded_dense, self.hidden_state_global, cn_global, epoch
        )
        GATraj_loss, full_pre_tra = self.mdn_loss(
            train_y.permute(2, 0, 1), mdn_out, 1, iftest
        )  # [K, H, N, 2]
        return GATraj_loss, full_pre_tra

    def mdn_loss(self, y, y_prime, goal_gt, iftest):
        batch_size = y.shape[1]
        y = y.permute(1, 0, 2)  # [N, H, 2]
        # [F, N, H, 2], [F, N, H, 2], [N, F]
        out_mu, out_sigma, out_pi = y_prime
        y_hat = torch.cat((out_mu, out_sigma), dim=-1)
        reg_loss, cls_loss = 0, 0
        full_pre_tra = []
        l2_norm = (torch.norm(out_mu - y, p=2, dim=-1)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(batch_size)]
        reg_loss += self.reg_loss(y_hat_best, y)
        soft_target = F.softmax(-l2_norm / self.args.pred_length, dim=0).t().detach()  # [N, F]
        cls_loss += self.cls_loss(out_pi, soft_target)
        loss = reg_loss + cls_loss
        # best ADE
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  # [H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs, sample_k), axis=0))
        # best FDE
        l2_norm_FDE = torch.norm(out_mu[:, :, -1, :] - y[:, -1, :], p=2, dim=-1)  # [F, N]
        best_mode = l2_norm_FDE.argmin(dim=0)
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  # [H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs, sample_k), axis=0))
        return loss, full_pre_tra


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------
class _GATrajArgs:
    """Minimal stand-in for the argparse.Namespace GATraj expects, shrunk for a
    fast tiny build (defaults per the official train.py argparse block, scaled down)."""

    def __init__(self):
        self.SR = True
        self.pass_time = 1  # paper default: 2
        self.final_mode = 3  # paper default: 20
        self.mlp_decoder = False
        self.input_offset = True
        self.input_position = False
        self.input_mix = False
        self.ifGaussian = False
        self.z_dim = 8  # paper default: 32
        self.hidden_size = 16  # paper default: 64
        self.x_encoder_layers = 1  # paper default: 3
        self.x_encoder_head = 4  # paper default: 8
        self.obs_length = 4  # paper default: 8
        self.pred_length = 3  # paper default: 12


def build_gatraj():
    return GATraj(_GATrajArgs())


def example_input_gatraj():
    args = _GATrajArgs()
    seq_length = args.obs_length + args.pred_length
    n_agents = 5  # total pedestrians/agents across the (single) batch element

    batch_abs_gt = torch.randn(seq_length, n_agents, 2)
    batch_norm_gt = torch.randn(seq_length, n_agents, 2)

    # nei_list_batch: list (len = num batch elements) of [seq_length, N, N] neighbor-exists
    # flags for that batch element's agent slice. Use one batch element covering all agents.
    nei_list = torch.randint(0, 2, (seq_length, n_agents, n_agents)).float()
    nei_list_batch = [nei_list]

    # nei_num_batch: [N, seq_length] neighbor counts per agent per frame.
    nei_num_batch = nei_list[..., 0].new_zeros((n_agents, seq_length))
    nei_num_batch += nei_list.sum(dim=-1).permute(1, 0)

    # batch_split: [B, 2] start/end agent-index bounds per batch element.
    batch_split = torch.tensor([[0, n_agents]])

    inputs = (batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split)
    epoch = 1
    return (inputs, epoch)


MENAGERIE_ENTRIES = [
    ("GATraj", "build_gatraj", "example_input_gatraj", 2023, "vendored-pytorch"),
]
