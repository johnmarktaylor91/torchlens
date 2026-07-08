# SOURCE: vendored from https://github.com/ml4bio/e2efold @ master
# Files: e2efold/models.py (ContactAttention_simple, ContactAttention_simple_fix_PE,
#        Lag_PP_zero, Lag_PP_mixed, RNA_SS_e2e) + e2efold/common/utils.py (soft_sign, inlined
#        as its containing module additionally pulls in sklearn/pandas/argparse purely for
#        unrelated training/eval helpers not needed to construct or run the model).
# Real construction (e2efold_productive/e2efold_productive_short.py, config.json
# model_type="att_simple_fix", pp_model="mixed"):
#   contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len, device=device)
#   lag_pp_net = Lag_PP_mixed(pp_steps, k, rho_per_position)
#   rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)
# MENAGERIE_ZOO = "vendored-pytorch"
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import diags
from torch.nn.modules.utils import _pair


def soft_sign(x, k):
    # e2efold/common/utils.py::soft_sign, verbatim
    return torch.sigmoid(k * x)


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride=1, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(
                1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, output_size[0], output_size[1]))
        else:
            self.register_parameter("bias", None)
        self.kernel_size = _pair(kernel_size)
        self.pad = int((kernel_size - 1) / 2)
        self.stride = _pair(stride)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class ContactAttention_simple(nn.Module):
    """docstring for ContactAttention_simple"""

    def __init__(self, d, L):
        super(ContactAttention_simple, self).__init__()
        self.d = d
        self.L = L
        self.conv1d1 = nn.Conv1d(
            in_channels=4, out_channels=d, kernel_size=9, padding=8, dilation=2
        )
        self.bn1 = nn.BatchNorm1d(d)

        self.conv_test_1 = nn.Conv2d(in_channels=6 * d, out_channels=d, kernel_size=1)
        self.bn_conv_1 = nn.BatchNorm2d(d)
        self.conv_test_2 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1)
        self.bn_conv_2 = nn.BatchNorm2d(d)
        self.conv_test_3 = nn.Conv2d(in_channels=d, out_channels=1, kernel_size=1)

        self.position_embedding_1d = nn.Parameter(torch.randn(1, d, 600))

        # transformer encoder for the input sequences
        self.encoder_layer = nn.TransformerEncoderLayer(2 * d, 2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)

    def forward(self, prior, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """
        position_embeds = self.position_embedding_1d.repeat(seq.shape[0], 1, 1)
        seq = seq.permute(0, 2, 1)  # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))  # d*L just for increase the capacity

        seq = torch.cat([seq, position_embeds], 1)  # 2d*L
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))
        seq = seq.permute(1, 2, 0)

        # what about apply attention on the the 2d map?
        seq_mat = self.matrix_rep(seq)  # 4d*L*L

        p_mat = self.matrix_rep(position_embeds)  # 2d*L*L

        infor = torch.cat([seq_mat, p_mat], 1)  # 6d*L*L

        contact = F.relu(self.bn_conv_1(self.conv_test_1(infor)))
        contact = F.relu(self.bn_conv_2(self.conv_test_2(contact)))
        contact = self.conv_test_3(contact)

        contact = contact.view(-1, self.L, self.L)
        contact = (contact + torch.transpose(contact, -1, -2)) / 2

        return contact.view(-1, self.L, self.L)

    def matrix_rep(self, x):
        """
        for each position i,j of the matrix, we concatenate the embedding of i and j
        """
        x = x.permute(0, 2, 1)  # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L, 1, 1)
        x2 = x2.repeat(1, 1, L, 1)
        mat = torch.cat([x, x2], -1)  # L*L*2d

        # make it symmetric
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2))  # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag
        return mat


class ContactAttention_simple_fix_PE(ContactAttention_simple):
    """docstring for ContactAttention_simple_fix_PE"""

    def __init__(self, d, L, device):
        super(ContactAttention_simple_fix_PE, self).__init__(d, L)
        self.PE_net = nn.Sequential(
            nn.Linear(111, 5 * d),
            nn.ReLU(),
            nn.Linear(5 * d, 5 * d),
            nn.ReLU(),
            nn.Linear(5 * d, d),
        )

    def forward(self, pe, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """
        position_embeds = self.PE_net(pe.view(-1, 111)).view(-1, self.L, self.d)  # N*L*111 -> N*L*d
        position_embeds = position_embeds.permute(0, 2, 1)  # N*d*L
        seq = seq.permute(0, 2, 1)  # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))  # d*L just for increase the capacity

        seq = torch.cat([seq, position_embeds], 1)  # 2d*L
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))
        seq = seq.permute(1, 2, 0)

        # what about apply attention on the the 2d map?
        seq_mat = self.matrix_rep(seq)  # 4d*L*L

        p_mat = self.matrix_rep(position_embeds)  # 2d*L*L

        infor = torch.cat([seq_mat, p_mat], 1)  # 6d*L*L

        contact = F.relu(self.bn_conv_1(self.conv_test_1(infor)))
        contact = F.relu(self.bn_conv_2(self.conv_test_2(contact)))
        contact = self.conv_test_3(contact)

        contact = contact.view(-1, self.L, self.L)
        contact = (contact + torch.transpose(contact, -1, -2)) / 2

        return contact.view(-1, self.L, self.L)


class Lag_PP_zero(nn.Module):
    """
    The definition of Lagrangian post-procssing with no parameters
    Instantiation:
    :steps: the number of unroll steps
    Input:
    :u: the utility matrix, batch*L*L
    :s: the sequence encoding, batch*L*4

    Output: a list of contact map of each step, batch*L*L
    """

    def __init__(self, steps, k):
        super(Lag_PP_zero, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps
        # the parameter for the soft sign
        self.k = k
        self.s = math.log(9.0)
        self.rho = 1.0
        self.alpha = 0.01
        self.beta = 0.1
        self.lr_decay = 0.99

    def forward(self, u, x):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x)  # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k).detach()
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = F.relu(torch.sum(a_tmp, dim=-1) - 1).detach()

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(self.steps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t
            )

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        return a_t_list[1:]

    def update_rule(self, u, m, lmbd, a, a_hat, t):
        grad_a = -u / 2 + (lmbd * soft_sign(torch.sum(a, dim=-1) - 1, self.k)).unsqueeze_(
            -1
        ).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))

        a_hat_updated = a_hat - self.alpha * grad
        self.alpha *= self.lr_decay
        a_hat_updated = F.relu(torch.abs(a_hat_updated) - self.rho * self.alpha)
        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_updated = lmbd + self.beta * lmbd_grad
        self.beta *= self.lr_decay

        return lmbd_updated, a_updated, a_hat_updated

    def constraint_matrix_batch(self, x):
        base_a = x[:, :, 0]
        base_u = x[:, :, 1]
        base_c = x[:, :, 2]
        base_g = x[:, :, 3]
        batch = base_a.shape[0]
        length = base_a.shape[1]
        au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
        au_ua = au + torch.transpose(au, -1, -2)
        cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
        cg_gc = cg + torch.transpose(cg, -1, -2)
        ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
        ug_gu = ug + torch.transpose(ug, -1, -2)
        m = au_ua + cg_gc + ug_gu

        mask = diags([1] * 7, [-3, -2, -1, 0, 1, 2, 3], shape=(m.shape[-2], m.shape[-1])).toarray()
        m = m.masked_fill(torch.Tensor(mask).bool().to(self.device), 0)
        return m

    def contact_a(self, a_hat, m):
        a = a_hat * a_hat
        a = (a + torch.transpose(a, -1, -2)) / 2
        a = a * m
        return a


class Lag_PP_mixed(Lag_PP_zero):
    """
    For the update of a and lambda, we use gradient descent with
    learnable parameters. For the rho, we use neural network to learn
    a position related threshold
    """

    def __init__(self, steps, k, rho_mode="fix"):
        super(Lag_PP_mixed, self).__init__(steps, k)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps
        self.k = k
        self.s = nn.Parameter(torch.Tensor([math.log(9.0)]))
        self.w = nn.Parameter(torch.randn(1))
        self.rho = nn.Parameter(torch.Tensor([1.0]))
        self.rho_m = nn.Parameter(torch.randn(600, 600))
        self.rho_net = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 1), nn.ReLU())
        # build the rho network, reused under every time step
        self.alpha = nn.Parameter(torch.Tensor([0.01]))
        self.beta = nn.Parameter(torch.Tensor([0.1]))
        self.lr_decay_alpha = nn.Parameter(torch.Tensor([0.99]))
        self.lr_decay_beta = nn.Parameter(torch.Tensor([0.99]))
        self.rho_mode = rho_mode

        pos_j, pos_i = np.meshgrid(np.arange(1, 600 + 1) / 600.0, np.arange(1, 600 + 1) / 600.0)
        self.rho_pos_fea = (
            torch.cat([torch.Tensor(pos_i).unsqueeze(-1), torch.Tensor(pos_j).unsqueeze(-1)], -1)
            .view(-1, 2)
            .to(self.device)
        )

        self.rho_pos_net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1), nn.ReLU())

    def forward(self, u, x):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x)  # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k).detach()
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = self.w * F.relu(torch.sum(a_tmp, dim=-1) - 1).detach()

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(self.steps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t
            )

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        return a_t_list[1:]

    def update_rule(self, u, m, lmbd, a, a_hat, t):
        grad_a = -u / 2 + (lmbd * soft_sign(torch.sum(a, dim=-1) - 1, self.k)).unsqueeze_(
            -1
        ).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))

        a_hat_updated = a_hat - self.alpha * torch.pow(self.lr_decay_alpha, t) * grad

        if self.rho_mode == "nn":
            input_features = torch.cat(
                [torch.unsqueeze(a_hat, -1), torch.unsqueeze(grad, -1), torch.unsqueeze(u, -1)], -1
            ).view(-1, 3)
            rho = self.rho_net(input_features).view(a_hat.shape)
            a_hat_updated = F.relu(torch.abs(a_hat_updated) - rho)
        elif self.rho_mode == "matrix":
            a_hat_updated = F.relu(
                torch.abs(a_hat_updated)
                - self.rho_m * self.alpha * torch.pow(self.lr_decay_alpha, t)
            )
        elif self.rho_mode == "nn_pos":
            rho = self.rho_pos_net(self.rho_pos_fea).view(
                a_hat_updated.shape[-2], a_hat_updated.shape[-1]
            )
            a_hat_updated = F.relu(torch.abs(a_hat_updated) - rho)
        else:
            a_hat_updated = F.relu(
                torch.abs(a_hat_updated) - self.rho * self.alpha * torch.pow(self.lr_decay_alpha, t)
            )

        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_updated = lmbd + self.beta * torch.pow(self.lr_decay_beta, t) * lmbd_grad

        return lmbd_updated, a_updated, a_hat_updated


class RNA_SS_e2e(nn.Module):
    def __init__(self, model_att, model_pp):
        super(RNA_SS_e2e, self).__init__()
        self.model_att = model_att
        self.model_pp = model_pp

    def forward(self, prior, seq, state):
        u = self.model_att(prior, seq, state)
        map_list = self.model_pp(u, seq)
        return u, map_list


def build_e2efold():
    # real construction from e2efold_productive/e2efold_productive_short.py with
    # config.json ("model_type": "att_simple_fix", "pp_model": "mixed", "u_net_d": 10,
    # "pp_steps": 20, "k": 1); seq_len shrunk for a tiny trace. rho_mode='matrix' bakes in a
    # hardcoded 600x600 buffer tied to the repo's max training length (600), so for a small
    # catalog trace we use the 'fix' branch instead -- a real, already-present config value
    # of the same Lag_PP_mixed class (not an architecture change), matching the
    # "rho_per_position" ablation the repo itself supports alongside "matrix"/"nn"/"nn_pos".
    torch.manual_seed(0)
    device = torch.device("cpu")
    seq_len = 32
    d = 10
    contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len, device=device)
    lag_pp_net = Lag_PP_mixed(steps=4, k=1, rho_mode="fix")
    # Lag_PP_zero.__init__ (the base class) hardcodes
    # `self.device = "cuda" if torch.cuda.is_available() else "cpu"` regardless of where the
    # module ends up living; on a CUDA-capable but CPU-tracing host that pins an internal mask
    # buffer to cuda while the module/inputs stay on cpu. Not an architecture change -- just
    # aligning the module's internal device bookkeeping with where we actually run it.
    lag_pp_net.device = device
    rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)
    return rna_ss_e2e


def example_input_e2efold():
    torch.manual_seed(0)
    seq_len = 32
    pe = torch.randn(1, seq_len, 111)
    seq = F.one_hot(torch.randint(0, 4, (1, seq_len)), num_classes=4).float()
    state = torch.zeros(1, seq_len, seq_len)
    return pe, seq, state


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    ("e2efold_rna_ss", "build_e2efold", "example_input_e2efold", 2020, "vendored-pytorch"),
]
