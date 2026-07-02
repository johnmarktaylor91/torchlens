# SOURCE: vendored from ecust-hc/ScaffoldGVAE @ master (model.py, utils.py)
#
# ScaffoldGVAE: a scaffold-conditioned graph-VAE for de novo molecule generation. A dual
# message-passing encoder (node-central NMPN + edge-central EMPN) produces atom-level
# embeddings, a graph-attention readout splits scaffold vs. side-chain latent codes, a VAE
# reparameterization step samples z, and a 3-layer GRU (MultiGRU) decodes SMILES tokens.
#
# The `DMPN`/`NMPN`/`EMPN`/`MultiGRU` classes below are the REAL model.py code, copied
# verbatim except:
#   - `create_var` (utils.py) unconditionally called `.cuda()`; this is CPU/GPU-agnostic
#     here (uses whatever device the input tensors are already on) since that .cuda() call
#     was a training-script convenience, not part of the architecture.
#   - `mol2graph`/`atom_if_sca` (utils.py) build the atom/bond graph tensors from RDKit
#     molecule objects (SMILES -> Chem.Mol -> atom/bond features). RDKit is not in the base
#     env, so this staging module builds structurally-equivalent atom/bond graph tensors
#     directly (same ATOM_FDIM/BOND_FDIM feature widths, same NMPN/EMPN adjacency-index
#     conventions) instead of parsing real SMILES -- this only replaces the *data
#     featurization* step, not the neural network.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

ELEM_LIST = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "Al",
    "I",
    "B",
    "K",
    "Se",
    "Zn",
    "H",
    "Cu",
    "Mn",
    "unknown",
]
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6


def create_var(tensor, requires_grad=None):
    """Device-agnostic replacement for utils.create_var (original hard-coded .cuda())."""
    if requires_grad is not None:
        tensor = tensor.clone().detach().requires_grad_(requires_grad)
    return tensor


def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)


# ---- Node-central Encoder (real model.py code) ----
class NMPN(nn.Module):
    def __init__(self, hidden_size, depth):
        super(NMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.W_nin = nn.Linear(ATOM_FDIM, hidden_size, bias=False)
        self.W_node = nn.Linear(hidden_size + BOND_FDIM, hidden_size, bias=False)

    def forward(self, mol_graph):
        fatoms, fbonds, aoutgraph, bgraph, aingraph, scope, all_bonds = mol_graph
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        aoutgraph = create_var(aoutgraph)

        h_0 = self.W_nin(fatoms)
        h_0 = nn.ReLU()(h_0)
        h_0 = h_0.t()
        H_n = h_0

        for i in range(self.depth):
            message = self.messagefunction(H_n, fbonds, all_bonds)
            nei_message = index_select_ND(message, 0, aoutgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_node(nei_message).t()
            H_n = nn.ReLU()(h_0 + nei_message)
        return H_n

    def messagefunction(self, H_n, fbonds, all_bonds):
        total_bonds = len(fbonds)
        in_n = []
        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            in_n.append(y)
        in_n = create_var(torch.tensor(in_n))
        message = H_n.index_select(1, in_n).t()
        zero = create_var(torch.unsqueeze(torch.zeros(message.size()[1:]), 0))
        message = torch.cat([zero, message], 0)
        message = torch.cat([message, fbonds], 1)
        return message


# ---- Edge-central Encoder (real model.py code) ----
class EMPN(nn.Module):
    def __init__(self, hidden_size, depth, out):
        super(EMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.out = out
        self.W_ein = nn.Linear(BOND_FDIM, hidden_size, bias=False)
        self.W_edge = nn.Linear(hidden_size + ATOM_FDIM, hidden_size, bias=False)
        self.W_eout = nn.Linear(hidden_size + ATOM_FDIM, out, bias=False)

    def forward(self, mol_graph):
        fatoms, fbonds, aoutgraph, bgraph, aingraph, scope, all_bonds = mol_graph
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        bgraph = create_var(bgraph)
        aingraph = create_var(aingraph)

        h_0 = self.W_ein(fbonds)
        h_0 = nn.ReLU()(h_0)
        H_e = h_0

        for i in range(self.depth):
            message = self.messagefunction(H_e, fatoms, all_bonds)
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_edge(nei_message)
            H_e = nn.ReLU()(h_0 + nei_message)

        message = self.messagefunction(H_e, fatoms, all_bonds)
        nei_message = index_select_ND(message, 0, aingraph)
        nei_message = nei_message.sum(dim=1)
        nei_message = self.W_eout(nei_message)
        H_e = nn.ReLU()(nei_message).t()
        return H_e

    def messagefunction(self, H_e, fatoms, all_bonds):
        total_bonds = len(all_bonds)
        out_n = []
        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            out_n.append(x)
        out_n = create_var(torch.tensor(out_n))
        message = fatoms.index_select(0, out_n)
        zero = create_var(torch.unsqueeze(torch.zeros(message.size()[1:]), 0))
        message = torch.cat([zero, message], 0)
        message = torch.cat([H_e, message], 1)
        return message


# ---- 3-layer GRU decoder (real model.py code) ----
class MultiGRU(nn.Module):
    """Implements a three layer GRU cell including an embedding layer
    and an output linear layer back to the size of the vocabulary"""

    def __init__(self, voc_size, h_size):
        super(MultiGRU, self).__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.gru_1 = nn.GRUCell(128, h_size)
        self.gru_2 = nn.GRUCell(h_size, h_size)
        self.gru_3 = nn.GRUCell(h_size, h_size)
        self.linear = nn.Linear(h_size, voc_size)

    def forward(self, x, h):
        x = self.embedding(x)
        h_out = create_var(torch.zeros(h.size()))
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out


class _Voc:
    """Minimal stand-in matching the real Vocabulary interface used by DMPN (vocab_size, vocab dict)."""

    def __init__(self, vocab_size=40):
        self.vocab_size = vocab_size
        self.vocab = {"GO": 0, "EOS": 1}


# ---- Double MPN + GRU (real model.py DMPN class) ----
class DMPN(nn.Module):
    def __init__(self, hidden_size, depth, out, atten_size, r, d_hid, d_z, voc, ver=False):
        super(DMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.out = out
        self.atten_size = atten_size
        self.r = r
        self.d_hid = d_hid
        self.d_z = d_z
        self.voc = voc

        self.NMPN = NMPN(self.hidden_size, self.depth)
        self.EMPN = EMPN(self.hidden_size, self.depth, self.out)
        self.W_1 = nn.Linear(self.hidden_size + self.out, self.atten_size, bias=False)
        self.W_2 = nn.Linear(self.atten_size, self.r, bias=False)
        self.W_3 = nn.Linear(self.r * (self.hidden_size + self.out), self.d_hid)
        if ver == False:  # noqa: E712 -- verbatim from real model.py
            self.rnn = MultiGRU(voc.vocab_size, (self.d_hid * 2))
            self.q_mu = nn.Linear(self.d_hid, self.d_z)
            self.q_logvar = nn.Linear(self.d_hid, self.d_z)
            self.decoder_lat = nn.Linear(self.d_z, self.d_hid)
        else:
            self.rnn = MultiGRU(voc.vocab_size, (self.d_hid * 2))
            self.q_mu = nn.Linear(self.d_hid * 2, self.d_z)
            self.q_logvar = nn.Linear(self.d_hid * 2, self.d_z)
            self.decoder_lat = nn.Linear(self.d_z, self.d_hid * 2)

    def forward(self, mol_batch, sca_batch, target):
        space_side, space_sca = self.forward_encoder(mol_batch, sca_batch)
        mu, logvar = self.q_mu(space_sca), self.q_logvar(space_sca)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps
        kl_loss = 0.5 * (logvar.exp() + mu**2 - 1 - logvar).sum(1).mean()

        sca_h = self.decoder_lat(z)
        gru_h0 = torch.cat([sca_h, space_side], 1)
        gru_h0 = torch.unsqueeze(gru_h0, 0).repeat([3, 1, 1]).type(torch.float32)

        recon_loss = self.forward_decoder(gru_h0, target)
        return kl_loss, recon_loss

    def read_out(self, h_node, s_sca):
        sca_index = []
        side_index = []
        for i in range(len(s_sca)):
            if s_sca[i] == 1:
                sca_index.append(i)
            else:
                side_index.append(i)

        sca_index = create_var(torch.tensor(sca_index))
        side_index = create_var(torch.tensor(side_index))
        sca_node = h_node.index_select(0, sca_index)
        side_node = h_node.index_select(0, side_index)
        sca_s = F.softmax(self.W_2(nn.Tanh()(self.W_1(sca_node))), 1)
        sca_s = sca_s.t()
        side_s = F.softmax(self.W_2(nn.Tanh()(self.W_1(side_node))), 1)
        side_s = side_s.t()
        sca_embeding = self.W_3(torch.flatten(torch.mm(sca_s, sca_node)))
        side_embeding = self.W_3(torch.flatten(torch.mm(side_s, side_node)))

        return sca_embeding, side_embeding

    def forward_encoder(self, mol_batch, sca_batch):
        # mol_batch/sca_batch are pre-built (mol_graph, S_sca) tuples in this staging
        # module (real repo builds them from SMILES via RDKit's mol2graph/atom_if_sca).
        mol_graph, S_sca = mol_batch
        H_n = self.NMPN(mol_graph)
        H_e = self.EMPN(mol_graph)
        H_node = torch.cat([H_n, H_e], 0).t()

        hidden_space_sca = []
        hidden_space_side = []
        for st, le in mol_graph[5]:
            s_sca = S_sca[st : st + le]
            cur_vecs_sca, cur_vecs_side = self.read_out(H_node[st : st + le], s_sca)
            hidden_space_sca.append(cur_vecs_sca)
            hidden_space_side.append(cur_vecs_side)

        space_sca = torch.stack(hidden_space_sca, 0)
        space_side = torch.stack(hidden_space_side, 0)
        return space_side, space_sca

    def forward_decoder(self, h_origin, target):
        batch_size, seq_length = target.size()
        start_token = create_var(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab["GO"]
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = h_origin

        log_probs = create_var(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            log_prob = F.log_softmax(logits, dim=1)
            log_probs += self.NLLLoss(log_prob, target[:, step])

        return log_probs.mean()

    def NLLLoss(self, inputs, targets):
        target_expanded = torch.zeros(inputs.size())
        target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
        loss = create_var(target_expanded) * inputs
        loss = -torch.sum(loss, 1)
        return loss


def _build_synthetic_mol_graph(n_atoms_per_mol, seed=0):
    """Builds a structurally-valid (fatoms, fbonds, aoutgraph, bgraph, aingraph, scope,
    all_bonds) mol_graph tuple with the same shapes/conventions as utils.mol2graph, using a
    simple deterministic path-graph topology per molecule (bypasses RDKit SMILES parsing,
    which is unavailable in the base env)."""
    g = torch.Generator().manual_seed(seed)
    padding = torch.zeros(BOND_FDIM)
    fatoms, fbonds = [], [padding]
    out_bonds, in_bonds, all_bonds = [], [], [(-1, -1)]
    scope = []
    total_atoms = 0

    for n_atoms in n_atoms_per_mol:
        for _ in range(n_atoms):
            fatoms.append(torch.rand(ATOM_FDIM, generator=g))
            in_bonds.append([])
            out_bonds.append([])
        # simple path-graph bonds: atom i -- atom i+1
        for i in range(n_atoms - 1):
            x = i + total_atoms
            y = i + 1 + total_atoms
            b = len(all_bonds)
            all_bonds.append((x, y))
            fbonds.append(torch.rand(BOND_FDIM, generator=g))
            in_bonds[y].append(b)
            out_bonds[x].append(b)

            b = len(all_bonds)
            all_bonds.append((y, x))
            fbonds.append(torch.rand(BOND_FDIM, generator=g))
            in_bonds[x].append(b)
            out_bonds[y].append(b)

        scope.append((total_atoms, n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    aoutgraph = torch.zeros(total_atoms, MAX_NB).long()
    aingraph = torch.zeros(total_atoms, MAX_NB).long()
    bgraph = torch.zeros(total_bonds, MAX_NB).long()

    for a in range(total_atoms):
        for i, b in enumerate(out_bonds[a]):
            aoutgraph[a, i] = b
        for i, b in enumerate(in_bonds[a]):
            aingraph[a, i] = b

    for b1 in range(1, total_bonds):
        x, y = all_bonds[b1]
        for i, b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1, i] = b2

    return fatoms, fbonds, aoutgraph, bgraph, aingraph, scope, all_bonds


def build_scaffoldgvae():
    """Tiny ScaffoldGVAE (DMPN) for tracing: small NMPN/EMPN encoder + attention readout + GRU decoder."""
    voc = _Voc(vocab_size=30)
    model = DMPN(
        hidden_size=8, depth=2, out=6, atten_size=8, r=2, d_hid=10, d_z=6, voc=voc, ver=False
    )
    model.eval()
    return model


def example_input_scaffoldgvae():
    n_atoms_per_mol = [6, 5]  # 2 molecules in the batch
    mol_graph = _build_synthetic_mol_graph(n_atoms_per_mol, seed=0)
    # first half of atoms in each molecule flagged as "scaffold", rest as "side chain"
    S_sca = []
    for st, le in mol_graph[5]:
        half = max(1, le // 2)
        S_sca.extend([1] * half + [0] * (le - half))
    mol_batch = (mol_graph, S_sca)
    target = torch.randint(2, 30, (len(n_atoms_per_mol), 12), dtype=torch.long)
    return (mol_batch, None, target)


MENAGERIE_ENTRIES = [
    ("ScaffoldGVAE", build_scaffoldgvae, example_input_scaffoldgvae, 2023, "vendored-pytorch"),
]
