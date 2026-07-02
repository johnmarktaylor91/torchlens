# SOURCE: vendored from lifanchen-simm/transformerCPI @ master
# https://raw.githubusercontent.com/lifanchen-simm/transformerCPI/master/Human,C.elegans/model_glu.py
#
# TransformerCPI (Chen et al., "TransformerCPI: improving compound-protein interaction
# prediction by sequence-based deep learning with self-attention mechanism and label
# reversal experiments", Bioinformatics 2020). A transformer encoder-decoder for
# compound-protein interaction prediction: a GLU-gated CNN "Encoder" extracts protein
# sequence features, a graph-convolution "gcn" step propagates atom features over the
# molecular adjacency graph, and a "Decoder" (cross-attention transformer, protein
# features as the attended-to source, compound atom features as the target) fuses the
# two modalities into an interaction logit. This module vendors the real classes
# verbatim (`SelfAttention`, `Encoder`, `PositionwiseFeedforward`, `DecoderLayer`,
# `Decoder`, `Predictor`) from `Human,C.elegans/model_glu.py` -- only the unrelated
# training/testing helper classes (`Trainer`, `Tester`) and sklearn-metric imports are
# dropped since they are not part of the architecture itself. `main_glu.py` in the real
# repo constructs the model with `protein_dim=100, atom_dim=34, hid_dim=64, n_layers=3,
# n_heads=8, pf_dim=256, kernel_size=5/7/9, dropout=0.1` (see lines 61-78); this module
# reuses those exact real hyperparameters for `build_transformercpi()`.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)

        return x


class Encoder(nn.Module):
    """protein feature extraction."""

    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2)
                for _ in range(self.n_layers)
            ]
        )  # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)

    def forward(self, protein):
        # protein = [batch size, protein len, protein_dim]
        conv_input = self.fc(protein)
        # conv_input=[batch size, protein len, hid dim]
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            # conved = [batch size, 2*hid dim, protein len]
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, protein len]
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        # conved = [batch size, protein len, hid dim]
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]
        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]
        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch size, pf dim, sent len]
        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]
        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device
    ):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg


class Decoder(nn.Module):
    """compound feature extraction."""

    def __init__(
        self,
        atom_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        decoder_layer,
        self_attention,
        positionwise_feedforward,
        dropout,
        device,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [
                decoder_layer(
                    hid_dim,
                    n_heads,
                    pf_dim,
                    self_attention,
                    positionwise_feedforward,
                    dropout,
                    device,
                )
                for _ in range(n_layers)
            ]
        )
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)
        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src)

        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant."""
        norm = torch.norm(trg, dim=2)
        # norm = [batch size, compound len]
        norm = F.softmax(norm, dim=1)
        # norm = [batch size, compound len]
        trg = torch.squeeze(trg, dim=0)
        norm = torch.squeeze(norm, dim=0)
        summed = torch.zeros((self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            v = trg[i,]
            v = v * norm[i]
            summed = summed + v
        summed = summed.unsqueeze(dim=0)

        # trg = [batch size, hid_dim]
        label = F.relu(self.fc_1(summed))
        label = self.fc_2(label)
        return label


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=34):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1.0 / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)

    def gcn(self, input, adj):
        # input = [num_node, atom_dim]
        # adj = [num_node, num_node]
        support = torch.mm(input, self.weight)
        # support = [num_node, atom_dim]
        output = torch.mm(adj, support)
        # output = [num_node, atom_dim]
        return output

    def forward(self, compound, adj, protein):
        # compound = [atom_num, atom_dim]
        # adj = [atom_num, atom_num]
        # protein = [protein len, protein_dim]
        compound = self.gcn(compound, adj)
        compound = torch.unsqueeze(compound, dim=0)
        # compound = [batch size=1, atom_num, atom_dim]

        protein = torch.unsqueeze(protein, dim=0)
        # protein = [batch size=1, protein len, protein_dim]
        enc_src = self.encoder(protein)
        # enc_src = [batch size, protein len, hid dim]

        out = self.decoder(compound, enc_src)
        # out = [batch size, 2]
        return out


MENAGERIE_ZOO = "vendored-pytorch"


def build_transformercpi():
    torch.manual_seed(0)
    device = torch.device("cpu")
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    kernel_size = 5
    dropout = 0.1

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(
        atom_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        DecoderLayer,
        SelfAttention,
        PositionwiseFeedforward,
        dropout,
        device,
    )
    model = Predictor(encoder, decoder, device, atom_dim=atom_dim)
    model.eval()
    return model


def example_input_transformercpi():
    torch.manual_seed(0)
    atom_dim = 34
    protein_dim = 100
    num_atoms = 12
    protein_len = 40

    compound = torch.randn(num_atoms, atom_dim)
    # symmetric normalized adjacency-like matrix (real featurizer output shape)
    adj_raw = torch.rand(num_atoms, num_atoms)
    adj = (adj_raw + adj_raw.t()) / 2.0
    protein = torch.randn(protein_len, protein_dim)
    return (compound, adj, protein)


MENAGERIE_ENTRIES = [
    ("TransformerCPI", "build_transformercpi", "example_input_transformercpi", 2020, MENAGERIE_ZOO),
]
