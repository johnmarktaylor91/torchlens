# FAITHFUL PORT of flatironinstitute/DeepFRI @ master (original framework: TensorFlow/Keras)
# Ported files: deepfrier/layers.py (GraphConv, FuncPredictor, SumPooling),
# deepfrier/DeepFRI.py (DeepFRI._build_model).
#
# DeepFRI predicts protein GO terms / EC numbers from a residue contact map (adjacency)
# plus per-residue sequence features, via a stack of Graph Convolution layers (Kipf &
# Welling, ICLR 2017 renormalization trick) over the contact-map graph, concatenated
# across layers, sum-pooled over residues, then an MLP + 2-way-softmax function
# predictor head. The default/paper config (`gc_layer=None` -> "NoGraphConv" is only a
# warned-fallback in the real code; the shipped pretrained models and the paper both use
# `gc_layer='GraphConv'`, so that is the layer ported here) uses n_channels=26 (one-hot
# amino acid + extra channels), gc_dims=[64, 128], fc_dims=[512].
#
# The optional pretrained LSTM language-model branch (`lm_model_name`) is orthogonal
# (a frozen external Keras LSTM concatenated into the AA embedding) and is not exercised
# here (lm_model=None), matching the real code's own default path.
#
# Faithful correspondence:
#   TF Dense(lm_dim, use_bias=False)                -> nn.Linear(n_channels, lm_dim, bias=False)
#   TF Activation('relu')                            -> F.relu
#   TF GraphConv (per-layer renormalized adjacency,
#     A_hat = A - diag(A) + I, D_hat^-1/2 A_hat D_hat^-1/2,
#     batched matmul, dense kernel, ELU)              -> GraphConv module below (identical math)
#   TF Concatenate over gc layer outputs              -> torch.cat(..., dim=-1)
#   TF SumPooling(axis=1)                             -> x.sum(dim=1)
#   TF Dense(fc_dims[l], relu) + Dropout((l+1)*drop)  -> nn.Linear + F.relu + nn.Dropout
#   TF FuncPredictor (Dense(2*output_dim) -> reshape
#     (output_dim, 2) -> softmax(axis=-1))            -> FuncPredictor module below (identical math)
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class GraphConv(nn.Module):
    """Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017).

    Ported from deepfrier/layers.py::GraphConv. Operates on a dense batched
    adjacency/contact-map tensor `A` of shape (B, N, N) and node features `x` of
    shape (B, N, C_in), matching the real Keras layer's `call([x, A])` signature.
    """

    def __init__(self, in_channels, output_dim, use_bias=False):
        super().__init__()
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel = nn.Parameter(torch.empty(in_channels, output_dim))
        nn.init.xavier_uniform_(self.kernel)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _normalize(a, eps=1e-6):
        n = a.shape[-1]
        eye = torch.eye(n, device=a.device, dtype=a.dtype)
        a = a - torch.diag_embed(torch.diagonal(a, dim1=-2, dim2=-1))
        a_hat = a + eye[None, :, :]
        deg = a_hat.sum(dim=2)
        d_hat = torch.diag_embed(1.0 / (eps + torch.sqrt(deg)))
        return torch.matmul(torch.matmul(d_hat, a_hat), d_hat)

    def forward(self, x, adjacency):
        norm_a = self._normalize(adjacency)
        out = torch.bmm(norm_a, x)
        out = torch.matmul(out, self.kernel)
        if self.use_bias:
            out = out + self.bias
        return F.elu(out)


class FuncPredictor(nn.Module):
    """Ported from deepfrier/layers.py::FuncPredictor."""

    def __init__(self, in_features, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.output_layer = nn.Linear(in_features, 2 * output_dim)

    def forward(self, x):
        x = self.output_layer(x)
        x = x.view(x.shape[0], self.output_dim, 2)
        return F.softmax(x, dim=-1)


class DeepFRI(nn.Module):
    """Ported from deepfrier/DeepFRI.py::DeepFRI._build_model (gc_layer='GraphConv',
    lm_model=None branch)."""

    def __init__(
        self,
        output_dim=6,
        n_channels=26,
        gc_dims=(16, 16),
        fc_dims=(32,),
        drop=0.3,
        lm_dim=32,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_channels = n_channels
        self.lm_dim = lm_dim

        self.aa_embedding = nn.Linear(n_channels, lm_dim, bias=False)

        self.gconv_layers = nn.ModuleList()
        in_dim = lm_dim
        for gc_dim in gc_dims:
            self.gconv_layers.append(GraphConv(in_dim, gc_dim, use_bias=False))
            in_dim = gc_dim

        concat_dim = sum(gc_dims)

        self.fc_layers = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()
        fc_in = concat_dim
        for i, fc_dim in enumerate(fc_dims):
            self.fc_layers.append(nn.Linear(fc_in, fc_dim))
            self.fc_dropouts.append(nn.Dropout((i + 1) * drop))
            fc_in = fc_dim

        self.func_predictor = FuncPredictor(fc_in, output_dim)

    def forward(self, input_seq, input_cmap):
        x_aa = self.aa_embedding(input_seq)
        x = F.relu(x_aa)

        gcnn_concat = []
        for gconv in self.gconv_layers:
            x = gconv(x, input_cmap)
            gcnn_concat.append(x)

        x = torch.cat(gcnn_concat, dim=-1) if len(gcnn_concat) > 1 else gcnn_concat[-1]

        x = x.sum(dim=1)

        for fc, drop in zip(self.fc_layers, self.fc_dropouts):
            x = F.relu(fc(x))
            x = drop(x)

        return self.func_predictor(x)


def build_deepfri():
    return DeepFRI(output_dim=6, n_channels=26, gc_dims=(16, 16), fc_dims=(32,), lm_dim=32)


def example_input_deepfri():
    torch.manual_seed(0)
    batch, n_residues, n_channels = 2, 12, 26
    seq = torch.randn(batch, n_residues, n_channels)
    cmap = (torch.rand(batch, n_residues, n_residues) > 0.6).float()
    cmap = torch.triu(cmap, diagonal=1)
    cmap = cmap + cmap.transpose(1, 2)
    return (seq, cmap)


MENAGERIE_ENTRIES = [
    ("DeepFRI", build_deepfri, example_input_deepfri, 2021, MENAGERIE_ZOO),
]
