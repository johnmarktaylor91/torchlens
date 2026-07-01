# SOURCE: vendored from samsledje/D-SCRIPT @ 23cbb0fbbb454d09ee41ff749c1454d8b3c8a4b8
# https://raw.githubusercontent.com/samsledje/D-SCRIPT/main/dscript/models/interaction.py
# https://raw.githubusercontent.com/samsledje/D-SCRIPT/main/dscript/models/contact.py
# https://raw.githubusercontent.com/samsledje/D-SCRIPT/main/dscript/models/embedding.py
#
# Sledzieski, Singh, Cowen, Berger. "D-SCRIPT translates genome to phenome with
# sequence-based, structure-aware, genome-scale predictions of protein-protein
# interactions" (Cell Systems, 2021). `DSCRIPTModel` (the real trainable top-level class,
# used e.g. by `dscript/commands/train.py`'s `interaction.DSCRIPTModel(**model_args)`) is
# `ModelInteraction` + `huggingface_hub.PyTorchModelHubMixin` for HF Hub push/pull. It
# wires a `FullyConnectedEmbed` projection (Linear+activation+dropout over language-model
# embeddings) into a `ContactCNN` (`FullyConnected` outer-difference/outer-product broadcast
# -> Conv2d -> `LogisticActivation`) that predicts a pairwise residue contact map, which
# `map_predict`/`predict` then reduce (learned Gaussian positional weighting `W` via
# `theta`/`lambda_`, quantile-style thresholded mean-pool with `gamma`, and a final
# `LogisticActivation` sigmoid) into a scalar interaction probability. All three files are
# copied verbatim; only the module docstrings' math markup is left as-is (harmless at
# runtime) and no forward/architecture code was altered.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch.nn.utils.rnn import PackedSequence


class IdentityEmbed(nn.Module):
    """
    Does not reduce the dimension of the language model embeddings, just passes them through to the contact model.
    """

    def forward(self, x):
        return x


class FullyConnectedEmbed(nn.Module):
    """
    Protein Projection Module. Takes embedding from language model and outputs low-dimensional interaction aware projection.
    """

    def __init__(self, nin, nout, dropout=0.5, activation=nn.ReLU()):
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.dropout_p = dropout

        self.transform = nn.Linear(nin, nout)
        self.drop = nn.Dropout(p=self.dropout_p)
        self.activation = activation

    def forward(self, x):
        t = self.transform(x)
        t = self.activation(t)
        t = self.drop(t)
        return t


class SkipLSTM(nn.Module):
    """
    Language model from Bepler & Berger. Loaded with pre-trained weights in embedding function.
    """

    def __init__(self, nin, nout, hidden_dim, num_layers, dropout=0, bidirectional=True):
        super().__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        dim = nin
        for i in range(num_layers):
            f = nn.LSTM(
                dim,
                hidden_dim,
                1,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.layers.append(f)
            if bidirectional:
                dim = 2 * hidden_dim
            else:
                dim = hidden_dim

        n = hidden_dim * num_layers + nin
        if bidirectional:
            n = 2 * hidden_dim * num_layers + nin

        self.proj = nn.Linear(n, nout)

    def to_one_hot(self, x):
        packed = type(x) is PackedSequence
        if packed:
            one_hot = x.data.new(x.data.size(0), self.nin).float().zero_()
            one_hot.scatter_(1, x.data.unsqueeze(1), 1)
            one_hot = PackedSequence(one_hot, x.batch_sizes)
        else:
            one_hot = x.new(x.size(0), x.size(1), self.nin).float().zero_()
            one_hot.scatter_(2, x.unsqueeze(2), 1)
        return one_hot

    def transform(self, x):
        one_hot = self.to_one_hot(x)
        hs = [one_hot]
        h_ = one_hot
        for f in self.layers:
            h, _ = f(h_)
            hs.append(h)
            h_ = h
        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            h = PackedSequence(h, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
        return h

    def forward(self, x):
        one_hot = self.to_one_hot(x)
        hs = [one_hot]
        h_ = one_hot

        for f in self.layers:
            h, _ = f(h_)
            hs.append(h)
            h_ = h

        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
            z = self.proj(h.view(-1, h.size(2)))
            z = z.view(x.size(0), x.size(1), -1)

        return z


class FullyConnected(nn.Module):
    """
    Performs part 1 of Contact Prediction Module. Takes embeddings from Projection module and produces broadcast tensor.
    """

    def __init__(self, embed_dim, hidden_dim, activation=nn.ReLU()):
        super().__init__()

        self.D = embed_dim
        self.H = hidden_dim
        self.conv = nn.Conv2d(2 * self.D, self.H, 1)
        self.batchnorm = nn.BatchNorm2d(self.H)
        self.activation = activation

    def forward(self, z0, z1):
        # z0 is (b,N,d), z1 is (b,M,d)
        z0 = z0.transpose(1, 2)
        z1 = z1.transpose(1, 2)

        # z0 is (b,d,N), z1 is (b,d,M)
        z_dif = torch.abs(z0.unsqueeze(3) - z1.unsqueeze(2))  # (b, d, N)
        z_mul = z0.unsqueeze(3) * z1.unsqueeze(2)
        z_cat = torch.cat([z_dif, z_mul], 1)

        c = self.conv(z_cat)
        c = self.activation(c)
        c = self.batchnorm(c)

        return c


class ContactCNN(nn.Module):
    """
    Residue Contact Prediction Module. Takes embeddings from Projection module and produces contact map, output of Contact module.
    """

    def __init__(self, embed_dim, hidden_dim=50, width=7, activation=nn.Sigmoid()):
        super().__init__()

        self.hidden = FullyConnected(embed_dim, hidden_dim)

        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.batchnorm = nn.BatchNorm2d(1)
        self.activation = activation
        self.clip()

    def clip(self):
        """
        Force the convolutional layer to be transpose invariant.
        """
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, z0, z1):
        C = self.cmap(z0, z1)
        return self.predict(C)

    def cmap(self, z0, z1):
        C = self.hidden(z0, z1)
        return C

    def predict(self, C):
        # S is (b,N,M)
        s = self.conv(C)
        s = self.batchnorm(s)
        s = self.activation(s)
        return s


class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    """

    def __init__(self, x0=0, k=1, train=False):
        super().__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requires_grad = train

    def forward(self, x):
        o = torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1)
        return o

    def clip(self):
        """
        Restricts sigmoid slope k to be greater than or equal to 0, if k is trained.
        """
        self.k.data.clamp_(min=0)


class ModelInteraction(nn.Module):
    def __init__(
        self,
        embedding,
        contact,
        use_cuda,
        do_w=True,
        do_sigmoid=True,
        do_pool=False,
        pool_size=9,
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
    ):
        """
        Main D-SCRIPT model. Contains an embedding and contact model and offers access to those models. Computes pooling operations on contact map to generate interaction probability.
        """
        super().__init__()
        self.use_cuda = use_cuda
        self.do_w = do_w
        self.do_sigmoid = do_sigmoid
        if do_sigmoid:
            self.activation = LogisticActivation(x0=0.5, k=20)

        self.embedding = embedding
        self.contact = contact

        if self.do_w:
            self.theta = nn.Parameter(torch.FloatTensor([theta_init]))
            self.lambda_ = nn.Parameter(torch.FloatTensor([lambda_init]))

        self.do_pool = do_pool
        self.pool_size = pool_size
        self.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)

        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))

        self.clip()

        self.xx = nn.Parameter(torch.arange(2000), requires_grad=False)

    def clip(self):
        """
        Clamp model values
        """
        self.contact.clip()

        if self.do_w:
            self.theta.data.clamp_(min=0, max=1)
            self.lambda_.data.clamp_(min=0)

        self.gamma.data.clamp_(min=0)

    def embed(self, x):
        if self.embedding is None:
            return x
        else:
            return self.embedding(x)

    def cpred(
        self,
        z0,
        z1,
        embed_foldseek=False,
        f0=None,
        f1=None,
    ):
        e0 = self.embed(z0)
        e1 = self.embed(z1)

        if embed_foldseek:
            assert f0 is not None and f1 is not None
            assert isinstance(f0, torch.Tensor) and isinstance(f1, torch.Tensor)
            assert z0.get_device() == f0.get_device() and z0.get_device() == f1.get_device()
            assert f0.shape[1] == z0.shape[1] and f1.shape[1] == z1.shape[1]

            # concatenate foldseek one hot embedding
            e0 = torch.concat([e0, f0], dim=2)
            e1 = torch.concat([e1, f1], dim=2)

        B = self.contact.cmap(e0, e1)
        C = self.contact.predict(B)
        return C

    def map_predict(
        self,
        z0,
        z1,
        embed_foldseek=False,
        f0=None,
        f1=None,
    ):
        if embed_foldseek:
            assert f0 is not None and f1 is not None
            assert isinstance(f0, torch.Tensor) and isinstance(f1, torch.Tensor)
            assert z0.get_device() == f0.get_device() and z0.get_device() == f1.get_device()
            assert f0.shape[1] == z0.shape[1] and f1.shape[1] == z1.shape[1]

        C = self.cpred(z0, z1, embed_foldseek, f0, f1)

        if self.do_w:
            N, M = C.shape[2:]

            x1 = -1 * torch.square((self.xx[:N] + 1 - ((N + 1) / 2)) / (-1 * ((N + 1) / 2)))

            x2 = -1 * torch.square((self.xx[:M] + 1 - ((M + 1) / 2)) / (-1 * ((M + 1) / 2)))

            x1 = torch.exp(self.lambda_ * x1)
            x2 = torch.exp(self.lambda_ * x2)

            W = x1.unsqueeze(1) * x2
            W = (1 - self.theta) * W + self.theta
            yhat = C * W

        else:
            yhat = C

        if self.do_pool:
            yhat = self.maxPool(yhat)

        # Mean of contact predictions where p_ij > mu + gamma*sigma
        mu = torch.mean(yhat)
        sigma = torch.var(yhat)
        Q = torch.relu(yhat - mu - (self.gamma * sigma))
        phat = torch.sum(Q) / (torch.sum(torch.sign(Q)) + 1)
        if self.do_sigmoid:
            phat = self.activation(phat).squeeze()
        return C, phat

    def predict(self, z0, z1, embed_foldseek=False, f0=None, f1=None):
        _, phat = self.map_predict(z0, z1, embed_foldseek=embed_foldseek, f0=f0, f1=f1)
        return phat

    def forward(self, z0, z1, embed_foldseek=False, f0=None, f1=None):
        return self.predict(z0, z1, embed_foldseek=embed_foldseek, f0=f0, f1=f1)


class DSCRIPTModel(ModelInteraction, PyTorchModelHubMixin):
    def __init__(
        self,
        emb_nin,
        emb_nout,
        emb_dropout,
        con_embed_dim,
        con_hidden_dim,
        con_width,
        use_cuda,
        emb_activation=nn.ReLU(),
        con_activation=nn.Sigmoid(),
        do_w=True,
        do_sigmoid=True,
        do_pool=False,
        pool_size=9,
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
    ):
        embedding = FullyConnectedEmbed(emb_nin, emb_nout, emb_dropout, emb_activation)
        contact = ContactCNN(con_embed_dim, con_hidden_dim, con_width, con_activation)
        super().__init__(
            embedding=embedding,
            contact=contact,
            use_cuda=use_cuda,
            do_w=do_w,
            do_sigmoid=do_sigmoid,
            do_pool=do_pool,
            pool_size=pool_size,
            theta_init=theta_init,
            lambda_init=lambda_init,
            gamma_init=gamma_init,
        )


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_dscript() -> nn.Module:
    # emb_nin=100 mimics a small language-model embedding width (real usage feeds
    # 6165-dim Bepler&Berger SkipLSTM embeddings or ESM embeddings); emb_nout/con_embed_dim
    # shrunk from real defaults (100->16) and con_hidden_dim (50->8) purely to keep the
    # trace fast -- con_width=7 and do_w/do_sigmoid/do_pool left at real defaults.
    model = DSCRIPTModel(
        emb_nin=100,
        emb_nout=16,
        emb_dropout=0.5,
        con_embed_dim=16,
        con_hidden_dim=8,
        con_width=7,
        use_cuda=False,
    )
    model.eval()
    return model


def example_input_dscript():
    # z0/z1: (batch, seq_len, emb_nin) per-residue language-model embeddings for the two
    # candidate interacting proteins, exactly what `DSCRIPTModel.forward(z0, z1)` /
    # `predict.py`'s `model.map_predict(...)` consumes.
    batch = 1
    n0, n1 = 12, 9
    z0 = torch.randn(batch, n0, 100)
    z1 = torch.randn(batch, n1, 100)
    return (z0, z1)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("D-SCRIPT", "build_dscript", "example_input_dscript", 2021, "vendored"),
]
