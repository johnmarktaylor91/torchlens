# SOURCE: vendored from calico/scnym @ master
# https://github.com/calico/scnym/blob/master/scnym/model.py
#
# scNym CellTypeCLF: a fully-connected residual classifier for single-cell
# RNA-seq cell-type annotation (semi-supervised training via MixMatch/DAN in
# the original repo, but the network itself is this plain feed-forward
# classifier). `ResBlock` and `CellTypeCLF` are copied verbatim; the file's
# other domain-adversarial (`DANN`/`GradReverse`) and autoencoder (`AE`)
# classes are unrelated model variants not needed to trace `CellTypeCLF` and
# are omitted.

import torch
import torch.nn as nn


MENAGERIE_ZOO = "vendored-pytorch"


class ResBlock(nn.Module):
    """Residual block.

    References
    ----------
    Deep Residual Learning for Image Recognition
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    arXiv:1512.03385
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
    ) -> None:
        """Residual block with fully-connected neural network
        layers.

        Parameters
        ----------
        n_inputs : int
            number of input dimensions.
        n_hidden : int
            number of hidden dimensions in the Residual Block.

        Returns
        -------
        None.
        """
        super(ResBlock, self).__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        # Build the initial projection layer
        self.linear00 = nn.Linear(self.n_inputs, self.n_hidden)
        self.norm00 = nn.BatchNorm1d(num_features=self.n_hidden)
        self.relu00 = nn.ReLU(inplace=True)

        # Map from the latent space to output space
        self.linear01 = nn.Linear(self.n_hidden, self.n_hidden)
        self.norm01 = nn.BatchNorm1d(num_features=self.n_hidden)
        self.relu01 = nn.ReLU(inplace=True)
        return

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Residual block forward pass.

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, self.n_inputs]

        Returns
        -------
        o : torch.FloatTensor
            [Batch, self.n_hidden]
        """
        identity = x

        # Project input to the latent space
        o = self.norm00(self.linear00(x))
        o = self.relu00(o)

        # Project from the latent space to output space
        o = self.norm01(self.linear01(o))

        # Make this a residual connection
        # by additive identity operation
        o += identity
        return self.relu01(o)


class CellTypeCLF(nn.Module):
    """Cell type classifier from expression data.

    Attributes
    ----------
    n_genes : int
        number of input genes in the model.
    n_cell_types : int
        number of output classes in the model.
    n_hidden : int
        number of hidden units in the model.
    n_layers : int
        number of hidden layers in the model.
    init_dropout : float
        dropout proportion prior to the first layer.
    residual : bool
        use residual connections.
    """

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        n_hidden: int = 256,
        n_hidden_init: int = 256,
        n_layers: int = 2,
        init_dropout: float = 0.0,
        residual: bool = False,
        batch_norm: bool = True,
        track_running_stats: bool = True,
        n_decoder_layers: int = 0,
        use_raw_counts: bool = False,
    ) -> None:
        """
        Cell type classifier from expression data.
        Linear layers with batch norm and dropout.

        Parameters
        ----------
        n_genes : int
            number of genes in the input
        n_cell_types : int
            number of cell types for the output
        n_hidden : int
            number of hidden unit
        n_hidden_init :
            number of hidden units for the initial encoding layer.
        n_layers : int
            number of hidden layers.
        init_dropout : float
            dropout proportion prior to the first layer.
        residual : bool
            use residual connections.
        batch_norm : bool
            use batch normalization in hidden layers.
        track_running_stats : bool
            track running statistics in batch norm layers.
        n_decoder_layers : int
            number of layers in the decoder.
        use_raw_counts : bool
            provide raw counts as input.

        Returns
        -------
        None.
        """
        super(CellTypeCLF, self).__init__()

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_hidden = n_hidden
        self.n_hidden_init = n_hidden_init
        self.n_decoder_layers = n_decoder_layers
        self.n_layers = n_layers
        self.init_dropout = init_dropout
        self.residual = residual
        self.batch_norm = batch_norm
        self.track_running_stats = track_running_stats
        self.use_raw_counts = use_raw_counts

        # simulate technical dropout of scRNAseq
        self.init_dropout = nn.Dropout(p=self.init_dropout)

        # Define a vanilla NN layer with batch norm, dropout, ReLU
        vanilla_layer = [
            nn.Linear(self.n_hidden, self.n_hidden),
        ]
        if self.batch_norm:
            vanilla_layer += [
                nn.BatchNorm1d(
                    num_features=self.n_hidden,
                    track_running_stats=self.track_running_stats,
                ),
            ]
        vanilla_layer += [
            nn.Dropout(),
            nn.ReLU(inplace=True),
        ]

        # Define a residual NN layer with batch norm, dropout, ReLU
        residual_layer = [
            ResBlock(self.n_hidden, self.n_hidden),
        ]
        if self.batch_norm:
            residual_layer += [
                nn.BatchNorm1d(
                    num_features=self.n_hidden,
                    track_running_stats=self.track_running_stats,
                ),
            ]

        residual_layer += [
            nn.Dropout(),
            nn.ReLU(inplace=True),
        ]

        # Build the intermediary layers of the model
        if self.residual:
            hidden_layer = residual_layer
        else:
            hidden_layer = vanilla_layer

        hidden_layers = hidden_layer * (self.n_layers - 1)

        # Build the classifier `nn.Module`.
        self.embed = nn.Sequential(
            nn.Linear(self.n_genes, self.n_hidden_init),
            nn.BatchNorm1d(
                num_features=self.n_hidden_init,
                track_running_stats=self.track_running_stats,
            ),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden_init, self.n_hidden),
            nn.BatchNorm1d(
                num_features=self.n_hidden,
                track_running_stats=self.track_running_stats,
            ),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            *hidden_layers,
        )

        dec_hidden = hidden_layer * (self.n_decoder_layers - 1)
        final_clf = nn.Linear(self.n_hidden, self.n_cell_types)
        self.classif = nn.Sequential(
            *dec_hidden,
            final_clf,
        )
        return

    def forward(
        self,
        x: torch.FloatTensor,
        return_embed: bool = False,
    ) -> torch.FloatTensor:
        """Perform a forward pass through the model

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, self.n_genes]
        return_embed : bool
            return the embedding and the class predictions.

        Returns
        -------
        pred : torch.FloatTensor
            [Batch, self.n_cell_types]
        embed : torch.FloatTensor, optional
            [Batch, n_hidden], only returned if `return_embed`.
        """
        # add initial dropout noise
        if self.init_dropout.p > 0 and not self.use_raw_counts:
            # counts are log1p(CPM)
            # expm1 to normed counts
            x = torch.expm1(x)
            x = self.init_dropout(x)
            # renorm to log1p CPM
            size = torch.sum(x, dim=1).reshape(-1, 1)
            prop_input_ = x / size
            norm_input_ = prop_input_ * 1e6
            x = torch.log1p(norm_input_)
        elif self.init_dropout.p > 0 and self.use_raw_counts:
            x = self.init_dropout(x)
        else:
            # we don't need to do initial dropout
            pass
        x_embed = self.embed(x)
        pred = self.classif(x_embed)

        if return_embed:
            r = (
                pred,
                x_embed,
            )
        else:
            r = pred
        return r


def build_scnym():
    return CellTypeCLF(
        n_genes=64,
        n_cell_types=10,
        n_hidden=32,
        n_hidden_init=32,
        n_layers=2,
        init_dropout=0.0,
        residual=True,
    )


def example_input_scnym():
    batch = 8
    n_genes = 64
    return (torch.rand(batch, n_genes) * 3.0,)


MENAGERIE_ENTRIES = [
    ("scNym", "build_scnym", "example_input_scnym", 2020, "vendored-pytorch"),
]
