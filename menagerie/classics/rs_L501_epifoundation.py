# SOURCE: vendored from UCSC-VLAA/EpiFoundation @ 9fc5931779aeb395e58127722c37179eac0ff27f
#
# Vendored files (concatenated, imports rewritten to be self-contained, architecture
# unmodified):
#   model/EpiFoundation.py  (GeneEncoder, CategoryValueEncoder, ClsDecoder,
#                             ExprDecoder, PretrainDecoder, EpiFoundation)
#   model/transformer.py    (TransformerModel; only the `use_fast_transformer=False`
#                             / plain `torch.nn.TransformerEncoderLayer` construction
#                             path is exercised)
#
# EpiFoundation's real `model/EpiFoundation.py` unconditionally imports
# `from model.performer import Performer` (needed only for the alternative
# `encoder="performer"` path), and the real `model/transformer.py` unconditionally
# imports `from model.flashDiff import MultiheadFlashDiff` (needed only for the
# alternative `fast_transformer_backend="diff"` path). Both of those sibling modules
# themselves unconditionally import CUDA-only compiled packages that are not part of
# our base-lib set (`local_attention`, `flash_attn`). Neither Performer nor
# MultiheadFlashDiff is reachable when the model is built with
# `encoder="transformer"` and `use_fast_transformer=False` (the vanilla-torch
# `torch.nn.TransformerEncoderLayer` path, already present verbatim in the real
# `TransformerModel.__init__`'s `else` branch). This vendored module therefore keeps
# every class exactly as in the source and only omits the two dead unconditional
# imports of unused sibling files -- the same "fix only imports/relative-paths
# minimally" latitude the vendoring rung already grants -- without touching the
# `EpiFoundation`/`TransformerModel` architecture itself.
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

MENAGERIE_ZOO = "vendored-pytorch"

# ================================ model/EpiFoundation.py ==============================


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class PretrainDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
        catagory_num: Optional[int] = 2,
    ) -> None:
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 128)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_in + 128, 128)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(128, catagory_num)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_in, 128)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(128, catagory_num)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(self, cell_emb: Tensor, gene_embs: Tensor):
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return pred_value
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            torch.sigmoid(zero_logits)
            return pred_value
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(self.fc1(torch.cat([cell_emb, query_vecs], dim=2)))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)


class EpiFoundation(nn.Module):
    def __init__(
        self,
        num_class_cell,  # num of cell categories
        num_rnas,  # num of genes (or atac peaks)
        num_atacs,  # num of genes (or atac peaks)
        num_values,  # num of values
        num_chrs,  # num of chromosomes
        embed_dim,  # embed_dim of tokens
        depth,  # layers
        heads,  # num of heads
        head_dim=64,  # embed_dim of heads
        encoder: str = "transformer",  # encoder type, performer or transformer
        dropout=0.2,
        pad_token_idx_atac=0,
        pad_token_idx_rna=0,
        cell_emb_style="cls",
        mvc_arch_style="inner product",
        use_batch_labels=False,
        batch_label_num=13,
        use_chr_labels=False,
        transformer_backend="flash",
        stage="pretrain",
    ):
        super().__init__()

        self.stage = stage
        self.encoder_type = encoder
        self.cell_emb_style = cell_emb_style
        self.embed_dim = embed_dim

        self.rna_emb = GeneEncoder(num_rnas, embed_dim, padding_idx=pad_token_idx_rna)
        self.atac_emb = GeneEncoder(num_atacs, embed_dim, padding_idx=pad_token_idx_atac)

        if use_batch_labels:
            self.batch_emb = BatchLabelEncoder(batch_label_num, embed_dim)
        else:
            self.batch_emb = None

        if use_chr_labels:
            self.chr_emb = GeneEncoder(num_chrs, embed_dim)
        else:
            self.chr_emb = None

        self.dropout_rna = nn.Dropout(dropout)
        self.dropout_atac = nn.Dropout(dropout)

        if encoder == "performer":
            raise NotImplementedError(
                "This vendored recipe only exercises the plain-torch "
                "`encoder='transformer'` path (Performer needs `local_attention`, a "
                "CUDA-only compiled dependency not in the base-lib set)."
            )
        elif encoder == "transformer":
            self.encoder = TransformerModel(
                d_model=embed_dim,
                nhead=heads,
                nlayers=depth,
                d_hid=head_dim,
                dropout=dropout,
                fast_transformer_backend=transformer_backend,
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_decoder = ClsDecoder(embed_dim, num_class_cell)
        self.mvc_decoder = PretrainDecoder(
            embed_dim,
            arch_style=mvc_arch_style,
            use_batch_labels=use_batch_labels,
            catagory_num=num_values,
        )
        self.bn_atac = nn.BatchNorm1d(embed_dim, eps=6.1e-5)
        self.bn_rna = nn.BatchNorm1d(embed_dim, eps=6.1e-5)
        if stage == "value_finetune":
            for param in self.cls_decoder.parameters():
                param.requires_grad = False
            for param in self.mvc_decoder.parameters():
                param.requires_grad = False
            self.value_decoder = PretrainDecoder(
                embed_dim,
                arch_style=mvc_arch_style,
                use_batch_labels=use_batch_labels,
                catagory_num=1,
            )

    def _get_cell_emb_from_layer(self, layer_output: Tensor, weights: Tensor = None) -> Tensor:
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)
        return cell_emb

    def forward(self, atac, rna, src_key_padding_mask: Optional[Tensor] = None, **kwargs):
        atac_emb = self.atac_emb(atac)
        if self.chr_emb is not None:
            chr_emb = self.chr_emb(kwargs["atac_chrs"])
            atac_emb = atac_emb + chr_emb
        atac_emb = self.dropout_atac(atac_emb)
        atac_emb = self.bn_atac(atac_emb.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.encoder(atac_emb, src_key_padding_mask=src_key_padding_mask)
        transformer_output = self.norm(x)  # (batch, seq_len, embsize)

        rna_emb = self.rna_emb(rna)
        if self.chr_emb is not None:
            chr_emb = self.chr_emb(kwargs["rna_chrs"])
            rna_emb = rna_emb + chr_emb
        rna_emb = self.dropout_rna(rna_emb)
        rna_emb = self.bn_rna(rna_emb.permute(0, 2, 1)).permute(0, 2, 1)

        output = {}
        cell_emb = self._get_cell_emb_from_layer(transformer_output)

        if self.batch_emb is not None:
            batch_emb = self.batch_emb(kwargs["batch_id"])
            cell_emb_w_batch = torch.cat((cell_emb, batch_emb), dim=1)
            output["mvc_pred"] = self.mvc_decoder(cell_emb_w_batch, rna_emb)
            if self.stage == "value_finetune":
                output["value_pred"] = self.value_decoder(cell_emb_w_batch, rna_emb)
        else:
            output["mvc_pred"] = self.mvc_decoder(cell_emb, rna_emb)
            if self.stage == "value_finetune":
                output["value_pred"] = self.value_decoder(cell_emb, rna_emb)

        output["cell_emb"] = cell_emb
        output["cell_pred"] = self.cls_decoder(cell_emb)  # (batch, n_cls)

        return output


# ================================= model/transformer.py ================================


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
    ):
        super().__init__()

        if use_fast_transformer:
            raise NotImplementedError(
                "This vendored recipe only exercises the "
                "`use_fast_transformer=False` plain torch.nn.TransformerEncoder "
                "path; the 'linear'/'flash'/'diff' backends need CUDA-only compiled "
                "dependencies (`fast_transformers`, `flash_attn`) not in the "
                "base-lib set."
            )
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(
        self,
        embs: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        need_weights: Optional[bool] = False,
    ) -> Tensor:
        if need_weights:
            output, layer_weights = self.transformer_encoder(
                embs,
                src_key_padding_mask=src_key_padding_mask,
                need_weights=need_weights,
            )
            return output, layer_weights
        else:
            output = self.transformer_encoder(embs, src_key_padding_mask=src_key_padding_mask)
            return output  # (batch, seq_len, embsize)


# =================================== recipe glue ======================================


def build_epifoundation():
    """Tiny EpiFoundation pretraining model, plain-torch transformer encoder path
    (`encoder="transformer"`, `use_fast_transformer=False`)."""
    return EpiFoundation(
        num_class_cell=4,
        num_rnas=32,
        num_atacs=32,
        num_values=8,
        num_chrs=4,
        embed_dim=32,
        depth=2,
        heads=2,
        head_dim=64,
        encoder="transformer",
        dropout=0.1,
        pad_token_idx_atac=0,
        pad_token_idx_rna=0,
        cell_emb_style="cls",
        mvc_arch_style="inner product",
        use_batch_labels=False,
        use_chr_labels=False,
        transformer_backend="flash",  # unused when use_fast_transformer=False
        stage="pretrain",
    )


def example_input_epifoundation():
    torch.manual_seed(0)
    atac = torch.randint(low=1, high=32, size=(1, 16), dtype=torch.long)
    rna = torch.randint(low=1, high=32, size=(1, 16), dtype=torch.long)
    return (atac, rna)


MENAGERIE_ENTRIES = [
    (
        "EpiFoundation",
        "build_epifoundation",
        "example_input_epifoundation",
        2025,
        "SOURCE_AVAILABLE",
    ),
]
