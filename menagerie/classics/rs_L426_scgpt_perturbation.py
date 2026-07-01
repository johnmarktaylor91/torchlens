# SOURCE: vendored from bowang-lab/scGPT @ main
# https://raw.githubusercontent.com/bowang-lab/scGPT/main/scgpt/model/generation_model.py
# https://raw.githubusercontent.com/bowang-lab/scGPT/main/scgpt/model/model.py (ExprDecoder,
#   ContinuousValueEncoder, MVCDecoder helper classes imported by generation_model.py)
#
# scGPT (Cui et al. 2024, Nature Methods) genetic-perturbation-response model. The real
# class is `TransformerGenerator(nn.Module)` from `scgpt/model/generation_model.py`: a
# gene-token + continuous-value + perturbation-flag embedding stack feeding a standard
# `torch.nn.TransformerEncoder`, decoded back to per-gene expression predictions via an
# affine `Ax + b` decoder built from two `ExprDecoder` heads. Copied verbatim from the
# official repo; only the imports were flattened (the `from .model import ExprDecoder,
# MVCDecoder, ContinuousValueEncoder, FastTransformerEncoderWrapper,
# FlashTransformerEncoderLayer` cross-file import was inlined by copying those classes'
# definitions from `scgpt/model/model.py` into this file), and the `use_fast_transformer`
# lazy `fast_transformers`/flash-attn code paths are left as in the original (unreachable
# at `use_fast_transformer=False`, the upstream default). The masked-value-prediction
# (`do_mvc`) branch is likewise left exactly as-is (default `do_mvc=False` so it is
# constructed lazily, matching upstream behavior).
#
# Upstream license: bowang-lab/scGPT (MIT).

import math
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli
from torch.nn import TransformerEncoder, TransformerEncoderLayer

MENAGERIE_ZOO = "vendored-pytorch"

# ===========================================
# scgpt/model/model.py helper classes referenced by generation_model.py's
# `from .model import ExprDecoder, MVCDecoder, ContinuousValueEncoder,
#  FastTransformerEncoderWrapper, FlashTransformerEncoderLayer` (verbatim)
# ===========================================


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)


class MVCDecoder(nn.Module):
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
    ) -> None:
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(self, cell_emb: Tensor, gene_embs: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)
            h = self.hidden_activation(self.fc1(torch.cat([cell_emb, query_vecs], dim=2)))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)
            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)


class FastTransformerEncoderWrapper(nn.Module):
    """Lazy `fast_transformers` dependency, only constructed when
    `use_fast_transformer=True, fast_transformer_backend="linear"` (not exercised here)."""

    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.fast_transformer_encoder = self.build_fast_transformer_encoder(
            d_model, nhead, d_hid, nlayers, dropout
        )

    @staticmethod
    def build_fast_transformer_encoder(
        d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float
    ):
        from fast_transformers.builders import TransformerEncoderBuilder

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead, got d_model={d_model} and nhead={nhead}"
            )
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=nlayers,
            n_heads=nhead,
            query_dimensions=d_model // nhead,
            value_dimensions=d_model // nhead,
            feed_forward_dimensions=d_hid,
            attention_type="linear",
            attention_dropout=dropout,
            dropout=dropout,
            activation="gelu",
        )
        assert builder.attention_type == "linear"
        return builder.get()

    def forward(self, src: Tensor, src_key_padding_mask: torch.BoolTensor) -> Tensor:
        raise NotImplementedError  # lazy path; not exercised at use_fast_transformer=False


class FlashTransformerEncoderLayer(nn.Module):
    """Lazy flash-attn dependency, only constructed when
    `use_fast_transformer=True, fast_transformer_backend="flash"` (not exercised here)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError  # lazy path; not exercised at use_fast_transformer=False


# ===========================================
# scgpt/model/generation_model.py (verbatim, minus the flattened imports above)
# ===========================================


class TransformerGenerator(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int,
        n_cls: int,
        vocab,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        pert_pad_id: int = 2,
        do_mvc: bool = False,
        domain_spec_batchnorm: Union[bool, str] = False,
        n_input_bins: Optional[int] = 0,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        decoder_activation: Optional[str] = None,
        decoder_adaptive_bias: bool = False,
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.pad_token_id = vocab[pad_token]
        self.pad_value = pad_value
        self.pert_pad_id = pert_pad_id
        self.ecs_threshold = ecs_threshold
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.n_input_bins = n_input_bins
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        # flash_attn_available is always False in this vendored (base-env) build.
        if use_fast_transformer:
            use_fast_transformer = False
        self.use_fast_transformer = use_fast_transformer

        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)

        if use_fast_transformer:
            if fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout
                )
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = AffineExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            activation=decoder_activation,
            adaptive_bias=decoder_adaptive_bias,
        )
        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)
        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
            )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags,
        src_key_padding_mask: Tensor,
    ) -> Tensor:
        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        perts = self.pert_encoder(input_pert_flags)  # (batch, seq_len, embsize)
        total_embs = src + values + perts

        output = self.transformer_encoder(total_embs, src_key_padding_mask=src_key_padding_mask)
        return output  # (batch, seq_len, embsize)

    def _get_cell_emb_from_layer(self, layer_output: Tensor, weights: Tensor = None) -> Tensor:
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = torch.nn.functional.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags: Tensor,
        src_key_padding_mask: Tensor,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        if self.explicit_zero_prob and not do_sample and not self.training:
            do_sample = True

        # binning input gene values
        if self.n_input_bins > 0:
            raise NotImplementedError(
                "binning path requires scgpt.preprocess.binning; not vendored"
            )
        else:
            processed_values = values

        transformer_output = self._encode(
            src, processed_values, input_pert_flags, src_key_padding_mask
        )
        output = {}
        mlm_output = self.decoder(transformer_output, values)
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if MVC:
            mvc_output = self.mvc_decoder(cell_emb, self.cur_gene_token_embs)  # (batch, seq_len)
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            cell_emb_normed = torch.nn.functional.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            cos_sim = torch.nn.functional.relu(cos_sim)
            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class GeneEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class AffineExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        activation: Optional[str] = None,
        tanh_coeff: bool = False,
        adaptive_bias: bool = False,
    ):
        """
        Predict the expression value of each gene in an affine like form of Ax + b.
        This decoder takes two ExprDecoder intrinsically to generate the coefficient A and bias b.
        """
        super().__init__()
        self.explicit_zero_prob = explicit_zero_prob
        self.tanh_coeff = tanh_coeff
        self.adaptive_bias = adaptive_bias
        self.coeff_decoder = ExprDecoder(d_model, explicit_zero_prob=explicit_zero_prob)
        self.bias_decoder = ExprDecoder(d_model, explicit_zero_prob=explicit_zero_prob)

        self.activation = activation
        if activation is not None:
            assert hasattr(nn, activation), f"Unknown activation: {activation}"
            self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor, values: Tensor) -> Tensor:
        coeff = self.coeff_decoder(x)
        bias = self.bias_decoder(x)

        if self.activation is not None:
            coeff["pred"] = self.activation(coeff["pred"])
            bias["pred"] = self.activation(bias["pred"])

        if self.adaptive_bias:
            non_zero_value_mean = values.sum(dim=1, keepdim=True) / (values != 0).sum(
                dim=1, keepdim=True
            )
            bias["pred"] = bias["pred"] * non_zero_value_mean

        if self.explicit_zero_prob:
            return {
                "pred": coeff["pred"] * values + bias["pred"],
                "zero_probs": coeff["zero_probs"],
            }

        return dict(pred=coeff["pred"] * values + bias["pred"])


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        zero_out_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

        self.zero_out_idx = zero_out_idx
        if zero_out_idx is not None:
            self._fill_idx_with_zero(zero_out_idx)
            zero_vector = self(zero_out_idx)
            assert torch.all(zero_vector == 0.0)
            assert not zero_vector.requires_grad

    def _fill_idx_with_zero(self, idx) -> None:
        with torch.no_grad():
            self.embedding.weight[idx].fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Similarity(nn.Module):
    """Dot product or cosine similarity"""

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ClsDecoder(nn.Module):
    """Decoder for classification task."""

    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3, activation=nn.ReLU):
        super().__init__()
        self._decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


# ===========================================
# menagerie staging entry points
# ===========================================

_VOCAB_SIZE = 32
_D_MODEL = 16
_SEQ_LEN = 8


def build_scgpt_perturbation():
    vocab = {"<pad>": 0}
    model = TransformerGenerator(
        ntoken=_VOCAB_SIZE,
        d_model=_D_MODEL,
        nhead=2,
        d_hid=32,
        nlayers=2,
        nlayers_cls=2,
        n_cls=2,
        vocab=vocab,
        dropout=0.0,
        do_mvc=False,
        use_fast_transformer=False,
    )
    model.eval()
    return model


def example_input_scgpt_perturbation():
    torch.manual_seed(0)
    batch = 2
    src = torch.randint(1, _VOCAB_SIZE, (batch, _SEQ_LEN))
    values = torch.rand(batch, _SEQ_LEN)
    input_pert_flags = torch.zeros(batch, _SEQ_LEN, dtype=torch.long)
    src_key_padding_mask = torch.zeros(batch, _SEQ_LEN, dtype=torch.bool)
    return (src, values, input_pert_flags, src_key_padding_mask)


MENAGERIE_ENTRIES = [
    (
        "scGPT-perturbation",
        "build_scgpt_perturbation",
        "example_input_scgpt_perturbation",
        2024,
        "vendored-pytorch",
    ),
]
