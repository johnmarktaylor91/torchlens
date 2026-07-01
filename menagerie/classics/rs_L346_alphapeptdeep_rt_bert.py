# SOURCE: vendored from MannLabs/alphapeptdeep @ main
# https://raw.githubusercontent.com/MannLabs/alphapeptdeep/main/peptdeep/model/building_block.py
# https://raw.githubusercontent.com/MannLabs/alphapeptdeep/main/peptdeep/model/rt.py
# https://raw.githubusercontent.com/MannLabs/alphapeptdeep/main/peptdeep/constants/model_const.yaml
#
# Zeng, Wen, Fischer, Buchholtz, Bech, Wang, Mann 2022 (Nat Commun) "AlphaPeptDeep:
# a modular deep learning framework to predict peptide properties for proteomics" --
# the AlphaX ecosystem's peptide retention-time (RT) model. `Model_RT_Bert` encodes a
# peptide sequence as an amino-acid embedding (27-symbol alphabet, one embedding
# slot reserved for N/C-terminal placeholders) fused with a per-position, per-site
# modification feature vector via a linear "fix-first-k" adapter
# (`Mod_Embedding_FixFirstK`), adds sinusoidal positional encoding
# (`PositionalEncoding`), runs the fused sequence representation through a
# HuggingFace `BertEncoder` stack wrapped in a pseudo-BERT config
# (`Hidden_HFace_Transformer` / `_Pseudo_Bert_Config`) with a residual mix
# (`hidden_x + x * 0.2`), pools the transformer output with a learned
# attention-weighted sum (`SeqAttentionSum`), and regresses a single scalar
# retention-time value through a small PReLU MLP head.
#
# `aa_embedding`/`ascii_embedding`/`zero_param`/`xavier_param`/`invert_attention_mask`,
# `_Pseudo_Bert_Config`, `Hidden_HFace_Transformer`, `SeqAttentionSum`,
# `PositionalEncoding`, `Mod_Embedding_FixFirstK`, `Input_26AA_Mod_PositionalEncoding`
# (aliased `AATransformerEncoding` in the real source), and `Model_RT_Bert` are copied
# verbatim from `peptdeep/model/building_block.py` and `peptdeep/model/rt.py` -- no
# architectural change. The only edits are: (1) inlining `mod_feature_size=109` and
# `aa_embedding_size=27`, the two `peptdeep.settings.model_const` constants the real
# module-level code reads from `constants/model_const.yaml` at import time (these are
# fixed config numbers, not architecture -- `mod_feature_size` is `len(mod_elements)`,
# the count of chemical-element/isotope columns in the modification-feature table, and
# `aa_embedding_size` is the fixed 27-symbol amino-acid vocabulary size used by every
# AlphaPeptDeep model), which lets this module run standalone without the `alphabase`/
# `alpharaw` package chain that only the *data-loading* half of the real package (not
# the model classes) actually needs; (2) dropping the unused `instrument_embedding`/
# `Meta_Embedding`/`max_instrument_num` machinery that belongs to the sibling MS2/CCS
# models, not `Model_RT_Bert`. `AlphaRTModel.build(model_class=Model_RT_Bert, ...)` is
# the real ModelInterface wrapper the package uses to construct/train this class.

import torch

# BERT from huggingface
from transformers.models.bert.modeling_bert import BertEncoder

torch.set_num_threads(2)

# Inlined from peptdeep/constants/model_const.yaml (peptdeep.settings.model_const):
# mod_feature_size = len(model_const["mod_elements"]) == 109
# aa_embedding_size = model_const["aa_embedding_size"] == 27
mod_feature_size = 109
aa_embedding_size = 27


def aa_embedding(hidden_size):
    return torch.nn.Embedding(aa_embedding_size, hidden_size, padding_idx=0)


def zero_param(*shape):
    return torch.nn.Parameter(torch.zeros(shape), requires_grad=False)


def xavier_param(*shape):
    x = torch.nn.Parameter(torch.empty(shape), requires_grad=False)
    torch.nn.init.xavier_uniform_(x)
    return x


def invert_attention_mask(
    encoder_attention_mask: torch.Tensor, dtype=torch.float32
) -> torch.FloatTensor:
    """
    See `invert_attention_mask()` in https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L737.
    Invert an attention mask (e.g., switches 0. and 1.).
    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.
    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
        dtype=dtype
    )  # fp16 compatibility
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(
        dtype
    ).min

    return encoder_extended_attention_mask


init_state = xavier_param


class _Pseudo_Bert_Config:
    def __init__(
        self,
        hidden_dim=256,
        intermediate_size=1024,
        num_attention_heads=8,
        num_bert_layers=4,
        dropout=0.1,
        output_attentions=False,
    ):
        self.add_cross_attention = False
        self.chunk_size_feed_forward = 0
        self.is_decoder = False
        self.seq_len_dim = 1
        self.training = False
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = dropout
        self.attention_probs_dropout_prob = dropout
        self.hidden_size = hidden_dim
        self.initializer_range = 0.02
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = 1e-8
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_bert_layers
        self.output_attentions = output_attentions
        self._attn_implementation = "eager"  # Add this for transformers-4.41.0


class Hidden_HFace_Transformer(torch.nn.Module):
    """
    Transformer NN based on HuggingFace's BertEncoder class
    """

    def __init__(
        self,
        hidden_dim,
        hidden_expand=4,
        nheads=8,
        nlayers=4,
        dropout=0.1,
        output_attentions=False,
    ):
        super().__init__()
        self.config = _Pseudo_Bert_Config(
            hidden_dim=hidden_dim,
            intermediate_size=hidden_dim * hidden_expand,
            num_attention_heads=nheads,
            num_bert_layers=nlayers,
            dropout=dropout,
            output_attentions=False,
        )
        self.output_attentions = output_attentions
        self.bert = BertEncoder(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> tuple:
        """
        Parameters
        ----------
        x : torch.Tensor
            shape = (batch, seq_len, dim)
        attention_mask : torch.Tensor
            shape = (batch, seq_len), [0,1] tensor , 1=enable

        Returns
        -------
        (Tensor, [Tensor])
            out[0] is the hidden layer output,
            and out[1] is the output attention
            if self.output_attentions==True
        """
        if attention_mask is not None:
            attention_mask = invert_attention_mask(attention_mask, dtype=x.dtype)
        return self.bert(
            x,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            return_dict=False,
        )


class SeqAttentionSum(torch.nn.Module):
    """
    apply linear transformation and tensor rescaling with softmax
    """

    def __init__(self, in_features):
        super().__init__()
        self.attn = torch.nn.Sequential(
            torch.nn.Linear(in_features, 1, bias=False),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        attn = self.attn(x)
        return torch.sum(torch.mul(x, attn), dim=1)


class PositionalEncoding(torch.nn.Module):
    """
    transform sequence input into a positional representation
    """

    def __init__(self, out_features=128, max_len=200):
        super().__init__()

        import numpy as np

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_features, 2) * (-np.log(max_len) / out_features))
        pe = torch.zeros(1, max_len, out_features)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class Mod_Embedding_FixFirstK(torch.nn.Module):
    """
    Encodes the modification vector in a single layer feed forward network, but not transforming the first k features
    """

    def __init__(
        self,
        out_features,
    ):
        super().__init__()
        self.k = 6
        self.nn = torch.nn.Linear(mod_feature_size - self.k, out_features - self.k, bias=False)

    def forward(
        self,
        mod_x,
    ):
        return torch.cat((mod_x[:, :, : self.k], self.nn(mod_x[:, :, self.k :])), 2)


class Input_26AA_Mod_PositionalEncoding(torch.nn.Module):
    """
    Encodes AA (26 AA letters) and modification vector
    """

    def __init__(self, out_features, max_len=200):
        super().__init__()
        mod_hidden = 8
        self.mod_nn = Mod_Embedding_FixFirstK(mod_hidden)
        self.aa_emb = aa_embedding(out_features - mod_hidden)
        self.pos_encoder = PositionalEncoding(out_features, max_len)

    def forward(self, aa_indices, mod_x):
        mod_x = self.mod_nn(mod_x)
        x = self.aa_emb(aa_indices)
        return self.pos_encoder(torch.cat((x, mod_x), 2))


# legacy
AATransformerEncoding = Input_26AA_Mod_PositionalEncoding


class Model_RT_Bert(torch.nn.Module):
    """Transformer model for RT prediction"""

    def __init__(
        self,
        dropout=0.1,
        nlayers=4,
        hidden=128,
        output_attentions=False,
        **kwargs,
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)

        self.input_nn = AATransformerEncoding(hidden)

        self._output_attentions = output_attentions

        self.hidden_nn = Hidden_HFace_Transformer(
            hidden,
            nlayers=nlayers,
            dropout=dropout,
            output_attentions=output_attentions,
        )

        self.output_nn = torch.nn.Sequential(
            SeqAttentionSum(hidden),
            torch.nn.PReLU(),
            self.dropout,
            torch.nn.Linear(hidden, 1),
        )

    @property
    def output_attentions(self):
        return self._output_attentions

    @output_attentions.setter
    def output_attentions(self, val: bool):
        self._output_attentions = val
        self.hidden_nn.output_attentions = val

    def forward(
        self,
        aa_indices,
        mod_x,
    ):
        x = self.dropout(self.input_nn(aa_indices, mod_x))

        hidden_x = self.hidden_nn(x)
        if self.output_attentions:
            self.attentions = hidden_x[1]
        else:
            self.attentions = None
        x = self.dropout(hidden_x[0] + x * 0.2)

        return self.output_nn(x).squeeze(1)


MENAGERIE_ZOO = "vendored-pytorch"


def build_alphapeptdeep_rt_bert():
    model = Model_RT_Bert(dropout=0.1, nlayers=2, hidden=32)
    model.eval()
    return model


def example_input_alphapeptdeep_rt_bert():
    seq_len = 10  # nAA + 2 (N-term + C-term placeholders), matching real featurize.py convention
    batch = 1
    aa_indices = torch.randint(0, aa_embedding_size, (batch, seq_len))
    mod_x = torch.zeros(batch, seq_len, mod_feature_size)
    return aa_indices, mod_x


MENAGERIE_ENTRIES = [
    (
        "AlphaPeptDeep (RT-BERT)",
        build_alphapeptdeep_rt_bert,
        example_input_alphapeptdeep_rt_bert,
        2022,
        MENAGERIE_ZOO,
    ),
]
