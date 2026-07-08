# SOURCE: vendored from https://github.com/leiboliu/HiLAT @ main
# (hilat/models/modeling.py)
# HiLAT: hierarchical label-wise attention Transformer for ICD coding (Liu, Perez-Concha
# et al., Journal of Biomedical Informatics 2022). The `CodingModelConfig`,
# `LableWiseAttentionLayer`, `ChunkAttentionLayer`, and `CodingModel` classes below are
# copied verbatim from the official repo's hilat/models/modeling.py: a transformer
# backbone (any HuggingFace `AutoModel`, e.g. BERT/XLNet/RoBERTa) processes long clinical
# notes chunk-by-chunk, a label-wise attention layer produces one attention-weighted
# vector per ICD code per chunk, and an (optional) chunk attention layer aggregates
# across chunks before the per-code binary classifier. Only the `_init_linear_weights`
# call to `initial_code_title_vectors` (which needs a labeled code-title dataframe plus
# a second transformer forward pass to build the code-embedding init; gated behind
# `use_code_representation`) is left disabled here via `use_code_representation=False`,
# matching a real supported config path in the repo (random-init code vectors -- see
# `LableWiseAttentionLayer._init_linear_weights`, the `else` branch when
# `use_code_representation` is False). No architectural line was changed.
"""Vendored HiLAT model definition (CodingModelConfig, LableWiseAttentionLayer,
ChunkAttentionLayer, CodingModel)."""

import torch
from torch.nn import BCEWithLogitsLoss, Dropout, Linear
from transformers import AutoModel, BertConfig, BertModel, XLNetModel

MENAGERIE_ZOO = "vendored-pytorch"


class CodingModelConfig:
    def __init__(
        self,
        transformer_model_name_or_path,
        transformer_tokenizer_name,
        transformer_layer_update_strategy,
        num_chunks,
        max_seq_length,
        dropout,
        dropout_att,
        d_model,
        label_dictionary,
        num_labels,
        use_code_representation,
        code_max_seq_length,
        code_batch_size,
        multi_head_att,
        chunk_att,
        linear_init_mean,
        linear_init_std,
        document_pooling_strategy,
        multi_head_chunk_attention,
    ):
        super(CodingModelConfig, self).__init__()
        self.transformer_model_name_or_path = transformer_model_name_or_path
        self.transformer_tokenizer_name = transformer_tokenizer_name
        self.transformer_layer_update_strategy = transformer_layer_update_strategy
        self.num_chunks = num_chunks
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.dropout_att = dropout_att
        self.d_model = d_model
        # labels_dictionary is a dataframe with columns: icd9_code, long_title
        self.label_dictionary = label_dictionary
        self.num_labels = num_labels
        self.use_code_representation = use_code_representation
        self.code_max_seq_length = code_max_seq_length
        self.code_batch_size = code_batch_size
        self.multi_head_att = multi_head_att
        self.chunk_att = chunk_att
        self.linear_init_mean = linear_init_mean
        self.linear_init_std = linear_init_std
        self.document_pooling_strategy = document_pooling_strategy
        self.multi_head_chunk_attention = multi_head_chunk_attention


class LableWiseAttentionLayer(torch.nn.Module):
    def __init__(self, coding_model_config, args):
        super(LableWiseAttentionLayer, self).__init__()

        self.config = coding_model_config
        self.args = args

        # layers
        self.l1_linear = torch.nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.tanh = torch.nn.Tanh()
        self.l2_linear = torch.nn.Linear(self.config.d_model, self.config.num_labels, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)

        # Mean pooling last hidden state of code title from transformer model as the initial code vectors
        self._init_linear_weights(
            mean=self.config.linear_init_mean, std=self.config.linear_init_std
        )

    def _init_linear_weights(self, mean, std):
        # normalize the l1 weights
        torch.nn.init.normal_(self.l1_linear.weight, mean, std)
        if self.l1_linear.bias is not None:
            self.l1_linear.bias.data.fill_(0)
        # initialize the l2
        if self.config.use_code_representation:
            # NOTE: `initial_code_title_vectors` lives in the real repo's
            # hilat/models/utils.py (needs redis/datasets/sklearn deps); this staging
            # module omits it because it's unreachable dead code under the
            # `use_code_representation=False` config path used by build_hilat() below.
            code_vectors = initial_code_title_vectors(  # noqa: F821 -- vendored; symbol lives in the real repo's utils.py (redis/datasets deps), unreachable dead code under use_code_representation=False
                self.config.label_dictionary,
                self.config.transformer_model_name_or_path,
                self.config.transformer_tokenizer_name
                if self.config.transformer_tokenizer_name
                else self.config.transformer_model_name_or_path,
                self.config.code_max_seq_length,
                self.config.code_batch_size,
                self.config.d_model,
                self.args.device,
            )

            self.l2_linear.weight = torch.nn.Parameter(code_vectors, requires_grad=True)
        torch.nn.init.normal_(self.l2_linear.weight, mean, std)
        if self.l2_linear.bias is not None:
            self.l2_linear.bias.data.fill_(0)

    def forward(self, x):
        # input: (batch_size, max_seq_length, transformer_hidden_size)
        # output: (batch_size, max_seq_length, transformer_hidden_size)
        # Z = Tan(WH)
        l1_output = self.tanh(self.l1_linear(x))
        # softmax(UZ)
        # l2_linear output shape: (batch_size, max_seq_length, num_labels)
        # attention_weight shape: (batch_size, num_labels, max_seq_length)
        attention_weight = self.softmax(self.l2_linear(l1_output)).transpose(1, 2)
        # attention_output shpae: (batch_size, num_labels, transformer_hidden_size)
        attention_output = torch.matmul(attention_weight, x)

        return attention_output, attention_weight


class ChunkAttentionLayer(torch.nn.Module):
    def __init__(self, coding_model_config, args):
        super(ChunkAttentionLayer, self).__init__()

        self.config = coding_model_config
        self.args = args

        # layers
        self.l1_linear = torch.nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.tanh = torch.nn.Tanh()
        self.l2_linear = torch.nn.Linear(self.config.d_model, 1, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)

        self._init_linear_weights(
            mean=self.config.linear_init_mean, std=self.config.linear_init_std
        )

    def _init_linear_weights(self, mean, std):
        # initialize the l1
        torch.nn.init.normal_(self.l1_linear.weight, mean, std)
        if self.l1_linear.bias is not None:
            self.l1_linear.bias.data.fill_(0)
        # initialize the l2
        torch.nn.init.normal_(self.l2_linear.weight, mean, std)
        if self.l2_linear.bias is not None:
            self.l2_linear.bias.data.fill_(0)

    def forward(self, x):
        # input: (batch_size, num_chunks, transformer_hidden_size)
        # output: (batch_size, num_chunks, transformer_hidden_size)
        # Z = Tan(WH)
        l1_output = self.tanh(self.l1_linear(x))
        # softmax(UZ)
        # l2_linear output shape: (batch_size, num_chunks, 1)
        # attention_weight shape: (batch_size, 1, num_chunks)
        attention_weight = self.softmax(self.l2_linear(l1_output)).transpose(1, 2)
        # attention_output shpae: (batch_size, 1, transformer_hidden_size)
        attention_output = torch.matmul(attention_weight, x)

        return attention_output, attention_weight


# define the model class
class CodingModel(torch.nn.Module):
    def __init__(self, coding_model_config, args):
        super(CodingModel, self).__init__()
        self.coding_model_config = coding_model_config
        self.args = args
        # layers
        self.transformer_layer = AutoModel.from_pretrained(
            self.coding_model_config.transformer_model_name_or_path
        )
        if isinstance(self.transformer_layer, XLNetModel):
            self.transformer_layer.config.use_mems_eval = False
        self.dropout = Dropout(p=self.coding_model_config.dropout)

        if self.coding_model_config.multi_head_att:
            # initial multi head attention according to the num_chunks
            self.label_wise_attention_layer = torch.nn.ModuleList(
                [
                    LableWiseAttentionLayer(coding_model_config, args)
                    for _ in range(self.coding_model_config.num_chunks)
                ]
            )
        else:
            self.label_wise_attention_layer = LableWiseAttentionLayer(coding_model_config, args)
        self.dropout_att = Dropout(p=self.coding_model_config.dropout_att)

        # initial chunk attention
        if self.coding_model_config.chunk_att:
            if self.coding_model_config.multi_head_chunk_attention:
                self.chunk_attention_layer = torch.nn.ModuleList(
                    [
                        ChunkAttentionLayer(coding_model_config, args)
                        for _ in range(self.coding_model_config.num_labels)
                    ]
                )
            else:
                self.chunk_attention_layer = ChunkAttentionLayer(coding_model_config, args)

            self.classifier_layer = Linear(
                self.coding_model_config.d_model, self.coding_model_config.num_labels
            )
        else:
            if self.coding_model_config.document_pooling_strategy == "flat":
                self.classifier_layer = Linear(
                    self.coding_model_config.num_chunks * self.coding_model_config.d_model,
                    self.coding_model_config.num_labels,
                )
            else:  # max or mean pooling
                self.classifier_layer = Linear(
                    self.coding_model_config.d_model, self.coding_model_config.num_labels
                )
        self.sigmoid = torch.nn.Sigmoid()

        if self.coding_model_config.transformer_layer_update_strategy == "no":
            self.freeze_all_transformer_layers()
        elif self.coding_model_config.transformer_layer_update_strategy == "last":
            self.freeze_all_transformer_layers()
            self.unfreeze_transformer_last_layers()

        # initialize the weights of classifier
        self._init_linear_weights(
            mean=self.coding_model_config.linear_init_mean,
            std=self.coding_model_config.linear_init_std,
        )

    def _init_linear_weights(self, mean, std):
        torch.nn.init.normal_(self.classifier_layer.weight, mean, std)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, targets=None):
        # input ids/mask/type_ids shape: (batch_size, num_chunks, max_seq_length)
        # labels shape: (batch_size, num_labels)
        transformer_output = []

        # pass chunk by chunk into transformer layer in the batches.
        # input (batch_size, sequence_length)
        for i in range(self.coding_model_config.num_chunks):
            l1_output = self.transformer_layer(
                input_ids=input_ids[:, i, :],
                attention_mask=attention_mask[:, i, :],
                token_type_ids=token_type_ids[:, i, :],
            )
            # output hidden state shape: (batch_size, sequence_length, hidden_size)
            transformer_output.append(l1_output[0])

        # transpose back chunk and batch size dimensions
        transformer_output = torch.stack(transformer_output)
        transformer_output = transformer_output.transpose(0, 1)
        # dropout transformer output
        l2_dropout = self.dropout(transformer_output)

        # Label-wise attention layers
        # output: (batch_size, num_chunks, num_labels, hidden_size)
        attention_output = []
        attention_weights = []

        for i in range(self.coding_model_config.num_chunks):
            # input: (batch_size, max_seq_length, transformer_hidden_size)
            if self.coding_model_config.multi_head_att:
                attention_layer = self.label_wise_attention_layer[i]
            else:
                attention_layer = self.label_wise_attention_layer
            l3_attention, attention_weight = attention_layer(l2_dropout[:, i, :])
            # l3_attention shape: (batch_size, num_labels, hidden_size)
            # attention_weight: (batch_size, num_labels, max_seq_length)
            attention_output.append(l3_attention)
            attention_weights.append(attention_weight)

        attention_output = torch.stack(attention_output)
        attention_output = attention_output.transpose(0, 1)
        attention_weights = torch.stack(attention_weights)
        attention_weights = attention_weights.transpose(0, 1)

        l3_dropout = self.dropout_att(attention_output)

        if self.coding_model_config.chunk_att:
            # Chunk attention layers
            # output: (batch_size, num_labels, hidden_size)
            chunk_attention_output = []
            chunk_attention_weights = []

            for i in range(self.coding_model_config.num_labels):
                if self.coding_model_config.multi_head_chunk_attention:
                    chunk_attention = self.chunk_attention_layer[i]
                else:
                    chunk_attention = self.chunk_attention_layer
                l4_chunk_attention, l4_chunk_attention_weights = chunk_attention(
                    l3_dropout[:, :, i]
                )
                chunk_attention_output.append(l4_chunk_attention.squeeze(dim=1))
                chunk_attention_weights.append(l4_chunk_attention_weights.squeeze(dim=1))

            chunk_attention_output = torch.stack(chunk_attention_output)
            chunk_attention_output = chunk_attention_output.transpose(0, 1)
            chunk_attention_weights = torch.stack(chunk_attention_weights)
            chunk_attention_weights = chunk_attention_weights.transpose(0, 1)
            # output shape: (batch_size, num_labels, hidden_size)
            l4_dropout = self.dropout_att(chunk_attention_output)
        else:
            # output shape: (batch_size, num_labels, hidden_size*num_chunks)
            l4_dropout = l3_dropout.transpose(1, 2)
            if self.coding_model_config.document_pooling_strategy == "flat":
                # Flatten layer. concatenate representation by labels
                l4_dropout = torch.flatten(l4_dropout, start_dim=2)
            elif self.coding_model_config.document_pooling_strategy == "max":
                l4_dropout = torch.amax(l4_dropout, 2)
            elif self.coding_model_config.document_pooling_strategy == "mean":
                l4_dropout = torch.mean(l4_dropout, 2)
            else:
                raise ValueError("Not supported pooling strategy")

        # classifier layer
        # each code has a binary linear formula
        logits = (
            self.classifier_layer.weight.mul(l4_dropout).sum(dim=2).add(self.classifier_layer.bias)
        )

        preds = self.sigmoid(logits)

        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, targets)

        return {
            "loss": loss,
            "logits": logits,
            "preds": preds,
            "label_attention_weights": attention_weights,
            "chunk_attention_weights": chunk_attention_weights
            if self.coding_model_config.chunk_att
            else [],
        }

    def freeze_all_transformer_layers(self):
        """
        Freeze all layer weight parameters. They will not be updated during training.
        """
        for param in self.transformer_layer.parameters():
            param.requires_grad = False

    def unfreeze_all_transformer_layers(self):
        """
        Unfreeze all layers weight parameters. They will be updated during training.
        """
        for param in self.transformer_layer.parameters():
            param.requires_grad = True

    def unfreeze_transformer_last_layers(self):
        for name, param in self.transformer_layer.named_parameters():
            if "layer.11" in name or "pooler" in name:
                param.requires_grad = True


class _Args:
    """Minimal stand-in for the repo's argparse `args` namespace (only `.device` is read)."""

    def __init__(self, device):
        self.device = device


def build_hilat():
    # Real repo defaults (run_coding.py argparse): num_chunks=2 (multi-head-att, one
    # attention layer per chunk), chunk_att=True, multi_head_chunk_attention=False,
    # document_pooling_strategy irrelevant when chunk_att=True. Backbone is any
    # HuggingFace AutoModel (BERT/XLNet/RoBERTa in the paper); use a tiny random-init
    # BertModel saved to a temp dir so `AutoModel.from_pretrained` resolves it exactly
    # as the real repo does (real repo path: bert-base-uncased / xlnet-base-cased).
    import tempfile

    tiny_bert_config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=2,
    )
    tmp_dir = tempfile.mkdtemp(prefix="hilat_tiny_bert_")
    BertModel(tiny_bert_config).save_pretrained(tmp_dir)

    config = CodingModelConfig(
        transformer_model_name_or_path=tmp_dir,
        transformer_tokenizer_name=None,
        transformer_layer_update_strategy="all",
        num_chunks=2,
        max_seq_length=16,
        dropout=0.1,
        dropout_att=0.1,
        d_model=32,
        label_dictionary=None,
        num_labels=10,
        use_code_representation=False,
        code_max_seq_length=16,
        code_batch_size=1,
        multi_head_att=True,
        chunk_att=True,
        linear_init_mean=0.0,
        linear_init_std=0.03,
        document_pooling_strategy="flat",
        multi_head_chunk_attention=False,
    )
    model = CodingModel(config, _Args(device=torch.device("cpu")))
    model.eval()
    return model


def example_input_hilat():
    # (batch_size, num_chunks, max_seq_length) ids/masks + (batch_size, num_labels) targets,
    # matching MimicIIIDataset's real tensor shapes.
    batch, num_chunks, seq_len, num_labels = 2, 2, 16, 10
    input_ids = torch.randint(0, 128, (batch, num_chunks, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch, num_chunks, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch, num_chunks, seq_len, dtype=torch.long)
    targets = torch.randint(0, 2, (batch, num_labels)).float()
    return (input_ids, attention_mask, token_type_ids, targets)


MENAGERIE_ENTRIES = [
    ("HiLAT", build_hilat, example_input_hilat, 2022, "vendored-pytorch"),
]
