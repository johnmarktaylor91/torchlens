# SOURCE: vendored from LeonYang95/PLELog @ main
#
# Files combined below (architecture untouched; only file-logger boilerplate and
# the hardcoded `.cuda(device)` call in `forward()` were removed/adjusted so the
# module runs on CPU with random init -- these are training-script infra, not
# architecture):
#   models/gru.py       -> AttGRUModel (the semi-supervised log-anomaly-detection
#                            classifier: word-embed -> bidirectional GRU -> linear
#                            attention pooling -> 2-way linear classifier head)
#   module/Attention.py  -> LinearAttention (masked_softmax attention over GRU
#                            hidden states, keyed by a learned attention guide)
#   module/Common.py     -> NonLinear (orthonormal-initialized Linear + activation)
#   module/CPUEmbedding.py -> CPUEmbedding (plain nn.Embedding kept resident on CPU
#                            for a large vocab; forward() is a normal embedding
#                            lookup)
#
# PLELog (ICSE 2021): "Semi-supervised Log-based Anomaly Detection via Probabilistic
# Label Estimation." The traceable model is the AttGRUModel classifier used at both
# training and inference time.
#
# Repo: https://github.com/LeonYang95/PLELog

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# module/Attention.py (only LinearAttention + the masked_softmax it needs)
# ---------------------------------------------------------------------------
def masked_softmax(vector, mask, dim=-1, memory_efficient=False, mask_fill_value=-1e32):
    if mask is None:
        result = F.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = F.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
            result = F.softmax(masked_vector, dim=dim)
    return result


def _get_combination(combination, tensors):
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == "*":
            return first_tensor * second_tensor
        elif operation == "/":
            return first_tensor / second_tensor
        elif operation == "+":
            return first_tensor + second_tensor
        elif operation == "-":
            return first_tensor - second_tensor
        else:
            raise Exception("Invalid operation: " + operation)


def _rindex(sequence, obj):
    for i in range(len(sequence) - 1, -1, -1):
        if sequence[i] == obj:
            return i
    raise ValueError(f"Unable to find {obj} in sequence {sequence}.")


def _get_combination_and_multiply(combination, tensors, weight):
    if combination.isdigit():
        index = int(combination) - 1
        return torch.matmul(tensors[index], weight)
    else:
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == "*":
            desired_dim = max(first_tensor.dim(), second_tensor.dim()) - 1
            if first_tensor.dim() == 4:
                expanded_dim = _rindex(first_tensor.size(), 1)
                first_tensor = first_tensor.squeeze(expanded_dim)
            if second_tensor.dim() == 4:
                expanded_dim = _rindex(second_tensor.size(), 1)
                second_tensor = second_tensor.squeeze(expanded_dim)
            intermediate = first_tensor * weight
            result = torch.matmul(intermediate, second_tensor.transpose(-1, -2))
            if result.dim() == desired_dim + 1:
                result = result.squeeze(-1)
            return result
        else:
            raise Exception("Invalid operation for this module: " + operation)


def get_combined_dim(combination, tensor_dims):
    combination = combination.replace("x", "1").replace("y", "2")
    return sum([_get_combination_dim(piece, tensor_dims) for piece in combination.split(",")])


def _get_combination_dim(combination, tensor_dims):
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        if first_tensor_dim != second_tensor_dim:
            raise Exception('Tensor dims must match for operation "{}"'.format(operation))
        return first_tensor_dim


def combine_tensors_and_multiply(combination, tensors, weights):
    combination = combination.replace("x", "1").replace("y", "2")
    pieces = combination.split(",")
    tensor_dims = [tensor.size(-1) for tensor in tensors]
    combination_dims = [_get_combination_dim(piece, tensor_dims) for piece in pieces]
    dims_so_far = 0
    to_sum = []
    for piece, combination_dim in zip(pieces, combination_dims):
        weight = weights[dims_so_far : (dims_so_far + combination_dim)]
        dims_so_far += combination_dim
        to_sum.append(_get_combination_and_multiply(piece, tensors, weight))
    result = to_sum[0]
    for result_piece in to_sum[1:]:
        result = result + result_piece
    return result


class LinearAttention(nn.Module):
    def __init__(self, tensor_1_dim, tensor_2_dim, combination="x,y", normalize=True):
        super(LinearAttention, self).__init__()
        self._combination = combination
        combined_dim = get_combined_dim(combination, [tensor_1_dim, tensor_2_dim])
        self._weight_vector = Parameter(torch.Tensor(combined_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = nn.Tanh()
        self._normalize = normalize
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
        self._weight_vector.data.uniform_(-std, std)
        self._bias.data.fill_(0)

    def forward(self, vector, matrix, matrix_mask=None):
        similarities = self._forward_internal(vector, matrix)
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        else:
            return similarities

    def _forward_internal(self, vector, matrix):
        combined_tensors = combine_tensors_and_multiply(
            self._combination, [vector.unsqueeze(1), matrix], self._weight_vector
        )
        return self._activation(combined_tensors.squeeze(1) + self._bias)


# ---------------------------------------------------------------------------
# module/Common.py (only NonLinear + its orthonormal_initializer + dropout
# helper AttGRUModel.forward uses)
# ---------------------------------------------------------------------------
def orthonormal_initializer(output_size, input_size):
    """adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py"""
    import numpy as np

    identity_mat = np.eye(output_size)
    lr = 0.1
    eps = 0.05 / (output_size + input_size)
    success = False
    tries = 0
    loss = 0.0
    Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - identity_mat
            loss = np.sum(QTQmI**2 / 2)
            Q2 = Q**2
            Q -= (
                lr
                * Q.dot(QTQmI)
                / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1)
                    + eps
                )
            )
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    return np.transpose(Q.astype(np.float32))


def drop_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    word_masks.requires_grad = False
    scale = 1.0 / (1.0 * word_masks + 1e-12)
    word_masks = word_masks * scale
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    return word_embeddings


class NonLinear(nn.Module):
    def __init__(self, input_size, hidden_size, activation=None):
        super(NonLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable: type={}".format(type(activation)))
            self._activate = activation
        self.reset_parameters()

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

    def reset_parameters(self):
        import numpy as np

        W = orthonormal_initializer(self.hidden_size, self.input_size)
        self.linear.weight.data.copy_(torch.from_numpy(W))
        b = np.zeros(self.hidden_size, dtype=np.float32)
        self.linear.bias.data.copy_(torch.from_numpy(b))


# ---------------------------------------------------------------------------
# module/CPUEmbedding.py
# ---------------------------------------------------------------------------
class CPUEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(CPUEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, (
                    "Padding_idx must be within num_embeddings"
                )
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, (
                    "Padding_idx must be within num_embeddings"
                )
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        return F.embedding(input, self.weight, self.padding_idx)

    def extra_repr(self):
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        return s.format(**self.__dict__)


# ---------------------------------------------------------------------------
# models/gru.py -> AttGRUModel
# (file-logger boilerplate removed; `.cuda(device)` in forward() replaced with
# `.to(next(self.parameters()).device)` so the model runs on whatever device it
# was placed on -- purely an infra fix, not an architectural change)
# ---------------------------------------------------------------------------
class _TinyVocab:
    """Minimal stand-in for PLELog's `utils.Vocab.Vocab`: only the attributes
    AttGRUModel.__init__ reads (vocab_size, word_dim, embeddings)."""

    def __init__(self, vocab_size, word_dim):
        import numpy as np

        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.embeddings = np.random.randn(vocab_size, word_dim).astype("float32")


class AttGRUModel(nn.Module):
    def __init__(self, vocab, lstm_layers, lstm_hiddens, dropout=0):
        super(AttGRUModel, self).__init__()
        self.dropout = dropout
        vocab_size, word_dims = vocab.vocab_size, vocab.word_dim
        self.word_embed = CPUEmbedding(vocab_size, word_dims, padding_idx=vocab_size - 1)
        self.word_embed.weight.data.copy_(torch.from_numpy(vocab.embeddings))
        self.word_embed.weight.requires_grad = False
        self.rnn = nn.GRU(
            input_size=word_dims,
            hidden_size=lstm_hiddens,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.sent_dim = 2 * lstm_hiddens
        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim)
        self.proj = NonLinear(self.sent_dim, 2)

    def forward(self, inputs):
        words, masks, word_len = inputs
        embed = self.word_embed(words)
        if self.training:
            embed = drop_input_independent(embed, self.dropout)
        embed = embed.to(next(self.parameters()).device)
        batch_size = embed.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        hiddens, state = self.rnn(embed)
        sent_probs = self.atten(atten_guide, hiddens, masks)
        batch_size, srclen, dim = hiddens.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        represents = hiddens * sent_probs
        represents = represents.sum(dim=1)
        outputs = self.proj(represents)
        return outputs


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_VOCAB_SIZE = 64
_WORD_DIM = 32
_SEQ_LEN = 16
_LSTM_LAYERS = 1
_LSTM_HIDDENS = 24


def build_plelog_attgru():
    vocab = _TinyVocab(_VOCAB_SIZE, _WORD_DIM)
    model = AttGRUModel(vocab, lstm_layers=_LSTM_LAYERS, lstm_hiddens=_LSTM_HIDDENS, dropout=0.0)
    model.eval()
    return model


def example_input_plelog_attgru():
    # AttGRUModel.forward(self, inputs) takes ONE positional arg that is itself the
    # (words, masks, word_len) tuple, so wrap it in a length-1 outer tuple: tl.trace
    # unpacks input_args as model(*input_args), and we want model((words, masks, word_len)).
    words = torch.randint(0, _VOCAB_SIZE - 1, (4, _SEQ_LEN))
    masks = torch.ones(4, _SEQ_LEN)
    word_len = torch.full((4,), _SEQ_LEN, dtype=torch.long)
    return ((words, masks, word_len),)


MENAGERIE_ENTRIES = [
    (
        "PLELog AttGRU",
        build_plelog_attgru,
        example_input_plelog_attgru,
        2021,
        "VENDOR",
    ),
]
