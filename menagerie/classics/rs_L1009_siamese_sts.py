# SOURCE: vendored from https://github.com/shahrukhx01/siamese-nn-semantic-text-similarity @ main
#   Vendored files:
#     - siamese_sts/siamese_net/siamese_lstm.py            (SiameseLSTM)
#     - siamese_sts/siamese_net/siamese_lstm_attention.py  (SiameseBiLSTMAttention + SelfAttention)
#     - siamese_sts/utils/utils.py                         (similarity_score helper)
#   Both model classes and the similarity helper are copied verbatim (only the relative
#   `from siamese_sts...` import for `similarity_score` is inlined since this is a
#   single-file staging module). Two queue candidates both point at this repo:
#     - "Neural Short Answer Scoring (NTN-based)" (row 1063)
#     - "Neural Short-Answer Grading Siamese/Attention Model" (row 1064)
#   The repo's actual architecture is a Siamese BiLSTM (siamese_lstm.py) and a Siamese
#   BiLSTM + self-attention variant (siamese_lstm_attention.py, per "A Structured
#   Self-Attentive Sentence Embedding", https://arxiv.org/pdf/1703.03130.pdf) -- not an
#   NTN (Neural Tensor Network); this is the real code the queue rows cite, so both queue
#   rows are satisfied by the two real classes here rather than by a from-scratch NTN.
#
# Both models take two sentence batches (sent1, sent2) plus their sequence lengths and
# produce an exponential-L1-distance similarity score in [0, 1] -- a Siamese architecture
# for semantic text similarity / short-answer-vs-reference scoring. Multi-tensor input
# (sent1_batch, sent2_batch, sent1_lengths, sent2_lengths) means this must be staged as a
# module (not a single-tensor recipe row).

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# siamese_sts/utils/utils.py
# ---------------------------------------------------------------------------
def similarity_score(input1, input2):
    # Get similarity predictions:
    dif = input1.squeeze() - input2.squeeze()

    norm = torch.norm(dif, p=1, dim=dif.dim() - 1)
    y_hat = torch.exp(-norm)
    y_hat = torch.clamp(y_hat, min=1e-7, max=1.0 - 1e-7)
    return y_hat


# ---------------------------------------------------------------------------
# siamese_sts/siamese_net/siamese_lstm.py
# ---------------------------------------------------------------------------
class SiameseLSTM(nn.Module):
    """
    Wrapper class using Pytorch nn.Module to create the architecture for our
    binary classification model
    """

    def __init__(
        self,
        batch_size: int,
        output_size: int,
        hidden_size: int,
        vocab_size: int,
        embedding_size: int,
        embedding_weights: torch.TensorType,
        lstm_layers: int,
        device: str,
    ):
        super(SiameseLSTM, self).__init__()
        ## model hyper parameters
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_layers = lstm_layers
        self.device = device

        ## model layers
        # initializing the look-up table.
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        # assigning the look-up table to the pre-trained fasttext word embeddings.
        self.word_embeddings.weight = nn.Parameter(
            embedding_weights.to(self.device), requires_grad=True
        )

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=lstm_layers)

    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """
        return (
            Variable(torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(self.device)),
            Variable(torch.zeros(self.lstm_layers, batch_size, self.hidden_size)).to(self.device),
        )

    def forward_once(self, batch, lengths):
        # embedded input of shape = (batch_size, sequence_len,  embedding_size)
        embeddings = self.word_embeddings(batch)

        # permute embedded input to shape = (sequence_len, batch_size, embedding_size)
        embeddings = embeddings.permute(1, 0, 2)

        # perform forward pass of LSTM
        output, (final_hidden_state, final_cell_state) = self.lstm(embeddings, self.hidden)

        return final_hidden_state[-1]

    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths):
        """
        Performs the forward pass for each batch
        """
        ## init context and hidden weights for lstm cell
        self.hidden = self.init_hidden(sent1_batch.size(0))

        self.sent1_out = self.forward_once(sent1_batch, sent1_lengths)
        self.sent2_out = self.forward_once(sent2_batch, sent2_lengths)
        similarity = similarity_score(self.sent1_out, self.sent2_out)
        return similarity


# ---------------------------------------------------------------------------
# siamese_sts/siamese_net/siamese_lstm_attention.py
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """
    Implementation of the attention block
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        ## corresponds to variable Ws1 in ICLR paper, we don't use the bias term as suggested in paper
        self.layer1 = nn.Linear(input_size, hidden_size, bias=False)
        ## corresponds to variable Ws2 in ICLR paper, we don't use the bias term as suggested in paper
        self.layer2 = nn.Linear(hidden_size, output_size, bias=False)

    ## the forward function would receive lstm's all hidden states as input
    def forward(self, attention_input):
        ## expected input shape: (batch_size , seq_len, num_lstm_layers * num_directions)
        out = self.layer1(attention_input)
        # out shape: (batch_size, seq_len, attention_hidden_size)
        out = torch.tanh(out)
        # out shape: (batch_size, seq_len, attention_out)
        out = self.layer2(out)
        ## out shape post permute: (batch_size, attention_out, seq_len)
        out = out.permute(0, 2, 1)
        out = F.softmax(out, dim=2)  ## softmax dimenion as per the paper

        return out  ## out shape: (batch_size, attention_out, seq_len)


class SiameseBiLSTMAttention(nn.Module):
    """
    Wrapper class using Pytorch nn.Module to create the architecture for our model
    Architecture is based on the paper:
    A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
    https://arxiv.org/pdf/1703.03130.pdf
    """

    def __init__(
        self,
        batch_size,
        output_size,
        hidden_size,
        vocab_size,
        embedding_size,
        embedding_weights,
        lstm_layers,
        device,
        bidirectional,
        self_attention_config,
        fc_hidden_size,
    ):
        super(SiameseBiLSTMAttention, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lstm_layers = lstm_layers
        self.device = device
        self.bidirectional = bidirectional
        self.fc_hidden_size = fc_hidden_size
        self.lstm_directions = (
            2 if self.bidirectional else 1
        )  ## decide directions based on input flag

        ## model layers
        # initializing the look-up table.
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        # assigning the look-up table to the pre-trained fasttext word embeddings.
        self.word_embeddings.weight = nn.Parameter(
            embedding_weights.to(self.device), requires_grad=True
        )

        ## initializng lstm layer
        self.bilstm = nn.LSTM(
            self.embedding_size,
            self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            bidirectional=self.bidirectional,
            dropout=0.5,
        )

        ## initializing self attention layers
        self.self_attention = None
        self.fc_layer = None

        ## incase we are using bi-directional lstm we'd have to take care of bi-directional outputs in
        ## subsequent layers

        self.self_attention = SelfAttention(
            self.lstm_hidden_size * self.lstm_directions,
            self_attention_config["hidden_size"],
            self_attention_config["output_size"],
        )
        ## this layer comes right after self attention computation
        self.fc_layer = nn.Linear(
            self.lstm_directions * self.lstm_hidden_size * self_attention_config["output_size"],
            self.fc_hidden_size,
        )

    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """
        layer_size = self.lstm_layers
        if self.bidirectional:
            layer_size *= 2  # since we have two layers instantiated for each lstm layer of bi-lstm
        return (
            Variable(torch.zeros(layer_size, batch_size, self.lstm_hidden_size).to(self.device)),
            Variable(torch.zeros(layer_size, batch_size, self.lstm_hidden_size)).to(self.device),
        )

    def forward_once(self, batch, lengths):
        """
        Performs the forward pass for each batch
        """

        ## batch shape: (num_sequences, batch_size)
        ## embeddings shape: (seq_len, batch_size, embedding_size)

        embeddings = self.word_embeddings(batch)

        # permute embedded input to shape = (sequence_len, batch_size, embedding_size)
        embeddings = embeddings.permute(1, 0, 2)

        """
		here padded output refers to variable 'H' from the ICLR paper
		as LSTM's output contains all the hidden states for a given a sequence
		hence we use this as variable 'H'
		output shape : (seq_len, batch_size, num_lstm_layers * num_directions)
		"""
        output, (final_hidden_state, final_cell_state) = self.bilstm(embeddings, self.hidden)

        # output post permute shape: (batch_size , seq_len, num_lstm_layers * num_directions)
        output = output.permute(1, 0, 2)

        ## refers to annotation matrix 'A' in ICLR paper
        annotation_weight_matrix = self.self_attention(output)

        """
		in the final step we compute output matrix 'M = AH' which has sentnece embeddings
		here the bmm (batch matrix mul) inputs have following shapes
		annotation_weight_matrix : (batch_size, attention_out, seq_len)
		output: (batch_size , seq_len, lstm_hidden_size * num_directions)
		sentence_embedding shape: (batch_size , attention_out, lstm_hidden_size * num_directions)
		"""
        sentence_embedding = torch.bmm(annotation_weight_matrix, output)

        """
		transforming the two lstm directions with attention output in sentence embedding matrix
		for fully connected layer
		sentence_embedding shape: (batch_size, lstm_directions*lstm_hidden_size*self_attention_output_size)
		"""
        sentence_embedding = sentence_embedding.view(
            -1, sentence_embedding.size()[1] * sentence_embedding.size()[2]
        )

        ## feeding sentence_embedding result to fully connected
        fc_out = self.fc_layer(sentence_embedding)

        return fc_out, annotation_weight_matrix

    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths):
        """
        Performs the forward pass for each batch
        """
        ## init context and hidden weights for lstm cell
        self.hidden = self.init_hidden(sent1_batch.size(0))

        self.sent1_out, sent1_annotation_weight_matrix = self.forward_once(
            sent1_batch, sent1_lengths
        )
        self.sent2_out, sent2_annotation_weight_matrix = self.forward_once(
            sent2_batch, sent2_lengths
        )
        similarity = similarity_score(self.sent1_out, self.sent2_out)
        return (
            similarity,
            sent1_annotation_weight_matrix,
            sent2_annotation_weight_matrix,
        )


# ---------------------------------------------------------------------------
# Menagerie staging hooks
# ---------------------------------------------------------------------------
_BATCH_SIZE = 2
_OUTPUT_SIZE = 1
_HIDDEN_SIZE = 8
_VOCAB_SIZE = 32
_EMBEDDING_SIZE = 6
_LSTM_LAYERS = 1
_SEQ_LEN = 5
_DEVICE = "cpu"


def _tiny_embedding_weights():
    return torch.randn(_VOCAB_SIZE, _EMBEDDING_SIZE)


def build_siamese_lstm():
    return SiameseLSTM(
        batch_size=_BATCH_SIZE,
        output_size=_OUTPUT_SIZE,
        hidden_size=_HIDDEN_SIZE,
        vocab_size=_VOCAB_SIZE,
        embedding_size=_EMBEDDING_SIZE,
        embedding_weights=_tiny_embedding_weights(),
        lstm_layers=_LSTM_LAYERS,
        device=_DEVICE,
    )


def example_input_siamese_lstm():
    sent1_batch = torch.randint(0, _VOCAB_SIZE, (_BATCH_SIZE, _SEQ_LEN))
    sent2_batch = torch.randint(0, _VOCAB_SIZE, (_BATCH_SIZE, _SEQ_LEN))
    sent1_lengths = torch.full((_BATCH_SIZE,), _SEQ_LEN, dtype=torch.long)
    sent2_lengths = torch.full((_BATCH_SIZE,), _SEQ_LEN, dtype=torch.long)
    return (sent1_batch, sent2_batch, sent1_lengths, sent2_lengths)


def build_siamese_bilstm_attention():
    self_attention_config = {
        "hidden_size": 6,  ## 'da' in the ICLR paper
        "output_size": 3,  ## 'r' in the ICLR paper
        "penalty": 0.6,
    }
    return SiameseBiLSTMAttention(
        batch_size=_BATCH_SIZE,
        output_size=_OUTPUT_SIZE,
        hidden_size=_HIDDEN_SIZE,
        vocab_size=_VOCAB_SIZE,
        embedding_size=_EMBEDDING_SIZE,
        embedding_weights=_tiny_embedding_weights(),
        lstm_layers=_LSTM_LAYERS,
        device=_DEVICE,
        bidirectional=True,
        self_attention_config=self_attention_config,
        fc_hidden_size=8,
    )


def example_input_siamese_bilstm_attention():
    return example_input_siamese_lstm()


MENAGERIE_ENTRIES = [
    (
        "Siamese-BiLSTM-STS",
        build_siamese_lstm,
        example_input_siamese_lstm,
        2020,
        "MENAGERIE_ZOO",
    ),
    (
        "Siamese-BiLSTM-SelfAttention-STS",
        build_siamese_bilstm_attention,
        example_input_siamese_bilstm_attention,
        2020,
        "MENAGERIE_ZOO",
    ),
]
