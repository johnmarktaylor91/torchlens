# FAITHFUL PORT of https://github.com/zlliang/essaysense @ master (original framework: TensorFlow 1.4.1)
#
# EssaySense ("AES"), by Zilong Liang & Jiancong Gao (Fudan University DATA130006 final
# project). The repo implements automated essay scoring (AES) and is the concrete,
# code-matching implementation of "SEEP" / hierarchical CNN+LSTM+attention AES models --
# the task-queue note says: "canonical SEEP name not in repo but architecture matches --
# multiple AES repos implement this pattern". This ports `SentenceLevelCnnLstmWithAttention`
# from `essaysense/models.py`, the hierarchical CNN+attention-pooling+LSTM+attention-pooling
# model (Dong et al. 2017, "Attention-based Recurrent Convolutional Neural Network for
# Automatic Essay Scoring") -- the entry whose network topology is exactly:
#   [essay] -> conv1 -> att_pool1 -> lstm -> att_pool2 -> dense -> [prediction score]
#
# The repo is TensorFlow 1.4.1 (`tf.contrib.rnn`, `tf.placeholder`, static graph
# construction via `tf.layers`/`tf.Variable`/`tf.tensordot`), which cannot run in a
# modern base env (`tf.contrib` was removed in TF2, and the whole file is TF1-graph-mode).
# Per the menagerie rung ladder this is a RUNG-3 FAITHFUL PORT: every op in
# `SentenceLevelCnnLstmWithAttention.define_graph` (conv2d word-level filter bank ->
# learned two-layer additive-attention pooling over the sentence axis -> BasicLSTMCell
# with output dropout -> a second learned additive-attention pooling over the essay/time
# axis -> sigmoid dense head) is transcribed faithfully into base-env torch, using
# equivalent torch primitives (`nn.Conv2d` for `tf.layers.conv2d`, `nn.LSTM` for
# `tf.nn.dynamic_rnn(BasicLSTMCell)`, explicit `nn.Parameter` matmuls for the
# `tf.Variable`+`tf.tensordot` attention blocks). No mechanism is invented or omitted.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class SentenceLevelCnnLstmWithAttention(nn.Module):
    """Faithful port of essaysense.models.SentenceLevelCnnLstmWithAttention.

    Original TF graph (see header):
        essays: [batch, e_len, s_len, w_dim]
        conv1 = tf.layers.conv2d(essays, filters=cnn_lstm_convunits_size,
                                  kernel_size=[1, w_window_len], padding='same',
                                  activation=relu)
        # Attention pooling over the sentence (word) axis:
        att1_mat  = tf.Variable(trunc_normal([C, C]))
        att1_bias = tf.Variable(trunc_normal([1,1,1,C]))
        att1_weight = tanh(tensordot(conv1, att1_mat, axes=[3,0]) + att1_bias)
        att1_vec  = tf.Variable(trunc_normal([C, 1]))
        att1_weight = softmax(tensordot(att1_weight, att1_vec, axes=[3,0]), dim=2)
        att1_output = reduce_sum(att1_weight * conv1, axis=2)     # -> [batch, e_len, C]
        # LSTM over the sentence-representation sequence:
        lstm_cell = DropoutWrapper(BasicLSTMCell(cnn_lstm_att_pool_size))
        lstm, _ = dynamic_rnn(lstm_cell, att1_output)              # -> [batch, e_len, H]
        # Attention pooling over the essay (time) axis:
        att2_mat  = tf.Variable(trunc_normal([H, H]))
        att2_bias = tf.Variable(trunc_normal([1,1,H]))
        att2_weight = tanh(tensordot(lstm, att2_mat, axes=[2,0]) + att2_bias)
        att2_vec  = tf.Variable(trunc_normal([H, 1]))
        att2_weight = softmax(tensordot(att2_weight, att2_vec, axes=[2,0]), dim=1)
        att2_output = reduce_sum(att2_weight * lstm, axis=1)       # -> [batch, H]
        dense = tf.layers.dense(att2_output, units=1, activation=sigmoid)
        preds = tf.reshape(dense, [-1])
    """

    def __init__(
        self,
        w_dim=50,
        w_window_len=5,
        cnn_lstm_convunits_size=80,
        cnn_lstm_att_pool_size=50,
        dropout_keep_prob=0.3,
    ):
        super().__init__()
        self.w_dim = w_dim
        self.C = cnn_lstm_convunits_size
        self.H = cnn_lstm_att_pool_size

        # conv1: tf.layers.conv2d(kernel_size=[1, w_window_len], padding='same')
        # operates over [batch, e_len(H_img), s_len(W_img), w_dim(channels)] in
        # TF's NHWC convention -> torch NCHW equivalent: in_channels=w_dim,
        # kernel=(1, w_window_len), padding=(0, w_window_len // 2) for 'same'.
        self.conv1 = nn.Conv2d(
            in_channels=w_dim,
            out_channels=cnn_lstm_convunits_size,
            kernel_size=(1, w_window_len),
            padding=(0, w_window_len // 2),
        )

        # Attention pooling 1 (over the sentence/word axis).
        self.att1_mat = nn.Parameter(torch.empty(self.C, self.C).normal_(0, 0.05))
        self.att1_bias = nn.Parameter(torch.empty(self.C).normal_(0, 0.05))
        self.att1_vec = nn.Parameter(torch.empty(self.C, 1).normal_(0, 0.05))

        # LSTM: BasicLSTMCell(cnn_lstm_att_pool_size) + output DropoutWrapper.
        self.lstm = nn.LSTM(input_size=self.C, hidden_size=self.H, batch_first=True)
        self.lstm_out_dropout = nn.Dropout(p=1.0 - dropout_keep_prob)

        # Attention pooling 2 (over the essay/time axis).
        self.att2_mat = nn.Parameter(torch.empty(self.H, self.H).normal_(0, 0.05))
        self.att2_bias = nn.Parameter(torch.empty(self.H).normal_(0, 0.05))
        self.att2_vec = nn.Parameter(torch.empty(self.H, 1).normal_(0, 0.05))

        # Dense scoring head: tf.layers.dense(units=1, activation=sigmoid).
        self.dense = nn.Linear(self.H, 1)

    def forward(self, essays):
        """essays: [batch, e_len, s_len, w_dim] (float)."""
        batch, e_len, s_len, w_dim = essays.shape

        # conv1: NHWC->NCHW for torch's Conv2d, then back to NHWC-style layout.
        x = essays.permute(0, 3, 1, 2)  # [batch, w_dim, e_len, s_len]
        conv1 = F.relu(self.conv1(x))  # [batch, C, e_len, s_len]
        conv1 = conv1.permute(0, 2, 3, 1)  # [batch, e_len, s_len, C]  (NHWC, matches TF)

        # Attention pooling 1: tensordot(conv1, att1_mat, axes=[3,0]) == matmul on last dim.
        att1_weight = torch.matmul(conv1, self.att1_mat) + self.att1_bias  # [b,e_len,s_len,C]
        att1_weight = torch.tanh(att1_weight)
        att1_weight = torch.matmul(att1_weight, self.att1_vec)  # [b,e_len,s_len,1]
        att1_weight = F.softmax(att1_weight, dim=2)  # softmax over s_len
        att1_output = torch.sum(att1_weight * conv1, dim=2)  # [b, e_len, C]

        # LSTM over the sentence-representation sequence (dynamic_rnn == batch_first LSTM).
        lstm_out, _ = self.lstm(att1_output)  # [b, e_len, H]
        lstm_out = self.lstm_out_dropout(lstm_out)

        # Attention pooling 2: tensordot(lstm, att2_mat, axes=[2,0]).
        att2_weight = torch.matmul(lstm_out, self.att2_mat) + self.att2_bias  # [b,e_len,H]
        att2_weight = torch.tanh(att2_weight)
        att2_weight = torch.matmul(att2_weight, self.att2_vec)  # [b,e_len,1]
        att2_weight = F.softmax(att2_weight, dim=1)  # softmax over e_len
        att2_output = torch.sum(att2_weight * lstm_out, dim=1)  # [b, H]

        dense = torch.sigmoid(self.dense(att2_output))  # [b, 1]
        preds = dense.reshape(-1)  # [b]
        return preds


def build_seep():
    model = SentenceLevelCnnLstmWithAttention(
        w_dim=12,
        w_window_len=5,
        cnn_lstm_convunits_size=16,
        cnn_lstm_att_pool_size=10,
        dropout_keep_prob=0.7,
    )
    model.eval()
    return model


def example_input_seep():
    """essays: [batch, e_len, s_len, w_dim] word-embedding tensor, matching
    essaysense.configs.HyperParameters (e_len=essay length in sentences,
    s_len=sentence length in words, w_dim=word embedding dim)."""
    batch, e_len, s_len, w_dim = 2, 6, 8, 12
    return torch.randn(batch, e_len, s_len, w_dim)


MENAGERIE_ENTRIES = [
    ("SEEP", "build_seep", "example_input_seep", 2017, MENAGERIE_ZOO),
]
