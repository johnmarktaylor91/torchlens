# FAITHFUL REIMPLEMENTATION from https://github.com/smaegol/PlasFlow @ master
# (Krawczyk, Lipinski, Dziembowski. 2018, "PlasFlow: predicting plasmid sequences in
# metagenomic data using genome signatures". Nucleic Acids Research 46(6):e35.)
#
# WHY REIMPLEMENT (not vendor/port): PlasFlow's actual classifier is built directly on
# `tf.contrib.learn.DNNClassifier` -- a canned TensorFlow-1.x "contrib" estimator API
# that was removed entirely in TF 2.0 and cannot run in any currently installable
# TensorFlow. The repo (https://raw.githubusercontent.com/smaegol/PlasFlow/master/PlasFlow.py,
# class `tf_classif`, `predict_proba_tf`/`predict_tf`) does not define its own network
# module at all -- it *is* a thin wrapper that instantiates the estimator:
#   feature_columns = [tf.contrib.layers.real_valued_column("", dimension=self.no_features)]
#   classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
#                                                hidden_units=self.hidden, n_classes=no_classes,
#                                                model_dir=self.modeldir)
# There is therefore no source code to vendor or port (the "architecture" lives inside
# a defunct closed API, not in the repo). The architecture itself is precisely and
# completely documented, both by TF's own (historical) `DNNClassifier` contract -- a
# stack of fully-connected layers (`hidden_units`) each followed by ReLU, terminating in
# a final Linear -> softmax classification head over `n_classes` -- and by PlasFlow's own
# shipped model configuration:
#   * kmer in {5, 6, 7} -> TF-IDF-transformed oligonucleotide-frequency feature vector of
#     dimension 4**kmer (`tf_classif.__init__`/`calculate_freq`, PlasFlow.py:154-243)
#   * hidden_units in {[30], [20, 20]} (PlasFlow.py:154-164, the two shipped hidden-layer
#     configs: "30" -> one hidden layer of 30 ReLU units; "20_20" -> two hidden layers of
#     20 ReLU units each)
#   * n_classes = 28 (models/class_labels_df.tsv has 28 label rows: per-phylum
#     chromosome/plasmid classes)
# This reimplementation is the faithful, standard DNNClassifier computation graph
# (Linear -> ReLU per hidden layer, final Linear -> softmax logits) built from base
# torch.nn, with PlasFlow's own shipped kmer/hidden/class-count configuration.
#
# What is dropped (data plumbing, not architecture): rpy2/Biostrings k-mer counting,
# scikit-learn TF-IDF transform, and FASTA batching are feature-preprocessing performed
# before the network sees its input, not part of the classifier itself.
#
# MENAGERIE_ZOO = "reimpl-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "reimpl-pytorch"


class PlasFlowDNNClassifier(nn.Module):
    """Faithful reimplementation of the `tf.contrib.learn.DNNClassifier` PlasFlow
    instantiates: a stack of `Linear -> ReLU` hidden layers (one per entry of
    `hidden_units`) followed by a final `Linear -> softmax` classification head,
    matching PlasFlow's own shipped kmer-frequency-vector input dimension, hidden-layer
    configuration, and 28-class label set."""

    def __init__(self, n_features, hidden_units, n_classes=28):
        super().__init__()
        layers = []
        in_features = n_features
        for h in hidden_units:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        self.hidden = nn.Sequential(*layers)
        self.logits = nn.Linear(in_features, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = self.hidden(x)
        logits = self.logits(h)
        probs = self.softmax(logits)
        return probs


def build_plasflow():
    # Shipped "kmer5_split_20_20_neurons_relu" config: kmer=5 -> 4**5=1024 TF-IDF
    # oligonucleotide-frequency features, hidden_units=[20, 20], n_classes=28.
    return PlasFlowDNNClassifier(n_features=4**5, hidden_units=[20, 20], n_classes=28)


def example_input_plasflow():
    # TF-IDF-transformed 5-mer frequency vector, batch x 4**5.
    return torch.randn(4, 4**5)


MENAGERIE_ENTRIES = [
    ("PlasFlow", "build_plasflow", "example_input_plasflow", 2018, "reimpl-pytorch"),
]
