# FAITHFUL PORT of xiaojunxu/dnn-binary-code-similarity @ master (original framework: TensorFlow 1.x)
# https://raw.githubusercontent.com/xiaojunxu/dnn-binary-code-similarity/master/graphnnSiamese.py
#
# "Gemini" (Xu, Liu, Feng, Yin, Song, Song; "Neural Network-based Graph Embedding for
# Cross-Platform Binary Code Similarity Detection", CCS 2017) -- NOT Google's Gemini
# model family; the name collision is coincidental (an earlier binary-similarity
# system named itself "Gemini" years before Google's LLM). The official repo is
# TensorFlow 1.x with `tf.placeholder`/`tf.Session`/Python-2 `print` statements
# (`graphnnSiamese.py`), which our installed TF/Keras stack cannot run, so this is a
# faithful transcription of the real `graph_embed` computation rather than a vendor.
#
# Gemini's architecture is a Siamese Structure2Vec-style graph neural network: each
# binary-function control-flow graph is embedded independently by the SAME
# `graph_embed` tower (weight-shared, per repo's single Wnode/Wembed/W_output/b_output
# variable set reused for both `embed1` and `embed2`), then compared by cosine
# similarity. This module ports one `graph_embed` tower verbatim from
# `graphnnSiamese.py`:
#   node_val = reshape(matmul(reshape(X, [-1, N_x]), Wnode), [B, N_node, N_embed])
#   cur_msg = relu(node_val)
#   for t in range(iter_level):
#       Li_t = matmul(msg_mask, cur_msg)                    # message aggregation
#       cur_info = reshape(Li_t, [-1, N_embed])
#       for Wi in Wembed: cur_info = relu(matmul(cur_info, Wi))     # except last Wi (no relu)
#       neigh_val_t = reshape(cur_info, shape(Li_t))
#       tot_val_t = node_val + neigh_val_t
#       cur_msg = tanh(tot_val_t)
#   g_embed = reduce_sum(cur_msg, axis=1)
#   output = matmul(g_embed, W_output) + b_output
# `X` is the per-node label/feature matrix [B, N_node, N_x] (handcrafted binary-code
# features: e.g. instruction-type histograms, per the paper); `msg_mask` is the dense
# [B, N_node, N_node] adjacency (CFG edges) used to aggregate neighbor messages via
# matmul, exactly as in the repo (no sparse-graph API is used there). `depth_embed`
# (len(Wembed), the repo's default 2) linear layers with ReLU on all but the last are
# applied to the aggregated message at every propagation step, matching the repo's
# `for Wi in Wembed: ... if Wi == Wembed[-1]: no relu else: relu` loop.
#
# This module ports ONE embedding tower (`GeminiGraphEmbed`, matching `embed1`'s
# subgraph in the repo) as the traceable forward architecture; the Siamese
# cosine-similarity head and the second (weight-tied) tower are training/inference
# scaffolding around the same tower and are omitted, matching how tl.trace captures a
# single forward-callable nn.Module.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class GeminiGraphEmbed(nn.Module):
    """Port of `graph_embed()` / `graphnn.__init__`'s embed1 tower in
    graphnnSiamese.py -- one Structure2Vec-style graph-embedding pass."""

    def __init__(self, n_x=8, n_embed=16, n_o=8, depth_embed=2, iter_level=3):
        super().__init__()
        self.n_x = n_x
        self.n_embed = n_embed
        self.iter_level = iter_level

        self.w_node = nn.Parameter(torch.empty(n_x, n_embed))
        nn.init.trunc_normal_(self.w_node, std=0.1)

        self.w_embed = nn.ParameterList()
        for _ in range(depth_embed):
            w = nn.Parameter(torch.empty(n_embed, n_embed))
            nn.init.trunc_normal_(w, std=0.1)
            self.w_embed.append(w)

        self.w_output = nn.Parameter(torch.empty(n_embed, n_o))
        nn.init.trunc_normal_(self.w_output, std=0.1)
        self.b_output = nn.Parameter(torch.zeros(n_o))

    def forward(self, x, msg_mask):
        # x: [B, N_node, N_x], msg_mask: [B, N_node, N_node]
        b = x.shape[0]
        node_val = torch.matmul(x.reshape(-1, self.n_x), self.w_node).reshape(b, -1, self.n_embed)

        cur_msg = torch.relu(node_val)
        for _ in range(self.iter_level):
            li_t = torch.matmul(msg_mask, cur_msg)
            cur_info = li_t.reshape(-1, self.n_embed)
            for i, w in enumerate(self.w_embed):
                cur_info = torch.matmul(cur_info, w)
                if i != len(self.w_embed) - 1:
                    cur_info = torch.relu(cur_info)
            neigh_val_t = cur_info.reshape(li_t.shape)
            tot_val_t = node_val + neigh_val_t
            cur_msg = torch.tanh(tot_val_t)

        g_embed = torch.sum(cur_msg, dim=1)
        output = torch.matmul(g_embed, self.w_output) + self.b_output
        return output


def build_gemini_structure2vec():
    model = GeminiGraphEmbed(n_x=8, n_embed=16, n_o=8, depth_embed=2, iter_level=3)
    model.eval()
    return model


def example_input_gemini_structure2vec():
    torch.manual_seed(0)
    n_node = 6
    x = torch.randn(2, n_node, 8)
    adj = (torch.rand(2, n_node, n_node) > 0.5).float()
    return (x, adj)


MENAGERIE_ENTRIES = [
    (
        "Gemini (binary code similarity graph embedding)",
        "build_gemini_structure2vec",
        "example_input_gemini_structure2vec",
        2017,
        "ported",
    ),
]
