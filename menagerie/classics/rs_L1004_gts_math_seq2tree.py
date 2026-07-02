# SOURCE: vendored from ShichaoSun/math_seq2tree @ master
#
# GTS: Goal-driven Tree-Structured MWP solver (Xie & Sun, IJCAI 2019, "A Goal-Driven
# Tree-Structured Neural Model for Math Word Problems"). This is the official
# ShichaoSun/math_seq2tree repo -- pure torch, no exotic dependencies. The four `nn.Module`
# classes below (`EncoderSeq`, `Prediction`, `GenerateNode`, `Merge`) are vendored verbatim
# from src/models.py: a bidirectional-GRU sentence encoder, then a top-down tree decoder
# (`Prediction` scores each tree node against generated-number/operator embeddings with
# "problem aware dynamic encoding" gated concat units, `GenerateNode` expands an operator
# node into left/right child embeddings, `Merge` folds a completed subtree back into its
# parent's sibling slot). `GTSTreeSolver.forward` is a greedy (beam_size=1), single-example,
# CUDA-stripped transcription of the real `evaluate_tree` inference loop in
# src/train_and_evaluate.py (verbatim control flow: node-stack expansion, leaf vs.
# operator branching via `out_token < num_start`, subtree merging while the embedding
# stack's top is terminal) -- only the beam-search bookkeeping (Beam/TreeBeam list-of-K,
# multiple candidate branches per step) is collapsed to K=1 since beam search is a
# decode-time search strategy, not part of the model architecture, and the module needs a
# single deterministic forward pass to trace.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(
            embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True
        )

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = (
            pade_outputs[-1, :, : self.hidden_size] + pade_outputs[0, :, self.hidden_size :]
        )
        pade_outputs = (
            pade_outputs[:, :, : self.hidden_size] + pade_outputs[:, :, self.hidden_size :]
        )  # S x B x H
        return pade_outputs, problem_output


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(
            -1, self.input_size + self.hidden_size
        )
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(
            -1, self.input_size + self.hidden_size
        )

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(
        self,
        node_stacks,
        left_childs,
        encoder_outputs,
        num_pades,
        padding_hidden,
        seq_mask,
        mask_nums,
    ):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for lc, c in zip(left_childs, current_embeddings):
            if lc is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(lc)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        op = self.ops(leaf_input)

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(
            self.generate_l(torch.cat((node_embedding, current_context, node_label), 1))
        )
        l_child_g = torch.sigmoid(
            self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1))
        )
        r_child = torch.tanh(
            self.generate_r(torch.cat((node_embedding, current_context, node_label), 1))
        )
        r_child_g = torch.sigmoid(
            self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1))
        )
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(
            self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1))
        )
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


def _get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    # Verbatim (CPU-only) transcription of
    # src/train_and_evaluate.py::get_all_number_encoder_outputs.
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.BoolTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0)


class _TreeEmbedding:  # matches src/train_and_evaluate.py::TreeEmbedding
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


class GTSTreeSolver(nn.Module):
    """Wraps the real EncoderSeq/Prediction/GenerateNode/Merge submodules and runs a
    greedy (beam_size=1) transcription of the real `evaluate_tree` inference loop
    (src/train_and_evaluate.py) for a single example, so the whole architecture traces in
    one forward call."""

    def __init__(self, encoder, predict, generate, merge, num_start, max_length=8):
        super().__init__()
        self.encoder = encoder
        self.predict = predict
        self.generate = generate
        self.merge = merge
        self.num_start = num_start
        self.max_length = max_length

    def forward(self, input_var, num_pos, generate_nums_count):
        # input_var: (seq_len, 1) LongTensor (S x B, B=1), matches evaluate_tree's input_var.
        # num_pos: positions of quantities in the sentence (len(num_pos) == "num_size" in
        # the source). num_mask width is len(num_pos) + generate_nums_count, matching
        # evaluate_tree's `num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums))`.
        seq_len = input_var.size(0)
        num_size = len(num_pos)
        seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=input_var.device)
        num_mask = torch.zeros(
            1, num_size + generate_nums_count, dtype=torch.bool, device=input_var.device
        )

        encoder_outputs, problem_output = self.encoder(input_var, [seq_len])

        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        all_nums_encoder_outputs = _get_all_number_encoder_outputs(
            encoder_outputs, [num_pos], 1, num_size, self.encoder.hidden_size
        )
        padding_hidden = torch.zeros(
            1, self.predict.hidden_size, dtype=encoder_outputs.dtype, device=encoder_outputs.device
        )

        embeddings_stacks = [[]]
        left_childs = [None]
        out_tokens = []

        for _ in range(self.max_length):
            if len(node_stacks[0]) == 0:
                break
            num_score, op, current_embeddings, current_context, current_nums_embeddings = (
                self.predict(
                    node_stacks,
                    left_childs,
                    encoder_outputs,
                    all_nums_encoder_outputs,
                    padding_hidden,
                    seq_mask,
                    num_mask,
                )
            )

            out_score = torch.nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
            out_token = int(out_score.topk(1)[1][0, 0])
            out_tokens.append(out_token)

            node_stacks[0].pop()

            if out_token < self.num_start:
                generate_input = torch.LongTensor([out_token]).to(input_var.device)
                left_child, right_child, node_label = self.generate(
                    current_embeddings, generate_input, current_context
                )

                node_stacks[0].append(TreeNode(right_child))
                node_stacks[0].append(TreeNode(left_child, left_flag=True))

                embeddings_stacks[0].append(_TreeEmbedding(node_label[0].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)

                while len(embeddings_stacks[0]) > 0 and embeddings_stacks[0][-1].terminal:
                    sub_stree = embeddings_stacks[0].pop()
                    op_embed = embeddings_stacks[0].pop()
                    current_num = self.merge(op_embed.embedding, sub_stree.embedding, current_num)
                embeddings_stacks[0].append(_TreeEmbedding(current_num, True))

            if len(embeddings_stacks[0]) > 0 and embeddings_stacks[0][-1].terminal:
                left_childs = [embeddings_stacks[0][-1].embedding]
            else:
                left_childs = [None]

        return torch.tensor(out_tokens, dtype=torch.long)


def build_gts_math_seq2tree():
    """Tiny GTS goal-driven tree-structured MWP solver for tracing. Every submodule
    (EncoderSeq, Prediction, GenerateNode, Merge) and the tree-decode control flow are
    unmodified from the vendored ShichaoSun/math_seq2tree source (beam search collapsed to
    greedy K=1 for a single deterministic trace)."""
    vocab_size = 40
    embedding_size = 16
    hidden_size = 24
    generate_nums = 2  # number of "generated" constant number tokens (e.g. 1, PI)
    op_nums = 5  # +, -, *, /, ^

    encoder = EncoderSeq(
        input_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        n_layers=1,
        dropout=0.0,
    )
    predict = Prediction(
        hidden_size=hidden_size, op_nums=op_nums, input_size=generate_nums, dropout=0.0
    )
    generate = GenerateNode(
        hidden_size=hidden_size, op_nums=op_nums, embedding_size=embedding_size, dropout=0.0
    )
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size, dropout=0.0)

    model = GTSTreeSolver(encoder, predict, generate, merge, num_start=op_nums, max_length=6)
    model.eval()
    return model


def example_input_gts_math_seq2tree():
    seq_len = 12
    input_var = torch.randint(1, 40, (seq_len, 1), dtype=torch.long)
    num_pos = [3, 7]  # positions of quantities in the sentence
    generate_nums_count = 2  # matches build_gts_math_seq2tree's generate_nums=2
    return (input_var, num_pos, generate_nums_count)


MENAGERIE_ENTRIES = [
    (
        "GTS_MathSeq2Tree",
        build_gts_math_seq2tree,
        example_input_gts_math_seq2tree,
        2019,
        "vendored-pytorch",
    ),
]
