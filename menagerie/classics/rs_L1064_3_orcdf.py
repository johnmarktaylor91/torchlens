# SOURCE: vendored from ECNU-ILOG/ORCDF @ main
# https://raw.githubusercontent.com/ECNU-ILOG/ORCDF/main/inscd/extractor/orcdf.py (ORCDF_Extractor)
# https://raw.githubusercontent.com/ECNU-ILOG/ORCDF/main/inscd/interfunc/dp.py (DP_IF, dot_product)
# https://raw.githubusercontent.com/ECNU-ILOG/ORCDF/main/inscd/interfunc/_util.py (none_neg_clipper)
# https://raw.githubusercontent.com/ECNU-ILOG/ORCDF/main/inscd/models/static/graph/orcdf.py (graph-build
#   logic __create_adj_se / __final_graph, transcribed 1:1 as module-level helpers -- these are the exact
#   NumPy/SciPy static methods the real `ORCDF.train()` calls to build `graph_dict` before extraction)
#
# Wang et al. 2024 (KDD) "ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student
# Learning in Online Education Systems" -- `ORCDF_Extractor` is a bipartite-graph cognitive-diagnosis
# encoder: it embeds student/exercise/knowledge nodes, builds two "response-aware" LightGCN-style
# k-layer sparse-graph convolutions over the student-exercise-knowledge tripartite graph split into a
# "right" (correct-response) and "wrong" (incorrect-response) sub-graph, concatenates+projects the two
# per layer (`concat_layer`), mean-pools across layers, and (in 'all' mode, the repo default) also runs
# the same convolution over a randomly "flipped" copy of the response labels and adds an InfoNCE
# self-supervised contrastive term between the real and flipped student/exercise embeddings -- this
# oversmoothing-resistant graph-contrastive step is the paper's core contribution over plain LightGCN-CD.
# `DP_IF` ("dot-product interaction function", `if_type='dp-linear'`, the repo's default/paper setting)
# is the downstream monotonic MLP that turns the extracted (student, exercise, knowledge) embeddings
# into a per-(student, exercise, knowledge-mask) response probability via a sigmoid dot-product kernel
# followed by a `nn.Sequential` MLP (`none_neg_clipper`-constrained after each optimizer step, same
# monotonicity trick as NeuralCDM). Both classes are copied verbatim from the source (only double
# `__name_mangled` attribute names are kept as-is; they are ordinary Python name mangling, not renamed).
#
# `ORCDFStaged` below is a thin composition wrapper (NOT part of the original architecture) that wires
# `ORCDF_Extractor.extract()` -> `DP_IF.compute()` into one `forward()` -- exactly the two-call sequence
# `_unifier.predict()`/`_unifier.train()` perform in the real repo -- and builds the static sparse graphs
# with `__create_adj_se`/`__final_graph` (transcribed verbatim from `ORCDF.train()`/`ORCDF.__final_graph`)
# from synthetic random response data at construction time, exactly mirroring what the real `ORCDF.train()`
# does before the first `extractor.extract()` call. No architectural change; the wrapper only performs the
# same orchestration the original top-level `ORCDF` class (in `inscd/models/static/graph/orcdf.py`) does,
# minus the CAT/AbstractModel/sklearn adaptive-testing scaffolding (irrelevant to the traced architecture).

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ---- inscd/interfunc/_util.py ----
class _NoneNegClipper(object):
    def __init__(self):
        super(_NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


none_neg_clipper = _NoneNegClipper()


# ---- inscd/interfunc/dp.py ----
def dot_product(A, B):
    return torch.sigmoid(torch.matmul(A, B.T))


class DP_IF(nn.Module):
    def __init__(self, knowledge_num: int, hidden_dims: list, dropout, device, dtype, kernel):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.kernel = kernel
        self.transform_kernel = self.get_kernel()
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        "linear0": nn.Linear(self.knowledge_num, hidden_dim, dtype=self.dtype),
                        "activation0": nn.Tanh(),
                    }
                )
            else:
                layers.update(
                    {
                        "dropout{}".format(idx): nn.Dropout(p=self.dropout),
                        "linear{}".format(idx): nn.Linear(
                            self.hidden_dims[idx - 1], hidden_dim, dtype=self.dtype
                        ),
                        "activation{}".format(idx): nn.Tanh(),
                    }
                )
        layers.update(
            {
                "dropout{}".format(len(self.hidden_dims)): nn.Dropout(p=self.dropout),
                "linear{}".format(len(self.hidden_dims)): nn.Linear(
                    self.hidden_dims[len(self.hidden_dims) - 1], 1, dtype=self.dtype
                ),
                "activation{}".format(len(self.hidden_dims)): nn.Sigmoid(),
            }
        )

        self.mlp = nn.Sequential(layers).to(self.device)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def get_kernel(self):
        if self.kernel == "dp-linear":
            return dot_product
        else:
            raise ValueError("We don not support such kernel")

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        knowledge_ts = kwargs["knowledge_ts"]
        q_mask = kwargs["q_mask"]
        input_x = (
            torch.sigmoid(disc_ts)
            * (
                self.transform_kernel(student_ts, knowledge_ts)
                - self.transform_kernel(diff_ts, knowledge_ts)
            )
            * q_mask
        )
        return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        return F.sigmoid(self.transform_kernel(mastery, knowledge))

    def monotonicity(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)


# ---- inscd/extractor/orcdf.py ----
class ORCDF_Extractor(nn.Module):
    def __init__(
        self,
        student_num: int,
        exercise_num: int,
        knowledge_num: int,
        latent_dim: int,
        device,
        dtype,
        gcn_layers=3,
        keep_prob=0.9,
        mode="all",
        ssl_temp=0.8,
        ssl_weight=1e-2,
        **kwargs,
    ):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.mode = mode
        self.ssl_temp = ssl_temp
        self.ssl_weight = ssl_weight

        self.device = device
        self.dtype = dtype
        self.gcn_layers = gcn_layers
        self.keep_prob = keep_prob
        self.gcn_drop = True
        self.graph_dict = ...

        self.__student_emb = nn.Embedding(self.student_num, self.latent_dim, dtype=self.dtype).to(
            self.device
        )
        self.__knowledge_emb = nn.Embedding(
            self.knowledge_num, self.latent_dim, dtype=self.dtype
        ).to(self.device)
        self.__exercise_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(
            self.device
        )
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)
        self.__knowledge_impact_emb = nn.Embedding(
            self.exercise_num, self.latent_dim, dtype=self.dtype
        ).to(self.device)

        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff": self.__exercise_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight,
        }

        self.concat_layer = nn.Linear(2 * self.latent_dim, self.latent_dim, dtype=self.dtype).to(
            self.device
        )

        self.transfer_student_layer = nn.Linear(
            self.latent_dim, self.knowledge_num, dtype=self.dtype
        ).to(self.device)
        self.transfer_exercise_layer = nn.Linear(
            self.latent_dim, self.knowledge_num, dtype=self.dtype
        ).to(self.device)
        self.transfer_knowledge_layer = nn.Linear(
            self.latent_dim, self.knowledge_num, dtype=self.dtype
        ).to(self.device)
        self.apply(self.initialize_weights)

    def get_graph_dict(self, graph_dict):
        self.graph_dict = graph_dict

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_all_emb(self):
        stu_emb, exer_emb, know_emb = (
            self.__student_emb.weight,
            self.__exercise_emb.weight,
            self.__knowledge_emb.weight,
        )
        all_emb = torch.cat([stu_emb, exer_emb, know_emb]).to(self.device)
        return all_emb

    def convolution(self, graph):
        all_emb = self.get_all_emb()
        emb = [all_emb]
        for layer in range(self.gcn_layers):
            all_emb = torch.sparse.mm(self.__graph_drop(graph), all_emb)
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb

    def __common_forward(self, right, wrong):
        all_emb = self.get_all_emb()
        emb = [all_emb]
        right_emb = all_emb
        wrong_emb = all_emb
        for layer in range(self.gcn_layers):
            right_emb = torch.sparse.mm(self.__graph_drop(right), right_emb)
            wrong_emb = torch.sparse.mm(self.__graph_drop(wrong), wrong_emb)
            all_emb = self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return (
            out_emb[: self.student_num],
            out_emb[self.student_num : self.student_num + self.exercise_num],
            out_emb[self.exercise_num + self.student_num :],
        )

    def __dropout(self, graph, keep_prob):
        if self.gcn_drop and self.training:
            size = graph.size()
            index = graph.indices().t()
            values = graph.values()
            random_index = torch.rand(len(values)) + keep_prob
            random_index = random_index.int().bool()
            index = index[random_index]
            values = values[random_index] / keep_prob
            g = torch.sparse.DoubleTensor(index.t(), values, size)
            return g
        else:
            return graph

    def __graph_drop(self, graph):
        g_dropped = self.__dropout(graph, self.keep_prob)
        return g_dropped

    def extract(self, student_id, exercise_id, q_mask):
        if "dis" not in self.mode:
            stu_forward, exer_forward, know_forward = self.__common_forward(
                self.graph_dict["right"], self.graph_dict["wrong"]
            )
            stu_forward_flip, exer_forward_flip, know_forward_flip = self.__common_forward(
                self.graph_dict["right_flip"], self.graph_dict["wrong_flip"]
            )
        else:
            out = self.convolution(self.graph_dict["all"])
            stu_forward, exer_forward, know_forward = (
                out[: self.student_num],
                out[self.student_num : self.student_num + self.exercise_num],
                out[self.exercise_num + self.student_num :],
            )

        extra_loss = 0

        def InfoNCE(view1, view2, temperature: float = 1.0, b_cos: bool = False):
            if b_cos:
                view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

            pos_score = (view1 @ view2.T) / temperature
            score = torch.diag(F.log_softmax(pos_score, dim=1))
            return -score.mean()

        if "cl" not in self.mode:
            extra_loss = self.ssl_weight * (
                InfoNCE(stu_forward, stu_forward_flip, temperature=self.ssl_temp)
                + InfoNCE(exer_forward, exer_forward_flip, temperature=self.ssl_temp)
            )
        student_ts = (
            self.transfer_student_layer(F.embedding(student_id, stu_forward))
            if "tf" not in self.mode
            else F.embedding(student_id, stu_forward)
        )
        diff_ts = (
            self.transfer_exercise_layer(F.embedding(exercise_id, exer_forward))
            if "tf" not in self.mode
            else F.embedding(exercise_id, exer_forward)
        )
        knowledge_ts = (
            self.transfer_knowledge_layer(know_forward) if "tf" not in self.mode else know_forward
        )

        disc_ts = self.__disc_emb(exercise_id)
        knowledge_impact_ts = self.__knowledge_impact_emb(exercise_id)
        return (
            student_ts,
            diff_ts,
            disc_ts,
            knowledge_ts,
            {"extra_loss": extra_loss, "knowledge_impact": knowledge_impact_ts},
        )

    def get_flip_graph(self):
        def get_flip_data(data):
            np_response_flip = data.copy()
            column = np_response_flip[:, 2]
            probability = np.random.choice(
                [True, False],
                size=column.shape,
                p=[self.graph_dict["flip_ratio"], 1 - self.graph_dict["flip_ratio"]],
            )
            column[probability] = 1 - column[probability]
            np_response_flip[:, 2] = column
            return np_response_flip

        response_flip = get_flip_data(self.graph_dict["response"])
        se_graph_right_flip, se_graph_wrong_flip = [
            self.__create_adj_se(response_flip, is_subgraph=True)[i] for i in range(2)
        ]
        ek_graph = self.graph_dict["Q_Matrix"]
        self.graph_dict["right_flip"], self.graph_dict["wrong_flip"] = (
            self.__final_graph(se_graph_right_flip, ek_graph),
            self.__final_graph(se_graph_wrong_flip, ek_graph),
        )

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        if "dis" not in self.mode:
            stu_forward, exer_forward, know_forward = self.__common_forward(
                self.graph_dict["right"], self.graph_dict["wrong"]
            )
        else:
            out = self.convolution(self.graph_dict["all"])
            stu_forward, exer_forward, know_forward = (
                out[: self.student_num],
                out[self.student_num : self.student_num + self.exercise_num],
                out[self.exercise_num + self.student_num :],
            )

        student_ts = (
            self.transfer_student_layer(stu_forward) if "tf" not in self.mode else stu_forward
        )
        diff_ts = (
            self.transfer_exercise_layer(exer_forward) if "tf" not in self.mode else exer_forward
        )
        knowledge_ts = (
            self.transfer_knowledge_layer(know_forward) if "tf" not in self.mode else know_forward
        )

        disc_ts = self.__disc_emb.weight
        self.__emb_map["mastery"] = student_ts
        self.__emb_map["diff"] = diff_ts
        self.__emb_map["disc"] = disc_ts
        self.__emb_map["knowledge"] = knowledge_ts
        return self.__emb_map[item]

    @staticmethod
    def __get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def __sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def __create_adj_se(self, np_response, is_subgraph=False):
        if is_subgraph:
            if self.mode == "R":
                return np.zeros(shape=(self.student_num, self.exercise_num)), np.zeros(
                    shape=(self.student_num, self.exercise_num)
                )

            train_stu_right = np_response[np_response[:, 2] == 1, 0]
            train_exer_right = np_response[np_response[:, 2] == 1, 1]
            train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
            train_exer_wrong = np_response[np_response[:, 2] == 0, 1]

            adj_se_right = self.__get_csr(
                train_stu_right, train_exer_right, shape=(self.student_num, self.exercise_num)
            )
            adj_se_wrong = self.__get_csr(
                train_stu_wrong, train_exer_wrong, shape=(self.student_num, self.exercise_num)
            )
            return adj_se_right.toarray(), adj_se_wrong.toarray()

        else:
            if self.mode == "R":
                return np.zeros(shape=(self.student_num, self.exercise_num))
            response_stu = np_response[:, 0]
            response_exer = np_response[:, 1]
            adj_se = self.__get_csr(
                response_stu, response_exer, shape=(self.student_num, self.exercise_num)
            )
            return adj_se.toarray()

    def __final_graph(self, se, ek):
        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp = np.zeros(shape=(sek_num, sek_num))
        tmp[: self.student_num, self.student_num : se_num] = se
        tmp[self.student_num : se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self.__sp_mat_to_sp_tensor(adj_matrix).to(self.device)


def inner_product(a, b):
    return torch.sum(a * b, dim=-1)


# ---- graph-build helpers, transcribed verbatim from inscd/models/static/graph/orcdf.py ----
# (ORCDF.__create_adj_se / ORCDF.__final_graph -- the exact static methods the real `ORCDF.train()`
# calls before the first `extractor.extract()`; reproduced at module scope so the staging wrapper can
# build a valid `graph_dict` from synthetic response data without importing the CAT/AbstractModel stack).
def _get_csr(rows, cols, shape):
    values = np.ones_like(rows, dtype=np.float64)
    return sp.csr_matrix((values, (rows, cols)), shape=shape)


def _sp_mat_to_sp_tensor(sp_mat, device):
    coo = sp_mat.tocoo().astype(np.float64)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return (
        torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64)
        .coalesce()
        .to(device)
    )


def _create_adj_se(np_response, student_num, exercise_num, is_subgraph=False):
    if is_subgraph:
        train_stu_right = np_response[np_response[:, 2] == 1, 0]
        train_exer_right = np_response[np_response[:, 2] == 1, 1]
        train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
        train_exer_wrong = np_response[np_response[:, 2] == 0, 1]

        adj_se_right = _get_csr(
            train_stu_right, train_exer_right, shape=(student_num, exercise_num)
        )
        adj_se_wrong = _get_csr(
            train_stu_wrong, train_exer_wrong, shape=(student_num, exercise_num)
        )
        return adj_se_right.toarray(), adj_se_wrong.toarray()
    else:
        response_stu = np_response[:, 0]
        response_exer = np_response[:, 1]
        adj_se = _get_csr(response_stu, response_exer, shape=(student_num, exercise_num))
        return adj_se.toarray()


def _final_graph(se, ek, student_num, exercise_num, knowledge_num, device):
    sek_num = student_num + exercise_num + knowledge_num
    se_num = student_num + exercise_num
    tmp = np.zeros(shape=(sek_num, sek_num))
    tmp[:student_num, student_num:se_num] = se
    tmp[student_num:se_num, se_num:sek_num] = ek
    graph = tmp + tmp.T + np.identity(sek_num)
    graph = sp.csr_matrix(graph)

    rowsum = np.array(graph.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    norm_adj_tmp = d_mat_inv.dot(graph)
    adj_matrix = norm_adj_tmp.dot(d_mat_inv)
    return _sp_mat_to_sp_tensor(adj_matrix, device)


class ORCDFStaged(nn.Module):
    """Composition wrapper: ORCDF_Extractor.extract() -> DP_IF.compute(), the same two-call sequence
    `_unifier.predict()` performs in the real repo. Not part of the original architecture."""

    def __init__(
        self,
        student_num,
        exercise_num,
        knowledge_num,
        latent_dim=16,
        gcn_layers=2,
        hidden_dims=None,
        device="cpu",
        dtype=torch.float64,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]
        self.extractor = ORCDF_Extractor(
            student_num=student_num,
            exercise_num=exercise_num,
            knowledge_num=knowledge_num,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            gcn_layers=gcn_layers,
            keep_prob=0.9,
            mode="all",
            ssl_temp=0.8,
            ssl_weight=1e-2,
        )
        self.inter_func = DP_IF(
            knowledge_num=knowledge_num,
            hidden_dims=hidden_dims,
            dropout=0.0,
            device=device,
            dtype=dtype,
            kernel="dp-linear",
        )

        # Build the static right/wrong/flip graphs from synthetic random response data, exactly as
        # `ORCDF.train()` does before the first `extractor.extract()` call.
        rng = np.random.default_rng(0)
        n_responses = max(student_num * 2, exercise_num * 2)
        response = np.stack(
            [
                rng.integers(0, student_num, n_responses),
                rng.integers(0, exercise_num, n_responses),
                rng.integers(0, 2, n_responses),
            ],
            axis=1,
        ).astype(np.int64)
        q_matrix = (rng.random((exercise_num, knowledge_num)) > 0.5).astype(np.float64)

        se_right, se_wrong = _create_adj_se(response, student_num, exercise_num, is_subgraph=True)
        graph_dict = {
            "right": _final_graph(
                se_right, q_matrix, student_num, exercise_num, knowledge_num, device
            ),
            "wrong": _final_graph(
                se_wrong, q_matrix, student_num, exercise_num, knowledge_num, device
            ),
            "response": response,
            "Q_Matrix": q_matrix,
            "flip_ratio": 0.1,
        }
        self.extractor.get_graph_dict(graph_dict)
        self.extractor.get_flip_graph()
        self.eval()

    def forward(self, student_id, exercise_id, q_mask):
        extracted = self.extractor.extract(student_id, exercise_id, q_mask)
        student_ts, diff_ts, disc_ts, knowledge_ts = extracted[:4]
        return self.inter_func.compute(
            student_ts=student_ts,
            diff_ts=diff_ts,
            disc_ts=disc_ts,
            q_mask=q_mask,
            knowledge_ts=knowledge_ts,
        )


def build_orcdf():
    return ORCDFStaged(
        student_num=12, exercise_num=10, knowledge_num=6, latent_dim=16, gcn_layers=2
    )


def example_input_orcdf():
    batch = 4
    student_id = torch.randint(0, 12, (batch,))
    exercise_id = torch.randint(0, 10, (batch,))
    q_mask = (torch.rand(batch, 6) > 0.5).double()
    return (student_id, exercise_id, q_mask)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("ORCDF", "build_orcdf", "example_input_orcdf", 2024, "vendored"),
]
