# SOURCE: vendored from idekerlab/DrugCell @ master
# https://raw.githubusercontent.com/idekerlab/DrugCell/master/code/drugcell_NN.py
#
# Ma, Kuenzi, Huang, Chen, Zhu, Fan, Guo, Eskin, Ideker 2021 (Nature Cancer) "Cancer drug
# response prediction using a hybrid dynamic neural network of biological pathways" (DrugCell).
# DrugCell ("Visible Neural Network") wires a `torch.nn.Module` tree directly onto a Gene
# Ontology DAG: one Linear+BatchNorm "term" block per GO term (built bottom-up from leaves via
# `construct_NN_graph`), one Linear layer per term that has directly-annotated genes
# (`contruct_direct_gene_layer`), a separate small drug-fingerprint MLP branch
# (`construct_NN_drug`), and a final fusion Linear+BatchNorm that combines the root term's
# hidden state with the drug branch's output. `drugcell_nn` is copied verbatim from
# `code/drugcell_NN.py` (module construction, `cal_term_dim`, `contruct_direct_gene_layer`,
# `construct_NN_drug`, `construct_NN_graph`, `forward`) -- no architectural change. The only
# edits: dropped the unused `sys`, `torch.autograd.Variable`, and `util` imports (the real file
# imports `util` only for training-script helpers, never referenced inside the class), and
# built a tiny synthetic 3-level Gene-Ontology DAG (in place of the real ~2,000-node GO/gene
# annotation files) with the exact structure `load_ontology()` in `code/util.py` builds:
# a directed acyclic graph of terms with `dG.add_edge(parent, child)` edges, a single root
# (in-degree-0 node), and a `term_direct_gene_map` giving each term its directly-annotated
# genes. `cell_dim`/`drug_dim` are the real model's two concatenated raw feature blocks (gene
# mutation vector + drug Morgan-fingerprint vector); the real `forward()` slices the single
# input tensor into these two blocks via `x.narrow(...)`, exactly as vendored below.

import networkx as nx
import torch
import torch.nn as nn


class drugcell_nn(nn.Module):
    def __init__(
        self,
        term_size_map,
        term_direct_gene_map,
        dG,
        ngene,
        ndrug,
        root,
        num_hiddens_genotype,
        num_hiddens_drug,
        num_hiddens_final,
    ):
        super(drugcell_nn, self).__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype
        self.num_hiddens_drug = num_hiddens_drug

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = term_direct_gene_map

        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.cal_term_dim(term_size_map)

        # ngenes, gene_dim are the number of all genes
        self.gene_dim = ngene
        self.drug_dim = ndrug

        # add modules for neural networks to process genotypes
        self.contruct_direct_gene_layer()
        self.construct_NN_graph(dG)

        # add modules for neural networks to process drugs
        self.construct_NN_drug()

        # add modules for final layer
        final_input_size = num_hiddens_genotype + num_hiddens_drug[-1]
        self.add_module("final_linear_layer", nn.Linear(final_input_size, num_hiddens_final))
        self.add_module("final_batchnorm_layer", nn.BatchNorm1d(num_hiddens_final))
        self.add_module("final_aux_linear_layer", nn.Linear(num_hiddens_final, 1))
        self.add_module("final_linear_layer_output", nn.Linear(1, 1))

    # calculate the number of values in a state (term)
    def cal_term_dim(self, term_size_map):
        self.term_dim_map = {}

        for term, term_size in term_size_map.items():
            num_output = self.num_hiddens_genotype

            # log the number of hidden variables per each term
            num_output = int(num_output)
            self.term_dim_map[term] = num_output

    # build a layer for forwarding gene that are directly annotated with the term
    def contruct_direct_gene_layer(self):
        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                print("There are no directed asscoiated genes for", term)
                raise ValueError("empty term gene set: " + term)

            # if there are some genes directly annotated with the term, add a layer taking in all genes and forwarding out only those genes
            self.add_module(term + "_direct_gene_layer", nn.Linear(self.gene_dim, len(gene_set)))

    # add modules for fully connected neural networks for drug processing
    def construct_NN_drug(self):
        input_size = self.drug_dim

        for i in range(len(self.num_hiddens_drug)):
            self.add_module(
                "drug_linear_layer_" + str(i + 1), nn.Linear(input_size, self.num_hiddens_drug[i])
            )
            self.add_module(
                "drug_batchnorm_layer_" + str(i + 1), nn.BatchNorm1d(self.num_hiddens_drug[i])
            )
            self.add_module(
                "drug_aux_linear_layer1_" + str(i + 1), nn.Linear(self.num_hiddens_drug[i], 1)
            )
            self.add_module("drug_aux_linear_layer2_" + str(i + 1), nn.Linear(1, 1))

            input_size = self.num_hiddens_drug[i]

    # start from bottom (leaves), and start building a neural network using the given ontology
    # adding modules --- the modules are not connected yet
    def construct_NN_graph(self, dG):
        self.term_layer_list = []  # term_layer_list stores the built neural network
        self.term_neighbor_map = {}

        # term_neighbor_map records all children of each term
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)

        while True:
            leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]

            if len(leaves) == 0:
                break

            self.term_layer_list.append(leaves)

            for term in leaves:
                # input size will be #chilren + #genes directly annotated by the term
                input_size = 0

                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])

                # term_hidden is the number of the hidden variables in each state
                term_hidden = self.term_dim_map[term]

                self.add_module(term + "_linear_layer", nn.Linear(input_size, term_hidden))
                self.add_module(term + "_batchnorm_layer", nn.BatchNorm1d(term_hidden))
                self.add_module(term + "_aux_linear_layer1", nn.Linear(term_hidden, 1))
                self.add_module(term + "_aux_linear_layer2", nn.Linear(1, 1))

            dG.remove_nodes_from(leaves)

    # definition of forward function
    def forward(self, x):
        gene_input = x.narrow(1, 0, self.gene_dim)
        drug_input = x.narrow(1, self.gene_dim, self.drug_dim)

        # define forward function for genotype dcell #############################################
        term_gene_out_map = {}

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + "_direct_gene_layer"](gene_input)

        term_NN_out_map = {}
        aux_out_map = {}

        for i, layer in enumerate(self.term_layer_list):
            for term in layer:
                child_input_list = []

                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child])

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list, 1)

                term_NN_out = self._modules[term + "_linear_layer"](child_input)

                Tanh_out = torch.tanh(term_NN_out)
                term_NN_out_map[term] = self._modules[term + "_batchnorm_layer"](Tanh_out)
                aux_layer1_out = torch.tanh(
                    self._modules[term + "_aux_linear_layer1"](term_NN_out_map[term])
                )
                aux_out_map[term] = self._modules[term + "_aux_linear_layer2"](aux_layer1_out)

        # define forward function for drug dcell #################################################
        drug_out = drug_input

        for i in range(1, len(self.num_hiddens_drug) + 1, 1):
            drug_out = self._modules["drug_batchnorm_layer_" + str(i)](
                torch.tanh(self._modules["drug_linear_layer_" + str(i)](drug_out))
            )
            term_NN_out_map["drug_" + str(i)] = drug_out

            aux_layer1_out = torch.tanh(self._modules["drug_aux_linear_layer1_" + str(i)](drug_out))
            aux_out_map["drug_" + str(i)] = self._modules["drug_aux_linear_layer2_" + str(i)](
                aux_layer1_out
            )

        # connect two neural networks at the top #################################################
        final_input = torch.cat((term_NN_out_map[self.root], drug_out), 1)

        out = self._modules["final_batchnorm_layer"](
            torch.tanh(self._modules["final_linear_layer"](final_input))
        )
        term_NN_out_map["final"] = out

        aux_layer_out = torch.tanh(self._modules["final_aux_linear_layer"](out))
        aux_out_map["final"] = self._modules["final_linear_layer_output"](aux_layer_out)

        return aux_out_map, term_NN_out_map


def _build_tiny_ontology():
    """Tiny synthetic Gene-Ontology-style DAG with the exact shape `load_ontology()` in the
    real `code/util.py` builds from a `.txt` ontology file: a rooted DAG of GO terms
    (`dG.add_edge(parent, child)`), plus a `term_direct_gene_map` giving genes directly
    annotated to specific terms. Real runs load a ~2,000-term GO DAG + a ~3,000-gene mutation
    matrix from `data/drugcell_ont.txt` / `data/gene2ind.txt`; this substitutes a hand-built
    3-level, 7-term DAG with 6 total genes so the traced example stays tiny -- the class
    construction logic (`construct_NN_graph`'s bottom-up leaf-stripping loop) is identical
    regardless of DAG size."""
    dG = nx.DiGraph()
    # root -> mid1, mid2 ; mid1 -> leaf1, leaf2 ; mid2 -> leaf2, leaf3
    dG.add_edge("root", "mid1")
    dG.add_edge("root", "mid2")
    dG.add_edge("mid1", "leaf1")
    dG.add_edge("mid1", "leaf2")
    dG.add_edge("mid2", "leaf2")
    dG.add_edge("mid2", "leaf3")

    term_direct_gene_map = {
        "leaf1": {0, 1},
        "leaf2": {2, 3},
        "leaf3": {4, 5},
        "mid1": {1},
        "root": {0},
    }

    term_size_map = {}
    for term in dG.nodes():
        descendants = nx.descendants(dG, term)
        gene_set = set(term_direct_gene_map.get(term, set()))
        for child in descendants:
            gene_set |= term_direct_gene_map.get(child, set())
        term_size_map[term] = len(gene_set)

    return dG, "root", term_size_map, term_direct_gene_map


def build_drugcell():
    dG, root, term_size_map, term_direct_gene_map = _build_tiny_ontology()
    ngene = 6
    ndrug = 8
    model = drugcell_nn(
        term_size_map=term_size_map,
        term_direct_gene_map=term_direct_gene_map,
        dG=dG,
        ngene=ngene,
        ndrug=ndrug,
        root=root,
        num_hiddens_genotype=4,
        num_hiddens_drug=[8, 4, 3],
        num_hiddens_final=3,
    )
    model.eval()
    return model


def example_input_drugcell():
    batch = 4
    ngene = 6
    ndrug = 8
    return torch.randn(batch, ngene + ndrug)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DrugCell", build_drugcell, example_input_drugcell, 2021, MENAGERIE_ZOO),
]
