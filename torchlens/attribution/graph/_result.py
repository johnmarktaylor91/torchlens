"""The attribution-graph result: typed nodes, a dense adjacency, and its accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["AttributionGraph", "GraphNode", "NODE_KINDS"]

#: The four node kinds, in the adjacency's node order.
NODE_KINDS: tuple[str, ...] = ("feature", "error", "token", "logit")


@dataclass(frozen=True)
class GraphNode:
    """One node of an attribution graph.

    Attributes
    ----------
    kind
        ``"feature"`` (an active transcoder feature), ``"error"`` (what the
        transcoders leave unexplained in one MLP output at one position),
        ``"token"`` (one position's embedding output) or ``"logit"`` (one
        target logit at the last position).
    site
        Module name the node belongs to: the MLP a feature's transcoder
        reads, the MLP whose output an error node completes, the embedding
        module, or the module whose output holds the logits.
    position
        Sequence position.
    feature
        Feature index inside its transcoder (feature nodes only).
    token
        Token id: the input token at ``position`` (token nodes) or the
        target vocabulary id (logit nodes).
    """

    kind: str
    site: str
    position: int
    feature: int | None = None
    token: int | None = None


@dataclass(frozen=True)
class AttributionGraph:
    """Direct linear effects between features, errors, tokens and logits.

    Node order is all feature nodes (by transcoder in execution order, then
    position, then feature index), then error nodes (by MLP in execution
    order, then position), then token nodes (by position), then logit nodes
    (in target order). ``adjacency_matrix[t, s]`` is the direct effect of
    source node ``s`` on target node ``t`` along the frozen linear graph:
    attention patterns, normalisation scales and every other site named in
    ``frozen`` are held at their captured values, and every MLP output with
    a transcoder is held fixed and explained by its feature and error nodes.

    Accounting: for every feature and logit node ``t``,
    ``adjacency_matrix[t].sum() + sum(c[t] for c in constants.values())``
    equals ``node_values[t]``. ``constants`` names the terms that are not
    nodes: ``"encoder_bias"`` (a feature's own encoder bias) and
    ``"decoder_bias"`` (the transcoders' decoder biases, read through the
    same frozen graph) are computed directly; ``"other"`` is the remainder,
    which on a frozen graph that reaches every source is exactly the
    model's own constant terms (its biases, position embeddings and any
    frozen site's offset) carried to the node. A remainder larger than
    those terms means a path to the node bypasses the graph's sources, for
    example an unfrozen nonlinearity. Error and token nodes have no
    incoming edges; their ``node_values`` and constants are NaN.

    Attributes
    ----------
    nodes
        One :class:`GraphNode` per row and column of the adjacency.
    adjacency_matrix
        ``(n_nodes, n_nodes)`` tensor, rows are targets and columns sources.
    node_values
        Each node's value: a feature's pre-activation, a logit's value
        (demeaned when ``demeaned_logits``), NaN for error and token nodes.
    constants
        Name to ``(n_nodes,)`` tensor of the non-node terms, NaN where a node
        has no incoming edges.
    active_features
        ``(n_features, 3)`` long tensor of (transcoder index, position,
        feature index), one row per feature node.
    activation_values
        ``(n_features,)`` activations of the feature nodes.
    transcoder_sites
        Module names of the MLPs whose inputs the transcoders read, in
        execution order (the transcoder index of ``active_features``).
    error_sites
        Module names of the MLP outputs that carry error nodes, in execution
        order.
    error_vectors
        ``(n_error_sites, positions, width)``: each MLP output minus what the
        transcoders reconstruct there, the vector an error node writes.
    input_tokens
        Input token ids, one per position.
    logit_tokens
        Target vocabulary ids, one per logit node.
    logit_probabilities
        Softmax probability of each target at the last position.
    demeaned_logits
        Whether a logit node's value is its logit minus the mean logit.
    frozen
        Labels of the sites held at their captured values.
    capture_outcome
        The capture's completeness verdict, as the trace reports it.
    disclosures
        Plain-language notes about what the graph does not cover.
    """

    nodes: tuple[GraphNode, ...]
    adjacency_matrix: torch.Tensor
    node_values: torch.Tensor
    constants: dict[str, torch.Tensor]
    active_features: torch.Tensor
    activation_values: torch.Tensor
    transcoder_sites: tuple[str, ...]
    error_sites: tuple[str, ...]
    error_vectors: torch.Tensor
    input_tokens: torch.Tensor
    logit_tokens: torch.Tensor
    logit_probabilities: torch.Tensor
    demeaned_logits: bool
    frozen: tuple[str, ...]
    capture_outcome: str | None
    disclosures: tuple[str, ...] = field(default=())

    @property
    def node_counts(self) -> dict[str, int]:
        """Number of nodes of each kind, in node order."""

        counts = dict.fromkeys(NODE_KINDS, 0)
        for node in self.nodes:
            counts[node.kind] += 1
        return counts

    def node_indices(self, kind: str) -> list[int]:
        """Return the indices of the nodes of one kind.

        Parameters
        ----------
        kind
            One of ``"feature"``, ``"error"``, ``"token"`` or ``"logit"``.

        Returns
        -------
        list[int]
            Node indices in node order, ready to index ``adjacency_matrix``.
        """

        return [index for index, node in enumerate(self.nodes) if node.kind == kind]
