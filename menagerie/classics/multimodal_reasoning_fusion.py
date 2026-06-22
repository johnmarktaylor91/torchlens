"""Multimodal reasoning and fusion architectures missing from the catalog.

Covered compact classics:

* Neural Module Networks (NMN), Andreas et al., CVPR 2016, arXiv:1511.02799.
  Distinctive primitive: a question-conditioned network dynamically composes reusable
  neural modules (find/transform/describe) over visual-region features.
* MAC Network, Hudson and Manning, ICLR 2018, arXiv:1803.03067.
  Distinctive primitive: recurrent Memory, Attention, and Composition cells with
  separated control and memory state for iterative visual reasoning.
* Tensor Fusion Network (TFN), Zadeh et al., EMNLP 2017, arXiv:1707.07250.
  Distinctive primitive: explicit Cartesian outer-product fusion of unimodal
  representations augmented with constant 1 terms.
* Low-rank Multimodal Fusion (LMF), Liu et al., ACL 2018, arXiv:1806.00064.
  Distinctive primitive: modality-specific low-rank factors approximate TFN's
  full fusion tensor without materializing the exponential outer-product feature.

All implementations are faithful compact forward-pass cores: random init, CPU-sized
inputs, and no dataset/parser/training losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralModuleNetwork(nn.Module):
    """Compact differentiable Neural Module Network for visual question answering."""

    def __init__(
        self,
        vocab_size: int = 64,
        region_dim: int = 24,
        word_dim: int = 32,
        hidden_dim: int = 32,
        num_answers: int = 10,
    ) -> None:
        """Initialize reusable NMN modules and a soft layout controller.

        Parameters
        ----------
        vocab_size:
            Number of question-token ids.
        region_dim:
            Width of each image-region feature vector.
        word_dim:
            Token embedding width.
        hidden_dim:
            Internal module width.
        num_answers:
            Number of VQA answer logits.
        """

        super().__init__()
        self.words = nn.Embedding(vocab_size, word_dim)
        self.query = nn.Linear(word_dim, hidden_dim)
        self.region = nn.Linear(region_dim, hidden_dim)
        self.layout = nn.Linear(word_dim, 4)
        self.find_gate = nn.Linear(hidden_dim, hidden_dim)
        self.relate_gate = nn.Linear(hidden_dim + word_dim, hidden_dim)
        self.describe = nn.Sequential(
            nn.Linear(hidden_dim + word_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_answers),
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Run a soft find-transform-describe module layout.

        Parameters
        ----------
        inputs:
            ``[regions, question]`` where regions has shape ``(B, R, region_dim)`` and
            question has shape ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Answer logits with shape ``(B, num_answers)``.
        """

        regions, question = inputs
        word_states = self.words(question.long())
        q = word_states.mean(dim=1)
        visual = torch.tanh(self.region(regions))
        query = torch.tanh(self.query(q))
        layout = torch.softmax(self.layout(q), dim=-1)

        find_logits = (visual * query.unsqueeze(1)).sum(dim=-1)
        attention = torch.softmax(find_logits, dim=-1)
        context = torch.bmm(attention.unsqueeze(1), visual).squeeze(1)

        for step in range(2):
            step_word = word_states[:, step % word_states.shape[1]]
            related = torch.tanh(self.relate_gate(torch.cat([context, step_word], dim=-1)))
            find_context = torch.tanh(self.find_gate(context))
            mix = layout[:, step : step + 1]
            context = mix * related + (1.0 - mix) * find_context

        return self.describe(torch.cat([context, q], dim=-1))


class MACCell(nn.Module):
    """One Memory-Attention-Composition reasoning cell."""

    def __init__(self, dim: int) -> None:
        """Initialize control, read, and write projections.

        Parameters
        ----------
        dim:
            Shared control, memory, and knowledge width.
        """

        super().__init__()
        self.control_proj = nn.Linear(dim * 2, dim)
        self.read_proj = nn.Linear(dim * 3, dim)
        self.write_proj = nn.Linear(dim * 3, dim)

    def forward(
        self,
        control: torch.Tensor,
        memory: torch.Tensor,
        words: torch.Tensor,
        knowledge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one MAC recurrent reasoning step.

        Parameters
        ----------
        control:
            Previous control state ``(B, D)``.
        memory:
            Previous memory state ``(B, D)``.
        words:
            Contextual question word features ``(B, T, D)``.
        knowledge:
            Visual knowledge base ``(B, R, D)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(control, memory)`` states.
        """

        control_query = self.control_proj(torch.cat([control, words.mean(dim=1)], dim=-1))
        control_attn = torch.softmax(
            torch.bmm(words, control_query.unsqueeze(-1)).squeeze(-1), dim=-1
        )
        next_control = torch.bmm(control_attn.unsqueeze(1), words).squeeze(1)

        memory_expand = memory.unsqueeze(1).expand_as(knowledge)
        control_expand = next_control.unsqueeze(1).expand_as(knowledge)
        read_features = torch.tanh(
            self.read_proj(torch.cat([knowledge, memory_expand, control_expand], dim=-1))
        )
        read_attn = torch.softmax((read_features * control_expand).sum(dim=-1), dim=-1)
        read = torch.bmm(read_attn.unsqueeze(1), knowledge).squeeze(1)
        next_memory = torch.tanh(self.write_proj(torch.cat([memory, read, next_control], dim=-1)))
        return next_control, next_memory


class MACNetwork(nn.Module):
    """Compact MAC network for iterative visual reasoning."""

    def __init__(
        self,
        vocab_size: int = 64,
        region_dim: int = 24,
        dim: int = 32,
        steps: int = 3,
        num_answers: int = 10,
    ) -> None:
        """Initialize the MAC question encoder, visual projection, and cells.

        Parameters
        ----------
        vocab_size:
            Number of question-token ids.
        region_dim:
            Width of visual-region input features.
        dim:
            Shared MAC state width.
        steps:
            Number of recurrent reasoning cells.
        num_answers:
            Number of output answer logits.
        """

        super().__init__()
        self.words = nn.Embedding(vocab_size, dim)
        self.knowledge = nn.Linear(region_dim, dim)
        self.cells = nn.ModuleList([MACCell(dim) for _ in range(steps)])
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, num_answers)
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Run recurrent MAC reasoning over regions and question tokens.

        Parameters
        ----------
        inputs:
            ``[regions, question]`` where regions has shape ``(B, R, region_dim)`` and
            question has shape ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Answer logits with shape ``(B, num_answers)``.
        """

        regions, question = inputs
        words = self.words(question.long())
        knowledge = torch.tanh(self.knowledge(regions))
        control = words.mean(dim=1)
        memory = torch.zeros_like(control)
        for cell in self.cells:
            control, memory = cell(control, memory, words, knowledge)
        return self.classifier(torch.cat([memory, control], dim=-1))


class ModalityEncoder(nn.Module):
    """Small unimodal sequence encoder used by TFN and LMF."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize a GRU encoder for one modality.

        Parameters
        ----------
        in_dim:
            Input feature width.
        out_dim:
            Output representation width.
        """

        super().__init__()
        self.rnn = nn.GRU(in_dim, out_dim, batch_first=True)
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a modality sequence.

        Parameters
        ----------
        x:
            Modality sequence ``(B, T, in_dim)``.

        Returns
        -------
        torch.Tensor
            Encoded modality vector ``(B, out_dim)``.
        """

        hidden = self.rnn(x)[1][-1]
        return torch.tanh(self.proj(hidden))


class TensorFusionNetwork(nn.Module):
    """Tensor Fusion Network with explicit tri-modal outer-product fusion."""

    def __init__(self, rep_dim: int = 8, output_dim: int = 4) -> None:
        """Initialize unimodal encoders and the TFN fusion head.

        Parameters
        ----------
        rep_dim:
            Width of each unimodal representation before appending the bias term.
        output_dim:
            Number of output logits.
        """

        super().__init__()
        self.text = ModalityEncoder(12, rep_dim)
        self.audio = ModalityEncoder(6, rep_dim)
        self.visual = ModalityEncoder(8, rep_dim)
        fusion_dim = (rep_dim + 1) ** 3
        self.head = nn.Sequential(nn.Linear(fusion_dim, 32), nn.GELU(), nn.Linear(32, output_dim))

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Fuse text, audio, and visual sequences with an explicit outer product.

        Parameters
        ----------
        inputs:
            ``[text, audio, visual]`` sequences.

        Returns
        -------
        torch.Tensor
            Output logits ``(B, output_dim)``.
        """

        text, audio, visual = inputs
        t = _append_one(self.text(text))
        a = _append_one(self.audio(audio))
        v = _append_one(self.visual(visual))
        ta = torch.einsum("bi,bj->bij", t, a).flatten(1)
        tav = torch.einsum("bi,bj->bij", ta, v).flatten(1)
        return self.head(tav)


class LowRankMultimodalFusion(nn.Module):
    """Low-rank Multimodal Fusion with modality-specific factors."""

    def __init__(self, rep_dim: int = 8, rank: int = 6, output_dim: int = 4) -> None:
        """Initialize encoders and low-rank factor tensors.

        Parameters
        ----------
        rep_dim:
            Width of each unimodal representation before appending the bias term.
        rank:
            Number of rank components in the tensor factorization.
        output_dim:
            Number of output logits.
        """

        super().__init__()
        self.text = ModalityEncoder(12, rep_dim)
        self.audio = ModalityEncoder(6, rep_dim)
        self.visual = ModalityEncoder(8, rep_dim)
        factor_shape = (rank, rep_dim + 1, output_dim)
        self.text_factor = nn.Parameter(torch.randn(factor_shape) * 0.02)
        self.audio_factor = nn.Parameter(torch.randn(factor_shape) * 0.02)
        self.visual_factor = nn.Parameter(torch.randn(factor_shape) * 0.02)
        self.fusion_weights = nn.Parameter(torch.ones(rank))
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Fuse three modalities through low-rank tensor factors.

        Parameters
        ----------
        inputs:
            ``[text, audio, visual]`` sequences.

        Returns
        -------
        torch.Tensor
            Output logits ``(B, output_dim)``.
        """

        text, audio, visual = inputs
        t = torch.einsum("bi,rio->bro", _append_one(self.text(text)), self.text_factor)
        a = torch.einsum("bi,rio->bro", _append_one(self.audio(audio)), self.audio_factor)
        v = torch.einsum("bi,rio->bro", _append_one(self.visual(visual)), self.visual_factor)
        fused = t * a * v
        return torch.einsum("r,bro->bo", self.fusion_weights, fused) + self.bias


def _append_one(x: torch.Tensor) -> torch.Tensor:
    """Append the constant-one channel used to preserve lower-order interactions.

    Parameters
    ----------
    x:
        Modality representation ``(B, D)``.

    Returns
    -------
    torch.Tensor
        Representation ``(B, D + 1)`` with a final constant-one column.
    """

    return torch.cat([x, x.new_ones(x.shape[0], 1)], dim=-1)


def build_neural_module_network() -> nn.Module:
    """Build a compact Neural Module Network classic.

    Returns
    -------
    nn.Module
        Configured ``NeuralModuleNetwork`` in eval mode.
    """

    return NeuralModuleNetwork().eval()


def build_mac_network() -> nn.Module:
    """Build a compact MAC Network classic.

    Returns
    -------
    nn.Module
        Configured ``MACNetwork`` in eval mode.
    """

    return MACNetwork().eval()


def build_tensor_fusion_network() -> nn.Module:
    """Build a compact Tensor Fusion Network classic.

    Returns
    -------
    nn.Module
        Configured ``TensorFusionNetwork`` in eval mode.
    """

    return TensorFusionNetwork().eval()


def build_low_rank_multimodal_fusion() -> nn.Module:
    """Build a compact Low-rank Multimodal Fusion classic.

    Returns
    -------
    nn.Module
        Configured ``LowRankMultimodalFusion`` in eval mode.
    """

    return LowRankMultimodalFusion().eval()


def example_input_vqa() -> list[torch.Tensor]:
    """Create visual-region and question-token inputs for VQA reasoning classics.

    Returns
    -------
    list[torch.Tensor]
        ``[regions (1, 6, 24), question (1, 5)]``.
    """

    return [torch.randn(1, 6, 24), torch.randint(0, 64, (1, 5))]


def example_input_fusion() -> list[torch.Tensor]:
    """Create text/audio/visual sequence inputs for multimodal fusion classics.

    Returns
    -------
    list[torch.Tensor]
        ``[text (1, 5, 12), audio (1, 7, 6), visual (1, 4, 8)]``.
    """

    return [torch.randn(1, 5, 12), torch.randn(1, 7, 6), torch.randn(1, 4, 8)]


MENAGERIE_ENTRIES = [
    (
        "Neural Module Networks (dynamic visual-question module composition)",
        "build_neural_module_network",
        "example_input_vqa",
        "2016",
        "DC",
    ),
    (
        "MAC Network (Memory-Attention-Composition recurrent reasoning cells)",
        "build_mac_network",
        "example_input_vqa",
        "2018",
        "DC",
    ),
    (
        "Tensor Fusion Network (explicit tri-modal outer-product fusion)",
        "build_tensor_fusion_network",
        "example_input_fusion",
        "2017",
        "DC",
    ),
    (
        "Low-rank Multimodal Fusion (modality-specific tensor factors)",
        "build_low_rank_multimodal_fusion",
        "example_input_fusion",
        "2018",
        "DC",
    ),
]
