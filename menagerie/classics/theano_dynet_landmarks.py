"""Theano/Lasagne/DyNet-era landmark architectures, 2014-2016.

Paper: Compact faithful reimplementations of notable Theano/Lasagne/DyNet-era
architectures that are less often available as modern PyTorch atlas models.

Included entries:
  - Spatial Transformer Network (Jaderberg et al., NeurIPS 2015): Lasagne
    recipe/TransformerLayer lineage with affine-grid localization.
  - Show, Attend and Tell (Xu et al., ICML 2015): Theano attention captioner
    lineage with soft visual attention over annotation vectors.
  - Neural Turing Machine (Graves et al., arXiv 2014): differentiable external
    memory with content addressing, interpolation, shift, sharpen, read/write.
  - Recurrent Neural Network Grammar (Dyer et al., NAACL 2016): DyNet stack-LSTM
    parser/generator lineage represented by fixed legal actions.
  - BiLSTM-CNN-CRF NER (Lample et al., NAACL 2016): DyNet Theano-era NER model
    with character CNN, word BiLSTM, emissions, and CRF forward score.

All models are random-init, CPU-friendly, and intentionally small for trace/draw.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformerClassifier(nn.Module):
    """Compact Spatial Transformer Network classifier with affine sampling."""

    def __init__(self, n_classes: int = 10) -> None:
        """Initialize the localization network and classifier.

        Parameters
        ----------
        n_classes:
            Number of output classes.
        """

        super().__init__()
        self.loc_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.loc_fc = nn.Linear(10, 6)
        with torch.no_grad():
            self.loc_fc.weight.zero_()
            self.loc_fc.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned affine warping, then classify the transformed image.

        Parameters
        ----------
        x:
            Image batch of shape ``(batch, 1, height, width)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(batch, n_classes)``.
        """

        theta = self.loc_fc(self.loc_conv(x)).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        warped = F.grid_sample(x, grid, align_corners=False)
        return self.classifier(warped)


class ShowAttendTellCaptioner(nn.Module):
    """Soft-attention image captioner over CNN annotation vectors."""

    def __init__(self, vocab_size: int = 64, d_model: int = 32, steps: int = 5) -> None:
        """Initialize image encoder, additive attention, and LSTM decoder.

        Parameters
        ----------
        vocab_size:
            Caption vocabulary size.
        d_model:
            Hidden and annotation width.
        steps:
            Number of unrolled decoding steps.
        """

        super().__init__()
        self.steps = steps
        self.vocab_size = vocab_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.embed = nn.Embedding(vocab_size, d_model)
        self.init_h = nn.Linear(d_model, d_model)
        self.init_c = nn.Linear(d_model, d_model)
        self.attn_ann = nn.Linear(d_model, d_model, bias=False)
        self.attn_dec = nn.Linear(d_model, d_model, bias=False)
        self.attn_score = nn.Linear(d_model, 1, bias=False)
        self.decoder = nn.LSTMCell(d_model + d_model, d_model)
        self.out = nn.Linear(d_model + d_model, vocab_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Decode a short caption with deterministic soft visual attention.

        Parameters
        ----------
        image:
            Image batch of shape ``(batch, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Vocabulary logits of shape ``(batch, steps, vocab_size)``.
        """

        annotations = self.encoder(image).flatten(2).transpose(1, 2)
        pooled = annotations.mean(dim=1)
        h = torch.tanh(self.init_h(pooled))
        c = torch.tanh(self.init_c(pooled))
        token = torch.zeros(image.shape[0], dtype=torch.long, device=image.device)
        outputs: list[torch.Tensor] = []
        ann_proj = self.attn_ann(annotations)
        for _ in range(self.steps):
            scores = self.attn_score(torch.tanh(ann_proj + self.attn_dec(h).unsqueeze(1)))
            alpha = torch.softmax(scores.squeeze(-1), dim=-1)
            context = torch.sum(alpha.unsqueeze(-1) * annotations, dim=1)
            h, c = self.decoder(torch.cat([self.embed(token), context], dim=-1), (h, c))
            logits = self.out(torch.cat([h, context], dim=-1))
            outputs.append(logits.unsqueeze(1))
            token = torch.argmax(logits, dim=-1)
        return torch.cat(outputs, dim=1)


class NeuralTuringMachine(nn.Module):
    """Small Neural Turing Machine with one read/write head."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 32,
        memory_slots: int = 8,
        memory_width: int = 8,
        steps: int = 5,
    ) -> None:
        """Initialize controller, head-parameter projection, and output head.

        Parameters
        ----------
        input_size:
            Width of each input vector.
        hidden_size:
            LSTM controller hidden width.
        memory_slots:
            Number of memory locations.
        memory_width:
            Width of each memory vector.
        steps:
            Number of sequence steps to unroll.
        """

        super().__init__()
        self.memory_slots = memory_slots
        self.memory_width = memory_width
        self.steps = steps
        self.controller = nn.LSTMCell(input_size + memory_width, hidden_size)
        self.head = nn.Linear(hidden_size, 4 * memory_width + 9)
        self.out = nn.Linear(hidden_size + memory_width, input_size)

    def _address(
        self,
        memory: torch.Tensor,
        prev_w: torch.Tensor,
        key: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        shift_logits: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NTM content-location addressing weights.

        Parameters
        ----------
        memory:
            Memory tensor of shape ``(batch, slots, width)``.
        prev_w:
            Previous attention weights of shape ``(batch, slots)``.
        key:
            Content key of shape ``(batch, width)``.
        beta:
            Positive content-strength scalar.
        gate:
            Interpolation gate scalar.
        shift_logits:
            Three-way shift logits for shifts ``[-1, 0, 1]``.
        gamma:
            Positive sharpening scalar.

        Returns
        -------
        torch.Tensor
            Addressing weights of shape ``(batch, slots)``.
        """

        similarity = F.cosine_similarity(memory, key.unsqueeze(1), dim=-1)
        content_w = torch.softmax(beta * similarity, dim=-1)
        gated = gate * content_w + (1.0 - gate) * prev_w
        shifts = torch.softmax(shift_logits, dim=-1)
        shifted = (
            shifts[:, 0:1] * torch.roll(gated, shifts=-1, dims=1)
            + shifts[:, 1:2] * gated
            + shifts[:, 2:3] * torch.roll(gated, shifts=1, dims=1)
        )
        sharpened = shifted.clamp_min(1e-6).pow(gamma)
        return sharpened / sharpened.sum(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the differentiable memory read/write loop.

        Parameters
        ----------
        x:
            Sequence tensor of shape ``(batch, steps, input_size)``.

        Returns
        -------
        torch.Tensor
            Output sequence of shape ``(batch, steps, input_size)``.
        """

        batch = x.shape[0]
        memory = x.new_zeros(batch, self.memory_slots, self.memory_width) + 1e-3
        read_w = x.new_full((batch, self.memory_slots), 1.0 / self.memory_slots)
        write_w = read_w
        read_vec = torch.sum(read_w.unsqueeze(-1) * memory, dim=1)
        h = x.new_zeros(batch, self.controller.hidden_size)
        c = x.new_zeros(batch, self.controller.hidden_size)
        outputs: list[torch.Tensor] = []
        for t in range(min(self.steps, x.shape[1])):
            h, c = self.controller(torch.cat([x[:, t], read_vec], dim=-1), (h, c))
            params = self.head(h)
            r_key, w_key, erase, add, scalars = torch.split(
                params,
                [self.memory_width, self.memory_width, self.memory_width, self.memory_width, 9],
                dim=-1,
            )
            r_beta, r_gate, r_gamma, w_beta, w_gate, w_gamma = scalars[:, :6].chunk(6, dim=-1)
            r_shift = scalars[:, 6:9]
            w_shift = torch.flip(r_shift, dims=[-1])
            read_w = self._address(
                memory,
                read_w,
                torch.tanh(r_key),
                F.softplus(r_beta) + 1.0,
                torch.sigmoid(r_gate),
                r_shift,
                F.softplus(r_gamma) + 1.0,
            )
            write_w = self._address(
                memory,
                write_w,
                torch.tanh(w_key),
                F.softplus(w_beta) + 1.0,
                torch.sigmoid(w_gate),
                w_shift,
                F.softplus(w_gamma) + 1.0,
            )
            erase_gate = torch.sigmoid(erase).unsqueeze(1)
            add_vec = torch.tanh(add).unsqueeze(1)
            memory = memory * (1.0 - write_w.unsqueeze(-1) * erase_gate)
            memory = memory + write_w.unsqueeze(-1) * add_vec
            read_vec = torch.sum(read_w.unsqueeze(-1) * memory, dim=1)
            outputs.append(self.out(torch.cat([h, read_vec], dim=-1)).unsqueeze(1))
        return torch.cat(outputs, dim=1)


class RecurrentNeuralNetworkGrammar(nn.Module):
    """Compact fixed-action RNNG with stack composition and action scoring."""

    def __init__(self, vocab_size: int = 48, action_size: int = 16, hidden_size: int = 32) -> None:
        """Initialize embeddings, recurrent state, composer, and action scorer.

        Parameters
        ----------
        vocab_size:
            Terminal vocabulary size.
        action_size:
            Action vocabulary size.
        hidden_size:
            Stack/action hidden width.
        """

        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, hidden_size)
        self.action_emb = nn.Embedding(action_size, hidden_size)
        self.stack_cell = nn.GRUCell(hidden_size, hidden_size)
        self.action_cell = nn.GRUCell(hidden_size, hidden_size)
        self.compose = nn.Linear(2 * hidden_size, hidden_size)
        self.score = nn.Linear(2 * hidden_size, action_size)

    def forward(self, words: torch.Tensor) -> torch.Tensor:
        """Execute a fixed legal NT/SHIFT/REDUCE-style action skeleton.

        Parameters
        ----------
        words:
            Token ids of shape ``(batch, 4)``.

        Returns
        -------
        torch.Tensor
            Action logits of shape ``(batch, 7, action_size)``.
        """

        batch = words.shape[0]
        stack_state = words.new_zeros(batch, self.stack_cell.hidden_size, dtype=torch.float32)
        action_state = words.new_zeros(batch, self.action_cell.hidden_size, dtype=torch.float32)
        stack_top = words.new_zeros(batch, self.stack_cell.hidden_size, dtype=torch.float32)
        actions = words.new_tensor([1, 2, 3, 3, 4, 3, 5])
        outputs: list[torch.Tensor] = []
        word_pos = 0
        for action in actions:
            action_vec = self.action_emb(action.expand(batch))
            action_state = self.action_cell(action_vec, action_state)
            if int(action.item()) == 3:
                stack_top = self.word_emb(words[:, word_pos])
                word_pos += 1
            elif int(action.item()) == 5:
                stack_top = torch.tanh(self.compose(torch.cat([stack_top, stack_state], dim=-1)))
            else:
                stack_top = action_vec
            stack_state = self.stack_cell(stack_top, stack_state)
            outputs.append(self.score(torch.cat([stack_state, action_state], dim=-1)).unsqueeze(1))
        return torch.cat(outputs, dim=1)


class BiLstmCnnCrfNer(nn.Module):
    """Lample-style word BiLSTM plus character CNN plus CRF forward score."""

    def __init__(
        self,
        vocab_size: int = 64,
        char_vocab_size: int = 32,
        tag_size: int = 7,
        word_dim: int = 24,
        char_dim: int = 12,
        hidden_size: int = 32,
    ) -> None:
        """Initialize token encoders, emission layer, and CRF transitions.

        Parameters
        ----------
        vocab_size:
            Word vocabulary size.
        char_vocab_size:
            Character vocabulary size.
        tag_size:
            Number of NER tags.
        word_dim:
            Word embedding width.
        char_dim:
            Character embedding and CNN width.
        hidden_size:
            Bidirectional LSTM output width.
        """

        super().__init__()
        self.tag_size = tag_size
        self.word_emb = nn.Embedding(vocab_size, word_dim)
        self.char_emb = nn.Embedding(char_vocab_size, char_dim)
        self.char_conv = nn.Conv1d(char_dim, char_dim, kernel_size=3, padding=1)
        self.word_lstm = nn.LSTM(
            word_dim + char_dim,
            hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.emission = nn.Linear(hidden_size, tag_size)
        self.transitions = nn.Parameter(torch.randn(tag_size, tag_size) * 0.02)
        self.start = nn.Parameter(torch.zeros(tag_size))

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute emissions and the CRF log-partition score.

        Parameters
        ----------
        inputs:
            Pair ``(word_ids, char_ids)`` where word ids are ``(batch, tokens)``
            and char ids are ``(batch, tokens, chars)``.

        Returns
        -------
        torch.Tensor
            Concatenated per-token emissions and sequence log-normalizer.
        """

        words, chars = inputs
        batch, tokens, chars_per_word = chars.shape
        char_flat = chars.reshape(batch * tokens, chars_per_word)
        char_feat = self.char_emb(char_flat).transpose(1, 2)
        char_feat = torch.max(torch.relu(self.char_conv(char_feat)), dim=-1).values
        char_feat = char_feat.reshape(batch, tokens, -1)
        word_feat = self.word_emb(words)
        encoded, _ = self.word_lstm(torch.cat([word_feat, char_feat], dim=-1))
        emissions = self.emission(encoded)
        alpha = self.start + emissions[:, 0]
        for t in range(1, tokens):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            alpha = torch.logsumexp(scores, dim=1) + emissions[:, t]
        partition = torch.logsumexp(alpha, dim=-1, keepdim=True)
        return torch.cat([emissions.flatten(start_dim=1), partition], dim=-1)


def build_spatial_transformer_network() -> nn.Module:
    """Build a compact Spatial Transformer Network.

    Returns
    -------
    nn.Module
        Random-initialized STN classifier.
    """

    return SpatialTransformerClassifier().eval()


def example_input_spatial_transformer() -> torch.Tensor:
    """Create a small grayscale image input.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 1, 28, 28)``.
    """

    return torch.randn(1, 1, 28, 28)


def build_show_attend_tell() -> nn.Module:
    """Build a compact Show-Attend-Tell captioner.

    Returns
    -------
    nn.Module
        Random-initialized visual-attention captioner.
    """

    return ShowAttendTellCaptioner().eval()


def example_input_show_attend_tell() -> torch.Tensor:
    """Create a small RGB image input.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


def build_neural_turing_machine() -> nn.Module:
    """Build a compact Neural Turing Machine.

    Returns
    -------
    nn.Module
        Random-initialized NTM.
    """

    return NeuralTuringMachine().eval()


def example_input_neural_turing_machine() -> torch.Tensor:
    """Create a short vector sequence input.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 5, 8)``.
    """

    return torch.randn(1, 5, 8)


def build_recurrent_neural_network_grammar() -> nn.Module:
    """Build a compact Recurrent Neural Network Grammar.

    Returns
    -------
    nn.Module
        Random-initialized RNNG.
    """

    return RecurrentNeuralNetworkGrammar().eval()


def example_input_recurrent_neural_network_grammar() -> torch.Tensor:
    """Create a short token sequence input.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 4)``.
    """

    return torch.randint(0, 48, (1, 4))


def build_bilstm_cnn_crf_ner() -> nn.Module:
    """Build a compact BiLSTM-CNN-CRF NER model.

    Returns
    -------
    nn.Module
        Random-initialized NER model.
    """

    return BiLstmCnnCrfNer().eval()


def example_input_bilstm_cnn_crf_ner() -> tuple[torch.Tensor, torch.Tensor]:
    """Create word and character ids for a short NER sentence.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Word ids of shape ``(1, 5)`` and char ids of shape ``(1, 5, 6)``.
    """

    return torch.randint(0, 64, (1, 5)), torch.randint(0, 32, (1, 5, 6))


MENAGERIE_ENTRIES = [
    (
        "Spatial Transformer Network (Lasagne affine-grid TransformerLayer)",
        "build_spatial_transformer_network",
        "example_input_spatial_transformer",
        "2015",
        "E5",
    ),
    (
        "Show-Attend-Tell soft visual attention captioner",
        "build_show_attend_tell",
        "example_input_show_attend_tell",
        "2015",
        "E5",
    ),
    (
        "Neural Turing Machine (content-location addressed external memory)",
        "build_neural_turing_machine",
        "example_input_neural_turing_machine",
        "2014",
        "E5",
    ),
    (
        "Recurrent Neural Network Grammar (DyNet stack-LSTM parser)",
        "build_recurrent_neural_network_grammar",
        "example_input_recurrent_neural_network_grammar",
        "2016",
        "E5",
    ),
    (
        "BiLSTM-CNN-CRF NER (Lample et al. DyNet architecture)",
        "build_bilstm_cnn_crf_ner",
        "example_input_bilstm_cnn_crf_ner",
        "2016",
        "E5",
    ),
]
