# FAITHFUL PORT of mit-han-lab/hardware-aware-transformers @ master (original framework: fairseq fork,
# vendored C/Cython extensions not installable in the base env)
# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han, ACL 2020
# https://arxiv.org/abs/2005.14187

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.module import _addindent


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
MENAGERIE_ZOO = "ported-pytorch"


def gelu_accurate(x: torch.Tensor) -> torch.Tensor:
    """Apply fairseq's accurate GELU approximation.

    Parameters
    ----------
    x
        Input tensor.

    Returns
    -------
    torch.Tensor
        Activated tensor.
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


def make_positions(
    tensor: torch.Tensor, padding_idx: int, onnx_trace: bool = False
) -> torch.Tensor:
    """Replace non-padding symbols with their position numbers.

    Parameters
    ----------
    tensor
        Token tensor of shape ``batch x time``.
    padding_idx
        Padding token id.
    onnx_trace
        Retained for API parity with fairseq; unused in this torch-only port.

    Returns
    -------
    torch.Tensor
        Position ids beginning at ``padding_idx + 1`` for non-padding tokens.
    """
    del onnx_trace
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def fill_with_neg_inf(t: torch.Tensor) -> torch.Tensor:
    """Fill a tensor with negative infinity in an fp16-compatible way.

    Parameters
    ----------
    t
        Tensor to fill.

    Returns
    -------
    torch.Tensor
        Filled tensor.
    """
    return t.float().fill_(float("-inf")).type_as(t)


def softmax(x: torch.Tensor, dim: int, onnx_trace: bool = False) -> torch.Tensor:
    """Apply fairseq's float32 softmax.

    Parameters
    ----------
    x
        Input tensor.
    dim
        Dimension to normalize.
    onnx_trace
        Retained for API parity with fairseq; unused in this torch-only port.

    Returns
    -------
    torch.Tensor
        Softmax probabilities.
    """
    del onnx_trace
    return F.softmax(x, dim=dim, dtype=torch.float32)


def get_activation_fn(activation: str) -> Any:
    """Return the activation function corresponding to ``activation``.

    Parameters
    ----------
    activation
        Activation name.

    Returns
    -------
    Any
        Callable activation.
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation in {"gelu_fast", "gelu_accurate"}:
        return gelu_accurate
    if activation == "tanh":
        return torch.tanh
    if activation == "linear":
        return lambda x: x
    raise RuntimeError(f"--activation-fn {activation} not supported")


def sample_weight(
    weight: torch.Tensor,
    sample_in_dim: int | None,
    sample_out_dim: int | None,
) -> torch.Tensor:
    """Slice a super linear weight to the sampled dimensions.

    Parameters
    ----------
    weight
        Supernet weight.
    sample_in_dim
        Sampled input dimension.
    sample_out_dim
        Sampled output dimension.

    Returns
    -------
    torch.Tensor
        Sampled weight.
    """
    sampled_weight = weight[:, :sample_in_dim]
    sampled_weight = sampled_weight[:sample_out_dim, :]
    return sampled_weight


def sample_bias(bias: torch.Tensor, sample_out_dim: int | None) -> torch.Tensor:
    """Slice a super linear bias to the sampled output dimension.

    Parameters
    ----------
    bias
        Supernet bias.
    sample_out_dim
        Sampled output dimension.

    Returns
    -------
    torch.Tensor
        Sampled bias.
    """
    return bias[:sample_out_dim]


class LinearSuper(nn.Linear):
    """Elastic linear layer from HAT's fairseq fork."""

    def __init__(
        self,
        super_in_dim: int,
        super_out_dim: int,
        bias: bool = True,
        uniform_: Any = None,
        non_linear: str = "linear",
    ) -> None:
        """Initialize the super linear layer.

        Parameters
        ----------
        super_in_dim
            Maximum input dimension.
        super_out_dim
            Maximum output dimension.
        bias
            Whether to include a bias.
        uniform_
            Optional initializer used by the original fairseq init hook.
        non_linear
            Nonlinearity name passed to the optional initializer.
        """
        super().__init__(super_in_dim, super_out_dim, bias=bias)
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim
        self.sample_in_dim: int | None = None
        self.sample_out_dim: int | None = None
        self.samples: dict[str, torch.Tensor | None] = {}
        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode: bool = True) -> None:
        """Toggle profiling mode.

        Parameters
        ----------
        mode
            Whether to resample parameters every forward.
        """
        self.profiling = mode

    def sample_parameters(self, resample: bool = False) -> dict[str, torch.Tensor | None]:
        """Return sampled parameters.

        Parameters
        ----------
        resample
            Whether to force resampling.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Sampled parameter dictionary.
        """
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias: bool, uniform_: Any, non_linear: str) -> None:
        """Initialize super parameters.

        Parameters
        ----------
        bias
            Whether bias exists.
        uniform_
            Optional initializer.
        non_linear
            Nonlinearity name passed to the optional initializer.
        """
        if uniform_ is None:
            nn.init.xavier_uniform_(self.weight)
        else:
            uniform_(self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def set_sample_config(self, sample_in_dim: int, sample_out_dim: int) -> None:
        """Set sampled input and output dimensions.

        Parameters
        ----------
        sample_in_dim
            Sampled input dimension.
        sample_out_dim
            Sampled output dimension.
        """
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self._sample_parameters()

    def _sample_parameters(self) -> dict[str, torch.Tensor | None]:
        """Slice parameters for the current sample.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Sampled parameter dictionary.
        """
        self.samples["weight"] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples["bias"] = self.bias
        if self.bias is not None:
            self.samples["bias"] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the sampled linear projection.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Projected tensor.
        """
        self.sample_parameters()
        return F.linear(x, self.samples["weight"], self.samples["bias"])

    def calc_sampled_param_num(self) -> int:
        """Count sampled parameters.

        Returns
        -------
        int
            Number of sampled parameters.
        """
        assert "weight" in self.samples
        weight_numel = self.samples["weight"].numel()
        bias_numel = self.samples["bias"].numel() if self.samples["bias"] is not None else 0
        return weight_numel + bias_numel


class LayerNormSuper(nn.LayerNorm):
    """Elastic layer norm from HAT's fairseq fork."""

    def __init__(self, super_embed_dim: int) -> None:
        """Initialize the layer norm.

        Parameters
        ----------
        super_embed_dim
            Maximum embedding dimension.
        """
        super().__init__(super_embed_dim)
        self.super_embed_dim = super_embed_dim
        self.sample_embed_dim: int | None = None
        self.samples: dict[str, torch.Tensor] = {}
        self.profiling = False

    def profile(self, mode: bool = True) -> None:
        """Toggle profiling mode.

        Parameters
        ----------
        mode
            Whether to resample parameters every forward.
        """
        self.profiling = mode

    def sample_parameters(self, resample: bool = False) -> dict[str, torch.Tensor]:
        """Return sampled parameters.

        Parameters
        ----------
        resample
            Whether to force resampling.

        Returns
        -------
        dict[str, torch.Tensor]
            Sampled parameter dictionary.
        """
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self) -> dict[str, torch.Tensor]:
        """Slice layer norm parameters.

        Returns
        -------
        dict[str, torch.Tensor]
            Sampled parameter dictionary.
        """
        self.samples["weight"] = self.weight[: self.sample_embed_dim]
        self.samples["bias"] = self.bias[: self.sample_embed_dim]
        return self.samples

    def set_sample_config(self, sample_embed_dim: int) -> None:
        """Set sampled embedding dimension.

        Parameters
        ----------
        sample_embed_dim
            Sampled embedding dimension.
        """
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sampled layer normalization.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        self.sample_parameters()
        return F.layer_norm(
            x,
            (self.sample_embed_dim,),
            weight=self.samples["weight"],
            bias=self.samples["bias"],
            eps=self.eps,
        )

    def calc_sampled_param_num(self) -> int:
        """Count sampled parameters.

        Returns
        -------
        int
            Number of sampled parameters.
        """
        assert "weight" in self.samples
        assert "bias" in self.samples
        return self.samples["weight"].numel() + self.samples["bias"].numel()


class EmbeddingSuper(nn.Embedding):
    """Elastic embedding from HAT's fairseq fork."""

    def __init__(self, num_embeddings: int, super_embed_dim: int, padding_idx: int) -> None:
        """Initialize the super embedding.

        Parameters
        ----------
        num_embeddings
            Vocabulary size.
        super_embed_dim
            Maximum embedding dimension.
        padding_idx
            Padding token id.
        """
        super().__init__(num_embeddings, super_embed_dim, padding_idx)
        self.super_embed_dim = {"encoder": super_embed_dim, "decoder": super_embed_dim}
        self.sample_embed_dim: dict[str, int | None] = {"encoder": None, "decoder": None}
        self.samples: dict[str, dict[str, torch.Tensor]] = {"encoder": {}, "decoder": {}}
        self.profiling = False
        self.reset_parameters()

    def profile(self, mode: bool = True) -> None:
        """Toggle profiling mode.

        Parameters
        ----------
        mode
            Whether to resample parameters every forward.
        """
        self.profiling = mode

    def reset_parameters(self) -> None:
        """Initialize embedding parameters."""
        super().reset_parameters()
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        nn.init.constant_(self.weight[self.padding_idx], 0)

    def set_sample_config(self, sample_embed_dim: int, part: str) -> None:
        """Set sampled embedding dimension for encoder or decoder use.

        Parameters
        ----------
        sample_embed_dim
            Sampled embedding dimension.
        part
            ``"encoder"`` or ``"decoder"``.
        """
        self.sample_embed_dim[part] = sample_embed_dim
        self._sample_parameters(part)

    def _sample_parameters(self, part: str) -> dict[str, dict[str, torch.Tensor]]:
        """Slice embedding parameters.

        Parameters
        ----------
        part
            ``"encoder"`` or ``"decoder"``.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Sampled parameter dictionary.
        """
        weight = self.weight[..., : self.sample_embed_dim[part]]
        self.samples[part]["weight"] = weight
        return self.samples

    def sample_parameters(
        self,
        part: str,
        resample: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Return sampled embedding parameters.

        Parameters
        ----------
        part
            ``"encoder"`` or ``"decoder"``.
        resample
            Whether to force resampling.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Sampled parameter dictionary.
        """
        return self._sample_parameters(part) if self.profiling or resample else self.samples

    def sampled_weight(self, part: str) -> torch.Tensor:
        """Return sampled embedding weight.

        Parameters
        ----------
        part
            ``"encoder"`` or ``"decoder"``.

        Returns
        -------
        torch.Tensor
            Sampled weight.
        """
        return self.sample_parameters(part)[part]["weight"]

    def forward(self, input: torch.Tensor, part: str = "encoder") -> torch.Tensor:
        """Embed tokens with the sampled weight.

        Parameters
        ----------
        input
            Token ids.
        part
            ``"encoder"`` or ``"decoder"``.

        Returns
        -------
        torch.Tensor
            Embedded tokens.
        """
        return F.embedding(
            input,
            self.sampled_weight(part),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class LearnedPositionalEmbedding(nn.Embedding):
    """Learned positional embeddings from fairseq."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int) -> None:
        """Initialize learned positional embeddings.

        Parameters
        ----------
        num_embeddings
            Number of positions.
        embedding_dim
            Embedding dimension.
        padding_idx
            Padding index.
        """
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False

    def forward(
        self,
        input: torch.Tensor,
        incremental_state: dict[str, Any] | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed positions for an input token tensor.

        Parameters
        ----------
        input
            Token ids of shape ``batch x time``.
        incremental_state
            Optional incremental decoding cache.
        positions
            Optional precomputed position ids.

        Returns
        -------
        torch.Tensor
            Positional embeddings.
        """
        assert (positions is None) or (self.padding_idx is None), (
            "If positions is pre-computed then padding_idx should not be set."
        )
        if positions is None:
            if incremental_state is not None:
                positions = input.data.new(1, 1).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = make_positions(input.data, self.padding_idx, onnx_trace=self.onnx_trace)
        return super().forward(positions)

    def max_positions(self) -> int:
        """Return the maximum number of supported positions.

        Returns
        -------
        int
            Maximum positions.
        """
        if self.padding_idx is not None:
            return self.num_embeddings - self.padding_idx - 1
        return self.num_embeddings


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings from fairseq."""

    def __init__(self, embedding_dim: int, padding_idx: int, init_size: int = 1024) -> None:
        """Initialize sinusoidal positional embeddings.

        Parameters
        ----------
        embedding_dim
            Embedding dimension.
        padding_idx
            Padding index.
        init_size
            Initial table size.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ) -> torch.Tensor:
        """Build sinusoidal embeddings.

        Parameters
        ----------
        num_embeddings
            Number of positions.
        embedding_dim
            Embedding dimension.
        padding_idx
            Optional padding index.

        Returns
        -------
        torch.Tensor
            Sinusoidal table.
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input: torch.Tensor,
        incremental_state: dict[str, Any] | None = None,
        timestep: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Embed positions for an input token tensor.

        Parameters
        ----------
        input
            Token ids of shape ``batch x time``.
        incremental_state
            Optional incremental decoding cache.
        timestep
            Optional timestep for incremental decoding.
        **kwargs
            Unused fairseq compatibility arguments.

        Returns
        -------
        torch.Tensor
            Positional embeddings.
        """
        del kwargs
        bsz, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)
        if incremental_state is not None:
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
        positions = make_positions(input, self.padding_idx, onnx_trace=self.onnx_trace)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self) -> int:
        """Return the maximum number of supported positions.

        Returns
        -------
        int
            Maximum positions.
        """
        return int(1e5)


def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    learned: bool = False,
) -> nn.Module:
    """Construct learned or sinusoidal positional embeddings.

    Parameters
    ----------
    num_embeddings
        Number of positions.
    embedding_dim
        Embedding dimension.
    padding_idx
        Padding index.
    learned
        Whether to use learned embeddings.

    Returns
    -------
    nn.Module
        Positional embedding module.
    """
    if learned:
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings + padding_idx + 1,
        )
    return m


class MultiheadAttentionSuper(nn.Module):
    """Elastic multi-head attention from HAT's fairseq fork."""

    def __init__(
        self,
        super_embed_dim: int,
        num_heads: int,
        is_encoder: bool,
        super_kdim: int | None = None,
        super_vdim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        out_dim: int | None = None,
        qkv_dim: int | None = None,
    ) -> None:
        """Initialize elastic attention.

        Parameters
        ----------
        super_embed_dim
            Maximum query embedding dimension.
        num_heads
            Number of attention heads in the supernet.
        is_encoder
            Whether the module is in the encoder.
        super_kdim
            Maximum key embedding dimension.
        super_vdim
            Maximum value embedding dimension.
        dropout
            Attention dropout probability.
        bias
            Whether projections use bias.
        add_bias_kv
            Whether to append learned key/value bias tokens.
        add_zero_attn
            Whether to append zero attention.
        self_attention
            Whether this is self attention.
        encoder_decoder_attention
            Whether this is encoder-decoder attention.
        out_dim
            Maximum output dimension.
        qkv_dim
            Projected q/k/v dimension.
        """
        super().__init__()
        self.super_q_embed_dim = super_embed_dim
        self.super_kv_embed_dim = super_kdim if super_kdim is not None else self.super_q_embed_dim
        if super_kdim is not None:
            assert super_kdim == super_vdim
        self.sample_q_embed_dim: int | None = None
        self.sample_kv_embed_dim: int | None = None
        self.qkv_dim = self.super_q_embed_dim if qkv_dim is None else qkv_dim
        self.qkv_same_dim = self.super_kv_embed_dim == self.super_q_embed_dim
        self.encoder = is_encoder
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.qkv_dim // num_heads
        assert self.head_dim * num_heads == self.qkv_dim, "qkv must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim
        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * self.qkv_dim, self.super_q_embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(self.qkv_dim, self.super_kv_embed_dim))
            self.v_proj_weight = Parameter(torch.Tensor(self.qkv_dim, self.super_kv_embed_dim))
            self.q_proj_weight = Parameter(torch.Tensor(self.qkv_dim, self.super_q_embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * self.qkv_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        if out_dim is None:
            out_dim = self.super_q_embed_dim
        self.out_proj = LinearSuper(super_in_dim=self.qkv_dim, super_out_dim=out_dim, bias=bias)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, self.super_q_embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.super_q_embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        self.onnx_trace = False

    def calc_sampled_param_num(self) -> int:
        """Count sampled attention projection parameters.

        Returns
        -------
        int
            Number of sampled parameters, excluding output projection.
        """
        assert self.in_proj_weight is not None and self.in_proj_bias is not None
        in_proj_q_weight_numel = self.sample_q_embed_dim * self.qkv_dim
        in_proj_v_weight_numel = self.sample_kv_embed_dim * self.qkv_dim
        in_proj_k_weight_numel = self.sample_kv_embed_dim * self.qkv_dim
        in_proj_bias_numel = self.in_proj_bias.numel()
        return (
            in_proj_q_weight_numel
            + in_proj_k_weight_numel
            + in_proj_v_weight_numel
            + in_proj_bias_numel
        )

    def set_sample_config(
        self,
        sample_q_embed_dim: int,
        sample_attention_heads: int,
        sample_kv_embed_dim: int | None = None,
    ) -> None:
        """Set sampled dimensions and head count.

        Parameters
        ----------
        sample_q_embed_dim
            Sampled query dimension.
        sample_attention_heads
            Sampled head count.
        sample_kv_embed_dim
            Sampled key/value dimension.
        """
        self.sample_q_embed_dim = sample_q_embed_dim
        self.sample_kv_embed_dim = (
            sample_q_embed_dim if sample_kv_embed_dim is None else sample_kv_embed_dim
        )
        self.num_heads = sample_attention_heads
        self.head_dim = self.qkv_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.qkv_dim
        self.scaling = self.head_dim**-0.5
        self.out_proj.set_sample_config(
            sample_in_dim=self.qkv_dim,
            sample_out_dim=self.sample_q_embed_dim,
        )

    def reset_parameters(self) -> None:
        """Initialize attention parameters."""
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None = None,
        incremental_state: dict[str, Any] | None = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply elastic scaled dot-product multi-head attention.

        Parameters
        ----------
        query
            Query tensor of shape ``time x batch x channel``.
        key
            Key tensor.
        value
            Value tensor.
        key_padding_mask
            Optional padding mask of shape ``batch x source_time``.
        incremental_state
            Optional incremental decoding cache.
        need_weights
            Whether to return averaged attention weights.
        static_kv
            Whether key/value cache is static.
        attn_mask
            Optional attention mask.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Attention output and optional attention weights.
        """
        tgt_len, bsz, _embed_dim = query.size()
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_key" in saved_state and static_kv:
                assert self.encoder_decoder_attention and not self.self_attention
                key = value = None
        else:
            saved_state = None
        if self.self_attention:
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if saved_state is not None:
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"].view(bsz * self.num_heads, -1, self.head_dim)
                k = prev_key if static_kv else torch.cat((prev_key, k), dim=1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"].view(bsz * self.num_heads, -1, self.head_dim)
                v = prev_value if static_kv else torch.cat((prev_value, v), dim=1)
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            self._set_input_buffer(incremental_state, saved_state)
        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None
        if key_padding_mask is not None:
            fil = key_padding_mask.new_ones(
                key_padding_mask.size(0),
                src_len - key_padding_mask.size(1),
            )
            key_padding_mask = torch.cat((key_padding_mask, fil), dim=1)
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace).type_as(
            attn_weights
        )
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.qkv_dim)
        attn = self.out_proj(attn)
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None
        return attn, attn_weights

    def in_proj_qkv(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project query to q/k/v for self attention.

        Parameters
        ----------
        query
            Query tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Query, key, and value projections.
        """
        return self._in_proj(query, sample_dim=self.sample_q_embed_dim).chunk(3, dim=-1)

    def in_proj_q(self, query: torch.Tensor) -> torch.Tensor:
        """Project query.

        Parameters
        ----------
        query
            Query tensor.

        Returns
        -------
        torch.Tensor
            Query projection.
        """
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.qkv_dim, sample_dim=self.sample_q_embed_dim)
        bias = self.in_proj_bias
        if bias is not None:
            bias = bias[: self.qkv_dim]
        return F.linear(query, self.q_proj_weight[..., : self.sample_q_embed_dim], bias)

    def in_proj_k(self, key: torch.Tensor) -> torch.Tensor:
        """Project key.

        Parameters
        ----------
        key
            Key tensor.

        Returns
        -------
        torch.Tensor
            Key projection.
        """
        if self.qkv_same_dim:
            return self._in_proj(
                key,
                start=self.qkv_dim,
                end=2 * self.qkv_dim,
                sample_dim=self.sample_kv_embed_dim,
            )
        weight = self.k_proj_weight
        bias = self.in_proj_bias
        if bias is not None:
            bias = bias[self.qkv_dim : 2 * self.qkv_dim]
        return F.linear(key, weight[..., : self.sample_kv_embed_dim], bias)

    def in_proj_v(self, value: torch.Tensor) -> torch.Tensor:
        """Project value.

        Parameters
        ----------
        value
            Value tensor.

        Returns
        -------
        torch.Tensor
            Value projection.
        """
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.qkv_dim, sample_dim=self.sample_kv_embed_dim)
        weight = self.v_proj_weight
        bias = self.in_proj_bias
        if bias is not None:
            bias = bias[2 * self.qkv_dim :]
        return F.linear(value, weight[..., : self.sample_kv_embed_dim], bias)

    def _in_proj(
        self,
        input: torch.Tensor,
        sample_dim: int | None,
        start: int = 0,
        end: int | None = None,
    ) -> torch.Tensor:
        """Apply a slice of the packed q/k/v projection.

        Parameters
        ----------
        input
            Input tensor.
        sample_dim
            Sampled input dimension.
        start
            Start row of the packed projection.
        end
            End row of the packed projection.

        Returns
        -------
        torch.Tensor
            Projected tensor.
        """
        weight = self.in_proj_weight[start:end, :sample_dim]
        bias = self.in_proj_bias
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(
        self,
        incremental_state: dict[str, Any],
        new_order: torch.Tensor,
    ) -> None:
        """Reorder cached incremental attention state.

        Parameters
        ----------
        incremental_state
            Incremental cache.
        new_order
            New batch order.
        """
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer:
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Get this module's incremental attention cache.

        Parameters
        ----------
        incremental_state
            Incremental cache.

        Returns
        -------
        dict[str, torch.Tensor]
            Module-specific cache.
        """
        return incremental_state.get(f"{id(self)}.attn_state", {})

    def _set_input_buffer(
        self,
        incremental_state: dict[str, Any],
        buffer: dict[str, torch.Tensor],
    ) -> None:
        """Set this module's incremental attention cache.

        Parameters
        ----------
        incremental_state
            Incremental cache.
        buffer
            Module-specific cache.
        """
        incremental_state[f"{id(self)}.attn_state"] = buffer

    def apply_sparse_mask(
        self,
        attn_weights: torch.Tensor,
        tgt_len: int,
        src_len: int,
        bsz: int,
    ) -> torch.Tensor:
        """Apply sparse attention mask hook.

        Parameters
        ----------
        attn_weights
            Attention weights.
        tgt_len
            Target length.
        src_len
            Source length.
        bsz
            Batch size.

        Returns
        -------
        torch.Tensor
            Unchanged attention weights.
        """
        del tgt_len, src_len, bsz
        return attn_weights

    def __repr__(self) -> str:
        """Return fairseq-style module representation.

        Returns
        -------
        str
            Representation string.
        """
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines
        main_str = (
            self._get_name()
            + "\tnum_heads:"
            + str(self.num_heads)
            + "\t qkv_dim:"
            + str(self.qkv_dim)
        )
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str


class SuperTransformerConfig:
    """Plain config object matching HAT ``base_architecture`` defaults."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize config with HAT defaults and optional overrides.

        Parameters
        ----------
        **kwargs
            Field overrides.
        """
        self.encoder_embed_dim = 512
        self.encoder_ffn_embed_dim = 2048
        self.encoder_layers = 6
        self.encoder_attention_heads = 8
        self.encoder_normalize_before = False
        self.encoder_learned_pos = False
        self.decoder_embed_dim = self.encoder_embed_dim
        self.decoder_ffn_embed_dim = self.encoder_ffn_embed_dim
        self.decoder_layers = 6
        self.decoder_attention_heads = 8
        self.decoder_normalize_before = False
        self.decoder_learned_pos = False
        self.attention_dropout = 0.0
        self.activation_dropout = 0.0
        self.activation_fn = "relu"
        self.dropout = 0.1
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        self.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        self.no_token_positional_embeddings = False
        self.vocab_original_scaling = False
        self.qkv_dim = None
        self.get_attn = False
        self.tie_adaptive_weights = False
        self.adaptive_softmax_cutoff = None
        self.adaptive_softmax_dropout = 0.0
        self.share_decoder_input_output_embed = False
        self.decoder_output_dim = self.decoder_embed_dim
        self.encoder_embed_choice = [512, 256, 128]
        self.decoder_embed_choice = [512, 256, 128]
        self.encoder_layer_num_choice = [7, 6, 5, 4, 3, 2]
        self.decoder_layer_num_choice = [7, 6, 5, 4, 3, 2]
        self.encoder_ffn_embed_dim_choice = [4096, 3072, 2048, 1024]
        self.decoder_ffn_embed_dim_choice = [4096, 3072, 2048, 1024]
        self.encoder_self_attention_heads_choice = [16, 8, 4, 2, 1]
        self.decoder_self_attention_heads_choice = [16, 8, 4, 2, 1]
        self.decoder_ende_attention_heads_choice = [16, 8, 4, 2, 1]
        self.decoder_arbitrary_ende_attn_choice = [-1, 1, 2]
        self.no_decoder_final_norm = False
        for key, value in kwargs.items():
            setattr(self, key, value)


class TransformerEncoderLayer(nn.Module):
    """HAT Transformer encoder layer."""

    def __init__(self, args: SuperTransformerConfig, layer_idx: int) -> None:
        """Initialize the encoder layer.

        Parameters
        ----------
        args
            HAT config.
        layer_idx
            Layer index.
        """
        super().__init__()
        del layer_idx
        self.super_embed_dim = args.encoder_embed_dim
        self.super_ffn_embed_dim_this_layer = args.encoder_ffn_embed_dim
        self.super_self_attention_heads_this_layer = args.encoder_attention_heads
        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, "activation_dropout", 0)
        self.sample_embed_dim: int | None = None
        self.sample_ffn_embed_dim_this_layer: int | None = None
        self.sample_self_attention_heads_this_layer: int | None = None
        self.sample_dropout: float | None = None
        self.sample_activation_dropout: float | None = None
        self.is_identity_layer: bool | None = None
        self.qkv_dim = args.qkv_dim
        self.self_attn = MultiheadAttentionSuper(
            super_embed_dim=self.super_embed_dim,
            num_heads=self.super_self_attention_heads_this_layer,
            is_encoder=True,
            dropout=args.attention_dropout,
            self_attention=True,
            qkv_dim=self.qkv_dim,
        )
        self.self_attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.dropout = args.dropout
        self.activation_fn = get_activation_fn(activation=getattr(args, "activation_fn", "relu"))
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = LinearSuper(
            super_in_dim=self.super_embed_dim,
            super_out_dim=self.super_ffn_embed_dim_this_layer,
            uniform_=None,
            non_linear="relu",
        )
        self.fc2 = LinearSuper(
            super_in_dim=self.super_ffn_embed_dim_this_layer,
            super_out_dim=self.super_embed_dim,
            uniform_=None,
            non_linear="linear",
        )
        self.final_layer_norm = LayerNormSuper(self.super_embed_dim)

    def set_sample_config(
        self,
        is_identity_layer: bool,
        sample_embed_dim: int | None = None,
        sample_ffn_embed_dim_this_layer: int | None = None,
        sample_self_attention_heads_this_layer: int | None = None,
        sample_dropout: float | None = None,
        sample_activation_dropout: float | None = None,
    ) -> None:
        """Set sampled layer configuration.

        Parameters
        ----------
        is_identity_layer
            Whether this layer is skipped.
        sample_embed_dim
            Sampled embedding dimension.
        sample_ffn_embed_dim_this_layer
            Sampled FFN dimension.
        sample_self_attention_heads_this_layer
            Sampled self-attention heads.
        sample_dropout
            Sampled dropout.
        sample_activation_dropout
            Sampled activation dropout.
        """
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_embed_dim = sample_embed_dim
        self.sample_ffn_embed_dim_this_layer = sample_ffn_embed_dim_this_layer
        self.sample_self_attention_heads_this_layer = sample_self_attention_heads_this_layer
        self.sample_dropout = sample_dropout
        self.sample_activation_dropout = sample_activation_dropout
        self.self_attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        self.self_attn.set_sample_config(
            sample_q_embed_dim=self.sample_embed_dim,
            sample_attention_heads=self.sample_self_attention_heads_this_layer,
        )
        self.fc1.set_sample_config(
            sample_in_dim=self.sample_embed_dim,
            sample_out_dim=self.sample_ffn_embed_dim_this_layer,
        )
        self.fc2.set_sample_config(
            sample_in_dim=self.sample_ffn_embed_dim_this_layer,
            sample_out_dim=self.sample_embed_dim,
        )
        self.final_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_padding_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the encoder layer.

        Parameters
        ----------
        x
            Input tensor of shape ``time x batch x channel``.
        encoder_padding_mask
            Optional encoder padding mask.
        attn_mask
            Optional attention mask.

        Returns
        -------
        torch.Tensor
            Layer output.
        """
        if self.is_identity_layer:
            return x
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.byte(), -1e8)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x[: residual.size(0), :, :] = residual + x[: residual.size(0), :, :]
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = residual + x
        return self.maybe_layer_norm(self.final_layer_norm, x, after=True)

    def maybe_layer_norm(
        self,
        layer_norm: LayerNormSuper,
        x: torch.Tensor,
        before: bool = False,
        after: bool = False,
    ) -> torch.Tensor:
        """Apply layer norm according to pre/post-norm mode.

        Parameters
        ----------
        layer_norm
            Layer norm module.
        x
            Input tensor.
        before
            Whether this is the pre-normalization site.
        after
            Whether this is the post-normalization site.

        Returns
        -------
        torch.Tensor
            Possibly normalized tensor.
        """
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """HAT Transformer decoder layer."""

    def __init__(
        self,
        args: SuperTransformerConfig,
        layer_idx: int,
        no_encoder_attn: bool = False,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
    ) -> None:
        """Initialize the decoder layer.

        Parameters
        ----------
        args
            HAT config.
        layer_idx
            Layer index.
        no_encoder_attn
            Whether to omit encoder attention.
        add_bias_kv
            Whether self-attention appends learned key/value bias.
        add_zero_attn
            Whether self-attention appends zero attention.
        """
        super().__init__()
        self.super_embed_dim = args.decoder_embed_dim
        self.super_encoder_embed_dim = args.encoder_embed_dim
        self.super_ffn_embed_dim_this_layer = args.decoder_ffn_embed_dim
        self.super_self_attention_heads_this_layer = args.decoder_attention_heads
        self.super_ende_attention_heads_this_layer = args.decoder_attention_heads
        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, "activation_dropout", 0)
        self.sample_embed_dim: int | None = None
        self.sample_encoder_embed_dim: int | None = None
        self.sample_ffn_embed_dim_this_layer: int | None = None
        self.sample_self_attention_heads_this_layer: int | None = None
        self.sample_ende_attention_heads_this_layer: int | None = None
        self.sample_dropout: float | None = None
        self.sample_activation_dropout: float | None = None
        self.is_identity_layer: bool | None = None
        self.qkv_dim = args.qkv_dim
        self.layer_idx = layer_idx
        self.self_attn = MultiheadAttentionSuper(
            is_encoder=False,
            super_embed_dim=self.super_embed_dim,
            num_heads=self.super_self_attention_heads_this_layer,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            qkv_dim=self.qkv_dim,
        )
        self.activation_fn = get_activation_fn(activation=getattr(args, "activation_fn", "relu"))
        self.normalize_before = args.decoder_normalize_before
        self.self_attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttentionSuper(
                super_embed_dim=self.super_embed_dim,
                num_heads=self.super_ende_attention_heads_this_layer,
                is_encoder=False,
                super_kdim=self.super_encoder_embed_dim,
                super_vdim=self.super_encoder_embed_dim,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                qkv_dim=self.qkv_dim,
            )
            self.encoder_attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.fc1 = LinearSuper(
            super_in_dim=self.super_embed_dim,
            super_out_dim=self.super_ffn_embed_dim_this_layer,
            uniform_=None,
            non_linear="relu",
        )
        self.fc2 = LinearSuper(
            super_in_dim=self.super_ffn_embed_dim_this_layer,
            super_out_dim=self.super_embed_dim,
            uniform_=None,
            non_linear="linear",
        )
        self.final_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.need_attn = True
        self.onnx_trace = False

    def set_sample_config(
        self,
        is_identity_layer: bool,
        sample_embed_dim: int | None = None,
        sample_encoder_embed_dim: int | None = None,
        sample_ffn_embed_dim_this_layer: int | None = None,
        sample_self_attention_heads_this_layer: int | None = None,
        sample_ende_attention_heads_this_layer: int | None = None,
        sample_dropout: float | None = None,
        sample_activation_dropout: float | None = None,
    ) -> None:
        """Set sampled layer configuration.

        Parameters
        ----------
        is_identity_layer
            Whether this layer is skipped.
        sample_embed_dim
            Sampled decoder embedding dimension.
        sample_encoder_embed_dim
            Sampled encoder embedding dimension.
        sample_ffn_embed_dim_this_layer
            Sampled FFN dimension.
        sample_self_attention_heads_this_layer
            Sampled self-attention heads.
        sample_ende_attention_heads_this_layer
            Sampled encoder-decoder attention heads.
        sample_dropout
            Sampled dropout.
        sample_activation_dropout
            Sampled activation dropout.
        """
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_embed_dim = sample_embed_dim
        self.sample_encoder_embed_dim = sample_encoder_embed_dim
        self.sample_ffn_embed_dim_this_layer = sample_ffn_embed_dim_this_layer
        self.sample_self_attention_heads_this_layer = sample_self_attention_heads_this_layer
        self.sample_ende_attention_heads_this_layer = sample_ende_attention_heads_this_layer
        self.sample_dropout = sample_dropout
        self.sample_activation_dropout = sample_activation_dropout
        self.self_attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        self.encoder_attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        self.self_attn.set_sample_config(
            sample_q_embed_dim=self.sample_embed_dim,
            sample_attention_heads=self.sample_self_attention_heads_this_layer,
        )
        self.encoder_attn.set_sample_config(
            sample_q_embed_dim=self.sample_embed_dim,
            sample_kv_embed_dim=self.sample_encoder_embed_dim,
            sample_attention_heads=self.sample_ende_attention_heads_this_layer,
        )
        self.fc1.set_sample_config(
            sample_in_dim=self.sample_embed_dim,
            sample_out_dim=self.sample_ffn_embed_dim_this_layer,
        )
        self.fc2.set_sample_config(
            sample_in_dim=self.sample_ffn_embed_dim_this_layer,
            sample_out_dim=self.sample_embed_dim,
        )
        self.final_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor | None = None,
        encoder_padding_mask: torch.Tensor | None = None,
        incremental_state: dict[str, Any] | None = None,
        prev_self_attn_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_attn_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        self_attn_mask: torch.Tensor | None = None,
        self_attn_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the decoder layer.

        Parameters
        ----------
        x
            Input tensor of shape ``time x batch x channel``.
        encoder_out
            Encoder output tensor.
        encoder_padding_mask
            Optional encoder padding mask.
        incremental_state
            Optional incremental cache.
        prev_self_attn_state
            Optional previous self-attention cache.
        prev_attn_state
            Optional previous encoder-attention cache.
        self_attn_mask
            Optional causal attention mask.
        self_attn_padding_mask
            Optional self-attention padding mask.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Layer output and optional attention.
        """
        if self.is_identity_layer:
            return x, None
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            self.self_attn._set_input_buffer(
                incremental_state,
                {"prev_key": prev_key, "prev_value": prev_value},
            )
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                self.encoder_attn._set_input_buffer(
                    incremental_state,
                    {"prev_key": prev_key, "prev_value": prev_value},
                )
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.sample_dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(
        self,
        layer_norm: LayerNormSuper,
        x: torch.Tensor,
        before: bool = False,
        after: bool = False,
    ) -> torch.Tensor:
        """Apply layer norm according to pre/post-norm mode.

        Parameters
        ----------
        layer_norm
            Layer norm module.
        x
            Input tensor.
        before
            Whether this is the pre-normalization site.
        after
            Whether this is the post-normalization site.

        Returns
        -------
        torch.Tensor
            Possibly normalized tensor.
        """
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        return x


class TransformerEncoder(nn.Module):
    """HAT Transformer encoder."""

    def __init__(
        self,
        args: SuperTransformerConfig,
        vocab_size: int,
        padding_idx: int,
        embed_tokens: EmbeddingSuper,
    ):
        """Initialize the encoder.

        Parameters
        ----------
        args
            HAT config.
        vocab_size
            Source vocabulary size.
        padding_idx
            Padding index.
        embed_tokens
            Encoder embedding module.
        """
        super().__init__()
        del vocab_size
        self.padding_idx = padding_idx
        self.super_embed_dim = args.encoder_embed_dim
        self.super_ffn_embed_dim = [args.encoder_ffn_embed_dim] * args.encoder_layers
        self.super_layer_num = args.encoder_layers
        self.super_self_attention_heads = [args.encoder_attention_heads] * args.encoder_layers
        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, "activation_dropout", 0)
        self.super_embed_scale = math.sqrt(self.super_embed_dim)
        self.sample_embed_dim: int | None = None
        self.sample_ffn_embed_dim: list[int] | None = None
        self.sample_layer_num: int | None = None
        self.sample_self_attention_heads: list[int] | None = None
        self.sample_dropout: float | None = None
        self.sample_activation_dropout: float | None = None
        self.sample_embed_scale: float | None = None
        self.register_buffer("version", torch.Tensor([3]))
        self.max_source_positions = args.max_source_positions
        self.embed_tokens = embed_tokens
        if not args.no_token_positional_embeddings:
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions,
                self.super_embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
        else:
            self.embed_positions = None
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args, layer_idx=i) for i in range(self.super_layer_num)]
        )
        self.layer_norm = (
            LayerNormSuper(self.super_embed_dim) if args.encoder_normalize_before else None
        )
        self.vocab_original_scaling = args.vocab_original_scaling

    def set_sample_config(self, config: dict[str, dict[str, Any]]) -> None:
        """Set sampled encoder configuration.

        Parameters
        ----------
        config
            Nested HAT sample config.
        """
        self.sample_embed_dim = config["encoder"]["encoder_embed_dim"]
        self.sample_ffn_embed_dim = config["encoder"]["encoder_ffn_embed_dim"]
        self.sample_layer_num = config["encoder"]["encoder_layer_num"]
        self.sample_self_attention_heads = config["encoder"]["encoder_self_attention_heads"]
        self.sample_dropout = calc_dropout(
            self.super_dropout, self.sample_embed_dim, self.super_embed_dim
        )
        self.sample_activation_dropout = calc_dropout(
            self.super_activation_dropout,
            self.sample_embed_dim,
            self.super_embed_dim,
        )
        self.sample_embed_scale = (
            math.sqrt(self.sample_embed_dim)
            if not self.vocab_original_scaling
            else self.super_embed_scale
        )
        self.embed_tokens.set_sample_config(sample_embed_dim=self.sample_embed_dim, part="encoder")
        if self.layer_norm is not None:
            self.layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        for i, layer in enumerate(self.layers):
            if i < self.sample_layer_num:
                layer.set_sample_config(
                    is_identity_layer=False,
                    sample_embed_dim=self.sample_embed_dim,
                    sample_ffn_embed_dim_this_layer=self.sample_ffn_embed_dim[i],
                    sample_self_attention_heads_this_layer=self.sample_self_attention_heads[i],
                    sample_dropout=self.sample_dropout,
                    sample_activation_dropout=self.sample_activation_dropout,
                )
            else:
                layer.set_sample_config(is_identity_layer=True)

    def forward(self, src_tokens: torch.Tensor, src_lengths: torch.Tensor) -> dict[str, Any]:
        """Apply the encoder.

        Parameters
        ----------
        src_tokens
            Source tokens of shape ``batch x time``.
        src_lengths
            Source lengths, retained for signature parity.

        Returns
        -------
        dict[str, Any]
            Encoder outputs, all layer outputs, and padding mask.
        """
        del src_lengths
        x = self.sample_embed_scale * self.embed_tokens(src_tokens, part="encoder")
        if self.embed_positions is not None:
            positions = self.embed_positions(src_tokens)
            x += positions[..., : self.sample_embed_dim]
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        all_x = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            all_x.append(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        return {
            "encoder_out": x,
            "encoder_out_all": all_x,
            "encoder_padding_mask": encoder_padding_mask,
        }

    def max_positions(self) -> int:
        """Return maximum supported source length.

        Returns
        -------
        int
            Maximum positions.
        """
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerDecoder(nn.Module):
    """HAT Transformer decoder."""

    def __init__(
        self,
        args: SuperTransformerConfig,
        vocab_size: int,
        padding_idx: int,
        embed_tokens: EmbeddingSuper,
        no_encoder_attn: bool = False,
    ) -> None:
        """Initialize the decoder.

        Parameters
        ----------
        args
            HAT config.
        vocab_size
            Target vocabulary size.
        padding_idx
            Padding index.
        embed_tokens
            Decoder embedding module.
        no_encoder_attn
            Whether to omit encoder attention.
        """
        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.super_embed_dim = args.decoder_embed_dim
        self.super_ffn_embed_dim = [args.decoder_ffn_embed_dim] * args.decoder_layers
        self.super_layer_num = args.decoder_layers
        self.super_self_attention_heads = [args.decoder_attention_heads] * args.decoder_layers
        self.super_ende_attention_heads = [args.decoder_attention_heads] * args.decoder_layers
        self.super_arbitrary_ende_attn = [-1] * args.decoder_layers
        self.super_dropout = args.dropout
        self.super_activation_dropout = getattr(args, "activation_dropout", 0)
        self.super_embed_scale = math.sqrt(self.super_embed_dim)
        self.sample_embed_dim: int | None = None
        self.sample_encoder_embed_dim: int | None = None
        self.sample_ffn_embed_dim: list[int] | None = None
        self.sample_layer_num: int | None = None
        self.sample_self_attention_heads: list[int] | None = None
        self.sample_ende_attention_heads: list[int] | None = None
        self.sample_arbitrary_ende_attn: list[int] | None = None
        self.sample_dropout: float | None = None
        self.sample_activation_dropout: float | None = None
        self.sample_embed_scale: float | None = None
        self.register_buffer("version", torch.Tensor([3]))
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.output_embed_dim = args.decoder_output_dim
        self.max_target_positions = args.max_target_positions
        self.embed_tokens = embed_tokens
        if not args.no_token_positional_embeddings:
            self.embed_positions = PositionalEmbedding(
                args.max_target_positions,
                self.super_embed_dim,
                padding_idx,
                learned=args.decoder_learned_pos,
            )
        else:
            self.embed_positions = None
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(args, layer_idx=i, no_encoder_attn=no_encoder_attn)
                for i in range(self.super_layer_num)
            ]
        )
        self.adaptive_softmax = None
        if self.super_embed_dim != self.output_embed_dim and not args.tie_adaptive_weights:
            self.project_out_dim = Linear(self.super_embed_dim, self.output_embed_dim, bias=False)
        else:
            self.project_out_dim = None
        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(vocab_size, self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim**-0.5)
        if args.decoder_normalize_before and not getattr(args, "no_decoder_final_norm", False):
            self.layer_norm = LayerNormSuper(self.super_embed_dim)
        else:
            self.layer_norm = None
        self.get_attn = args.get_attn
        self.vocab_original_scaling = args.vocab_original_scaling
        self._future_mask: torch.Tensor | None = None

    def set_sample_config(self, config: dict[str, dict[str, Any]]) -> None:
        """Set sampled decoder configuration.

        Parameters
        ----------
        config
            Nested HAT sample config.
        """
        self.sample_embed_dim = config["decoder"]["decoder_embed_dim"]
        self.sample_encoder_embed_dim = config["encoder"]["encoder_embed_dim"]
        self.sample_ffn_embed_dim = config["decoder"]["decoder_ffn_embed_dim"]
        self.sample_self_attention_heads = config["decoder"]["decoder_self_attention_heads"]
        self.sample_ende_attention_heads = config["decoder"]["decoder_ende_attention_heads"]
        self.sample_arbitrary_ende_attn = config["decoder"]["decoder_arbitrary_ende_attn"]
        self.sample_layer_num = config["decoder"]["decoder_layer_num"]
        self.sample_dropout = calc_dropout(
            self.super_dropout, self.sample_embed_dim, self.super_embed_dim
        )
        self.sample_activation_dropout = calc_dropout(
            self.super_activation_dropout,
            self.sample_embed_dim,
            self.super_embed_dim,
        )
        self.sample_embed_scale = (
            math.sqrt(self.sample_embed_dim)
            if not self.vocab_original_scaling
            else self.super_embed_scale
        )
        self.embed_tokens.set_sample_config(sample_embed_dim=self.sample_embed_dim, part="decoder")
        if self.layer_norm is not None:
            self.layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        for i, layer in enumerate(self.layers):
            if i < self.sample_layer_num:
                layer.set_sample_config(
                    is_identity_layer=False,
                    sample_embed_dim=self.sample_embed_dim,
                    sample_encoder_embed_dim=self.sample_encoder_embed_dim,
                    sample_ffn_embed_dim_this_layer=self.sample_ffn_embed_dim[i],
                    sample_self_attention_heads_this_layer=self.sample_self_attention_heads[i],
                    sample_ende_attention_heads_this_layer=self.sample_ende_attention_heads[i],
                    sample_dropout=self.sample_dropout,
                    sample_activation_dropout=self.sample_activation_dropout,
                )
            else:
                layer.set_sample_config(is_identity_layer=True)

    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: dict[str, Any] | None = None,
        incremental_state: dict[str, Any] | None = None,
        **unused: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply the decoder and output projection.

        Parameters
        ----------
        prev_output_tokens
            Previous decoder tokens of shape ``batch x time``.
        encoder_out
            Optional encoder output dictionary.
        incremental_state
            Optional incremental cache.
        **unused
            Unused fairseq compatibility arguments.

        Returns
        -------
        tuple[torch.Tensor, dict[str, Any]]
            Decoder logits and extra outputs.
        """
        del unused
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state)
        x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: dict[str, Any] | None = None,
        incremental_state: dict[str, Any] | None = None,
        **unused: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Extract decoder features.

        Parameters
        ----------
        prev_output_tokens
            Previous decoder tokens of shape ``batch x time``.
        encoder_out
            Optional encoder output dictionary.
        incremental_state
            Optional incremental cache.
        **unused
            Unused fairseq compatibility arguments.

        Returns
        -------
        tuple[torch.Tensor, dict[str, Any]]
            Decoder features and extra outputs.
        """
        del unused
        positions = (
            self.embed_positions(prev_output_tokens, incremental_state=incremental_state)
            if self.embed_positions is not None
            else None
        )
        if positions is not None:
            positions = positions[..., : self.sample_embed_dim]
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        x = self.sample_embed_scale * self.embed_tokens(prev_output_tokens, part="decoder")
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = x.transpose(0, 1)
        attn = None
        attns = []
        inner_states = [x]
        for i, layer in enumerate(self.layers):
            encoder_out_feed = None
            encoder_padding_mask_feed = None
            if encoder_out is not None:
                if i >= self.sample_layer_num or self.sample_arbitrary_ende_attn[i] == -1:
                    encoder_out_feed = encoder_out["encoder_out"]
                elif self.sample_arbitrary_ende_attn[i] == 1:
                    encoder_out_feed = torch.cat(
                        [encoder_out["encoder_out"], encoder_out["encoder_out_all"][-2]],
                        dim=0,
                    )
                elif self.sample_arbitrary_ende_attn[i] == 2:
                    encoder_out_feed = torch.cat(
                        [
                            encoder_out["encoder_out"],
                            encoder_out["encoder_out_all"][-2],
                            encoder_out["encoder_out_all"][-3],
                        ],
                        dim=0,
                    )
                else:
                    raise NotImplementedError("arbitrary_ende_attn should in [-1, 1, 2]")
                if encoder_out["encoder_padding_mask"] is not None:
                    if i >= self.sample_layer_num or self.sample_arbitrary_ende_attn[i] == -1:
                        encoder_padding_mask_feed = encoder_out["encoder_padding_mask"]
                    elif self.sample_arbitrary_ende_attn[i] == 1:
                        encoder_padding_mask_feed = torch.cat(
                            [
                                encoder_out["encoder_padding_mask"],
                                encoder_out["encoder_padding_mask"],
                            ],
                            dim=1,
                        )
                    elif self.sample_arbitrary_ende_attn[i] == 2:
                        encoder_padding_mask_feed = torch.cat(
                            [
                                encoder_out["encoder_padding_mask"],
                                encoder_out["encoder_padding_mask"],
                                encoder_out["encoder_padding_mask"],
                            ],
                            dim=1,
                        )
                    else:
                        raise NotImplementedError("arbitrary_ende_attn should in [-1, 1, 2]")
            x, attn = layer(
                x,
                encoder_out_feed,
                encoder_padding_mask_feed,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)
            attns.append(attn)
        if self.layer_norm:
            x = self.layer_norm(x)
        x = x.transpose(0, 1)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        if not self.get_attn:
            attns = attns[-1]
        return x, {"attn": attns, "inner_states": inner_states}

    def output_layer(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Project decoder features to vocabulary logits.

        Parameters
        ----------
        features
            Decoder features.
        **kwargs
            Unused fairseq compatibility arguments.

        Returns
        -------
        torch.Tensor
            Vocabulary logits or adaptive-softmax features.
        """
        del kwargs
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.sampled_weight("decoder"))
            return F.linear(features, self.embed_out[:, : self.sample_embed_dim])
        return features

    def max_positions(self) -> int:
        """Return maximum supported target length.

        Returns
        -------
        int
            Maximum positions.
        """
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a causal future mask.

        Parameters
        ----------
        tensor
            Decoder state tensor whose leading dimension defines mask size.

        Returns
        -------
        torch.Tensor
            Causal mask.
        """
        dim = tensor.size(0)
        if (
            self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class TransformerSuperModel(nn.Module):
    """Thin encoder-decoder wrapper matching fairseq's forward wiring."""

    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder) -> None:
        """Initialize the model.

        Parameters
        ----------
        encoder
            Encoder module.
        decoder
            Decoder module.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_output_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run encoder then decoder.

        Parameters
        ----------
        src_tokens
            Source tokens.
        src_lengths
            Source lengths.
        prev_output_tokens
            Previous decoder tokens.

        Returns
        -------
        tuple[torch.Tensor, dict[str, Any]]
            Decoder output.
        """
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        return decoder_out

    def set_sample_config(self, config: dict[str, dict[str, Any]]) -> None:
        """Set sampled encoder and decoder configs.

        Parameters
        ----------
        config
            Nested HAT sample config.
        """
        self.encoder.set_sample_config(config)
        self.decoder.set_sample_config(config)


def calc_dropout(dropout: float, sample_embed_dim: int, super_embed_dim: int) -> float:
    """Scale dropout by sampled embedding width, as in HAT.

    Parameters
    ----------
    dropout
        Supernet dropout.
    sample_embed_dim
        Sampled embedding dimension.
    super_embed_dim
        Supernet embedding dimension.

    Returns
    -------
    float
        Sampled dropout.
    """
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


def Embedding(num_embeddings: int, embedding_dim: int, padding_idx: int) -> EmbeddingSuper:
    """Construct HAT's elastic embedding helper.

    Parameters
    ----------
    num_embeddings
        Vocabulary size.
    embedding_dim
        Maximum embedding dimension.
    padding_idx
        Padding index.

    Returns
    -------
    EmbeddingSuper
        Elastic embedding.
    """
    return EmbeddingSuper(num_embeddings, embedding_dim, padding_idx=padding_idx)


def Linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    uniform_: Any = None,
    non_linear: str = "linear",
) -> nn.Linear:
    """Construct a Xavier-initialized linear helper.

    Parameters
    ----------
    in_features
        Input dimension.
    out_features
        Output dimension.
    bias
        Whether to include bias.
    uniform_
        Optional fairseq initializer hook.
    non_linear
        Nonlinearity name passed to the optional initializer.

    Returns
    -------
    nn.Linear
        Initialized linear layer.
    """
    m = nn.Linear(in_features, out_features, bias)
    if uniform_ is None:
        nn.init.xavier_uniform_(m.weight)
    else:
        uniform_(m.weight, non_linear=non_linear)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def build_hat() -> TransformerSuperModel:
    """Build a tiny sampled HAT SuperTransformer for TorchLens tracing.

    Returns
    -------
    TransformerSuperModel
        Tiny HAT model in eval mode.
    """
    torch.manual_seed(0)
    vocab_size = 64
    padding_idx = 0
    args = SuperTransformerConfig(
        encoder_embed_dim=32,
        encoder_ffn_embed_dim=64,
        encoder_layers=2,
        encoder_attention_heads=4,
        decoder_embed_dim=32,
        decoder_ffn_embed_dim=64,
        decoder_layers=2,
        decoder_attention_heads=4,
        decoder_output_dim=32,
        qkv_dim=32,
        max_source_positions=32,
        max_target_positions=32,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        share_decoder_input_output_embed=True,
    )
    encoder_embed_tokens = Embedding(vocab_size, args.encoder_embed_dim, padding_idx)
    decoder_embed_tokens = Embedding(vocab_size, args.decoder_embed_dim, padding_idx)
    encoder = TransformerEncoder(args, vocab_size, padding_idx, encoder_embed_tokens)
    decoder = TransformerDecoder(args, vocab_size, padding_idx, decoder_embed_tokens)
    model = TransformerSuperModel(encoder, decoder)
    sample_config = {
        "encoder": {
            "encoder_embed_dim": 32,
            "encoder_ffn_embed_dim": [64, 64],
            "encoder_layer_num": 2,
            "encoder_self_attention_heads": [4, 4],
        },
        "decoder": {
            "decoder_embed_dim": 32,
            "decoder_ffn_embed_dim": [64, 64],
            "decoder_layer_num": 2,
            "decoder_self_attention_heads": [4, 4],
            "decoder_ende_attention_heads": [4, 4],
            "decoder_arbitrary_ende_attn": [-1, -1],
        },
    }
    model.set_sample_config(sample_config)
    model.eval()
    return model


def example_input_hat() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a multi-input batch for HAT tracing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Source tokens, source lengths, and previous decoder tokens.
    """
    src_tokens = torch.tensor(
        [
            [5, 7, 9, 11, 13, 15, 0, 0],
            [4, 6, 8, 10, 12, 14, 16, 18],
        ],
        dtype=torch.long,
    )
    src_lengths = torch.tensor([6, 8], dtype=torch.long)
    prev_output_tokens = torch.tensor(
        [
            [2, 21, 22, 23, 24, 25, 0, 0],
            [2, 31, 32, 33, 34, 35, 36, 37],
        ],
        dtype=torch.long,
    )
    return (src_tokens, src_lengths, prev_output_tokens)


MENAGERIE_ENTRIES = [
    ("Hardware-Aware Transformer (HAT)", "build_hat", "example_input_hat", 2020, MENAGERIE_ZOO),
]
