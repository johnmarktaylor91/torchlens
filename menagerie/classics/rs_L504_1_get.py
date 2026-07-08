# SOURCE: vendored from GET-Foundation/get_model @ master
#   get_model/model/modules.py  (BaseConfig, BaseModule, RegionEmbedConfig, RegionEmbed,
#                                 ExpressionHeadConfig, ExpressionHead)
#   get_model/model/transformer.py  (Attention, Mlp, Block, GETTransformer)
#   get_model/model/model.py  (LossConfig, MetricsConfig, EncoderConfig, GETLoss,
#                               RegressionMetrics, BaseGETModelConfig, BaseGETModel,
#                               GETRegionFinetuneModelConfig, GETRegionFinetune)
#
# GET (General Expression Transformer): a modular framework for multimodal cross-cell-type
# transcriptional regulation modeling (He et al., Nature 2024, "A foundation model of
# transcription across human cell types"). GETRegionFinetune is the region-level expression
# fine-tuning model: per-region motif-accessibility features are linearly embedded, passed
# through a ViT-style transformer encoder (GETTransformer/Block/Attention/Mlp, identical in
# structure to timm's VisionTransformer blocks), and decoded per-region by a linear expression
# head with a Softplus nonlinearity. Copied verbatim from the real repo's classes, with only
# the cross-file `from get_model.model.X import Y` edges collapsed into this single file (the
# unused MotifScanner/ATACHead/ContactMapHead/HiCHead heads, the flash-attn and axial-attention
# optional variants, and the pretrain/HiC/contact-map model subclasses are dropped -- none of
# them are reached by GETRegionFinetune's forward pass). GETLoss/RegressionMetrics are kept
# so BaseGETModel.__init__ constructs identically to the original; both are unused by forward().
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import trunc_normal_ as trunc_normal_init

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# get_model/model/modules.py
# ---------------------------------------------------------------------------
@dataclass
class BaseConfig:
    """Dummy configuration class with arbitrary generated values."""

    _target_: str = "get_model.model.modules.BaseConfig"
    freezed: bool | str = False


class BaseModule(nn.Module):
    """Base model class with methods to generate dummy data and forward function."""

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg

    def generate_dummy_data(self, batch_size=1):
        raise NotImplementedError("Dummy data generation must be implemented in subclasses.")

    def forward(self, x):
        raise NotImplementedError("Forward function must be implemented in subclasses.")

    def test(self, device="cpu"):
        x = self.generate_dummy_data()
        self.to(device)
        return self(**x)

    def freeze_parameters(self):
        if self.cfg.freezed:
            if self.cfg.freezed == True:  # noqa: E712
                for param in self.parameters():
                    param.requires_grad = False
            else:
                for name, param in self.named_parameters():
                    if self.cfg.freezed in name:
                        param.requires_grad = False


@dataclass
class RegionEmbedConfig(BaseConfig):
    """Configuration class for the region embedding module."""

    _target_: str = "get_model.model.modules.RegionEmbedConfig"
    num_features: int = 800
    embed_dim: int = 768


class RegionEmbed(BaseModule):
    """A simple region embedding transforming motif features to region embeddings."""

    def __init__(self, cfg: RegionEmbedConfig):
        super().__init__(cfg)
        self.embed = nn.Linear(cfg.num_features, cfg.embed_dim)

    def forward(self, x, **kwargs):
        x = self.embed(x)
        return x

    def generate_dummy_data(self, batch_size=1):
        return torch.rand(batch_size, 5, self.cfg.num_features)


@dataclass
class ExpressionHeadConfig(BaseConfig):
    """Configuration class for the expression head."""

    embed_dim: int = 768
    output_dim: int = 2
    use_atac: bool = False


class ExpressionHead(BaseModule):
    """Expression head"""

    def __init__(self, cfg: ExpressionHeadConfig):
        super().__init__(cfg)
        self.use_atac = cfg.use_atac
        if self.use_atac:
            self.head = nn.Linear(cfg.embed_dim + 1, cfg.output_dim)
        else:
            self.head = nn.Linear(cfg.embed_dim, cfg.output_dim)

        trunc_normal_init(self.head.weight, std=0.02)

        self.head.weight.data.mul_(0.001)
        self.head.bias.data.mul_(0.001)

    def forward(self, x, atac=None):
        if self.use_atac:
            x = torch.cat([x, atac], dim=-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# get_model/model/transformer.py
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask=None, attention_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if attention_bias is not None:
            attn = attn + attention_bias
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0,
        attn_drop=0,
        drop_path=0.1,
        init_values=0.001,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        flash_attn=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attention_mask=None, attention_bias=None):
        if self.gamma_1 is None:
            x_attn, attn = self.attn(
                self.norm1(x),
                attention_mask=attention_mask,
                attention_bias=attention_bias,
            )
            x = x + self.drop_path(x_attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_attn, attn = self.attn(
                self.norm1(x),
                attention_mask=attention_mask,
                attention_bias=attention_bias,
            )
            x = x + self.drop_path(self.gamma_1 * x_attn)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn


class GETTransformer(nn.Module):
    """A transformer module for GET model."""

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        num_layers=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0,
        use_mean_pooling=False,
        flash_attn=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    flash_attn=flash_attn,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None, return_attns=False, bias=None):
        attn_output = [] if return_attns else None
        for blk in self.blocks:
            x, attn = blk(x, mask, bias)
            if return_attns:
                attn_output.append(attn)
        x = self.norm(x)
        if self.fc_norm is not None:
            x = self.fc_norm(x.mean(1))
        return x, attn_output


# ---------------------------------------------------------------------------
# get_model/model/model.py
# ---------------------------------------------------------------------------
@dataclass
class LossConfig:
    components: dict = MISSING
    weights: dict = MISSING


@dataclass
class MetricsConfig:
    components: dict = MISSING


@dataclass
class EncoderConfig:
    num_heads: int = MISSING
    embed_dim: int = MISSING
    num_layers: int = MISSING
    drop_path_rate: float = MISSING
    drop_rate: float = MISSING
    attn_drop_rate: float = MISSING
    use_mean_pooling: bool = False
    flash_attn: bool = MISSING


class GETLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super(GETLoss, self).__init__()
        self.cfg = cfg
        if isinstance(cfg, DictConfig):
            self.losses = {
                name: (component, cfg.weights[f"{name}"])
                for name, component in cfg.components.items()
            }
        else:
            self.losses = instantiate(cfg)

    def forward(self, pred, obs):
        if isinstance(self.losses, dict):
            return {
                f"{name}_loss": loss_fn(pred[name], obs[name]) * weight
                for name, (loss_fn, weight) in self.losses.items()
            }
        elif isinstance(self.losses, nn.Module):
            return self.losses(pred, obs)

    def freeze_component(self, component_name):
        if component_name in self.losses:
            self.losses[component_name] = (self.losses[component_name][0], 0)
        else:
            raise ValueError(f"Component '{component_name}' not found in the loss function.")


class RegressionMetrics(nn.Module):
    def __init__(self, _cfg_: MetricsConfig):
        super(RegressionMetrics, self).__init__()
        self.cfg = _cfg_
        self.metrics = nn.ModuleDict(
            {
                target: nn.ModuleDict(
                    {metric_name: self._get_metric(metric_name) for metric_name in metric_names}
                )
                for target, metric_names in _cfg_.components.items()
            }
        )

    def _get_metric(self, metric_name):
        if metric_name == "pearson":
            return torchmetrics.PearsonCorrCoef()
        elif metric_name == "spearman":
            return torchmetrics.SpearmanCorrCoef()
        elif metric_name == "mse":
            return torchmetrics.MeanSquaredError()
        elif metric_name == "r2":
            return torchmetrics.R2Score()
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    def forward(self, _pred_, _obs_):
        result = {
            target: {
                metric_name: metric(_pred_[target].reshape(-1, 1), _obs_[target].reshape(-1, 1))
                for metric_name, metric in target_metrics.items()
            }
            for target, target_metrics in self.metrics.items()
        }
        result = {
            f"{target}_{metric_name}": result[target][metric_name]
            for target in result
            for metric_name in result[target]
        }
        return result


@dataclass
class BaseGETModelConfig:
    freezed: bool | str = False
    loss: LossConfig = field(default_factory=LossConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


class BaseGETModel(BaseModule):
    def __init__(self, cfg: BaseConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.loss = GETLoss(cfg.loss)
        self.metrics = RegressionMetrics(cfg.metrics)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_input(self, batch):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

    def before_loss(self, output, batch):
        raise NotImplementedError

    def after_loss(self, loss):
        if isinstance(loss, dict):
            return sum(loss.values())
        else:
            return loss

    def generate_dummy_data(self):
        raise NotImplementedError

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}


@dataclass
class GETRegionFinetuneModelConfig(BaseGETModelConfig):
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_exp: ExpressionHeadConfig = field(default_factory=ExpressionHeadConfig)
    use_atac: bool = False


class GETRegionFinetune(BaseGETModel):
    """The GET (General Expression Transformer) region-level expression fine-tuning model."""

    def __init__(self, cfg: GETRegionFinetuneModelConfig):
        super().__init__(cfg)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):
        return {"region_motif": batch["region_motif"]}

    def forward(self, region_motif):
        x = self.region_embed(region_motif)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, _ = self.encoder(x)
        x = x[:, 1:]
        exp = nn.Softplus()(self.head_exp(x))
        return exp

    def before_loss(self, output, batch):
        pred = {"exp": output}
        obs = {"exp": batch["exp_label"]}
        return pred, obs

    def generate_dummy_data(self):
        B, R, M = 2, 900, 283
        return {"region_motif": torch.randn(B, R, M).float().abs()}


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------
def _get_config():
    num_regions, num_motif, embed_dim, num_layers, num_heads = 20, 32, 32, 2, 4
    cfg_dict = {
        "freezed": False,
        "num_regions": num_regions,
        "num_motif": num_motif,
        "embed_dim": embed_dim,
        "region_embed": {"num_features": num_motif, "embed_dim": embed_dim},
        "encoder": {
            "num_heads": num_heads,
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "drop_path_rate": 0.1,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "use_mean_pooling": False,
            "flash_attn": False,
        },
        "head_exp": {"embed_dim": embed_dim, "output_dim": 2, "use_atac": False},
        "loss": {"components": {}, "weights": {}},
        "metrics": {"components": {}},
        "use_atac": False,
    }
    return OmegaConf.create(cfg_dict), num_regions, num_motif


def build_get_region_finetune():
    cfg, _, _ = _get_config()
    model = GETRegionFinetune(cfg)
    model.eval()
    return model


def example_input_get_region_finetune():
    _, num_regions, num_motif = _get_config()
    return torch.randn(2, num_regions, num_motif).float().abs()


MENAGERIE_ENTRIES = [
    (
        "GET (General Expression Transformer)",
        build_get_region_finetune,
        example_input_get_region_finetune,
        2024,
        MENAGERIE_ZOO,
    ),
]
