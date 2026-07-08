# SOURCE: vendored from reginabarzilaygroup/Sybil @ main
# https://raw.githubusercontent.com/reginabarzilaygroup/Sybil/main/sybil/models/sybil.py
# https://raw.githubusercontent.com/reginabarzilaygroup/Sybil/main/sybil/models/pooling_layer.py
# https://raw.githubusercontent.com/reginabarzilaygroup/Sybil/main/sybil/models/cumulative_probability_layer.py
#
# "Sybil: A Validated Deep Learning Model to Predict Future Lung Cancer Risk From a Single
# Low-Dose Chest Computed Tomography" (Mikhael et al., J Clin Oncol 2023). SybilNet =
# torchvision 3D-ResNet-18 (`r3d_18`) video encoder over a CT volume stack, followed by a
# multi-branch attention-pooling head (MultiAttentionPool: per-image attention pool,
# per-volume attention pool, per-frame max pool, conv1d attention pool, global max pool --
# fused via linear layers) and a cumulative-probability (discrete-time hazard) output layer
# that predicts lung-cancer risk over `max_followup` years. Classes copied verbatim from the
# real repo; only trims:
#   - `SybilNet.load` (checkpoint-file loading classmethod, not part of the forward
#     architecture) is dropped.
#   - `RiskFactorPredictor` (auxiliary subclass predicting NLST risk-factor fields, used only
#     for the optional risk-factor auxiliary loss) is dropped; it depends on
#     `sybil.datasets.nlst_risk_factors.NLSTRiskFactorVectorizer`, which reads an external NLST
#     metadata CSV and is unrelated to the core imaging architecture.
#   - `r3d_18(pretrained=True)` is instantiated with `pretrained=False` in build_sybil() (we
#     only need the real architecture at random init; downloading ImageNet/Kinetics weights is
#     a network dependency, not an architectural one).
import torch
import torch.nn as nn
import torchvision

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from sybil/models/cumulative_probability_layer.py ---
class Cumulative_Probability_Layer(nn.Module):
    def __init__(self, num_features, args, max_followup):
        super(Cumulative_Probability_Layer, self).__init__()
        self.args = args
        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        hazards = self.hazards(x)
        B, T = hazards.size()  # hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T)  # expanded_hazards is (B,T, T)
        masked_hazards = expanded_hazards * self.upper_triagular_mask  # masked_hazards now (B,T, T)
        base_hazard = self.base_hazard_fc(x)
        cum_prob = torch.sum(masked_hazards, dim=1) + base_hazard
        return cum_prob


# --- vendored from sybil/models/pooling_layer.py ---
class GlobalMaxPool(nn.Module):
    """
    Pool to obtain the maximum value for each channel
    """

    def __init__(self):
        super(GlobalMaxPool, self).__init__()

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict. output['hidden'] is (B, C)
        """
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        hidden, _ = torch.max(x, dim=-1)
        return {"hidden": hidden}


class PerFrameMaxPool(nn.Module):
    """
    Pool to obtain the maximum value for each slice in 3D input
    """

    def __init__(self):
        super(PerFrameMaxPool, self).__init__()

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict.
                + output['multi_image_hidden'] is (B, C, T)
        """
        assert len(x.shape) == 5
        output = {}
        spatially_flat_size = (*x.size()[:3], -1)
        x = x.view(spatially_flat_size)
        output["multi_image_hidden"], _ = torch.max(x, dim=-1)
        return output


class Simple_AttentionPool(nn.Module):
    """
    Pool to learn an attention over the slices
    """

    def __init__(self, **kwargs):
        super(Simple_AttentionPool, self).__init__()

        self.attention_fc = nn.Linear(kwargs["num_chan"], 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, N)
        returns:
            - output: dict
                + output['volume_attention']: tensor (B, N)
                + output['hidden']: tensor (B, C)
        """
        output = {}
        B = x.shape[0]
        spatially_flat_size = (*x.size()[:2], -1)  # B, C, N

        x = x.view(spatially_flat_size)
        attention_scores = self.attention_fc(x.transpose(1, 2))  # B, N, 1

        output["volume_attention"] = self.logsoftmax(attention_scores.transpose(1, 2)).view(B, -1)
        attention_scores = self.softmax(attention_scores.transpose(1, 2))  # B, 1, N

        x = x * attention_scores  # B, C, N
        output["hidden"] = torch.sum(x, dim=-1)
        return output


class Simple_AttentionPool_MultiImg(nn.Module):
    """
    Pool to learn an attention over the slices and the volume
    """

    def __init__(self, **kwargs):
        super(Simple_AttentionPool_MultiImg, self).__init__()

        self.attention_fc = nn.Linear(kwargs["num_chan"], 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict
                + output['image_attention']: tensor (B, T, W*H)
                + output['multi_image_hidden']: tensor (B, C, T)
                + output['hidden']: tensor (B, T*C)
        """
        output = {}
        B, C, T, W, H = x.size()
        x = x.permute([0, 2, 1, 3, 4])
        x = x.contiguous().view(B * T, C, W * H)
        attention_scores = self.attention_fc(x.transpose(1, 2))  # BT, WH , 1

        output["image_attention"] = self.logsoftmax(attention_scores.transpose(1, 2)).view(B, T, -1)
        attention_scores = self.softmax(attention_scores.transpose(1, 2))  # BT, 1, WH

        x = x * attention_scores  # BT, C, WH
        x = torch.sum(x, dim=-1)
        output["multi_image_hidden"] = x.view(B, T, C).permute([0, 2, 1]).contiguous()
        output["hidden"] = x.view(B, T * C)
        return output


class Conv1d_AttnPool(nn.Module):
    """
    Pool to learn an attention over the slices after convolution
    """

    def __init__(self, **kwargs):
        super(Conv1d_AttnPool, self).__init__()
        self.conv1d = nn.Conv1d(
            kwargs["num_chan"],
            kwargs["num_chan"],
            kernel_size=kwargs["conv_pool_kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["conv_pool_kernel_size"] // 2,
            bias=False,
        )
        self.aggregate = Simple_AttentionPool(**kwargs)

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, T)
        returns:
            - output: dict
                + output['attention_scores']: tensor (B, C)
                + output['hidden']: tensor (B, C)
        """
        # X: B, C, N
        x = self.conv1d(x)  # B, C, N'
        return self.aggregate(x)


class MultiAttentionPool(nn.Module):
    def __init__(self):
        super(MultiAttentionPool, self).__init__()
        params = {"num_chan": 512, "conv_pool_kernel_size": 11, "stride": 1}
        self.image_pool1 = Simple_AttentionPool_MultiImg(**params)
        self.volume_pool1 = Simple_AttentionPool(**params)

        self.image_pool2 = PerFrameMaxPool()
        self.volume_pool2 = Conv1d_AttnPool(**params)

        self.global_max_pool = GlobalMaxPool()

        self.multi_img_hidden_fc = nn.Linear(2 * 512, 512)
        self.hidden_fc = nn.Linear(3 * 512, 512)

    def forward(self, x):
        # X dim: B, C, T, W, H
        output = {}

        image_pool_out1 = self.image_pool1(
            x
        )  # contains keys: "multi_image_hidden", "image_attention"
        volume_pool_out1 = self.volume_pool1(
            image_pool_out1["multi_image_hidden"]
        )  # contains keys: "hidden", "volume_attention"

        image_pool_out2 = self.image_pool2(x)  # contains keys: "multi_image_hidden"
        volume_pool_out2 = self.volume_pool2(
            image_pool_out2["multi_image_hidden"]
        )  # contains keys: "hidden", "volume_attention"

        for pool_out, num in [
            (image_pool_out1, 1),
            (volume_pool_out1, 1),
            (image_pool_out2, 2),
            (volume_pool_out2, 2),
        ]:
            for key, val in pool_out.items():
                output["{}_{}".format(key, num)] = val

        maxpool_out = self.global_max_pool(x)
        output["maxpool_hidden"] = maxpool_out["hidden"]

        multi_image_hidden = torch.cat(
            [image_pool_out1["multi_image_hidden"], image_pool_out2["multi_image_hidden"]], dim=-2
        )
        output["multi_image_hidden"] = (
            self.multi_img_hidden_fc(multi_image_hidden.permute([0, 2, 1]).contiguous())
            .permute([0, 2, 1])
            .contiguous()
        )

        hidden = torch.cat(
            [volume_pool_out1["hidden"], volume_pool_out2["hidden"], output["maxpool_hidden"]],
            dim=-1,
        )
        output["hidden"] = self.hidden_fc(hidden)

        return output


# --- vendored from sybil/models/sybil.py (load(), RiskFactorPredictor dropped; see header) ---
class SybilNet(nn.Module):
    def __init__(self, args):
        super(SybilNet, self).__init__()

        self.hidden_dim = 512

        encoder = torchvision.models.video.r3d_18(pretrained=False)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])

        self.pool = MultiAttentionPool()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)

        self.prob_of_failure_layer = Cumulative_Probability_Layer(
            self.hidden_dim, args, max_followup=args.max_followup
        )

    def forward(self, x, batch=None):
        output = {}
        x = self.image_encoder(x)
        pool_output = self.aggregate_and_classify(x)
        output["activ"] = x
        output.update(pool_output)
        output["prob"] = pool_output["logit"].sigmoid()

        return output

    def aggregate_and_classify(self, x):
        pool_output = self.pool(x)

        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"])

        return pool_output


class _SybilArgs:
    """Minimal stand-in for the argparse.Namespace SybilNet expects (dropout, max_followup)."""

    def __init__(self, dropout=0.25, max_followup=6):
        self.dropout = dropout
        self.max_followup = max_followup


def build_sybil():
    torch.manual_seed(0)
    args = _SybilArgs(dropout=0.0, max_followup=6)
    model = SybilNet(args)
    model.eval()
    return model


def example_input_sybil():
    torch.manual_seed(0)
    # (B, C, T, W, H): 3-channel CT volume stack, small spatial/temporal extent for tracing.
    return torch.randn(1, 3, 16, 32, 32)


MENAGERIE_ENTRIES = [
    ("Sybil", "build_sybil", "example_input_sybil", 2023, "vendored"),
]
