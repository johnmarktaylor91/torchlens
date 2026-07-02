# SOURCE: vendored from zju3dv/object_nerf @ 6b7e2f634671d3a94006475c135ce74c5747bb2a
# https://github.com/zju3dv/object_nerf -- "Learning Object-Compositional Neural Radiance
# Field for Editable Scene Rendering" (Yang, Chen, Chen, Bao, Zhang, ICCV 2021 oral),
# official ZJU3DV repo. `ObjectNeRF` (models/nerf_model.py) is a dual-branch NeRF: a
# standard scene-level MLP branch (xyz/skip-connection MLP -> sigma, then a
# direction-conditioned MLP -> RGB) plus a parallel object/instance-level branch that
# additionally conditions on a per-object learned instance code (models/code_library.py's
# `CodeLibrary.embedding_instance`, an `nn.Embedding` over object IDs) to predict
# per-object sigma/RGB for editable, object-compositional scene rendering. `ObjectNeRF` is
# transcribed verbatim from models/nerf_model.py; `Embedding` (positional encoding for xyz
# and view direction) is transcribed verbatim from models/embedding_helper.py. The staging
# `build_objectnerf`/`example_input_objectnerf` helpers construct `ObjectNeRF` with
# `use_voxel_embedding=False` (the repo's own default fallback path when no voxel grid is
# supplied) and precompute positional-encoded xyz/dir to match the shapes `ObjectNeRF`'s
# default `forward` (the scene-level branch, invoked by `nn.Module.__call__`) expects,
# mirroring how render_tools/multi_rendering.py assembles the per-point input dict at
# inference time. The instance/object-compositional branch (`forward_instance`, the paper's
# additional architectural contribution over vanilla NeRF) shares identical MLP topology
# and is included verbatim above; it is not the entry point `tl.trace` walks by default
# since capture follows the module's `forward`, but every layer it is built from
# (`initialize_object_branch`) is still constructed and present in the traced module graph.
import torch
import torch.nn.functional as F  # noqa: F401 -- kept for parity with nerf_model.py imports
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from models/embedding_helper.py ----
class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


# ---- verbatim from models/nerf_model.py ----
class ObjectNeRF(nn.Module):
    def __init__(
        self,
        model_config,
    ):
        super(ObjectNeRF, self).__init__()
        self.model_config = model_config
        self.use_voxel_embedding = self.model_config.use_voxel_embedding
        # initialize neural model with config
        self.initialize_scene_branch(model_config)
        self.initialize_object_branch(model_config)

    def initialize_scene_branch(self, model_config):
        self.D = model_config["D"]
        self.W = model_config["W"]
        self.N_freq_xyz = model_config["N_freq_xyz"]
        self.N_freq_dir = model_config["N_freq_dir"]
        self.skips = model_config["skips"]
        # embedding size for voxel representation
        if self.use_voxel_embedding:
            self.N_scn_voxel_size = model_config.get("N_scn_voxel_size", 0)
            self.N_freq_voxel = model_config["N_freq_voxel"]
            voxel_emb_size = self.N_scn_voxel_size + self.N_scn_voxel_size * self.N_freq_voxel * 2
        else:
            voxel_emb_size = 0
        # embedding size for NeRF xyz
        xyz_emb_size = 3 + 3 * self.N_freq_xyz * 2
        self.in_channels_xyz = xyz_emb_size + voxel_emb_size
        self.in_channels_dir = 3 + 3 * self.N_freq_dir * 2

        self.activation = nn.LeakyReLU(inplace=True)

        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"xyz_encoding_{i + 1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        self.rgb = nn.Sequential(nn.Linear(self.W // 2, 3), nn.Sigmoid())
        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(self.W + self.in_channels_dir, self.W // 2), self.activation
        )

    def initialize_object_branch(self, model_config):
        # instance encoding
        N_obj_code_length = model_config["N_obj_code_length"]
        if self.use_voxel_embedding:
            N_obj_voxel_size = model_config.get("N_obj_voxel_size", 0)
            inst_voxel_emb_size = N_obj_voxel_size + N_obj_voxel_size * self.N_freq_voxel * 2
        else:
            inst_voxel_emb_size = 0
        self.inst_channel_in = self.in_channels_xyz + N_obj_code_length + inst_voxel_emb_size
        self.inst_D = model_config["inst_D"]
        self.inst_W = model_config["inst_W"]
        self.inst_skips = model_config["inst_skips"]

        for i in range(self.inst_D):
            if i == 0:
                layer = nn.Linear(self.inst_channel_in, self.inst_W)
            elif i in self.inst_skips:
                layer = nn.Linear(self.inst_W + self.inst_channel_in, self.inst_W)
            else:
                layer = nn.Linear(self.inst_W, self.inst_W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"instance_encoding_{i + 1}", layer)
        self.instance_encoding_final = nn.Sequential(
            nn.Linear(self.inst_W, self.inst_W),
        )
        self.instance_sigma = nn.Linear(self.inst_W, 1)

        self.inst_dir_encoding = nn.Sequential(
            nn.Linear(self.inst_W + self.in_channels_dir, self.inst_W // 2),
            self.activation,
        )
        self.inst_rgb = nn.Sequential(nn.Linear(self.inst_W // 2, 3), nn.Sigmoid())

    def forward(self, inputs, sigma_only=False):
        output_dict = {}
        input_xyz = inputs["emb_xyz"]
        input_dir = inputs.get("emb_dir", None)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        sigma = self.sigma(xyz_)
        output_dict["sigma"] = sigma

        if sigma_only:
            return output_dict

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        output_dict["rgb"] = rgb

        return output_dict

    def forward_instance(self, inputs, sigma_only=False):
        output_dict = {}
        emb_xyz = inputs["emb_xyz"]
        input_dir = inputs.get("emb_dir", None)
        obj_code = inputs["obj_code"]
        if self.use_voxel_embedding:
            obj_voxel = inputs["obj_voxel"]
            input_x = torch.cat([emb_xyz, obj_voxel, obj_code], -1)
        else:
            input_x = torch.cat([emb_xyz, obj_code], -1)

        x_ = input_x

        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([input_x, x_], -1)
            x_ = getattr(self, f"instance_encoding_{i + 1}")(x_)
        inst_sigma = self.instance_sigma(x_)
        output_dict["inst_sigma"] = inst_sigma

        if sigma_only:
            return output_dict

        x_final = self.instance_encoding_final(x_)
        dir_encoding_input = torch.cat([x_final, input_dir], -1)
        dir_encoding = self.inst_dir_encoding(dir_encoding_input)
        rgb = self.inst_rgb(dir_encoding)
        output_dict["inst_rgb"] = rgb

        return output_dict


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
class _ObjectNeRFConfig(dict):
    """Tiny attribute-access shim so `model_config.use_voxel_embedding` (used in
    `ObjectNeRF.__init__`) works alongside the dict-style `model_config["D"]` /
    `model_config.get(...)` accesses used elsewhere in the real `__init__` body --
    the real training config in the source repo is an addict/omegaconf-style object
    that supports both accessor styles simultaneously."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e


def build_objectnerf():
    torch.manual_seed(0)
    model_config = _ObjectNeRFConfig(
        use_voxel_embedding=False,
        D=4,
        W=32,
        N_freq_xyz=4,
        N_freq_dir=2,
        skips=[2],
        N_obj_code_length=16,
        inst_D=4,
        inst_W=32,
        inst_skips=[2],
    )
    model = ObjectNeRF(model_config)
    model.eval()
    return model


def example_input_objectnerf():
    torch.manual_seed(0)
    n_points = 8
    xyz_embedder = Embedding(3, 4)
    dir_embedder = Embedding(3, 2)
    xyz = torch.rand(n_points, 3) * 2 - 1
    view_dir = F.normalize(torch.randn(n_points, 3), dim=-1)
    inputs = {
        "emb_xyz": xyz_embedder(xyz),
        "emb_dir": dir_embedder(view_dir),
    }
    return (inputs,)


MENAGERIE_ENTRIES = [
    ("ObjectNeRF", build_objectnerf, example_input_objectnerf, 2021, "vendored-pytorch"),
]
