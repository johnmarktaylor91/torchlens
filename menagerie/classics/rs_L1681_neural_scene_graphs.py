# FAITHFUL PORT of https://github.com/princeton-computational-imaging/neural-scene-graphs @ main (8d3d9ce9) (original framework: TensorFlow 1.x/tf.keras)
#
# Ost, Mannan, Thuerey, Knodt, Heide, "Neural Scene Graphs for Dynamic Scenes"
# (CVPR 2021). Official Princeton CIL repo, neural_scene_graph_helper.py /
# main.py. The repo is written against `tensorflow.compat.v1` with a
# `tf.keras.Model` functional-API NeRF MLP (`init_nerf_model`), run through a
# TF1 `tf.Session`-driven training loop -- it cannot be installed/run in this
# torch-only env, so we transcribe the actual trainable-network architecture
# faithfully into self-contained torch:
#
#   - `Embedder`/`get_embedder` (helper.py L23-77): the exact sin/cos
#     positional encoding (include_input + 2^k log-sampled frequency bands)
#     used for both 3D point and view-direction inputs.
#   - `init_nerf_model` (helper.py L80-123): the object-level NeRF MLP -- D
#     Dense(W, relu) layers with an input-concat skip at layer index 4
#     (matching `skips=[4]`), then (with `use_viewdirs=True`) an alpha head
#     off the pre-skip trunk, a bottleneck Dense(256), concatenation with the
#     view-direction/object-pose embedding, 4 more Dense(W//2, relu) layers,
#     and a final Dense(3) RGB head concatenated with the alpha channel.
#   - `create_nerf`'s scene-graph composition (main.py L841-937): the "scene
#     graph" contribution is instantiating one such NeRF MLP per dynamic
#     object *class* (`models_dynamic_dict`), each conditioned by a learned
#     per-object latent code (`init_latent_vector`, concatenated onto the
#     point embedding per `run_network`, main.py L31-60) in addition to a
#     shared background-only NeRF MLP -- multiple latent-conditioned NeRFs
#     composited at render time via non-parametric ray marching/z-sorting
#     (`render_rays`/`combine_z`), which is scene geometry, not a trainable
#     layer, and is intentionally not part of this traced module.
import torch
import torch.nn as nn


class Embedder:
    """Port of helper.py Embedder: sin/cos positional encoding."""

    def __init__(
        self, include_input, input_dims, max_freq_log2, num_freqs, log_sampling, periodic_fns
    ):
        self.include_input = include_input
        self.input_dims = input_dims
        self.periodic_fns = periodic_fns
        embed_fns = []
        d = input_dims
        out_dim = 0
        if include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq_log2, num_freqs)

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


def get_embedder(multires, input_dims=3):
    """Port of helper.py get_embedder (the i==-1 identity branch is handled
    by callers passing multires=0, matching the real `i` disable path)."""
    embedder_obj = Embedder(
        include_input=True,
        input_dims=input_dims,
        max_freq_log2=multires - 1,
        num_freqs=multires,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos],
    )
    return embedder_obj.embed, embedder_obj.out_dim


class NeRFObjectModel(nn.Module):
    """Port of helper.py init_nerf_model: the per-class / background object
    NeRF MLP used throughout the scene graph (D Dense(W) trunk with a
    skip-concat at `skips`, optional view-direction-conditioned color head).
    """

    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_color_head=3,
        output_ch=4,
        skips=(4,),
        use_viewdirs=False,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_color_head = input_ch_color_head
        self.skips = set(skips)
        self.use_viewdirs = use_viewdirs

        trunk = []
        in_dim = input_ch
        for i in range(D):
            trunk.append(nn.Linear(in_dim, W))
            in_dim = W
            if i in self.skips:
                in_dim = W + input_ch
        self.trunk = nn.ModuleList(trunk)
        self.relu = nn.ReLU()

        if use_viewdirs:
            self.alpha_head = nn.Linear(in_dim, 1)
            self.bottleneck = nn.Linear(in_dim, 256)
            self.view_layers = nn.ModuleList(
                [
                    nn.Linear(256 + input_ch_color_head if i == 0 else W // 2, W // 2)
                    for i in range(4)
                ]
            )
            self.rgb_head = nn.Linear(W // 2, 3)
        else:
            self.output_head = nn.Linear(in_dim, output_ch)

    def forward(self, inputs):
        # inputs: (..., input_ch + input_ch_color_head), matching the real
        # tf.keras.Input(shape=(input_ch + input_ch_color_head)) + tf.split.
        inputs_pts = inputs[..., : self.input_ch]
        inputs_color_head = inputs[..., self.input_ch :]

        outputs = inputs_pts
        for i, layer in enumerate(self.trunk):
            outputs = self.relu(layer(outputs))
            if i in self.skips:
                outputs = torch.cat([inputs_pts, outputs], dim=-1)

        if self.use_viewdirs:
            alpha_out = self.alpha_head(outputs)
            bottleneck = self.bottleneck(outputs)
            outputs = torch.cat([bottleneck, inputs_color_head], dim=-1)
            for layer in self.view_layers:
                outputs = self.relu(layer(outputs))
            outputs = self.rgb_head(outputs)
            outputs = torch.cat([outputs, alpha_out], dim=-1)
        else:
            outputs = self.output_head(outputs)
        return outputs


class NeuralSceneGraph(nn.Module):
    """Port of main.py create_nerf's network set: a shared background NeRF
    (`model`) plus one latent-conditioned NeRF per dynamic-object class
    (`models_dynamic_dict`, "Version b: one network for all similar objects
    of the same class"), each object instance contributing a learned latent
    code (`init_latent_vector`) concatenated onto its positional embedding
    exactly as `run_network` does (helper.py L31-60 / main.py L31-60).
    """

    def __init__(
        self,
        D=8,
        W=64,
        multires=4,
        multires_views=2,
        latent_size=8,
        n_object_classes=2,
        use_viewdirs=True,
    ):
        super().__init__()
        self.embed_fn, input_ch_pts = get_embedder(multires, input_dims=3)
        self.embeddirs_fn, input_ch_views = get_embedder(multires_views, input_dims=3)
        self.latent_size = latent_size

        # Background network: pure NeRF MLP on positional + view embedding.
        self.background_model = NeRFObjectModel(
            D=D,
            W=W,
            input_ch=input_ch_pts,
            input_ch_color_head=input_ch_views,
            output_ch=4,
            skips=(4,),
            use_viewdirs=use_viewdirs,
        )

        # One NeRF per object class; input augmented with the latent code
        # (Version b in create_nerf: `input_ch = input_ch + latent_size`).
        self.object_models = nn.ModuleList(
            [
                NeRFObjectModel(
                    D=D,
                    W=W,
                    input_ch=input_ch_pts + latent_size,
                    input_ch_color_head=input_ch_views,
                    output_ch=4,
                    skips=(4,),
                    use_viewdirs=use_viewdirs,
                )
                for _ in range(n_object_classes)
            ]
        )
        # Learned per-object latent vectors (init_latent_vector), one per
        # object instance sharing the object_models[class_id] network.
        self.latent_vectors = nn.ParameterList(
            [nn.Parameter(torch.randn(latent_size) * 0.01) for _ in range(n_object_classes)]
        )

    def forward(self, pts_bg, dirs_bg, pts_obj, dirs_obj, class_ids):
        # Background: run_network's embed_fn(points) + embeddirs_fn(views).
        bg_embedded = torch.cat([self.embed_fn(pts_bg), self.embeddirs_fn(dirs_bg)], dim=-1)
        bg_out = self.background_model(bg_embedded)

        # Objects: embed points, concat this instance's latent code, embed
        # view directions -- matching run_network's NeRF+Latent-Code branch.
        obj_pts_embedded = self.embed_fn(pts_obj)
        obj_dirs_embedded = self.embeddirs_fn(dirs_obj)
        obj_outs = []
        for i, class_id in enumerate(class_ids):
            latent = self.latent_vectors[class_id].unsqueeze(0).expand(pts_obj.shape[0], -1)
            embedded = torch.cat(
                [obj_pts_embedded[:, i, :], latent, obj_dirs_embedded[:, i, :]], dim=-1
            )
            obj_outs.append(self.object_models[class_id](embedded))
        obj_out = torch.stack(obj_outs, dim=1)
        return bg_out, obj_out


# --- staging harness: build + example input ---------------------------------


def build_neural_scene_graph():
    # Shrunk from the paper's real config (D=8, W=256, multires=10,
    # multires_views=4, latent_size up to 32+) to a tiny architecturally
    # faithful set: same MLP depth/skip/viewdir-head structure, small width.
    return NeuralSceneGraph(
        D=8,
        W=16,
        multires=4,
        multires_views=2,
        latent_size=4,
        n_object_classes=2,
        use_viewdirs=True,
    ).eval()


def example_input_neural_scene_graph():
    n_rays = 3
    pts_bg = torch.rand(n_rays, 3)
    dirs_bg = torch.rand(n_rays, 3)
    n_objects = 2
    pts_obj = torch.rand(n_rays, n_objects, 3)
    dirs_obj = torch.rand(n_rays, n_objects, 3)
    class_ids = [0, 1]
    return (pts_bg, dirs_bg, pts_obj, dirs_obj, class_ids)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "NeuralSceneGraphRendering",
        build_neural_scene_graph,
        example_input_neural_scene_graph,
        2021,
        MENAGERIE_ZOO,
    ),
]
