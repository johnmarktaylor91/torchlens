# SOURCE: vendored from E2E-AD/AD-MLP @ main
#
# https://github.com/E2E-AD/AD-MLP
# https://raw.githubusercontent.com/E2E-AD/AD-MLP/main/pytorch/admlp/planner.py
#
# AD-MLP ("Rethinking the Open-Loop Evaluation of End-to-End Autonomous
# Driving in nuScenes", Zhai et al. 2023). The real network is the
# `VanillaPlanHead2.plan_head` `nn.Sequential` defined in
# `pytorch/admlp/planner.py` -- a plain MLP planner
# (`Linear(21,512) -> ReLU -> Linear(512,512) -> ReLU -> Linear(512,21)`,
# 21 = velocity_dim(3) * (past_frame(5)+2)) that maps a flattened
# past-ego-state/velocity history vector directly to future waypoints,
# WITHOUT any perception backbone -- this is the paper's central point (a
# bare MLP over past trajectory/velocity beats perception-heavy planners on
# the nuScenes open-loop metric). The real `VanillaPlanHead2.__init__` (the
# architecture) is copied verbatim.
#
# Dropped from the vendor: `VanillaPlanHead2.forward`/`.inference`/`.loss`,
# which are NOT architecture -- they are dataset-plumbing that (a) opens a
# pickled per-scene-token lookup table `stp3_val/data_nuscene.pkl`
# (real-data-dependent, not present in this env) to gather each sample's
# already-computed 21-d velocity feature vector and ground-truth trajectory,
# then (b) call `self.plan_head(input)` (the actual network forward) and
# (c) compute an L1 planning loss / package outputs into a waypoint dict.
# `example_input_admlp` below supplies the flattened `(bs, 21)` feature
# tensor that `VanillaPlanHead2.forward`'s `velocitys.cat(...).permute(1,0)`
# would itself construct from the pickle file, and `build_admlp`'s wrapper
# calls the identical `self.plan_head(input)` op the real `forward` calls --
# same op, same weights, same shapes; only the pickle-file token lookup
# (not a tensor op) is elided. Also dropped: `PlanningMetric_3` (an
# evaluation-only class using `skimage.draw.polygon` for collision-metric
# computation, never part of the forward network) and the free functions
# `gen_dx_bx`/`calculate_birds_eye_view_parameters`/`in_same_segment`/
# `l1_loss`/`l2_loss`/`count_layers`, which are only used by
# `PlanningMetric_3`/`VanillaPlanHead2.loss`/the token-gathering forward,
# not by the `plan_head` network itself. `torchmetrics`/`skimage` (imported
# at the top of the real file for the dropped evaluation code) are not
# needed for the network and are not imported here.

import torch
import torch.nn as nn


class VanillaPlanHead2(nn.Module):
    def __init__(
        self,
        num_heads=8,
        hidden_dim=256,
        dropout=0.1,
        activation="relu",
        ffn_channels=256,
        future_frames=6,
    ):
        super().__init__()
        self.velocity_dim = 3
        self.past_frame = 5
        self.plan_head = nn.Sequential(
            nn.Linear(self.velocity_dim * (self.past_frame + 2), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 7 * 3),
        )

    def forward(self, velocitys):
        # Real `VanillaPlanHead2.forward` gathers `velocitys` (bs, 21) from a
        # pickled per-token feature table and then does exactly this call
        # (`input = velocitys; input = self.plan_head(input)`) before
        # reshaping into a per-waypoint dict for loss computation. Vendored
        # here as a direct tensor-in/tensor-out call to the real network.
        return self.plan_head(velocitys)


def build_admlp():
    return VanillaPlanHead2()


def example_input_admlp():
    # velocity_dim(3) * (past_frame(5) + 2) = 21, matching the real
    # `plan_head`'s first `nn.Linear(21, 512)`.
    return torch.randn(1, 21)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("AD-MLP", "build_admlp", "example_input_admlp", 2023, "vendored"),
]
