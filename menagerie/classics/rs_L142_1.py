# SOURCE: vendored from maudzung/Complex-YOLOv4-Pytorch @ 2f32996fc4d8942a156545596f76bbc267d5cd8e
# https://raw.githubusercontent.com/maudzung/Complex-YOLOv4-Pytorch/2f32996fc4d8942a156545596f76bbc267d5cd8e/src/models/darknet2pytorch.py
# https://raw.githubusercontent.com/maudzung/Complex-YOLOv4-Pytorch/2f32996fc4d8942a156545596f76bbc267d5cd8e/src/models/darknet_utils.py
# https://raw.githubusercontent.com/maudzung/Complex-YOLOv4-Pytorch/2f32996fc4d8942a156545596f76bbc267d5cd8e/src/models/yolo_layer.py
# https://raw.githubusercontent.com/maudzung/Complex-YOLOv4-Pytorch/2f32996fc4d8942a156545596f76bbc267d5cd8e/src/utils/torch_utils.py
# https://raw.githubusercontent.com/maudzung/Complex-YOLOv4-Pytorch/2f32996fc4d8942a156545596f76bbc267d5cd8e/src/utils/iou_rotated_boxes_utils.py
# https://raw.githubusercontent.com/maudzung/Complex-YOLOv4-Pytorch/2f32996fc4d8942a156545596f76bbc267d5cd8e/src/utils/cal_intersection_rotated_boxes.py
# https://raw.githubusercontent.com/maudzung/Complex-YOLOv4-Pytorch/2f32996fc4d8942a156545596f76bbc267d5cd8e/src/config/cfg/complex_yolov4_tiny.cfg
#
# Complex-YOLOv4 (Nguyen Mau Dung, 2020) -- a LiDAR bird's-eye-view (BEV) 3D object
# detector: the real Darknet/YOLOv4-tiny CNN backbone (parsed from the repo's own
# `.cfg` config-file grammar via `parse_cfg`/`create_network`, exactly as upstream)
# feeding two `YoloLayer` detection heads that regress rotated boxes with an extra
# `(im, re)` = `(sin(yaw), cos(yaw))` orientation channel pair (the "Complex" in
# Complex-YOLO) alongside the usual (x, y, w, h, conf, cls) YOLO outputs. This
# im/re rotated-box parameterization plus the polygon-IoU anchor matching
# (`get_polygons_areas_fix_xy` / `iou_rotated_boxes_targets_vs_anchors`, real
# shapely/scipy convex-hull code from the repo) is the architectural contribution
# over stock YOLO.
#
# `Darknet` builds its `nn.ModuleList` at construction time by parsing the block
# list from a `.cfg` file (real Darknet/YOLO config-file format: `[net]`,
# `[convolutional]`, `[route]`, `[maxpool]`, `[upsample]`, `[yolo]` sections). We
# use the *real* `complex_yolov4_tiny.cfg` from the repo's own `src/config/cfg/`
# directory (embedded verbatim below as a string and written to a real temp file
# at build time so the unmodified `parse_cfg(cfgfile)` signature -- which opens a
# path -- keeps working unchanged). All classes (`Mish`, `MaxPoolDark`,
# `Upsample_expand`, `Reorg`, `GlobalAvgPool2d`, `EmptyModule`, `Darknet`,
# `YoloLayer`) and helper functions (`parse_cfg`, `print_cfg`, `load_conv`,
# `load_conv_bn`, `load_fc`, `convert2cpu`, `to_cpu`, and the polygon/IoU helpers
# in `iou_rotated_boxes_utils.py` / `cal_intersection_rotated_boxes.py`) are
# copied verbatim from the four upstream files. Only mechanical fixes were made
# for self-containment:
#   - `sys.path.append('../')` + package-relative imports (`from models.yolo_layer
#     import YoloLayer`, `from utils.torch_utils import to_cpu`, etc.) collapsed
#     into flat same-module references, since everything now lives in one file.
#   - `parse_cfg` is called on a real temp file written from the embedded cfg
#     text (`tempfile.NamedTemporaryFile`), not a monkeypatched string reader.
#   - `shapely` and `scipy` (both already present in this environment as real
#     packages, not stubs) back `iou_rotated_boxes_utils.get_polygons_areas_fix_xy`,
#     which `YoloLayer.compute_grid_offsets` calls unconditionally on every
#     forward pass (not just the loss path) -- vendored verbatim rather than
#     stripped, since it genuinely runs during plain inference.
#   - `build_targets` / loss computation is real repo code but is never invoked
#     for a forward pass with `targets=None`; kept intact and unmodified for
#     faithfulness even though the trace below does not exercise it.
#   - `use_giou_loss` kwarg on `Darknet.__init__` set to `False` in the builder;
#     purely a constructor argument, no architecture change.

import math
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

# --- utils/torch_utils.py (verbatim) ---------------------------------------

__torch_utils_all__ = ["convert2cpu", "convert2cpu_long", "to_cpu"]


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def to_cpu(tensor):
    return tensor.detach().cpu()


# --- utils/cal_intersection_rotated_boxes.py (verbatim) ---------------------


class Line:
    # ax + by + c = 0
    def __init__(self, p1, p2):
        """

        Args:
            p1: (x, y)
            p2: (x, y)
        """
        self.a = p2[1] - p1[1]
        self.b = p1[0] - p2[0]
        self.c = p2[0] * p1[1] - p2[1] * p1[0]  # cross
        self.device = p1.device

    def cal_values(self, pts):
        return self.a * pts[:, 0] + self.b * pts[:, 1] + self.c

    def find_intersection(self, other):
        # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a * other.b - self.b * other.a
        return torch.tensor(
            [(self.b * other.c - self.c * other.b) / w, (self.c * other.a - self.a * other.c) / w],
            device=self.device,
        )


def intersection_area(rect1, rect2):
    """Calculate the inter

    Args:
        rect1: vertices of the rectangles (4, 2)
        rect2: vertices of the rectangles (4, 2)

    Returns:

    """

    # Use the vertices of the first rectangle as, starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    roll_rect2 = torch.roll(rect2, -1, dims=0)
    for p, q in zip(rect2, roll_rect2):
        if len(intersection) <= 2:
            break  # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".
        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = line.cal_values(intersection)
        roll_intersection = torch.roll(intersection, -1, dims=0)
        roll_line_values = torch.roll(line_values, -1, dims=0)
        for s, t, s_value, t_value in zip(
            intersection, roll_intersection, line_values, roll_line_values
        ):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.find_intersection(Line(s, t))
                new_intersection.append(intersection_point)

        if len(new_intersection) > 0:
            intersection = torch.stack(new_intersection)
        else:
            break

    # Calculate area
    if len(intersection) <= 2:
        return 0.0

    return PolyArea2D(intersection)


def PolyArea2D(pts):
    roll_pts = torch.roll(pts, -1, dims=0)
    area = (pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]).sum().abs() * 0.5
    return area


# --- utils/iou_rotated_boxes_utils.py (verbatim) -----------------------------


def cvt_box_2_polygon(box):
    """
    :param array: an array of shape [num_conners, 2]
    :return: a shapely.geometry.Polygon object
    """
    # use .buffer(0) to fix a line polygon
    # more infor: https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera
    return Polygon([(box[i, 0], box[i, 1]) for i in range(len(box))]).buffer(0)


def get_corners_vectorize(x, y, w, l, yaw):  # noqa: E741 -- verbatim upstream
    """bev image coordinates format - vectorization

    :param x, y, w, l, yaw: [num_boxes,]
    :return: num_boxes x (x,y) of 4 conners
    """
    device = x.device
    bbox2 = torch.zeros((x.size(0), 4, 2), device=device, dtype=torch.float)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # front left
    bbox2[:, 0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bbox2[:, 0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bbox2[:, 1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bbox2[:, 1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bbox2[:, 2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bbox2[:, 2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bbox2[:, 3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bbox2[:, 3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bbox2


def get_polygons_areas_fix_xy(boxes, fix_xy=100.0):
    """
    Args:
        box: (num_boxes, 4) --> w, l, im, re
    """
    device = boxes.device
    n_boxes = boxes.size(0)
    x = torch.full(size=(n_boxes,), fill_value=fix_xy, device=device, dtype=torch.float)
    y = torch.full(size=(n_boxes,), fill_value=fix_xy, device=device, dtype=torch.float)
    w, l, im, re = boxes.t()  # noqa: E741 -- verbatim upstream
    yaw = torch.atan2(im, re)
    boxes_conners = get_corners_vectorize(x, y, w, l, yaw)
    boxes_polygons = [cvt_box_2_polygon(box_) for box_ in boxes_conners]
    boxes_areas = w * l

    return boxes_polygons, boxes_areas


def iou_rotated_boxes_targets_vs_anchors(
    anchors_polygons, anchors_areas, targets_polygons, targets_areas
):
    device = anchors_areas.device
    num_anchors = len(anchors_areas)
    num_targets_boxes = len(targets_areas)

    ious = torch.zeros(size=(num_anchors, num_targets_boxes), device=device, dtype=torch.float)

    for a_idx in range(num_anchors):
        for tg_idx in range(num_targets_boxes):
            intersection = anchors_polygons[a_idx].intersection(targets_polygons[tg_idx]).area
            iou = intersection / (
                anchors_areas[a_idx] + targets_areas[tg_idx] - intersection + 1e-16
            )
            ious[a_idx, tg_idx] = iou

    return ious


def iou_pred_vs_target_boxes(pred_boxes, target_boxes, GIoU=False, DIoU=False, CIoU=False):
    assert pred_boxes.size() == target_boxes.size(), "Unmatch size of pred_boxes and target_boxes"
    device = pred_boxes.device
    n_boxes = pred_boxes.size(0)

    t_x, t_y, t_w, t_l, t_im, t_re = target_boxes.t()
    t_yaw = torch.atan2(t_im, t_re)
    t_conners = get_corners_vectorize(t_x, t_y, t_w, t_l, t_yaw)
    t_areas = t_w * t_l

    p_x, p_y, p_w, p_l, p_im, p_re = pred_boxes.t()
    p_yaw = torch.atan2(p_im, p_re)
    p_conners = get_corners_vectorize(p_x, p_y, p_w, p_l, p_yaw)
    p_areas = p_w * p_l

    ious = []
    giou_loss = torch.tensor([0.0], device=device, dtype=torch.float)
    # Thinking to apply vectorization this step
    for box_idx in range(n_boxes):
        p_cons, t_cons = p_conners[box_idx], t_conners[box_idx]
        if not GIoU:
            p_poly, t_poly = cvt_box_2_polygon(p_cons), cvt_box_2_polygon(t_cons)
            intersection = p_poly.intersection(t_poly).area
        else:
            intersection = intersection_area(p_cons, t_cons)

        p_area, t_area = p_areas[box_idx], t_areas[box_idx]
        union = p_area + t_area - intersection
        iou = intersection / (union + 1e-16)

        if GIoU:
            convex_conners = torch.cat((p_cons, t_cons), dim=0)
            hull = ConvexHull(
                convex_conners.clone().detach().cpu().numpy()
            )  # done on cpu, just need indices output
            convex_conners = convex_conners[hull.vertices]
            convex_area = PolyArea2D(convex_conners)
            giou_loss += 1.0 - (iou - (convex_area - union) / (convex_area + 1e-16))
        else:
            giou_loss += 1.0 - iou

        if DIoU or CIoU:
            raise NotImplementedError

        ious.append(iou)

    return torch.tensor(ious, device=device, dtype=torch.float), giou_loss


# --- models/darknet_utils.py (verbatim) --------------------------------------


def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, "r")
    block = None
    line = fp.readline()
    while line != "":
        line = line.rstrip()
        if line == "" or line[0] == "#":
            line = fp.readline()
            continue
        elif line[0] == "[":
            if block:
                blocks.append(block)
            block = dict()
            block["type"] = line.lstrip("[").rstrip("]")
            # set default value
            if block["type"] == "convolutional":
                block["batch_normalize"] = 0
        else:
            key, value = line.split("=")
            key = key.strip()
            if key == "type":
                key = "_type"
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks


def print_cfg(blocks):
    print("layer     filters    size              input                output")
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block["type"] == "net":
            prev_width = int(block["width"])
            prev_height = int(block["height"])
            continue
        elif block["type"] == "convolutional":
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            is_pad = int(block["pad"])
            pad = (kernel_size - 1) // 2 if is_pad else 0
            width = (prev_width + 2 * pad - kernel_size) // stride + 1
            height = (prev_height + 2 * pad - kernel_size) // stride + 1
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "maxpool":
            pool_size = int(block["size"])  # noqa: F841 -- verbatim upstream (dead var in original)
            stride = int(block["stride"])
            width = prev_width // stride
            height = prev_height // stride
            prev_width = width
            prev_height = height
            prev_filters = prev_filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "route":
            layers = block["layers"].split(",")
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            elif len(layers) == 4:
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = (
                    out_filters[layers[0]]
                    + out_filters[layers[1]]
                    + out_filters[layers[2]]
                    + out_filters[layers[3]]
                )
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] in ["region", "yolo"]:
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "shortcut":
            from_id = int(block["from"])
            from_id = from_id if from_id > 0 else from_id + ind
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "upsample":
            stride = int(block["stride"])
            filters = prev_filters
            width = prev_width * stride
            height = prev_height * stride
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(
        torch.from_numpy(buf[start : start + num_w]).reshape(conv_model.weight.data.shape)
    )
    start = start + num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(
        torch.from_numpy(buf[start : start + num_w]).reshape(conv_model.weight.data.shape)
    )
    start = start + num_w
    return start


def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start : start + num_w]))
    start = start + num_w
    return start


# --- models/yolo_layer.py (verbatim) -----------------------------------------


class YoloLayer(nn.Module):
    """Yolo layer"""

    def __init__(self, num_classes, anchors, stride, scale_x_y, ignore_thresh):
        super(YoloLayer, self).__init__()
        # Update the attributions when parsing the cfg during create the darknet
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.stride = stride
        self.scale_x_y = scale_x_y
        self.ignore_thresh = ignore_thresh

        self.noobj_scale = 100
        self.obj_scale = 1
        self.lgiou_scale = 3.54
        self.leular_scale = 3.54
        self.lobj_scale = 64.3
        self.lcls_scale = 37.4

        self.seen = 0
        # Initialize dummy variables
        self.grid_size = 0
        self.img_size = 0
        self.metrics = {}

    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_size / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = (
            torch.arange(g, device=self.device, dtype=torch.float).repeat(g, 1).view([1, 1, g, g])
        )
        self.grid_y = (
            torch.arange(g, device=self.device, dtype=torch.float)
            .repeat(g, 1)
            .t()
            .view([1, 1, g, g])
        )
        self.scaled_anchors = torch.tensor(
            [(a_w / self.stride, a_h / self.stride, im, re) for a_w, a_h, im, re in self.anchors],
            device=self.device,
            dtype=torch.float,
        )
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Pre compute polygons and areas of anchors
        self.scaled_anchors_polygons, self.scaled_anchors_areas = get_polygons_areas_fix_xy(
            self.scaled_anchors
        )

    def build_targets(self, pred_boxes, pred_cls, target, anchors):
        """Built yolo targets to compute loss
        :param out_boxes: [num_samples or batch, num_anchors, grid_size, grid_size, 6]
        :param pred_cls: [num_samples or batch, num_anchors, grid_size, grid_size, num_classes]
        :param target: [num_boxes, 8]
        :param anchors: [num_anchors, 4]
        :return:
        """
        nB, nA, nG, _, nC = pred_cls.size()
        n_target_boxes = target.size(0)

        # Create output tensors on "device"
        obj_mask = torch.full(
            size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.uint8
        )
        noobj_mask = torch.full(
            size=(nB, nA, nG, nG), fill_value=1, device=self.device, dtype=torch.uint8
        )
        class_mask = torch.full(
            size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float
        )
        iou_scores = torch.full(
            size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float
        )
        tx = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        ty = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tw = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        th = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tim = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tre = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tcls = torch.full(
            size=(nB, nA, nG, nG, nC), fill_value=0, device=self.device, dtype=torch.float
        )
        tconf = obj_mask.float()
        giou_loss = torch.tensor([0.0], device=self.device, dtype=torch.float)

        if n_target_boxes > 0:  # Make sure that there is at least 1 box
            b, target_labels = target[:, :2].long().t()
            target_boxes = torch.cat(
                (target[:, 2:6] * nG, target[:, 6:8]), dim=-1
            )  # scale up x, y, w, h

            gxy = target_boxes[:, :2]
            gwh = target_boxes[:, 2:4]
            gimre = target_boxes[:, 4:6]

            targets_polygons, targets_areas = get_polygons_areas_fix_xy(target_boxes[:, 2:6])
            # Get anchors with best iou
            ious = iou_rotated_boxes_targets_vs_anchors(
                self.scaled_anchors_polygons,
                self.scaled_anchors_areas,
                targets_polygons,
                targets_areas,
            )
            best_ious, best_n = ious.max(0)

            gx, gy = gxy.t()
            gw, gh = gwh.t()
            gim, gre = gimre.t()
            gi, gj = gxy.long().t()
            # Set masks
            obj_mask[b, best_n, gj, gi] = 1
            noobj_mask[b, best_n, gj, gi] = 0

            # Set noobj mask to zero where iou exceeds ignore threshold
            for i, anchor_ious in enumerate(ious.t()):
                noobj_mask[b[i], anchor_ious > self.ignore_thresh, gj[i], gi[i]] = 0

            # Coordinates
            tx[b, best_n, gj, gi] = gx - gx.floor()
            ty[b, best_n, gj, gi] = gy - gy.floor()
            # Width and height
            tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
            th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
            # Im and real part
            tim[b, best_n, gj, gi] = gim
            tre[b, best_n, gj, gi] = gre

            # One-hot encoding of label
            tcls[b, best_n, gj, gi, target_labels] = 1
            class_mask[b, best_n, gj, gi] = (
                pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels
            ).float()
            ious, giou_loss = iou_pred_vs_target_boxes(
                pred_boxes[b, best_n, gj, gi], target_boxes, GIoU=self.use_giou_loss
            )
            iou_scores[b, best_n, gj, gi] = ious
            if self.reduction == "mean":
                giou_loss /= n_target_boxes
            tconf = obj_mask.float()

        return (
            iou_scores,
            giou_loss,
            class_mask,
            obj_mask.type(torch.bool),
            noobj_mask.type(torch.bool),
            tx,
            ty,
            tw,
            th,
            tim,
            tre,
            tcls,
            tconf,
        )

    def forward(self, x, targets=None, img_size=608, use_giou_loss=False):
        """
        :param x: [num_samples or batch, num_anchors * (6 + 1 + num_classes), grid_size, grid_size]
        :param targets: [num boxes, 8] (box_idx, class, x, y, w, l, sin(yaw), cos(yaw))
        :param img_size: default 608
        :return:
        """
        self.img_size = img_size
        self.use_giou_loss = use_giou_loss
        self.device = x.device
        num_samples, _, _, grid_size = x.size()

        prediction = x.view(
            num_samples, self.num_anchors, self.num_classes + 7, grid_size, grid_size
        )
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        # prediction size: [num_samples, num_anchors, grid_size, grid_size, num_classes + 7]

        # Get outputs
        pred_x = torch.sigmoid(prediction[..., 0])
        pred_y = torch.sigmoid(prediction[..., 1])
        pred_w = prediction[..., 2]  # Width
        pred_h = prediction[..., 3]  # Height
        pred_im = prediction[..., 4]  # angle imaginary part
        pred_re = prediction[..., 5]  # angle real part
        pred_conf = torch.sigmoid(prediction[..., 6])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 7:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size)

        # Add offset and scale with anchors
        # pred_boxes size: [num_samples, num_anchors, grid_size, grid_size, 6]
        pred_boxes = torch.empty(prediction[..., :6].shape, device=self.device, dtype=torch.float)
        pred_boxes[..., 0] = pred_x + self.grid_x
        pred_boxes[..., 1] = pred_y + self.grid_y
        pred_boxes[..., 2] = torch.exp(pred_w).clamp(max=1e3) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(pred_h).clamp(max=1e3) * self.anchor_h
        pred_boxes[..., 4] = pred_im
        pred_boxes[..., 5] = pred_re

        output = torch.cat(
            (
                pred_boxes[..., :4].view(num_samples, -1, 4) * self.stride,
                pred_boxes[..., 4:6].view(num_samples, -1, 2),
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            dim=-1,
        )
        # output size: [num_samples, num boxes, 7 + num_classes]

        if targets is None:
            return output, 0
        else:
            self.reduction = "mean"
            (
                iou_scores,
                giou_loss,
                class_mask,
                obj_mask,
                noobj_mask,
                tx,
                ty,
                tw,
                th,
                tim,
                tre,
                tcls,
                tconf,
            ) = self.build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
            )

            loss_x = F.mse_loss(pred_x[obj_mask], tx[obj_mask], reduction=self.reduction)
            loss_y = F.mse_loss(pred_y[obj_mask], ty[obj_mask], reduction=self.reduction)
            loss_w = F.mse_loss(pred_w[obj_mask], tw[obj_mask], reduction=self.reduction)
            loss_h = F.mse_loss(pred_h[obj_mask], th[obj_mask], reduction=self.reduction)
            loss_im = F.mse_loss(pred_im[obj_mask], tim[obj_mask], reduction=self.reduction)
            loss_re = F.mse_loss(pred_re[obj_mask], tre[obj_mask], reduction=self.reduction)
            loss_im_re = (
                1.0 - torch.sqrt(pred_im[obj_mask] ** 2 + pred_re[obj_mask] ** 2)
            ) ** 2  # as tim^2 + tre^2 = 1
            loss_im_re_red = loss_im_re.sum() if self.reduction == "sum" else loss_im_re.mean()
            loss_eular = loss_im + loss_re + loss_im_re_red

            loss_conf_obj = F.binary_cross_entropy(
                pred_conf[obj_mask], tconf[obj_mask], reduction=self.reduction
            )
            loss_conf_noobj = F.binary_cross_entropy(
                pred_conf[noobj_mask], tconf[noobj_mask], reduction=self.reduction
            )
            loss_cls = F.binary_cross_entropy(
                pred_cls[obj_mask], tcls[obj_mask], reduction=self.reduction
            )

            if self.use_giou_loss:
                loss_obj = loss_conf_obj + loss_conf_noobj
                total_loss = (
                    giou_loss * self.lgiou_scale
                    + loss_eular * self.leular_scale
                    + loss_obj * self.lobj_scale
                    + loss_cls * self.lcls_scale
                )
            else:
                loss_obj = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
                total_loss = loss_x + loss_y + loss_w + loss_h + loss_eular + loss_obj + loss_cls

                # Metrics (store loss values using tensorboard)
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "iou_score": to_cpu(iou_scores[obj_mask].mean()).item(),
                "giou_loss": to_cpu(giou_loss).item(),
                "loss_x": to_cpu(loss_x).item(),
                "loss_y": to_cpu(loss_y).item(),
                "loss_w": to_cpu(loss_w).item(),
                "loss_h": to_cpu(loss_h).item(),
                "loss_eular": to_cpu(loss_eular).item(),
                "loss_im": to_cpu(loss_im).item(),
                "loss_re": to_cpu(loss_re).item(),
                "loss_obj": to_cpu(loss_obj).item(),
                "loss_cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
            }

            return output, total_loss


# --- models/darknet2pytorch.py (verbatim) ------------------------------------


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class MaxPoolDark(nn.Module):
    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        """
        darknet output_size = (input_size + p - k) / s +1
        p : padding = k - 1
        k : size
        s : stride
        torch output_size = (input_size + 2*p -k) / s +1
        p : padding = k//2
        """
        p = self.size // 2
        if ((x.shape[2] - 1) // self.stride) != ((x.shape[2] + 2 * p - self.size) // self.stride):
            padding1 = (self.size - 1) // 2
            padding2 = padding1 + 1
        else:
            padding1 = (self.size - 1) // 2
            padding2 = padding1
        if ((x.shape[3] - 1) // self.stride) != ((x.shape[3] + 2 * p - self.size) // self.stride):
            padding3 = (self.size - 1) // 2
            padding4 = padding3 + 1
        else:
            padding3 = (self.size - 1) // 2
            padding4 = padding3
        x = F.max_pool2d(
            F.pad(x, (padding3, padding4, padding1, padding2), mode="replicate"),
            self.size,
            stride=self.stride,
        )
        return x


class Upsample_expand(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride  # noqa: F841 -- verbatim upstream (dead var in original)
        hs = stride  # noqa: F841 -- verbatim upstream (dead var in original)
        x = (
            x.view(B, C, H, 1, W, 1)
            .expand(B, C, H, stride, W, stride)
            .contiguous()
            .view(B, C, H * stride, W * stride)
        )
        return x


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert H % stride == 0
        assert W % stride == 0
        ws = stride
        hs = stride
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile, use_giou_loss):
        super(Darknet, self).__init__()
        self.use_giou_loss = use_giou_loss
        self.blocks = parse_cfg(cfgfile)
        self.width = int(self.blocks[0]["width"])
        self.height = int(self.blocks[0]["height"])

        self.models = self.create_network(self.blocks)  # merge conv, bn,leaky
        self.yolo_layers = [
            layer for layer in self.models if layer.__class__.__name__ == "YoloLayer"
        ]

        self.loss = self.models[len(self.models) - 1]

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x, targets=None):
        # batch_size, c, h, w
        img_size = x.size(2)
        ind = -2
        self.loss = None
        outputs = dict()
        loss = 0.0
        yolo_outputs = []
        for block in self.blocks:
            ind = ind + 1
            # if ind > 0:
            #    return x

            if block["type"] == "net":
                continue
            elif block["type"] in [
                "convolutional",
                "maxpool",
                "reorg",
                "upsample",
                "avgpool",
                "softmax",
                "connected",
            ]:
                x = self.models[ind](x)
                outputs[ind] = x
            elif block["type"] == "route":
                layers = block["layers"].split(",")
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if "groups" not in block.keys() or int(block["groups"]) == 1:
                        x = outputs[layers[0]]
                        outputs[ind] = x
                    else:
                        groups = int(block["groups"])
                        group_id = int(block["group_id"])
                        _, b, _, _ = outputs[layers[0]].shape
                        x = outputs[layers[0]][
                            :, b // groups * group_id : b // groups * (group_id + 1)
                        ]
                        outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
                elif len(layers) == 4:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x3 = outputs[layers[2]]
                    x4 = outputs[layers[3]]
                    x = torch.cat((x1, x2, x3, x4), 1)
                    outputs[ind] = x
                else:
                    print("rounte number > 2 ,is {}".format(len(layers)))

            elif block["type"] == "shortcut":
                from_layer = int(block["from"])
                activation = block["activation"]
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == "leaky":
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == "relu":
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block["type"] == "yolo":
                x, layer_loss = self.models[ind](x, targets, img_size, self.use_giou_loss)
                loss += layer_loss
                yolo_outputs.append(x)
            elif block["type"] == "cost":
                continue
            else:
                print("unknown type %s" % (block["type"]))
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))

        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()

        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        for block in blocks:
            if block["type"] == "net":
                prev_filters = int(block["channels"])
                continue
            elif block["type"] == "convolutional":
                conv_id = conv_id + 1
                batch_normalize = int(block["batch_normalize"])
                filters = int(block["filters"])
                kernel_size = int(block["size"])
                stride = int(block["stride"])
                is_pad = int(block["pad"])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block["activation"]
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module(
                        "conv{0}".format(conv_id),
                        nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False),
                    )
                    model.add_module("bn{0}".format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module(
                        "conv{0}".format(conv_id),
                        nn.Conv2d(prev_filters, filters, kernel_size, stride, pad),
                    )
                if activation == "leaky":
                    model.add_module("leaky{0}".format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == "relu":
                    model.add_module("relu{0}".format(conv_id), nn.ReLU(inplace=True))
                elif activation == "mish":
                    model.add_module("mish{0}".format(conv_id), Mish())
                else:
                    print("[INFO] No error, the convolution haven't activate {}".format(activation))

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block["type"] == "maxpool":
                pool_size = int(block["size"])
                stride = int(block["stride"])
                if stride == 1 and pool_size % 2:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=3 stride=1
                    model = nn.MaxPool2d(
                        kernel_size=pool_size, stride=stride, padding=pool_size // 2
                    )
                elif stride == pool_size:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=2 stride=2
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=0)
                else:
                    model = MaxPoolDark(pool_size, stride)
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block["type"] == "avgpool":
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block["type"] == "softmax":
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block["type"] == "cost":
                if block["_type"] == "sse":
                    model = nn.MSELoss(size_average=True)
                elif block["_type"] == "L1":
                    model = nn.L1Loss(size_average=True)
                elif block["_type"] == "smooth":
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block["type"] == "reorg":
                stride = int(block["stride"])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                prev_stride = prev_stride * stride
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block["type"] == "upsample":
                stride = int(block["stride"])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)

                models.append(Upsample_expand(stride))
                # models.append(Upsample_interpolate(stride))

            elif block["type"] == "route":
                layers = block["layers"].split(",")
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if "groups" not in block.keys() or int(block["groups"]) == 1:
                        prev_filters = out_filters[layers[0]]
                        prev_stride = out_strides[layers[0]]
                    else:
                        prev_filters = out_filters[layers[0]] // int(block["groups"])
                        prev_stride = out_strides[layers[0]] // int(block["groups"])
                elif len(layers) == 2:
                    assert layers[0] == ind - 1 or layers[1] == ind - 1
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 4:
                    assert layers[0] == ind - 1
                    prev_filters = (
                        out_filters[layers[0]]
                        + out_filters[layers[1]]
                        + out_filters[layers[2]]
                        + out_filters[layers[3]]
                    )
                    prev_stride = out_strides[layers[0]]
                else:
                    print("route error!!!")

                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block["type"] == "shortcut":
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block["type"] == "connected":
                filters = int(block["output"])
                if block["activation"] == "linear":
                    model = nn.Linear(prev_filters, filters)
                elif block["activation"] == "leaky":
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters), nn.LeakyReLU(0.1, inplace=True)
                    )
                elif block["activation"] == "relu":
                    model = nn.Sequential(nn.Linear(prev_filters, filters), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block["type"] == "yolo":
                anchor_masks = [int(i) for i in block["mask"].split(",")]
                anchors = [float(i) for i in block["anchors"].split(",")]
                anchors = [
                    (anchors[i], anchors[i + 1], math.sin(anchors[i + 2]), math.cos(anchors[i + 2]))
                    for i in range(0, len(anchors), 3)
                ]
                anchors = [anchors[i] for i in anchor_masks]

                num_classes = int(block["classes"])
                self.num_classes = num_classes
                scale_x_y = float(block["scale_x_y"])
                ignore_thresh = float(block["ignore_thresh"])

                yolo_layer = YoloLayer(
                    num_classes=num_classes,
                    anchors=anchors,
                    stride=prev_stride,
                    scale_x_y=scale_x_y,
                    ignore_thresh=ignore_thresh,
                )

                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                print("unknown type %s" % (block["type"]))

        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block["type"] == "net":
                continue
            elif block["type"] == "convolutional":
                model = self.models[ind]
                batch_normalize = int(block["batch_normalize"])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block["type"] == "connected":
                model = self.models[ind]
                if block["activation"] != "linear":
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block["type"] in (
                "maxpool",
                "reorg",
                "upsample",
                "route",
                "shortcut",
                "yolo",
                "avgpool",
                "softmax",
                "cost",
            ):
                pass
            else:
                print("unknown type %s" % (block["type"]))


# --- menagerie build/example glue -------------------------------------------

# The real `src/config/cfg/complex_yolov4_tiny.cfg` from the repo, embedded
# verbatim (unmodified content) so `build_complex_yolov4_tiny()` is
# self-contained with no external file dependency at import time.
_COMPLEX_YOLOV4_TINY_CFG = """[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch=64
subdivisions=1
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

##################################

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=30
activation=linear



[yolo]
mask = 3,4,5
anchors = 11, 15, 0, 11, 25, 0, 23, 49, 0, 23, 55, 0, 24, 53, 0, 25, 61, 0
classes=3
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
resize=1.5
nms_kind=greedynms
beta_nms=0.6

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 23

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=30
activation=linear

[yolo]
mask = 0,1,2
anchors = 11, 15, 0, 11, 25, 0, 23, 49, 0, 23, 55, 0, 24, 53, 0, 25, 61, 0
classes=3
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
resize=1.5
nms_kind=greedynms
beta_nms=0.6
"""


def build_complex_yolov4_tiny():
    # Real repo cfg (complex_yolov4_tiny.cfg), written to a real temp file so the
    # unmodified `parse_cfg(cfgfile)` (which opens a path) works unchanged. Real
    # `Darknet` class builds the exact YOLOv4-tiny CSP backbone + two rotated-box
    # `YoloLayer` heads defined by that cfg (3 classes, 6 anchors/scale, as shipped).
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(_COMPLEX_YOLOV4_TINY_CFG)
        cfg_path = f.name
    model = Darknet(cfgfile=cfg_path, use_giou_loss=False)
    model.eval()
    return model


def example_input_complex_yolov4_tiny():
    # BEV pseudo-image input matching the cfg's [net] width/height/channels (416x416x3).
    return torch.randn(1, 3, 416, 416)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Complex-YOLOv4-tiny",
        "build_complex_yolov4_tiny",
        "example_input_complex_yolov4_tiny",
        2020,
        "vendored",
    ),
]
