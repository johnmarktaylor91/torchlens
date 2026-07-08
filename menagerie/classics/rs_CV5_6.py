# SOURCE: vendored from vidursatija/BlazePalm @ aa31761
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Taken from https://github.com/vidursatija/BlazeFace-CoreML/blob/master/ML/blazeface.py
class ResModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResModule, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        # kernel size is always 3
        kernel_size = 3

        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        layers = [ResModule(in_channels, in_channels) for _ in range(7)]

        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


# From https://github.com/google/mediapipe/blob/master/mediapipe/models/palm_detection.tflite
class PalmDetector(nn.Module):
    def __init__(self):
        super(PalmDetector, self).__init__()

        self.backbone1 = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), value=0.0),
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0, bias=True
            ),
            nn.ReLU(inplace=True),
            ResBlock(32),
            ResModule(32, 64, stride=2),
            ResBlock(64),
            ResModule(64, 128, stride=2),
            ResBlock(128),
        )

        self.backbone2 = nn.Sequential(ResModule(128, 256, stride=2), ResBlock(256))

        self.backbone3 = nn.Sequential(ResModule(256, 256, stride=2), ResBlock(256))

        self.upscale8to16 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True
            ),
            nn.ReLU(inplace=True),
        )
        self.scaled16add = ResModule(256, 256)

        self.upscale16to32 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True
            ),
            nn.ReLU(inplace=True),
        )
        self.scaled32add = ResModule(128, 128)

        self.class_32 = nn.Conv2d(
            in_channels=128, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.class_16 = nn.Conv2d(
            in_channels=256, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.class_8 = nn.Conv2d(
            in_channels=256, out_channels=6, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.reg_32 = nn.Conv2d(
            in_channels=128, out_channels=36, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.reg_16 = nn.Conv2d(
            in_channels=256, out_channels=36, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.reg_8 = nn.Conv2d(
            in_channels=256, out_channels=108, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        b1 = self.backbone1(x)  # 32x32
        # print(b1.size())

        b2 = self.backbone2(b1)  # 16x16
        # print(b2.size())

        b3 = self.backbone3(b2)  # 8x8
        # print(b3.size())

        b2 = self.upscale8to16(b3) + b2  # 16x16
        b2 = self.scaled16add(b2)  # 16x16
        # print(b2.size())

        b1 = self.upscale16to32(b2) + b1  # 32x32
        b1 = self.scaled32add(b1)
        # print(b1.size())

        c8 = self.class_8(b3).permute(0, 2, 3, 1).reshape(-1, 384, 1)
        c16 = self.class_16(b2).permute(0, 2, 3, 1).reshape(-1, 512, 1)
        c32 = self.class_32(b1).permute(0, 2, 3, 1).reshape(-1, 2048, 1)

        r8 = self.reg_8(b3).permute(0, 2, 3, 1).reshape(-1, 384, 18)
        r16 = self.reg_16(b2).permute(0, 2, 3, 1).reshape(-1, 512, 18)
        r32 = self.reg_32(b1).permute(0, 2, 3, 1).reshape(-1, 2048, 18)

        c = torch.cat([c32, c16, c8], dim=1)
        r = torch.cat([r32, r16, r8], dim=1)  # needs to be anchored

        return c, r

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_anchors(self, path):
        self.anchors = torch.tensor(np.load(path), dtype=torch.float32)
        assert self.anchors.ndimension() == 2
        assert self.anchors.shape[0] == 2944
        assert self.anchors.shape[1] == 4

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)

        thresh = 100
        raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)

        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= 0.7

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

    def predict_on_image(self, img):
        """Makes a prediction on a single image.
        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be
                 128 pixels.
        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))

    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.
        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.
        Returns:
            A list containing a tensor of face detections for each image in
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).
        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == 256
        assert x.shape[3] == 256

        # 1. Preprocess the images into tensors:
        # x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        detections = self._tensors_to_detections(out[1], out[0], self.anchors)

        # 4. Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            if len(faces) > 0:
                faces = torch.stack(faces)
                filtered_detections.append(faces)

        return filtered_detections

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = torch.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / 256 * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / 256 * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / 256 * anchors[:, 2] * 2.6
        h = raw_boxes[..., 3] / 256 * anchors[:, 3] * 2.6

        y_center = y_center - h / 5.2

        boxes[..., 0] = x_center - w / 2.0  # ymin
        boxes[..., 1] = y_center - h / 2.0  # xmin
        boxes[..., 2] = x_center + w / 2.0  # ymax
        boxes[..., 3] = y_center + h / 2.0  # xmax

        for k in range(7):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[..., offset] / 256 * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / 256 * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:
        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."
        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.
        The input detections should be a Tensor of shape (count, 17).
        Returns a list of PyTorch tensors, one for each detected face.

        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0:
            return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = torch.argsort(detections[:, 18], descending=True)

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious >= 0.3
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :18]
                scores = detections[overlapping, 18:19]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted_detection[:18] = weighted
                weighted_detection[18] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections


# IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py


def intersect(box_a, box_b):
    """We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2)
    )
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2)
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = (
        ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    )  # [A,B]
    area_b = (
        ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    )  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)


if __name__ == "__main__":
    m = PalmDetector()
    import coremltools as ct

    m.load_weights("./palmdetector.pth")
    m.load_anchors("./anchors.npy")
    m.eval()

    # np.random.seed(0)
    # a = torch.from_numpy(np.random.rand(1, 3, 256, 256).astype(np.float32))
    # bb = m(a)
    # np.save('npseed0reg_torch.npy', bb[1].detach().numpy())
    traced_model = torch.jit.trace(m, torch.rand(1, 3, 256, 256), check_trace=True)
    # print(traced_m)
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=ct.Shape(
                    shape=(
                        1,
                        3,
                        256,
                        256,
                    )
                ),
                bias=[-1, -1, -1],
                scale=1 / 127.5,
            )
        ],
    )
    mlmodel.save("./BlazePalmDetector.mlmodel")


def build_blazepalm_detector():
    """Build the vendored BlazePalm palm detector.

    Returns
    -------
    torch.nn.Module
        Randomly initialized PalmDetector from the source repository.
    """
    return PalmDetector().eval()


def example_input_blazepalm_detector():
    """Return an example image tensor for BlazePalm.

    Returns
    -------
    torch.Tensor
        Example RGB image batch with shape ``(1, 3, 256, 256)``.
    """
    return torch.randn(1, 3, 256, 256)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "BlazePalm + BlazeLandmark (MediaPipe Hands)",
        "build_blazepalm_detector",
        "example_input_blazepalm_detector",
        2019,
        "CV5",
    )
]
