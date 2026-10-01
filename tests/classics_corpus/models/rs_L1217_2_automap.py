# FAITHFUL PORT of MattRosenLab/AUTOMAP @ master (original framework: TensorFlow 1.x / Keras)
# https://raw.githubusercontent.com/MattRosenLab/AUTOMAP/master/models/automap_model.py
# The official repo (Zhu, Bo, et al. "Image reconstruction by domain-transform manifold
# learning." Nature 555 (2018): 487-492 -- AUTOMAP) ships only a TF1/Keras functional-API
# model (`AUTOMAP_Basic_Model` in models/automap_model.py) built against
# tensorflow==1.x-era APIs incompatible with the installed tf/keras3 stack in this env, so
# the architecture is transcribed faithfully into torch (rung 3). Every layer/mechanism in
# the original functional graph is preserved in the same order:
#   Input (flattened k-space, fc_input_dim)
#   -> Dense(fc_hidden_dim, tanh)             [fc_2, on 'gpu:0' in the original]
#   -> Dense(fc_output_dim, tanh)             [fc_3, on 'gpu:1' in the original]
#   -> reshape to (im_h, im_w, 1)
#   -> ZeroPad2d(4)                            (pads all 4 spatial sides by 4)
#   -> Conv2d(1->64, k=5, stride=1, pad='same', ReLU)   [c_1]
#   -> Conv2d(64->64, k=5, stride=1, pad='same', ReLU)  [c_2]
#   -> ConvTranspose2d(64->1, k=7, stride=1, pad='same', linear) [c_3]
#   -> reshape/flatten to ((im_h+8)*(im_w+8),)
# The original Keras model returns two outputs, [c_2, output] (a debug feature map and the
# flattened reconstruction); the port keeps both outputs, matching `keras.Model(inputs=fc_1,
# outputs=[c_2, output])` exactly. The GPU device-placement context managers in the original
# (`tf.device('/gpu:0')` / `'/gpu:1'`) are a multi-GPU training convenience with no effect on
# architecture and are omitted (both layers run on whatever device the input tensor is on).
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class AutomapBasicModel(nn.Module):
    """Direct port of `AUTOMAP_Basic_Model(config)` (models/automap_model.py)."""

    def __init__(self, fc_input_dim, fc_hidden_dim, fc_output_dim, im_h, im_w):
        super().__init__()
        self.im_h = im_h
        self.im_w = im_w

        # fc_2 = Dense(fc_hidden_dim, activation='tanh')(fc_1)
        self.fc_2 = nn.Linear(fc_input_dim, fc_hidden_dim)
        # fc_3 = Dense(fc_output_dim, activation='tanh')(fc_2)
        self.fc_3 = nn.Linear(fc_hidden_dim, fc_output_dim)

        assert fc_output_dim == im_h * im_w, "fc_output_dim must reshape exactly to (im_h, im_w, 1)"

        # fc_3 = ZeroPadding2D(4)(fc_3)
        self.zero_pad = nn.ZeroPad2d(4)

        # c_1 = Conv2D(64, 5, strides=1, padding='same', activation='relu')(fc_3)
        self.c_1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding="same")
        # c_2 = Conv2D(64, 5, strides=1, padding='same', activation='relu')(c_1)
        self.c_2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding="same")
        # c_3 = Conv2DTranspose(1, 7, strides=1, padding='same')(c_2)  (no activation, linear)
        self.c_3 = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, fc_1):
        x = torch.tanh(self.fc_2(fc_1))
        x = torch.tanh(self.fc_3(x))

        # Reshape((im_h, im_w, 1)) in Keras (channels-last); torch is channels-first (N,C,H,W)
        x = x.reshape(-1, 1, self.im_h, self.im_w)

        x = self.zero_pad(x)

        c1 = torch.relu(self.c_1(x))
        c2 = torch.relu(self.c_2(c1))
        c3 = self.c_3(c2)

        # Reshape(((im_h+8)*(im_w+8),))(c_3): flatten the padded spatial map
        output = c3.reshape(c3.shape[0], -1)

        # keras.Model(inputs=fc_1, outputs=[c_2, output], name='output')
        return c2, output


def build_automap():
    # Tiny menagerie-scale config: a 16x16 image with matching fc dims, mirroring the
    # released 64x64 / 128x128 example configs (configs/*.json) at a much smaller size.
    im_h = im_w = 16
    fc_output_dim = im_h * im_w
    return AutomapBasicModel(
        fc_input_dim=fc_output_dim,
        fc_hidden_dim=fc_output_dim,
        fc_output_dim=fc_output_dim,
        im_h=im_h,
        im_w=im_w,
    )


def example_input_automap():
    torch.manual_seed(0)
    im_h = im_w = 16
    return (torch.randn(2, im_h * im_w),)


MENAGERIE_ENTRIES = [
    (
        "AUTOMAP (Automated Transform by Manifold Approximation)",
        "build_automap",
        "example_input_automap",
        2018,
        "ported-pytorch",
    ),
]
