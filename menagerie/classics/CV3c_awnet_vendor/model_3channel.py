import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import DWT, IWT
from .modules_3channel import (
    shortcutblock,
    GCIWTResUp,
    GCWTResDown,
    GCRDB,
    ContextBlock2d,
    SE_net,
    PSPModule,
)
import functools


class AWNet(nn.Module):
    def __init__(self, in_channels, out_channels, block=[2, 2, 2, 3, 4]):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # layer1
        _layer_1_dw = []
        for i in range(block[0]):
            _layer_1_dw.append(GCRDB(64, ContextBlock2d))
        _layer_1_dw.append(GCWTResDown(64, ContextBlock2d, norm_layer=None))
        self.layer1 = nn.Sequential(*_layer_1_dw)

        # layer 2
        _layer_2_dw = []
        for i in range(block[1]):
            _layer_2_dw.append(GCRDB(128, ContextBlock2d))
        _layer_2_dw.append(GCWTResDown(128, ContextBlock2d, norm_layer=None))
        self.layer2 = nn.Sequential(*_layer_2_dw)

        # layer 3
        _layer_3_dw = []
        for i in range(block[2]):
            _layer_3_dw.append(GCRDB(256, ContextBlock2d))
        _layer_3_dw.append(GCWTResDown(256, ContextBlock2d, norm_layer=None))
        self.layer3 = nn.Sequential(*_layer_3_dw)

        # layer 4
        _layer_4_dw = []
        for i in range(block[3]):
            _layer_4_dw.append(GCRDB(512, ContextBlock2d))
        _layer_4_dw.append(GCWTResDown(512, ContextBlock2d, norm_layer=None))
        self.layer4 = nn.Sequential(*_layer_4_dw)

        # layer 5
        _layer_5_dw = []
        for i in range(block[4]):
            _layer_5_dw.append(GCRDB(1024, ContextBlock2d))
        self.layer5 = nn.Sequential(*_layer_5_dw)

        # upsample4
        self.layer4_up = GCIWTResUp(1024, ContextBlock2d)

        # upsample3
        self.layer3_up = GCIWTResUp(512, ContextBlock2d)

        # upsample2
        self.layer2_up = GCIWTResUp(256, ContextBlock2d)

        # upsample1
        self.layer1_up = GCIWTResUp(128, ContextBlock2d)

        self.sc_x1 = shortcutblock(64)
        self.sc_x2 = shortcutblock(128)
        self.sc_x3 = shortcutblock(256)
        self.sc_x4 = shortcutblock(512)

        self.scale_5 = nn.Conv2d(1024, out_channels, kernel_size=3, padding=1)
        self.scale_4 = nn.Conv2d(512, out_channels, kernel_size=3, padding=1)
        self.scale_3 = nn.Conv2d(256, out_channels, kernel_size=3, padding=1)
        self.scale_2 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self.se1 = SE_net(64, 64)
        self.se2 = SE_net(128, 128)
        self.se3 = SE_net(256, 256)
        self.se4 = SE_net(512, 512)
        self.se5 = SE_net(1024, 1024)

        self.enhance = PSPModule(features=64, out_features=64, sizes=(1, 2, 3, 6))

    def forward(self, x, target=None, teacher_latent=None):
        x1 = self.conv1(x)

        x2, x2_dwt = self.layer1(self.se1(x1))
        x3, x3_dwt = self.layer2(self.se2(x2))
        x4, x4_dwt = self.layer3(self.se3(x3))
        x5, x5_dwt = self.layer4(self.se4(x4))
        x5_latent = self.layer5(self.se5(x5))

        x5_out = self.scale_5(x5_latent)
        x5_out = F.sigmoid(x5_out)
        x4_up = self.layer4_up(x5_latent, x5_dwt) + self.sc_x4(x4)
        x4_out = self.scale_4(x4_up)
        x4_out = F.sigmoid(x4_out)
        x3_up = self.layer3_up(x4_up, x4_dwt) + self.sc_x3(x3)
        x3_out = self.scale_3(x3_up)
        x3_out = F.sigmoid(x3_out)
        x2_up = self.layer2_up(x3_up, x3_dwt) + self.sc_x2(x2)
        x2_out = self.scale_2(x2_up)
        x2_out = F.sigmoid(x2_out)
        x1_up = self.layer1_up(x2_up, x2_dwt) + self.sc_x1(x1)
        x1_up = self.enhance(x1_up)
        out = self.final_conv(x1_up)
        out = F.sigmoid(out)

        return (out, x2_out, x3_out, x4_out, x5_out), x5_latent
