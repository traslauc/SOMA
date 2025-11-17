from typing import Optional, Union
import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import gc
from models.utils import get_autocast_params


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None, 
                 dilation = None, freeze_bn = True, anti_aliased = False, early_exit = False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights = weights,replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            net = self.net
            feats = {1:x}
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            feats[2] = x 
            x = net.maxpool(x)
            x = net.layer1(x)
            feats[4] = x 
            x = net.layer2(x)
            feats[8] = x
            if self.early_exit:
                return feats
            x = net.layer3(x)
            feats[16] = x
            x = net.layer4(x)
            feats[32] = x
            # print('1:',feats[1].shape,'2:',feats[2].shape,'4:',feats[4].shape,'8:',feats[8].shape,'16:',feats[16].shape,'32:',feats[32].shape)
            # In fact, we find that for SOMA, we don't actually need to stack five layers of features. Feel free to adjust it according to the task needs!
            return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass

class SRU(nn.Module):
    def __init__(
            self,
            channels: int,
            group_num: int = 4,
            gate_threshold: float = 0.5,
    ):
        super(SRU, self).__init__()
        self.gn = nn.GroupNorm(group_num, channels)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        gn_x = self.gn(x)
        w = (self.gn.weight / torch.sum(self.gn.weight)).view(1, -1, 1, 1)
        w = self.sigmoid(w * gn_x)
        infor_mask = w >= self.gate_threshold
        less_infor_maks = w < self.gate_threshold
        x1 = infor_mask * gn_x
        x2 = less_infor_maks * gn_x
        x11, x12 = torch.split(x1, x1.size(1) // 2, dim=1)
        x21, x22 = torch.split(x2, x2.size(1) // 2, dim=1)
        out = torch.cat([x11 + x22, x12 + x21], dim=1)
        return out


class CRU(nn.Module):
    def __init__(
            self,
            channels: int,
            alpha: float = 0.5,
            squeeze_ratio: int = 2,
            groups: int = 2,
            stride: int = 1,
    ):
        super(CRU, self).__init__()
        self.upper_channel = int(channels * alpha)
        self.low_channel = channels - self.upper_channel
        s_up_c, s_low_c = self.upper_channel // squeeze_ratio, self.low_channel // squeeze_ratio
        self.squeeze_up = nn.Conv2d(self.upper_channel, s_up_c, 1, stride=stride, bias=False)
        self.squeeze_low = nn.Conv2d(self.low_channel, s_low_c, 1, stride=stride, bias=False)
        
        self.gwc = nn.Conv2d(s_up_c, channels, 3, stride=1, padding=1, groups=groups)
        self.pwc1 = nn.Conv2d(s_up_c, channels, 1, bias=False)

        self.pwc2 = nn.Conv2d(s_low_c, channels - s_low_c, 1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        up, low = torch.split(x, [self.upper_channel, self.low_channel], dim=1)
        up, low = self.squeeze_up(up), self.squeeze_low(low)

        y1 = self.gwc(up) + self.pwc1(up)
        y2 = torch.cat((low, self.pwc2(low)), dim=1)

        out = torch.cat((y1, y2), dim=1)
        out_s = self.softmax(self.gap(out))
        out = out * out_s
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)

        return out1 + out2

class SC(nn.Module):
    def __init__(
            self,
            channels: int,
            group_num: int = 4,
            gate_threshold: int = 0.5,
            alpha: float = 0.5,
            squeeze_ratio: int = 2,
            groups: int = 2,
            stride: int = 1,
    ):
        super(SC, self).__init__()
        self.sru = SRU(channels, group_num, gate_threshold)
        self.cru = CRU(channels, alpha, squeeze_ratio, groups, stride)

    def forward(self, x: torch.Tensor):
        x = self.sru(x)
        x = self.cru(x)
        return x