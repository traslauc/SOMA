import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.hub import load
import math
import gc
from models.backbones import ResNet50,SC
# from torchsummary import summary


def check_na(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} has NaN!")
        exit()
    if torch.isinf(tensor).any():
        print(f"{name} has Inf!")
        exit()


class GradientFilter(nn.Module):
    def __init__(self, in_channels, branch_channels, num_directions=8, kernel_size=3):
        super().__init__()
        self.num_directions = num_directions
        self.branches = nn.ModuleList()
        angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5][:num_directions]
        for angle in angles:
            branch = nn.Conv2d(in_channels, branch_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
            weight = self.get_rotated_sobel_kernel(angle, in_channels, branch_channels, kernel_size)
            branch.weight.data.copy_(weight)
            self.branches.append(branch)
        
        self.fusion_conv = nn.Conv2d(branch_channels * num_directions, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)

    def get_rotated_sobel_kernel(self, angle, in_channels, out_channels, kernel_size):
        base_kernel = torch.tensor([[-1., 0., 1.],
                                     [-2., 0., 2.],
                                     [-1., 0., 1.]], dtype=torch.float32)
        theta = math.radians(angle)
        rotated = torch.zeros_like(base_kernel)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = j - center
                y = i - center
                xr = x * math.cos(theta) - y * math.sin(theta)
                yr = x * math.sin(theta) + y * math.cos(theta)
                xi = int(round(xr)) + center
                yi = int(round(yr)) + center
                if 0 <= xi < kernel_size and 0 <= yi < kernel_size:
                    rotated[i, j] = base_kernel[yi, xi]

        weight = rotated.unsqueeze(0).unsqueeze(0).repeat(out_channels, in_channels, 1, 1)
        return weight

    def forward(self, x):
        branch_outs = []
        for branch in self.branches:
            out = branch(x)
            out = F.relu(out)
            branch_outs.append(out)
        multi_grad = torch.cat(branch_outs, dim=1)
        fused = self.fusion_conv(multi_grad)
        fused = self.norm(fused)
        return fused


# class LearnableConv(nn.Module):
#     def __init__(self, in_channels, branch_channels, num_directions=8, kernel_size=3):
#         super().__init__()
#         self.num_directions = num_directions
#         self.branches = nn.ModuleList()

#         for _ in range(num_directions):
#             branch = nn.Sequential(
#                 nn.Conv2d(in_channels, branch_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
#                 nn.ReLU(inplace=True)
#             )
#             self.branches.append(branch)

#         self.fusion_conv = nn.Conv2d(branch_channels * num_directions, in_channels, kernel_size=1)
#         self.norm = nn.BatchNorm2d(in_channels)

#     def forward(self, x):
#         branch_outs = [branch(x) for branch in self.branches]
#         multi_feat = torch.cat(branch_outs, dim=1)
#         fused = self.fusion_conv(multi_feat)
#         return self.norm(fused)
    

class FGE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.sc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            SC(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.md_gradient = GradientFilter(
            in_channels, branch_channels=in_channels // 2, num_directions=8
        )

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

        self.dilated_conv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=d, padding=d)
            for d in [1, 2, 3]
        ])

        self.fusion_multi_scale = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(in_channels, eps=1e-6)

        self.gaussian = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        nn.init.normal_(self.gaussian.weight, mean=0.0, std=0.01)

    def forward(self, x):
        x = x.float()

        feat = self.sc(x) + x
        
        grad_feat = self.md_gradient(feat)
        fused_feat = feat + grad_feat

        ca = self.channel_att(fused_feat)
        sa = self.spatial_att(fused_feat)
        att_feat = fused_feat * ca * sa + fused_feat

        H, W = att_feat.shape[2:]

        ms_features = []
        for conv in self.dilated_conv:
            feat_d = conv(att_feat)
            ms_features.append(F.interpolate(feat_d, size=(H, W), mode='bilinear', align_corners=False))

        ms_fused = self.fusion_multi_scale(torch.cat(ms_features, dim=1))

        normed = self.norm(ms_fused)
        filtered = self.gaussian(normed)

        output = filtered + att_feat
        return output




# ---------------- OPT Encoder ----------------
class OPTEncoder(nn.Module):
    def __init__(self,dino_model):
        super(OPTEncoder, self).__init__()
        self.resnet_opt = ResNet50(pretrained=False)
        self.fgeset = nn.ModuleDict({
            '2': FGE(64),
            '4': FGE(256),
            '8': FGE(512),
        })
        self.dino_model = dino_model

    def forward(self, x):
        x = x.float()
        B, C, H, W = x.shape
        with torch.no_grad():
            x_dino = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)
            dino_feats = self.dino_model.forward_features(x_dino)
            dino_out = dino_feats['x_norm_patchtokens'].permute(0, 2, 1).reshape(B, 1024, 32, 32)
            dino_out = dino_out.float()
            check_na(dino_out, "DINOv2 output")

        feats_opt = self.resnet_opt(x)  # dict[int, tensor]
        feats_opt = {k: v.float() for k, v in feats_opt.items()} 

        filtered_feats_opt = {}

        for scale in [ '2', '4', '8']:

            feat = feats_opt[int(scale)]
            filtered_feats_opt[scale] = self.fgeset[scale](feat)
            
            check_na(filtered_feats_opt[scale], f"ResNet opt {scale}+ filtered")
        
        filtered_feats_opt['16'] = dino_out

        feats_opt_str = {str(k): v for k, v in feats_opt.items()}
        feats_opt_str.update(filtered_feats_opt)

        return feats_opt_str


class SAREncoder(nn.Module):
    def __init__(self,dino_model):
        super(SAREncoder, self).__init__()
        self.resnet_sar = ResNet50(pretrained=False)
        self.fgeset = nn.ModuleDict({
            '2': FGE(64),
            '4': FGE(256),
            '8': FGE(512),
        })
        self.dino_model = dino_model

    def forward(self, x):
        x = x.float()
        B, C, H, W = x.shape

        with torch.no_grad():
            x_dino = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)
            dino_feats = self.dino_model.forward_features(x_dino)
            dino_out = dino_feats['x_norm_patchtokens'].permute(0, 2, 1).reshape(B, 1024, 32, 32)
            dino_out = dino_out.float()
            check_na(dino_out, "DINOv2 output")

        feats_sar = self.resnet_sar(x)  # dict[int, tensor]
        feats_sar = {k: v.float() for k, v in feats_sar.items()}  # ✅ 加这一行

        filtered_feats_sar = {}
        for scale in ['2','4','8']:
            feat = feats_sar[int(scale)]
            filtered_feats_sar[scale] = self.fgeset[scale](feat)
   
            check_na(filtered_feats_sar[scale], f"ResNet sar {scale}+ filtered")

        filtered_feats_sar['16'] = dino_out

        feats_sar_str = {str(k): v for k, v in feats_sar.items()}
        feats_sar_str.update(filtered_feats_sar)

        return feats_sar_str


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.dino_model = load(
            '/home/F/whd/proj/somatchv5/facebookresearch_dinov2_main', # set your dinov2 model address here
            'dinov2_vitl14_reg',
            source='local'
        )
        print("DINOv2 model loaded")
        self.dino_model.eval()
        for param in self.dino_model.parameters():
            param.requires_grad = False

        self.opt_encoder = OPTEncoder(self.dino_model)
        self.sar_encoder = SAREncoder(self.dino_model)

    def forward(self, x, label):
        if label == 'sar':
            sar_feats = self.sar_encoder(x)
            return {'sar': sar_feats}
        elif label == 'opt':
            opt_feats = self.opt_encoder(x)
            return {'opt': opt_feats}



