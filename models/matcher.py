import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class FeatureCorrelation(nn.Module):
    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        feat1 = feat1.view(B, C, -1)
        feat2 = feat2.view(B, C, -1)
        corr = torch.bmm(feat1.transpose(1, 2), feat2)
        corr = corr.view(B, H, W, H, W)
        return corr

def warp_image(image, flow):
    B, C, H, W = image.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing='ij'
    )
    base_grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(B, 1, 1, 1)  
    norm_flow = torch.zeros_like(flow)
    norm_flow[:, 0] = flow[:, 0] / ((W - 1) / 2)
    norm_flow[:, 1] = flow[:, 1] / ((H - 1) / 2)

    grid = base_grid + norm_flow.permute(0, 2, 3, 1)

    warped = F.grid_sample(
        image, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return warped

def flow_loss(pred_flow, gt_flow, mask=None, certainty=None, threshold=2.0):
    epe = torch.norm(pred_flow - gt_flow, dim=1)  # [B, H, W]
    epe_sq = epe ** 2

    if mask is not None:
        epe_sq = epe_sq * mask
        rmse = torch.sqrt(epe_sq.sum() / (mask.sum() + 1e-6))
    else:
        rmse = torch.sqrt(epe_sq.mean())

    result = {'rmse': rmse}

    if certainty is not None:
        epe_ = epe.unsqueeze(1)  
        gt_target = torch.exp(-epe_)
        loss_certainty = F.mse_loss(torch.sigmoid(certainty), gt_target)
        result['certainty_loss'] = loss_certainty

    return result

def rb_loss(pred, target, alpha=0.5, eps=1e-6, reduction='mean'):
    diff = pred - target
    loss = torch.pow((diff ** 2 + eps ** 2), alpha)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  

def flow_lossv2(flow_pred, flow_gt, certainty_logits, eps=1e-6):
    flow_diff = torch.norm(flow_pred - flow_gt, dim=1)    
    certainty_prob = torch.sigmoid(certainty_logits).squeeze(1)  

    weighted_sq_error = (flow_diff ** 2) * certainty_prob
    loss = torch.sqrt(weighted_sq_error.sum() / (certainty_prob.sum() + eps))

    return loss

def pncc(feats1, feats2, window_size=5, eps=1e-5, scales=None):
    def zncc_map(x, y):
        mean_x = F.avg_pool2d(x, window_size, 1, window_size // 2)
        mean_y = F.avg_pool2d(y, window_size, 1, window_size // 2)
        x_centered = x - mean_x
        y_centered = y - mean_y
        std_x = torch.sqrt(F.avg_pool2d(x_centered**2, window_size, 1, window_size // 2) + eps)
        std_y = torch.sqrt(F.avg_pool2d(y_centered**2, window_size, 1, window_size // 2) + eps)
        zncc = (x_centered * y_centered) / (std_x * std_y + eps)
        return zncc.mean(dim=1, keepdim=True)  # → [B,1,H,W]

    scores = []
    for key in scales:
        f1 = feats1[key]
        f2 = feats2[key]
        assert f1.shape == f2.shape, f"Feature shape mismatch at scale {key}"
        zncc = zncc_map(f1, f2).mean()  # scalar per batch
        scores.append(zncc)

    return torch.stack(scores).mean()  # → scalar


def uni_loss(flow_pred, flow_gt, num_patches=4):
    B, C, H, W = flow_pred.shape
    patch_H = H // num_patches
    patch_W = W // num_patches

    losses = []
    for b in range(B):
        patch_rmses = []
        for i in range(num_patches):
            for j in range(num_patches):
                h_start = i * patch_H
                h_end = (i + 1) * patch_H
                w_start = j * patch_W
                w_end = (j + 1) * patch_W

                pred_patch = flow_pred[b, :, h_start:h_end, w_start:w_end]
                gt_patch = flow_gt[b, :, h_start:h_end, w_start:w_end]

                error = (pred_patch - gt_patch).pow(2).sum(dim=0).sqrt()
                rmse = error.mean()  
                patch_rmses.append(rmse)

        patch_rmses = torch.stack(patch_rmses)  # [num_patches^2]
        std = torch.std(patch_rmses)  
        losses.append(std)

    return torch.mean(torch.stack(losses))


def visualize_flow(flow):
    import numpy as np
    import cv2
    flow = flow[0].detach().cpu().numpy()
    fx, fy = flow[0], flow[1]
    H, W = fx.shape
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb_tensor = torch.from_numpy(rgb).float() / 255.0
    return rgb_tensor.permute(2, 0, 1)

class ChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.reduce(x)

class CoarseMatcher(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 6)
        )

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        theta = self.regressor(x)
        theta = theta.view(-1, 2, 3)
        return theta

class TMatcher(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 6)
        )

    def forward(self, feat1, feat2, theta):
        B, C, H, W = feat2.shape
        # 根据当前 theta 构造采样网格，warp feat2
        grid = F.affine_grid(theta, size=(B, feat2.size(1), H, W), align_corners=False)
        warped_feat2 = F.grid_sample(feat2, grid, align_corners=False)
        x = torch.cat([feat1, warped_feat2], dim=1)
        x = self.encoder(x)
        delta_theta = self.decoder(x)
        delta_theta = delta_theta.view(B, 2, 3)
        return delta_theta

def compose_theta(theta, delta_theta):
    B = theta.shape[0]
    ones = torch.tensor([0, 0, 1], dtype=theta.dtype, device=theta.device).view(1, 1, 3).expand(B, 1, 3)
    theta_h = torch.cat([theta, ones], dim=1)         # (B, 3, 3)
    delta_theta_h = torch.cat([delta_theta, ones], dim=1) # (B, 3, 3)
    composed = torch.bmm(delta_theta_h, theta_h)
    return composed[:, :2, :]  

def theta_to_flow(theta, H, W):
    B = theta.shape[0]
    grid = F.affine_grid(theta, size=(B, 1, H, W), align_corners=False)  # (B, H, W, 2)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=theta.device),
        torch.linspace(-1, 1, W, device=theta.device),
        indexing='ij'
    )
    base_grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
    norm_flow = grid - base_grid  
    flow = torch.zeros(B, 2, H, W, device=theta.device)
    flow[:, 0] = norm_flow[..., 0] * ((W - 1) / 2)
    flow[:, 1] = norm_flow[..., 1] * ((H - 1) / 2)
    return flow


class RegressEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stage1 = self._make_block(in_channels, 128, kernel_size=3)
        self.stage2 = self._make_block(128, 128, kernel_size=3)
        self.stage5 = self._make_block(128, 64, kernel_size=3)
    def _make_block(self, in_ch, out_ch, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            DeformableBlock(out_ch, out_ch, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True)
            
        )
    def forward(self, x):
        x = self.stage1(x)  
        x = self.stage2(x)  
        x = self.stage5(x)  
        return x

class DeformableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=3, padding=1
        )
        self.defconv = DeformConv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.defconv(x, offset)


class TFMatcher(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = RegressEncoder(in_channels * 2)
        
        self.theta_decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 6),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flow_decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )
        
    def forward(self, feat1, feat2, theta):
        B, C, H, W = feat2.shape
        
        grid = F.affine_grid(theta, size=(B, feat2.size(1), H, W), align_corners=False)
        warped_feat2 = F.grid_sample(feat2, grid, align_corners=False)

        x = torch.cat([feat1, warped_feat2], dim=1)
        x_encoded = self.encoder(x)

        delta_theta = self.theta_decoder(x_encoded).view(B, 2, 3)
        
        pred_flow = self.flow_decoder(x_encoded)
        
        return delta_theta, pred_flow
    

class RefineMatcher(nn.Module):
    def __init__(self, in_channels, use_certainty=True, concat_logits=True):
        super().__init__()
        self.use_certainty = use_certainty
        self.concat_logits = concat_logits

        self.disp_emb = nn.Conv2d(2, 16, 1)
        base_in_dim = in_channels * 2 + 16 
        self.in_dim = base_in_dim + (1 if concat_logits else 0)

        self.encoder = RegressEncoder(in_channels=self.in_dim)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x_feat, y_feat, flow, logits=None):
        B, C, H, W = x_feat.shape
        if flow.shape[2:] != (H, W):
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)

        if self.concat_logits and logits is not None:
            if logits.shape[2:] != (H, W):
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        norm_flow = torch.stack([
            flow[:, 0] / ((W - 1) / 2),
            flow[:, 1] / ((H - 1) / 2)
        ], dim=1)
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1 + 1/H, 1 - 1/H, H, device=x_feat.device),
            torch.linspace(-1 + 1/W, 1 - 1/W, W, device=x_feat.device),
            indexing='ij'
        ), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        warped_y = F.grid_sample(y_feat, (norm_flow + grid).permute(0, 2, 3, 1), align_corners=False)

        disp = flow  # pixel unit
        disp_emb = self.disp_emb(disp)

        features = [x_feat, warped_y, disp_emb]
        if self.concat_logits and logits is not None:
            features.append(logits)
        x = torch.cat(features, dim=1)

        x_encoded = self.encoder(x)
        out = self.out_conv(x_encoded)

        flow_refined = out[:, :2]
        certainty = out[:, 2:3] 
        return flow_refined, certainty



class GLAM(nn.Module):
    def __init__(self, reduce_channels=512):
        super().__init__()
        self.reducer_16_1 = ChannelReducer(in_channels=1024, out_channels=reduce_channels)
        self.reducer_32_1 = ChannelReducer(in_channels=2048, out_channels=reduce_channels)
        self.reducer_16_2 = ChannelReducer(in_channels=1024, out_channels=reduce_channels)
        self.reducer_32_2 = ChannelReducer(in_channels=2048, out_channels=reduce_channels)
 
        coarse_in_channels = reduce_channels * 4
        self.coarse_matcher = CoarseMatcher(in_channels=coarse_in_channels)

        self.fine_matcher_8 = TFMatcher(in_channels=512)
        self.fine_matcher_4 = TFMatcher(in_channels=256)

        self.fine_matcher_2 = RefineMatcher(in_channels=64, use_certainty=False, concat_logits=False)
        self.fine_matcher_1 = RefineMatcher(in_channels=3, use_certainty=True, concat_logits=True)

    def forward(self, feats1, feats2):

        f1_16 = self.reducer_16_1(feats1['16'])
        f1_32 = self.reducer_32_1(feats1['32'])
        f2_16 = self.reducer_16_2(feats2['16'])
        f2_32 = self.reducer_32_2(feats2['32'])
        f1_32 = F.interpolate(f1_32, size=f1_16.shape[2:], mode='bilinear', align_corners=False)
        f2_32 = F.interpolate(f2_32, size=f2_16.shape[2:], mode='bilinear', align_corners=False)

        feat1_coarse = torch.cat([f1_16, f1_32], dim=1)
        feat2_coarse = torch.cat([f2_16, f2_32], dim=1)
        theta = self.coarse_matcher(feat1_coarse, feat2_coarse)  # (B,2,3)

        B, _, H8, W8 = feats1['8'].shape
        theta_upsampled_8 = theta
        flow_8 = theta_to_flow(theta, H=512,W=512)
        delta_theta_8 , pred_flow_8 = self.fine_matcher_8(feats1['8'], feats2['8'], theta_upsampled_8)
        delta_flow_8 = theta_to_flow(delta_theta_8, H=H8, W=W8)
        theta_updated_8 = compose_theta(theta_upsampled_8, delta_theta_8)

        B, _, H4, W4 = feats1['4'].shape
        theta_upsampled_4 = theta_updated_8
        flow_4 = theta_to_flow(theta_upsampled_4, H=512, W=512)
        delta_theta_4, pred_flow_4 = self.fine_matcher_4(feats1['4'], feats2['4'], theta_upsampled_4)
        delta_flow_4 = theta_to_flow(delta_theta_4, H=H4, W=W4)
        theta_updated_4 = compose_theta(theta_upsampled_4, delta_theta_4)

        B, _, H2, W2 = feats1['2'].shape
        theta_upsampled_2 = theta_updated_4  
        flow_2 = theta_to_flow(theta_upsampled_2, H=256, W=256)
        delta_flow_2,cer2 = self.fine_matcher_2(feats1['2'], feats2['2'], flow_2)
        computed_flow_2 = delta_flow_2 + flow_2

        flow_2 = F.interpolate(flow_2, size=(512, 512), mode='bilinear', align_corners=False)
        delta_flow_2 = F.interpolate(delta_flow_2, size=(512, 512), mode='bilinear', align_corners=False)
        computed_flow_2 = F.interpolate(computed_flow_2, size=(512, 512), mode='bilinear', align_corners=False)
        cer2 = F.interpolate(cer2, size=(512, 512), mode='bilinear', align_corners=False)

        B, _, H1, W1 = feats1['1'].shape
        delta_flow_1, cer = self.fine_matcher_1(feats1['1'], feats2['1'], computed_flow_2, logits=cer2)
        fine_flow = delta_flow_1 + computed_flow_2
        cer = cer2 + cer


        return {
            'coarse_theta': theta,
            'flow_8_pred': pred_flow_8,
            'flow_8_delta': delta_flow_8,
            'flow_4_pred': pred_flow_4,
            'flow_4_delta': delta_flow_4,
            'flow_2_delta': delta_flow_2,
            'flow_1_delta': delta_flow_1,
            
            'flow_final': fine_flow,
            'certainty': cer,
            # 'valid_mask': valid_mask
            'flow_8': flow_8,
            'flow_4': flow_4,
            'flow_2': flow_2,
            'flow_1': computed_flow_2
        }
