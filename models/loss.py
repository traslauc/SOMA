import torch 
import torch.nn as nn
import torch.nn.functional as F


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

