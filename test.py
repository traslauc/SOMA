import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.encoder import Encoder
from models.matcher import GLAM, warp_image
from models.loss import flow_loss
from datasets.dataloader import SODataset
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchsummary import summary


@torch.no_grad()
def test_model(encoder, matcher, dataloader, device):
    encoder.eval()
    matcher.eval()

    total_rmse = 0
    num_batches = 0
    total_images = 0
    total_valid_pixels = 0
    total_valid_error = 0

    all_pixel_errors = []
    all_image_rmses = []
    image_rmse_below_5 = []
    image_rmse_below_4 = []
    image_rmse_below_3 = []
    image_rmse_below_2 = []
    image_rmse_below_1 = []

    os.makedirs("vis/samples", exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
        sar = batch['sar_transformed'].to(device)
        sar_ori = batch['sar_original'].to(device)
        opt = batch['opt_original'].to(device)
        gt_flow = batch['warp'].to(device).permute(0, 3, 1, 2)

        feats_sar = encoder(sar, label='sar')['sar']
        feats_opt = encoder(opt, label='opt')['opt']
        output = matcher(feats_sar, feats_opt)

        pred_flow = output['flow_final']
        certainty = output['certainty']

        pred_flow_up = F.interpolate(pred_flow, size=gt_flow.shape[2:], mode='bilinear', align_corners=False)
        rmse_batch = flow_loss(pred_flow_up, gt_flow)['rmse']
        total_rmse += rmse_batch.item()
        num_batches += 1

        for i in range(sar.size(0)):
            error_map = torch.norm(pred_flow_up[i] - gt_flow[i], dim=0)  # [H, W]
            diff = pred_flow_up[i] - gt_flow[i]
            error_squared_map = torch.sum(diff**2, dim=0)   # [H, W]
            mse = error_squared_map.mean()
            rmse = torch.sqrt(mse)
            image_rmse = rmse.item()

            all_image_rmses.append(image_rmse)

            if image_rmse < 5:
                image_rmse_below_5.append(image_rmse)
            if image_rmse < 4:
                image_rmse_below_4.append(image_rmse)
            if image_rmse < 3:
                image_rmse_below_3.append(image_rmse)
            if image_rmse < 2:
                image_rmse_below_2.append(image_rmse)
            if image_rmse < 1:
                image_rmse_below_1.append(image_rmse)

            valid_mask = error_map < 2
            total_valid_pixels += valid_mask.sum().item()
            total_valid_error += error_map[valid_mask].sum().item()
            all_pixel_errors.append(error_map.view(-1))

            if (total_images + i) % 10 == 0:
                save_visualization_local(
                    warp_image(opt[i].unsqueeze(0), gt_flow[i].unsqueeze(0))[0],
                    warp_image(sar_ori[i].unsqueeze(0), pred_flow_up[i].unsqueeze(0))[0],
                    certainty[i, 0],
                    error_map,
                    image_rmse,
                    total_images + i
                )
        total_images += sar.size(0)

    plt.figure()
    plt.hist(all_image_rmses, bins=30, color='dodgerblue', edgecolor='black')
    plt.title("Image-wise RMSE Distribution")
    plt.xlabel("RMSE")
    plt.ylabel("Number of Image Pairs")
    plt.grid(True)
    plt.savefig("vis/rmse_hist.png")
    plt.close()

    avg_rmse = total_rmse / num_batches
    ratio_below_5 = len(image_rmse_below_5) / total_images
    ratio_below_4 = len(image_rmse_below_4) / total_images
    ratio_below_3 = len(image_rmse_below_3) / total_images
    ratio_below_2 = len(image_rmse_below_2) / total_images
    ratio_below_1 = len(image_rmse_below_1) / total_images

    avg_rmse_below_5 = np.mean(image_rmse_below_5) if image_rmse_below_5 else 0
    all_pixel_errors = torch.cat(all_pixel_errors).cpu().numpy()
    avg_valid_pixels = total_valid_pixels / total_images
    avg_valid_error = total_valid_error / total_valid_pixels if total_valid_pixels > 0 else 0

    print(f"\n Average RMSE: {avg_rmse:.4f}")
    print(f"RMSE < 5 Image Pairs: {ratio_below_5:.2%}")
    print(f"RMSE < 4 Image Pairs: {ratio_below_4:.2%}")
    print(f"RMSE < 3 Image Pairs: {ratio_below_3:.2%}")
    print(f"RMSE < 2 Image Pairs: {ratio_below_2:.2%}")
    print(f"RMSE < 1 Image Pairs: {ratio_below_1:.2%}")

    print(f"Avg RMSE of Image Pairs with RMSE < 5: {avg_rmse_below_5:.4f}")
    print(f"Avg #Pixels with Error < 2: {avg_valid_pixels:.2f}")
    print(f"Avg Error of Pixels < 2: {avg_valid_error:.4f}")
    print(f"Histogram saved to vis/rmse_hist.png")

    return avg_rmse


def save_visualization_local(sar, warped_sar, certainty, error_map, rmse_value, index, save_dir="vis/samples"):
    def to_numpy(img):
        img = img.detach().cpu()
        if img.dim() == 3:
            img = img.permute(1, 2, 0)
        img = img.numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        return img

    sar_np = to_numpy(sar)
    warped_np = to_numpy(warped_sar)
    cert_np = to_numpy(certainty)
    err_np = to_numpy(error_map)

    cert_color = cv2.applyColorMap(cert_np, cv2.COLORMAP_JET)
    err_color = cv2.applyColorMap(err_np, cv2.COLORMAP_JET)

    combined = cv2.hconcat([sar_np, warped_np, cert_color, err_color])
    cv2.putText(combined, f"RMSE: {rmse_value:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"sample_{index:03d}.png"), combined)

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor()])
    test_set = SODataset(root=args.data_root, split='test', transform=transform, apply_transform=True ,dataset_folder=['SEN1-2','WHU-SEN-City','OSdataset'])
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    encoder = Encoder().to(device)
    matcher = GLAM().to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        matcher.load_state_dict(ckpt['matcher'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    test_model(encoder, matcher, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='sopatch', help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model.pth')
    args = parser.parse_args()
    main(args)
