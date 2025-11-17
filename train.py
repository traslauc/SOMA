import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import time

from models.encoder import Encoder
from models.matcher import GLAM, warp_image
from models.loss import flow_loss, rb_loss, uni_loss
from datasets.dataloader import SODataset


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.backends.cudnn.benchmark = True

def evaluate(encoder, matcher, dataloader, device):
    encoder.eval()
    matcher.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            sar = batch['sar_transformed'].to(device)
            opt = batch['opt_original'].to(device)
            gt_flow = batch['warp'].to(device).permute(0, 3, 1, 2)

            feats_sar = encoder(sar, 'sar')['sar']
            feats_opt = encoder(opt, 'opt')['opt']
            output = matcher(feats_sar, feats_opt)
            pred_flow = output['flow_final']
            pred_flow_up = F.interpolate(pred_flow, size=gt_flow.shape[2:], mode='bilinear', align_corners=False)
            loss = flow_loss(pred_flow_up, gt_flow)
            loss = loss['rmse']
            total_loss += loss.item()
    return total_loss / len(dataloader)

def sinkhorn_wasserstein(f1, f2):
    return F.mse_loss(f1, f2)

def train(config):
    wandb.init(project="somatch-v2", config=config,mode="disabled")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using GPUs:", torch.cuda.device_count())

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("Using TF32:", torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32)
    print("Using CUDNN Benchmark:", torch.backends.cudnn.benchmark)

    start_epoch = 0
    best_val_loss = float('inf')
    patience = 0
    checkpoint_path = 'checkpoints/backup.pth'
    encoder = torch.nn.DataParallel(Encoder()).to(device)
    matcher = torch.nn.DataParallel(GLAM()).to(device)

    optimizer = torch.optim.AdamW([
        {'params': matcher.parameters(), 'lr': config['lr']},
        {'params': encoder.parameters(), 'lr': config['lr'] * 0.5},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler()

    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    warmup_epochs = config.get('warmup_epochs', 0)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        encoder.module.load_state_dict(checkpoint['encoder'])
        matcher.module.load_state_dict(checkpoint['matcher'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        print(f"Resumed from epoch {start_epoch}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = SODataset(root=config['data_root'], split='train', transform=transform, apply_transform=True,dataset_folder=['SEN1-2'])
    val_set   = SODataset(root=config['data_root'], split='val',   transform=transform, apply_transform=True,dataset_folder=['SEN1-2'])
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    for epoch in range(start_epoch, config['epochs']):
        # —— Linear warm-up
        if epoch < warmup_epochs:
            factor = float(epoch + 1) / warmup_epochs
            for base_lr, pg in zip(base_lrs, optimizer.param_groups):
                pg['lr'] = base_lr * factor
        print(f"Epoch {epoch} — LR groups: {[pg['lr'] for pg in optimizer.param_groups]}")

        encoder.train()
        matcher.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(progress):
            # t0 = time.time()
            sar     = batch['sar_transformed'].to(device)
            sar_ori = batch['sar_original'].to(device)
            opt     = batch['opt_original'].to(device)
            gt_flow = batch['warp'].to(device).permute(0, 3, 1, 2)
            

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'): 
                feats_sar_out = encoder(sar, 'sar')
                feats_opt_out = encoder(opt, 'opt')

                feats_sar = feats_sar_out['sar']
                feats_opt = feats_opt_out['opt']
           
                output = matcher(feats_sar, feats_opt)
                pred_flow = output['flow_final']
                cer = output['certainty']
                flow_1 = output['flow_1']
                flow_2 = output['flow_2']
                flow_4 = output['flow_4']
                flow_8 = output['flow_8']

                pred_flow_4 = output['flow_4_pred']
                delta_flow_4 = output['flow_4_delta']

                pred_flow_8 = output['flow_8_pred']
                delta_flow_8 = output['flow_8_delta']

                delta_flow_2 = output['flow_2_delta']
                delta_flow_1 = output['flow_1_delta']

                tf_loss_8 = flow_loss(pred_flow_8, delta_flow_8)
                tf_loss_4 = flow_loss(pred_flow_4, delta_flow_4)
                tf_loss = tf_loss_4['rmse'] + tf_loss_8['rmse']

                dict = flow_loss(pred_flow, gt_flow ,certainty=cer)
                dict1 = rb_loss(flow_1, gt_flow)
                dict2 = rb_loss(flow_2, gt_flow)
                dict4 = rb_loss(flow_4, gt_flow)
                dict8 = rb_loss(flow_8, gt_flow)

                #delta 损失
                delta_flow_8 = F.interpolate(delta_flow_8, size=(512, 512), mode='bilinear', align_corners=False)
                delta_flow_4 = F.interpolate(delta_flow_4, size=(512, 512), mode='bilinear', align_corners=False)
               
                delta_supervision_8 = F.l1_loss(delta_flow_8, gt_flow - flow_8)
                delta_supervision_4 = F.l1_loss(delta_flow_4, gt_flow - flow_4)
                delta_supervision_2 = F.l1_loss(delta_flow_2, gt_flow - flow_2)
                delta_supervision_1 = F.l1_loss(delta_flow_1, gt_flow - flow_1)
                delta_supervision = 0.125 * delta_supervision_8 + 0.25 * delta_supervision_4 + 0.5 * delta_supervision_2 + delta_supervision_1


                warp_rmse = dict['rmse']
                warp_rmse_1 = dict1
                warp_rmse_2 = dict2
                warp_rmse_4 = dict4
                warp_rmse_8 = dict8
                scale_loss = 0.125 * warp_rmse_8 + 0.25 * warp_rmse_4 + 0.5 * warp_rmse_2 + 0.75 * warp_rmse_1
                warp_loss = rb_loss(pred_flow, gt_flow)

                uni_loss_val = uni_loss(pred_flow, gt_flow)
                certainty_loss = dict['certainty_loss']

                loss = warp_loss + 0.5 * tf_loss + 0.1 * uni_loss_val + 0.1 * delta_supervision + 0.1 * certainty_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(matcher.parameters()) + list(encoder.parameters()), config.get('grad_clip', 1.0))
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress.set_postfix({'warp': f"{warp_rmse.item():.4f}", 'total': f"{loss.item():.4f}"})
            wandb.log({
                'loss/rmse_8':dict8,
                'loss/rmse_4':dict4,
                'loss/rmse_2':dict2,
                'loss/rmse_1':dict1,
                'loss/rmse': dict['rmse'],
                'loss/total': loss,
                'loss/certainty': certainty_loss,
                'loss/uni_loss': uni_loss_val,
                'loss/tf_loss': tf_loss,
                'epoch': epoch,
                'lr_group0': optimizer.param_groups[0]['lr']
            })


        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(encoder, matcher, val_loader, device)

        with open('val_loss.txt', 'a') as f:
            f.write(f"Epoch {epoch} | Val Loss: {val_loss:.4f}\n")

        scheduler.step(val_loss)
        wandb.log({'val_loss': val_loss})

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        

        torch.save({
            'epoch': epoch,
            'encoder': encoder.module.state_dict(),
            'matcher': matcher.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience': patience
        }, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save({
                'encoder': encoder.module.state_dict(),
                'matcher': matcher.module.state_dict(),
            }, 'checkpoints/best_modelss_ex.pth')
            print("New best model saved.")

        else:
            patience += 1
            if patience >= config['early_stop']:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    config = {
        'data_root': 'sodata',
        'batch_size': 4,
        'lr': 5e-5,
        'epochs': 100,
        'early_stop': 15,
        'grad_clip': 1.0,
        'warmup_epochs': 5
    }
    train(config)
