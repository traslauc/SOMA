import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2

class SODataset(Dataset):
    def __init__(self, root, split='train', transform=None, apply_transform=True,dataset_folder=None):
        self.transform = transform
        self.apply_transform = apply_transform
        self.pairs = []

        if dataset_folder is None:
            dataset_folders = ['OSdataset', 'WHU-SEN-City', 'SEN1-2']
        else:
            dataset_folders = dataset_folder
            
        for ds in dataset_folders:
            sar_dir = os.path.join(root, ds, split, 'sar')
            opt_dir = os.path.join(root, ds, split, 'opt')
      
            if not os.path.isdir(sar_dir) or not os.path.isdir(opt_dir):
                continue

            if ds == 'GFGE':
                sar_files = sorted([f for f in os.listdir(sar_dir) if f.startswith('sar')])
                opt_files = sorted([f for f in os.listdir(opt_dir) if f.startswith('opt')])

                for opt_file in opt_files:
                    idx = opt_file.replace('opt', '').replace('.png', '')
                    sar_file = f'sar{idx}.png'
                    sar_path = os.path.join(sar_dir, sar_file)
                    opt_path = os.path.join(opt_dir, opt_file)

                    if os.path.isfile(sar_path) and os.path.isfile(opt_path):
                        self.pairs.append((sar_path, opt_path))

            for filename in sorted(os.listdir(sar_dir)):
                sar_path = os.path.join(sar_dir, filename)
                opt_path = os.path.join(opt_dir, filename)
                if os.path.isfile(sar_path) and os.path.isfile(opt_path):
                    self.pairs.append((sar_path, opt_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        sar_path, opt_path = self.pairs[index]
        sar_original = Image.open(sar_path).convert('RGB')
        opt_original = Image.open(opt_path).convert('RGB')

        warp_map = None
        sar_transformed = sar_original
        if self.apply_transform:
            sar_transformed, warp_map , M= self.apply_geometric_transform(sar_original)

        if self.transform:
            sar_original = self.transform(sar_original)
            opt_original = self.transform(opt_original)
            sar_transformed = self.transform(sar_transformed)

        out = {
            'sar_original': sar_original,
            'opt_original': opt_original,
            'sar_transformed': sar_transformed,
        }
        if self.apply_transform and warp_map is not None:
            out['warp'] = warp_map
            out['M'] = M
        return out


    def apply_geometric_transform(self, img):
        img_np = np.array(img)
        h, w, _ = img_np.shape

        # Generate random affine transformation parameters, including rotation, translation, and scaling.
        # You can adjust the range of the parameters.
        angle = np.random.uniform(-20, 20)
        tx = np.random.uniform(-50, 50)
        ty = np.random.uniform(-50, 50)
        scale = np.random.uniform(1.0, 1.0)

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[:, 2] += [tx, ty]

        flow = self.compute_flow_map(h, w, M)
        transformed_img = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_LINEAR)

        return Image.fromarray(transformed_img), flow , M

    def compute_flow_map(self, h, w, M):
        M_inv = cv2.invertAffineTransform(M)
        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)  # (H, W, 2)
        coords_homo = np.concatenate([coords, np.ones((*coords.shape[:2], 1))], axis=-1)
        new_coords = np.einsum('ij,hwj->hwi', M_inv, coords_homo)
        flow = new_coords - coords.astype(np.float32)
       
        return torch.tensor(flow, dtype=torch.float32)