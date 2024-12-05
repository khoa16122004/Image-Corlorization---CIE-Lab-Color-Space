import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from skimage import color
from skimage.io import imsave
import os
import argparse
from dataset import get_dataset
from arch import get_architecture
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale


def asign_centroid_color_batch(bins_pred, ab_bins, grid_size=12):
    a_bins, b_bins = ab_bins  # start points of bins

    bins_pred = bins_pred.squeeze(1)  # batch x W x H
    a_idx = bins_pred % (a_bins.shape[0] - 1)
    b_idx = bins_pred // (b_bins.shape[0] - 1)

    a_start = a_bins[a_idx]
    b_start = b_bins[b_idx]

    a = a_start + grid_size / 2
    b = b_start + grid_size / 2

    ab = torch.stack([a, b], dim=1)  # batch x 2 x W x H
    return ab # not normalize

def calculate_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    return structural_similarity(img1, img2,  win_size=3, channel_axis=2, data_range=255)


def test_recontruct_with_metrics(args, model, test_loader):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0

    with torch.no_grad():
        for img, _ in tqdm(test_loader):
            img = img.cuda()
            L, ab = (img[:, 0:1, :, :] / 100).cuda(), (img[:, 1:, :, :] / 128).cuda()
            ab_pred = model(L) * 128
            ab = ab * 128
            ab = torch.clamp(ab, min=-110, max=110) 
            ab_pred = torch.clamp(ab_pred, min=-110, max=110) 
            print(ab_pred.min(), ab_pred.max())
            print(ab.min(), ab.max())
  
            
            # Chuyển đổi kết quả từ mô hình
            lab_pred = torch.cat([L * 100, ab_pred], dim=1).cpu().numpy()
            lab_gt = torch.cat([L * 100, ab], dim=1).cpu().numpy()

            for i in range(img.size(0)):
                rgb_gt = color.lab2rgb(lab_gt[i].transpose(1, 2, 0))
                rgb_pred = color.lab2rgb(lab_pred[i].transpose(1, 2, 0))
                
                rgb_gt_uint8 = (rgb_gt * 255).astype(np.uint8)
                rgb_pred_uint8 = (rgb_pred * 255).astype(np.uint8)
                
                psnr = calculate_psnr(rgb_gt_uint8, rgb_pred_uint8)
                ssim = calculate_ssim(rgb_gt_uint8, rgb_pred_uint8)
                
                total_psnr += psnr
                total_ssim += ssim
                num_images += 1

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")


def create_ab_bins(grid_size=12):
    a_bins = torch.arange(-110, 110 + grid_size, grid_size) 
    b_bins = torch.arange(-110, 110 + grid_size, grid_size) 
    
    return a_bins.cuda(), b_bins.cuda()
def encode_ab(ab, ab_bins):  # ab: Batch x 2 x 96 x 96
    a_bins, b_bins = ab_bins
    batch_size, _, height, width = ab.shape

    a, b = ab[:, 0].contiguous() * 128, ab[:, 1].contiguous() * 128 # Batch x 96 x 96

    a_idx = torch.bucketize(a, a_bins) - 1  # Batch x 96 x 96
    b_idx = torch.bucketize(b, b_bins) - 1  # Batch x 96 x 96

    a_idx = torch.clamp(a_idx, min=0, max=360)  
    b_idx = torch.clamp(b_idx, min=0, max=360) 

    ab_target = b_idx * (len(a_bins) - 1) + a_idx  # Batch x 96 x 96

    return ab_target # Batch x 1 x 96 x 96
def test_classification_with_metrics(args, model, test_loader):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0

    ab_bins = create_ab_bins()
    with torch.no_grad():
        for img, _ in tqdm(test_loader):
            img = img.cuda()
            L, ab = (img[:, 0:1, :, :] / 100).cuda(), (img[:, 1:, :, :] / 128).cuda()
            ab_logits = model(L)
            bin_preds = torch.argmax(ab_logits, dim=1)
            ab_pred = asign_centroid_color_batch(bin_preds, ab_bins)

            ab = torch.clamp(ab, min=-110, max=110) 
            ab_pred = torch.clamp(ab_pred * 128, min=-110, max=110) 
            
            lab_pred = torch.cat([L * 100, ab_pred], dim=1).cpu().numpy()
            lab_gt = torch.cat([L * 100, ab], dim=1).cpu().numpy()

            for i in range(img.size(0)):
                rgb_gt = color.lab2rgb(lab_gt[i].transpose(1, 2, 0))
                rgb_pred = color.lab2rgb(lab_pred[i].transpose(1, 2, 0))
                
                rgb_gt_uint8 = (rgb_gt * 255).astype(np.uint8)
                rgb_pred_uint8 = (rgb_pred * 255).astype(np.uint8)
                
                psnr = calculate_psnr(rgb_gt_uint8, rgb_pred_uint8)
                ssim = calculate_ssim(rgb_gt_uint8, rgb_pred_uint8)
                
                total_psnr += psnr
                total_ssim += ssim
                num_images += 1

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")



def main():
    parser = argparse.ArgumentParser(description="Train Colorization Model")
    
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate cho Adam')
    parser.add_argument('--step_size', type=int, default=10, help='Step size cho scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma cho scheduler')
    parser.add_argument('--colorizer', type=str, default="simple_CNN")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--objective', type=str, default='reconstruction')
    parser.add_argument('--arch', type=str, default='simple_CNN')
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--dataset', type=str, default='stl10')
    args = parser.parse_args()

    model = get_architecture(args.arch, args.objective).cuda()
    
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    
    train_dataset, test_dataset = get_dataset(args.dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.objective == "reconstruction":
        test_recontruct_with_metrics(args, model, test_loader)

    elif args.objective == "classification":
        test_classification_with_metrics(args, model, test_loader)
    

    
if __name__ == "__main__":
    main()

