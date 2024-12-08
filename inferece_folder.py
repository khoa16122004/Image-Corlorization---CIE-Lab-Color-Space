import torch
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
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def create_ab_bins(grid_size=12):
    a_bins = torch.arange(-110, 110 + grid_size, grid_size) 
    b_bins = torch.arange(-110, 110 + grid_size, grid_size) 
    
    return a_bins.cuda(), b_bins.cuda()

def save_colored_sample(L, ab_pred, output_path, classification=False):
    L_single = L[0][0].cpu().numpy() * 100  # 1 x 96 x 96
    ab_pred_single = ab_pred[0].cpu().numpy() * 128  # 2 x 96 x 96

    if classification == True:
        ab_pred_single /= 128
        
    lab_pred = np.array([L_single, ab_pred_single[0], ab_pred_single[1]])  # 3 x 96 x 96
    gray_scale = np.array([L_single, np.zeros_like(ab_pred_single[0]),  np.zeros_like(ab_pred_single[1])])  # 3 x 96 x 96
    
    rgb_pred = color.lab2rgb(np.transpose(lab_pred, (1, 2, 0)))
    gray_scale = color.lab2rgb(np.transpose(gray_scale, (1, 2, 0)))

    rgb_pred = (rgb_pred * 255).astype(np.uint8)
    gray_scale = (gray_scale * 255).astype(np.uint8)

    combined = np.hstack((gray_scale, rgb_pred))  # Ghép theo chiều ngang

    imsave(output_path, combined)

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

def asign_nearly_color_batch(ab_logits, ab_bins, T=0.1, k=5, grid_size=12):
    a_bins, b_bins = ab_bins
    
    scaled_logits = ab_logits / T
    prob = F.softmax(scaled_logits, dim=1)
    
    topk_probs, topk_indices = torch.topk(prob, k=k, dim=1)
    
    a_coords = (topk_indices % (a_bins.shape[0] - 1)).float()
    b_coords = (topk_indices // (b_bins.shape[0] - 1)).float()
    
    a_pred = torch.sum(topk_probs * (a_bins[a_coords.long()] + grid_size / 2), dim=1)
    b_pred = torch.sum(topk_probs * (b_bins[b_coords.long()] + grid_size / 2), dim=1)
    
    return torch.stack([a_pred, b_pred], dim=1)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Batch Colorization Inference")
    
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the model checkpoint')
    parser.add_argument('--colorizer', type=str, default="simple_CNN")
    parser.add_argument('--objective', type=str, default='reconstruction')
    parser.add_argument('--input_folder', type=str, required=True, 
                        help='Folder containing input grayscale images')
    parser.add_argument('--output_folder', type=str, required=True, 
                        help='Folder to save colorized images')
    parser.add_argument('--method', type=str, 
                        choices=['centroid', 'nearly'], 
                        help='Colorization method')
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Load model
    model = get_architecture("simple_CNN", args.objective).cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # Prepare bins for classification methods
    ab_bins = create_ab_bins() if args.objective == 'classification' else None

    # Process images
    for filename in tqdm(os.listdir(args.input_folder)):
        # Check if file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Full path to input and output images
            input_path = os.path.join(args.input_folder, filename)
            output_filename = f"colorized_{filename}"
            output_path = os.path.join(args.output_folder, output_filename)

            # Open and preprocess image
            img = Image.open(input_path).convert('L').resize((96, 96))
            img_ = transforms.ToTensor()(img).unsqueeze(0).cuda()

            # Colorize based on method
            if args.objective == 'reconstruction':
                ab_pred = model(img_)
                save_colored_sample(img_, ab_pred, output_path)
            elif args.objective == 'classification':
                ab_logits = model(img_)
                

                if args.method == 'centroid':
                    bin_preds = torch.argmax(ab_logits, dim=1)
                    ab_preds = asign_centroid_color_batch(bin_preds, ab_bins)
                    save_colored_sample(img_, ab_preds, output_path, True)
                elif args.method == 'nearly':
                    ab_preds = asign_nearly_color_batch(ab_logits, ab_bins)
                    save_colored_sample(img_, ab_preds, output_path, True)

    print(f"Colorization complete. Images saved in {args.output_folder}")
        
if __name__ == "__main__":
    main()