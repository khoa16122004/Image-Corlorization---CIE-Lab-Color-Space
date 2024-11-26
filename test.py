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
# from train import save_colored_sample



def save_colored_sample(L, ab_pred, classification=False):
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

    imsave(f'colorize.png', combined)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Train Colorization Model")
    
    parser.add_argument('--checkpoint', type=str, default='samples/best_model_recontruction.pth')
    parser.add_argument('--colorizer', type=str, default="simple_CNN")
    parser.add_argument('--objective', type=str, default='reconstruction')
    parser.add_argument('--img_path', type=str)
    args = parser.parse_args()


    model = get_architecture("simple_CNN", args.objective).cuda()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    img = Image.open(args.img_path).convert('L').resize((96, 96))  # 255 space  
    img_ = transforms.ToTensor()(img).unsqueeze(0).cuda() # 0,1
    ab_pred = model(img_)
    if args.objective == 'reconstruction':
        save_colored_sample(img_, ab_pred )
    elif args.objective == 'classification':
        pass
        
if __name__ == "__main__":
    main()