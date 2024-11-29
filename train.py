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
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt


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


def create_ab_bins(grid_size=12):
    a_bins = torch.arange(-110, 110 + grid_size, grid_size) 
    b_bins = torch.arange(-110, 110 + grid_size, grid_size) 
    
    return a_bins.cuda(), b_bins.cuda()


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


def save_colored_sample(args, L, ab, ab_pred, epoch, classification=False):
    
    L_single = L[0][0].cpu().numpy() * 100  # 1 x 96 x 96
    ab_pred_single = ab_pred[0].cpu().numpy() * 128  # 2 x 96 x 96
    ab_single = ab[0].cpu().numpy() * 128  # 2 x 96 x 96

    if classification == True:
        ab_pred_single /= 128
        ab_single / 128
        
    lab_gt = np.array([L_single, ab_single[0], ab_single[1]])  # 3 x 96 x 96
    lab_pred = np.array([L_single, ab_pred_single[0], ab_pred_single[1]])  # 3 x 96 x 96
    gray_scale = np.array([L_single, np.zeros_like(ab_pred_single[0]),  np.zeros_like(ab_pred_single[1])])  # 3 x 96 x 96
    
    rgb_gt = color.lab2rgb(np.transpose(lab_gt, (1, 2, 0)))  
    rgb_pred = color.lab2rgb(np.transpose(lab_pred, (1, 2, 0)))
    gray_scale = color.lab2rgb(np.transpose(gray_scale, (1, 2, 0)))

    rgb_gt = (rgb_gt * 255).astype(np.uint8)
    rgb_pred = (rgb_pred * 255).astype(np.uint8)
    gray_scale = (gray_scale * 255).astype(np.uint8)

    combined = np.hstack((rgb_gt, gray_scale, rgb_pred))  # Ghép theo chiều ngang

    imsave(os.path.join(args.outdir, f"combined_epoch{epoch}.png"), combined)
    # groutruth, scale, rgb


def train_RGB_reconstruction_objective(args, model, train_loader, test_loader, epochs=20, lr=1e-3, step_size=10, gamma=0.1):
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()

    best_loss = None
    for epoch in range(epochs): 
        model.train()
        epoch_loss = 0.0
        for img, _ in tqdm(train_loader): # img: batch x 3 x w x h
            img = img.cuda()
            img_gray = rgb_to_grayscale(img * 255) / 255
            img_pred = model(img_gray)
            loss = criterion(img_pred, img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if not best_loss or epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(args.outdir, 'best_model.pth'))
        
        # if epoch % 5 != 0:
        #     continue
        with torch.no_grad():
            model.eval()
            for img, _ in test_loader:
                img = img.cuda()
                img_gray = rgb_to_grayscale(img * 255) / 255
                img_pred = model(img_gray)
                save_image(img_gray, os.path.join(args.outdir, f"gray.png"))
                save_image(img_pred, os.path.join(args.outdir, f"colorize.png"))
                break  
        
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader)}")
        scheduler.step()

def train_reconstruction_objective(args, model, train_loader, test_loader, epochs=20, lr=1e-3, step_size=10, gamma=0.1):
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()

    best_loss = None
    for epoch in range(epochs): 
        model.train()
        epoch_loss = 0.0
        for img, _ in tqdm(train_loader): # img: batch x 3 x w x h
            # L: 0-> 1, ab: -1 -> 1
            L, ab = (img[:, 0:1, :, :] / 100).cuda(), (img[:, 1:, :, :] / 128).cuda()
            
            optimizer.zero_grad()
            ab_pred = model(L)
            loss = criterion(ab_pred, ab)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # break
        
        if not best_loss or epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(args.outdir, 'best_model.pth'))
        
        if epoch % 5 != 0:
            continue
        with torch.no_grad():
            model.eval()
            for img, _ in test_loader:
                L, ab = (img[:, 0:1, :, :] / 100).cuda(), (img[:, 1:, :, :] / 128).cuda()
                ab_pred = model(L)
                save_colored_sample(args, L, ab, ab_pred, epoch)
                break  
        
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader)}")
        scheduler.step()


def train_classification_objective(args, model, train_loader, test_loader, epochs=20, lr=1e-3, step_size=10, gamma=0.1):
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    ab_bins = create_ab_bins()
    best_loss = None
    
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0.0
        for img, _ in tqdm(train_loader):
            L, ab = (img[:, 0:1, :, :] / 100).cuda(), (img[:, 1:, :, :] / 128).cuda() # L, ab : batch x 3 x 96 x 96
            ab_target = encode_ab(ab, ab_bins) # batch x W x H
            optimizer.zero_grad()
            ab_logits = model(L) # batch x 361 x W x H
            loss = F.cross_entropy(ab_logits, ab_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # break
        
        if not best_loss or epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(args.outdir,"best_model.pth"))
        
        if epoch % 1 != 0:
            continue
        with torch.no_grad():
            model.eval()
            for img, _ in test_loader:
                L, ab = (img[:, 0:1, :, :] / 100).cuda(), (img[:, 1:, :, :] / 128).cuda()
                ab_target = encode_ab(ab, ab_bins) # batch x W x H
                ab_logits = model(L)
                bin_preds = torch.argmax(ab_logits, dim=1)
                # print(bin_preds)
                
                ab_preds = asign_centroid_color_batch(bin_preds, ab_bins)
                save_colored_sample(args, L, ab, ab_preds, epoch, True)
                break  
        
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader)}")
        scheduler.step()

def train_wgan_objective(args, generator, discriminator, train_loader, test_loader):
    betas = (0.5, 0.999)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=betas)
    generator = generator.to('cuda')
    discriminator = discriminator.to('cuda')
    metrics = {'train_g_loss': [], 'train_d_loss': []}
    best_g_loss, best_d_loss = None, None
    ab_bins = create_ab_bins()

    def train_step (batch, ab_bins):
        # TRAIN DISCRIMINATOR
        for _ in range(args.n_critic):  
            img, _ = batch
            L, ab = (img[:, 0:1, :, :] / 100).to('cuda'), (img[:, 1:, :, :] / 128).to('cuda')
            ab_encoded = encode_ab(ab, ab_bins).unsqueeze(1).float()
            
            with torch.no_grad():
                fake_images_probs = generator(L)
                
            if args.objective == "classification":
                # print(ab_encoded.shape)
                real_validity = discriminator(ab_encoded) # B x 361 x W x H
                fake_images = torch.argmax(fake_images_probs, dim=1, keepdim=True).float()
            else: 
                real_validity = discriminator(ab) # B x 2 x W x H
                fake_images = fake_images_probs
                
            fake_validity = discriminator(fake_images)
    
            optimizer_D.zero_grad()
    
            # Gradient penalty
            alpha = torch.rand(ab_encoded.size(0), 1, 1, 1).to('cuda')
            interpolated = alpha * ab_encoded + (1 - alpha) * fake_images
            interpolated.requires_grad_(True)
            
            interpolated_validity = discriminator(interpolated)
            
            gradients = torch.autograd.grad(
                outputs=interpolated_validity,
                inputs=interpolated,
                grad_outputs=torch.ones_like(interpolated_validity).to('cuda'),
                create_graph=True,
                retain_graph=True, 
            )[0]
            
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + args.lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
    
            optimizer_G.zero_grad()

        # TRAIN GENERATOR
        fake_images_probs = generator(L)
        if args.objective == "classification":
            fake_images = torch.argmax(fake_images_probs, dim=1, keepdim=True).float()
        if args.objective == "reconstruction":
            fake_images = fake_images_probs

        fake_validity = discriminator(fake_images)
        g_loss = -torch.mean(fake_validity)

        g_loss.backward()
        optimizer_G.step()
        return g_loss.item(), d_loss.item()
    
    def train_epoch(train_loader, ab_bins):
        generator.train()
        discriminator.train()
        total_g_loss, total_d_loss = 0, 0

        for batch in train_loader:
            g_loss, d_loss = train_step(batch, ab_bins)
            total_g_loss += g_loss
            total_d_loss += d_loss
        
        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)
        
        metrics['train_g_loss'].append(avg_g_loss)
        metrics['train_d_loss'].append(avg_d_loss)
        
        return avg_g_loss, avg_d_loss
    
    def plot_metrics():        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(metrics['train_g_loss'], label='Train Generator Loss')
        plt.title('Generator Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(metrics['train_d_loss'], label='Train Discriminator Loss')
        plt.title('Discriminator Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(args.outdir, 'lossWGAN.png'))

    for ep in tqdm(range(args.epochs)):
        g_loss, d_loss = train_epoch(train_loader = train_loader, ab_bins = ab_bins)
        if not best_g_loss or g_loss < best_g_loss:
                best_g_loss = g_loss
                torch.save(generator.state_dict(), os.path.join(args.outdir,"best_G.pth"))
        if not best_d_loss or d_loss < best_d_loss:
                best_d_loss = d_loss
                torch.save(discriminator.state_dict(), os.path.join(args.outdir,"best_D.pth"))
       
        with open(os.path.join(args.outdir, 'lossWGAN.txt'), 'a') as f:
            f.write(f"Epoch {ep}, G Loss: {g_loss}\t|\t D Loss {d_loss}\n")
       
        if ep % 5 != 0:
            continue
        with torch.no_grad():
            generator.eval()
            discriminator.eval()
            for i,batch in enumerate(test_loader):
                img, _ = batch
                L, ab = (img[:, 0:1, :, :] / 100).to('cuda'), (img[:, 1:, :, :] / 128).to('cuda')
                # ab_encoded = encode_ab(ab, ab_bins).unsqueeze(1).float()
        
                # Generate fake images
                fake_images_probs = generator(L)
        
                if args.objective == "classification":
                    # real_validity = discriminator(ab_encoded)  # B x 361 x W x H
                    fake_images = torch.argmax(fake_images_probs, dim=1, keepdim=True).float()
                    ab_preds = asign_centroid_color_batch(fake_images, ab_bins)
                    save_colored_sample(args, L, ab, ab_preds, ep, True)
                else: 
                    # real_validity = discriminator(ab)  # B x 2 x W x H
                    fake_images = fake_images_probs
                    save_colored_sample(args, L, ab, fake_images, ep)
        
    plot_metrics()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Colorization Model")
    
    parser.add_argument('--epochs', type=int, default=100, help='Số epoch để huấn luyện')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint-G', type=str, default=None, help='Checkpoint của generator')
    parser.add_argument('--checkpoint-D', type=str, default=None, help='Checkpoint của discriminator')
    parser.add_argument('--lambda_gp', type=float, default=10, help='Lambda cho gradient penalty')
    parser.add_argument('--n_critic', type=int, default=5, help='Số lần huấn luyện discriminator trước khi huấn luyện generator')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate cho Adam')
    parser.add_argument('--step_size', type=int, default=10, help='Step size cho scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma cho scheduler')
    parser.add_argument('--colorizer', type=str, default="simple_CNN")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--objective', type=str, default='recontruction')
    parser.add_argument('--arch', type=str, default='simple_CNN')
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--dataset', type=str, default='stl10')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.arch == "wgan":
        generator, discriminator = get_architecture(args.arch, args.objective)
    else:
        model = get_architecture(args.arch, args.objective)
    
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    if args.checkpoint_G:
        generator.load_state_dict(torch.load(args.checkpoint_G))
    if args.checkpoint_D:
        discriminator.load_state_dict(torch.load(args.checkpoint_D))
    
    train_dataset, test_dataset = get_dataset(args.dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.arch == "wgan":
        train_wgan_objective(args, generator, discriminator, train_loader, test_loader)
    else:
        if args.objective == "recontruction":
                train_reconstruction_objective(args, model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, step_size=args.step_size, gamma=args.gamma)

        elif args.objective == "classification":
            train_classification_objective(args, model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, step_size=args.step_size, gamma=args.gamma)
        
        elif args.objective == "upscale":
            train_RGB_reconstruction_objective(args, model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, step_size=args.step_size, gamma=args.gamma)

    
if __name__ == "__main__":
    main()
