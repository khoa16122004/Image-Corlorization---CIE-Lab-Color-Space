# import numpy as np
# from skimage.color import lab2rgb
# import matplotlib.pyplot as plt

# L = 50 
# bin_size = 12

# a_bins = np.arange(-110, 110 + bin_size, bin_size)
# b_bins = np.arange(-110, 110 + bin_size, bin_size)

# a_centroids = (a_bins[:-1] + a_bins[1:]) / 2
# b_centroids = (b_bins[:-1] + b_bins[1:]) / 2

# a_grid, b_grid = np.meshgrid(a_centroids, b_centroids)
# lab = np.zeros((a_grid.size, 3))
# lab[:, 0] = L
# lab[:, 1] = a_grid.ravel()
# lab[:, 2] = b_grid.ravel()

# rgb = lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)

# valid_mask = np.all((rgb >= 0) & (rgb <= 1), axis=1)
# valid_a = a_grid.ravel()[valid_mask]
# valid_b = b_grid.ravel()[valid_mask]
# valid_rgb = rgb[valid_mask]

# fig, ax = plt.subplots(figsize=(8, 8))

# # Duyệt qua từng ô và vẽ màu RGB cùng số chỉ mục
# for idx, (a, b, color) in enumerate(zip(valid_a, valid_b, valid_rgb)):
#     rect = plt.Rectangle((a - bin_size / 2, b - bin_size / 2), bin_size, bin_size, color=color)
#     ax.add_patch(rect)
#     ax.text(a, b, str(idx), color='black', ha='center', va='center', fontsize=8)  # Hiển thị chỉ số tại trung tâm

# ax.set_xlim(-120, 120)
# ax.set_ylim(-120, 120)
# ax.set_xticks(a_bins)
# ax.set_yticks(b_bins)
# plt.xlabel("a")
# plt.ylabel("b")
# plt.title("Màu RGB tại centroid của mỗi bin (a, b) với chỉ số")
# plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
# plt.show()
import torch

def encode_ab(ab, ab_bins):  # ab: Batch x 2 x 96 x 96
    a_bins, b_bins = ab_bins
    batch_size, _, height, width = ab.shape

    a, b = ab[:, 0].contiguous() * 128, ab[:, 1].contiguous() * 128  # Batch x 96 x 96

    # Tìm chỉ số bin cho từng giá trị trong `a` và `b`
    a_idx = torch.bucketize(a, a_bins) - 1  # Batch x 96 x 96
    b_idx = torch.bucketize(b, b_bins) - 1  # Batch x 96 x 96

    a_idx = torch.clamp(a_idx, min=0)  # Nếu a_idx = -1 -> gán thành 0
    b_idx = torch.clamp(b_idx, min=0)  # Nếu b_idx = -1 -> gán thành 0

    print(a_idx, b_idx)
    # Gán nhãn (b trước -> a sau)
    ab_target = b_idx * (len(a_bins) - 1) + a_idx  # Batch x 96 x 96
    return ab_target.unsqueeze(1)  # Batch x 1 x 96 x 96


def create_ab_bins(grid_size=12):
    a_bins = torch.arange(-110, 110 + grid_size, grid_size) 
    b_bins = torch.arange(-110, 110 + grid_size, grid_size) 
    
    return a_bins, b_bins

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
    return ab


# Test

# ab_bins = create_ab_bins(grid_size=12)
# bins_pred = torch.randint(0, 360, (2, 1, 2, 2)) # batch x 1 x W x H
# print(bins_pred) 
# ab =asign_centroid_color_batch(bins_pred, ab_bins)
# print(ab)


grid_size = 12
ab_bins = create_ab_bins(grid_size=grid_size)

# Tạo dữ liệu giả lập
batch_size, height, width = 2, 96, 96
ab = torch.rand(batch_size, 2, height, width) * 2 - 1  # Giá trị trong khoảng [-1, 1]

# Gọi hàm encode_ab
ab_target = encode_ab(ab, ab_bins)

# In kết quả
print("Input ab shape:", ab.shape)
print("Output ab_target shape:", ab_target.shape)
print("Sample ab_target:", ab_target[0, :5, :5])  # Hiển thị một phần nhỏ của kết quả