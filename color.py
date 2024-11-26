from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import cv2 as cv

def get_rgb_combinations():
    rgb = np.zeros((256, 256, 256, 3), dtype=np.uint8)

    for i in range(256):
        for j in range(256):
            for k in range(256):
                rgb[i, j, k] = [i, j, k]

    return rgb

def get_in_gamut_ab():
    rgb = get_rgb_combinations()
    lab = color.rgb2lab(rgb)
    print(lab.min(), lab.max())

    ab = lab[:, :, :, 1:]
    ab = ab.reshape(ab.shape[0] * ab.shape[1] * ab.shape[2], 2)
    return ab

def get_ab_centroids():
    centroids = np.arange(-110, 120, 10, dtype=np.float32)
    centroids = np.transpose([np.tile(centroids, len(centroids)), np.repeat(centroids, len(centroids))])
    #centroids[:, [0, 1]] = centroids[:, [1, 0]]
    return centroids

def get_closest_centroids(ab, n=5):
    centroids = get_ab_centroids()
    neigh = NearestNeighbors(n_neighbors=n).fit(centroids)
    _, indices = neigh.kneighbors(ab)
    return indices

def get_simple_ab_mask(ab):
    mask, _, _ = np.histogram2d(ab[:, 0], ab[:, 1], bins=np.arange(-115, 125, 10))
    mask = (mask != 0).astype(np.uint8)
    return mask

def soften_ab_mask(ab, mask):
    centroids = get_ab_centroids()
    neigh = NearestNeighbors(n_neighbors=6).fit(centroids)
    _, indices = neigh.kneighbors(ab)
    indices = indices[:, 1:]

    unique, _ = np.unique(indices, return_counts=True)

    soft_mask = mask.reshape(mask.shape[0] * mask.shape[1])
    soft_mask[unique] = 1
    soft_mask = soft_mask.reshape(mask.shape[0], mask.shape[1])

    return soft_mask

def get_in_gamut_mask():
    ab = get_in_gamut_ab()
    mask = get_simple_ab_mask(ab)
    soft_mask = soften_ab_mask(ab, mask)
    return soft_mask

def get_in_gamut_rgb():
    mask = get_in_gamut_mask()
    gamut = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                gamut[i, j] = [50, -110 + i * 10, -110 + j * 10]
            else:
                gamut[i, j] = [100, 0, 0]
    
    return gamut

def in_hull_centroids():
    mask = get_in_gamut_mask()
    centroids = get_ab_centroids().reshape(23, 23, 2)
    hull = centroids[mask == 1]
    
    return hull

if __name__ == "__main__":
    # gamut = get_in_gamut_rgb()
    # print(gamut.shape)
    # gamut = color.lab2rgb(gamut)

    # plt.imshow(gamut)
    # plt.savefig("gamut.png")

    get_in_gamut_ab()
