import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh mức xám
gray_image = cv2.imread(r'D:\Image-Corlorization-CIE-Lab-Color-Space\img\car_1.png', cv2.IMREAD_GRAYSCALE)

# Tạo ảnh màu với kích thước tương tự ảnh gốc
color_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

# Định nghĩa màu sắc cho từng khoảng giá trị
def apply_color(pixel_value):
    if pixel_value < 64:
        return [255, 0, 0]  # Màu đỏ
    elif pixel_value < 128:
        return [0, 255, 0]  # Màu xanh lá
    elif pixel_value < 192:
        return [0, 0, 255]  # Màu xanh dương
    else:
        return [255, 255, 0]  # Màu vàng

# Áp dụng màu sắc dựa trên giá trị pixel
for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
        color_image[i, j] = apply_color(gray_image[i, j])

# Hiển thị ảnh gốc và ảnh sau khi tô màu
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Ảnh gốc (Grayscale)")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Ảnh sau khi tô màu (Rule-based)")
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
