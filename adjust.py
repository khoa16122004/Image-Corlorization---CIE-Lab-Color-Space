import cv2

def enhance_color_lab(image):
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    l = cv2.equalizeHist(l)  # Tăng cường độ sáng
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_img

def auto_white_balance(image):
    result = cv2.xphoto.createGrayworldWB()
    result.setSaturationThreshold(0.99)  # Giảm độ thiên lệch màu
    balanced_img = result.balanceWhite(image)
    return balanced_img

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    filtered_img = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_img

def adjust_saturation(image, alpha=1.3):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    s = cv2.convertScaleAbs(s, alpha=alpha)
    saturated_img = cv2.merge((h, s, v))
    return cv2.cvtColor(saturated_img, cv2.COLOR_HSV2BGR)


colorized_img = cv2.imread(r'D:\Image-Corlorization-CIE-Lab-Color-Space\img\nearly.png')
colorized_img_rgb = cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB)

enhanced_img = bilateral_filter( cv2.cvtColor(colorized_img_rgb, cv2.COLOR_BGR2RGB))
cv2.imwrite('enhanced_image.jpg', enhanced_img)
