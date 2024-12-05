import cv2 as cv

img = cv.imread(r"D:\Image-Corlorization-CIE-Lab-Color-Space\img\deer.jpg")

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_rize = cv.cvtColor(cv.resize(img, (160, 160)), cv.COLOR_BGR2RGB)
cv.imwrite("resize.png",img_rize)