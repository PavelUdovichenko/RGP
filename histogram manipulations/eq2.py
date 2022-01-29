import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# сначала прочитаем нашу картинку
img = cv.imread("img.jpg", cv.IMREAD_GRAYSCALE)  # считаем в оттенках серого
cv.imshow("before", img)

# зададим колличество пикселей и их интенсивность
hist, bins = np.histogram(img, 256, [0, 255])
# считаем куммулятивную функцию распределения
cdf = hist.cumsum()
# построим наш график исходного изображения
plt.plot(hist, 'b')

# растягиваем пиксели в наш диапазон [0,255]
cdf = (cdf-cdf[0])*255/(cdf[-1]-1)
cdf = cdf.astype(np.uint8)  # Transform from float64 back to unit8

# generate img after Histogram Equalization
img2 = np.zeros((384, 495, 1), dtype=np.uint8)
img2 = cdf[img]

hist2, bins2 = np.histogram(img2, 256, [0, 255])
cdf2 = hist2.cumsum()
plt.plot(hist2, 'g')

cv.imshow("after", img2)
plt.show()
cv.waitKey(0)
