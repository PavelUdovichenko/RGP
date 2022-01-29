import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

# читаем картинку в градациях серого
img = cv.imread('img.jpg', cv.IMREAD_GRAYSCALE)
# imgS = cv.resize(img, (640, 400))
cv.imshow('before', img)
# гистограмма данныъ изображения
plt.hist(img.ravel(), 256, [0, 255])  # данные о колличестве пикселей и их интенсивности от 0 до 255
hist, bins = np.histogram(img.ravel(), 256, [0, 255])
plt.plot(hist, 'b')

# для реализации линейного растяжения нам потребуются граничные значения нашей гистограммы
# cl
a = 0
b = 255
while (sum(hist[a:(b+1)])/sum(hist)>0.95):  # отрезаем 5 процентов от общей площади
    if hist[a]<hist[b]:
        a+=1
    else:
        b-=1

img2 = img/(b-a)
img2 = img2 * 255
img2 = np.around(img2)

hist2, bins2 = np.histogram(img2, 256, [0, 255])
plt.plot(hist2, 'g')

# print(out)
# plt.hist(img2.ravel(),256,[0,256])
img2 = img/(b-a)
cv.imshow("after", img2)

plt.show()
cv.waitKey(0)
