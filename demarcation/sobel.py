import cv2 as cv
import numpy as np

img = cv.imread('img2.jpg', cv.IMREAD_GRAYSCALE)
chk = cv.imread('check.jpeg', cv.IMREAD_GRAYSCALE)

x = np.array([[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]])

y = np.array([[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]])

rows = np.size(img, 0)
columns = np.size(img, 0)
out = np.zeros(img.shape)
rows2 = np.size(chk, 0)
columns2 = np.size(chk, 0)
out2 = np.zeros(chk.shape)

for r in range(0, rows - 2):
    for c in range(0, columns - 2):
        h1 = sum(sum(x * img[r:r + 3, c:c + 3]))
        h2 = sum(sum(y * img[r:r + 3, c:c + 3]))
        out[r + 1, c + 1] = np.sqrt(np.square(h1) + np.square(h2))

for k in range(0, rows2 - 2):
    for l in range(0, columns2 - 2):
        h3 = sum(sum(x * chk[k:k + 3, l:l + 3]))
        h4 = sum(sum(y * chk[k:k + 3, l:l + 3]))
        out2[k + 1, l + 1] = np.sqrt(np.square(h3) + np.square(h4))

out = out / np.max(out) * 255
out = out.astype('uint8')
out2 = out2 / np.max(out2) * 255
out2 = out2.astype('uint8')

cv.imshow("Source", img)
cv.imshow("Result Sobel", out)
cv.imshow("Diagnostic image", chk)
cv.imshow("Result for diagnostic", out2)
cv.waitKey(0)