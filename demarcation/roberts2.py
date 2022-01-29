import cv2 as cv
import numpy as np


img = cv.imread('img2.jpg', cv.IMREAD_GRAYSCALE)
chk = cv.imread('check.jpeg', cv.IMREAD_GRAYSCALE)

rows = np.size(img, 0)
columns = np.size(img, 0)
out = np.zeros(img.shape)
rows2 = np.size(chk, 0)
columns2 = np.size(chk, 0)
out2 = np.zeros(chk.shape)

for r in range(0, rows - 2):
    for c in range(0, columns - 2):
        H1 = int(img[r, c]) - img[r - 1, c - 1]
        H2 = int(img[r, c - 1]) - img[r - 1, c]
        out[r + 1, c + 1] = np.sqrt(np.square(H1) + np.square(H2))
for k in range(0, rows2 - 2):
    for l in range(0, columns2 - 2):
        H3 = int(chk[k, l]) - chk[k - 1, l - 1]
        H4 = int(chk[k, l - 1]) - chk[k - 1, l]
        out2[k + 1, l + 1] = np.sqrt(np.square(H3) + np.square(H4))


out = out / np.max(out) * 255
out = out.astype('uint8')
out2 = out2 / np.max(out2) * 255
out2 = out2.astype('uint8')

cv.imshow("Source", img)
cv.imshow("Result", out)
cv.imshow("Diagnostic image", chk)
cv.imshow("Result for diagnostic", out2)
cv.waitKey(0)