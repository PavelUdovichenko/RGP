import numpy as np
import matplotlib.pyplot as plt
import cv2

img=cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

def corr(img, mask):
    row,col=img.shape
    m,n=mask.shape
    new=np.zeros((row+m-1,col+n-1))
    n=n//2
    m=m//2
    filtered_img=np.zeros(img.shape)
    new[m:new.shape[0]-m,n:new.shape[1]-n]=img
    for i in range(m,new.shape[0]-m):
        for j in range(n,new.shape[1]-n):
            temp=new[i-m:i+m+1,j-m:j+m+1]
            result=temp*mask
            filtered_img[i-m,j-n]=result.sum()

    return filtered_img

def gaussian(m,n,sigma):
    gaussian=np.zeros((m,n))
    m=m//2
    n=n//2
    for x in range(-m,m+1):
        for y in range(-n,n+1):
            x1=sigma*(2*np.pi)**2
            x2=np.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian[x+m,y+n]=(1/x1)*x2
    return gaussian

def show_gaussian(img,sigma):
    g = gaussian(5, 5, sigma)
    n = corr(img, g)
    plt.imshow(img, cmap='gray')
    plt.figure()
    plt.imshow(n, cmap='gray')
    plt.show()


if __name__ == '__main__':
    show_gaussian(img,3)




