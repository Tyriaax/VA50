import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

from scipy.ndimage import gaussian_filter

img = cv.imread(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'Samples', "Cards", "black.jpg")))

#hist0 = cv.calcHist(img, [0,1,2], None, [256],[0,256])
hist1 = cv.calcHist([img],[0],None,[256],[0,256])
hist2 = cv.calcHist([img],[1],None,[256],[0,256])
hist3 = cv.calcHist([img],[2],None,[256],[0,256])

plt.subplot(221), plt.imshow(img)
plt.subplot(222),plt.plot(hist3)
plt.xlim([0,256])

#, plt.plot(hist1), plt.plot(hist2)

plt.show()

"""cv.imshow('ras', img)

img = cv.GaussianBlur(img,(0,0),5)
#img = gaussian_filter(img, sigma= 5)


ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
#th2 = cv.medianBlur(th2,4)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

ret3,th1 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[2],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

cv.imwrite(os.path.abspath(os.path.join(os.path.dirname( __file__ ), "white_binary.png")), images[2])
"""