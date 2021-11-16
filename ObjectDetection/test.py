import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", "LQ", "CardsWithContour3", "CPurple.jpg"))
img = cv2.imread(path)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]

clahe = cv2.createCLAHE(clipLimit= 5.0, tileGridSize=(3,3))
v = clahe.apply(v)

hsv_img = np.dstack((h,s,v))

rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

cv2.imshow("1", img)
cv2.imshow("2", rgb)
cv2.waitKey(50000)

