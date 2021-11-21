
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def increaseContrast(img):
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
  clahe = cv2.createCLAHE(clipLimit= 5.0, tileGridSize=(5,5))
  v = clahe.apply(v)
  hsv_img = np.dstack((h,s,v))
  rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
  return rgb

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", "LQ", "CardsWithContour"))

a = os.listdir(path)
for file in a:
  img = cv2.imread(path + '/' + file)
  rgb = increaseContrast(img)

  cv2.imshow("1", img)
  cv2.imshow("2", rgb)
  cv2.waitKey(50000)

