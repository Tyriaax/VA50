import os
from zncc import *
import numpy as np

# This function loads the samples, in this case the zncc which is the image converted in gray levels
def loadSamples(path, resolution = None, circleMask = False, applySharpen = False):
  dir = os.listdir(path)

  samplesZncc = []

  for image in dir:
    img = cv2.imread(os.path.join(path, image))

    samplesZncc.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

  return samplesZncc

def increaseImgColorContrast(img):
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
  clahe = cv2.createCLAHE(clipLimit= 1.0, tileGridSize=(16,16))
  v = clahe.apply(v)
  hsv_img = np.dstack((h,s,v))
  rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
  return rgb
