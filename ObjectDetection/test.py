
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.exposure import is_low_contrast
from imutils.paths import list_images
import argparse
import imutils
import cv2
from cv2 import dnn_sup

def increaseContrast(img):
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
  clahe = cv2.createCLAHE(clipLimit= 5.0, tileGridSize=(5,5))
  v = clahe.apply(v)
  hsv_img = np.dstack((h,s,v))
  rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
  return rgb

def contrast(image):

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# blur the image slightly and perform edge detection
  blurred = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_DEFAULT)
  edged = cv2.Canny(blurred, 30, 150)
	# initialize the text and color to indicate that the input image
	# is *not* low contrast
  text = "Low contrast: No"
  color = (0, 255, 0)
  if is_low_contrast(gray, fraction_threshold= 255):
		# update the text and color
    text = "Low contrast: Yes"
    color = (0, 0, 255)
  # otherwise, the image is *not* low contrast, so we can continue
  # processing it
  else:
    # find contours in the edge map and find the largest one,
    # which we'll assume is the outline of our color correction
    # card
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # draw the largest contour on the image
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
  # draw the text on the output image
  cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
    color, 2)
  # show the output image and edge map
  cv2.imshow("Image", image)
  cv2.imshow("Edge", edged)
  cv2.waitKey(0)


path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", "LQ", "CardsWithContour3"))

a = os.listdir(path)
for file in a:
  img = cv2.imread(path + '/' + file)
  img = imutils.resize(img, width=200)
  rgb = increaseContrast(img)
  #contrast(img)

  cv2.imshow("1", img)
  cv2.imshow("2", rgb)
  cv2.waitKey(50000)

