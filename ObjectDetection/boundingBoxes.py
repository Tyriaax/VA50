import cv2
import numpy as np
from skimage import color,io
from skimage.draw.draw import rectangle
from skimage.transform import hough_ellipse
from skimage.feature import canny
from skimage.draw import ellipse_perimeter
from skimage.util import img_as_float
from matplotlib import pyplot as plt

def imageProcessingForFindingContours(img):
  # First we convert the frame to a grayscale image
  img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # We then use a median blur technique to reduce the noise
  img2 = cv2.medianBlur(img2, 5)

  # We then apply a sharpening filter to enhance the edges
  sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
  img2 = cv2.filter2D(img2, -1, sharpen_kernel)

  # We can then threshold to get a binary image
  img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

  # We invert the image so that the contours can get detected
  img2 = 255 - img2

  # We also apply a close morphology transformation to get rid of the imperfections inside the shape
  morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, morph_kernel, iterations=3)  # We apply a close transformation

  return img2

def getBoundingBoxes(img,maxarea,minarea):
  rectangles = []

  img2 = imageProcessingForFindingContours(img)

  # We then use findContours to get the contours of the shape
  cnts = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  # We then loop through all the detected contours to onl²y retrieve the ones with a desired area
  for c in cnts:
    area = cv2.contourArea(c)
    if minarea <= area <= maxarea:
      x, y, w, h = cv2.boundingRect(c)
      rectangle = [x, y, x+w, y+h]
      rectangles.append(rectangle)

  return rectangles

def getBoundingBox(img):
  img2 = imageProcessingForFindingContours(img)
  minuspixels = 10
  maxarea = (img2.shape[0]-minuspixels)*(img2.shape[1]-minuspixels)

  # We then use findContours to get the contours of the shape
  cnts = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  maxfoundarea = 0
  rectangle = []
  # We then loop through all the detected contours to onl²y retrieve the ones with a desired area
  for c in cnts:
    area = cv2.contourArea(c)
    if (area > maxfoundarea) and (area < maxarea):
      x, y, w, h = cv2.boundingRect(c)
      rectangle = [x, y, x + w, y + h]
      maxfoundarea = area

  return rectangle

def getCirclesBb(img, boundingBoxes):
  finalBbs = []
  for boundingBox in boundingBoxes:
    rectangle = []
    #rectangle = houghEllipseDetection(img[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]])
    #rectangle = getBoundingBox(img[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]])
    if(len(rectangle)>0):
      finalBbs.append([boundingBox[0] + rectangle[0],boundingBox[1] + rectangle[1],boundingBox[0] + rectangle[2],boundingBox[1] + rectangle[3]])
    else:
      finalBbs.append(boundingBox)

  return finalBbs

def houghEllipseDetection(img):
  addedpixels = 0
  rectangle = []

  image = img_as_float(img)
  image_gray = color.rgb2gray(image)
  #io.imshow(image_gray)
  #plt.show()

  edges = canny(image_gray, sigma=2.0,
                low_threshold=0.55, high_threshold=0.8)

  # Perform a Hough Transform
  # The accuracy corresponds to the bin size of a major axis.
  # The value is chosen in order to get a single high accumulator.
  # The threshold eliminates low accumulators
  result = hough_ellipse(edges, accuracy=10, threshold=250,
                         min_size=1, max_size=100)
  if(len(result)>0):
    print("Cercle detecte")
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image[cy, cx] = (0, 0, 255)
    io.imshow(image)
    plt.show()

  return rectangle