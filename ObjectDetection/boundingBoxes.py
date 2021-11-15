import cv2
import numpy as np
from skimage.transform import hough_ellipse

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
    rectangle = houghCircleDetection(img[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]])
    #rectangle = getBoundingBox(img[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]])
    if(len(rectangle)>0):
      finalBbs.append([boundingBox[0] + rectangle[0],boundingBox[1] + rectangle[1],boundingBox[0] + rectangle[2],boundingBox[1] + rectangle[3]])
    else:
      finalBbs.append(boundingBox)

  return finalBbs

def houghCircleDetection(img):
  addedpixels = 0

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Blur using 3 * 3 kernel.
  gray_blurred = cv2.blur(gray, (3, 3))

  rectangle = []


  """
  detected_circles = cv2.HoughEllipse(gray_blurred,cv2.HOUGH_GRADIENT,1,10,param1 = 40, param2 = 40)
  
  if detected_circles is not None:
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
    pt = detected_circles[0, 0]
    detected_object = (pt[0], pt[1], pt[2])  # x,y,rayon
    cv2.circle(img, (detected_object[0], detected_object[1]), detected_object[2], (255, 0, 0), 5)
    cv2.imshow("Test",img)

    xcircle = detected_circles[0][0]
    ycircle = detected_circles[0][1]
    radius = detected_circles[0][2]

    xtl = xcircle - (radius + addedpixels)
    ytl = ycircle - (radius + addedpixels)
    xbr = xcircle + (radius + addedpixels)
    ybr = ycircle + (radius + addedpixels)
    rectangle = [xtl,ytl,xbr,ybr]
    cv2.imshow("test", gray_blurred)
    """
  return rectangle