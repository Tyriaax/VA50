import cv2
import numpy as np

rectangleMaxRatioDifference = 0.4

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

def getBoundingBoxes(img,maxarea,minarea,inspectInsideCountours = False):
  rectangles = []

  img2 = imageProcessingForFindingContours(img)

  # We then use findContours to get the contours of the shape
  if not inspectInsideCountours:
    retrievalMode = cv2.RETR_EXTERNAL
  else:
    retrievalMode = cv2.RETR_LIST

  cnts = cv2.findContours(img2, retrievalMode, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  # We then loop through all the detected contours to only retrieve the ones with a desired area and ratio
  for c in cnts:
    area = cv2.contourArea(c)
    if minarea <= area <= maxarea:
      x, y, w, h = cv2.boundingRect(c)
      if (1-rectangleMaxRatioDifference)*h <= w <= (1+rectangleMaxRatioDifference)*h:
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
  # We then loop through all the detected contours to onl²y retrieve the ones with a desired area and ratio
  for c in cnts:
    area = cv2.contourArea(c)
    if (area > maxfoundarea) and (area < maxarea):
      x, y, w, h = cv2.boundingRect(c)
      rectangle = [x, y, x + w, y + h]
      maxfoundarea = area

  return rectangle

# This fonction adds offset to bounding box values, for example when the coordinates of the Bb were found in a cropped image
def addOffsetToBb(boundingBoxes, overlayX,overlayY):
    for boundingBox in boundingBoxes:
      boundingBox[0] = boundingBox[0] + overlayX
      boundingBox[1] = boundingBox[1] + overlayY
      boundingBox[2] = boundingBox[2] + overlayX
      boundingBox[3] = boundingBox[3] + overlayY

    return boundingBoxes