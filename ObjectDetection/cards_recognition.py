from enum import Enum

from samples import *
from boundingBoxes import *
from probabilities import *
import numpy

class Cards(Enum):
  CBlack = 0
  CBlue = 1
  CBrown = 2
  CGreen = 3
  COrange = 4
  CPurple = 5
  CRose = 6
  CWhite = 7
  CYellow = 8

class CardsRecognitionHelper:
  selectedSamplesQuality = "LQ"

  def __init__(self, height, width):
    if self.selectedSamplesQuality == "HQ":
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "Cards"))
    elif self.selectedSamplesQuality == "LQ":
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "CardsWithContour2"))

    [self.samplesSiftInfos, self.samplesHistograms] = loadSamples(path)

  rectangles = []

  def GetScreenPortions(self, img):
    height, width = img.shape[0],img.shape[1] 
    width_portion = int(width / 3)
    height_portion = int(height / 3)
    proportionh = int(0.2 * height_portion)
    proportionw = int(0.24 * width_portion)

    for i in range(3):
      for j in range(3):
        x = i * width_portion + proportionw
        w = (i + 1) * width_portion - proportionw
        y = j * height_portion + proportionh
        h = (j + 1) * height_portion - proportionh

        self.rectangles.append([x,y,w,h])
        #houghCircleDetection()

  def ComputeFrame(self, img, coordinates):
    selectedimg = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
    boundingBoxes = getCirclesBb(selectedimg, self.rectangles)

    if(len(boundingBoxes) > 0):
      siftProbabilities = []
      histoProbabilities = []
      for boundingBox in boundingBoxes:
        currentimg = selectedimg[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
        siftProbabilities.append(sift_detection(currentimg, self.samplesSiftInfos))
        histoProbabilities.append(histogram_Probabilities(currentimg, self.samplesHistograms))
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.5, 0.5])
      selectedimg = drawRectangleWithProbabilities(selectedimg, finalProbabilities, boundingBoxes, [], Cards)

    img[coordinates[1]:coordinates[3],coordinates[0]:coordinates[2]] = selectedimg
    return img

def houghCircleDetection(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Blur using 3 * 3 kernel.
  gray_blurred = cv2.blur(gray, (3, 3))
    
    #TRY SIMPLIFY IMAGE FIRST
  detected_circles = cv2.HoughCircles(gray_blurred, 
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 120, #->
                param2 = 120, minRadius = 1, maxRadius = 200) #param2 : 120, 130, 1, 200
  # Draw circles that are detected.
  detected_object = []
  if detected_circles is not None:
      # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
    for pt in detected_circles[0, :]:
      detected_object = (pt[0], pt[1], pt[2])  #x,y,rayon
      cv2.circle(img,(detected_object[0], detected_object[1]), detected_object[2], (255,0,0), 5)

  return img, detected_object
