from enum import Enum

from samples import *
from boundingBoxes import *
from probabilities import *

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
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "CardsWithoutContour"))

    [self.samplesSiftInfos, self.samplesHistograms] = loadSamples(path)

  rectangles = []

  def GetScreenPortions(self, height, width):
    width_portion = int(width / 3)
    height_portion = int(height / 3)
    proportionh = int(0.2 * height_portion)
    proportionw = int(0.2 * width_portion)

    for i in range(3):
      for j in range(3):
        x = j * width_portion + proportionw
        w = (j + 1) * width_portion - proportionw
        y = i * height_portion + proportionh
        h = (i + 1) * height_portion - proportionh

        self.rectangles.append([x,y,x+w,y+h])

  def ComputeFrame(self, img):
    boundingBoxes = self.rectangles.copy()

    if(len(boundingBoxes) > 0):
      siftProbabilities = []
      histoProbabilities = []
      for boundingBox in boundingBoxes:
        currentimg = img[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
        siftProbabilities.append(sift_detection(currentimg, self.samplesSiftInfos))
        histoProbabilities.append(histogram_Probabilities(currentimg, self.samplesHistograms))

      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.5, 0.5])
      img = drawRectangleWithProbabilities(img, finalProbabilities, boundingBoxes, [], Cards)

    return img