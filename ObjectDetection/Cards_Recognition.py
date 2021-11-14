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

  def getScreenPortions(self, height, width):
    height_portion = int(height / 3)
    proportion = int(0.2 * height_portion)

    for i in range(3):
      for j in range(3):
        x = j * height_portion + proportion
        w = (j + 1) * height_portion - proportion
        y = i * height_portion + proportion
        h = (i + 1) * height_portion - proportion

        self.rectangles.append([x,y,x+w,y+h])

  def ComputeFrame(self, img):

    if(len(self.rectangles) > 0):
      siftProbabilities = []
      histoProbabilities = []
      for rectangle in self.rectangles:
        currentimg = img[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]
        siftProbabilities.append(sift_detection(currentimg, self.samplesSiftInfos))
        histoProbabilities.append(histogram_Probabilities(currentimg, self.samplesHistograms))

      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.5, 0.5])
      img = drawRectangleWithProbabilities(img, finalProbabilities, self.rectangles, [], Cards)

    return img