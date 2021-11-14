from enum import Enum

from samples import *
from boundingBoxes import *
from probabilities import *

class ActionPawns(Enum):
  APChangeCard = 0
  APReturn = 1
  APSherlock = 2
  APToby = 3
  APWatson = 4

class DetectivePawns(Enum):
  DPSherlock = 0
  DPToby = 1
  DPWatson = 2

class PawnsRecognitionHelper:
  selectedEnum = DetectivePawns
  selectedSamplesQuality = "LQ"

  selectedSamplesResolution = 400

  maxAreaDivider = 4
  minAreaDivider = 12

  def __init__(self, height, width):
    if self.selectedEnum == DetectivePawns:
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "Pawns", "DetectivePawns"))
    elif self.selectedEnum == ActionPawns:
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "Pawns", "ActionPawns"))

    if (self.selectedSamplesQuality == "LQ"):
      [self.samplesSiftInfos, self.samplesHistograms] = loadSamples(path, self.selectedSamplesResolution)
    elif (self.selectedSamplesQuality == "HQ"):
      [self.samplesSiftInfos, self.samplesHistograms] = loadSamples(path, self.selectedSamplesResolution)

    self.bBmaxArea = height / self.maxAreaDivider * width / self.maxAreaDivider  # TODO Find better way ?
    self.bBminArea = height / self.minAreaDivider * width / self.minAreaDivider  # TODO Find better way ?

  def ComputeFrame(self, img):
    boundingBoxes = getBoundingBoxes(img, self.bBmaxArea, self.bBminArea)

    siftProbabilities = []
    histoProbabilities = []
    for boundingBox in boundingBoxes:
      currentimg = img[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
      siftProbabilities.append(sift_detection(currentimg, self.samplesSiftInfos))
      histoProbabilities.append(histogram_Probabilities(currentimg, self.samplesHistograms))

    if (len(boundingBoxes) > 0):
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.5, 0.5])
      img = drawRectangleWithProbabilities(img, finalProbabilities, boundingBoxes, [], self.selectedEnum)

    return img