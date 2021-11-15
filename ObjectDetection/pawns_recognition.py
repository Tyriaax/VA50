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

  def ComputeFrame(self, img, coordinates):
    rectangles = []
    rectangles.append([coordinates[0],0 , coordinates[2], coordinates[1]])
    rectangles.append([0, coordinates[1], coordinates[0], coordinates[3]])
    rectangles.append([coordinates[2], coordinates[1],coordinates[0] + coordinates[2], coordinates[3]])
    rectangles.append([coordinates[0], coordinates[3], coordinates[2], coordinates[1] + coordinates[3]])

    for rectangle in rectangles:
      workingimg = img[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]

      boundingBoxes = getBoundingBoxes(workingimg, self.bBmaxArea, self.bBminArea)

      siftProbabilities = []
      histoProbabilities = []
      for boundingBox in boundingBoxes:
        currentimg = workingimg[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
        siftProbabilities.append(sift_detection(currentimg, self.samplesSiftInfos))
        histoProbabilities.append(histogram_Probabilities(currentimg, self.samplesHistograms))

      if (len(boundingBoxes) > 0):
        finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.5, 0.5])
        workingimg = drawRectangleWithProbabilities(workingimg, finalProbabilities, boundingBoxes, [], self.selectedEnum)

    img[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = workingimg

    return img
