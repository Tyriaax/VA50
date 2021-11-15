from enum import Enum

import cv2.cv2

from samples import *
from boundingBoxes import *
from probabilities import *
from game_board_recognition import*

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

  maxAreaDivider = 10
  minAreaDivider = 20

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
    mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)

    board = Board()
    detectivePawn = board.getDetectivePawn()

    cv2.rectangle(mask, (coordinates[0],coordinates[1]), (coordinates[2],coordinates[3]), 0, -1)

    selectedimg = cv2.bitwise_and(img, img, mask=mask)

    boundingBoxes = getBoundingBoxes(selectedimg, self.bBmaxArea, self.bBminArea)

    siftProbabilities = []
    histoProbabilities = []
    for boundingBox in boundingBoxes:
      currentimg = selectedimg[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
      siftProbabilities.append(sift_detection(currentimg, self.samplesSiftInfos))
      histoProbabilities.append(histogram_Probabilities(currentimg, self.samplesHistograms))

    if (len(boundingBoxes) > 0):
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.5, 0.5])
      selectedimg = drawRectangleWithProbabilities(selectedimg, finalProbabilities, boundingBoxes, self.selectedEnum, detectivePawn)

    selectedimg[coordinates[1]-1:coordinates[3]+1, coordinates[0]-1:coordinates[2]+1] = img[coordinates[1]-1:coordinates[3]+1, coordinates[0]-1:coordinates[2]+1]

    return selectedimg
