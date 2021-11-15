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

  def __init__(self, height, width):
    if self.selectedEnum == DetectivePawns:
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "Pawns", "DetectivePawns"))
    elif self.selectedEnum == ActionPawns:
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "Pawns", "ActionPawns"))

    if (self.selectedSamplesQuality == "LQ"):
      [self.samplesSiftInfos, self.samplesHistograms] = loadSamples(path, self.selectedSamplesResolution)
    elif (self.selectedSamplesQuality == "HQ"):
      [self.samplesSiftInfos, self.samplesHistograms] = loadSamples(path, self.selectedSamplesResolution)

  def GetScreenPortion(self,img, coordinates):
    height, width = img.shape[0], img.shape[1]

    #Generate Mask
    self.mask = np.full((height, width), 255, dtype=np.uint8)
    cv2.rectangle(self.mask, (coordinates[0],coordinates[1]), (coordinates[2],coordinates[3]), 0, -1)

    cardSize = ((coordinates[2]-coordinates[0])/3,(coordinates[3]-coordinates[1])/3)
    self.bBmaxArea = (cardSize[0]*cardSize[1])/6
    self.bBminArea = (cardSize[0]*cardSize[1])/30

    self.coordinates = coordinates

  def ComputeFrame(self, img):
    board = Board()
    detectivePawn = board.getDetectivePawn()

    selectedimg = cv2.bitwise_and(img, img, mask=self.mask)

    boundingBoxes = getBoundingBoxes(selectedimg, self.bBmaxArea, self.bBminArea)

    siftProbabilities = []
    histoProbabilities = []
    for i in range(min(len(boundingBoxes), len(self.selectedEnum))):
    #for boundingBox in boundingBoxes:
      currentimg = selectedimg[boundingBoxes[i][1]:boundingBoxes[i][3], boundingBoxes[i][0]:boundingBoxes[i][2]]
      siftProbabilities.append(sift_detection(currentimg, self.samplesSiftInfos))
      histoProbabilities.append(histogram_Probabilities(currentimg, self.samplesHistograms))

    if (len(boundingBoxes) > 0):
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.5, 0.5])
      selectedimg = drawRectangleWithProbabilities(selectedimg, finalProbabilities, boundingBoxes, self.selectedEnum, detectivePawn)

    selectedimg[self.coordinates[1]-1:self.coordinates[3]+1, self.coordinates[0]-1:self.coordinates[2]+1] = img[self.coordinates[1]-1:self.coordinates[3]+1, self.coordinates[0]-1:self.coordinates[2]+1]

    return selectedimg
