from enum import Enum

import cv2.cv2

from samples import *
from boundingBoxes import *
from probabilities import *
from GameBoard import *

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

  def __init__(self, height, width, gameBoard):
    if self.selectedEnum == DetectivePawns:
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "Pawns", "DetectivePawns"))
    elif self.selectedEnum == ActionPawns:
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "Pawns", "ActionPawns"))

    if (self.selectedSamplesQuality == "LQ"):
      [self.samplesSiftInfos, self.samplesHistograms] = loadSamples(path, self.selectedSamplesResolution)
    elif (self.selectedSamplesQuality == "HQ"):
      [self.samplesSiftInfos, self.samplesHistograms] = loadSamples(path, self.selectedSamplesResolution)

    self.boardReference = gameBoard

  def GetScreenPortion(self,img, coordinates):
    self.coordinates = coordinates
   
    cardSize = ((coordinates[2]-coordinates[0])/3, (coordinates[3]-coordinates[1])/3)
    self.dpOverlaySizePx = int(cardSize[0]/2)
    self.bBmaxArea = (cardSize[0]*cardSize[1])/6
    self.bBminArea = (cardSize[0]*cardSize[1])/30

    height, width = img.shape[0], img.shape[1]

    # Generate Masks
    self.mask = np.full((height, width), 0, dtype=np.uint8)
    cv2.rectangle(self.mask, (coordinates[0] - self.dpOverlaySizePx, coordinates[1] - self.dpOverlaySizePx), (coordinates[2] + self.dpOverlaySizePx, coordinates[3] + self.dpOverlaySizePx), 255, -1)
    cv2.rectangle(self.mask, (coordinates[0],coordinates[1]), (coordinates[2],coordinates[3]), 0, -1)
    self.invertedmask = cv2.bitwise_not(self.mask)

  def ComputeFrame(self, img):
    maskedimg = cv2.bitwise_and(img, img, mask=self.mask)
    selectedimg = maskedimg[self.coordinates[1] - self.dpOverlaySizePx:self.coordinates[3] + self.dpOverlaySizePx, self.coordinates[0] - self.dpOverlaySizePx:self.coordinates[2] + self.dpOverlaySizePx]

    boundingBoxes = getBoundingBoxes(selectedimg, self.bBmaxArea, self.bBminArea)

    siftProbabilities = []
    histoProbabilities = []
    for i in range(min(len(boundingBoxes), len(self.selectedEnum))):
      currentimg = selectedimg[boundingBoxes[i][1]:boundingBoxes[i][3], boundingBoxes[i][0]:boundingBoxes[i][2]]
      siftProbabilities.append(sift_detection(currentimg, self.samplesSiftInfos))
      histoProbabilities.append(histogramProbabilities(currentimg, self.samplesHistograms))

    if (len(boundingBoxes) > 0):
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.5, 0.5])

      #selectedimg = drawRectangleWithProbabilities(selectedimg, finalProbabilities, boundingBoxes, self.selectedEnum, detectivePawns)

      assignedObjects = linearAssignment(finalProbabilities, self.selectedEnum)
      selectedimg = drawRectanglesWithAssignment(selectedimg, assignedObjects, boundingBoxes)
      self.boardReference.setDetectivePawns = assignedObjects

    maskedimg[self.coordinates[1] - self.dpOverlaySizePx:self.coordinates[3] + self.dpOverlaySizePx, self.coordinates[0] - self.dpOverlaySizePx:self.coordinates[2] + self.dpOverlaySizePx] = selectedimg
    img = cv2.bitwise_and(img, img, mask=self.invertedmask)
    img = img+maskedimg

    #Draw the zones rectangles
    cv2.rectangle(img, (self.coordinates[0] - self.dpOverlaySizePx, self.coordinates[1] - self.dpOverlaySizePx), (self.coordinates[2] + self.dpOverlaySizePx, self.coordinates[3] + self.dpOverlaySizePx),(0, 255, 0), 2)
    cv2.rectangle(img, (self.coordinates[0],self.coordinates[1]), (self.coordinates[2],self.coordinates[3]),(0, 255, 0), 2)

    return img
