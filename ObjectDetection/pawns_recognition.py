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
    self.detectivePawnsLocations = list()

  def GetScreenPortion(self,img, coordinates):
    self.coordinates = coordinates
   
    self.cardSize = (int((coordinates[2]-coordinates[0])/3), int((coordinates[3]-coordinates[1])/3))
    self.dpOverlaySizePx = int(self.cardSize[0]/2)
    self.bBmaxArea = (self.cardSize[0]*self.cardSize[1])/6
    self.bBminArea = (self.cardSize[0]*self.cardSize[1])/30

    height, width = img.shape[0], img.shape[1]

    # Generate Masks
    self.mask = np.full((height, width), 0, dtype=np.uint8)
    self.detectivePawnsRectangle = [coordinates[0] - self.dpOverlaySizePx, coordinates[1] - self.dpOverlaySizePx, coordinates[2] + self.dpOverlaySizePx, coordinates[3] + self.dpOverlaySizePx]
    cv2.rectangle(self.mask, (self.detectivePawnsRectangle[0], (self.detectivePawnsRectangle[1])),(self.detectivePawnsRectangle[2], (self.detectivePawnsRectangle[3])), 255, -1)
    cv2.rectangle(self.mask, (coordinates[0],coordinates[1]), (coordinates[2],coordinates[3]), 0, -1)
    self.invertedmask = cv2.bitwise_not(self.mask)

    for j in range(12):
      if (j // 3 == 0):
        ymin = self.detectivePawnsRectangle[1]
        ymax = self.coordinates[1]
        xmin = self.coordinates[0] + self.cardSize[0] * (j % 3)
        xmax = self.coordinates[0] + self.cardSize[0] * (j % 3 + 1)
      elif (j // 3 == 2):
        ymin = self.coordinates[3]
        ymax = self.detectivePawnsRectangle[3]
        xmax = self.coordinates[2] - self.cardSize[0] * (j % 3)
        xmin = self.coordinates[2] - self.cardSize[0] * (j % 3 + 1)

      elif (j // 3 == 1):
        xmin = self.coordinates[2]
        xmax = self.detectivePawnsRectangle[2]
        ymin = self.coordinates[1] + self.cardSize[1] * (j % 3)
        ymax = self.coordinates[1] + self.cardSize[1] * (j % 3 + 1)
      elif (j // 3 == 3):
        xmin = self.detectivePawnsRectangle[0]
        xmax = self.coordinates[0]
        ymax = self.coordinates[3] - self.cardSize[1] * (j % 3)
        ymin = self.coordinates[3] - self.cardSize[1] * (j % 3 + 1)

      rectangle =[xmin,ymin,xmax,ymax]
      self.detectivePawnsLocations.append(rectangle)

  def ComputeFrame(self, img):
    maskedimg = cv2.bitwise_and(img, img, mask=self.mask)
    selectedimg = maskedimg[self.detectivePawnsRectangle[1]:self.detectivePawnsRectangle[3],self.detectivePawnsRectangle[0]:self.detectivePawnsRectangle[2]]

    boundingBoxes = getBoundingBoxes(selectedimg, self.bBmaxArea, self.bBminArea)

    siftProbabilities = []
    histoProbabilities = []
    for i in range(min(len(boundingBoxes), len(self.selectedEnum))):
      currentimg = selectedimg[boundingBoxes[i][1]:boundingBoxes[i][3], boundingBoxes[i][0]:boundingBoxes[i][2]]
      siftProbabilities.append(sift_detection(currentimg, self.samplesSiftInfos,self.selectedSamplesResolution))
      histoProbabilities.append(histogramProbabilities(currentimg, self.samplesHistograms))

    if (len(boundingBoxes) > 0):
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.3, 0.7])

      #selectedimg = drawRectangleWithProbabilities(selectedimg, finalProbabilities, boundingBoxes, self.selectedEnum, detectivePawns)

      assignedObjects = linearAssignment(finalProbabilities, self.selectedEnum)
      selectedimg = drawRectanglesWithAssignment(selectedimg, assignedObjects, boundingBoxes)
      DPpawnspositions = self.getDetectivePawnsPositions(assignedObjects,boundingBoxes)
      self.boardReference.setDetectivePawns(DPpawnspositions)

    maskedimg[self.detectivePawnsRectangle[1]:self.detectivePawnsRectangle[3],self.detectivePawnsRectangle[0]:self.detectivePawnsRectangle[2]] = selectedimg
    img = cv2.bitwise_and(img, img, mask=self.invertedmask)
    img = img+maskedimg

    #Draw the zones rectangles
    cv2.rectangle(img, (self.coordinates[0] - self.dpOverlaySizePx, self.coordinates[1] - self.dpOverlaySizePx), (self.coordinates[2] + self.dpOverlaySizePx, self.coordinates[3] + self.dpOverlaySizePx),(0, 255, 0), 2)
    cv2.rectangle(img, (self.coordinates[0],self.coordinates[1]), (self.coordinates[2],self.coordinates[3]),(0, 255, 0), 2)

    return img

  def getDetectivePawnsPositions(self, assignedObjects, boundingBoxes):
    positions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(boundingBoxes)):
      x = boundingBoxes[i][0]+ int((boundingBoxes[i][2]-boundingBoxes[i][0])/2) + self.detectivePawnsRectangle[0] #adding the overlay to the boudingbox
      y = boundingBoxes[i][1]+ int((boundingBoxes[i][3]-boundingBoxes[i][1])/2) + self.detectivePawnsRectangle[1] #adding the overlay to the boudingbox

      for j in range(12):
        xmin = self.detectivePawnsLocations[j][0]
        ymin = self.detectivePawnsLocations[j][1]
        xmax = self.detectivePawnsLocations[j][2]
        ymax = self.detectivePawnsLocations[j][3]

        if((xmin < x < xmax) and (ymin < y < ymax)):
          positions[j]=assignedObjects[i]

    return positions