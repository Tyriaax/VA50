from enum import Enum

import cv2.cv2

from samples import *
from boundingBoxes import *
from probabilities import *
from GameBoard import *

class ActionPawns(Enum):
  APSherlock = 0
  APAlibi = 1
  APToby = 2
  APWatson = 3
  APJoker = 4
  APReturn = 5
  APChangeCard = 6
  APReturn2 = 7


class DetectivePawns(Enum):
  DPSherlock = 0
  DPToby = 1
  DPWatson = 2

class PawnsRecognitionHelper:
  selectedSamplesQuality = "HQ" #TODO Change back

  selectedSamplesResolution = 400

  maxThrownActionPawnsNumber = 4

  def __init__(self, height, width, gameBoard):

    DPpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "Pawns", "DetectivePawns"))
    APpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "Pawns", "ActionPawns3"))


    [self.DPsamplesSiftInfos, self.DPsamplesHistograms, self.DPsamplesZncc] = loadSamples(DPpath, self.selectedSamplesResolution)

    [self.APsamplesSiftInfos, self.APsamplesHistograms, self.APsamplesZncc] = loadSamples(APpath, self.selectedSamplesResolution)

    self.boardReference = gameBoard
    self.detectivePawnsLocations = list()
    self.actionPawnsBb = list()

  def GetScreenPortion(self,img, coordinates):
    self.coordinates = coordinates
   
    self.cardSize = (int((coordinates[2]-coordinates[0])/3), int((coordinates[3]-coordinates[1])/3))

    dPOverlayCardRatio = 0.6
    self.dpOverlaySizePx = int(self.cardSize[0]*dPOverlayCardRatio)

    self.bBmaxArea = (self.cardSize[0]*self.cardSize[1])/2
    self.bBminArea = (self.cardSize[0]*self.cardSize[1])/12

    height, width = img.shape[0], img.shape[1]

    # Generate Masks
    self.detectivePawnsRectangle = [coordinates[0] - self.dpOverlaySizePx, coordinates[1] - self.dpOverlaySizePx,
                                    coordinates[2] + self.dpOverlaySizePx, coordinates[3] + self.dpOverlaySizePx]

    self.DPmask = np.full((height, width), 0, dtype=np.uint8)
    cv2.rectangle(self.DPmask, (self.detectivePawnsRectangle[0], (self.detectivePawnsRectangle[1])),(self.detectivePawnsRectangle[2], (self.detectivePawnsRectangle[3])), 255, -1)
    cv2.rectangle(self.DPmask, (coordinates[0],coordinates[1]), (coordinates[2],coordinates[3]), 0, -1)

    self.APmask = np.full((height, width), 255, dtype=np.uint8)
    cv2.rectangle(self.APmask, (self.detectivePawnsRectangle[0], (self.detectivePawnsRectangle[1])),(self.detectivePawnsRectangle[2], (self.detectivePawnsRectangle[3])), 0, -1)

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

      rectangle =[xmin, ymin, xmax, ymax]
      self.detectivePawnsLocations.append(rectangle)

  def ComputeDetectivePawns(self, img):
    maskedimg = cv2.bitwise_and(img, img, mask=self.DPmask)
    selectedimg = maskedimg[self.detectivePawnsRectangle[1]:self.detectivePawnsRectangle[3],self.detectivePawnsRectangle[0]:self.detectivePawnsRectangle[2]]

    boundingBoxes = getBoundingBoxes(selectedimg, self.bBmaxArea, self.bBminArea)
    boundingBoxes = addOffsetToBb(boundingBoxes,self.detectivePawnsRectangle[0],self.detectivePawnsRectangle[1])

    siftProbabilities = []
    histoProbabilities = []
    znccProbabilities = []
    for i in range(min(len(boundingBoxes), len(DetectivePawns))):
      currentimg = img[boundingBoxes[i][1]:boundingBoxes[i][3], boundingBoxes[i][0]:boundingBoxes[i][2]]
      siftProbabilities.append(sift_detection(currentimg, self.DPsamplesSiftInfos,self.selectedSamplesResolution))
      histoProbabilities.append(histogramProbabilities(currentimg, self.DPsamplesHistograms))
      znccProbabilities.append(zncc_pawn(currentimg,self.DPsamplesZncc))

    if (len(boundingBoxes) > 0):
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities,znccProbabilities], [0.5,0.5,0])

      assignedObjects = linearAssignment(finalProbabilities, DetectivePawns)
      DPpawnspositions = self.getDetectivePawnsPositions(assignedObjects,boundingBoxes)
      self.boardReference.setDetectivePawns(DPpawnspositions)
      self.detectivePawnsBb = boundingBoxes[0:len(DetectivePawns)]
      self.detectivePawnsBbOrder = assignedObjects

  def ComputeActionPawns(self, img):
    maskedimg = cv2.bitwise_and(img, img, mask=self.APmask)
    #cv2.imshow("ActionPawnsImg",maskedimg)
    boundingBoxes = getBoundingBoxes(maskedimg, self.bBmaxArea, self.bBminArea, inspectInsideCountours = True)

    siftProbabilities = []
    histoProbabilities = []
    znccProbabilites = []
    for i in range(min(len(boundingBoxes), self.maxThrownActionPawnsNumber)):
      currentimg = maskedimg[boundingBoxes[i][1]:boundingBoxes[i][3], boundingBoxes[i][0]:boundingBoxes[i][2]]
      siftProbabilities.append(sift_detection(currentimg, self.APsamplesSiftInfos, self.selectedSamplesResolution))
      histoProbabilities.append(histogramProbabilities(currentimg, self.APsamplesHistograms))
      znccProbabilites.append(zncc_pawn(currentimg, self.APsamplesZncc))

    if (len(boundingBoxes) > 0):
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities, znccProbabilites],  [0,0, 1])

      assignedObjects = linearAssignment(finalProbabilities, ActionPawns)
      self.boardReference.setActionPawns(assignedObjects)
      self.actionPawnsBb = boundingBoxes[0:self.maxThrownActionPawnsNumber]

  def ComputeFrame(self, img):
    self.ComputeActionPawns(img)
    self.ComputeDetectivePawns(img)

  def DrawFrame(self, img):
    img = self.DrawZonesRectangles(img, drawOffset=True)
    img = self.DrawDetectivePawns(img)
    img = self.DrawActionPawns(img)

    return img

  def DrawDetectivePawns(self, img):
    """
    detectivePawns = self.boardReference.getDetectivePawns()
    for i in range(len(detectivePawns)):
      if detectivePawns[i] != 0:
        img = drawRectangle(img,self.detectivePawnsLocations[i],detectivePawns[i])
    """
    img = drawRectanglesWithAssignment(img, self.detectivePawnsBbOrder, self.detectivePawnsBb)

    return img

  def DrawActionPawns(self, img):
    actionPawns = self.boardReference.getActionPawns()
    img = drawRectanglesWithAssignment(img, actionPawns, self.actionPawnsBb)

    return img

  def DrawZonesRectangles(self, img, drawOffset = False):
    offsetpx = 10
    cv2.rectangle(img, (self.coordinates[0] - self.dpOverlaySizePx, self.coordinates[1] - self.dpOverlaySizePx),
                  (self.coordinates[2] + self.dpOverlaySizePx, self.coordinates[3] + self.dpOverlaySizePx), (0, 0, 255),
                  2)
    cv2.rectangle(img, (self.coordinates[0], self.coordinates[1]), (self.coordinates[2], self.coordinates[3]),
                  (0, 0, 255), 2)
    if (drawOffset):
      cv2.rectangle(img, (self.coordinates[0] - self.dpOverlaySizePx + offsetpx, self.coordinates[1] - self.dpOverlaySizePx + offsetpx),
                  (self.coordinates[2] + self.dpOverlaySizePx - offsetpx, self.coordinates[3] + self.dpOverlaySizePx - offsetpx), (255, 0, 0),
                  2)
      cv2.rectangle(img, (self.coordinates[0] - offsetpx, self.coordinates[1] - offsetpx), (self.coordinates[2] + offsetpx, self.coordinates[3] + offsetpx),
                  (255, 0, 0), 2)
    return img

  def getDetectivePawnsPositions(self, assignedObjects, boundingBoxes):
    positions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(boundingBoxes)):
      x = boundingBoxes[i][0]+ int((boundingBoxes[i][2]-boundingBoxes[i][0])/2)
      y = boundingBoxes[i][1]+ int((boundingBoxes[i][3]-boundingBoxes[i][1])/2)

      for j in range(12):
        xmin = self.detectivePawnsLocations[j][0]
        ymin = self.detectivePawnsLocations[j][1]
        xmax = self.detectivePawnsLocations[j][2]
        ymax = self.detectivePawnsLocations[j][3]

        if((xmin < x < xmax) and (ymin < y < ymax)):
          if positions[j] == 0:
            positions[j]=assignedObjects[i]
          else:
            positions[j]=[positions[j], assignedObjects[i]]

    return positions

  def actionPawnClick(self, click_coordinates):
    actionPawnIndex = None

    for i in range(len(self.actionPawnsBb)):
      if ((self.actionPawnsBb[i][0] < click_coordinates[0] <  self.actionPawnsBb[i][2]) and
      (self.actionPawnsBb[i][1] < click_coordinates[1] <  self.actionPawnsBb[i][3])):
        actionPawnIndex= i

    return actionPawnIndex

  def actionPawnUsed(self, actionPawn):
    actionPawns = self.boardReference.getActionPawns()

    actionPawnIndex = actionPawns.index(actionPawn.name)

    del self.actionPawnsBb[actionPawnIndex]
    del actionPawns[actionPawnIndex]
    self.boardReference.setActionPawns(actionPawns)
