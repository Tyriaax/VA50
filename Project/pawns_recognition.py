import cv2.cv2

from GameBoard import *
from boundingBoxes import *
from cnn import *
from probabilities import *
from samples import *

class PawnsRecognitionHelper:
  selectedSamplesQuality = "LQ"

  selectedSamplesResolution = 200

  maxThrownActionPawnsNumber = 4

  def __init__(self, gameBoard):
    self.applyCircleMask = True
    self.applySharpenFilter = True

    self.boardReference = gameBoard
    self.detectivePawnsLocations = list()
    self.actionPawnsBb = list()
    self.actionPawnCNN = cnnHelper("AP")
    self.detectivePawnCNN = cnnHelper("DP")

  def GetScreenPortion(self,img, coordinates):
    # We store here the coordinates of the 4 points of homography
    self.coordinates = coordinates
   
    self.cardSize = (int((coordinates[2]-coordinates[0])/3), int((coordinates[3]-coordinates[1])/3))

    # Specifies the range of area for the bouding boxes that we will find
    self.bBmaxArea = (self.cardSize[0]*self.cardSize[1])/2
    self.bBminArea = (self.cardSize[0]*self.cardSize[1])/12

    height, width = img.shape[0], img.shape[1]

    # Specifies the number of pixels that we want to add around for the detective pawns zone
    dPOverlayCardRatio = 0.6
    self.dpOverlaySizePx = int(self.cardSize[0] * dPOverlayCardRatio)

    # Generate Masks for detection of action and detective pawns
    self.detectivePawnsRectangle = [coordinates[0] - self.dpOverlaySizePx, coordinates[1] - self.dpOverlaySizePx,
                                    coordinates[2] + self.dpOverlaySizePx, coordinates[3] + self.dpOverlaySizePx]

    self.DPmask = np.full((height, width), 0, dtype=np.uint8)
    cv2.rectangle(self.DPmask, (self.detectivePawnsRectangle[0], (self.detectivePawnsRectangle[1])),(self.detectivePawnsRectangle[2], (self.detectivePawnsRectangle[3])), 255, -1)
    cv2.rectangle(self.DPmask, (coordinates[0],coordinates[1]), (coordinates[2],coordinates[3]), 0, -1)

    self.APmask = np.full((height, width), 255, dtype=np.uint8)
    cv2.rectangle(self.APmask, (self.detectivePawnsRectangle[0], (self.detectivePawnsRectangle[1])),(self.detectivePawnsRectangle[2], (self.detectivePawnsRectangle[3])), 0, -1)

    # Here we build the list of coordinates for the detective pawns to be considered facing a line of cards
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

    # Here we apply mask on the image before finding the bounding boxes
    maskedimg = cv2.bitwise_and(img, img, mask=self.DPmask)
    selectedimg = maskedimg[self.detectivePawnsRectangle[1]:self.detectivePawnsRectangle[3],self.detectivePawnsRectangle[0]:self.detectivePawnsRectangle[2]]

    boundingBoxes = getBoundingBoxes(selectedimg, self.bBmaxArea, self.bBminArea)
    boundingBoxes = addOffsetToBb(boundingBoxes,self.detectivePawnsRectangle[0],self.detectivePawnsRectangle[1]) # We add back the offset to the boundingboxes since we cropped the image

    cnnProbabilities = []
    for i in range(min(len(boundingBoxes), len(DetectivePawns))):
      # We can choose to appy a circle mask to the pawn image
      if self.applyCircleMask:
        currentimg = self.CircleMask(img[boundingBoxes[i][1]:boundingBoxes[i][3], boundingBoxes[i][0]:boundingBoxes[i][2]],self.selectedSamplesResolution)
      else:
        currentimg = img[boundingBoxes[i][1]:boundingBoxes[i][3], boundingBoxes[i][0]:boundingBoxes[i][2]]

      cnnProbabilities.append(self.detectivePawnCNN.ComputeImage(currentimg,self.selectedSamplesResolution))

    if (len(boundingBoxes) > 0):
      assignedObjects = linearAssignment(cnnProbabilities, DetectivePawns)

      DPpawnspositions = self.getDetectivePawnsPositions(assignedObjects,boundingBoxes) # This function is called to convert the assignement to the list of pawns by position
      self.boardReference.setDetectivePawns(DPpawnspositions)
      self.detectivePawnsBb = boundingBoxes[0:len(DetectivePawns)]
      self.detectivePawnsBbOrder = assignedObjects

  def ComputeActionPawns(self, img):

    # Here we apply mask on the image before finding the bounding boxes
    maskedimg = cv2.bitwise_and(img, img, mask=self.APmask)
    boundingBoxes = getBoundingBoxes(maskedimg, self.bBmaxArea, self.bBminArea, inspectInsideCountours = True)

    cnnProbabilities = []
    for i in range(min(len(boundingBoxes), self.maxThrownActionPawnsNumber)):
      # We can choose to appy a circle mask to the pawn image
      if self.applyCircleMask:
        currentimg = self.CircleMask( maskedimg[boundingBoxes[i][1]:boundingBoxes[i][3], boundingBoxes[i][0]:boundingBoxes[i][2]], self.selectedSamplesResolution)
      else:
        currentimg = maskedimg[boundingBoxes[i][1]:boundingBoxes[i][3], boundingBoxes[i][0]:boundingBoxes[i][2]]

      cnnProbabilities.append(self.actionPawnCNN.ComputeImage(currentimg,self.selectedSamplesResolution))

    if (len(boundingBoxes) > 0):
      cnnProbabilities = FormatActionPawnProbabilitiesMissingSample(cnnProbabilities)
      assignedObjects = linearAssignment(cnnProbabilities, ActionPawns)
      self.boardReference.setActionPawns(assignedObjects)
      self.actionPawnsBb = boundingBoxes[0:self.maxThrownActionPawnsNumber]

  # Function to add a circle mask to a pawn image
  # Needs to be resized in square first with given dimension
  def CircleMask(self, img, resolution):
    dim = (resolution, resolution)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    height, width = img.shape[:2]
    mask = np.full((height, width), 0, dtype=np.uint8)
    cv2.circle(mask, ( width // 2, height // 2), height // 2, 255, -1)

    img = cv2.bitwise_and(img, img, mask=mask)

    return img

  def ComputeFrame(self, img):
    self.ComputeActionPawns(img)
    self.ComputeDetectivePawns(img)

  def DrawFrame(self, img):
    img = self.DrawZonesRectangles(img, drawOffset=True)
    img = self.DrawDetectivePawns(img)
    img = self.DrawActionPawns(img)

    return img

  def DrawDetectivePawns(self, img):
    img = drawRectanglesWithAssignment(img, self.detectivePawnsBbOrder, self.detectivePawnsBb)

    return img

  def DrawActionPawns(self, img):
    actionPawns = self.boardReference.getActionPawns()
    img = drawRectanglesWithAssignment(img, actionPawns, self.actionPawnsBb)

    return img

  def DrawDetectivePawnByName(self, img, detectivePawnName):
    detectivePawns = self.detectivePawnsBbOrder
    if detectivePawnName in detectivePawns:
      detectivePawnsIndex = detectivePawns.index(detectivePawnName)
      img = drawRectanglesWithAssignment(img, [detectivePawnName], [self.detectivePawnsBb[detectivePawnsIndex]])

    return img

  # This function draws the rectangles for the cards and detective pawns part, with the option to add to more zones for a little offset
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

  #This function converts the images and there assignement to a list representing the detective pawns based on their position on the board
  def getDetectivePawnsPositions(self, assignedObjects, boundingBoxes):
    positions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(min(len(boundingBoxes), len(DetectivePawns))):
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
            # This is in the case 2 or 3 pawns are on the same position
            if type(positions[j]) == type(str()):
              positions[j] = [positions[j], assignedObjects[i]]
            else:
              positions[j].append(assignedObjects[i])

    return positions

  # This function takes the coordinates of a click and gives the name of the action pawn clicked at this position if there exists one
  def actionPawnClick(self, click_coordinates):
    actionPawnIndex = None

    for i in range(len(self.actionPawnsBb)):
      if ((self.actionPawnsBb[i][0] < click_coordinates[0] <  self.actionPawnsBb[i][2]) and
      (self.actionPawnsBb[i][1] < click_coordinates[1] <  self.actionPawnsBb[i][3])):
        actionPawnIndex= i

    return actionPawnIndex

  # This function removes an action pawn that has been used
  def actionPawnUsed(self, actionPawn):
    actionPawns = self.boardReference.getActionPawns()

    actionPawnIndex = actionPawns.index(actionPawn.name)

    del self.actionPawnsBb[actionPawnIndex]
    del actionPawns[actionPawnIndex]
    self.boardReference.setActionPawns(actionPawns)
