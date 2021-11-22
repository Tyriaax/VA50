from enum import Enum

from samples import *
from boundingBoxes import *
from probabilities import *
import numpy
from GameBoard import *

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

  def __init__(self, height, width, gameBoard):
    if self.selectedSamplesQuality == "HQ":
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "Cards"))
    elif self.selectedSamplesQuality == "LQ":
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "CardsWithContour3"))

    self.boardReference = gameBoard
    self.cardRectangle = list()
    self.rectangles = list()

    [self.samplesSiftInfos, self.samplesHistograms] = loadSamples(path)

 

  def GetScreenPortions(self, img,coordinates):
    height, width = img.shape[0],img.shape[1] 
    width_portion = int(width / 3)
    height_portion = int(height / 3)
    proportionh = int(0.28 * height_portion)
    proportionw = int(0.28 * width_portion)

    for i in range(3):
      for j in range(3):
        x = j * width_portion + proportionw
        w = (j + 1) * width_portion - proportionw
        y = i * height_portion + proportionh
        h = (i + 1) * height_portion - proportionh

        self.rectangles.append([x,y,w,h])
        self.cardRectangle.append([width_portion * j, i * height_portion, (j + 1) * width_portion, (i + 1) * height_portion])

    self.coordinates = coordinates

  def ComputeFrame(self, img):
    selectedimg = img[self.coordinates[1]:self.coordinates[3], self.coordinates[0]:self.coordinates[2]]

    boundingBoxes = getCirclesBb(selectedimg, self.rectangles)

    if(len(boundingBoxes) > 0):
      siftProbabilities = []
      histoProbabilities = []
      for boundingBox in boundingBoxes:
        currentimg = selectedimg[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]

        #currentimg = increaseImgColorContrast(currentimg)

        siftProbabilities.append(sift_detection(currentimg, self.samplesSiftInfos))
        histoProbabilities.append(histogramProbabilities(currentimg, self.samplesHistograms))#histogram_Probabilities(currentimg, self.samplesHistograms))
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0, 1])

      #selectedimg = drawRectangleWithProbabilities(selectedimg, finalProbabilities, boundingBoxes, Cards, cards)

      assignedObjects = linearAssignment(finalProbabilities,Cards)
      self.boardReference.setCards(assignedObjects)
      selectedimg = drawRectanglesWithAssignment(selectedimg, assignedObjects, boundingBoxes)

    img[self.coordinates[1]:self.coordinates[3],self.coordinates[0]:self.coordinates[2]] = selectedimg
    return img

  def isInLineOfSight(self, img, detectivePosition, jackPosition):
    cardBackMask = ((0,6,119),(28,71,255))
    copy = img.copy()
    selectedimg = copy[self.coordinates[1]:self.coordinates[3], self.coordinates[0]:self.coordinates[2]]


    for i in range(len(self.cardRectangle)):

      yCoordCardMid = (self.cardRectangle[i][3] - self.cardRectangle[i][1])/2
      portionImg = selectedimg[self.cardRectangle[i][1]:self.cardRectangle[i][3], self.cardRectangle[i][0]:self.cardRectangle[i][2]]
      currentImg = cv2.cvtColor(portionImg, cv2.COLOR_BGR2GRAY)
      #currentImg = cv2.GaussianBlur(currentImg, (7,7), cv2.BORDER_DEFAULT)
      #kernel = np.ones((6,6), np.uint8)
      #currentImg = cv2.erode(currentImg, kernel, cv2.BORDER_REFLECT) 

      th, imageThresholded= cv2.threshold(src=currentImg, thresh=100, maxval= 255, type=cv2.THRESH_BINARY)
      cv2.imshow(str(i), imageThresholded)

        """
        currentImg = cv2.cvtColor(portionImg, cv2.COLOR_BGR2HSV)

        portionImg = cv2.GaussianBlur(portionImg, (5,5), cv2.BORDER_DEFAULT)

        currentImg = cv2.cvtColor(portionImg, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(currentImg, cardBackMask[0], cardBackMask[1])
        blue = cv2.bitwise_and(portionImg, portionImg, mask = mask) #Affichage du mask
        cv2.imshow(str(i), blue)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
          for contour in contours:
            if cv2.contourArea(contour) > 150:
              x, y, w, h = cv2.boundingRect(contour)
              cv2.rectangle(portionImg, (x,y), (x + w, y + h), (0,255,255), 3)
              cv2.putText(portionImg, "path", (x,y),1,1,(0,0,255),3)"""
              
              
