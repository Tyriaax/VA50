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
  
  def inSight(self, detectivePos, orientation, cards : list, heightCard, widthCard, inSightList):

    if len(cards) == 0:
      return

    pursue = False
    if orientation == "Horizontal":
      if detectivePos[1] == 0: #La cart le détective est sur la partie gauche
        card = cards.pop(-1)
        if card[0][int(heightCard/2) - 1][0] == 255:
          inSightList.append(card)
          if card[0][int(heightCard/2) - 1][widthCard - 1] == 255:
            pursue = True

      elif detectivePos[1] == 4: #La cart le détective est sur la partie droite
        card = cards.pop(0)
        if card[0][int(heightCard/2) - 1][widthCard - 1] == 255:
          inSightList.append(card)
          if card[0][int(heightCard/2) - 1][0] == 255:
            pursue = True

    elif orientation == "Vertical":
      if detectivePos[0] == 0: #La cart le détective est sur la partie haute
        card = cards.pop(0)
        if card[0][0][int(widthCard/2) - 1] == 255:
          inSightList.append(card)
          if card[0][heightCard - 1][int(widthCard/2) - 1] == 255:
            pursue = True

      elif detectivePos[0] == 4: #La cart le détective est sur la partie basse
        card = cards.pop(-1)
        if card[0][heightCard - 1][int(widthCard/2)- 1] == 255:
          inSightList.append(card)
          if card[0][0][int(widthCard/2)- 1] == 255:
            pursue = True   

    if pursue == True:
      self.inSight(detectivePos, orientation, cards, heightCard, widthCard, inSightList)
    else:
      return

  def isInLineOfSight(self, img, board : list, detectivePosition : tuple, jackPosition : tuple):
    #DectivePosition : ligne, colonne
    possibleDetectivePos = (1,2,3)
    copy = img.copy()
    selectedimg = copy[self.coordinates[1]:self.coordinates[3], self.coordinates[0]:self.coordinates[2]]

    sight = ""
    inSightPos = []
    cardList = []

    cards = self.boardReference.getCards()
    cards = np.array(cards)
    cards.resize((3,3))

    if bool(detectivePosition) and bool(jackPosition):
      #Not empty
      if detectivePosition[0] in possibleDetectivePos and jackPosition[0] == detectivePosition[0]:
        sight = "Horizontal"

      elif detectivePosition[1] in possibleDetectivePos and jackPosition[1] == detectivePosition[1]:
        sight = "Vertical"

    if bool(sight):
      for i in range(3):

        if sight == "Vertical":
          rectangleCard = self.cardRectangle[i * 3 + detectivePosition[1] - 1] 
          index = i * 3 + detectivePosition[1] - 1
        elif sight == "Horizontal":
          rectangleCard = self.cardRectangle[detectivePosition[0] * 3 - 1 - i ] 
          index = detectivePosition[0] * 3 - 1 - i 

        portionImg = selectedimg[rectangleCard[1]:rectangleCard[3], rectangleCard[0]:rectangleCard[2]]
        heightCard,widthCard, _ = portionImg.shape
        portionImg = cv2.cvtColor(portionImg, cv2.COLOR_BGR2GRAY)
        portionImg = cv2.GaussianBlur(portionImg, (7,7), cv2.BORDER_DEFAULT)
        kernel = np.ones((5,5), np.uint8)
        portionImg = cv2.erode(portionImg, kernel, cv2.BORDER_REFLECT) 

        th, cardThreshold= cv2.threshold(src=portionImg, thresh= 95, maxval= 255, type=cv2.THRESH_BINARY)
        cv2.imshow(str(i),cardThreshold )

        cardList.append([cardThreshold, index])

      self.inSight(detectivePosition, sight, cardList, heightCard, widthCard, inSightPos)

      if len(inSightPos) > 0 :
        #print(len(inSightPos), "people in sight")
        for pos in inSightPos:
          x, y = pos[1]//3 + 1, pos[1]%3 + 1
          print(x,y)
          if jackPosition[0] == x and jackPosition[1] == y:
            print("JACK IN SIGHT")
            return True
        
    return False

