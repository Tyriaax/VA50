from enum import Enum
from os import path
from samples import *
from boundingBoxes import *
from probabilities import *
import numpy
from GameBoard import *
from cnn import *

class SamplesQuality(Enum):
  LQ = 1
  HQ = 2
  LAHQ = 3

class CardsRecognitionHelper:
  selectedSamplesQuality = SamplesQuality.LQ

  selectedSamplesResolution = 200

  def __init__(self, height, width, gameBoard):
    pathHQ = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", "HQ", "Cards"))
    pathLQ = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", "LQ", "CardsWithContour"))

    self.boardReference = gameBoard
    self.cardRectangle = list()
    self.rectangles = list()
    self.gameBoard = np.zeros((9,2), dtype= np.chararray)

    if self.selectedSamplesQuality == SamplesQuality.HQ:
      [self.samplesSiftInfos, self.samplesHistograms, self.samplesZncc] = loadSamples(pathHQ,self.selectedSamplesResolution)
    else:
      [self.samplesSiftInfos, self.samplesHistograms, self.samplesZncc] = loadSamples(pathLQ, self.selectedSamplesResolution)
      """
      if self.selectedSamplesQuality == SamplesQuality.LAHQ:
        [self.samplesSiftInfos2, self.samplesHistograms2, self.samplesZncc] = loadSamples(pathHQ, self.selectedSamplesResolution)
      """

    # self.selectedCirclesResolution = int(0.42*self.selectedSamplesResolution)

    self.cardsCNN = cnnHelper("CARDS")

  def GetScreenPortions(self, img,coordinates):
    cardToCircleProportion = 0.26
    img = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
    height, width = img.shape[0],img.shape[1] 
    width_portion = int(width / 3)
    height_portion = int(height / 3)
    proportionh = int(cardToCircleProportion * height_portion)
    proportionw = int(cardToCircleProportion * width_portion)

    for i in range(3):
      for j in range(3):
        x = j * width_portion + proportionw
        w = (j + 1) * width_portion - proportionw
        y = i * height_portion + proportionh
        h = (i + 1) * height_portion - proportionh

        self.rectangles.append([x,y,w,h])
        self.cardRectangle.append([width_portion * j, i * height_portion, (j + 1) * width_portion, (i + 1) * height_portion])

    self.rectangles = addOffsetToBb(self.rectangles,coordinates[0],coordinates[1])
    self.cardRectangle = addOffsetToBb(self.cardRectangle, coordinates[0], coordinates[1])
    self.coordinates = coordinates

  def ComputeFrame(self, img):
    self.ComputeCardsOrientation(img)
    self.ComputeCards(img)


  def ComputeCards(self, img, dontCheckReturnedAndInnocentedCards = False):
    if(len(self.rectangles) > 0):
      """
      siftProbabilities = []
      histoProbabilities = []
      """
      znccProbabilities = []
      cnnProbabilities = []

      """Try with more than 1 sample
      siftProbabilities2 = []
      histoProbabilities2 = []
      """

      boundingBoxes = getCirclesBb(img,self.rectangles)

      # Remove the samples of innocentCards
      innocentCards = self.boardReference.getInnocentCardsIndex()
      #samplesSift = self.samplesSiftInfos.copy()
      samplesZncc = self.samplesZncc.copy()
      j = 0
      for i in range(len(self.rectangles)):
        if i in innocentCards:
          #samplesSift.pop(j)
          samplesZncc.pop(j)
        else:
          j = j+1

      for i in range(len(self.rectangles)):
        # Only add the probabilities if the card is on the front side
        if(self.gameBoard[i][1] == "front"):
          cardimg = img[self.cardRectangle[i][1]:self.cardRectangle[i][3], self.cardRectangle[i][0]:self.cardRectangle[i][2]]
          circleimg = img[self.rectangles[i][1]:self.rectangles[i][3], self.rectangles[i][0]:self.rectangles[i][2]]

          """
          siftProbabilities.append(sift_detection(circleimg, samplesSift, self.selectedSamplesResolution))
          histoProbability = histogramProbabilities(circleimg, self.samplesHistograms)
          """

          znccProbabilities.append(zncc_score(circleimg,samplesZncc, orientation=self.gameBoard[i][0]))
          cnnProbability = self.cardsCNN.ComputeImage(cardimg)

          # Remove the unwanted HistogramProbabilities and CNNProbabilities
          j = 0
          for i in range(len(self.rectangles)):
            if i in innocentCards:
              # histoProbability.pop(j)
              cnnProbability.pop(j)
            else:
              j = j + 1

          #histoProbabilities.append(histoProbability)
          cnnProbabilities.append(cnnProbability)

          """
          if self.selectedSamplesQuality == SamplesQuality.LAHQ:
            siftProbabilities2.append(sift_detection(cardimg, self.samplesSiftInfos, self.selectedCirclesResolution))
            histoProbabilities2.append(histogramProbabilities(circleimg, self.samplesHistograms))
          """

      """
      if self.selectedSamplesQuality == SamplesQuality.LAHQ:
        finalProbabilities = combineProbabilities([siftProbabilities,siftProbabilities2, histoProbabilities, histoProbabilities2, znccProbabilities], [0.0,0.0,0.0,0.0,1])
      else:
      """

      #finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities, znccProbabilities], [0,0,1]) # TODO PUT BACK
      finalProbabilities = combineProbabilities([znccProbabilities, cnnProbabilities], [0.4, 0.6])

      # We also need to change the list of cards we give to the assignment algorithm
      CurrentCards = []
      for card in Cards:
        if card.value not in innocentCards:
          CurrentCards.append({'name': card.name, 'value': card.value})

      assignedObjects = linearAssignment(finalProbabilities, CurrentCards)

      # Now we assign back the objects to only the front turned cards TODO ADD SAFETY IN CASE A CARD INNOCENTED IS NOT DETECTED AS TURNED
      finalAssignedObjects = []
      j = 0 # Counts the cards returned
      for i in range(len(self.gameBoard)):
        if (self.gameBoard[i][1] == "front"):
          finalAssignedObjects.append(assignedObjects[j])
          j = j + 1
        else:
          finalAssignedObjects.append(None)

      self.boardReference.setCards(finalAssignedObjects)

  def DrawFrame(self, img):
    cards = self.boardReference.getCards()
    img = drawRectanglesWithAssignment(img, cards, self.rectangles)

    return img

  def DrawBoxesByIndex(self, img, indexes):
    for index in indexes:
      cv2.rectangle(img, (self.rectangles[index][0], self.rectangles[index][1]), (self.rectangles[index][2], self.rectangles[index][3]), (0, 255, 0), 2)

    return img

  def DrawBoxesByName(self, img, names):
    cards = self.boardReference.getCards()

    for i in range(len(cards)):
      if cards[i] in names:
        cv2.rectangle(img, (self.rectangles[i][0], self.rectangles[i][1]), (self.rectangles[i][2], self.rectangles[i][3]), (0, 255, 0), 2)

    return img
  
  def ComputeCardsOrientation(self, img):
    self.gameBoard = np.zeros((9,2), dtype= np.chararray)
    self.GetEmptySideCards(img)
    self.getFrontSideCards(img)

  def getMeanPathValuesCards(self, img, heightCard, widthCard):
    pathPortion = 0.3
    offset = 7
    meanValues = list()
    pathValues = [
      img[0:int(pathPortion*(heightCard - 1)), int(widthCard/2) - offset:int(widthCard/2) + offset], #up
      img[(heightCard - 1) - int(pathPortion*(heightCard - 1)):heightCard - 1, int(widthCard/2) - offset:int(widthCard/2) + offset], #down
      img[int(heightCard/2) - offset:int(heightCard/2) + offset, 0:int(pathPortion*(widthCard - 1))], #left
      img[int(heightCard/2) - offset:int(heightCard/2) + offset, (widthCard - 1) - int(pathPortion*(widthCard - 1)):widthCard - 1] #right
      ]
  
    return pathValues

  def BinarizeCard(self, img, heightCard, widthCard):
    meanValues = list()
    portionImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pathValues = self.getMeanPathValuesCards(img, heightCard, widthCard)

    for pathValue in pathValues:
      meanValues.append(pathValue.mean())

    binValue = sum(meanValues)/len(meanValues)

    #portionImg = cv2.GaussianBlur(portionImg, (3,3), cv2.BORDER_DEFAULT)
    kernel = np.ones((5,5), np.uint8)
    portionImg = cv2.dilate(portionImg, kernel, iterations=1)

    th, cardThresholded = cv2.threshold(src=portionImg, thresh= binValue, maxval= 255, type=cv2.THRESH_BINARY)

    return cardThresholded

  def getFrontSideCards(self, img):
    index = 0
    if(len(self.rectangles) == 9):
      for boundingBox in self.cardRectangle:
        if np.array_equal(self.gameBoard[index], np.array([0, 0], dtype=np.chararray)):
          currentimg = img[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
          heightCard,widthCard, _ = currentimg.shape
          currentimgbinar = self.BinarizeCard(currentimg, heightCard, widthCard)
          a = self.getMeanPathValuesCards(currentimgbinar, heightCard, widthCard)
          
          meanValues = list()
          for pathValue in a:
            meanValues.append(pathValue.mean())

          up, down, left, right = meanValues[0], meanValues[1], meanValues[2], meanValues[3]
          sortedMean = sorted(meanValues, reverse=True)[:3]
          
          if up in sortedMean and down in sortedMean and left in sortedMean:
            self.gameBoard[index] = ["left", "front"]
          elif up in sortedMean and down in sortedMean and right in sortedMean:
            self.gameBoard[index] = ["right", "front"]
          elif right in sortedMean and down in sortedMean and left in sortedMean:
            self.gameBoard[index] = ["down", "front"]
          elif right in sortedMean and up in sortedMean and left in sortedMean:
            self.gameBoard[index] = ["up", "front"]

        index += 1
    self.boardReference.setCardsState(self.gameBoard)

  def GetEmptySideCards(self, img):

    index = 0
    if(len(self.rectangles) > 0):
      for boundingBox in self.cardRectangle:
        horizontalHalfLeft = False
        horizontalHalfRight = False
        horizontalLine = True

        verticalHalfUp = False
        verticalHalfDown = False
        verticalLine = True

        currentimg = img[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
        heightCard,widthCard, _ = currentimg.shape
        currentimgbinar = self.BinarizeCard(currentimg, heightCard, widthCard)

        pathPortion = 0.15
        for a, b in zip(range(int(pathPortion * widthCard),widthCard), range(int(widthCard - pathPortion * widthCard), int(pathPortion * widthCard), -1)):
          if currentimgbinar[int(heightCard/2) - 1][a] != 255:
            horizontalLine = False
            if a >= int(widthCard/2):
              horizontalHalfLeft = True

          if currentimgbinar[int(heightCard/2) - 1][b] != 255:
            horizontalLine = False
            if b <= int(widthCard/2):
              horizontalHalfRight = True

        for c, d in zip(range(int(pathPortion * heightCard), heightCard), range(int(heightCard - pathPortion * heightCard), int(pathPortion * heightCard), -1)):
          if currentimgbinar[c][int(widthCard/2) - 1] != 255:
            verticalLine = False
            if c >= int(heightCard/2):
              verticalHalfUp = True

          if currentimgbinar[d][int(widthCard/2) - 1] != 255:
            verticalLine = False
            if d <= int(heightCard/2):
              verticalHalfDown = True

        if verticalLine and not horizontalLine:
          if horizontalHalfLeft and not horizontalHalfRight:
            self.gameBoard[index] = ["Left", "returned"]
          elif not horizontalHalfLeft and horizontalHalfRight:
            self.gameBoard[index] = ["Right", "returned"]
        elif horizontalLine and not verticalLine:
          if verticalHalfUp and not verticalHalfDown:
            self.gameBoard[index] = ["Up", "returned"]
          elif not verticalHalfUp and verticalHalfDown:
            self.gameBoard[index] = ["Down", "returned"]
        elif horizontalLine and verticalLine:
          self.gameBoard[index] = ["Cross", "returned"]

        index += 1
  
  def InSight(self, detectivePos, orientation, cards : list, heightCard, widthCard, inSightList):

    if len(cards) == 0:
      return

    pathPortion = 0.15
    pursue = False
    if orientation == "Horizontal":
      if detectivePos[1] == 0: #La cart le détective est sur la partie gauche
        card = cards.pop(-1)
        if card[0][int(heightCard/2) - 1][int(pathPortion * widthCard)] == 255:
          inSightList.append(card)
          if card[0][int(heightCard/2) - 1][int(widthCard - pathPortion * widthCard)] == 255:
            pursue = True

      elif detectivePos[1] == 4: #La cart le détective est sur la partie droite
        card = cards.pop(0)
        if card[0][int(heightCard/2) - 1][int(widthCard - pathPortion * widthCard)] == 255:
          inSightList.append(card)
          if card[0][int(heightCard/2) - 1][int(pathPortion * widthCard)] == 255:
            pursue = True

    elif orientation == "Vertical":
      if detectivePos[0] == 0: #La cart le détective est sur la partie haute
        card = cards.pop(0)
        if card[0][int(pathPortion * heightCard)][int(widthCard/2) - 1] == 255:
          inSightList.append(card)
          if card[0][int(heightCard - pathPortion * heightCard)][int(widthCard/2) - 1] == 255:
            pursue = True

      elif detectivePos[0] == 4: #La cart le détective est sur la partie basse
        card = cards.pop(-1)
        if card[0][int(heightCard - pathPortion * heightCard)][int(widthCard/2)- 1] == 255:
          inSightList.append(card)
          if card[0][int(pathPortion * heightCard)][int(widthCard/2)- 1] == 255:
            pursue = True

    if pursue == True:
      self.InSight(detectivePos, orientation, cards, heightCard, widthCard, inSightList)
    else:
      return

  def IsInLineOfSight(self, img):
 
    possibleDetectivePos = (1,2,3)
    copy = img.copy()

    inSightList = list()
    jackInSight = False
    inSightPos = list()

    cards = self.boardReference.getCards()
    cards = np.array(cards)
    cards.resize((3,3))

    jackPosition = self.boardReference.getJackPos()
    detectivesPosition = self.boardReference.getDetectivesPos()

    if None not in [jackPosition, detectivesPosition]:
      for detectivePosition in detectivesPosition:
        cardList = []
        sight = ""
        
        if detectivePosition[0] in possibleDetectivePos and detectivePosition[1] in [0,4]:#jackPosition[0] == detectivePosition[0]:
          sight = "Horizontal"

        elif detectivePosition[1] in possibleDetectivePos and detectivePosition[0] in [0,4]: #jackPosition[1] == detectivePosition[1]:
          sight = "Vertical"

        if bool(sight):
          for i in range(3):
            if sight == "Vertical":
              index = i * 3 + detectivePosition[1] - 1
              rectangleCard = self.cardRectangle[index] 
              
            elif sight == "Horizontal":
              index = detectivePosition[0] * 3 - 1 - i 
              rectangleCard = self.cardRectangle[index] 
              
            portionImg = copy[rectangleCard[1]:rectangleCard[3], rectangleCard[0]:rectangleCard[2]]
            heightCard,widthCard, _ = portionImg.shape
            
            cardList.append([self.BinarizeCard(portionImg, heightCard, widthCard), index])

          self.InSight(detectivePosition, sight, cardList, heightCard, widthCard, inSightPos)
     
    if len(inSightPos) > 0 :
      for pos in inSightPos:            
        if jackPosition == pos[1]:
          jackInSight = True

    cards = self.boardReference.getCards()
    if jackInSight:
      print("JACK IN SIGHT")
      inSightList = [cards[index] for index in range(9)]
      for pos in inSightPos: 
        inSightList.pop(pos[1]) #Index des cartes en lignes de vues par les détectiives
  
      self.boardReference.addInnocentCards(inSightList)
      return True
    else:
      print("Jack not in sight") 
      for pos in inSightPos: 
        inSightList.append(cards[pos[1]]) #Index des cartes en lignes de vues par les détectiives  
      self.boardReference.addInnocentCards(inSightList)
      return False

