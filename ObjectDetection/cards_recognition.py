from enum import Enum

from samples import *
from boundingBoxes import *
from probabilities import *
import numpy
from game_board_recognition import*

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

  def __init__(self, height, width):
    if self.selectedSamplesQuality == "HQ":
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "Cards"))
    elif self.selectedSamplesQuality == "LQ":
      path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples", self.selectedSamplesQuality, "CardsWithContour3"))

    [self.samplesSiftInfos, self.samplesHistograms] = loadSamples(path)

  rectangles = []

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

    self.coordinates = coordinates

  def ComputeFrame(self, img):
    selectedimg = img[self.coordinates[1]:self.coordinates[3], self.coordinates[0]:self.coordinates[2]]
    boundingBoxes = getCirclesBb(selectedimg, self.rectangles)

    board = Board()
    cardBoard = board.getBoard()

    if(len(boundingBoxes) > 0):
      siftProbabilities = []
      histoProbabilities = []
      for boundingBox in boundingBoxes:
        currentimg = selectedimg[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
        siftProbabilities.append(sift_detection(currentimg, self.samplesSiftInfos))
        histoProbabilities.append(histogram_Probabilities(currentimg, self.samplesHistograms))
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.1, 0.9])
      selectedimg = drawRectangleWithProbabilities(selectedimg, finalProbabilities, boundingBoxes, Cards, cardBoard)

    img[self.coordinates[1]:self.coordinates[3],self.coordinates[0]:self.coordinates[2]] = selectedimg
    board.printBoard()
    return img
