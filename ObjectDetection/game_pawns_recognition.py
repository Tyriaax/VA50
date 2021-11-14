import cv2
import os
import numpy as np
from enum import Enum

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

selectedEnum = ActionPawns
selectedSamplesQuality = "LQ"

class SiftInfo:
  def __init__(self, img = None, squaredim = None):
    if(squaredim):
      dim = (squaredim, squaredim)
      img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    self.keypoints, self.descriptors = sift.detectAndCompute(img, None)

def getHisto(img):
  img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  h_bins = 50
  s_bins = 60
  histSize = [h_bins, s_bins]

  # hue varies from 0 to 179, saturation from 0 to 255
  h_ranges = [0, 180]
  s_ranges = [0, 256]
  ranges = h_ranges + s_ranges  # concat lists

  # Use the 0-th and 1-st channels
  channels = [0, 1]

  hist_img = cv2.calcHist([img_hsv], channels, None, histSize, ranges, accumulate=False)
  cv2.normalize(hist_img, hist_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

  return hist_img

def loadSamples():
  squareDim = 400

  if selectedEnum == DetectivePawns:
    PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Samples",selectedSamplesQuality, "Pawns","DetectivePawns"))
  elif (selectedEnum == ActionPawns):
    PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples",selectedSamplesQuality ,"Pawns", "ActionPawns"))
  dir = os.listdir(PATH_SAMPLES)

  samplesSiftInfoList = []
  samplesHistoList = []


  for image in dir:
    img = cv2.imread(os.path.join(PATH_SAMPLES, image))
    if selectedSamplesQuality == "HQ":
      samplesSiftInfoList.append(SiftInfo(img,squareDim))
    elif selectedSamplesQuality == "LQ":
      samplesSiftInfoList.append(SiftInfo(img))
    samplesHistoList.append(getHisto(img))

  return [samplesSiftInfoList,samplesHistoList]


list_board_coords = []
def mousePoints(event,x,y,flags,params):
  if event == cv2.EVENT_LBUTTONDOWN and len(list_board_coords) < 4:
    list_board_coords.append([x,y])

def get_homography_matrix(img, pts_src, w, h):
  pts_dst = np.array([[0,0],[w - 1, 0],[w-1, h-1],[0, h-1]])
  mat, status = cv2.findHomography(pts_src, pts_dst)

  return mat

def imageProcessingForFindingContours(img):
  # First we convert the frame to a grayscale image
  img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # We then use a median blur technique to reduce the noise
  img2 = cv2.medianBlur(img2, 5)

  # We then apply a sharpening filter to enhance the edges
  sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
  img2 = cv2.filter2D(img2, -1, sharpen_kernel)

  # We can then threshold to get a binary image
  img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

  # We invert the image so that the contours can get detected
  img2 = 255 - img2

  # We also apply a close morphology transformation to get rid of the imperfections inside the shape
  morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, morph_kernel, iterations=3)  # We apply a close transformation

  return img2

def getBoundingBoxes(img,maxarea,minarea):
  rectangles = []

  img2 = imageProcessingForFindingContours(img)

  # We then use findContours to get the contours of the shape
  cnts = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  # We then loop through all the detected contours to onl²y retrieve the ones with a desired area
  for c in cnts:
    area = cv2.contourArea(c)
    if minarea <= area <= maxarea:
      x, y, w, h = cv2.boundingRect(c)
      rectangle = [x, y, x+w, y+h]
      rectangles.append(rectangle)

  return rectangles

def sift_detection(img, samplesSiftInfos):
  minMatches = 0
  knnDistance = 0.3

  index_params = dict(algorithm=1, trees=5)
  search_params = dict(checks=50)

  siftInfosImg = SiftInfo(img)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  numberoffoundpoints = []

  for sampleSiftInfo in samplesSiftInfos:
    numberoffoundpoints.append(0)
    if(sampleSiftInfo.descriptors is not None) and (siftInfosImg.descriptors is not None):
      if (len(sampleSiftInfo.descriptors) >= 2) and (len(siftInfosImg.descriptors) >= 2):
        matches = flann.knnMatch(sampleSiftInfo.descriptors, sampleSiftInfo.descriptors, k = 2)
        foundpoints = []
        for m, n in matches:
          if m.distance < knnDistance * n.distance:
            foundpoints.append(m)

        if(len(foundpoints) > minMatches):
          numberoffoundpoints[len(numberoffoundpoints)-1] = len(foundpoints)

  totalsum = sum(numberoffoundpoints)

  probabilities = [0 for i in range(len(numberoffoundpoints))]

  if totalsum > 2*minMatches:
    for i in range(len(numberoffoundpoints)):
      probabilities[i] = numberoffoundpoints[i]/totalsum

  return probabilities

def histogram_Probabilities(img, samplesHistograms):
  compareMethod = 0
  imghist = getHisto(img)

  comparisonValues = []
  for sampleHistogram in samplesHistograms:
    comparisonValues.append(cv2.compareHist(sampleHistogram, imghist, compareMethod))

  totalsum = sum(comparisonValues)

  probabilities = []

  for i in range(len(comparisonValues)):
    probabilities.append(comparisonValues[i] / totalsum)

  return probabilities

def combineProbabilities(probabilitiesList,weights):
  numberOfProbabilitiesToCombine = len(probabilitiesList)
  numberOfObjects = len(probabilitiesList[0])
  numberOfSamples = len(probabilitiesList[0][0])

  combinedProbability = [[0 for i in range(numberOfSamples)] for j in range(numberOfObjects)]

  for i in range(numberOfProbabilitiesToCombine):
    for j in range(numberOfObjects):
      for k in range(numberOfSamples):
        combinedProbability[j][k] = combinedProbability[j][k] + probabilitiesList[i][j][k]*weights[i]

  for i in range(numberOfObjects):
    sumValue = sum(combinedProbability[i])
    for j in range(numberOfSamples):
      combinedProbability[i][j] = combinedProbability[i][j]/sumValue

  return combinedProbability

def drawRectangleWithProbabilities(img,probabilities,boundingBoxes,alreadydetectedobjects):
  minProbability = 0

  maxproba = []
  for i in range(len(probabilities)):
    maxproba.append(max(probabilities[i]))

  maxValueBb = max(maxproba)
  indexMaxValueBb = maxproba.index(maxValueBb)

  maxValue = max(probabilities[indexMaxValueBb])
  if (maxValue > minProbability):
    indexMaxValue = probabilities[indexMaxValueBb ].index(maxValue)

    if indexMaxValue not in alreadydetectedobjects:
      alreadydetectedobjects.append(indexMaxValue)
      img = drawRectangle(img, boundingBoxes[indexMaxValueBb], selectedEnum(indexMaxValue))
      boundingBoxes.remove(boundingBoxes[indexMaxValueBb])
      probabilities.remove(probabilities[indexMaxValueBb])
    else:
      probabilities[indexMaxValueBb][indexMaxValue] = 0

    if(len(boundingBoxes) > 0):
      img = drawRectangleWithProbabilities(img, probabilities, boundingBoxes,alreadydetectedobjects)

  return img

def drawRectangle(img,boundingBox,object):
  cv2.rectangle(img, (boundingBox[0], boundingBox[1]), (boundingBox[2], boundingBox[3]), (0, 255, 0), 2)
  cv2.putText(img, object.name, (boundingBox[0], boundingBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(36, 255, 12), 2)

  return img

def video_recognition():
  window_name = "JACK"
  height = 720
  width = 1280

  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #Tochange in case

  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  [samplesSiftInfos, samplesHistograms] = loadSamples()

  maxAreaDivider = 4
  minAreaDivider = 12
  bBmaxArea = height/maxAreaDivider*width/maxAreaDivider #TODO Find better way ?
  bBminArea = height/minAreaDivider*width/minAreaDivider #TODO Find better way ?

  homographymatrixfound = False

  _, img1 = cap.read()
  cv2.imshow(window_name, img1)
  while True:
    _, img = cap.read()

    if len(list_board_coords) < 4:
      cv2.setMouseCallback(window_name, mousePoints)
      for coord in list_board_coords:
        cv2.circle(img,coord,10,(0,255,0),-1)
    else:
      if not homographymatrixfound:
        homographymatrix = get_homography_matrix(img, np.array(list_board_coords), width, height)
        homographymatrixfound = True
      else:
        img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))

    boundingBoxes = getBoundingBoxes(img,bBmaxArea,bBminArea)

    siftProbabilities = []
    histoProbabilities = []
    for boundingBox in boundingBoxes:
      currentimg = img[boundingBox[1]:boundingBox[3],boundingBox[0]:boundingBox[2]]
      siftProbabilities.append(sift_detection(currentimg, samplesSiftInfos))
      histoProbabilities.append(histogram_Probabilities(currentimg, samplesHistograms))

    if (len(boundingBoxes) > 0):
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.7, 0.3])
      img = drawRectangleWithProbabilities(img,finalProbabilities,boundingBoxes,[])

    cv2.imshow(window_name, img)
    #cv2.waitKey(1000)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break  
    
  cap.release()
  cv2.destroyAllWindows()


video_recognition()

