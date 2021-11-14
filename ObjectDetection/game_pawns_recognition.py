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

def load_kp_samples():
  squareDim = 400

  if selectedEnum == DetectivePawns:
    PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Samples",selectedSamplesQuality, "Pawns","DetectivePawns"))
  elif (selectedEnum == ActionPawns):
    PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples",selectedSamplesQuality ,"Pawns", "ActionPawns"))
  dir = os.listdir(PATH_SAMPLES)

  samplesInfoList = []

  for image in dir:
    img = cv2.imread(os.path.join(PATH_SAMPLES, image))
    samplesInfoList.append(SiftInfo(img))

  return samplesInfoList


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

def sift_detection_with_Bb(img, samplesInfos):
  minMatches = 50
  knnDistance = 0.1

  index_params = dict(algorithm=1, trees=5)
  search_params = dict(checks=50)

  siftInfosImg = SiftInfo(img)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  numberoffoundpoints = []

  for samplesInfo in samplesInfos:
    numberoffoundpoints.append(0)
    if(samplesInfo.descriptors is not None) and (siftInfosImg.descriptors is not None):
      if (len(samplesInfo.descriptors) >= 2) and (len(siftInfosImg.descriptors) >= 2):
        matches = flann.knnMatch(samplesInfo.descriptors, samplesInfo.descriptors, k = 2)
        foundpoints = []
        for m, n in matches:
          if m.distance < knnDistance * n.distance:
            foundpoints.append(m)

        if(len(foundpoints) > minMatches):
          numberoffoundpoints[len(numberoffoundpoints)-1] = len(foundpoints)

  totalsum = sum(numberoffoundpoints)

  probabilities = []
  if totalsum > 2*minMatches:
    for i in range(len(numberoffoundpoints)):
      probabilities.append(numberoffoundpoints[i]/totalsum)
  else:
    probabilities.append(0)

  return probabilities

def drawRectangleWithProbabilities(img,probabilities,boundingBoxes,alreadydetectedobjects):
  maxproba = []
  for i in range(len(probabilities)):
    maxproba.append(max(probabilities[i]))

  maxValueBb = max(maxproba)
  indexMaxValueBb = maxproba.index(maxValueBb)

  maxValue = max(probabilities[indexMaxValueBb])
  if (maxValue > 0):
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

  samplesInfos = load_kp_samples()

  bBmaxArea = height/3*width/3 #TODO Find better way ?
  bBminArea = height/10*width/10 #TODO Find better way ?

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
    for boundingBox in boundingBoxes:
      siftProbabilities.append(sift_detection_with_Bb(img[boundingBox[1]:boundingBox[3],boundingBox[0]:boundingBox[2]], samplesInfos))

    if (len(boundingBoxes) > 0):
      img = drawRectangleWithProbabilities(img,siftProbabilities,boundingBoxes,[])

    cv2.imshow(window_name, img)
    cv2.waitKey(500)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break  
    
  cap.release()
  cv2.destroyAllWindows()


video_recognition()

