import cv2
import os
import numpy as np
from enum import Enum

from samples import *
from homography import *
from boundingBoxes import *
from probabilities import *

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

def video_recognition():
  window_name = "JACK"
  height = 720
  width = 1280

  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #Tochange in case

  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  selectedEnum = DetectivePawns
  selectedSamplesQuality = "LQ"
  selectedSamplesResolution = 400

  if selectedEnum == DetectivePawns:
    path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Samples",selectedSamplesQuality, "Pawns","DetectivePawns"))
  elif selectedEnum == ActionPawns:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Samples",selectedSamplesQuality ,"Pawns", "ActionPawns"))

  if(selectedSamplesQuality == "LQ"):
    [samplesSiftInfos, samplesHistograms] = loadSamples(path, selectedSamplesResolution)
  elif (selectedSamplesQuality == "HQ"):
    [samplesSiftInfos, samplesHistograms] = loadSamples(path, selectedSamplesResolution)

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
      finalProbabilities = combineProbabilities([siftProbabilities, histoProbabilities], [0.5, 0.5])
      img = drawRectangleWithProbabilities(img,finalProbabilities,boundingBoxes,[],selectedEnum)

    cv2.imshow(window_name, img)
    #cv2.waitKey(1000)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break  
    
  cap.release()
  cv2.destroyAllWindows()


video_recognition()

