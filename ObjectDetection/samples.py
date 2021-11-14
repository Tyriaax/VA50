import os
from sift import *
from histo import *

def loadSamples(path, resolution = None):
  dir = os.listdir(path)

  samplesSiftInfoList = []
  samplesHistoList = []

  for image in dir:
    img = cv2.imread(os.path.join(path, image))
    if resolution:
      samplesSiftInfoList.append(SiftInfo(img,resolution))
    else:
      samplesSiftInfoList.append(SiftInfo(img))
    samplesHistoList.append(getHisto(img))

  return [samplesSiftInfoList,samplesHistoList]

