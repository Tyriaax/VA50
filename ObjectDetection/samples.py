import os
from sift import *
from histo import *
from HistogramComparison import*

def loadSamples(path, resolution = None):
  dir = os.listdir(path)

  samplesSiftInfoList = []
  samplesHistoList = []
  histoClassifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128], hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')

  for image in dir:
    img = cv2.imread(os.path.join(path, image))
    if resolution:
      samplesSiftInfoList.append(SiftInfo(img,resolution))
    else:
      samplesSiftInfoList.append(SiftInfo(img))

    #samplesHistoList.append(getHisto(img))
    addToClassifier(histoClassifier, img, image)

  return [samplesSiftInfoList, histoClassifier]  #samplesHistoList]

