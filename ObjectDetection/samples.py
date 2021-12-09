import os
from sift import *
from HistogramComparison import*
from zncc import *

def loadSamples(path, resolution = None):
  dir = os.listdir(path)

  samplesSiftInfoList = []
  histoClassifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128], hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')
  samplesZncc = []

  for image in dir:
    img = cv2.imread(os.path.join(path, image))
    if resolution:
      samplesSiftInfoList.append(SiftInfo(img,resolution))
    else:
      samplesSiftInfoList.append(SiftInfo(img))

    #samplesHistoList.append(getHisto(img))
    #img = increaseImgColorContrast(img)
    #cv2.imshow(image, img)
    addToClassifier(histoClassifier, img, image)

    samplesZncc.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

  return [samplesSiftInfoList, histoClassifier, samplesZncc]  

def increaseImgColorContrast(img):
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
  clahe = cv2.createCLAHE(clipLimit= 1.0, tileGridSize=(16,16))
  v = clahe.apply(v)
  hsv_img = np.dstack((h,s,v))
  rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
  return rgb
