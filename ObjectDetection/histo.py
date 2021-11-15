import cv2
from HistogramComparison import*


def histogram_Probabilities(img, samplesHistograms : HistogramColorClassifier):

  probabilities = CompareHist(img, samplesHistograms)

  return probabilities