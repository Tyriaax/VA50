import cv2
import numpy as np
from matplotlib import pyplot as plt
from HistogramColorClassifier import HistogramColorClassifier
import os
from PIL import ImageEnhance, Image

#Defining the classifier

def addToClassifier(my_classifier : HistogramColorClassifier, model, file):
  my_classifier.addModelHistogram(model, file.split(".")[0])


def histogramProbabilities(image, my_classifier : HistogramColorClassifier):
  #image = cv2.imread(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Samples', 'LQ', 'CardsWithoutContour','CBlue.jpg'))) #Load the image

  """image = Image.open(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Samples', 'LQ', 'CardsWithoutContour','CBlue.jpg')))
  enhancer = ImageEnhance.Contrast(image)
  enhanced_im = np.array(enhancer.enhance(1.8))
  cv2.imshow("res", enhanced_im)"""

  #Get a numpy array which contains the comparison values
  #between the model and the input image
  comparison_array = my_classifier.returnHistogramComparisonArray(image, method="intersection")
  #Normalisation of the array
  comparison_distribution = comparison_array / np.sum(comparison_array)

  return comparison_distribution.tolist()