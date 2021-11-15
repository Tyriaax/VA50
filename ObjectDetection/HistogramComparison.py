import cv2
import numpy as np
from matplotlib import pyplot as plt
from HistogramColorClassifier import HistogramColorClassifier
import os
from PIL import ImageEnhance, Image

#Defining the classifier
my_classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128], hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')

label_objects = []
path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Samples', 'LQ', 'CardsWithoutContour'))
for file in os.listdir(path):
  model = cv2.imread(path + '/' + file)
  print(model)
  my_classifier.addModelHistogram(model)
  label_objects.append(file.split(".")[0])


image = cv2.imread(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Samples', 'LQ', 'CardsWithoutContour','CBlue.jpg'))) #Load the image

"""image = Image.open(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Samples', 'LQ', 'CardsWithoutContour','CBlue.jpg')))
enhancer = ImageEnhance.Contrast(image)
enhanced_im = np.array(enhancer.enhance(1.8))
cv2.imshow("res", enhanced_im)"""

#Get a numpy array which contains the comparison values
#between the model and the input image
comparison_array = my_classifier.returnHistogramComparisonArray(image, method="intersection")
#Normalisation of the array
comparison_distribution = comparison_array / np.sum(comparison_array)

#Printing the arrays
print("Comparison Array:")
print(comparison_array)
print("Distribution Array: ")
print(comparison_distribution)

#Plotting a bar chart with the probability distribution
#If you are comparing more than 8 superheroes you have to
#change the total objects variable and add new labels in 
total_objects = 9
#label_objects = ('Flash', 'Batman', 'Hulk', 'Superman', 'Capt. America', 'Wonder Woman', 'Iron Man', 'Wolverine')
font_size = 20
width = 0.5 
plt.barh(np.arange(total_objects), comparison_distribution, width, color='r')
plt.yticks(np.arange(total_objects) + width/2.,label_objects , rotation=0, size=font_size)
plt.xlim(0.0, 1.0)
plt.ylim(-0.5, 8.0)
plt.xlabel('Probability', size=font_size)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()