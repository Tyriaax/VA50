import cv2

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

def histogram_Probabilities(img, samplesHistograms):
  compareMethod = 2
  imghist = getHisto(img)

  comparisonValues = []
  for sampleHistogram in samplesHistograms:
    comparisonValues.append(cv2.compareHist(sampleHistogram, imghist, compareMethod))

  totalsum = sum(comparisonValues)

  probabilities = []

  for i in range(len(comparisonValues)):
    probabilities.append(comparisonValues[i] / totalsum)

  return probabilities