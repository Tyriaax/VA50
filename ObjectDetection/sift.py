import cv2
import numpy as np
import random

class SiftInfo:
  def __init__(self, img = None, squaredim = None, circleMask = False, applySharpen = False):
    """
    if (applySharpen):
      sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
      img = cv2.filter2D(img, -1, sharpen_kernel)
    """

    if(squaredim):
      dim = (squaredim, squaredim)
      img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    # """
    if (applySharpen):
      sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
      img = cv2.filter2D(img, -1, sharpen_kernel)
    # """

    self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if circleMask:
      height, width = img.shape[:2]
      mask = np.full((height, width), 0, dtype=np.uint8)
      cv2.circle(mask, (height//2,width//2), height//2, 255, -1)

      self.img = cv2.bitwise_and(self.img, self.img, mask=mask)

    #self.img = cv2.equalizeHist(self.img)
    sift = cv2.SIFT_create()

    if circleMask:
      self.keypoints, self.descriptors = sift.detectAndCompute(img, mask)
    else:
      self.keypoints, self.descriptors = sift.detectAndCompute(img, None)

def sift_detection(img, samplesSiftInfos, resolution = None, circleMask = False, applySharpen = False):
  minMatches = 0
  knnDistance = 0.7 # Or 0.3 ?

  index_params = dict(algorithm=1, trees=5)
  search_params = dict(checks=50)

  siftInfosImg = SiftInfo(img,resolution, circleMask, applySharpen)

  """
  cv2.imshow("Img",siftInfosImg.img)
  """

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  numberoffoundpoints = []

  # i = 0
  for sampleSiftInfo in samplesSiftInfos:
    """
    cv2.imshow("ImgSample" + str(i), sampleSiftInfo.img)
    i = i + 1
    """
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

  probabilities = [0] * len(numberoffoundpoints)

  if totalsum > 2*minMatches:
    for i in range(len(numberoffoundpoints)):
      probabilities[i] = numberoffoundpoints[i]/totalsum

  return probabilities