import cv2

class SiftInfo:
  def __init__(self, img = None, squaredim = None):
    if(squaredim):
      dim = (squaredim, squaredim)
      img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #self.img = cv2.equalizeHist(self.img)
    sift = cv2.SIFT_create()
    self.keypoints, self.descriptors = sift.detectAndCompute(img, None)

def sift_detection(img, samplesSiftInfos):
  minMatches = 0
  knnDistance = 0.3

  index_params = dict(algorithm=1, trees=5)
  search_params = dict(checks=50)

  siftInfosImg = SiftInfo(img)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  numberoffoundpoints = []

  for sampleSiftInfo in samplesSiftInfos:
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

  probabilities = [0 for i in range(len(numberoffoundpoints))]

  if totalsum > 2*minMatches:
    for i in range(len(numberoffoundpoints)):
      probabilities[i] = numberoffoundpoints[i]/totalsum

  return probabilities