import cv2

def drawRectangle(img,boundingBox,name):
  cv2.rectangle(img, (boundingBox[0], boundingBox[1]), (boundingBox[2], boundingBox[3]), (0, 255, 0), 2)
  cv2.putText(img, name, (boundingBox[0], boundingBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(36, 255, 12), 2)
  return img

def drawRectanglesWithAssignment(img, foundObjects, boundingBoxes):
  for i in range(len(boundingBoxes)):
    img = drawRectangle(img, boundingBoxes[i], foundObjects[i])

  return img
