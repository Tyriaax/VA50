import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

def drawRectangle(img,boundingBox,name):
  cv2.rectangle(img, (boundingBox[0], boundingBox[1]), (boundingBox[2], boundingBox[3]), (0, 255, 0), 2)
  cv2.putText(img, name, (boundingBox[0], boundingBox[1] - 10), font, 0.9,(0, 255, 0), 2)
  return img

def drawRectanglesWithAssignment(img, foundObjects, boundingBoxes):
  for i in range(len(boundingBoxes)):
    img = drawRectangle(img, boundingBoxes[i], foundObjects[i])

  return img

def drawText(img, text):
  # get boundary of this text
  textsize = cv2.getTextSize(text, font, 1, 2)[0]

  # get coords based on boundary
  textX = (img.shape[1] - textsize[0]) / 2
  textY = (img.shape[0] + textsize[1]) / 2

  # add text centered on image
  cv2.putText(img, text, (int(textX), int(textY)), font, 1, (0, 255, 0), 2)
  return img