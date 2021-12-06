import cv2
from enum import Enum

class TextPositions(Enum):
  TPTopL = 1
  TPTopR = 2
  TPCenter = 3

font = cv2.FONT_HERSHEY_SIMPLEX

def drawRectangle(img,boundingBox,name):
  cv2.rectangle(img, (boundingBox[0], boundingBox[1]), (boundingBox[2], boundingBox[3]), (0, 255, 0), 2)
  cv2.putText(img, name, (boundingBox[0], boundingBox[1] - 10), font, 0.9,(0, 255, 0), 2)
  return img

def drawRectanglesWithAssignment(img, foundObjects, boundingBoxes):
  for i in range(len(boundingBoxes)):
    img = drawRectangle(img, boundingBoxes[i], foundObjects[i])

  return img

marginTop = 4
def drawText(img, text, position, offset = 0):
  textsize = cv2.getTextSize(text, font, 1, 2)[0]

  if position == TextPositions.TPCenter:
    textX = (img.shape[1] - textsize[0]) / 2
    textY = (img.shape[0]/2) - (textsize[1]/2) + (textsize[1]*offset)
  elif position == TextPositions.TPTopL:
    textX = 0
    textY = (textsize[1]+marginTop) * (offset+1)
  elif position == TextPositions.TPTopR:
    textX = img.shape[1] - textsize[0]
    textY = (textsize[1] + marginTop) * (offset + 1)

  cv2.putText(img, text, (int(textX), int(textY)), font, 1, (0, 255, 0), 2)
  return img