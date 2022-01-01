import cv2
import numpy as np
import random
import enum as Enum
from pathlib import Path

objectToSample = "DP"
numberOfSamples = 3
cropCircle = True
resizeDim = 100 #px

"""
class Cards(Enum):
  CBlack = 0
  CBlue = 1
  CBrown = 2
  CGreen = 3
  COrange = 4
  CPink = 5
  CPurple = 6
  CWhite = 7
  CYellow = 8

class ActionPawns(Enum):
  APSherlock = 0
  APAlibi = 1
  APToby = 2
  APWatson = 3
  APJoker = 4
  APReturn = 5
  APChangeCard = 6
  APReturn2 = 7

class DetectivePawns(Enum):
  DPSherlock = 0
  DPToby = 1
  DPWatson = 2
"""

savePath = objectToSample+"data"
validationProbability = 5

Bb_click_coordinates = None
boundingBoxes = []

def BbClick(click_coordinates):
    global Bb_click_coordinates
    global boundingBoxes

    for i in range(len(boundingBoxes)):
        if ((boundingBoxes[i][0] < click_coordinates[0] < boundingBoxes[i][2]) and
        (boundingBoxes[i][1] < click_coordinates[1] < boundingBoxes[i][3])):
            Bb_click_coordinates = boundingBoxes[i]

list_board_coords = []
def ComputeMouseInput(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # If we are still at the homography stage record the first 4 points coordinates
        if len(list_board_coords) < 4:
            list_board_coords.append([x, y])
        else:
            BbClick([x,y])

def imageProcessingForFindingContours(img):
    # First we convert the frame to a grayscale image
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # We then use a median blur technique to reduce the noise
    img2 = cv2.medianBlur(img2, 5)

    # We then apply a sharpening filter to enhance the edges
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img2 = cv2.filter2D(img2, -1, sharpen_kernel)

    # We can then threshold to get a binary image
    img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # We invert the image so that the contours can get detected
    img2 = 255 - img2

    # We also apply a close morphology transformation to get rid of the imperfections inside the shape
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, morph_kernel, iterations=3)  # We apply a close transformation

    return img2

rectangleMaxRatioDifference = 0.4
def getBoundingBoxes(img,maxarea,minarea, inspectInsideCountours = False):
    rectangles = []

    img2 = imageProcessingForFindingContours(img)

    # We then use findContours to get the contours of the shape
    if not inspectInsideCountours:
        retrievalMode = cv2.RETR_EXTERNAL
    else:
        retrievalMode = cv2.RETR_LIST

    #cnts = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(img2, retrievalMode, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  # We then loop through all the detected contours to only retrieve the ones with a desired area
    for c in cnts:
        area = cv2.contourArea(c)
        if minarea <= area <= maxarea:
            x, y, w, h = cv2.boundingRect(c)
            if (1-rectangleMaxRatioDifference)*h <= w <= (1+rectangleMaxRatioDifference)*h:
                rectangle = [x, y, x+w, y+h]
                rectangles.append(rectangle)

    return rectangles

def DrawBoundingBoxes(img, boundingBoxes):
    for boundingBox in boundingBoxes:
        cv2.rectangle(img, (boundingBox[0], boundingBox[1]), (boundingBox[2], boundingBox[3]), (0, 255, 0), 2)

        return img

def get_homography_matrix(img, pts_src, w, h):

  factor = 0.15
  top_factor = 0.3
  side_factor = 0.3

  #box_cards = (int(factor * w), int(factor * h), int(w-factor *w), int(h-factor *h))
  box_cards = (int(side_factor * w), int(top_factor * h), int(w-side_factor *w), int(h-factor *h))

  #pts_dst_cards = np.array([[factor * w, factor * h],[w - factor *w, factor * h],[w-factor *w, h-factor *h],[factor *w, h-factor *h]])
  pts_dst_cards = np.array([[side_factor * w, top_factor * h],[w - side_factor *w, top_factor * h],[w-side_factor *w, h-factor *h],[side_factor *w, h-factor *h]])
  mat_cards, status = cv2.findHomography(pts_src, pts_dst_cards)

  return mat_cards, box_cards

def generateFolders():
    for i in range(0,numberOfSamples):
        Path.mkdir(Path(objectToSample + 'data/val/' + objectToSample + str(i)), 511, 1, 1)
        Path.mkdir(Path(objectToSample + 'data/train/' + objectToSample + str(i) ), 511, 1, 1)

def video_recognition():
    global Bb_click_coordinates
    global boundingBoxes

    window_name = "CNNSampleHelper"
    height = 720
    width = 1280

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, img = cap.read()
    cv2.imshow(window_name, img) #Need to show the window first so that we can set our MouseCallback on the windows

    homographymatrixfound = False

    cv2.setMouseCallback(window_name, ComputeMouseInput)

    imgToSample = None

    generateFolders()

    while True:
        ret, img = cap.read()
        if ret:
            # If we are still waiting for the homography points
            if not homographymatrixfound:
                modifiedimg = img.copy()

                for coord in list_board_coords:
                    cv2.circle(modifiedimg, coord, 10, (0, 255, 0), -1)

                # When we have the 4 coordinates we can compute it
                if len(list_board_coords) == 4:
                    # When we get the points for homography, set homographymatrix
                    homographymatrix, coordinates = get_homography_matrix(img,np.array(list_board_coords),width, height)

                    homographymatrixfound = True
                    img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))

                    cardSize = (int((coordinates[2]-coordinates[0])/3), int((coordinates[3]-coordinates[1])/3))

                    bBmaxArea = (cardSize[0] * cardSize[1]) / 2
                    bBminArea = (cardSize[0] * cardSize[1]) / 12
            else:
                # We need to apply homography here so that the frames are correctly computed
                img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))

                boundingBoxes = getBoundingBoxes(img, bBmaxArea, bBminArea, inspectInsideCountours=True)
                modifiedimg = img.copy()

                DrawBoundingBoxes(modifiedimg, boundingBoxes)

                if Bb_click_coordinates is not None:
                    print("Sample selectionne veuillez lui assigner un numero")
                    imgToSample = img[Bb_click_coordinates[1]:Bb_click_coordinates[3],Bb_click_coordinates[0]:Bb_click_coordinates[2]]
                    if cropCircle:
                        dim = (resizeDim,resizeDim)
                        imgToSample = cv2.resize(imgToSample, dim, interpolation=cv2.INTER_LINEAR)
                        height, width = imgToSample.shape[:2]
                        mask = np.full((height, width), 0, dtype=np.uint8)
                        cv2.circle(mask, (height // 2, width // 2), height // 2, 255, -1)

                        imgToSample = cv2.bitwise_and(imgToSample, imgToSample, mask=mask)

                    Bb_click_coordinates = None

            cv2.imshow(window_name, modifiedimg)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

            if 48 <= key <= 48+numberOfSamples-1:
                if imgToSample is not None:
                    objectName = objectToSample + str(key-48)
                    print("Je sauvegarde le sample " + objectName)
                    if random.randint(0,validationProbability) == 0:
                        pathname = savePath + "/val/" + objectName + "/" + objectName + "_" + str(random.randint(0,1000000)) + ".jpg"
                        cv2.imwrite(pathname, imgToSample)
                    else:
                        pathname = savePath + "/train/" + objectName + "/" + objectName + "_" + str(random.randint(0, 1000000)) + ".jpg"
                        cv2.imwrite(pathname, imgToSample)

                    imgToSample = None

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_recognition()

if __name__ == '__main__':
    main()