import cv2
import os

from homography import *
from pawns_recognition import *
from cards_recognition import *

def image_recognition(path):
    window_name = "JACK"

    fiximg = cv2.imread(path)

    height = img.shape[0]
    width = img.shape[1]

    pawnsRecognitionHelper = PawnsRecognitionHelper(height,width)
    cardsRecognitionHelper = CardsRecognitionHelper(height,width)

    homographymatrixfound = False

    img = fiximg.copy()
    cv2.imshow(window_name, img)
    while True:
        if len(list_board_coords) < 4:
            cv2.setMouseCallback(window_name, mousePoints)
            for coord in list_board_coords:
                cv2.circle(img, coord, 10, (0, 255, 0), -1)

        else:
            if not homographymatrixfound:
                homographymatrix = get_homography_matrix(img, np.array(list_board_coords), width, height)
                img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))
                cardsRecognitionHelper.GetScreenPortions(img.shape[0],img.shape[1])
                homographymatrixfound = True

        #img = pawnsRecognitionHelper.ComputeFrame(img)
        img = cardsRecognitionHelper.ComputeFrame(img)

        cv2.imshow(window_name, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

