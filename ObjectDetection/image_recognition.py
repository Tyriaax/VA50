import cv2
import os

from homography import *
from pawns_recognition import *
from cards_recognition import *

def image_recognition(path):
    window_name = "JACK"

    fiximg = cv2.imread(path)
    img = fiximg.copy()

    height = img.shape[0]
    width = img.shape[1]

    gameBoard = GameBoard()

    pawnsRecognitionHelper = PawnsRecognitionHelper(height,width, gameBoard)
    cardsRecognitionHelper = CardsRecognitionHelper(height,width, gameBoard)

    homographymatrixfound = False

    cv2.imshow(window_name, img)
    while True:
        img = fiximg.copy()

        if len(list_board_coords) < 4:
            cv2.setMouseCallback(window_name, mousePoints)
            for coord in list_board_coords:
                cv2.circle(img, coord, 10, (0, 255, 0), -1)

        else:
            if not homographymatrixfound:
                homographymatrix, coordinates = get_homography_matrix(img, np.array(list_board_coords), width, height)
                img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))
                cardsRecognitionHelper.GetScreenPortions(img[coordinates[1]:coordinates[3],coordinates[0]:coordinates[2]],coordinates)
                pawnsRecognitionHelper.GetScreenPortion(img, coordinates)
                homographymatrixfound = True
            else:
                img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))
                img = cardsRecognitionHelper.ComputeFrame(img)
                img = pawnsRecognitionHelper.ComputeFrame(img)

        cv2.imshow(window_name, img)
        gameBoard.printState()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

