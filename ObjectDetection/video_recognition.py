import cv2
import os

from homography import *
from pawns_recognition import *
from cards_recognition import *
from GameBoard import *
from drawing import *

def video_recognition(path = None):
    capEveryFrame = False

    window_name = "JACK"
    height = 720
    width = 1280

    if(path):
        cap = cv2.VideoCapture(path)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Tochange in case

    gameBoard = GameBoard()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    pawnsRecognitionHelper = PawnsRecognitionHelper(height, width, gameBoard)
    cardsRecognitionHelper = CardsRecognitionHelper(height, width, gameBoard)

    homographymatrixfound = False

    _, img1 = cap.read()
    cv2.imshow(window_name, img1)
    while True:
        ret, img = cap.read()
        if ret:
            if len(list_board_coords) < 4:
                cv2.setMouseCallback(window_name, mousePoints)
                for coord in list_board_coords:
                    cv2.circle(img, coord, 10, (0, 255, 0), -1)
            else:
                if not homographymatrixfound:
                    homographymatrix, coordinates = get_homography_matrix(img, np.array(list_board_coords), width, height)
                    
                    cardsRecognitionHelper.GetScreenPortions(img[coordinates[1]:coordinates[3],coordinates[0]:coordinates[2]],coordinates)
                    pawnsRecognitionHelper.GetScreenPortion(img,coordinates)
                    homographymatrixfound = True
                    gameBoard.updateGameStatus()

                img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))

            modifiedimg = img.copy()

            if gameBoard.getGameStatus() == GameStates.GSWaitingHomography:
                modifiedimg = drawText(modifiedimg,"Veuillez selectionner les quatres coins des 9 cartes")
            if gameBoard.getGameStatus() == GameStates.GSWaitingFirstRecognition:
                modifiedimg = pawnsRecognitionHelper.DrawZonesRectangles(modifiedimg)
                modifiedimg = drawText(modifiedimg,"Veuillez appuyer sur espace pour lancer une nouvelle detection")
            if gameBoard.getGameStatus() == GameStates.GSGameStarted:
                modifiedimg = pawnsRecognitionHelper.DrawFrame(modifiedimg)
                modifiedimg = cardsRecognitionHelper.DrawFrame(modifiedimg)

            cv2.imshow(window_name, modifiedimg)


            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            if (key == 32 or capEveryFrame) and (gameBoard.getGameStatus().value > GameStates.GSWaitingHomography.value):
                gameBoard.updateGameStatus()
                cardsRecognitionHelper.isInLineOfSight(img, [], (0, 3), (1, 3))  # (1,0), (1,3))
                cardsRecognitionHelper.ComputeFrame(img)
                pawnsRecognitionHelper.ComputeFrame(img)
                gameBoard.printState()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    cv2.destroyAllWindows()

