import cv2
import os

from GameProcessor import *

def video_recognition(path = None):
    window_name = "JACK"
    height = 720
    width = 1280

    if(path):
        cap = cv2.VideoCapture(path)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # ToChange in case

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, img = cap.read()
    cv2.imshow(window_name, img) #Need to show the window first so that we can set our MouseCallback on the windows

    gameProcessor = GameProcessor(img, window_name)

    while True:
        ret, img = cap.read()
        if ret:
            img = gameProcessor.ComputeFrame(img)
            modifiedimg = gameProcessor.DrawFrame(img)
            cv2.imshow(window_name, modifiedimg)
            if not gameProcessor.ComputeInputs(img) :
                break
            if (key == 32 or capEveryFrame) and (gameBoard.getGameStatus().value > GameStates.GSWaitingHomography.value):
                gameBoard.updateGameStatus()
                cardsRecognitionHelper.IsInLineOfSight(img, [], (0, 3), (1, 3))  # (1,0), (1,3))
                cardsRecognitionHelper.GetEmptySideCards(img)
                cardsRecognitionHelper.getFrontSideCards(img)
                cardsRecognitionHelper.ComputeFrame(img)            
                pawnsRecognitionHelper.ComputeFrame(img)
                gameBoard.printState()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    cv2.destroyAllWindows()
