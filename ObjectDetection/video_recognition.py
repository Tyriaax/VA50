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

    gameProcessor = GameProcessor(img, window_name)

    while True:
        ret, img = cap.read()
        if ret:
            img = gameProcessor.ComputeFrame(img)
            modifiedimg = gameProcessor.DrawFrame(img)
            cv2.imshow(window_name, modifiedimg)
            if not gameProcessor.ComputeInputs(img) :
                break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    cv2.destroyAllWindows()
