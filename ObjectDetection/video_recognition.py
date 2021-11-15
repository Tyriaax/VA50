import cv2
import os

from homography import *
from pawns_recognition import *
from cards_recognition import *

def video_recognition(path = None):
    window_name = "JACK"
    height = 720
    width = 1280

    if(path):
        cap = cv2.VideoCapture(path)
        
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Tochange in case

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    pawnsRecognitionHelper = PawnsRecognitionHelper(height,width)
    cardsRecognitionHelper = CardsRecognitionHelper(height,width)

    homographymatrixfound = False
    coordinates = []

    _, img1 = cap.read()
    cv2.imshow(window_name, img1)
    while True:
        _, img = cap.read()
        if len(list_board_coords) < 4:
            cv2.setMouseCallback(window_name, mousePoints)
            for coord in list_board_coords:
                cv2.circle(img, coord, 10, (0, 255, 0), -1)
        else:
            if not homographymatrixfound:
                homographymatrix, coordinates = get_homography_matrix(img, np.array(list_board_coords), width, height)
                img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))
                cardsRecognitionHelper.GetScreenPortions(img[coordinates[1]:coordinates[3],coordinates[0]:coordinates[2]])
                homographymatrixfound = True
            else:
                img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))
                cards_with_bounding_boxes = cardsRecognitionHelper.ComputeFrame(img[coordinates[1]:coordinates[3],coordinates[0]:coordinates[2]])
                img[coordinates[1]:coordinates[3],coordinates[0]:coordinates[2]] = cards_with_bounding_boxes
        #img = pawnsRecognitionHelper.ComputeFrame(img)
        

        cv2.imshow(window_name, img)
        # cv2.waitKey(1000)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

