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
                homographymatrix, coordinates_box = get_homography_matrix(img, np.array(list_board_coords), width, height)
                img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))
                
                #upper_homography_matrix, upper_box = get_upper_homography_matrix(np.array(list_board_coords),coordinates_box)

                
                cardsRecognitionHelper.GetScreenPortions(img[coordinates_box[1]:coordinates_box[3],coordinates_box[0]:coordinates_box[2]])
                homographymatrixfound = True
            else:
                img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))
                cards_with_bounding_boxes = cardsRecognitionHelper.ComputeFrame(img[coordinates_box[1]:coordinates_box[3],coordinates_box[0]:coordinates_box[2]])
                img[coordinates_box[1]:coordinates_box[3],coordinates_box[0]:coordinates_box[2]] = cards_with_bounding_boxes

                img = cv2.rectangle(img,(coordinates_box[0], 0),(coordinates_box[2], coordinates_box[1]), (0, 255, 0), 3) #Partie supérieur au dessus du plateau
                img = cv2.rectangle(img,(0, coordinates_box[1]),(coordinates_box[0], coordinates_box[3]), (0, 255, 0), 3) #Partie gauche
                img = cv2.rectangle(img,(coordinates_box[2], coordinates_box[1]),(coordinates_box[0] + coordinates_box[2], coordinates_box[3]), (0, 255, 0), 3) #Partie droite
                img = cv2.rectangle(img,(coordinates_box[0], coordinates_box[3]),( coordinates_box[2], coordinates_box[1] + coordinates_box[3]), (0, 255, 0), 3) #partie basse

        #img = pawnsRecognitionHelper.ComputeFrame(img)
        

        cv2.imshow(window_name, img)
        # cv2.waitKey(1000)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

