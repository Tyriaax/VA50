import cv2
import os

from homography import *
from Pawns_recognition import *

def video_recognition():
    window_name = "JACK"
    height = 720
    width = 1280

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Tochange in case

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    pawnRecognitionHelper = PawnsRecognitionHelper(height,width)

    homographymatrixfound = False

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
                homographymatrix = get_homography_matrix(img, np.array(list_board_coords), width, height)
                homographymatrixfound = True
            else:
                img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))

        img = pawnRecognitionHelper.ComputeFrame(img)

        cv2.imshow(window_name, img)
        # cv2.waitKey(1000)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

video_recognition()
