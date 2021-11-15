import cv2
import numpy as np

list_board_coords = []
def mousePoints(event,x,y,flags,params):
  if event == cv2.EVENT_LBUTTONDOWN and len(list_board_coords) < 4:
    list_board_coords.append([x,y])

def get_homography_matrix(img, pts_src, w, h):

  factor = 0.15
  pts_dst = np.array([[factor * w, factor * h],[w - factor *w, factor * h],[w-factor *w, h-factor *h],[factor *w, h-factor *h]])
  mat, status = cv2.findHomography(pts_src, pts_dst)

  return mat, (int(factor * w), int(factor * h), int(w-factor *w), int(h-factor *h))

