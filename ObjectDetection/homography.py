import cv2
import numpy as np

list_board_coords = []
def mousePoints(event,x,y,flags,params):
  if event == cv2.EVENT_LBUTTONDOWN and len(list_board_coords) < 4:
    list_board_coords.append([x,y])

def get_homography_matrix(img, pts_src, w, h):
  pts_dst = np.array([[0,0],[w - 1, 0],[w-1, h-1],[0, h-1]])
  mat, status = cv2.findHomography(pts_src, pts_dst)

  return mat

