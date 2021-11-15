import cv2
import numpy as np

list_board_coords = []
def mousePoints(event,x,y,flags,params):
  if event == cv2.EVENT_LBUTTONDOWN and len(list_board_coords) < 4:
    list_board_coords.append([x,y])

def get_homography_matrix(img, pts_src, w, h):

  factor = 0.15
  bot_factor = 0.1

  box_cards = (int(factor * w), int(factor * h), int(w-factor *w), int(h-factor *h))

  pts_dst_cards = np.array([[factor * w, factor * h],[w - factor *w, factor * h],[w-factor *w, h-factor *h],[factor *w, h-factor *h]])
  mat_cards, status = cv2.findHomography(pts_src, pts_dst_cards)


  return mat_cards, box_cards

def get_upper_homography_matrix(pts_src, box_cards):

  pts_src_upper_box = [[pts_src[0][0], 0], [pts_src[1][0], 0], [pts_src[1][0], pts_src[1][1]], [pts_src[0][0],pts_src[0][1]]]
  pts_dst_upper_box = [[box_cards[0],0], [box_cards[0] + box_cards[2] ,0], [box_cards[0] + box_cards[2], box_cards[1]], [box_cards[0], box_cards[1]]]
  mat_upper, status_upper = cv2.findHomography(pts_src_upper_box, pts_dst_upper_box)
  upper_box = (box_cards[0], 0, box_cards[2], box_cards[1])

  return  mat_upper, upper_box