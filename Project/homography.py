import cv2
import numpy as np

def get_homography_matrix(img, pts_src, w, h):

  factor = 0.15
  top_factor = 0.3
  side_factor = 0.3

  box_cards = (int(side_factor * w), int(top_factor * h), int(w-side_factor *w), int(h-factor *h))

  pts_dst_cards = np.array([[side_factor * w, top_factor * h],[w - side_factor *w, top_factor * h],[w-side_factor *w, h-factor *h],[side_factor *w, h-factor *h]])
  mat_cards, status = cv2.findHomography(pts_src, pts_dst_cards)

  return mat_cards, box_cards

def get_upper_homography_matrix(pts_src, box_cards):

  pts_src_upper_box = [[pts_src[0][0], 0], [pts_src[1][0], 0], [pts_src[1][0], pts_src[1][1]], [pts_src[0][0],pts_src[0][1]]]
  pts_dst_upper_box = [[box_cards[0],0], [box_cards[0] + box_cards[2] ,0], [box_cards[0] + box_cards[2], box_cards[1]], [box_cards[0], box_cards[1]]]
  mat_upper, status_upper = cv2.findHomography(pts_src_upper_box, pts_dst_upper_box)
  upper_box = (box_cards[0], 0, box_cards[2], box_cards[1])

  return  mat_upper, upper_box