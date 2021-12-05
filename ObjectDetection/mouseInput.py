import cv2

list_board_coords = []
action_pawn_click_coord = (None,None)
def mousePoints(event,x,y,flags,params):
  if event == cv2.EVENT_LBUTTONDOWN:
    if len(list_board_coords) < 4:
      list_board_coords.append([x,y])
    else:
      action_pawn_click_coord = (x,y)