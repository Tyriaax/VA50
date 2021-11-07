import cv2
import os
import numpy as np
from pynput.mouse import Listener



def get_object_position():
  pass

def get_pieces_orientation():
  pass

def detect_pieces_through_color(frame, card_center, image_info):

  colors_detected = [
    [np.array([90,108,196]), np.array([107,169,247]), "blue" ], #blue
    [np.array([90,20,231]), np.array([113,44,255]), "white"], #white
    [np.array([13,88,196]), np.array([17,138,231]), "orange" ],
    [np.array([27,50,186]), np.array([39,101,243]), "yellow" ],
    [np.array([47,28,149]), np.array([70,50,191]), "green" ] ,
    [np.array([145,39,207]), np.array([157,70,242]), "rose" ],
    [np.array([124,43,95]), np.array([130,79,204]), "purple" ],
    [np.array([98,79,104]), np.array([107,140,164]), "black" ],
    [np.array([113,0,167]), np.array([162,17,204]), "brown" ]
  ]

  #image = cv2.cvtColor(card_center, cv2.COLOR_BGR2HSV)
  image = cv2.cvtColor(frame[card_center[0]:card_center[1], card_center[2]:card_center[3]], cv2.COLOR_BGR2HSV)
  for color in colors_detected:
    if image_info[3] == color[2]:
      mask = cv2.inRange(image, color[0], color[1])

      contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      if len(contours) != 0:
        for contour in contours:
          if cv2.contourArea(contour) > 350 :
              #x, y, w, h = cv2.boundingRect(contour)
              #isRecognised = sift_detection(image[y: y + h, x : x + w], image_infos)
              #if isRecognised:
                #cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 3)
            cv2.rectangle(frame, (card_center[2],card_center[0]), (card_center[3],card_center[1]), (0,0,255), 3)
                  #y, h, x , w 
            cv2.putText(frame, color[2], (card_center[2], card_center[0] - 10),1,1,(0,0,255),2)
            print("detected : " + color[2])

  return frame

def get_homographied_board(img, pts_src, w, h):

  pts_dst = np.array([[0,0],[w - 1, 0],[w-1, h-1],[0, h-1]])

  mat, status = cv2.findHomography(pts_src, pts_dst)
  im_out = cv2.warpPerspective(img, mat, (img.shape[1], img.shape[0]))

  return im_out

def get_keypoints(images):
  list_image_info = []
  for image in images:
    #dim = (400,400)
    #image = cv2.resize(cv2.imread(image), dim, interpolation=cv2.INTER_LINEAR)  
    image =cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image,None)
    image_info = [image,keypoints,descriptors]
    list_image_info.append(image_info)

  return list_image_info

def sift_detection(current_img, card_center, images_infos : list):

  MIN_MATCHES = 40
  img = cv2.cvtColor(current_img[card_center[0]:card_center[1], card_center[2]:card_center[3]], cv2.COLOR_BGR2GRAY)
  sift = cv2.SIFT_create()
  keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)

  index_params = dict(algorithm = 0, trees = 5)
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  good_points = []
  info = ""
  
  for image_info in images_infos:
    if(image_info[2] is not None and len(image_info[2]) > 2) and (descriptors_1 is not None and len(image_info[2]) > 2):
      matches = flann.knnMatch(image_info[2], descriptors_1, k = 2)
      temp = []
      for m, n in matches:
        if m.distance < 0.8 * n.distance:
          good_points.append(m)
        """if(len(temp) >= len(good_points)):
          good_points = temp
          info = image_info"""
      if(len(good_points) >= MIN_MATCHES):
            #img3 = draw_boxes(matches,keypoints_1,img1,keypoints_2,img2)
            #matchedImg = cv2.drawMatches(img, keypoints_1, image_info[0], #image_info[1], matches[:30], image_info[0], flags=2)
        current_img = detect_pieces_through_color(current_img, card_center, image_info)

  return current_img

def get_screen_portion(img, images_infos):
  height, width, channels = img.shape
  #print(height,width)
  
  height_portion = int(height/3) 
  proportion = int(0.24 * height_portion)

  for i in range(3):
    for j in range(3):
      center_position = (i * height_portion + proportion, (i + 1) * height_portion - proportion, j * height_portion + proportion, (j + 1) * height_portion - proportion)
      #y, h, x , w 
      print("---------------------\nposition : ", i , j)
      frame = sift_detection(img, center_position,images_infos)
      #frame = detect_pieces_through_color(img, center_position, images_infos)

  return frame

def load_kp_samples():

  PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Samples","Cards"))
  list_image_info = []
  dir = os.listdir(PATH_SAMPLES)

  for image in dir:
    #images.append(os.path.join(PATH_SAMPLES,image))
    dim = (400,400)
    #im_out = cv2.resize(cv2.imread(os.path.join(PATH_SAMPLES, image)), dim, interpolation=cv2.INTER_LINEAR)  
    im_out = cv2.imread(os.path.join(PATH_SAMPLES, image))
    im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(im_out,None)
    image_info = [im_out,keypoints,descriptors, image.split(".")[0]]
    list_image_info.append(image_info)

  return list_image_info

list_board_coords = []
def mousePoints(event,x,y,flags,params):
  if event == cv2.EVENT_LBUTTONDOWN and len(list_board_coords) < 4:
    list_board_coords.append([x,y])

def video_recognition():

  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #Tochange in case

  height = 720
  width = 720
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  images_infos = load_kp_samples()

  window_name = "JACK"

  _, img1 = cap.read()
  cv2.imshow(window_name, img1)

  while True:
    _, img = cap.read()
  
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    if len(list_board_coords) < 4:
      cv2.setMouseCallback(window_name, mousePoints)
      for coord in list_board_coords:
        cv2.circle(img,coord,10,(0,255,0),-1)
    else:  
      img = get_homographied_board(img, np.array(list_board_coords), width, height)
      get_screen_portion(img, images_infos)

    cv2.imshow(window_name, img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break  
    
  cap.release()
  cv2.destroyAllWindows()


def image_recognition():

  height = 720
  width = 720

  PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "plateau.jpg"))
  plateau = cv2.imread(PATH_SAMPLES)
  images_infos = load_kp_samples()

  window_name = "JACK"

  cv2.imshow(window_name, plateau)

  while True:
    img = plateau
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    if len(list_board_coords) < 4:
      cv2.setMouseCallback(window_name, mousePoints)
      for coord in list_board_coords:
        cv2.circle(img,coord,10,(0,255,0),-1)
    else:  
      img = get_homographied_board(img, np.array(list_board_coords), width, height)
      get_screen_portion(img, images_infos)

    cv2.imshow(window_name, img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break  
    
  cv2.destroyAllWindows()


image_recognition()
