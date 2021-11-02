import cv2
import os
import numpy as np
from pynput.mouse import Listener



def get_object_position():
  pass

def get_pieces_orientation():
  pass

def detect_pieces_through_color(frame, image_infos):
  """lower = np.array([15,150,20]) #yellow askip
  upper = np.array([35,255,255])

  lower = np.array([161,155,84]) #red askip
  upper = np.array([179,255,255])"""

  #lower = np.array([94,80,2]) #blue askip
  #upper = np.array([126,255,255])

  #lower = np.array([0,0,128]) #white askip
  #upper = np.array([255,255,255])

  colors_detected = [
    [np.array([94,80,2]), np.array([126,255,255]), "Sherlock" ], #blue
    #[np.array([0,0,128]), np.array([255,255,255]), "Ogre"] #white
  ]

  image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  for color in colors_detected:
    #mask = cv2.inRange(image, lower, upper)
    mask = cv2.inRange(image, color[0], color[1])
    #blue = cv2.bitwise_and(frame, frame, mask = mask) Affichage du mask

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
      for contour in contours:
        if cv2.contourArea(contour) > 500:
          x, y, w, h = cv2.boundingRect(contour)
          isRecognised = sift_detection(frame[y: y + h, x : x + w], image_infos)
          if isRecognised:
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 3)
            cv2.putText(frame, color[2], (x,y),1,1,(0,0,255),3)


  return frame

def get_homographied_board(img, pts_src):

  #im_src = cv2.imread(os.path.abspath(os.path.join(os.path.dirname( __file__ ), "plateau.jpg")))
  #window_name = 'image'
  
  # Four corners of the book in destination image.
  w = 1280
  h = 720

  pts_dst = np.array([[0,0],[w - 1, 0],[w-1, h-1],[0, h-1]])

  mat, status = cv2.findHomography(pts_src, pts_dst)
  im_out = cv2.warpPerspective(img, mat, (img.shape[1], img.shape[0]))
  return im_out

def get_keypoints(images):
  list_image_info = []
  for image in images:
    dim = (400,400)
    image = cv2.resize(cv2.imread(image), dim, interpolation=cv2.INTER_LINEAR)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image,None)
    image_info = [image,keypoints,descriptors]
    list_image_info.append(image_info)

  return list_image_info

def sift_detection(current_img, images_infos : list):
  MIN_MATCHES = 20

  for image_info in images_infos:
    
    img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

    #sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, image_info[2])
    matches = sorted(matches, key = lambda x:x.distance)

    if(len(matches) >= MIN_MATCHES):
      #img3 = draw_boxes(matches,keypoints_1,img1,keypoints_2,img2)
      #matchedImg = cv2.drawMatches(img, keypoints_1, image_info[0], #image_info[1], matches[:30], image_info[0], flags=2)
      return True

  return False

def load_kp_samples(PATH_SAMPLES):

  PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Samples"))
  list_image_info = []
  dir = os.listdir(PATH_SAMPLES)

  for image in dir:
    print(image)
    #images.append(os.path.join(PATH_SAMPLES,image))
    dim = (400,400)
    image = cv2.resize(cv2.imread(os.path.join(PATH_SAMPLES, image)), dim, interpolation=cv2.INTER_LINEAR)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image,None)
    image_info = [image,keypoints,descriptors]
    list_image_info.append(image_info)

  return list_image_info


list_board_coords = []
def mousePoints(event,x,y,flags,params):
  if event == cv2.EVENT_LBUTTONDOWN and len(list_board_coords) < 4:
    list_board_coords.append([x,y])

def video_recognition():

  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #Tochange in case

  height = 720
  width = 1280
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  images_infos = load_kp_samples(PATH_SAMPLES = "")

  window_name = "JACK"

  _, img1 = cap.read()
  cv2.imshow(window_name, img1)

  while True:
    _, img = cap.read()
    #surfImage = sift_detection(img, images_infos)  
    #cv2.imshow('Result', surfImage)

    if len(list_board_coords) < 4:
      cv2.setMouseCallback(window_name, mousePoints)
      for coord in list_board_coords:
        cv2.circle(img,coord,10,(0,255,0),-1)
    else:  
      img = get_homographied_board(img, np.array(list_board_coords))
      img = detect_pieces_through_color(img, images_infos)
      #cv2.imshow("mask", mask)

    cv2.imshow(window_name, img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break  
    
  cap.release()
  cv2.destroyAllWindows()


video_recognition()
