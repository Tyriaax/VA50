import cv2
import os
import numpy as np
from pynput.mouse import Listener



def get_object_position():
  pass

def get_pieces_orientation():
  pass

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
  MIN_MATCHES = 55

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
      matchedImg = cv2.drawMatches(img, keypoints_1, image_info[0], image_info[1], matches[:30], image_info[0], flags=2)
      return matchedImg

  return current_img

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

    cv2.imshow(window_name, img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break  
    
  cap.release()
  cv2.destroyAllWindows()


video_recognition()
