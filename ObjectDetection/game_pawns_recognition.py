import cv2
import os
import numpy as np
from pynput.mouse import Listener



def get_object_position():
  pass

def get_pieces_orientation():
  pass

def load_kp_samples():

  PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Samples","Pawns"))
  list_image_info = []
  dir = os.listdir(PATH_SAMPLES)

  for image in dir:
    print(image)
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

def get_homographied_board(img, pts_src, w, h):

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

  MIN_MATCHES = 25
  img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
  sift = cv2.SIFT_create()
  keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)

  index_params = dict(algorithm = 0, trees = 5)
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  good_points = []
  info = ""

  for image_info in images_infos:
    if(image_info[2] is not None) and (descriptors_1 is not None):
      matches = flann.knnMatch(image_info[2], descriptors_1, k = 2)
      temp = []
      for m, n in matches:
        if m.distance < 0.8 * n.distance:
          temp.append(m)
      if(len(temp) > len(good_points)):
        good_points = temp
        info = image_info

  if(len(good_points) >= MIN_MATCHES):
      #img3 = draw_boxes(matches,keypoints_1,img1,keypoints_2,img2)
      #matchedImg = cv2.drawMatches(img, keypoints_1, image_info[0], #image_info[1], matches[:30], image_info[0], flags=2)
    print("detected " + info[3])

  return current_img

def video_recognition():

  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #Tochange in case

  height = 720
  width = 1280
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  images_infos = load_kp_samples()

  window_name = "JACK"

  _, img1 = cap.read()
  cv2.imshow(window_name, img1)

  while True:
    _, img = cap.read()
  
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    """
    if len(list_board_coords) < 4:
      cv2.setMouseCallback(window_name, mousePoints)
      for coord in list_board_coords:
        cv2.circle(img,coord,10,(0,255,0),-1)
    else:  
      img = get_homographied_board(img, np.array(list_board_coords), width, height)
    """

    img = sift_detection(img, images_infos)
    cv2.imshow(window_name, img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break  
    
  cap.release()
  cv2.destroyAllWindows()


video_recognition()

