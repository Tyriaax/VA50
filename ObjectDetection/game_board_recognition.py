import cv2
import os
import numpy as np
from numpy.lib.type_check import imag
from pynput.mouse import Listener



def get_object_position():
  pass

def get_pieces_orientation():
  pass

def detect_pieces_through_color(frame, card_center, image_info):

  colors_detected_original = [
    [np.array([90,108,196]), np.array([107,169,247]), "blue_binary" ], #blue
    [np.array([90,20,231]), np.array([113,44,255]), "white_binary"], #white
    [np.array([13,88,196]), np.array([17,138,231]), "orange" ],
    [np.array([27,50,186]), np.array([39,101,243]), "yellow" ],
    [np.array([47,28,149]), np.array([70,50,191]), "green" ] ,
    [np.array([145,39,207]), np.array([157,70,242]), "pink" ],
    [np.array([124,43,95]), np.array([130,79,204]), "purple" ],
    [np.array([98,79,104]), np.array([107,140,164]), "black" ],
    [np.array([113,0,167]), np.array([162,17,204]), "brown" ]
  ]

  colors_detected = [
    [np.array([112,54,73]), np.array([139,124,131]), "blue" ], #blue
    [np.array([122,0,136]), np.array([163,23,226]), "white"], #white
    [np.array([8,76,177]), np.array([13,109,208]), "orange" ],
    [np.array([18,52,186]), np.array([29,90,238]), "yellow" ],
    [np.array([20,50,70]), np.array([29,79,103]), "green" ] ,
    [np.array([162,69,140]), np.array([173,108,244]), "pink" ],
    [np.array([141,45,109]), np.array([160,57,146]), "purple" ],
    [np.array([6,41,44]), np.array([15,80,75]), "black" ],
    [np.array([4,76,103]), np.array([5,98,126]), "brown" ]
  ]

  #image = cv2.cvtColor(card_center, cv2.COLOR_BGR2HSV)
  image = cv2.cvtColor(frame[card_center[0]:card_center[1], card_center[2]:card_center[3]], cv2.COLOR_BGR2HSV)
  for color in colors_detected:
    if image_info[3] == color[2]:
      mask = cv2.inRange(image, color[0], color[1])

      contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      if len(contours) != 0:
        for contour in contours:
          if cv2.contourArea(contour) > 250 :
              #x, y, w, h = cv2.boundingRect(contour)
              #isRecognised = sift_detection(image[y: y + h, x : x + w], image_infos)
              #if isRecognised:
                #cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 3)
            cv2.rectangle(frame, (card_center[2],card_center[0]), (card_center[3],card_center[1]), (0,0,255), 3)
                  #y, h, x , w 
            cv2.putText(frame, color[2], (card_center[2], card_center[0] - 10),1,1,(0,0,255),2)
            print("detected : " + color[2])
            return frame

  return frame

def get_homographied_board(img, pts_src, w, h):

  pts_dst = np.array([[0,0],[w - 1, 0],[w-1, h-1],[0, h-1]])

  mat, status = cv2.findHomography(pts_src, pts_dst)
  im_out = cv2.warpPerspective(img, mat, (img.shape[1], img.shape[0]))

  return im_out


def sift_detection(current_img, card_center, images_infos : list):

  MIN_MATCHES = 10
  img = cv2.cvtColor(current_img[card_center[0]:card_center[1], card_center[2]:card_center[3]], cv2.COLOR_BGR2GRAY)
  sift = cv2.SIFT_create()
  keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)

  index_params = dict(algorithm = 1, trees = 5)
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  good_points = []
  info = ""
  
  for image_info in images_infos:
    if(image_info[2] is not None and len(image_info[2]) > 2) and (descriptors_1 is not None and len(image_info[2]) > 2):
      matches = flann.knnMatch(image_info[2], descriptors_1, k = 2)
      temp = []
      for m, n in matches:
        if m.distance < 0.8 * n.distance: #0.8 - 50
          temp.append(m)
        if(len(temp) >= len(good_points)):
          good_points = temp
          info = image_info
        

  if(len(good_points) >= MIN_MATCHES):
            #img3 = draw_boxes(matches,keypoints_1,img1,keypoints_2,img2)
            #matchedImg = cv2.drawMatches(img, keypoints_1, image_info[0], #image_info[1], matches[:30], image_info[0], flags=2)
    print(info[3])
    current_img = detect_pieces_through_color(current_img, card_center, info)

  return current_img

def get_screen_portion(img, images_infos):
  height, width, channels = img.shape
  #print(height,width)
  
  height_portion = int(height/3) 
  proportion = int(0.2 * height_portion)

  for i in range(3):
    for j in range(3):
      center_position = (i * height_portion + proportion, (i + 1) * height_portion - proportion, j * height_portion + proportion, (j + 1) * height_portion - proportion)
      #y, h, x , w 
      print("---------------------\nposition : ", i , j)
      frame = sift_detection(img, center_position,images_infos)
      #frame = detect_pieces_through_color(img, center_position, images_infos)

      """PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Samples","Cards2", str(i) + str(j) + ".jpg"))
      dim = (600,600)
      im_out = cv2.resize(frame[center_position[0]:center_position[1], center_position[2]:center_position[3]], dim, interpolation=cv2.INTER_LINEAR)  
      cv2.imwrite(PATH_SAMPLES, im_out)"""

  return frame

from scipy.ndimage import gaussian_filter

def load_kp_samples():

  PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Samples","Cards3"))
  list_image_info = []
  dir = os.listdir(PATH_SAMPLES)

  for image in dir:

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
    _, frame = cap.read()
    img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

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

class Camera:
  def __init__(self, width, height):
    self.image_width = width
    self.image_height = height
  
  def get_image(self):
    return self.image_width, self.image_height

def from_video_file_recognition():
  PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'zebi.mp4'))#"first_cut1.mp4"))

  cap= cv2.VideoCapture(PATH_SAMPLES)
  height = 720
  width = 720
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  camera = Camera(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  i=0
  images = load_kp_samples()
  window_name = "result"

  _,plateau = cap.read()

  cv2.imshow(window_name, plateau)

  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    if len(list_board_coords) < 4:
      cv2.setMouseCallback(window_name, mousePoints)
      for coord in list_board_coords:
        cv2.circle(img,coord,10,(0,255,0),-1)
    else:  
      img = get_homographied_board(img, np.array(list_board_coords), width, height)
      get_screen_portion(img, images)

    cv2.imshow(window_name, img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break  
    cv2.waitKey(1)

  cap.release()
  cv2.destroyAllWindows()

video_recognition()
#qfrom_video_file_recognition()
