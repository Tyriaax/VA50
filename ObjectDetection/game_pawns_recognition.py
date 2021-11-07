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

def get_homography_matrix(img, pts_src, w, h):

  pts_dst = np.array([[0,0],[w - 1, 0],[w-1, h-1],[0, h-1]])

  mat, status = cv2.findHomography(pts_src, pts_dst)

  return mat

"""
def sift_detection(current_img, images_infos : list):

  MIN_MATCHES = 15
  KnnDistance = 0.5

  img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
  sift = cv2.SIFT_create()
  keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)

  index_params = dict(algorithm = 0 , trees = 5)
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  good_points = []

  for image_info in images_infos:
    if(image_info[2] is not None) and (descriptors_1 is not None):
      if (len(image_info[2]) >= 2) and (len(descriptors_1) >= 2):
        matches = flann.knnMatch(image_info[2], descriptors_1, k = 2)
        temp = []
        for m, n in matches:
          if m.distance < KnnDistance * n.distance:
            temp.append(m)
        if (len(temp) > len(good_points)):
          good_points = temp
          info = image_info
          selected_match = matches

  if (len(good_points) >= MIN_MATCHES):
    print("detected " + info[3])
    src_pts = np.float32([info[1][m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_1[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Méthode 1
    pts = src_pts[mask == 1]
    if (len(pts) > 0):
      min_x, min_y = np.int32(pts.min(axis=0))
      max_x, max_y = np.int32(pts.max(axis=0))

      #cv2.rectangle(current_img, (min_x, min_y), (max_x, max_y), 255, 2)

    # Méthode 2
    #if(len(M) > 0):
    matchesMask = mask.ravel().tolist()

    h, w = info[0].shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts[None,:,:], M)

    current_img = cv2.polylines(current_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # Méthode 3
    matchesMask = [[0, 0] for i in range(len(selected_match))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(selected_match):
      if m.distance < KnnDistance * n.distance:
        matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                     singlePointColor=(255, 0, 0),
                     matchesMask=matchesMask,
                     flags=0)

    imresize = cv2.resize(info[0],(100,100))
    current_img = cv2.drawMatchesKnn(imresize,info[1],current_img,keypoints_1,selected_match,None,**draw_params)

  return current_img
"""

def getBoundingBoxes(img,maxarea,minarea):
  rectangles = []

  # First we convert the frame to a grayscale image
  img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # We then use a median blur technique to reduce the noise
  img2 = cv2.medianBlur(img2, 5)

  # We then apply a sharpening filter to enhance the edges
  sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
  img2 = cv2.filter2D(img2, -1, sharpen_kernel)

  # We can then threshold to get a binary image
  img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

  # We invert the image so that the contours can get detected
  img2 = 255 - img2

  # We also apply a close morphology transformation to get rid of the imperfections inside the shape
  morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, morph_kernel, iterations=3)  # We apply a close transformation

  # We then use findContours to get the contours of the shape
  cnts = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  # We then loop through all the detected contours to only retrieve the ones with a desired area
  for c in cnts:
    area = cv2.contourArea(c)
    if minarea <= area <= maxarea:
      x, y, w, h = cv2.boundingRect(c)
      rectangle = [x, y, x+w, y+h]
      rectangles.append(rectangle)
      #cv2.rectangle(img, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), (0, 255, 0), 2)

  #cv2.imshow("BoundingBoxes", img)

  return rectangles

def sift_detection_with_Bb(current_img, images_infos : list, boundingBox):
  MIN_MATCHES = 15
  KnnDistance = 0.5

  img = cv2.cvtColor(current_img[boundingBox[0]:boundingBox[1],boundingBox[2]:boundingBox[0]], cv2.COLOR_BGR2GRAY)
  sift = cv2.SIFT_create()
  keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)

  index_params = dict(algorithm = 0 , trees = 5)
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  good_points = []

  for image_info in images_infos:
    if(image_info[2] is not None) and (descriptors_1 is not None):
      if (len(image_info[2]) >= 2) and (len(descriptors_1) >= 2):
        matches = flann.knnMatch(image_info[2], descriptors_1, k = 2)
        temp = []
        for m, n in matches:
          if m.distance < KnnDistance * n.distance:
            temp.append(m)
        if (len(temp) > len(good_points)):
          good_points = temp
          info = image_info

  if (len(good_points) >= MIN_MATCHES):
    print("detected " + info[3])
    current_img.rectangle(img, (boundingBox[0], boundingBox[1]), (boundingBox[2], boundingBox[3]), (0, 255, 0), 2)
    cv2.putText(current_img, info[3], (boundingBox[0], boundingBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

  return current_img

def video_recognition():

  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #Tochange in case

  height = 720
  width = 1280
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  images_infos = load_kp_samples()

  window_name = "JACK"

  totalimgArea = height * width
  bBmaxArea = totalimgArea / 6  # TODO A VOIR
  bBminArea = totalimgArea / 12  # TODO A VOIR

  while True:
    _, img = cap.read()

    if len(list_board_coords) < 4:
      cv2.setMouseCallback(window_name, mousePoints)
      for coord in list_board_coords:
        cv2.circle(img,coord,10,(0,255,0),-1)
    else:
      if (homographymatrix == None):
        homographymatrix = get_homography_matrix(img, np.array(list_board_coords), width, height)
      else:
        img = cv2.warpPerspective(img, homographymatrix, (img.shape[1], img.shape[0]))

    boundingBoxes = getBoundingBoxes(img,bBmaxArea,bBmaxArea)
    """
    if len(list_board_coords) < 4:
      cv2.setMouseCallback(window_name, mousePoints)
      for coord in list_board_coords:
        cv2.circle(img,coord,10,(0,255,0),-1)
    else:  
      img = get_homographied_board(img, np.array(list_board_coords), width, height)
    """

    for boundingBox in boundingBoxes:
      img = sift_detection_with_Bb(img, images_infos, boundingBox)
    cv2.imshow(window_name, img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break  
    
  cap.release()
  cv2.destroyAllWindows()


video_recognition()

