import cv2
import os

def get_object_position():
  pass

def get_homographied_board():
  pass

def sift_detection(currentImg, imageSamples : list):
  MIN_MATCHES = 60

  #img1 = cv2.imread(os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Samples", "DetectiveCard106.jpg")))  
  #img2 = cv2.imread(os.path.abspath(os.path.join(os.path.dirname( __file__ ), "DetectiveCard181.jpg"))) 
  for imageSample in imageSamples:
    dim = (400,400)
    imageSample = cv2.resize(cv2.imread(imageSample), dim, interpolation=cv2.INTER_LINEAR)  
    
    img1 = cv2.cvtColor(imageSample, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(currentImg, cv2.COLOR_BGR2GRAY)

    #sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    if(len(matches) >= MIN_MATCHES):
      #img3 = draw_boxes(matches,keypoints_1,img1,keypoints_2,img2)
      matchedImg = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
      return matchedImg

  return currentImg

def load_samples():
  PATH_SAMPLES = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Samples"))
  images = []
  dir = os.listdir(PATH_SAMPLES)
  for image in dir:
    images.append(os.path.join(PATH_SAMPLES,image))

  return images

def sift_landing_area():

  cap = cv2.VideoCapture(0) #Tochange in case
  images = load_samples()

  print(images)
  while True:
    success, img = cap.read()
    surfImage = sift_detection(img, images)  
    cv2.imshow('Result', surfImage)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break  
  cap.release()
  cv2.destroyAllWindows()


sift_landing_area()
