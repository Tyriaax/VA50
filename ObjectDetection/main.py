import os

from video_recognition import *
from image_recognition import *

def main():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'zebi.mp4'))
    img = cv2.imread(path)
    video_recognition()
    #video_recognition(path)

if __name__ == '__main__':
    main()