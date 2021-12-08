import os

from video_recognition import *
from image_recognition import *

def main():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'game.mp4'))
    imagepath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plateau.jpg'))

    #image_recognition(imagepath)
    video_recognition()

if __name__ == '__main__':
    main()