from __future__ import print_function
from pathlib import Path
import numpy as np
import cv2 as cv
import easygui as eagui
from GenerateXmlFile import *
import random


# Strings
windowTitle = 'XML File Generator'
modeWebcam = "webcam"
modeVideoFile = "videofile"
frametitle = "Use ESC or Q to stop now"

# Variables For Webcam
framerate = 30
fourccCodec = cv.VideoWriter_fourcc(*'XVID') # Codec
temporaryWebcamFilePath = './webcam_temp.avi' # Temporary File Name

# Folder Names
trainfolder = "train"
validationfolder = "validation"
imagesfolder = "images"
xmlfolder = "labels"

def openFileOrWebcam():
    global modeSelected
    if eagui.ynbox('Choose if you want to select a video file or access  webcam directly', windowTitle, ("Webcam", "Video File"),"","Webcam"):
        modeSelected = modeWebcam
        capture = cv.VideoCapture(1, cv.CAP_DSHOW)

        if not capture.isOpened:
            print('Unable to open Webcam')
            exit(0)

        height = 720
        width = 1280

        capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        global videoWriter
        videoWriter = cv.VideoWriter(temporaryWebcamFilePath, fourccCodec, framerate, (width, height))
    else:
        modeSelected = modeVideoFile
        path = eagui.fileopenbox('Select the video file you want to analyze', windowTitle)
        capture = cv.VideoCapture(cv.samples.findFileOrKeep(path))

        if not capture.isOpened:
            print('Unable to open Video File : ' + path)
            exit(0)

    return capture

def getRectanglesFromVideo(capture):
    rectangles = []
    i = 0

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        # First we convert the frame to a grayscale image
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # We then use a median blur technique to reduce the noise
        blur = cv.medianBlur(gray, 5)

        # We then apply a sharpening filter to enhance the edges
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv.filter2D(blur, -1, sharpen_kernel)

        # We can then threshold to get a binary image
        thresh = cv.threshold(sharpen, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

        # We invert the image so that the contours can get detected
        inverted = 255 - thresh

        # We also apply a close morphology transformation to get rid of the imperfections inside the shape
        morph_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        close = cv.morphologyEx(inverted, cv.MORPH_CLOSE, morph_kernel, iterations=3)  # We apply a close transformation

        # We then use findContours to get the contours of the shape
        cnts = cv.findContours(inverted, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        maxarea = 0
        # We then loop through all the detected contours to only retrieve the one rectangle with the maximum area
        for c in cnts:
            area = cv.contourArea(c)
            if area > maxarea:
                selectedcontour = c
                maxarea = area

        # We can then finally get the rectangle data and draw it on top of the image
        framecopy = frame.copy()

        x, y, w, h = cv.boundingRect(selectedcontour)
        cv.rectangle(framecopy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow(frametitle, framecopy)
        if modeSelected == modeWebcam:
            videoWriter.write(frame)

        rectangles.append([x, y, w, h])

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

        i = i + 1

    if modeSelected == modeWebcam:
        capture.release()
        videoWriter.release()

    cv.destroyAllWindows()
    return rectangles, i

def generateFolders():
    name = eagui.enterbox('Enter the name of your object', windowTitle)
    if name == "":
        name = "default"
    path = eagui.diropenbox('Enter the folder where you want the data to be generated (subfolders will be created)', windowTitle)
    path = path + '/' + name
    pathTrain = path + '/' + trainfolder
    pathValidation = path + '/' + validationfolder
    Path.mkdir(Path(pathTrain + '/' + imagesfolder),511,1,1)
    Path.mkdir(Path(pathTrain + '/' + xmlfolder),511,1,1)
    Path.mkdir(Path(pathValidation + '/' + imagesfolder),511,1,1)
    Path.mkdir(Path(pathValidation + '/' + xmlfolder),511,1,1)

    return name, pathTrain, pathValidation

def rewindCaptureOrOpenWebcamFile(capture):
    if modeSelected == modeVideoFile:
        capture.set(cv.CAP_PROP_POS_FRAMES, 0)
    else:
        capture = cv.VideoCapture(cv.samples.findFileOrKeep(temporaryWebcamFilePath))

        if not capture.isOpened:
            print('Unable to open Webcam Temporary File : ' + temporaryWebcamFilePath)
            exit(0)

    return capture

def generateJPGandXMLFiles(capture, rectangles, pathVariables, maxNumberOfFrames):
    i = 0
    name = pathVariables[0]
    pathTrain = pathVariables[1]
    pathValidation = pathVariables[2]

    while True:
        ret, frame = capture.read()
        if (frame is None) or (i > maxNumberOfFrames) :
            break


        size = {
            "width": rectangles[i][2],
            "height": rectangles[i][3],
            "depth": 3
        }
        bndbox = {
            "xmin": rectangles[i][0],
            "ymin": rectangles[i][1],
            "xmax": rectangles[i][0] + rectangles[i][2],
            "ymax": rectangles[i][1] + rectangles[i][3]
        }

        objectxml = {
            "name": name,
            "pose": 'Unspecified',
            "truncated": 0,
            "difficult": 0,
            "bndbox": bndbox
        }

        selectedPath = pathTrain
        if (random.randrange(10) == 1):
            selectedPath = pathValidation

        framename = name + str(i)

        cv.imwrite(selectedPath + "/images/" + framename + '.jpg', frame)
        annotation = Annotation(name, framename, selectedPath + "/labels/", sourceDict, size, 0, objectxml)
        annotation.generateAnnotationFile()
        i = i + 1

def main():
    random.seed

    capture = openFileOrWebcam()

    (rectangles, maxNumberOfFrames) = getRectanglesFromVideo(capture)

    if maxNumberOfFrames > 0:
        if eagui.ynbox('Are the object borders correctly detected ?', windowTitle, ('Yes', 'No'),"",'Yes','No'):
            pathVariables = generateFolders()
            capture = rewindCaptureOrOpenWebcamFile(capture)
            generateJPGandXMLFiles(capture, rectangles, pathVariables, maxNumberOfFrames)
        else:
            if eagui.ynbox('Do you want to restart the Capture ?', windowTitle, ('Yes', 'No'),"",'Yes','No'):
                main()

    capture.release()
    Path.unlink(Path(temporaryWebcamFilePath), 1)

if __name__ == '__main__':
    main()