from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
import easygui as eagui
from GenerateXmlFile import *

Title = 'XML File Generator'
path = eagui.enterbox('Enter the video file you want to analyze', Title)

capture = cv.VideoCapture(cv.samples.findFileOrKeep(path))
if not capture.isOpened:
    print('Unable to open: ' + path)
    exit(0)

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
    thresh = cv.threshold(sharpen, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    # We invert the image so that the contours can get detected
    inverted = 255 - thresh

    # We also apply a close morphology transformation to get rid of the imperfections inside the shape
    morph_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    close = cv.morphologyEx(inverted, cv.MORPH_CLOSE, morph_kernel, iterations=3)  # We apply a close transformation

    # We then use findContours to get the contours of the shape
    cnts = cv.findContours(inverted, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    maxarea = 0
    #We then loop through all the detected contours to only retrieve the one with the maximum area
    for c in cnts:
        area = cv.contourArea(c)
        if area > maxarea:
            selectedcontour = c
            maxarea = area

    # We can then finally draw the contour on tof of the image
    framecopy = frame

    x, y, w, h = cv.boundingRect(selectedcontour)
    cv.rectangle(framecopy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('frame', framecopy)

    rectangles.append([x, y, w, h])

    i = i + 1

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

if eagui.ynbox('Are the object borders correctly detected ?', Title, ('Yes', 'No')):
    name = eagui.enterbox('Enter the name of your object', Title)
    pathtosave = eagui.enterbox('Enter the path where you want your files to be generated', Title)

    i = 0
    capture.set(cv.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        size = {
            "width": rectangles[i][2],
            "height": rectangles[i][3],
            "depth": 3
        }
        bndbox = {
            "xmin": rectangles[i][0],
            "ymin": rectangles[i][1],
            "xmax": rectangles[i][0]+rectangles[i][2],
            "ymax": rectangles[i][1]+rectangles[i][3]
        }

        objectxml = {
            "name" : name,
            "pose" : 'Unspecified',
            "truncated" : 0,
            "difficult" : 0,
            "bndbox" : bndbox
        }

        framename = name+str(i)

        cv.imwrite(pathtosave + "/" + framename + '.jpg', frame)
        annotation = Annotation(name, framename, pathtosave, sourceDict, size, 0, objectxml)
        annotation.generateAnnotationFile()
        i = i+1
