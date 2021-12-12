import cv2
import os
from scipy import ndimage

from numpy.core.fromnumeric import resize

def get_average(img, u, v, n):
    s = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += img[u+i][v+j]
    return float(s)/(2*n+1)**2

def get_standard_deviation(img, u, v, n):
    s = 0
    avg = get_average(img, u, v, n)
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += (img[u+i][v+j] - avg)**2
    return (s**0.5)/(2*n+1) 


def zncc(img1, img2, u1, v1, u2, v2, n):
    """
    Calculate the ZNCC value for img1 and img2.
    """
    std_deviation1 = get_standard_deviation(img1, u1, v1, n)
    std_deviation2 = get_standard_deviation(img2, u2, v2, n)
    avg1 = get_average(img1, u1, v1, n)
    avg2 = get_average(img2, u2, v2, n)

    s = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += (img1[u1+i][v1+j] - avg1)*(img2[u2+i][v2+j] - avg2)
    return float(s)/((2*n+1)**2 * std_deviation1 * std_deviation2)

def zncc_score(circleimg, samples = [], orientation = 'up'):

    cardCCscore = []

    for sample in samples:
        resizedSample = sample
        if(orientation == 'left'):
            resizedSample =cv2.rotate(resizedSample,cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif(orientation == 'right'):
            resizedSample =cv2.rotate(resizedSample,cv2.ROTATE_90_CLOCKWISE)
        elif(orientation == 'down'):
            resizedSample =cv2.rotate(resizedSample,cv2.ROTATE_180)
        resizedSample = cv2.resize(resizedSample, (circleimg.shape[1],circleimg.shape[0]))

        cardCCscore.append(max(0,zncc(cv2.cvtColor(circleimg,cv2.COLOR_BGR2GRAY), resizedSample, 15, 15, 15, 15, 25)))

    total = sum(cardCCscore)
    cardProba = []

    for i in range(len(cardCCscore)):
        cardProba.append(cardCCscore[i]/total)

    return cardProba

def zncc_pawn(img, samples = []):

    pawnCCscore = []
    for sample in samples:
        resizedSample = cv2.resize(sample, (img.shape[1],img.shape[0]))
        [h,w] = resizedSample.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        #cv2.imshow(str(-1), resizedSample)

        maxScore = max(0,zncc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), resizedSample, 15, 15, 15, 15, 25)) # TODO Max a voir ?
        for i in range(7):
            M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
            resizedSample = cv2.warpAffine(resizedSample, M, (w, h))

            #cv2.imshow(str(i),resizedSample)
            maxScore = max(maxScore,zncc(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), resizedSample, 15, 15, 15, 15, 25))

        pawnCCscore.append(maxScore)

    total = sum(pawnCCscore)
    pawnProba = []

    if (total>0):
        for i in range(len(pawnCCscore)):
            pawnProba.append(pawnCCscore[i]/total)
    else:
        pawnProba = pawnCCscore

    return pawnProba