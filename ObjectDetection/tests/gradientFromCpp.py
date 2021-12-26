import cv2

def gradient(pathImg):

    img = cv2.imread(pathImg,0)
    sz = img.shape
    sz = sz*4
    img = cv2.resize(img,(sz[0]*2,sz[1]*2))
    cv2.imshow("img",img)

    # set the kernel size, depending on whether we are using the Sobel
    # operator of the Scharr operator, then compute the gradients along
    # the x and y axis, respectively
    ksize = 3 #if args["scharr"] > 0 else 3
    gX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    # the gradient magnitude images are now of the floating point data
    # type, so we need to take care to convert them back a to unsigned
    # 8-bit integer representation so other OpenCV functions can operate
    # on them and visualize them
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    # combine the gradient representations into a single image
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    # show our output images
    
    cv2.imshow("Sobel/Scharr X", gX)
    cv2.imshow("Sobel/Scharr Y", gY)
    cv2.imshow("Sobel/Scharr Combined", combined)

    # cv2.addWeighted( gX, 0.5, gY, 0.5, 0, combined)

    # orientation = [gX, gY]

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break  

gradient("..\Samples\LQ\Pawns\ActionPawns\AP1Sherlock.png")

