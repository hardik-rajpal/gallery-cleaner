import cv2
import numpy as np
import sys
import os
photosDir = './photos'
faceClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
)
def detectFaces(photo:cv2.UMat):
    detectedFaces = []
    photoGray = cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
    rects,_,wts = faceClassifier.detectMultiScale3(photoGray,outputRejectLevels=True)
    return rects
def main(args):
    if(len(args)<2):
        print('Usage: python3 main.py photosDir facesDir')
        exit(0)
    photosDir = args[1]
    facesDir = args[2]
    # faces = []
    # for facePath in os.listdir(facesDir):
    #     faces.append(cv2.imread(facePath))

    for photoName in os.listdir(photosDir):
        path = os.path.join(photosDir,photoName)
        print(path)
        photo = cv2.imread(path)
        detectedFaces = detectFaces(photo)
        if(len(detectedFaces)>0):
            os.rename(path,os.path.join(facesDir,photoName))
        else:
            pass
            # leave non-face photos there.
if(__name__=='__main__'):
    main(sys.argv)