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
    faceCoords = faceClassifier.detectMultiScale(photoGray)
    
def main(args):
    if(len(args)<2):
        print('Usage: python3 main.py photosDir facesDir')
        exit(0)
    photosDir = args[0]
    facesDir = args[1]
    faces = []
    for facePath in os.listdir(facesDir):
        faces.append(cv2.imread(facePath))

    for photoPath in photosDir:
        photo = cv2.imread(photoPath)
        detectedFaces = detectFaces(photo)
if(__name__=='__main__'):
    main(sys.argv)