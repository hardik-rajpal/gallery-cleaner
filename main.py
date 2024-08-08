import cv2
import numpy as np
import sys
import os
photosDir = './photos'
faceClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
)
import face_recognition
def detectFaces(photo:cv2.UMat):
    photoGray = cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
    rects,_,wts = faceClassifier.detectMultiScale3(photoGray,outputRejectLevels=True)
    return rects
"""
returns the index of the best match in knownFaceEncodings
-1 if no match is found.
"""
def getKnownFaceIndex(knownFaceEncodings,img):
    img = cv2.resize(img,dsize=(0,0),fx=0.5,fy=0.5)
    imgRGB = img[:,:,::-1]
    imgRGB = cv2.blur(imgRGB,(5,5))
    faceLocs = face_recognition.face_locations(imgRGB)
    # cv2.imshow('wer',imgRGB)
    # cv2.waitKey(0)
    faceEncodings = face_recognition.face_encodings(imgRGB,faceLocs)
    for encoding in faceEncodings:
        matches = face_recognition.compare_faces(knownFaceEncodings,encoding)
        faceDistances = face_recognition.face_distance(knownFaceEncodings, encoding)
        bestMatchIndex = np.argmin(faceDistances)
        minDistance = faceDistances[bestMatchIndex]
        if(minDistance > 0.5):
            # neat threshold for true positives, manually verified.
            continue
        if matches[bestMatchIndex]:
            return bestMatchIndex,minDistance
    return -1,-1
def main(args):
    if(len(args)<4):
        print('Usage: python3 main.py photosDir knownFacesDir outFacesDir')
        exit(0)
    photosDir = args[1]
    knownFacesDir = args[2]
    knownFaceEncodings = []
    knownFaceNames = []
    for knownFace in os.listdir(knownFacesDir):
        knownFacePath = os.path.join(knownFacesDir,knownFace)
        knownFaceImg = face_recognition.load_image_file(knownFacePath)
        knownFaceNames.append(knownFace.split('.')[0])
        knownFaceEncodings.append(face_recognition.face_encodings(knownFaceImg)[0])

    outFacesDir = args[3]


    # faces = []
    # for facePath in os.listdir(facesDir):
    #     faces.append(cv2.imread(facePath))
    photos = os.listdir(photosDir)
    i = 0
    totalPhotos = len(photos)
    for photoName in photos:
        print(f'processing: {i+1}/{totalPhotos}; ',end='')
        path = os.path.join(photosDir,photoName)
        photo = cv2.imread(path)
        # detectedFaces = detectFaces(photo)
        knownFaceIndex,distance = getKnownFaceIndex(knownFaceEncodings,photo)
        if(-1!=knownFaceIndex):
            os.rename(path,os.path.join(outFacesDir,photoName))
            print('Found '+knownFaceNames[knownFaceIndex]+\
                  ' ('+str(distance)+') in '+photoName)
        else:
            print('No known face in '+photoName)
            pass
            # leave non-face photos there.
        i+=1
if(__name__=='__main__'):
    main(sys.argv)