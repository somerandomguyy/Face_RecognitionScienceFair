import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images (attendance)'
images = []
classNames = []
mylist = os.listdir(path)
# print(mylist)
for cls in mylist:
    currentImg = cv2.imread(f'{path}/{cls}')
    images.append(currentImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)


def findencodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markattendance(name):
    with open('Mark_attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        namelist = []
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')


encodeListknown = findencodings(images)
print('Encoding Complete')

capture = cv2.VideoCapture(0)

while True:
    Success, img = capture.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    curFrameFaces = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, curFrameFaces)

    for encodeFace, faceLocation in zip(encodesCurFrame, curFrameFaces):
        matches = face_recognition.compare_faces(encodeListknown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListknown, encodeFace)
        # print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (35, 181, 224), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (35, 181, 224), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
            markattendance(name)

        else:
            name = 'UNKNOWN'
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (35, 181, 224), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (35, 181, 224), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
            markattendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)