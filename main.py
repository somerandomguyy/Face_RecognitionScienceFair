import cv2
import numpy as np
import face_recognition

imgMain = face_recognition.load_image_file('images (main)/Leonardo DiCaprio.jpg')
imgMain = cv2.cvtColor(imgMain, cv2.COLOR_BGR2RGB)
imgMainTest = face_recognition.load_image_file('images (main)/Leonardo DiCaprio Test.jpg')
imgMainTest = cv2.cvtColor(imgMainTest, cv2.COLOR_BGR2RGB)

faceLocation = face_recognition.face_locations(imgMain)[0]
encodeDicaprio = face_recognition.face_encodings(imgMain)[0]
cv2.rectangle(imgMain, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (33, 71, 218), 2)

faceLocationTest = face_recognition.face_locations(imgMainTest)[0]
encodeDicaprioTest = face_recognition.face_encodings(imgMainTest)[0]
cv2.rectangle(imgMainTest, (faceLocationTest[3], faceLocationTest[0]), (faceLocationTest[1], faceLocationTest[2]), (33, 71, 218), 2)

results = face_recognition.compare_faces([encodeDicaprio], encodeDicaprioTest)
faceDistance = face_recognition.face_distance([encodeDicaprio],encodeDicaprioTest)
print(results, faceDistance)
cv2.putText(imgMainTest, f'{results} {round(faceDistance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

cv2.imshow('Leonardo DiCaprio', imgMain)
cv2.imshow('Leonardo DiCaprio Test', imgMainTest)
cv2.waitKey(0)