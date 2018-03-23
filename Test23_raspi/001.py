import cv2
import os
import sys

OUTPUT_DIR = '../../my_faces'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

count = 0
while True:
    print(count)
    if count < 10000:
        _, img = cam.read()

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_haar.detectMultiScale(gray_image, 1.3, 5)
        for face_x, face_y, face_w, face_h in faces:
            face = img[face_y:face_y + face_h, face_x:face_x + face_w]

            face = cv2.resize(face, (64, 64))
            cv2.imshow('img', face)
            cv2.imwrite(os.path.join(OUTPUT_DIR, str(count) + '.jpg'), face)
            count += 1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        break