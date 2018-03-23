__author__ = 'hooloongge'
# -*- coding: utf-8 -*-
import cv2
import numpy as np


cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
print (fps)
fps = 30
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print (size)
# fourcc = cv2.cv.CV_FOURCC(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,fps,size)
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)


    if cv2.waitKey(10) &0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()