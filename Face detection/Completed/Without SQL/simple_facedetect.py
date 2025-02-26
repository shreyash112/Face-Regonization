##Codacus youtube channel
"""
Created on Mon Jun 04 20:05:50 2018

@author: Shreyash
"""

## start

import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);

while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("Faces",img);
    if (cv2.waitKey(1) & 0xFF  == ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
    