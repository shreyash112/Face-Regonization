import cv2
import numpy as np

#eyesDetect = cv2.CascadeClassifier('lbpcascade_silverware.xml');
facesDetect = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
cam = cv2.VideoCapture(0);

while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
   # eyes = eyesDetect.detectMultiScale(gray, 1.3, 5);
    faces = facesDetect.detectMultiScale(gray, 1.3, 5)
    #for(x,y,w,h) in eyes:
     #   cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)
    cv2.imshow("Faces",img);
    if (cv2.waitKey(1) & 0xFF  == ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
    
