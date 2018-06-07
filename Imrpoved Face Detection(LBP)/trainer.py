#@author: Shreyash

import os
import cv2
import numpy as np 
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create() 
path = 'dataset'

def getImagesNid(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples =[]
    IDs =[]
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L');
        facenp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(facenp)
        IDs.append(ID)
        print(ID)
    return faceSamples,IDs

faceSamples,IDs = getImagesNid('dataset')
recognizer.train(faceSamples,np.array(IDs))
recognizer.save('recognizer/trainingData.yml')
