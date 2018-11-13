# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:53:31 2018

@author: Varun
"""

import dlib
import cv2
from cv2 import imread,imshow,imwrite
import numpy as np 
import os
import errno
from skimage import io
import matplotlib.pyplot as plt

imsize = 128
ch = 3 


root='E:/Acads/9th sem/SVM/training_data_straight' 
sroot='E:/Acads/9th sem/SVM/training_datas' 
unkroot = 'E:/Acads/9th sem/SVM/training_data/unk'
sunkroot = 'E:/Acads/9th sem/SVM/training_datas/unks'

"""
Get classe names from the root directory
calsses = classes in the root directory 
num_classes =  number of classes 
"""
classes = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) and item!="unk"]
classes= [int(x) for x in classes]
classes.sort()
classes = [str(i) for i in classes]
print(classes)
num_classes = len(classes)

"""
classpath = list of class paths for each class (folders) in the root directory
Note: images are not imported here
"""
image_size = 128
num_channels = 3

classpath = [root+'/'+str(x) for x in classes]

#img = cv2.resize(img, (imsize, imsize),0,0, cv2.INTER_LINEAR)#.reshape((1,imsize,imsize,ch))
#face_detector = dlib.get_frontal_face_detector()
#detected_faces = face_detector(img, 1)
#face_frames = [(x.left(), x.top(),
#                x.right(), x.bottom()) for x in detected_faces]
#imgycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
#imgycc[:,:,0] = cv2.equalizeHist(imgycc[:,:,0])
#imgeq = cv2.cvtColor(imgycc, cv2.COLOR_YCR_CB2BGR)
#imshow('n',img)
#imshow('eq',imgeq)
#imcrop = imgeq[15:105,25:115,:]
#imshow('c',imcrop)


def imRead(path, imsize, ch):
  img = imread(path,1)
  face_detector = dlib.get_frontal_face_detector()
  faces = face_detector(img, 1)
  frame = [(x.left(), x.top(),
                x.right(), x.bottom()) for x in faces]
  if len(frame)!=1 :
    print("invalid")
    return cv2.resize(img, (imsize, imsize),0,0, cv2.INTER_LINEAR)
  else:
    frame = frame[0]
  imgycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
  imgycc[:,:,0] = cv2.equalizeHist(imgycc[:,:,0])
  img = cv2.cvtColor(imgycc, cv2.COLOR_YCR_CB2BGR)
  img = img[frame[0]:frame[2],frame[1]:frame[3]]  
  return cv2.resize(img, (imsize, imsize),0,0, cv2.INTER_LINEAR)

def imPaths(i):
  imagepaths = [classpath[i]+'/'+item for item in os.listdir(classpath[i])]
  return imagepaths
def createPath(path):
  if not os.path.exists(os.path.dirname(path)):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
#os.makedirs(sroot)
#os.makedirs(sunkroot)
for pathi in range(num_classes):
  imagepaths = imPaths(pathi)
  try: 
    os.makedirs(sroot+'/'+classes[pathi])
  except:
    pass
  for i in range(len(imagepaths)):
    imwrite(sroot+'/'+classes[pathi]+'/'+'s'+str(i)+'.jpg',imRead(imagepaths[i], image_size, num_channels))

imagepaths = [unkroot+'/'+item for item in os.listdir(unkroot)]   
for i in range(len(imagepaths)):
  imwrite(sunkroot+'/'+'s'+str(i)+'.jpg',imRead(imagepaths[i], image_size, num_channels))
