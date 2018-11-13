# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:54:01 2018

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
from distutils.dir_util import copy_tree

root='E:/Acads/9th sem/SVM/lfw-deepfunneled_sort51' 
sroot='E:/Acads/9th sem/SVM/lfw-deepfunneled_sort514'
smodel = 'E:/Acads/9th sem/SVM/shape_predictor_68_face_landmarks.dat'

img = imread(sroot+"/Abdullah_Gul/Abdullah_Gul_0001.jpg",1)
shaper = dlib.shape_predictor(smodel) 
detector = dlib.get_frontal_face_detector()  
image = dlib.get_face_chip(img, shaper(img,detector(img, 1)[0]))


def createPath(path):
  try:
    os.makedirs(path)
  except:
    pass
createPath(sroot)
classes = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
imsize = 160
for i in range(len(classes)):
  createPath(sroot+"/"+classes[i])
  impaths = [ item for item in os.listdir(root+"/"+classes[i]) ]
  for j in impaths[:4]:
    img = imread(root+"/"+classes[i]+"/"+j,1)
    img = cv2.resize(img, (imsize, imsize),0,0, cv2.INTER_LINEAR)
    imwrite(sroot+"/"+classes[i]+"/"+j,img)
    
    
    
    
    