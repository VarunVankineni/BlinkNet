# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:38:39 2018

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

root='E:/Acads/9th sem/SVM/lfw-deepfunneled' 
sroot='E:/Acads/9th sem/SVM/lfw-deepfunneled_sorted5'
model = 'E:/Acads/9th sem/SVM/dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(model)



def createPath(path):
  try:
    os.makedirs(path)
  except:
    pass
createPath(sroot)
classes = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
for c in classes:
  d = root+"/"+c
  sd = sroot+"/"+c
  if(len([item for item in os.listdir(d)])>5):
    createPath(sd)
    copy_tree(d, sd)
    
