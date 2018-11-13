# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:31:06 2018

@author: Varun
"""

import cv2
from cv2 import imread,imshow,imwrite
import numpy as np 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
import os
import errno

root='E:/Acads/9th sem/SVM/training_data' 
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

def imEqualize(path, imsize, ch):
  img = imread(path,1)
  for i in range(3):
    img[:,:,i] = cv2.equalizeHist(img[:,:,i])
  img = cv2.resize(img, (imsize, imsize),0,0, cv2.INTER_LINEAR).reshape((1,imsize,imsize,ch))
  return img

def imRead(path, imsize, ch):
  img = imread(path,1)
  for i in range(3):
    img[:,:,i] = cv2.equalizeHist(img[:,:,i])
  img = cv2.resize(img, (imsize, imsize),0,0, cv2.INTER_LINEAR).reshape((1,imsize,imsize,ch))
  return img

def imPaths(i):
  imagepaths = [classpath[i]+'/'+item for item in os.listdir(classpath[i])]
  return imagepaths
def createPath(path):
  if not os.path.exists(os.path.dirname(path)):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exc: 
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
    imwrite(sroot+'/'+classes[pathi]+'/'+'s'+str(i)+'.jpg',imRead(imagepaths[i], image_size, num_channels)[0])

imagepaths = [unkroot+'/'+item for item in os.listdir(unkroot)]   
for i in range(len(imagepaths)):
  imwrite(sunkroot+'/'+'s'+str(i)+'.jpg',imRead(imagepaths[i], image_size, num_channels)[0])

  









