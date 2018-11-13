# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:31:32 2018

@author: Varun
"""

import dlib
import cv2
from cv2 import imread,imshow,imwrite
import numpy as np 
import os
import errno
import matplotlib.pyplot as plt
from distutils.dir_util import copy_tree
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

root='E:/Acads/9th sem/SVM/training_data' 
unkroot = 'E:/Acads/9th sem/SVM/training_data/unk'
unkroots = 'E:/Acads/9th sem/SVM/training_data_others'
smodel = 'E:/Acads/9th sem/SVM/shape_predictor_68_face_landmarks.dat'
model = 'E:/Acads/9th sem/SVM/dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(model)
shaper = dlib.shape_predictor(smodel)
detector = dlib.get_frontal_face_detector()
"""
Get classe names from the root directory
calsses = classes in the root directory 
num_classes =  number of classes 
"""
def labelData(path):
  classes = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) and item!="unk"]
  return classes, [root+'/'+str(x) for x in classes],len(classes)

classes, classpath,num_classes = labelData(root)

image_size = 128
num_channels = 3

def getVec(path):
  img  = imread(path,1)
  if (len(detector(img, 1)) == 1):
    return np.array(facerec.compute_face_descriptor(img,shaper(img,detector(img, 1)[0])))
  else:
    return -np.ones((128))
  
def dist(v1,v2):
  np.sum((v1-v2)**2)**0.5
  
def imPaths(i):
  imagepaths = [classpath[i]+'/'+item for item in os.listdir(classpath[i])]
  return imagepaths

total = 0
for pathi in range(num_classes):
  imagepaths = imPaths(pathi)
  l = len(imagepaths)
  total += l

vecs_train = np.zeros((num_classes*4,129))
vecs_test = np.zeros((total - num_classes*4 ,129))
jtr = 0
jts = 0
for pathi in range(num_classes):
  imagepaths = imPaths(pathi)
  l = len(imagepaths)
  for i in range(4):
    vecs_train[jtr,0] = classes[pathi]
    vecs_train[jtr,1:] = np.array(getVec(imagepaths[i])).reshape((128))
    jtr+=1
  for i in range(4,l):
    vecs_test[jts,0] = classes[pathi]
    vecs_test[jts,1:] = np.array(getVec(imagepaths[i])).reshape((128))
    jts+=1
  
V_train = np.delete(vecs_train,28,0)[:,1:]
V_test = vecs_test[:,1:]
y_train = np.delete(vecs_train,28,0)[:,0]
y_test = vecs_test[:,0]

reg = LinearSVC()
reg.fit(V_train,y_train)
reg.score(V_test,y_test)


imagepaths = [unkroot+'/'+item for item in os.listdir(unkroot)]
l = len(imagepaths)
vecs_unk = np.zeros((l ,128))
for i in range(l):
    vecs_unk[i,:] = np.array(getVec(imagepaths[i])).reshape((128))
V_unk = vecs_unk

folders = [item for item in os.listdir(unkroots) if os.path.isdir(os.path.join(unkroots, item)) ]
total = 0
for j in folders:
  imagepaths = [unkroots+'/'+j+'/'+item for item in os.listdir(unkroots+'/'+j)]
  l = len(imagepaths)
  total += l
vecs_unk2 = np.zeros((total,128))
ju = 0
for j in folders:
  imagepaths = [unkroots+'/'+j+'/'+item for item in os.listdir(unkroots+'/'+j)]
  l = len(imagepaths)
  for i in range(l):
    vecs_unk2[ju,:] = np.array(getVec(imagepaths[i])).reshape((128))
    ju+=1
V_unk2 = vecs_unk2

Eucdist = np.zeros((V_unk.shape[0]+V_test.shape[0]+V_unk2.shape[0],V_train.shape[0]))
for i in range(V_train.shape[0]):
  for j in range(V_unk.shape[0]):
    Eucdist[j,i] = np.linalg.norm(V_train[i,:]-V_unk[j,:])
for i in range(V_train.shape[0]):
  for j in range(V_unk2.shape[0]):
    Eucdist[j+V_unk.shape[0],i] = np.linalg.norm(V_train[i,:]-V_unk2[j,:])
for i in range(V_train.shape[0]):
  for j in range(V_test.shape[0]):
    Eucdist[j+V_unk.shape[0]+V_unk2.shape[0],i] = np.linalg.norm(V_train[i,:]-V_test[j,:])
Euarg = np.argmin(Eucdist,axis =1)
Eu = np.c_[np.min(Eucdist,axis =1),np.argmin(Eucdist,axis =1)]




bar = 0.3
def predictClass(fpath,reg=reg,V_train= V_train,bar = bar):
  vec = np.array(getVec(fpath)).reshape((128))
  Eucdist = np.zeros((1,V_train.shape[0]))
  for i in range(V_train.shape[0]):
    Eucdist[0,i] = np.linalg.norm(V_train[i,:]-vec)
  if(np.min(Eucdist,axis =1)>=bar):
    print("Unknown Image")
    return -1
  else:
    print(reg.predict(vec.reshape(1, -1)))
    return reg.predict(vec.reshape(1, -1))
  
def predictFolder(fpath,reg=reg,V_train= V_train,bar = bar):
  imagepaths = [fpath+'/'+item for item in os.listdir(fpath)]
  return [predictClass(imagepaths[i],reg=reg,V_train= V_train,bar = bar) for i in range(len(imagepaths))]
    
    
pred = predictFolder('E:/Acads/9th sem/SVM/training_data/103')
























