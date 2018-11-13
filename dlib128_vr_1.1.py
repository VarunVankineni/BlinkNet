# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:24:47 2018

@author: Varun
"""

import dlib
import cv2
from cv2 import imread
import numpy as np 
import os
from sklearn.svm import LinearSVC
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from scipy.optimize import minimize
"""
%%%Paths%%%
"""
folder = 'E:/Acads/9th sem/SVM/'
root='E:/Acads/9th sem/SVM/lfw-deepfunneled_sort5b'
unkroot = 'E:/Acads/9th sem/SVM/training_data/unk1'
unkroots = 'E:/Acads/9th sem/SVM/lfwunk1'
smodel = 'E:/Acads/9th sem/SVM/shape_predictor_68_face_landmarks.dat'
model = 'E:/Acads/9th sem/SVM/dlib_face_recognition_resnet_model_v1.dat'

"""
%%%Parameters%%%
"""


image_size = 128
num_channels = 3
bar = 70           #percentage similiraity for detection 
train_nos = 4
bar = (100-bar)/100

"""
Initialize Models 
"""
facerec = dlib.face_recognition_model_v1(model)  #feature vector resnet model
shaper = dlib.shape_predictor(smodel)            #coordinate helper for splicing face from image
detector = dlib.get_frontal_face_detector()      #face box detector

"""
Get labels from the given folder with paths for all the labels and number of labels
"""
def labelData(path):
  classes = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) and item!="unk"]
  return classes, [root+'/'+str(x) for x in classes],len(classes)

"""
Helper function to convert image at path into feature space with side variable 
denoting the number of face detected
"""
def getVec(path):
  img  = imread(path,1)
  output = np.zeros(129)
  dets,scores,idx = detector.run(img,1)
  output[0] = len(dets)
  if (output[0] > 0):
    output[1:] = np.array(facerec.compute_face_descriptor(img,shaper(img,dets[int(np.argmax(scores))])))
  else:
    output[1:] = np.zeros((128))
    print(path)
  return output

def getVecI(img):
  output = np.zeros(129)
  dets,scores,idx = detector.run(img,1)
  output[0] = np.sign(len(dets))
  if (output[0] > 0):
    output[1:] = np.array(facerec.compute_face_descriptor(img,shaper(img,dets[int(np.argmax(scores))])))
  else:
    output[1:] = np.zeros((128))
  return output

"""
imPaths = Getting paths from the label using its index in classpath
collectValid = get feature vecs that are valid i.e, single face only images (only used for training and testing)
train = train the classfier and return the model
"""
def imPaths(i):
  imagepaths = [classpath[i]+'/'+item for item in os.listdir(classpath[i])]
  return imagepaths

def collectValid(vecs):
  V = np.delete(vecs[vecs[:,1]==1,:],1, 1)
  return V[:,1:], V[:,0]

def train(V_train,y_train):
  reg = LinearSVC()
  reg.fit(V_train,y_train)
  return reg

classes, classpath,num_classes = labelData(root)
num_classes = 100
"""
Find the number of images in the available dataset
Intitalize training and testing feature vectors accordingly
Iterate over each label and save the feature vectors of each image with
corresponding index of the label
Get the valid feature vetors from the total dataset
Train the model and output the prediction score on the test dataset
"""
total = 0
for pathi in range(num_classes):
  imagepaths = imPaths(pathi)
  l = len(imagepaths)
  total += l
#vecs_full = np.zeros((total,130))
vecs_train = np.zeros((num_classes*train_nos,130))
vecs_test = np.zeros((total - num_classes*train_nos ,130))
jtr = 0
jts = 0
jtf = 0
for pathi in range(num_classes):
  imagepaths = imPaths(pathi)
  l = min(10,len(imagepaths))
#  for i in range(l):
#    vecs_full[jtf,0] = pathi
#    vecs_full[jtf,1:] = np.array(getVec(imagepaths[i])).reshape((129))
#    jtf+=1
    
  for i in range(train_nos):
    vecs_train[jtr,0] = pathi
    vecs_train[jtr,1:] = np.array(getVec(imagepaths[i])).reshape((129))
    jtr+=1
  for i in range(train_nos,l):
    vecs_test[jts,0] = pathi
    vecs_test[jts,1:] = np.array(getVec(imagepaths[i])).reshape((129))
    jts+=1
  print("Extracted Class "+str(pathi+1)+" : "+ classes[pathi]) 

#np.save('E:/Acads/9th sem/SVM/train.npy', vecs_train)
#np.save('E:/Acads/9th sem/SVM/test.npy', vecs_test)
#vecs_train = np.load('E:/Acads/9th sem/SVM/train.npy')
#vecs_test = np.load('E:/Acads/9th sem/SVM/test.npy')

V_train, y_train = collectValid(vecs_train)
V_test, y_test = collectValid(vecs_test)

reg = train(V_train,y_train)
print(reg.score(V_test,y_test))


"""
getVecs = return all the image path and feature vectors of the corresponding
images from the given folder. Removes invalid images internally.
"""
def getVecs(path):
  total = 0
  for p, subdirs, files in os.walk(path):
    for name in files:
      total+=1
  print("extracting "+str(total)+" images")
  vecs = np.zeros((total,129))
  pathlist = [0 for x in range(total)]
  itr = 0
  for p, subdirs, files in os.walk(path):
    for name in files:
      if int(10*itr/total)%10 == 0 : print(str(int(10*itr/total))+"% extracted")  
      pathlist[itr] = os.path.join(p, name)
      vecs[itr,:] = np.array(getVec(pathlist[itr])).reshape((129))
      itr+=1
  vcol = vecs[:,0]
  vecs = vecs[vcol==1,:][:,1:]
  pathlist = [pathlist[x] for x in range(total) if vcol[x]]
  return vecs,pathlist

"""
Get feature vectors for unknown images
Get the minimum distance for analysis
"""
V_unk,path_unk = getVecs(unkroot)
V_unk2,path_unk2 = getVecs(unkroots)

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

cosdist = np.zeros((V_unk.shape[0]+V_test.shape[0]+V_unk2.shape[0],V_train.shape[0]))
for i in range(V_train.shape[0]):
  for j in range(V_unk.shape[0]):
    cosdist[j,i] = cosine(V_train[i,:],V_unk[j,:])
for i in range(V_train.shape[0]):
  for j in range(V_unk2.shape[0]):
    cosdist[j+V_unk.shape[0],i] = cosine(V_train[i,:],V_unk2[j,:])
for i in range(V_train.shape[0]):
  for j in range(V_test.shape[0]):
    cosdist[j+V_unk.shape[0]+V_unk2.shape[0],i] = cosine(V_train[i,:],V_test[j,:])
cosarg = np.argmin(cosdist,axis =1)
cos = np.c_[np.min(cosdist,axis =1),np.argmin(cosdist,axis =1)]

"""
predictClass = predicts the class of the image at given path, other cases are 
also covered
predictFolder = predicts classes for all the images in a folder
"""
def predictClass(path,reg=reg,V_train= V_train,bar = bar):
  vec = np.array(getVec(path)).reshape((129))
  faces = vec[0]
  if(faces==0):
    return 'Face Not detected'
  elif(faces>1):
    return str(faces-1) + 'extra faces detected' 
  vec = vec[1:]
  Eucdist = np.zeros((1,V_train.shape[0]))
  for i in range(V_train.shape[0]):
    Eucdist[0,i] = np.linalg.norm(V_train[i,:]-vec)
  if(np.min(Eucdist,axis =1)>=bar):
    return 'Unknown'
  else:
    return classes[int(reg.predict(vec.reshape(1, -1))[0])]
  
def predictImage(img,reg=reg,V_train= V_train,bar = bar):
  vec = np.array(getVecI(img)).reshape((129))
  faces = vec[0]
  if(faces==0):
    return 'Face Not detected'
  elif(faces>1):
    return str(faces-1) + 'extra faces detected' 
  vec = vec[1:]
  Eucdist = np.zeros((1,V_train.shape[0]))
  for i in range(V_train.shape[0]):
    Eucdist[0,i] = np.linalg.norm(V_train[i,:]-vec)
    
  print(np.min(Eucdist,axis =1))
  if(np.min(Eucdist,axis =1)>=bar):
    return 'Unknown'
  else:
    return classes[int(reg.predict(vec.reshape(1, -1))[0])]
  
def predictFolder(fpath,reg=reg,V_train= V_train,bar = bar):
  imagepaths = [fpath+'/'+item for item in os.listdir(fpath)]
  return [predictClass(imagepaths[i],reg=reg,V_train= V_train,bar = bar) for i in range(len(imagepaths))]

#pred = predictFolder('E:/Acads/9th sem/SVM/training_data/103')
#print(pred)

def distribution(data, bins = 10):
  tests = np.histogram(data,bins)
  plt.bar(tests[1][:-1], tests[0], width = tests[1][5]/100)
  
a = V_unk.shape[0]
b = V_unk2.shape[0]
distribution(cos[:a,0])
distribution(cos[a:a+b,0])
distribution(cos[a+b:,0])
distribution(Eu[:a,0])
distribution(Eu[a:a+b,0])
distribution(Eu[a+b:,0])

def separation_error(m,Eu,a,w1=1,w2=1):
  n1 = 1 - (sum(Eu[:a,0]>m)/a)
  n2 = 1 - (sum(Eu[a:,0]<m)/(len(Eu)-a))
  return ((n1**(1/w1))*a + (n2**(1/w2))*(len(Eu)-a))/len(Eu)

def raw_error(m,Eu,a):
  n1 = 1 - (sum(Eu[:a,0]>m)/a)
  n2 = 1 - (sum(Eu[a:,0]<m)/(len(Eu)-a))
  return n1,n2
erlist = [separation_error(0.3+(i/250),Eu,b,1,1) for i in range(100)]
erlist2 = [separation_error(0.3+(i/250),Eu,b,3,0.5) for i in range(100)]

plt.plot(erlist)
plt.plot(erlist2)

res = minimize(separation_error,0.6,(Eu,b,3,0.5),method='Nelder-Mead', tol=1e-6)
bar = res.x
raw = raw_error(bar,Eu,b)
accuracy = [100*(1-i) for i in raw]
print(raw)
print(accuracy)

seconds = 2
arnold = cv2.VideoCapture(folder+'Arnold.mp4')
for j in range(10):
  for i in range(int(29*seconds)): ret, img = arnold.read()
  cv2.imshow('img', img)
  print(predictImage(img[:,:800,:]))

harrison= cv2.VideoCapture(folder+'harrison.mp4')
for j in range(5):
  for i in range(int(29*seconds)): ret, img = harrison.read()
  cv2.imshow('img', img)
  print(predictImage(img[:,:,:]))

img = cv2.imread('E:/Acads/9th sem/SVM/training_data/139/1392.png',1)
color = ('b','g','r')
for i,col in enumerate(color):
  histr = cv2.calcHist([img],[i],None,[256],[0,256])
  plt.plot(histr,color = col)
  plt.xlim([0,256])
plt.show()