# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:41:56 2018

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


"""
root directories 
root : training data with folder names as classes inside this directory
unkroot : unknown images with are not present in both training/ validation dataset
"""
root='E:/Acads/9th sem/SVM/training_data' 
unkroot = 'E:/Acads/9th sem/SVM/training_data/unk'

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
classpath = [root+'/'+str(x) for x in classes]

"""
Reset tensorflow board for retraining 
"""

tf.reset_default_graph()

"""
Placeholder variables 
x = input variable for first image
x1 = input variable for second image
y = class label of the first image 
y1 = class label of the second image
"""
x = tf.placeholder(tf.float32, (None, 128, 128, 3))
x1 = tf.placeholder(tf.float32, (None, 128, 128, 3))
y = tf.placeholder(tf.float32, (None))
y1 = tf.placeholder(tf.float32, (None))

"""
%%PARAMETERS

training_size = size of the training set, either this or a minimum of 2 images will be used for training
img_size = all images will be input into this shape [img_size, img_size]
num_channels = number of channels of the input image, RGB = 3, Greyscale  = 1
"""
training_size = 0.2
training_size = 1/training_size
image_size = 128
num_channels = 3

"""
Network Parameters
"""
filters1 = 64           #number of filters in 1st convolution
frame1 = [10,10]    #frame of the convolution filter1
filters2 =  128   #number of filters in 2nd convolution
frame2 = [7,7]     #frame of the convolution filter2
filters3 = 256    #number of filters in 3rd convolution
frame3 = [4,4]     #frame of the convolution filter3
filters4 = 512    #number of filters in 4th convolution
frame4 = [3,3]     #frame of the convolution filter4
maxpoolframe = [2,2]   #maxpooling frame size
maxpoolstrides = [2,2] #stride of maxpooling frame
fconshape1 = 4096       #neurons in 1st fully connected layer

"""
Optimizer parameters
"""
a = 0.0000001                #learning rate 
p = 10                #length between same class images
P = 100               #length between different class images
maxitr = 100         #maximum number of epochs
maxclasses = min(15,len(classes))  # number of classes to use of the classes given

"""
Network Architecture 
x suffix = This is the siamese twin network which shares the weights with the original network
"""
conv1 = tf.layers.conv2d(x, filters1, frame1, activation = tf.nn.relu, name = 'conv1', reuse = None)
pool1 = tf.layers.max_pooling2d(conv1,maxpoolframe,maxpoolstrides)
conv2 = tf.layers.conv2d(pool1, filters2, frame2, activation = tf.nn.relu, name = 'conv2', reuse = None)
pool2 = tf.layers.max_pooling2d(conv2,maxpoolframe,maxpoolstrides)
conv3 = tf.layers.conv2d(pool2, filters3, frame3, activation = tf.nn.relu, name = 'conv3', reuse = None)
pool3 = tf.layers.max_pooling2d(conv3,maxpoolframe,maxpoolstrides)
conv4 = tf.layers.conv2d(pool3, filters4, frame4, activation = tf.nn.relu, name = 'conv4', reuse = None)
pool4 = tf.layers.max_pooling2d(conv4,maxpoolframe,maxpoolstrides)

conv1x = tf.layers.conv2d(x1, filters1, frame1, activation = tf.nn.relu, name = 'conv1', reuse = True)
pool1x= tf.layers.max_pooling2d(conv1x,maxpoolframe,maxpoolstrides)
conv2x = tf.layers.conv2d(pool1x, filters2, frame2, activation = tf.nn.relu, name = 'conv2', reuse = True)
pool2x = tf.layers.max_pooling2d(conv2x,maxpoolframe,maxpoolstrides)
conv3x = tf.layers.conv2d(pool2x, filters3, frame3, activation = tf.nn.relu, name = 'conv3', reuse = True)
pool3x = tf.layers.max_pooling2d(conv3x,maxpoolframe,maxpoolstrides)
conv4x = tf.layers.conv2d(pool3x, filters4, frame4, activation = tf.nn.relu, name = 'conv4', reuse = True)
pool4x = tf.layers.max_pooling2d(conv4x,maxpoolframe,maxpoolstrides)

fcon0 = flatten(pool4)
fcon1 = tf.layers.dense(fcon0,fconshape1,tf.nn.relu, name = 'fcon1')

fcon0x = flatten(pool4x)
fcon1x = tf.layers.dense(fcon0x,fconshape1,tf.nn.relu, name = 'fcon1', reuse = True)

"""
Loss function and optimizer algorithm definition
dist = eucledian distance between the vectors of both the images
Y = class mapper for use in loss 
loss = loss function 
algo = optimizer to be used : SGD(a = 0.0000001, p = 10, P = 100)
Note : the loss and learning rate are inter related so optimizing these parameters
is necessary and may explode if not kept correctly
"""
dist = tf.sqrt(tf.reduce_sum(tf.pow(fcon1 - fcon1x,2)))
Y = tf.abs(tf.sign(y-y1))
loss = ((1-Y)*tf.pow(tf.maximum(dist-p,0),2)) + (Y*tf.pow(tf.maximum(P-dist,0),2))
algo = tf.train.GradientDescentOptimizer(a).minimize(loss)

"""
Start session and initialize the global variables and placeholders
"""
sess = tf.Session()
sess.run(tf.global_variables_initializer()) 
saver = tf.train.Saver(save_relative_paths=True)
saver = tf.train.import_meta_graph('training_data.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))


def imRead(path, imsize, ch):
  img = imread(path,1)
  img = cv2.resize(img, (imsize, imsize),0,0, cv2.INTER_LINEAR).reshape((1,imsize,imsize,ch))
  return img

def imPaths(i):
  imagepaths = [classpath[i]+'/'+item for item in os.listdir(classpath[i])]
  return imagepaths

def getVectors(start = 2, kind = 0):
  fullvecs = np.empty((fconshape1+1))
  if (kind == 0):
    for pathi in range(len(classes[:maxclasses])):
      imagepaths = imPaths(pathi)
      curvecs = np.zeros((fconshape1+1,2))
      for i in range(2):
        img1 = imRead(imagepaths[i], image_size, num_channels)
        curvecs[1:,i] = sess.run(fcon1,feed_dict = {x : img1})
        curvecs[0,i] = classes[pathi]
      fullvecs = np.c_[fullvecs, curvecs]
  elif (kind ==1):
    for pathi in range(len(classes[:maxclasses])):
      imagepaths = imPaths(pathi)
      lenimages = len(imagepaths)
      curvecs = np.zeros((fconshape1+1,lenimages - start))
      for i in range(0, lenimages - start):    
        img1 = imRead(imagepaths[i], image_size, num_channels)
        curvecs[1:,i] = sess.run(fcon1,feed_dict = {x : img1})
        curvecs[0,i] = classes[pathi]
      fullvecs = np.c_[fullvecs, curvecs]
  elif (kind ==2):
    imagepaths = [unkroot+'/'+item for item in os.listdir(unkroot)]
    lenimages = len(imagepaths)
    curvecs = np.zeros((fconshape1+1,lenimages))
    for i in range(lenimages):
      img1 = imRead(imagepaths[i], image_size, num_channels)
      curvecs[1:,i] = sess.run(fcon1,feed_dict = {x : img1})
      curvecs[0,i] = -100
    fullvecs = np.c_[fullvecs, curvecs]
  else:
    print("Unknown kind of vectors")
  return fullvecs[:,1:].T

def getVector(path, label):
  img1 = imRead(path, image_size, num_channels)
  curvecs = np.zeros((fconshape1+1,1))
  curvecs[1:,0] = sess.run(fcon1,feed_dict = {x : img1})
  curvecs[0,0] = label
  return curvecs.T
  

"""
#Load saved model, run only when required
"""
#saver = tf.train.import_meta_graph('training_data.meta')
#saver.restore(sess,tf.train.latest_checkpoint('./'))



"""
get vectors for trained network
vecs = training data
vecs1 = validation data
fullvecst = unknown data, should get unmathed for thse 
"""
vecs = getVectors()
vecs1 = getVectors(kind = 1)
vecst = getVectors(kind = 2)

"""
Linear SVC model for training
"""
reg = LinearSVC(loss = 'hinge', tol = 1e-8)
reg.fit(vecs[:,1:],vecs[:,0])

"""
sample vector and prediction
"""
Eucdist = np.zeros((vecst.shape[0]+vecs1.shape[0],vecs.shape[0]))
for i in range(vecs.shape[0]):
  for j in range(vecst.shape[0]):
    Eucdist[j,i] = np.linalg.norm(vecs[i,1:]-vecst[j,1:])
for i in range(vecs.shape[0]):
  for j in range(vecs1.shape[0]):
    Eucdist[j+vecst.shape[0],i] = np.linalg.norm(vecs[i,1:]-vecs1[j,1:])
Euarg = np.argmin(Eucdist,axis =1)

Eu = np.c_[np.min(Eucdist,axis =1),np.argmin(Eucdist,axis =1)]

#vec = getVector(unkroot+'/18.jpg', classes[6])
#conf = reg.decision_function(vec[:,1:])
#conf = np.sign(np.sign(conf-0.1)+1)
#if(conf.sum()!=1):
#  pred = 'unk'
#else:
#  pred = reg.predict(vec[:,1:])
"""
acc = accuracy of predicting validation data without confidence analysis
confi = confidence values for validaton data
conft = confidence values for unknown images
confi2 = wrappend confidence values for validation data
confit2 = wrapped confidence values for unknown images
"""
acc = reg.score(vecs1[:,1:], vecs1[:,0])
#prob = reg.predict_proba(vecs1[:,1:])
#probt = reg.predict_proba(vecst[:,1:])
#
#confi = reg.decision_function(vecs1[:,1:])
#confit = reg.decision_function(vecst[:,1:])
#confi2 = np.sign((np.sign(confi - 0.1)+1))
#confit2 = np.sign((np.sign(confit - 0.1)+1))
#
#"""
#unknownacc = accuracy for unknown image prediction
#"""
#sumerct = confit2.sum(axis = 1)
#sumerct[sumerct>1] = 0
#err = sumerct.sum()
#unknownacc = 1- (err/len(sumerct))
#
#"""
#acc = accuracy for validation image prediction
#args = prediction class labels
#"""
#args = np.array([int(classes[x]) for x in np.argmax(confi2,axis = 1)])
#args1 = np.abs(np.sign(args - vecs1[:,0]))
#err = args1.sum()
#acc = 1 - (err/len(args1))
