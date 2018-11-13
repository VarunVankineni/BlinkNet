# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:38:43 2018

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
"""

"""
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
filters1 = 32           #number of filters in 1st convolution
frame1 = [10,10]    #frame of the convolution filter1
filters2 = 64    #number of filters in 2nd convolution
frame2 = [7,7]     #frame of the convolution filter2
filters3 = 128    #number of filters in 3rd convolution
frame3 = [5,5]     #frame of the convolution filter3
filters4 = 128    #number of filters in 4th convolution
frame4 = [3,3]     #frame of the convolution filter4
maxpoolframe = [2,2]   #maxpooling frame size
maxpoolstrides = [2,2] #stride of maxpooling frame
fconshape1 = 384       #neurons in 1st fully connected layer

"""
Optimizer parameters
"""
a = 0.0000001                #learning rate 
p = 10                #length between same class images
P = 100               #length between different class images
maxitr = 100          #maximum number of epochs

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


for itr in range(maxitr):
  
  """
  pathi = iterable for going over the classes (like i in a general for loop)
  imagename = paths for the images in the classpath[pathi] which is the path for classes[pathi]
  lenimages = no.of images to train on, atleast 2 images are required for siamese network
  """
  for pathi in range(len(classes[:10])):
    imagename = [item for item in os.listdir(classpath[pathi])]
    imagepaths = [classpath[pathi]+'/'+x for x in imagename]   
    lenimages = max(2,len(imagepaths)//training_size)
    
    """
    i is the iterable over the images in imagepaths - > gives the image at path imagepaths[i]
    j is the second iterable -> gives the image at path imagepaths[j]
    img1 = image 1
    img2 = image 2
    """
    for i in range(lenimages):
      for j in range(i+1,lenimages):
        img1 = imread(imagepaths[i],1)
        img1 = cv2.resize(img1, (image_size, image_size),0,0, cv2.INTER_LINEAR).reshape((1,128,128,3))
        img2 = imread(imagepaths[j],1)
        img2 = cv2.resize(img2, (image_size, image_size),0,0, cv2.INTER_LINEAR).reshape((1,128,128,3))
        """
        hard constraint for not trying to train same images
        """
        if(np.array_equal(img1,img2)):
          continue     
        """
        train for same classes, currently input same labels different, in future use class labels directly
        """
        sess.run(algo, feed_dict = {x : img1, x1 : img2, y : np.ones((1,1)), y1 : np.ones((1,1))})
#    print(classes[pathi],classes[pathi])
    for pathj in range(pathi+1, len(classes[:10])):
      imagenamej = [item for item in os.listdir(classpath[pathj])]
      imagepathsj = [classpath[pathj]+'/'+x for x in imagenamej]   
      lenimagesj = max(2,len(imagepaths)//training_size)
      for i in range(lenimages):
        for j in range(lenimagesj):
          img1 = imread(imagepaths[i],1)
          img1 = cv2.resize(img1, (image_size, image_size),0,0, cv2.INTER_LINEAR).reshape((1,128,128,3))
          img2 = imread(imagepathsj[j],1)
          img2 = cv2.resize(img2, (image_size, image_size),0,0, cv2.INTER_LINEAR).reshape((1,128,128,3))
#          print(sess.run(dist, feed_dict = {x : img1, x1 : img2}))
          if(np.array_equal(img1,img2)):
            continue
          sess.run(algo, feed_dict = {x : img1, x1 : img2, y : np.ones((1,1)), y1 : np.zeros((1,1))})
#      print(classes[pathi],classes[pathj])
  print('epoch '+str(itr)+' done')
  
imagename = [item for item in os.listdir(classpath[0])]
imagepaths = [classpath[0]+'/'+x for x in imagename] 



fullvecs = np.empty((fconshape1+1))
for pathi in range(len(classes[:10])):
    imagename = [item for item in os.listdir(classpath[pathi])]
    imagepaths = [classpath[pathi]+'/'+x for x in imagename]   
    lenimages = len(imagepaths)//5
    curvecs = np.zeros((fconshape1+1,lenimages))
    for i in range(lenimages):
      img1 = imread(imagepaths[i],1)
      img1 = cv2.resize(img1, (image_size, image_size),0,0, cv2.INTER_LINEAR).reshape((1,128,128,3))
      curvecs[1:,i] = sess.run(fcon1,feed_dict = {x : img1})
      curvecs[0,i] = classes[pathi]
    fullvecs = np.c_[fullvecs, curvecs]
fullvecs = fullvecs[:,1:].T

fullvecs1 = np.empty((fconshape1+1))
for pathi in range(len(classes[:10])):
    imagename = [item for item in os.listdir(classpath[pathi])]
    imagepaths = [classpath[pathi]+'/'+x for x in imagename]   
    lenimages = len(imagepaths)
    len2 = len(imagepaths)//5
    curvecs = np.zeros((fconshape1+1,lenimages-len2))
    for i in range(len2,lenimages):
      i = i-len2      
      img1 = imread(imagepaths[i],1)
      img1 = cv2.resize(img1, (image_size, image_size),0,0, cv2.INTER_LINEAR).reshape((1,128,128,3))
      curvecs[1:,i] = sess.run(fcon1,feed_dict = {x : img1})
      curvecs[0,i] = classes[pathi]
    fullvecs1 = np.c_[fullvecs1, curvecs]
fullvecs1 = fullvecs1[:,1:].T
  
fullvecst = np.empty((fconshape1+1))

imagename = [item for item in os.listdir(unkroot)]
imagepaths = [unkroot+'/'+x for x in imagename]   
lenimages = len(imagepaths)
curvecs = np.zeros((fconshape1+1,lenimages))
for i in range(lenimages):
  img1 = imread(imagepaths[i],1)
  img1 = cv2.resize(img1, (image_size, image_size),0,0, cv2.INTER_LINEAR).reshape((1,128,128,3))
  curvecs[1:,i] = sess.run(fcon1,feed_dict = {x : img1})
  curvecs[0,i] = 1000
fullvecst = np.c_[fullvecst, curvecs]

fullvecst = fullvecst[:,1:].T




reg = LinearSVC()
#reg = SVC(kernel = 'linear' , probability = True)
reg.fit(fullvecs[:,1:],fullvecs[:,0])

ypred = reg.predict_proba(fullvecst[:,1:])
ypred1 = reg.predict(fullvecs1[:,1:])
ypred1p = reg.predict_proba(fullvecs1[:,1:])
acc = reg.score(fullvecs1[:,1:], fullvecs1[:,0])
confi = reg.decision_function(fullvecs1[:,1:])
confit = reg.decision_function(fullvecst[:,1:])

confit2 = np.sign(confit - 0.1)
confi2 = np.sign(confi - 0.1)


for k in range(10):
  imagename = [item for item in os.listdir(classpath[k])]
  imagepaths = [classpath[k]+'/'+x for x in imagename] 
  for i in imagepathsj[:2]:
    for j in imagepaths[2:]:
      img1 = imread(i,1)
      img1 = cv2.resize(img1, (image_size, image_size),0,0, cv2.INTER_LINEAR).reshape((1,128,128,3))
      img2 = imread(j,1)
      img2 = cv2.resize(img2, (image_size, image_size),0,0, cv2.INTER_LINEAR).reshape((1,128,128,3))
      print(sess.run(dist, feed_dict = {x : img1, x1 : img2}))
        
#  print(sess.run(dist,feed_dict = {x : img1, x1 : img2} ))
#  print(sess.run(dist,feed_dict = {x : img1, x1 : imgt} ))

#img11 = imread(data_path+'1003.png',1).reshape((1,160,160,3))
#imgt1 = imread(data_path+'1033.png',1).reshape((1,160,160,3))
#print(sess.run(dist,feed_dict = {x : img1, x1 : img11} ))
#print(sess.run(dist,feed_dict = {x : img1, x1 : imgt1} ))
#



