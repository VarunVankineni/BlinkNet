# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:31:51 2018

@author: Varun
"""
from cv2 import imread,imshow,imwrite
import numpy as np 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten

data_path = 'E:/Acads/9th sem/SVM/siamese dataset/'
tf.reset_default_graph()

img1 = imread(data_path+'1001.png',1).reshape((1,160,160,3))
img2 = imread(data_path+'1002.png',1).reshape((1,160,160,3))
imgt = imread(data_path+'1031.png',1).reshape((1,160,160,3))


x = tf.placeholder(tf.float32, (None, 160, 160, 3))
x1 = tf.placeholder(tf.float32, (None, 160, 160, 3))
y = tf.placeholder(tf.float32, (None))
y1 = tf.placeholder(tf.float32, (None))
#onehoty = tf.one_hot(y, 10)


"""
%%PARAMETERS
"""
filters1 = 16           #number of filters in 1st convolution
frame1 = [5,5]    #frame of the convolution filter1
filters2 = 32    #number of filters in 2nd convolution
frame2 = [3,3]     #frame of the convolution filter2
filters3 = 64    #number of filters in 3rd convolution
frame3 = [3,3]     #frame of the convolution filter3
filters4 = 128    #number of filters in 4th convolution
frame4 = [3,3]     #frame of the convolution filter4
maxpoolframe = [2,2]   #maxpooling frame size
maxpoolstrides = [2,2] #stride of maxpooling frame
fconshape1 = 128       #neurons in 1st fully connected layer
a = 0.000001                #learning rate 
maxitr = 3             #total number of iterations over the data set 
batchsize = 1        #batchsize for each gradient descent run
p = 10
P = 100

"""
Network Architecture 
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
"""

dist = tf.sqrt(tf.reduce_sum(tf.pow(fcon1 - fcon1x,2)))
Y = tf.abs(tf.sign(y-y1))
loss = ((1-Y)*tf.pow(dist-p,2)) + (Y*tf.pow(tf.maximum(P-dist,0),2))
algo = tf.train.GradientDescentOptimizer(a).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) 
batch_size = 1
for i in range(100):
  sess.run(algo, feed_dict = {x : img1, x1 : img2, y : np.ones((1,1)), y1 : np.ones((1,1))})
  sess.run(algo, feed_dict = {x : img1, x1 : imgt, y : np.ones((1,1)), y1 : np.zeros((1,1))})
  print(sess.run(dist,feed_dict = {x : img1, x1 : img2} ))
  print(sess.run(dist,feed_dict = {x : img1, x1 : imgt} ))

img11 = imread(data_path+'1003.png',1).reshape((1,160,160,3))
imgt1 = imread(data_path+'1033.png',1).reshape((1,160,160,3))
print(sess.run(dist,feed_dict = {x : img1, x1 : img11} ))
print(sess.run(dist,feed_dict = {x : img1, x1 : imgt1} ))




