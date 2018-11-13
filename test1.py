import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

tf.reset_default_graph()

#os.environ['CUDA_VISIBLE_DEVICES'] = ''
root='E:/Acads/9th sem/SVM/training_data' 
classes = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
classes= [int(x) for x in classes]
classes.sort()
classes = [str(i) for i in classes]
print(classes)
#Prepare input data
#classes = ['dogs','cats']
#classes=['160','161','162','163','164','165','166','151','160','161','162','163','164','165','166','167','168','169','170']
#classes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
#classes = []
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.1
img_size = 128
num_channels = 3
train_path='training_data'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(root, img_size, classes, validation_size=validation_size)


print("Complete reading input data")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None,img_size ,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape= [None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
print(y_true_cls)


"""
%%PARAMETERS
"""
filters1 = 8           #number of filters in 1st convolution
filterframe = [3,3]    #frame of the convolution filters
filters2 = 16        #number of filters in 2nd convolution
filterframe2 = [3,3]    #frame of the convolution filters
maxpoolframe = [2,2]   #maxpooling frame size
maxpoolstrides = [2,2] #stride of maxpooling frame
fconshape1 = 512       #neurons in 1st fully connected layer
fconshape2 = 128     #neurons in 2nd fully connected layer 
classes = num_classes           #final neurons for logits  
a = 0.001             #learning rate 
maxitr = 3             #total number of iterations over the data set 
batch_size = 4        #batchsize for each gradient descent run


"""
Network Architecture 
"""
conv0 = tf.layers.conv2d(x, filters1, filterframe, activation = tf.nn.relu)
#pool1 = tf.layers.max_pooling2d(conv0,maxpoolframe,maxpoolstrides)
conv1 = tf.layers.conv2d(conv0, filters1, filterframe, activation = tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv1,maxpoolframe,maxpoolstrides)
conv2 = tf.layers.conv2d(pool2, filters2, filterframe2, activation = tf.nn.relu)
pool3 = tf.layers.max_pooling2d(conv2,maxpoolframe,maxpoolstrides)
fcon0 = flatten(pool3)
fcon1 = tf.layers.dense(fcon0,fconshape1,tf.nn.relu)
fcon2 = tf.layers.dense(fcon1,fconshape2,tf.nn.relu)
logits = tf.layers.dense(fcon2, classes)

"""
Loss function and optimizer algorithm definition
"""
y_pred = tf.nn.softmax(logits,name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)
entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_true)
loss = tf.reduce_mean(entropy)
algo = tf.train.AdamOptimizer(a).minimize(loss)
#algo = tf.train.AdamOptimizer(a).minimize(loss)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
"""
Intitiate session
"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())

"""
Batch gradient descent
"""
#for i in range(maxitr):
#    for start in range(0,X_train.shape[0],batchsize):
#        end = start+batchsize
#        sess.run(algo,feed_dict = {x:X_train[start:end],y:y_train[start:end]})
#
#    print(i,calc_accuracy(X_train,y_train,batchsize))


def show_progress(epoch, feed_dict_train, feed_dict_validate,tr_loss, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

saver = tf.train.Saver()
total_iterations = 10
def train(num_iteration):
    session.run(tf.global_variables_initializer()) 
    global total_iterations
    for i in range(total_iterations,
                   total_iterations + num_iteration):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(algo, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0:
            tr_loss = session.run(loss, feed_dict=feed_dict_tr)
            val_loss = session.run(loss, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, tr_loss, val_loss)
            saver.save(session, 'E:/Acads/9th sem/SVM/training_data/savedmodel') 


    total_iterations += num_iteration
train(num_iteration=1000)