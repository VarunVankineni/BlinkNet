import tensorflow as tf
import numpy as np
import os,glob,cv2
#import sys,argparse
#import sys

#first_arg = sys.argv[1]
# First, pass the path of the image
#dir_path = os.path.dirname(os.path.realpath(__file__))
#image_path=sys.argv[1]

#image_path="IMG-20180220-WA0006_20180220_122543417.jpg" 
predictlist = np.zeros((58,2))
for i in range(14,57):
  filename ='E:/Acads/9th sem/SVM/unknownimages/'+str(i)+'.jpg'
  #filename ='E:/Acads/9th sem/SVM/New folder/Aaron_Eckhart/Aaron_Eckhart_0001.png'
  print(filename)
  image_size=128
  num_channels=3
  images = []
  # Reading the image using OpenCV
  image = cv2.imread(filename)
  # Resizing the image to our desired size and preprocessing will be done exactly as done during training
  image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
  images.append(image)
  images = np.array(images, dtype=np.uint8)
  images = images.astype('float32')
  images = np.multiply(images, 1.0/255.0) 
  #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
  x_batch = images.reshape(1, image_size,image_size,num_channels)
  ## Let us restore the saved model 
  sess = tf.Session()
  # Step-1: Recreate the network graph. At this step only graph is created.
  saver = tf.train.import_meta_graph('E:/Acads/9th sem/SVM/training_data/savedmodel.meta')
  # Step-2: Now let's load the weights saved using the restore method.
  saver.restore(sess, tf.train.latest_checkpoint('E:/Acads/9th sem/SVM/training_data/'))
  # Accessing the default graph which we have restored
  graph = tf.get_default_graph()
  # Now, let's get hold of the op that we can be processed to get the output.
  # In the original network y_pred is the tensor that is the prediction of the network
  y_pred = graph.get_tensor_by_name("y_pred:0")
  print(y_pred)
  ## Let's feed the images to the input placeholders
  x= graph.get_tensor_by_name("x:0") 
  y_true = graph.get_tensor_by_name("y_true:0") 
  #y_test_images = np.array([[0,1]])
  root='E:/Acads/9th sem/SVM/training_data/' 
  classes = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
  z=len(classes)
  c = list(range(z))
  y_test_images=np.asarray([c])
  print(y_test_images)
  #y_test_images=np.array([[y_test_images]])
  classes= [int(x) for x in classes]
  classes.sort()
  #y_test_images = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]])
  ### Creating the feed_dict that is required to be fed to calculate y_pred 
  feed_dict_testing = {x: x_batch, y_true: y_test_images}
  result=sess.run(y_pred, feed_dict=feed_dict_testing)
  print(result[0])
  #[,'151','160','161','162','163','164','165','166','167','168','169','170',25,26,27,28,29,30,31,32,33,34,35]
  #classes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
  #classes=['0','1']
  print(classes)
  a=np.max(result[0])
  z=[i for i,x in enumerate(result[0]) if x == a]
  print("the class of max value is",classes[z[0]])
  predictlist[i,0] = classes[z[0]]
  #if (a<0.3):
   #   print("unknown faces found")
  print("max values",a)
  predictlist[i,1] = a
  
  