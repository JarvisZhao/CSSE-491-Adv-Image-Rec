from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib as plot
import os
import re
import sys
import tarfile
from PIL import Image
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO);

def input(directory):
	"""Args: the directory of your images
	#Returns: 1-d list of Image Object
			  1-d np-array of labels
	"""	
	num = 0;  	
	labelList = os.listdir(directory)

	result = list()
	labels = list()
	for k in range(len(labelList)):
		print(labelList[k])
		imageList = os.listdir(directory+'/'+labelList[k])
		
		num = num + len(imageList)
		for i in range(len(imageList)):
			img = Image.open(directory+'/'+labelList[k]+'/'+imageList[i])
			img = img.resize([56,56])
			labels.append(k)
			result.append(img)
	
	return result ,labels

def prePoccess(input,labels):
	"""
	Args:
		input- a list of 3-d image object from 
	"""
	images = list()
	for k in range(len(input)):
		images.append(np.array(input[k]).flatten())

	images = np.array(images,dtype = 'float32')*1.0/255;
	label = np.array(labels)
	return images, label

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 56, 56, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)

  print("con1_shape: ", conv1.shape)
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  conv3 = tf.layers.conv2d(
  	inputs = pool2,
  	filters = 64,
  	kernel_size = [3,3],
  	padding="same",
  	activation = tf.nn.relu)

  pool3 = tf.layers.max_pooling2d(inputs = conv3,pool_size=[2,2],strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense1 = tf.layers.dense(inputs=pool3_flat, units=100, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.1, training=mode == learn.ModeKeys.TRAIN)

  dense2 = tf.layers.dense(inputs = dropout1,units = 100,activation = tf.nn.relu)

  dropout2 = tf.layers.dropout(
  	inputs=dense2, rate = 0.1,training = mode == learn.ModeKeys.TRAIN)

  dense3 = tf.layers.dense(inputs = dropout2,units = 100,activation = tf.nn.relu)

  dropout3 = tf.layers.dropout(
  	inputs=dense3, rate = 0.1,training = mode == learn.ModeKeys.TRAIN)

  # dense4 = tf.layers.dense(inputs=dropout3, units=100, activation=tf.nn.relu)

  # # Add dropout operation; 0.6 probability that element will be kept
  # dropout4 = tf.layers.dropout(
  #     inputs=dense4, rate=0.1, training=mode == learn.ModeKeys.TRAIN)

  # dense5 = tf.layers.dense(inputs = dropout4,units = 100,activation = tf.nn.relu)

  # dropout5 = tf.layers.dropout(
  # 	inputs=dense5, rate = 0.1,training = mode == learn.ModeKeys.TRAIN)

  # dense6 = tf.layers.dense(inputs = dropout5,units = 100,activation = tf.nn.relu)

  # dropout6 = tf.layers.dropout(
  # 	inputs=dense6, rate = 0.1,training = mode == learn.ModeKeys.TRAIN)

  # dense7 = tf.layers.dense(inputs=dropout6, units=100, activation=tf.nn.relu)

  # # Add dropout operation; 0.6 probability that element will be kept
  # dropout7 = tf.layers.dropout(
  #     inputs=dense7, rate=0.1, training=mode == learn.ModeKeys.TRAIN)

  # dense8 = tf.layers.dense(inputs = dropout7,units = 100,activation = tf.nn.relu)

  # dropout8 = tf.layers.dropout(
  # 	inputs=dense8, rate = 0.1,training = mode == learn.ModeKeys.TRAIN)

  # dense9 = tf.layers.dense(inputs = dropout8,units = 100,activation = tf.nn.relu)

  # dropout9 = tf.layers.dropout(
  # 	inputs=dense9, rate = 0.1,training = mode == learn.ModeKeys.TRAIN)

  # dense10 = tf.layers.dense(inputs = dropout9,units = 100,activation = tf.nn.relu)

  # dropout10 = tf.layers.dropout(
  # 	inputs=dense10, rate = 0.1,training = mode == learn.ModeKeys.TRAIN)




  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 9]
  logits = tf.layers.dense(inputs=dropout3, units=9)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=9)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
  # Load training and eval data
  images, labels = input("Images/Train")
  trainImg, trainLabels = prePoccess(images,labels)
  images, labels = input("Images/Test")
  testImg, testLabels = prePoccess(images,labels)

  # Create the Estimator
  mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/car5_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  mnist_classifier.fit(
      x=trainImg,
      y=trainLabels,
      batch_size=5100,
      steps=30000,
      monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }
  # Evaluate the model and print results
  eval_results = mnist_classifier.evaluate(
      x=testImg, y=testLabels, metrics=metrics)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()


