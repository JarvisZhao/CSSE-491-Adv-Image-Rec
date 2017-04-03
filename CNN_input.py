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


# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000

def ImageReader(directory):
	"""Args: the directory of your images
	#Returns: 1-d list of Image Object
			  1-d np-array of labels
	"""	  	
	imageList = os.listdir(directory)
	result = list()
	for k in range(len(imageList)):
		img = Image.open(directory + '/' +imageList[k])
		img = img.resize([64,64])
		result.append(img)
	labels = np.zeros(len(imageList))
	if 'Sunset' in directory:
		labels[:] = 0;
	if 'Nonsunsets' in directory:
		labels[:] = 1;

	return result ,labels


def prePoccess(input):
	"""
	Args:
		input- a list of 3-d image object from 
	"""
	images = list()
	for k in range(len(input)):
		images.append(np.array(input[k]))#.flatten()

	images = np.array(images,dtype = 'float32')
	return images


def inputs():
	testPosFeatures,testPosLabels = ImageReader("Images/TestSunset")
	testNegFeatures,testNegLabels = ImageReader("Images/TestNonsunsets")
	trainPosFeatures,trainPosLabels = ImageReader("Images/TrainSunset")
	trainNegFeatures,trainNegLabels = ImageReader("Images/TrainNonsunsets")

	trainPosFeatures=prePoccess(trainPosFeatures)
	trainNegFeatures = prePoccess(trainNegFeatures)
	testPosFeatures = prePoccess(testPosFeatures)
	testNegFeatures = prePoccess(testNegFeatures)

	trainLabels = np.concatenate((trainPosLabels,trainNegLabels))
	testLabels=np.concatenate((testPosLabels,testNegLabels))
	trainFeatures = np.concatenate((trainPosFeatures,trainNegFeatures))
	testFeatures=np.concatenate((testPosFeatures,testNegFeatures))

	return trainFeatures,testFeatures,trainLabels,testLabels



def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  print("image_size3: ",image.shape)
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size
        # min_after_dequeue=min_queue_examples)
        )
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  print("iamges_shape2: ", images.shape)
  print("label_batch: ", label_batch)
  print("labels_shape2: ", tf.reshape(label_batch, [batch_size]))
  return images, tf.reshape(label_batch, [batch_size])