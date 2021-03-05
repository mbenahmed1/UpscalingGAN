"""Contains the training loop and preparation routines.

This file contains the main training loop and all other routines for loading
and preparing train data.
"""

# imports
import tensorflow as tf
import utils
import constants
import os
import matplotlib.pyplot as plt

# disabling gpu if needed
if not constants.USEGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Create a generator
rng = tf.random.Generator.from_seed(123, alg='philox')
seed = rng.make_seeds(2)[1]

# loading paths
list_ds = tf.data.Dataset.list_files(constants.DATAPATH)

# loading and preparing images
ds = list_ds.map(utils.prepare_images, constants.NUMPARALLELCALLS)

# size of the dataset
size = len(list_ds)

# TODO: make random from image to image
# applying some augmentations for testing
ds = ds.map(lambda x: utils.saturate(x, constants.SATURATIONMIN, constants.SATURATIONMAX, seed), constants.NUMPARALLELCALLS)
ds = ds.map(lambda x: utils.flip_left_right(x), constants.NUMPARALLELCALLS)
ds = ds.map(lambda x: utils.flip_up_down(x), constants.NUMPARALLELCALLS)
ds = ds.map(lambda x: utils.brighten(x, constants.BRIGHTNESSMAXDETLA, seed), constants.NUMPARALLELCALLS)
ds = ds.map(lambda x: utils.contrast(x, constants.CONTRASTMIN, constants.CONTRASTMAX, seed), constants.NUMPARALLELCALLS)

# making pairs of the original and the scaled images
ds = ds.map(utils.make_full_low_pairs, constants.NUMPARALLELCALLS)

# splitting dataset into test and train
test_dataset = ds.take(int(constants.TESTSPLITSIZE * size))
train_dataset = ds.skip(int(constants.TESTSPLITSIZE * size))

# batching
test_dataset = test_dataset.batch(constants.BATCHSIZE)
train_dataset = train_dataset.batch(constants.BATCHSIZE)

# shuffling
test_dataset = test_dataset.shuffle(buffer_size=constants.BUFFERSIZE)
train_dataset = train_dataset.shuffle(buffer_size=constants.BUFFERSIZE)

# prefetching
train_dataset = train_dataset.prefetch(constants.PREFETCHSIZE)
test_dataset = test_dataset.prefetch(constants.PREFETCHSIZE)

# plotting some images from the ds
for low_image, full_image in test_dataset.take(5):
    plt.imshow(low_image)
    plt.show()
    plt.imshow(full_image)
    plt.show()