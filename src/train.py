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
seed = rng.make_seeds(2)[0]

# loading paths
list_ds = tf.data.Dataset.list_files(constants.DATAPATH)

# loading and preparing images
ds = list_ds.map(utils.prepare_images)


# applying some augmentations for testing
ds = ds.map(lambda x: utils.brighten(x, 0.2, seed))
ds = ds.map(lambda x: utils.contrast(x, 0.5, 0.8, seed))
ds = ds.map(lambda x: utils.flip_left_right(x))
ds = ds.map(lambda x: utils.flip_up_down(x))
ds = ds.map(lambda x: utils.saturate(x, 2.1, 3.2, seed))


# plotting some images from the ds
for full_image in ds.take(5):
    plt.imshow(full_image)
    plt.show()
