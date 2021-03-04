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

# loading paths
list_ds = tf.data.Dataset.list_files(constants.DATAPATH)

# loading and preparing images
ds = list_ds.map(utils.prepare_images)

# plotting some high and low res image pairs
for low_image, full_image in ds.take(5):
    plt.imshow(low_image)
    plt.show()
    plt.imshow(full_image)
    plt.show()
