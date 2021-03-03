"""Contains the training loop and preparation routines.

This file contains the main training loop and all other routines for loading
and preparing train data.
"""

# imports
import tensorflow as tf
import utils
import constants
import os

# disabling gpu if needed
if not constants.USEGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load(path_tensor, other):
    image = tf.image.decode_jpeg(tf.io.read_file(path_tensor))
    full_image = tf.image.resize_with_crop_or_pad(
        image, constants.FULLIMAGESIZE, constants.FULLIMAGESIZE)
    low_image = tf.image.resize(
        full_image, [constants.LOWIMAGESIZE,constants.LOWIMAGESIZE], preserve_aspect_ratio=False)
    return low_image, full_image


# array = utils.load_images('../data/unlabeled2017/')

path_tensor = utils.path_to_tensor('../data/unlabeled2017/')
ds = tf.data.Dataset.from_tensor_slices((path_tensor, path_tensor))
ds = ds.map(load, num_parallel_calls=constants.NUMPARALLELCALLS)

