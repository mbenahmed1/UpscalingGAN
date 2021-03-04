"""Contains utility functions.

This file contains utility functions such as file I/O
"""

# imports
import tensorflow as tf
import pathlib
import constants
import PIL
import PIL.Image
import numpy as np
import imageio
import time


def load_images(path: str):
    """Loads images from directory path and creates a numpy array
    containing numpy arrays of the images.

    Parameters:
        path:       the path to the directory of the images
    """
    # creating file list
    dir_path = pathlib.Path(path)
    image_paths = list(dir_path.glob('*.' + constants.IMAGEFILEEXTENSION))
    number_of_files = len(image_paths)
    print(str(number_of_files) + " images found.")

    # creating a list of all images as numpy arrays
    array_list = []
    minimum_width = constants.MAXIMAGEWIDTH
    minumum_heigth = constants.MAXIMAGEHEIGHT

    # time estimation prep
    posted_time_estimation = False
    start = time.time()

    # shrinking path list if necessary (ran out of memory lol)
    number_of_files *= constants.DATASETSCALINGFACTOR
    number_of_files = int(number_of_files)

    # iterating over (a part of) the path list
    print("Started loading images.")
    print("The dataset scaling factor is \'" +
          str(constants.DATASETSCALINGFACTOR) + "\'.")
    for i, path in enumerate(image_paths[0:number_of_files]):
        # convert image to nparray
        im = imageio.imread(path)
        array_list.append(im)
        # find min shapes
        if im.shape[0] < minimum_width:
            minimum_width = im.shape[0]
        if im.shape[1] < minumum_heigth:
            minumum_heigth = im.shape[1]
        # post time estimation
        if(i > constants.TIMEESTIMATIONCOUNTER and not posted_time_estimation):
            posted_time_estimation = True
            end = time.time()
            print("Estimated time remaining: " + str(round((end - start) *
                                                           number_of_files / constants.TIMEESTIMATIONCOUNTER - (end - start), 1)) + " s.")

    print("Finished loading images.")
    print("Found minimum width: " + str(minimum_width) + ".")
    print("Found minimum height: " + str(minumum_heigth) + ".")
    # converting array list to array of arrays
    return np.asarray(array_list, dtype=object)


def path_to_tensor(path):
    """Loads paths from directory path and creates tensor

    Parameters:
        path:       the path to the directory of the images
    """
    # creating file list
    dir_path = pathlib.Path(path)
    image_paths = list(dir_path.glob('*.' + constants.IMAGEFILEEXTENSION))
    number_of_files = len(image_paths)
    print(str(number_of_files) + " images found.")

    # converting posixpaths to strings
    path_strings = []
    for path in image_paths:
        path_strings.append(path.as_posix())

    # converting list of strings to tensor
    image_paths = tf.convert_to_tensor(path_strings, dtype=tf.string)
    return image_paths


def prepare_images(path: str):
    """Reads paths, load them into tensors and does image preprocessing
    
    Parameters:
        path:       the path to the directory of the images
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # crop and pad if image does not fit in target size
    full_image = tf.image.resize_with_crop_or_pad(
        image, constants.FULLIMAGESIZE, constants.FULLIMAGESIZE)
    full_image = tf.image.per_image_standardization(full_image)

    # creating smaller low resolution image from full resolution image
    low_image = tf.image.resize(
        full_image, [constants.LOWIMAGESIZE, constants.LOWIMAGESIZE])
    low_image = tf.image.per_image_standardization(low_image)

    return low_image, full_image
