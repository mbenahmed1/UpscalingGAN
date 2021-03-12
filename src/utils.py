"""Contains utility functions.

This file contains utility functions such as file I/O
and data augmentation.
"""

# imports
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import config
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
    image_paths = list(dir_path.glob('*.' + config.IMAGEFILEEXTENSION))
    number_of_files = len(image_paths)
    print(str(number_of_files) + " images found.")

    # creating a list of all images as numpy arrays
    array_list = []
    minimum_width = config.MAXIMAGEWIDTH
    minumum_heigth = config.MAXIMAGEHEIGHT

    # time estimation prep
    posted_time_estimation = False
    start = time.time()

    # shrinking path list if necessary (ran out of memory lol)
    number_of_files *= config.DATASETSCALINGFACTOR
    number_of_files = int(number_of_files)

    # iterating over (a part of) the path list
    print("Started loading images.")
    print("The dataset scaling factor is \'" +
          str(config.DATASETSCALINGFACTOR) + "\'.")
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
        if(i > config.TIMEESTIMATIONCOUNTER and not posted_time_estimation):
            posted_time_estimation = True
            end = time.time()
            print("Estimated time remaining: " + str(round((end - start) *
                                                           number_of_files / config.TIMEESTIMATIONCOUNTER - (end - start), 1)) + " s.")

    print("Finished loading images.")
    print("Found minimum width: " + str(minimum_width) + ".")
    print("Found minimum height: " + str(minumum_heigth) + ".")
    # converting array list to array of arrays
    return np.asarray(array_list, dtype=object)


def path_to_tensor(path: str) -> tf.Tensor:
    """Loads paths from directory path and creates tensor

    Parameters:
        path:       the path to the directory of the images
    """
    # creating file list
    dir_path = pathlib.Path(path)
    image_paths = list(dir_path.glob('*.' + config.IMAGEFILEEXTENSION))
    number_of_files = len(image_paths)
    print(str(number_of_files) + " images found.")

    # converting posixpaths to strings
    path_strings = []
    for path in image_paths:
        path_strings.append(path.as_posix())

    # converting list of strings to tensor
    image_paths = tf.convert_to_tensor(path_strings, dtype=tf.string)
    return image_paths


def prepare_images(path: str) -> tf.Tensor:
    """Reads paths, load them into tensors and does image preprocessing,
    such as converting dtype and resizing and cropping

    Parameters:
        path:       the path to the directory of the images
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if image.shape[2] == 3:
        image = tf.image.rgb_to_grayscale(image)
    

    # crop and pad if image does not fit in target size
    full_image = tf.image.resize_with_crop_or_pad(
        image, config.FULLIMAGESIZE, config.FULLIMAGESIZE)
    # full_image = tf.image.per_image_standardization(full_image)

    return full_image


def make_full_low_pairs(full_image: tf.Tensor) -> tf.Tensor:
    """Makes pairs of full and low resolution images. 
    Therefore, it resizes the full size image.

    Parameters:
        full_image:     the full size image
    """
    low_image = tf.image.resize(full_image, [config.LOWIMAGESIZE, config.LOWIMAGESIZE])
    return low_image, full_image

def visualize(original: tf.Tensor, augmented: tf.Tensor):
    """Visualizes original image vs augmented.

    Paramers:
        original:   original image
        augmented:  augmented image
    """
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)


def flip_left_right(image: tf.Tensor) -> tf.Tensor:
    """Takes an image and flips it left-right

    Parameters:
        image:  the image to be modified
    """
    if tf.random.uniform([1], 0, 1) < config.AUGMENTATIONPROBABILITY:
        return tf.image.flip_left_right(image)
    else:
        return image
    


def flip_up_down(image: tf.Tensor) -> tf.Tensor:
    """Takes an image and flips it up-down

    Parameters:
        image:  the image to be modified
    """
    if tf.random.uniform([1], 0, 1) < config.AUGMENTATIONPROBABILITY:
        return tf.image.flip_up_down(image)
    else:
        return image
    


def saturate(image: tf.Tensor, lower: float, upper: float, seed: tf.Tensor) -> tf.Tensor:
    """Takes an image and saturates it by some random factor

    Parameters:
        image:  the image to be modified
        lower:  lower bound for saturation adjustment
        upper:  upper bound for saturation adjustment
        seed:   the random seed
    """
    if tf.random.uniform([1], 0, 1) < config.AUGMENTATIONPROBABILITY:
        return tf.image.stateless_random_saturation(image, lower, upper, seed)
    else:
        return image
    


def brighten(image: tf.Tensor, max_delta: float, seed: tf.Tensor) -> tf.Tensor:
    """Takes an image and adjusts its brightness by some random factor

    Parameters:
        image:  the image to be modified
        max_delta: the maximum brightness delta
        seed:   the random seed
    """
    if tf.random.uniform([1], 0, 1) < config.AUGMENTATIONPROBABILITY:
        return tf.image.stateless_random_brightness(image, max_delta, seed)
    else:
        return image
    

def contrast(image: tf.Tensor, lower: float, upper: float, seed: tf.Tensor) -> tf.Tensor:
    """Takes an image and adjusts its contrast by some random factor

    Parameters:
        image:  the image to be modified
        lower:  the lower bound for contrast adjustment
        upper:  the upper bound for constrast adjustment
        seed:   the random seed
    """
    if tf.random.uniform([1], 0, 1) < config.AUGMENTATIONPROBABILITY:
        return tf.image.stateless_random_contrast(image, lower, upper, seed)
    else:
        return image

def write_log(path: str, start_string: str):
    """Write log file with all important paramters.
    
    Parameters:
        path:           the path to the folder
        start_string:   the string containing the starting time
    """
    # writing training paramters to file
    text_file = open(f'{path}/{config.OUTPUTFILENAME}', "w+")
    text_file.write(f'Run:              {start_string}\n')
    text_file.write("\n")
    text_file.write(f'Full:             {config.FULLIMAGESIZE}\n')
    text_file.write(f'Low:              {config.LOWIMAGESIZE}\n')
    text_file.write(f'Channels:         {config.NUMCHANNELS}\n')
    text_file.write(f'ParallelCalls:    {config.NUMPARALLELCALLS}\n')
    text_file.write(f'Splitsize:        {config.TESTSPLITSIZE}\n')
    text_file.write(f'Batchsize:        {config.BATCHSIZE}\n')
    text_file.write(f'Buffersize:       {config.BUFFERSIZE}\n')
    text_file.write(f'Prefetchsize:     {config.PREFETCHSIZE}\n')
    text_file.write(f'Epochs:           {config.EPOCHS}\n')
    text_file.write(f'ResBlocks:        {config.NUMRESBLOCKS}\n')
    text_file.close()