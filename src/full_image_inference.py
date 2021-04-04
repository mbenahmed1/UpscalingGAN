"""inference.py

This file contains a routine to load the model and then apply it to a given image.
"""

# imports
import sys
import tensorflow as tf
import config
import numpy as np
from PIL import Image


def print_usage():
    """ Prints the usage of the program to console.
    """
    print("usage: $ python3 inference.py [image.jpg]")


def upscale_patches(img: np.ndarray):
    """ Takes an image as np array and returns a list of upscaled np array patches.
    """
    sizeX = img.shape[1]
    sizeY = img.shape[0]
    print('')
    print('original image size:')
    print('')
    print(f'    x: {sizeX}')
    print(f'    y: {sizeY}')

    nRows = int(sizeY / config.LOWIMAGESIZE)
    mCols = int(sizeX / config.LOWIMAGESIZE)

    print(f'    number of rows: {nRows}')
    print(f'    number of cols: {mCols}')

    # waring the user if image has to be cropped
    if sizeY % nRows != 0 or sizeX % mCols != 0:
        print('')
        print('Warning: cropped image in x or y')
    print('')

    # cropping image
    img = img[:nRows * config.LOWIMAGESIZE *
              config.IMAGESCALINGFACTOR, :mCols * config.LOWIMAGESIZE * config.IMAGESCALINGFACTOR, :]

    # empty array for bigger image
    bigger_image = np.zeros(shape=(nRows * config.IMAGESCALINGFACTOR * config.LOWIMAGESIZE,
                                   mCols * config.IMAGESCALINGFACTOR * config.LOWIMAGESIZE, config.NUMCHANNELS))

    number_of_patches = nRows * mCols
    counter = 0

    for i in range(0, nRows):
        for j in range(0, mCols):
            # select patch from original image
            patch = img[i*config.LOWIMAGESIZE:i*config.LOWIMAGESIZE + config.LOWIMAGESIZE,
                        j*config.LOWIMAGESIZE:j*config.LOWIMAGESIZE + config.LOWIMAGESIZE]
            patch = tf.convert_to_tensor(patch, dtype=tf.float32)
            patch = tf.image.resize_with_crop_or_pad(
                patch, config.LOWIMAGESIZE, config.LOWIMAGESIZE)
            patch = tf.image.per_image_standardization(patch)
            patch = tf.expand_dims(patch, axis=0)

            # perform upscaling
            upscaled_patch = generator(patch, training=False)
            upscaled_patch = (upscaled_patch + 1) / 2
            upscaled_patch = upscaled_patch.numpy()

            # reassembling patches to bigger image
            bigger_image[config.IMAGESCALINGFACTOR * i * config.LOWIMAGESIZE:config.IMAGESCALINGFACTOR * i * config.LOWIMAGESIZE + config.IMAGESCALINGFACTOR * config.LOWIMAGESIZE,
                         config.IMAGESCALINGFACTOR * j * config.LOWIMAGESIZE:config.IMAGESCALINGFACTOR * j * config.LOWIMAGESIZE + config.IMAGESCALINGFACTOR * config.LOWIMAGESIZE] = upscaled_patch

            print(f'processed patch {counter + 1} of {number_of_patches}')

            counter += 1

    return bigger_image


# loading generator from data folder (unpack first)
generator = tf.keras.models.load_model(config.MODELPATH)
generator.compile(optimizer="adam", loss="mean_squared_error")


if len(sys.argv) > 1:
    # opening image from argument path
    img = Image.open(sys.argv[1])
    # converting to numpy array
    img = np.array(img)
    # scaling up
    upscaled_image = upscale_patches(img)

    # saving upscaled image
    up_im = Image.fromarray((upscaled_image * 255).astype(np.uint8))
    up_im.save('upscaled.jpg')
    print('saved image as "upscaled.jpg"')

else:
    print_usage()
    exit
