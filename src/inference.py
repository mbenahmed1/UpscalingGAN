"""inference.py

This file contains a routine to load the model and then apply it to a given image.
"""

# imports
import sys
import tensorflow as tf
import config
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


def print_usage():
    """ Prints the usage of the program to console.
    """
    print("usage: $ python3 inference.py [image.jpg]")

# @tf.function


def upscale_tiles(path: str):
    M = config.LOWIMAGESIZE
    N = config.LOWIMAGESIZE
    # loading image from passed argument
    image = tf.io.read_file(sys.argv[1])
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    im = np.asarray(Image.open(path))

    # creating tiles from full size image
    tiles = [im[x:x+M, y:y+N]
             for x in range(0, im.shape[0], M) for y in range(0, im.shape[1], N)]

    counter = 0
    for tile in tiles:
        # convert tiles to tensor
        image_tile = np.array(tile)
        image_tile = tf.image.resize_with_crop_or_pad(
            image_tile, config.LOWIMAGESIZE, config.LOWIMAGESIZE)
        image_tile = tf.convert_to_tensor(tile, dtype=tf.float32)
        image_tile = tf.image.per_image_standardization(image_tile)
        image_tile = tf.expand_dims(image_tile, axis=0)

        # perform upscaling
        upscaled_image = generator(image_tile, training=False)
        upscaled_image = (upscaled_image + 1) / 2
        upscaled_image = upscaled_image.numpy()

        # normalizing values to [0,...,255]
        plt.imshow((upscaled_image[0, :, :, :] * 255).astype(np.uint8))
        plt.savefig(f'./{counter}.png')

        print(f'processed tile {counter + 1} of {len(tiles)}')
        counter += 1


def upscale_patches(img: np.ndarray):
    """ Takes an image as np array and returns a list of upscaled np array patches.
    """
    sizeX = img.shape[1]
    sizeY = img.shape[0]

    print(f'x: {sizeX}')
    print(f'y: {sizeY}')

    nRows = int(sizeY / config.LOWIMAGESIZE)
    mCols = int(sizeX / config.LOWIMAGESIZE)

    print(nRows)
    print(mCols)

    bigger_image = np.zeros(shape=(nRows * 4, mCols * 4))

    for i in range(0, nRows):
        for j in range(0, mCols):
            patch = img[i*int(sizeY/nRows):i*int(sizeY/nRows) + int(
                sizeY/nRows), j*int(sizeX/mCols):j*int(sizeX/mCols) + int(sizeX/mCols)]
            patch = tf.convert_to_tensor(patch, dtype=tf.float32)
            patch = tf.image.resize_with_crop_or_pad(
                patch, config.LOWIMAGESIZE, config.LOWIMAGESIZE)
            patch = tf.image.per_image_standardization(patch)
            patch = tf.expand_dims(patch, axis=0)

            upscaled_patch = generator(patch, training=False)
            upscaled_patch = (upscaled_patch + 1) / 2
            upscaled_patch = upscaled_patch.numpy()

            upscaled_patch = np.reshape(upscaled_patch, (64, 64))
            bigger_image[4 * i*int(sizeY/nRows):4 * i*int(sizeY/nRows) + int(
                sizeY/nRows), 4 * j*int(sizeX/mCols):4 * j*int(sizeX/mCols) + int(sizeX/mCols)] = upscaled_patch


# loading generator from data folder (unpack first)
generator = tf.keras.models.load_model(config.MODELPATH)
generator.compile(optimizer="adam", loss="mean_squared_error")


if len(sys.argv) > 1:
    img = cv2.imread(sys.argv[1])
    # upscale_patches(img)
    upscale_tiles(sys.argv[1])

else:
    print_usage()
    exit
