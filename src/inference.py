"""inference.py

This file contains a routine to load the model and then apply it to a given image.
"""

# imports
import sys
import tensorflow as tf
import config
import matplotlib.pyplot as plt


def print_usage():
    """ Prints the usage of the program to console.
    """
    print("usage: $ python3 inference.py [image.jpg]")

# loading generator from data folder (unpack first)
generator = tf.keras.models.load_model(config.MODELPATH)
generator.compile(optimizer="adam", loss="mean_squared_error")
generator.save(config.MODELPATH)

if len(sys.argv) > 1:
    # loading image from passed argument
    image = tf.io.read_file(sys.argv[1])
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # resize (TODO: remove)
    image = tf.image.resize_with_crop_or_pad(image, config.LOWIMAGESIZE, config.LOWIMAGESIZE)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, axis=0)
    

    # perform upscaling
    upscaled_image = generator(image, training=False)
    upscaled_image = (upscaled_image + 1) / 2
    plt.imshow(upscaled_image[0, :, :, :])
    plt.savefig("./upscaled.pdf")
    
else:
    print_usage()
    exit