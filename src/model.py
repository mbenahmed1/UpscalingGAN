"""Contains the GAN model definition.

This file contains the model definition of the GAN
which consists of the Generator and the Discriminator Part.
"""

# imports
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
from tensorflow.keras.layers import Layer
import config

class ResidualBlock(Layer):
    """ Implements a residual block"""
    def __init__(self):

        super(ResidualBlock, self).__init__()

        self.conv_1     = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")
        self.bn_1       = layers.BatchNormalization()
        self.prelu_1    = layers.PReLU()
        
        self.conv_2     = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")
        self.bn_2       = layers.BatchNormalization()
        self.prelu_2    = layers.PReLU()

    def call(self, inp, training):
        """Runs the data through the layers.

        Attributes:
            inp:            the data
            training:       training flag
        """
        x = self.conv_1(inp)
        x = self.bn_1(x, training)
        x = self.prelu_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x, training)
        x = self.prelu_2(x)

        x += inp

        return x

class Generator(tf.keras.Model):
    """A class that represents the Generator part of the model.

    This class represents the Generator part of the neural
    network model and its layers.

    """

    def __init__(self):
        """Inits the Generator."""
        super(Generator, self).__init__()

        # k9n64s1
        self.conv_1 = layers.Conv2D(filters=64, kernel_size=(9,9), strides=(1,1), padding="same", input_shape=(config.BATCHSIZE, config.LOWIMAGESIZE, config.LOWIMAGESIZE, config.NUMCHANNELS))
        self.act_1 = layers.PReLU()

        # residual blocks
        self.residual_blocks = [ResidualBlock() for i in range(config.NUMRESBLOCKS)]

        # k3n64s1
        self.conv_2 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")
        self.bn_1 = layers.BatchNormalization()

        # k3n256s1
        self.conv_3 = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")
        self.convT_1 = layers.Conv2DTranspose(filters=256, kernel_size=(3,3), strides=(2,2), padding="same")
        self.act_2 = layers.PReLU()

        # k3n256s1
        self.conv_4 = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")
        self.convT_1 = layers.Conv2DTranspose(filters=256, kernel_size=(3,3), strides=(2,2), padding="same")
        self.act_3 = layers.PReLU()

        # k9n3s1
        self.conv_5 = layers.Conv2D(filters=3, kernel_size=(9,9), strides=(1,1), padding="same")

    def call(self, inp, training: bool):
        """Runs the data through the layers.

        Attributes:
            x:              the data
            training:       training flag
        """

        x = self.conv_1(inp)
        x = self.bn_1(x, training)
        x = self.act_1(x)

        for layer in self.residual_blocks:
            x = layer(x, training)

        x = self.conv_2(x)
        x = self.bn_1(x, training)

        #x += inp

        x = self.conv_3(x)
        x = self.convT_1(x)
        x = self.act_2(x)

        x = self.conv_4(x)
        x = self.convT_1(x)
        x = self.act_3(x)

        x = self.conv_5(x)

        return x


# Discriminator Model

class Discriminator(tf.keras.Model):
    """A class that represents the Discriminator part of the model.

    This class represents the Discriminator part of the neural
    network model and its layers.

    """

    def __init__(self):
        """Inits the Discriminator."""
        super(Discriminator, self).__init__()

        # k3n64s1
        self.conv_1 = layers.Conv2D(filters=64, kernel_size=(9,9), strides=(1,1), padding="same")
        self.act_1 = layers.LeakyReLU()

        # k3n64s2
        self.conv_2 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same")
        self.bn_1 = layers.BatchNormalization()
        self.act_2 = layers.LeakyReLU()

        # k3n128s1
        self.conv_3 = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same")
        self.bn_2 = layers.BatchNormalization()
        self.act_3 = layers.LeakyReLU()

        # k3n128s2
        self.conv_4 = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same")
        self.bn_3 = layers.BatchNormalization()
        self.act_4 = layers.LeakyReLU()

        # k3n256s1
        self.conv_5 = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")
        self.bn_4 = layers.BatchNormalization()
        self.act_5 = layers.LeakyReLU()

        # k3n256s2
        self.conv_6 = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same")
        self.bn_5 = layers.BatchNormalization()
        self.act_6 = layers.LeakyReLU()

        # k3n512s1
        self.conv_7 = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")
        self.bn_6 = layers.BatchNormalization()
        self.act_7 = layers.LeakyReLU()

        # k3n512s2
        self.conv_8 = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same")
        self.bn_7 = layers.BatchNormalization()
        self.act_8 = layers.LeakyReLU()

        self.dense_1 = layers.Dense(units=1024)
        self.act_9 = layers.LeakyReLU()
        self.out = layers.Dense(units=1, activation=activations.sigmoid)

    def call(self, x, training: bool):
        """Runs the data through the layers.

        Attributes:
            x:              the data
            training:       training flag
        """
        x = self.conv_1(x)
        x = self.act_1(x)

        x = self.conv_2(x)
        x = self.bn_1(x, training)
        x = self.act_2(x)

        x = self.conv_3(x)
        x = self.bn_2(x, training)
        x = self.act_3(x)

        x = self.conv_4(x)
        x = self.bn_3(x, training)
        x = self.act_4(x)

        x = self.conv_5(x)
        x = self.bn_4(x, training)
        x = self.act_5(x)

        x = self.conv_6(x)
        x = self.bn_5(x, training)
        x = self.act_6(x)

        x = self.conv_7(x)
        x = self.bn_6(x, training)
        x = self.act_7(x)

        x = self.conv_8(x)
        x = self.bn_7(x, training)
        x = self.act_8(x)

        x = self.dense_1(x)
        x = self.act_9(x)
        x = self.out(x)
        
        return x
