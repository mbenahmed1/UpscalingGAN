"""Contains the GAN model definition.

This file contains the model definition of the GAN
which consists of the Generator and the Discriminator Part.
"""

# imports
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations


class Generator(tf.keras.Model):
    """A class that represents the Generator part of the model.

    This class represents the Generator part of the neural
    network model and its layers.

    """

    def __init__(self):
        """Inits the Generator."""
        super(Generator, self).__init__()

        self.input_1 = layers.Dense(
7*7*256, use_bias=False, input_shape=(100,))
        self.batch_norm_1 = layers.BatchNormalization()
        #self.act_1 = activations.sigmoid
        self.act_1 = layers.LeakyReLU()
        self.reshape_1 = layers.Reshape((7, 7, 256))

        self.conv_2 = layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.batch_norm_2 = layers.BatchNormalization()
        #self.act_2 = activations.sigmoid
        self.act_2 = layers.LeakyReLU()

        self.conv_3 = layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm_3 = layers.BatchNormalization()
        #self.act_3 = activations.sigmoid
        self.act_3 = layers.LeakyReLU()

        self.conv_4 = layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        #self.act_4 = activations.sigmoid
        self.act_4 = layers.LeakyReLU()

    def call(self, x, training: bool):
        """Runs the data through the layers.

        Attributes:
            x:              the data
            training:       training flag
        """
        x = self.input_1(x)
        x = self.batch_norm_1(x, training)
        x = self.act_1(x)
        x = self.reshape_1(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x, training)
        x = self.act_2(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x, training)
        x = self.act_3(x)

        x = self.conv_4(x)
        x = self.act_4(x)

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

        self.conv_1 = layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', input_shape=[28, 28, 1])
        self.act_1 = activations.sigmoid
        self.dropout_1 = layers.Dropout(0.3)

        self.conv_2 = layers.Conv2D(
            128, (5, 5), strides=(2, 2), padding='same')
        self.act_2 = activations.sigmoid
        self.dropout_2 = layers.Dropout(0.3)

        self.flatten_3 = layers.Flatten()
        self.out_3 = layers.Dense(1)

    def call(self, x, training: bool):
        """Runs the data through the layers.

        Attributes:
            x:              the data
            training:       training flag
        """
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.dropout_1(x)

        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.dropout_2(x)

        x = self.flatten_3(x)
        x = self.out_3(x)

        return x
