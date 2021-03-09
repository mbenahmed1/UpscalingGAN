"""Contains the training loop and preparation routines.

This file contains the main training loop and all other routines for loading
and preparing train data.
"""

# imports
import tensorflow as tf
import utils
import model
import constants
import os
import matplotlib.pyplot as plt
import time

# disabling gpu if needed
if not constants.USEGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(d_pred):
    """
    d_pred: discriminator's prediction  of the image generated by the generator  
    take cross entropy between predicted label and all labels as 1s
    because we want to minimize the difference between them
    --> the more the discriminator thinks the images are real, the better our generator
    """
    return bce(tf.ones_like(d_pred), d_pred)  # tf.ones_like(d_pred) is same as tf.ones(d_pred.shape)


def discriminator_loss(real_img_lbl, fake_img_lbl):
    """
    real_img_lbl: labels that discriminator predicted when seeing real images (should ideally be all 1s)
    fake_img_lbl: labels that discriminator predicted when seeing fake images (should ideally be all 0s)
    """
    real_loss = bce(tf.ones(real_img_lbl.shape), real_img_lbl)
    fake_loss = bce(tf.zeros(fake_img_lbl.shape), fake_img_lbl)

    return real_loss + fake_loss


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

    # DEBUG
    tessst = dataset.take(1)
    print(tessst)

    for data in dataset:
        (loss1, loss2) = train_step(data)
        loss_gen.append(loss1)
        loss_desc.append(loss2)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


#@tf.function
def train_step(images):
    noise = tf.random.normal([constants.BATCHSIZE, noise_dim])
    image_lowRes = images[0]
    images_highRes = images[1]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(image_lowRes, training=True)

        real_output = discriminator(images_highRes, training=True)
        fake_output = discriminator(generated_image, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))

    return (gen_loss, disc_loss)


# Create a generator
rng = tf.random.Generator.from_seed(123, alg='philox')
seed = rng.make_seeds(2)[1]

# loading paths
#list_ds = tf.data.Dataset.list_files(constants.DATAPATH)

# loading and preparing images
#ds = list_ds.map(utils.prepare_images, constants.NUMPARALLELCALLS)

ds = tf.keras.preprocessing.image_dataset_from_directory('../data', label_mode=None, image_size=(512, 512), batch_size=1)
print('Einlesen der Bilder erfolgreich')

# size of the dataset
#size = len(list_ds)

# TODO: make random from image to image
# applying some augmentations for testing
ds = ds.map(lambda x: utils.saturate(x, constants.SATURATIONMIN,
                                     constants.SATURATIONMAX, seed), constants.NUMPARALLELCALLS)
ds = ds.map(lambda x: utils.flip_left_right(x), constants.NUMPARALLELCALLS)
ds = ds.map(lambda x: utils.flip_up_down(x), constants.NUMPARALLELCALLS)
ds = ds.map(lambda x: utils.brighten(
    x, constants.BRIGHTNESSMAXDETLA, seed), constants.NUMPARALLELCALLS)
ds = ds.map(lambda x: utils.contrast(x, constants.CONTRASTMIN,
                                     constants.CONTRASTMAX, seed), constants.NUMPARALLELCALLS)

# making pairs of the original and the scaled images
ds = ds.map(utils.make_full_low_pairs, constants.NUMPARALLELCALLS)

# splitting dataset into test and train
test_dataset = ds.take(int(1000))
train_dataset = ds.skip(int(1000)).take(1000)

# plot some samples
"""
for low_image, full_image in test_dataset.take(5):
    print(low_image.shape, full_image.shape)
    plt.imshow(low_image)
    plt.show()
    plt.imshow(full_image)
    plt.show()
"""
# batching
#test_dataset = test_dataset.batch(constants.BATCHSIZE)
#train_dataset = train_dataset.batch(constants.BATCHSIZE)

#test_dataset = test_dataset.batch(1)
#train_dataset = train_dataset.batch(1)

# shuffling
#test_dataset = test_dataset.shuffle(buffer_size=constants.BUFFERSIZE)
#train_dataset = train_dataset.shuffle(buffer_size=constants.BUFFERSIZE)

# prefetching
train_dataset = train_dataset.prefetch(constants.PREFETCHSIZE)
test_dataset = test_dataset.prefetch(constants.PREFETCHSIZE)


# training

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

noise_dim = 100
num_examples_to_generate = 16


generator = model.Generator()
discriminator = model.Discriminator()
seed = tf.random.normal([num_examples_to_generate, noise_dim])

loss_gen = []
loss_desc = []

print("beginne Training")

train(test_dataset, constants.EPOCHS)
