"""Contains the training loop and preparation routines.

This file contains the main training loop and all other routines for loading
and preparing train data.
"""

# imports
import tensorflow as tf
import utils
import model
import config
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime

# taking start time for weight saving
started_training = datetime.now()
start_string = started_training.strftime(config.WHEIGTSPATH)
started_training_time = time.time()

# creating dir
path = f'{config.WEIGHTFOLDERPATH}{start_string}/'
try:
    os.makedirs(path)
except OSError:
    print("Creating weights directory has failed.")
else:
    print("Created weights directory.")

utils.write_log(path, start_string)

# disabling gpu if needed
if not config.USEGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# memory growth
devices = tf.config.list_physical_devices('GPU')

if devices:
    try:
        tf.config.experimental.set_memory_growth(devices[0], True)
    except:
        print('Could not enable memory growth.')
    if config.ENABLEGPUMEMLIMIT:
        try:
            tf.config.experimental.set_virtual_device_configuration(devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=config.GPUMEMLIMIT)])
            print('setting memory limit of ', config.GPUMEMLIMIT)
        except RuntimeError as e:
            print(e)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(d_true, d_pred):
    """
    d_pred: discriminator's prediction  of the image generated by the generator  
    take cross entropy between predicted label and all labels as 1s
    because we want to minimize the difference between them
    --> the more the discriminator thinks the images are real, the better our generator
    """

    return tf.keras.losses.MSE(d_true, d_pred) + bce(tf.ones_like(d_pred), d_pred)


def discriminator_loss(real_img_lbl, fake_img_lbl):
    """
    real_img_lbl: labels that discriminator predicted when seeing real images (should ideally be all 1s)
    fake_img_lbl: labels that discriminator predicted when seeing fake images (should ideally be all 0s)
    """
    real_loss = bce(tf.ones(real_img_lbl.shape), real_img_lbl)
    fake_loss = bce(tf.zeros(fake_img_lbl.shape), fake_img_lbl)

    return real_loss + fake_loss


def train(dataset, low_res_image, full_res_image, epochs):

    # saving low res sample image to weights folder
    plt.imshow((low_res_image[0, :, :, :] + 1) / 2)
    plt.savefig(f'{config.WEIGHTFOLDERPATH}{start_string}/{config.LOWRESIMAGENAME}')
    
    # saving full res sample image to weights folder
    plt.imshow((full_res_image[0, :, :, :] + 1) / 2)
    plt.savefig(f'{config.WEIGHTFOLDERPATH}{start_string}/{config.FULLRESIMAGENAME}')
    
    # run training steps for number of epochs
    for epoch in range(epochs):
        start = time.time()
        
        for data in dataset:
            (loss1, loss2) = train_step(data)
            loss_gen.append(loss1)
            loss_desc.append(loss2)

        # save weights every CHECKPOINTINTERVAL time
        if epoch % config.CHECKPOINTINTERVAL == 0:
            generator.save_weights(f'{config.WEIGHTFOLDERPATH}{start_string}/{config.GENERATORFILENAME}{int(epoch)}')
            discriminator.save_weights(f'{config.WEIGHTFOLDERPATH}{start_string}/{config.DISCRIMINATORFILENAME}{int(epoch)}')

        # print time elapsed for this particular epoch
        time_per_epoch = int(time.time() - start)
        epoch_time_string = (f'Time for epoch {epoch} is {time_per_epoch} s')
        print(epoch_time_string)

        # write time elapsed for this epoch to file
        text_file = open(f'{config.WEIGHTFOLDERPATH}{start_string}/{config.OUTPUTFILENAME}', "a")
        text_file.write(f'Time e_{epoch}:       {time_per_epoch} s \n')
        text_file.close()

        # generate one upscaled image from low res sample and write to folder
        upscaled_image = generator(low_res_image, training=False)
        upscaled_image = (upscaled_image + 1) / 2
        plt.imshow(upscaled_image[0, :, :, :])
        plt.savefig(f'{config.WEIGHTFOLDERPATH}{start_string}/{epoch}{config.UPSCALEDIMAGENAME}')

@tf.function
def train_step(images):
    image_lowRes = images[0]
    images_highRes = images[1]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(image_lowRes, training=True)

        real_output = discriminator(images_highRes, training=True)
        fake_output = discriminator(generated_image, training=True)

        gen_loss = generator_loss(images_highRes, generated_image)
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

with tf.device('/cpu:0'):

    # Create a generator
    rng = tf.random.Generator.from_seed(123, alg='philox')
    seed = rng.make_seeds(2)[1]

    # loading paths
    list_ds = tf.data.Dataset.list_files(config.DATAPATH)

    # loading and preparing images
    ds = list_ds.map(utils.prepare_images, config.NUMPARALLELCALLS)

    # size of the dataset
    size = len(list_ds)

    # TODO: make random from image to image
    # applying some augmentations for testing
    ds = ds.map(lambda x: utils.saturate(x, config.SATURATIONMIN,
                                        config.SATURATIONMAX, seed), config.NUMPARALLELCALLS)
    ds = ds.map(lambda x: utils.flip_left_right(x), config.NUMPARALLELCALLS)
    ds = ds.map(lambda x: utils.flip_up_down(x), config.NUMPARALLELCALLS)
    ds = ds.map(lambda x: utils.brighten(
        x, config.BRIGHTNESSMAXDETLA, seed), config.NUMPARALLELCALLS)
    ds = ds.map(lambda x: utils.contrast(x, config.CONTRASTMIN,
                                        config.CONTRASTMAX, seed), config.NUMPARALLELCALLS)


    # making pairs of the original and the scaled images
    ds = ds.map(utils.make_full_low_pairs, config.NUMPARALLELCALLS)
    ds = ds.take(int(size * config.DATASETSCALINGFACTOR))
    test_dataset = ds.take(int(size * config.TESTSPLITSIZE))
    train_dataset = ds.skip(int(size * config.TESTSPLITSIZE))

    # plot some samples

    if config.SHOWSAMPLES:
        for low_image, full_image in test_dataset.take(5):
            print(low_image.shape, full_image.shape)
            print(full_image)
            plt.imshow(low_image[:, :, :])
            plt.show()
            plt.imshow(full_image[:, :, :])
            plt.show()

    # batching
    test_dataset = test_dataset.batch(config.BATCHSIZE)
    train_dataset = train_dataset.batch(config.BATCHSIZE)

    # shuffling
    test_dataset = test_dataset.shuffle(buffer_size=config.BUFFERSIZE)
    train_dataset = train_dataset.shuffle(buffer_size=config.BUFFERSIZE)

    # prefetching
    train_dataset = train_dataset.prefetch(config.PREFETCHSIZE)
    test_dataset = test_dataset.prefetch(config.PREFETCHSIZE)


# *** TRAINING ***

# create optimizers
generator_optimizer = tf.keras.optimizers.Adam(config.GENLEARNINGRATE)
discriminator_optimizer = tf.keras.optimizers.Adam(config.DISLEARNINGRATE)

noise_dim = 100
num_examples_to_generate = 16

# create models
generator = model.Generator()
discriminator = model.Discriminator()
seed = tf.random.normal([num_examples_to_generate, noise_dim])

loss_gen = []
loss_desc = []
print("")
print("")
print("Start training")

if config.SHOWSUMMARY:
    generator.build((config.BATCHSIZE, config.LOWIMAGESIZE, config.LOWIMAGESIZE, config.NUMCHANNELS))
    discriminator.build((config.BATCHSIZE, config.FULLIMAGESIZE, config.FULLIMAGESIZE, config.NUMCHANNELS))
    print("")
    print("")
    print(generator.summary())
    print(discriminator.summary())

full_res_image = []
low_res_image = []
for low, full in test_dataset.take(1):
    low_res_image = low
    full_res_image = full

train(test_dataset, low_res_image, full_res_image, config.EPOCHS)

# plot loss

fig = plt.figure()
line1, = plt.plot(loss_gen)
line2, = plt.plot(loss_desc)
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.legend((line1,line2),("generator","discriminator"))
plt.savefig(f'{config.WEIGHTFOLDERPATH}{start_string}/{config.LOSSFILENAME}')

# write complete time elapsed to file
time_elapsed = int((time.time() - started_training_time) / 60)
text_file = open(f'{config.WEIGHTFOLDERPATH}{start_string}{config.OUTPUTFILENAME}', "a")
text_file.write(f'Time elapsed:     {time_elapsed} min\n')
text_file.close()
