# UpscalingGAN
This repository contains an Image Upscaling project using a Generative Adversarial Network (GAN) model. 

<img src="https://user-images.githubusercontent.com/33060086/113520398-1df10d00-9593-11eb-873d-96536435cb74.jpg" data-canonical-src="https://user-images.githubusercontent.com/33060086/113520398-1df10d00-9593-11eb-873d-96536435cb74.jpg" width="300" height="300" /> <img src="https://user-images.githubusercontent.com/33060086/113520402-23e6ee00-9593-11eb-97cb-2ef84ee4fba8.jpg" data-canonical-src="https://user-images.githubusercontent.com/33060086/113520402-23e6ee00-9593-11eb-97cb-2ef84ee4fba8.jpg" width="300" height="300" />


## Usage
### Inference
In order to try out our trained model, please unpack the `data/production_model/gen.tar.gz` archive by typing </br>

`$ tar xf gen.tar.gz`

in the command line. The path to the unpacked folder is defined in `src/config.py`. After unpacking, tensorflow is able to read the model from the `gen` folder and you can upscale an image by typing

`$ cd src/`

`$ python3 full_image_inference.py [image.jpg]`.

Note that `[image.jpg]` is a parameter and must contain a valid path to a jpg image. The output will be saved as `upscaled.jpg` in your current folder.

### Train
In order to train the network, it is necessary to download the dataset first. We used parts of the Coco dataset which can be downloaded from http://images.cocodataset.org/zips/unlabeled2017.zip. We used the 
unlabled images with 123k samples. Is has to be extracted and placed in a folder `data` in the root of the directory in order for the algorithm to find it. The path can be redefined in the config, if needed.
As these images are a mixture of grayscale and color images, everyone of them needs to be converted to rgb in order for the network to handle them.
This is achieved using the `$ convert_to_rgb.py` in the following manner:

`$ python3 convert_to_rgb.py [path to folder containing images]`

After this conversion, the training process can be started with 

`$ python3 train.py`.

This will create a folder containing intermediate results.

A conda environment is provided with `UpscalingGAN.yml`.

## Abstract


New state of the art neural network architectures and ever increasing computational resources
offer a large potential for new applications of deep learning in real life problems.
In the context of images, GANs play a big role in creating new plausible sample data from a specified input.
Amongst others, GANs can solve the problem of upscaling an image while increasing its level of detail, making the image look better when enlarging it. Upscaling is especially useful in context of video games, surveillance cameras or old data with a very low resolution.
In this report we describe our approach on recreating a GAN for image super resolution, heavily
inspired by <cite>Ledig et al. [1]</cite>.

## More information

More information concerning the used model and training process can be found in the report https://github.com/mbenahmed1/UpscalingGAN/blob/main/doc/report/report.pdf.



[1]: C. Ledig, L. Theis, F. Huszar, J. Caballero, A. P. Aitken,
A. Tejani, J. Totz, Z. Wang, and W. Shi, “Photo-realistic single
image super-resolution using a generative adversarial network,”
CoRR, vol. abs/1609.04802, 2016.
