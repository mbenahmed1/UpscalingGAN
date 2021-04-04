"""
This file contains all the constants used in the project. 
"""

# general
USEGPU = True
SHOWSAMPLES = False
SHOWSUMMARY = False

# training
EPOCHS = 1000
GENLEARNINGRATE = 0.0001
DISLEARNINGRATE = 0.0001
NUMPARALLELCALLS = 8
TESTSPLITSIZE = 0.2
BATCHSIZE = 8
BUFFERSIZE = 32
PREFETCHSIZE = 1

# inference
MODELPATH = "../data/production_model/gen/"

# file I/O
DATAPATH = '../data/unlabeled2017/*'
IMAGEFILEEXTENSION = "jpg"
MAXIMAGEWIDTH = 8192
MAXIMAGEHEIGHT = 8192
TIMEESTIMATIONCOUNTER = 5000
DATASETSCALINGFACTOR = 1.0

# data augmentation
AUGMENTATIONPROBABILITY = 0.4
SATURATIONMIN = 0.8
SATURATIONMAX = 1.5
CONTRASTMAX = 0.9
CONTRASTMIN = 0.5
BRIGHTNESSMAXDETLA = 0.3

# checkpoints and logging
CHECKPOINTINTERVAL = 5
WHEIGTSPATH = "run_%d_%m_%Y__%H-%M-%S"
WEIGHTFOLDERPATH = "../weights/"
OUTPUTFILENAME = "log.txt"
GENERATORFILENAME = "gen"
DISCRIMINATORFILENAME = "dis"
LOWRESIMAGENAME = "low_res_image.pdf"
FULLRESIMAGENAME = "full_res_image.pdf"
UPSCALEDIMAGENAME = "upscaled.pdf"
LOSSFILENAME = "loss.pdf"

# model
LOWIMAGESIZE = 64
IMAGESCALINGFACTOR = 4  # this is bound to the model
FULLIMAGESIZE = LOWIMAGESIZE * IMAGESCALINGFACTOR
NUMCHANNELS = 3
NUMRESBLOCKS = 8
