"""
This file contains all the constants used in the project. 
"""

# general
USEGPU = True

# file I/O
DATAPATH = '../data/unlabeled2017/*'
IMAGEFILEEXTENSION = "jpg"
MAXIMAGEWIDTH = 8192
MAXIMAGEHEIGHT = 8192
TIMEESTIMATIONCOUNTER = 5000
DATASETSCALINGFACTOR = 0.5

# data prep
FULLIMAGESIZE = 512
LOWIMAGESIZE = 128
NUMPARALLELCALLS = 8

# data augmentation
BRIGHTNESSMAXDETLA = 0.5