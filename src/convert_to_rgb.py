"""This file contains a scritp to convert images to RGB.
"""

from PIL import Image
import sys

if len(sys.argv) != 2:
    print("Usage: $ python convert_to_rgb.py [0.jpg] [1.jpg] [2.jpg] ...")

for arg in sys.argv[1:]:
    image = Image.open(str(arg))
    rgb = image.convert('RGB')
    rgb.save(str(arg))
