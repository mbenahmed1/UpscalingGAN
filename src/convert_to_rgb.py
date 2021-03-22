"""This file contains a scritp to convert images to RGB.
"""

from PIL import Image
import sys
from os import walk

if len(sys.argv) != 2:
    print("Usage: $ python convert_to_rgb.py [folder_dest]")

_, _, filenames = next(walk(sys.argv[1]))

num_files = len(filenames)
print(str(num_files) + " files found in the given directory.")
print("Starting conversion...")
counter = 0
first = True
perc_count = 0
split = int(num_files / 100)

for img_path in filenames:

    if counter == split or first:
        print(str(perc_count) + "%")
        perc_count += 1
        counter = 0
        first = False

    full_path = sys.argv[1] + str(img_path)
    image = Image.open(str(full_path))
    rgb = image.convert('RGB')
    rgb.save(str(full_path))
    counter += 1
