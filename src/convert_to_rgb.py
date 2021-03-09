from PIL import Image
import constants
import sys


for arg in sys.argv[1:]:
    image = Image.open(str(arg))
    rgb = image.convert('RGB')
    rgb.save(str(arg))
