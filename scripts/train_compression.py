"""Train the compression algorithm using the training set in the provided directory.

Usage:
  compress_image.py <directoryname>

"""
import sys
sys.path.append('')

from docopt import docopt
from skimage import io as image_io
from skimage import img_as_uint
from skimage import color
import os


def run(args):
    directory = args['<directoryname>']

    image_names = []

    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            image_names.append(directory + file)

    images = image_io.imread_collection(image_names)

    grey_images = map(color.rgb2grey, images)


if __name__ == '__main__':
    run(docopt(__doc__))
