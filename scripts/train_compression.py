"""Train the compression algorithm using the training set in the provided directory.

Usage:
  compress_image.py <directoryname> <ncomponents>

"""
import sys
sys.path.append('')

from docopt import docopt
from numpy import matrix
from numpy import reshape
from skimage import io as image_io
from skimage import color
from skimage import transform
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.linalg import svd
import os
import pickle

IMAGE_SIZE = (256, 192)
BLOCK_SIZE = (64, 64)


def run(args):
    directory = args['<directoryname>']
    num_components = int(args['<ncomponents>'])

    image_names = []

    for image_file in os.listdir(directory):
        if image_file.endswith(".JPG"):
            image_names.append(directory + image_file)

    print 'Loading images...'

    images = image_io.imread_collection(image_names)

    print 'Resizing and greyscaling...'

    images_processed = [transform.resize(color.rgb2grey(image), IMAGE_SIZE) for image in images]

    print 'Blocking images...'

    blocks = [get_blocks(image) for image in images_processed]

    training_rows = []

    print 'Building training matrix...'

    for block in blocks[0]:
        vect = block_to_vect(block)
        training_rows.append(vect)

    training_matrix = matrix(training_rows).transpose()

    print 'Computing SVD...'

    u_matrix, sing_vals, v_matrix = svd(training_matrix, full_matrices=True) # TruncatedSVD(n_components=num_components)

    compression_matrix = u_matrix[:, :num_components]

    print 'Writing matrix to file...'

    with open(directory + 'compression.matrix', 'w') as outfile:
        pickle.dump(compression_matrix, outfile)


def block_to_vect(block):
    row = [item for sublist in block for item in sublist]
    return row


def get_blocks(image):
    return extract_patches_2d(image, BLOCK_SIZE)


def recover_block(vect):
    return reshape(vect, BLOCK_SIZE)


if __name__ == '__main__':
    run(docopt(__doc__))
