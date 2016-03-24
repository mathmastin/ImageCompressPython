"""Train the compression algorithm using the training set in the provided directory.

Usage:
  compress_image.py <directoryname> <ncomponents>

"""
import sys
sys.path.append('')

from docopt import docopt
from numpy import split
from numpy import matrix
from numpy import reshape
from skimage import io as image_io
from skimage import img_as_ubyte
from skimage import color
from skimage import transform
from skimage.viewer import ImageViewer
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
import os
import pickle


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

    images_processed = [transform.resize(color.rgb2grey(image), (256, 192)) for image in images]

    print 'Blocking images...'

    blocks = [get_blocks(image) for image in images_processed]

    training_rows = []

    print 'Building training matrix...'

    for block in blocks[0]:
        vect = block_to_vect(block)
        training_rows.append(vect)

    training_matrix = matrix(images_processed[0]).transpose()

    print 'Computing SVD...'

    trun_svd = TruncatedSVD(n_components=num_components)

    compression_matrix = trun_svd.fit_transform(training_matrix)

    with open(directory + 'compression.matrix', 'w') as outfile:
        pickle.dump(compression_matrix, outfile)


def block_to_vect(block):
    row = [item for sublist in block for item in sublist]
    return row


def get_blocks(image):
    return extract_patches_2d(image, (64, 64))


def recover_block(vect):
    return reshape(vect, (64, 64))


if __name__ == '__main__':
    run(docopt(__doc__))
