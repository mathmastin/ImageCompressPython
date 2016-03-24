"""Compress an image using the training matrix in the provided directory.

Usage:
  compress_image.py <filename> <directoryname>

"""
import sys
sys.path.append('')

from docopt import docopt
import pickle
from sklearn import linear_model
from skimage import io as image_io
from skimage import color
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage import transform
from numpy import reshape
from numpy import matrix

def run(args):
    filename = args['<filename>']
    directoryname = args['<directoryname>']

    with open(directoryname + 'compression.matrix', 'r') as infile:
        compression_matrix = pickle.load(infile)

    print len(compression_matrix)
    print len(compression_matrix[0])
    print compression_matrix[0]

    image = transform.resize(color.rgb2grey(image_io.imread(filename)), (256, 192))

    image_blocks = get_blocks(image)

    image_vect = matrix(block_to_vect(image_blocks[0]))

    print len(image_vect)

    clf = linear_model.LinearRegression()

    clf.fit(compression_matrix, image_vect.transpose())


def block_to_vect(block):
    row = [item for sublist in block for item in sublist]
    return row


def get_blocks(image):
    return extract_patches_2d(image, (64, 64))


def recover_block(vect):
    return reshape(vect, (64, 64))


if __name__ == '__main__':
    run(docopt(__doc__))
