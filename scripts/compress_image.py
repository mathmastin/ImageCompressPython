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
from numpy import array
from numpy import dot
from numpy import reshape
from numpy import matrix

IMAGE_SIZE = (128, 99)
BLOCK_SIZE = (64, 64)

def run(args):
    filename = args['<filename>']
    directoryname = args['<directoryname>']

    try:
        with open(directoryname + 'compression.matrix', 'r') as infile:
           compression_matrix = pickle.load(infile)
    except:
        print 'Compression matrix not found. Create the compression matrix with'
        print 'train_compression.py and pass the location as argument 2 to this script.'

    print 'Resizing image...'
    image = transform.resize(color.rgb2grey(image_io.imread(filename)), IMAGE_SIZE)

    print 'Saving resized, uncompressed image...'
    image_io.imsave('original.jpg', image)

    print 'Blocking image...'
    image_blocks = get_blocks(image)

    print 'Vectorizing blocks...'
    image_vects = [matrix(block_to_vect(image_blocks[i])) for i in range(0, len(image_blocks))]

    print 'Compressing block vectors...'
    compressed_vects = [compress_vect(image_vects[i], compression_matrix) for i in range(0, len(image_vects))]

    print 'Compression ratio = ' + str(float(len(compressed_vects[0][0]))/float(len(image_vects[0][0])))

    print 'Decompressing blocks...'
    decomp_blocks = array([recover_block(dot(compression_matrix, compressed_vects[i].transpose())) for i in range(0, len(compressed_vects))])

    print 'Saving image...'
    image_io.imsave('compressed.' + str(len(compressed_vects[0][0])) + '.jpg', reconstruct_from_patches_2d(decomp_blocks, IMAGE_SIZE))


def block_to_vect(block):
    row = [item for sublist in block for item in sublist]
    return row


def get_blocks(image):
    return extract_patches_2d(image, BLOCK_SIZE)


def recover_block(vect):
    new_vect = []
    for i in vect:
        if abs(i) < 1:
            new_vect.append(i)
        else:
            new_vect.append(i/i)
    return reshape(array(new_vect), BLOCK_SIZE)


def compress_vect(image_vect, compression_matrix):
    clf = linear_model.LinearRegression()
    clf.fit(compression_matrix, image_vect.transpose())
    return clf.coef_


if __name__ == '__main__':
    run(docopt(__doc__))
