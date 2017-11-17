"""  Bare-bones code for extracting CNN features (via tensorflow).

  Example usage:

  python nets.py ./NIPS_1000/Test ./Output

"""

__author__ = "mjp"
__date__ = "november, 2017"


import os, sys
import pdb

import numpy as np
from scipy.misc import imread, imsave
from scipy.io import savemat


import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, resnet_v2
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
slim = tf.contrib.slim


#-------------------------------------------------------------------------------
# Helper functions for data I/O
#-------------------------------------------------------------------------------
def input_filenames(input_dir):
  all_files = tf.gfile.Glob(os.path.join(input_dir, '*.png'))
  all_files.sort()
  return all_files


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      length of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]

  for filepath in input_filenames(input_dir):
    with tf.gfile.Open(filepath, mode='rb') as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0

    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))

    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0

  # This is a partial batch left over at end.
  # Note that images will still have the proper size.
  if idx > 0:
    yield filenames, images


#-------------------------------------------------------------------------------
# Networks
#-------------------------------------------------------------------------------

class InceptionV3:
  """ Just the first layer of InceptionV3
  """

  def __init__(self, sess):
    self.batch_shape = [16, 299, 299, 3]
    self._num_classes = 1001
    self._scope = 'InceptionV3'
    self._weights_file = './Weights/inception_v3.ckpt'
    output_layer = 'Conv2d_1a_3x3'

    #
    # network inputs
    #
    self.x_tf = tf.placeholder(tf.float32, shape=self.batch_shape)

    #
    # network outputs
    #
    with slim.arg_scope(inception.inception_v3_arg_scope()): 
      with arg_scope([layers_lib.batch_norm, layers_lib.dropout], is_training=False): # we truncate before these layers, so this is not really necessary...
        net, endpoints = inception.inception_v3_base(self.x_tf, final_endpoint=output_layer, scope=self._scope)
        self.output = endpoints[output_layer]

    #
    # load weights
    #
    saver = tf.train.Saver(slim.get_model_variables(scope=self._scope))
    saver.restore(sess, self._weights_file)


#-------------------------------------------------------------------------------
if __name__ == "__main__":
  if len(sys.argv) < 3:
    print('\nUSAGE:  python %s input_directory output_directory\n' % sys.argv[0])
    sys.exit(1)

  input_dir = sys.argv[1]
  output_dir = sys.argv[2]

  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  with tf.Graph().as_default(), tf.Session() as sess:
    model = InceptionV3(sess)

    # XXX: we may want to change the output format to make it more convenient
    #      for downstream processing later...
    for batch_id, (filenames, x) in enumerate(load_images(input_dir, model.batch_shape)):
      features = sess.run(model.output, feed_dict={model.x_tf : x})
      n = len(filenames)

      fn = os.path.join(output_dir, 'batch_%02d.txt')
      with open(fn, 'w') as f:
        f.writelines(filenames) 

      fn = os.path.join(output_dir, 'batch_%02d.mat')
      savemat(fn, {'X' : features[:n,...]})

      print('[info]: processed mini-batch # %d of size %d' % (batch_id, n))
